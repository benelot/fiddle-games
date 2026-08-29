"""
Worm locomotion simulation — Geijtenbeek-style muscle-tendon in Brax/JAX.

Physics:
  • Brax built-in viscous + density fluid (density=4000, viscosity=0.1)
    — matches the standard Brax swimmer environment approach
  • Hill-type CE+PEE muscles, rigid tendon (Zajac 1989 formulation)
  • First-order activation dynamics: τ_act ≈ 15 ms, τ_deact ≈ 50 ms
  • Sinusoidal CPG traveling wave

Rendering matches Geijtenbeek's visual style:
  • Light gray capsule segments on off-white background
  • Muscle spindles colored by activation level (jet: blue→green→red)

Run:
    python ecology/worm/simulate.py

Outputs:
    ecology/worm/worm_sim.html   — interactive Brax viewer
    ecology/worm/worm_sim.gif    — Geijtenbeek-style animated visualization
"""

import os
import time
import warnings
import sys as _sys

warnings.filterwarnings("ignore", category=UserWarning, module="brax")

import jax
import jax.numpy as jnp
import numpy as np

from brax.io import mjcf, html
import brax.generalized.pipeline as pipeline

# ---------------------------------------------------------------------------
# Monkey-patch: Brax's fluid.py has a jnp.clip(a_min=...) API mismatch with
# this JAX version. Replace with our anisotropic RFT model which also gives
# better propulsion (C_n/C_t ≈ 4, matching Gray & Hancock 1955).
# ---------------------------------------------------------------------------
import brax.fluid as _brax_fluid
import brax.generalized.dynamics as _brax_dyn
from brax.base import Force, Transform

_C_N   = 5.0    # normal (lateral) drag coefficient [N·s/m per segment]
_C_T   = 1.25   # tangential (axial) drag coefficient [N·s/m]
_C_ROT = 0.003  # rotational drag [N·m·s/rad] — near-physical value for the body size


def _rft_fluid_force(sys, x, xd, mass, inertia, root_com=None):
    """
    RFT anisotropic drag in body frame (Gray & Hancock 1955).
    xd_i.vel is in the body's inertia frame; long axis = [1,0,0] in body frame.
    Forces are rotated back to world frame before returning.
    """
    x_i    = x.vmap().do(sys.link.inertia.transform)
    offset = x_i.pos - x.pos if root_com is None else x_i.pos - root_com
    xd_i   = x_i.replace(pos=offset).vmap().do(xd)
    v      = xd_i.vel                  # (n, 3) in body frame

    v_tang = jnp.concatenate([v[:, 0:1], jnp.zeros_like(v[:, 1:])], axis=-1)
    v_norm = v - v_tang

    f_lin  = -_C_N * v_norm - _C_T * v_tang
    f_ang  = -_C_ROT * xd_i.ang

    frc = Force(vel=f_lin, ang=f_ang)
    frc = Transform.create(rot=x_i.rot).vmap().do(frc)
    return frc


_brax_fluid.force = _rft_fluid_force
_brax_dyn.fluid   = _brax_fluid

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
_WORM_DIR = os.path.dirname(os.path.abspath(__file__))
_sys.path.insert(0, _WORM_DIR)
from muscle import MuscleParams, joint_torque

# ---------------------------------------------------------------------------
# Simulation constants
# ---------------------------------------------------------------------------
DT           = 0.002    # physics timestep (s) — matches worm.xml
SIM_DURATION = 10.0     # total simulation time (s)
RENDER_FPS   = 40       # frames/s stored for GIF
N_JOINTS     = 3
N_SEGMENTS   = 4

# ---------------------------------------------------------------------------
# Muscle parameters — scaled to overcome Brax fluid drag at worm body size.
# Brax swimmer uses 150 N·m gear; with moment arm r=0.04m that's F_max=3750N.
# We use a lower F_max because our segments are much smaller (drag ∝ A × v²),
# so less force is needed to drive undulatory locomotion.
# ---------------------------------------------------------------------------
MUSCLE = MuscleParams(
    F_max    = 200.0,  # N — strong enough to overcome viscous + density drag
    l_opt    = 0.08,   # m  optimal CE fibre length
    l_slack  = 0.05,   # m  tendon slack (rigid-tendon assumption)
    v_max    = 0.6,    # m/s
    r        = 0.040,  # m  moment arm
    w        = 0.56,   # Gaussian width (fraction of l_opt)
    k_pee    = 0.35,   # PEE stiffness (× F_max)
    pee_slack= 1.0,    # PEE rest at l_opt
)

# ---------------------------------------------------------------------------
# CPG controller: sinusoidal traveling wave
# ---------------------------------------------------------------------------
FREQ         = 1.3     # Hz — empirically optimal for this worm geometry
PHASE_OFFSET = -1.15   # rad — tail-to-head wave propagation → +X thrust
AMPLITUDE    = 0.90    # peak neural drive ∈ [0, 1]

# ---------------------------------------------------------------------------
# Activation dynamics — Geijtenbeek 2013 first-order model
# ---------------------------------------------------------------------------
TAU_ACT   = 0.015   # s  activation rise  (≈ 15 ms)
TAU_DEACT = 0.050   # s  activation decay (≈ 50 ms)

# ---------------------------------------------------------------------------
# Joint limits — soft damped spring (Brax does not enforce XML range limits).
# Spring natural freq ≈ √(K/I_eff); with I_eff≈0.05 kg·m² and K=5:
#   ω_n ≈ 10 rad/s, period ≈ 0.6s >> DT → integration is stable.
# ---------------------------------------------------------------------------
_LIMIT_RAD = jnp.radians(72.0)   # just inside XML range="-75 75"
_K_LIMIT   = 3.0                  # N·m/rad — soft spring
_D_LIMIT   = 0.08                 # N·m·s/rad — light damping at the limit


def _cpg_drive(t: jnp.ndarray) -> jnp.ndarray:
    """Raw CPG neural drive u(t) ∈ [0, 1] for all muscles."""
    acts = []
    for i in range(N_JOINTS):
        phase = 2.0 * jnp.pi * FREQ * t - i * PHASE_OFFSET
        acts.append(AMPLITUDE * jnp.maximum(0.0,  jnp.sin(phase)))   # flexor
        acts.append(AMPLITUDE * jnp.maximum(0.0, -jnp.sin(phase)))   # extensor
    return jnp.stack(acts)


@jax.jit
def _update_activation(a: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
    """First-order activation filter: da/dt = (u - a) / τ(u, a)."""
    tau = jnp.where(u > a, TAU_ACT, TAU_DEACT)
    return a + (u - a) / tau * DT


def _compute_torques(q: jnp.ndarray, qd: jnp.ndarray,
                     a_act: jnp.ndarray) -> jnp.ndarray:
    """Compute joint torques from muscle activations, with soft joint limits."""
    theta   = q[3:6]
    theta_d = qd[3:6]
    torques = []
    for i in range(N_JOINTS):
        tau = joint_torque(
            a_flex    = a_act[2 * i],
            a_ext     = a_act[2 * i + 1],
            theta     = theta[i],
            theta_dot = theta_d[i],
            p         = MUSCLE,
        )
        # Soft damped spring: pushes joint back when beyond ±68°
        excess = jnp.maximum(0.0, jnp.abs(theta[i]) - _LIMIT_RAD)
        past   = jnp.float32(excess > 0.0)
        tau    = tau - _K_LIMIT * excess * jnp.sign(theta[i])
        tau    = tau - past * _D_LIMIT * theta_d[i]
        torques.append(tau)
    return jnp.stack(torques)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    xml_path = os.path.join(_WORM_DIR, "worm.xml")
    print(f"Loading {xml_path} …")
    sys = mjcf.load(xml_path)

    q0    = jnp.zeros(sys.q_size())
    qd0   = jnp.zeros(sys.qd_size())
    state = pipeline.init(sys, q0, qd0)

    @jax.jit
    def step_fn(state, a_act):
        torques = _compute_torques(state.q, state.qd, a_act)
        return pipeline.step(sys, state, torques)

    print("Compiling … (one-time cost)")
    t0 = time.time()
    a0 = jnp.zeros(2 * N_JOINTS)
    _ = step_fn(state, a0).q.block_until_ready()
    print(f"  compile done in {time.time()-t0:.1f}s")

    n_steps    = int(SIM_DURATION / DT)
    keep_every = max(1, int(1.0 / (RENDER_FPS * DT)))

    a_state = jnp.zeros(2 * N_JOINTS)
    stored  = [(state, a_state)]

    print(f"Simulating {SIM_DURATION}s ({n_steps} steps, {n_steps//keep_every} frames) …")
    t0 = time.time()

    for i in range(1, n_steps + 1):
        t_sim   = jnp.float32(i * DT)
        u       = _cpg_drive(t_sim)
        a_state = _update_activation(a_state, u)
        state   = step_fn(state, a_state)
        if i % keep_every == 0:
            stored.append((state, a_state))
        if i % (n_steps // 10) == 0:
            q_np = np.array(state.q)
            mj   = float(np.max(np.abs(np.degrees(q_np[3:6]))))
            print(f"  {100*i//n_steps}%  sim_t={i*DT:.1f}s  x={float(q_np[0]):.3f}m  max_j={mj:.1f}°", flush=True)

    elapsed  = time.time() - t0
    n_frames = len(stored)
    print(f"Done in {elapsed:.1f}s  ({n_steps/elapsed:.0f} steps/s, {n_frames} frames)")

    x_start = float(stored[0][0].q[0])
    x_end   = float(state.q[0])
    print(f"Displacement along X: {x_end - x_start:.3f} m over {SIM_DURATION}s")

    html_path = os.path.join(_WORM_DIR, "worm_sim.html")
    print(f"Writing HTML viewer → {html_path}")
    html_str = html.render(sys, [s for s, _ in stored], height=540, colab=False)
    with open(html_path, "w") as f:
        f.write(html_str)

    gif_path = os.path.join(_WORM_DIR, "worm_sim.gif")
    _render_gif(stored, gif_path, DT * keep_every)

    print("\nDone.")
    print(f"  Open {html_path} in your browser for the 3D viewer.")
    print(f"  GIF: {gif_path}")


# ---------------------------------------------------------------------------
# Visualization — Geijtenbeek 2013 style
# ---------------------------------------------------------------------------
def _render_gif(stored, path: str, frame_dt: float):
    """Geijtenbeek-style 2D visualization: gray bodies + activation-heatmap muscles."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    import matplotlib.cm as cm
    from matplotlib.patches import Polygon

    print(f"Rendering GIF ({len(stored)} frames) → {path}")

    SEGS       = [(0.090, 0.040), (0.075, 0.036), (0.065, 0.031), (0.055, 0.025)]
    HL         = [s[0] for s in SEGS]
    VIS_RADII  = [s[1] * 0.55 for s in SEGS]

    BG        = "#f8f8f6"
    SEG_FILL  = "#b8c2be"
    SEG_EDGE  = "#6a827e"
    SKE_COL   = "#9ab0aa"
    PIVOT_COL = "#445550"

    R_MOM    = MUSCLE.r
    ATTACH_D = 0.020

    def _capsule_xy(cx, cy, hl, r, angle, n=14):
        tr = np.linspace(-np.pi / 2, np.pi / 2, n)
        tl = np.linspace(np.pi / 2, 3 * np.pi / 2, n)
        lx = np.concatenate([hl + r * np.cos(tr), -hl + r * np.cos(tl)])
        ly = np.concatenate([r * np.sin(tr), r * np.sin(tl)])
        ca, sa = np.cos(angle), np.sin(angle)
        return cx + ca * lx - sa * ly, cy + sa * lx + ca * ly

    def _spindle_xy(p0, p1, belly):
        dx, dy = p1[0] - p0[0], p1[1] - p0[1]
        L  = max(float(np.hypot(dx, dy)), 1e-6)
        px, py = -dy / L, dx / L
        mx, my = (p0[0] + p1[0]) * 0.5, (p0[1] + p1[1]) * 0.5
        t  = np.linspace(0, 1, 16)
        tx = (1 - t)**2 * p0[0] + 2 * (1 - t) * t * (mx + px * belly) + t**2 * p1[0]
        ty = (1 - t)**2 * p0[1] + 2 * (1 - t) * t * (my + py * belly) + t**2 * p1[1]
        bx = (1 - t)**2 * p0[0] + 2 * (1 - t) * t * (mx - px * belly) + t**2 * p1[0]
        by = (1 - t)**2 * p0[1] + 2 * (1 - t) * t * (my - py * belly) + t**2 * p1[1]
        return np.concatenate([tx, bx[::-1]]), np.concatenate([ty, by[::-1]])

    def _force_frac(act, p0, p1):
        lmtu = float(np.hypot(p1[0] - p0[0], p1[1] - p0[1]))
        lCE  = max(lmtu - MUSCLE.l_slack, 0.01 * MUSCLE.l_opt)
        fl   = float(np.exp(-((lCE / MUSCLE.l_opt - 1.0) / MUSCLE.w) ** 2))
        return float(act) * fl

    window_w = 0.85
    window_h = 0.46
    fig, ax  = plt.subplots(figsize=(10, 4.0), facecolor=BG)
    ax.set_facecolor(BG)
    ax.set_aspect("equal")
    ax.axis("off")

    x_start_f = float(stored[0][0].q[0])

    def animate(frame_idx):
        ax.cla()
        ax.set_facecolor(BG)
        ax.set_aspect("equal")
        ax.axis("off")

        state, a_act = stored[frame_idx]
        q     = np.array(state.q)
        a_act = np.array(a_act)
        t_sim = frame_idx * frame_dt

        jp     = np.array(state.x.pos[:, :2])   # (4, 2)
        angles = [float(q[2])]
        for k in range(3):
            angles.append(angles[-1] + float(q[3 + k]))

        centers = [jp[0].copy()]
        for k in range(1, N_SEGMENTS):
            a = angles[k]
            centers.append(jp[k] + HL[k] * np.array([np.cos(a), np.sin(a)]))

        a3       = angles[3]
        tail_end = jp[3] + 2 * HL[3] * np.array([np.cos(a3), np.sin(a3)])
        cx_mid   = float((centers[0][0] + tail_end[0]) / 2)
        ax.set_xlim(cx_mid - 0.55 * window_w, cx_mid + 0.45 * window_w)
        ax.set_ylim(-window_h / 2, window_h / 2)

        ax.plot([c[0] for c in centers], [c[1] for c in centers],
                color=SKE_COL, lw=1.4, zorder=2, solid_capstyle="round")

        for k in range(N_JOINTS):
            pivot  = jp[k + 1]
            a_k    = angles[k];    a_k1 = angles[k + 1]
            ax_k   = np.array([ np.cos(a_k),  np.sin(a_k)])
            ax_k1  = np.array([ np.cos(a_k1), np.sin(a_k1)])
            perp_k  = np.array([-np.sin(a_k),  np.cos(a_k)])
            perp_k1 = np.array([-np.sin(a_k1), np.cos(a_k1)])

            for side, mu_act in [(+1, a_act[2 * k]), (-1, a_act[2 * k + 1])]:
                prox  = pivot - ATTACH_D * ax_k  + side * R_MOM * perp_k
                dist  = pivot + ATTACH_D * ax_k1 + side * R_MOM * perp_k1
                ff    = _force_frac(mu_act, prox, dist)
                belly = 0.002 + 0.014 * np.sqrt(max(0.0, ff))
                rgba  = cm.jet(float(mu_act))
                color = rgba[:3]
                alpha = 0.45 + 0.55 * float(mu_act)
                sx, sy = _spindle_xy(prox, dist, belly)
                ax.add_patch(Polygon(np.column_stack([sx, sy]), closed=True,
                                     facecolor=color, alpha=alpha,
                                     edgecolor=color, linewidth=0.15, zorder=3))

        for j in range(N_SEGMENTS):
            cx, cy = float(centers[j][0]), float(centers[j][1])
            xs, ys = _capsule_xy(cx, cy, HL[j], VIS_RADII[j], angles[j])
            ax.add_patch(Polygon(np.column_stack([xs, ys]), closed=True,
                                 facecolor=SEG_FILL, alpha=0.92,
                                 edgecolor=SEG_EDGE, linewidth=0.75, zorder=4))

        for k in range(N_JOINTS):
            ax.plot(float(jp[k + 1, 0]), float(jp[k + 1, 1]),
                    'o', color=PIVOT_COL, ms=4.0, zorder=6,
                    markeredgecolor=SEG_EDGE, markeredgewidth=0.5)

        ax.text(0.015, 0.96, f"t = {t_sim:.2f} s", transform=ax.transAxes,
                color="#2a3a36", fontsize=8, va="top", fontfamily="monospace")
        if t_sim > 0:
            dx    = float(state.q[0]) - x_start_f
            speed = dx / t_sim
            ax.text(0.985, 0.96, f"v̄ = {speed:.3f} m/s",
                    transform=ax.transAxes, color="#2a3a36",
                    fontsize=8, va="top", ha="right", fontfamily="monospace")
        return []

    anim = animation.FuncAnimation(
        fig, animate, frames=len(stored),
        interval=int(frame_dt * 1000), blit=False,
    )

    try:
        anim.save(path, writer="pillow", fps=int(1.0 / frame_dt),
                  savefig_kwargs={"facecolor": BG})
        print(f"  GIF saved ({os.path.getsize(path) // 1024} KB)")
    except Exception as exc:
        print(f"  GIF export failed: {exc}")
    finally:
        plt.close(fig)


if __name__ == "__main__":
    main()
