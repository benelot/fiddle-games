"""
Worm locomotion simulation — Geijtenbeek-style muscle-tendon model in Brax/JAX.

Run:
    python ecology/worm/simulate.py

Outputs:
    ecology/worm/worm_sim.html   — interactive Three.js viewer (open in browser)
    ecology/worm/worm_sim.gif    — top-down animated GIF
"""

import os
import time
import warnings
import sys as _sys

# Suppress Brax deprecation noise
warnings.filterwarnings("ignore", category=UserWarning, module="brax")

import jax
import jax.numpy as jnp
import numpy as np

from brax.io import mjcf, html
import brax.generalized.pipeline as pipeline

# ---------------------------------------------------------------------------
# Monkey-patch: brax/fluid.py uses jnp.clip(a_min=...) which was removed in
# JAX 0.4.x. Replace the 'force' function with a corrected copy.
# ---------------------------------------------------------------------------
import brax.fluid as _brax_fluid
from brax.base import Force, Motion, Transform
import jax as _jax

def _patched_fluid_force(sys, x, xd, mass, inertia, root_com=None):
    x_i = x.vmap().do(sys.link.inertia.transform)
    offset = x_i.pos - x.pos if root_com is None else x_i.pos - root_com
    xd_i = x_i.replace(pos=offset).vmap().do(xd)

    diag_inertia = _jax.vmap(jnp.diag)(inertia)
    diag_inertia_v = jnp.repeat(diag_inertia, 3, axis=-2).reshape((-1, 3, 3))
    diag_inertia_v *= jnp.ones((3, 3)) - 2 * jnp.eye(3)
    # Fixed: min= instead of deprecated a_min=
    box = 6.0 * jnp.clip(jnp.sum(diag_inertia_v, axis=-1), min=1e-12)
    box = jnp.sqrt(box / mass[:, None])

    frc = _brax_fluid._box_viscosity(box, xd_i, sys.viscosity)
    frc += _brax_fluid._box_density(box, xd_i, sys.density)
    frc = Transform.create(rot=x_i.rot).vmap().do(frc)
    return frc

_brax_fluid.force = _patched_fluid_force
# Also patch the reference inside the dynamics module (already imported)
import brax.generalized.dynamics as _brax_dyn
_brax_dyn.fluid = _brax_fluid

# Add parent dir so muscle.py resolves when run from repo root
_WORM_DIR = os.path.dirname(os.path.abspath(__file__))
_sys.path.insert(0, _WORM_DIR)
from muscle import MuscleParams, joint_torque

# ---------------------------------------------------------------------------
# Simulation constants
# ---------------------------------------------------------------------------
DT            = 0.002   # physics timestep (s)
SIM_DURATION  = 10.0    # total simulation time (s)
RENDER_FPS    = 40      # frames per second stored for rendering
N_JOINTS      = 3
N_SEGMENTS    = 4
# Fluid forces come from worm.xml option density+viscosity (Brax built-in).
# No manual drag needed.

# ---------------------------------------------------------------------------
# Muscle parameters — same for all three joints (tuned for worm scale)
# ---------------------------------------------------------------------------
MUSCLE = MuscleParams(
    F_max   = 18.0,  # N  — strong enough to drive clear undulation
    l_opt   = 0.08,  # m  — optimal fibre length
    l_slack = 0.05,  # m  — tendon slack length
    v_max   = 0.6,   # m/s
    r       = 0.040, # m  — moment arm
    w       = 0.56,
    k_pee   = 0.35,
    pee_slack = 1.0,
)

# ---------------------------------------------------------------------------
# Controller: sinusoidal traveling wave (head → tail)
#
# Rectified cosines produce smooth "burst" activations for each muscle:
#   flexor  activation at joint i: A · max(0,  sin(2πft – i·φ))
#   extensor activation at joint i: A · max(0, –sin(2πft – i·φ))
#
# Phase offset φ ≈ π/3 creates a wave that propagates rearward.
# ---------------------------------------------------------------------------
FREQ         = 1.3    # Hz
PHASE_OFFSET = -1.15  # rad; negative → wave propagates tail→head → body moves +X
AMPLITUDE    = 0.85   # peak activation


def _cpg_activations(t: jnp.ndarray) -> jnp.ndarray:
    """Return [a_flex0, a_ext0, a_flex1, a_ext1, a_flex2, a_ext2]."""
    acts = []
    for i in range(N_JOINTS):
        phase = 2.0 * jnp.pi * FREQ * t - i * PHASE_OFFSET
        acts.append(AMPLITUDE * jnp.maximum(0.0, jnp.sin(phase)))   # flexor
        acts.append(AMPLITUDE * jnp.maximum(0.0, -jnp.sin(phase)))  # extensor
    return jnp.stack(acts)


def _compute_act(q: jnp.ndarray, qd: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
    """
    Full muscle-torque computation.

    New 6-DOF layout (slide_x, slide_y, root_rot, j0, j1, j2):
      q[3:6]  — hinge joint angles (rad)
      qd[3:6] — hinge joint velocities (rad/s)
    Returns act of shape (3,) — one torque per motor actuator.
    """
    acts = _cpg_activations(t)
    theta  = q[3:6]
    theta_d = qd[3:6]

    torques = []
    for i in range(N_JOINTS):
        tau = joint_torque(
            a_flex   = acts[2 * i],
            a_ext    = acts[2 * i + 1],
            theta    = theta[i],
            theta_dot = theta_d[i],
            p        = MUSCLE,
        )
        torques.append(tau)
    return jnp.stack(torques)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    xml_path = os.path.join(_WORM_DIR, "worm.xml")
    print(f"Loading {xml_path} …")
    sys = mjcf.load(xml_path)
    # timestep is already 0.002 from worm.xml (opt.timestep)

    # Initial state: worm straight along X, elevated just above the ground
    # q layout: [x, y, root_rot, j0, j1, j2]  (6 DOF, matching swimmer.xml pattern)
    q0  = jnp.zeros(sys.q_size())
    qd0 = jnp.zeros(sys.qd_size())

    state = pipeline.init(sys, q0, qd0)

    @jax.jit
    def step_fn(state, t):
        act = _compute_act(state.q, state.qd, t)
        # Brax fluid forces (from option density+viscosity) handle drag.
        return pipeline.step(sys, state, act)

    # Warm-up JIT compile
    print("Compiling … (one-time cost)")
    t0 = time.time()
    _ = step_fn(state, jnp.float32(0.0)).q.block_until_ready()
    print(f"  compile done in {time.time()-t0:.1f}s")

    # Simulation rollout
    n_steps      = int(SIM_DURATION / DT)
    keep_every   = max(1, int(1.0 / (RENDER_FPS * DT)))
    stored_states = [state]

    print(f"Simulating {SIM_DURATION}s ({n_steps} steps, {n_steps//keep_every} frames) …")
    t0 = time.time()

    for i in range(1, n_steps + 1):
        state = step_fn(state, jnp.float32(i * DT))
        if i % keep_every == 0:
            stored_states.append(state)
        if i % (n_steps // 10) == 0:
            print(f"  {100*i//n_steps}%  sim_t={i*DT:.1f}s", flush=True)

    elapsed = time.time() - t0
    n_frames = len(stored_states)
    print(f"Done in {elapsed:.1f}s  ({n_steps/elapsed:.0f} steps/s, {n_frames} frames)")

    # --- displacement report ---
    x_start = float(stored_states[0].q[0])  # slider_x position
    x_end   = float(state.q[0])
    print(f"Displacement along X: {x_end - x_start:.3f} m over {SIM_DURATION}s")

    # --- HTML viewer ---
    html_path = os.path.join(_WORM_DIR, "worm_sim.html")
    print(f"Writing HTML viewer → {html_path}")
    html_str = html.render(sys, stored_states, height=540, colab=False)
    with open(html_path, "w") as f:
        f.write(html_str)

    # --- GIF (top-down view) ---
    gif_path = os.path.join(_WORM_DIR, "worm_sim.gif")
    _render_gif(stored_states, gif_path, DT * keep_every)

    print("\nDone.")
    print(f"  Open {html_path} in your browser for the 3D viewer.")
    print(f"  GIF: {gif_path}")


def _render_gif(states, path: str, frame_dt: float):
    """Top-down 2D animated GIF — Geijtenbeek-style capsule bodies + Hill muscle spindles."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib.patches import Polygon

    print(f"Rendering GIF ({len(states)} frames) → {path}")

    # Body geometry from worm.xml (halfLen, radius)
    SEGS = [(0.090, 0.040), (0.075, 0.036), (0.065, 0.031), (0.055, 0.025)]
    SEG_COLORS = ["#59d1bc", "#4dbfaa", "#40ad98", "#339b87"]
    R_MOM = MUSCLE.r   # moment arm (m)

    # Hill force-length for visual weight (isometric, rigid tendon)
    def _force_frac(act, p0, p1):
        lmtu = float(np.hypot(p1[0]-p0[0], p1[1]-p0[1]))
        lCE  = max(lmtu - MUSCLE.l_slack, 0.01 * MUSCLE.l_opt)
        fl   = float(np.exp(-((lCE / MUSCLE.l_opt - 1.0) / MUSCLE.w) ** 2))
        return float(act) * fl

    def _capsule_xy(cx, cy, hl, r, angle, n=10):
        tr = np.linspace(-np.pi/2, np.pi/2, n)
        tl = np.linspace(np.pi/2, 3*np.pi/2, n)
        lx = np.concatenate([hl + r*np.cos(tr), -hl + r*np.cos(tl)])
        ly = np.concatenate([r*np.sin(tr), r*np.sin(tl)])
        ca, sa = np.cos(angle), np.sin(angle)
        return cx + ca*lx - sa*ly, cy + sa*lx + ca*ly

    def _spindle_xy(p0, p1, belly):
        dx, dy = p1[0]-p0[0], p1[1]-p0[1]
        L  = max(float(np.hypot(dx, dy)), 1e-6)
        px, py = -dy/L, dx/L
        mx, my = (p0[0]+p1[0])*0.5, (p0[1]+p1[1])*0.5
        t = np.linspace(0, 1, 14)
        tx = (1-t)**2*p0[0] + 2*(1-t)*t*(mx+px*belly) + t**2*p1[0]
        ty = (1-t)**2*p0[1] + 2*(1-t)*t*(my+py*belly) + t**2*p1[1]
        bx = (1-t)**2*p0[0] + 2*(1-t)*t*(mx-px*belly) + t**2*p1[0]
        by = (1-t)**2*p0[1] + 2*(1-t)*t*(my-py*belly) + t**2*p1[1]
        return np.concatenate([tx, bx[::-1]]), np.concatenate([ty, by[::-1]])

    # Visual radii: thinner than physics so joint gaps are apparent
    VIS_RADII = [r * 0.55 for _, r in SEGS]

    BG = "#0d1a12"
    fig, ax = plt.subplots(figsize=(9, 5), facecolor=BG)
    ax.set_facecolor(BG)
    ax.set_aspect("equal")
    ax.axis("off")

    window_w = 0.85
    window_h = 0.55

    def animate(frame_idx):
        ax.cla()
        ax.set_facecolor(BG)
        ax.set_aspect("equal")
        ax.axis("off")

        state   = states[frame_idx]
        q       = np.array(state.q)
        xy      = np.array(state.x.pos[:, :2])   # (4, 2) body centers in world
        t_sim   = frame_idx * frame_dt

        # World orientation of each segment (cumulative joint angles)
        angles = [float(q[2])]
        for k in range(3):
            angles.append(angles[-1] + float(q[3 + k]))

        # Camera follows worm midpoint
        cx_mid = float((xy[0, 0] + xy[-1, 0]) / 2)
        ax.set_xlim(cx_mid - 0.55*window_w, cx_mid + 0.45*window_w)
        ax.set_ylim(-window_h/2, window_h/2)

        # CPG activations at this time
        acts = []
        for k in range(N_JOINTS):
            phase = 2.0*np.pi*FREQ*t_sim - k*PHASE_OFFSET
            acts.append(AMPLITUDE * max(0.0,  np.sin(phase)))   # flexor
            acts.append(AMPLITUDE * max(0.0, -np.sin(phase)))   # extensor

        # Skeleton — dark centerline shows the bend shape clearly
        ax.plot([float(xy[j, 0]) for j in range(N_SEGMENTS)],
                [float(xy[j, 1]) for j in range(N_SEGMENTS)],
                color="#1a3028", lw=2.5, zorder=2, solid_capstyle="round")

        # Draw muscles (behind bodies)
        for k in range(N_JOINTS):
            hl0, _ = SEGS[k]
            hl1, _ = SEGS[k+1]
            a0, a1 = angles[k], angles[k+1]
            ca0, sa0 = np.cos(a0), np.sin(a0)
            ca1, sa1 = np.cos(a1), np.sin(a1)
            cx0, cy0 = float(xy[k,   0]), float(xy[k,   1])
            cx1, cy1 = float(xy[k+1, 0]), float(xy[k+1, 1])
            perp0 = np.array([-sa0, ca0])
            perp1 = np.array([-sa1, ca1])

            for side, color, a_act in [
                (+1, "#d94040", acts[2*k]),     # flexor  — red, +perp
                (-1, "#3070c8", acts[2*k+1]),   # extensor — blue, -perp
            ]:
                prox = np.array([cx0 + hl0*ca0 + side*R_MOM*perp0[0],
                                 cy0 + hl0*sa0 + side*R_MOM*perp0[1]])
                dist = np.array([cx1 - hl1*ca1 + side*R_MOM*perp1[0],
                                 cy1 - hl1*sa1 + side*R_MOM*perp1[1]])

                ff    = _force_frac(a_act, prox, dist)
                belly = 0.004 + 0.016*np.sqrt(max(0.0, ff))
                alpha = 0.40 + 0.60*a_act

                sx, sy = _spindle_xy(prox, dist, belly)
                ax.add_patch(Polygon(np.column_stack([sx, sy]), closed=True,
                                     facecolor=color, alpha=alpha,
                                     edgecolor=color, linewidth=0.3, zorder=3))

        # Draw body capsules with thinner visual radius (shows joint gaps)
        for j in range(N_SEGMENTS):
            hl, _ = SEGS[j]
            xs, ys = _capsule_xy(float(xy[j,0]), float(xy[j,1]), hl, VIS_RADII[j], angles[j])
            ax.add_patch(Polygon(np.column_stack([xs, ys]), closed=True,
                                 facecolor=SEG_COLORS[j], alpha=0.80,
                                 edgecolor="#a0e8d4", linewidth=0.9, zorder=4))

        # Joint pivot markers — show the hinge center
        for k in range(N_JOINTS):
            ax.plot(float(xy[k+1, 0]), float(xy[k+1, 1]),
                    'o', color='#ffffff', ms=4.5, zorder=6, alpha=0.9,
                    markeredgecolor="#88e8d6", markeredgewidth=0.6)

        ax.text(0.02, 0.95, f"t = {t_sim:.2f}s", transform=ax.transAxes,
                color="#6ab89a", fontsize=9, va="top", fontfamily="monospace")
        return []

    anim = animation.FuncAnimation(
        fig, animate, frames=len(states),
        interval=int(frame_dt * 1000), blit=False,
    )

    try:
        anim.save(path, writer="pillow", fps=int(1.0 / frame_dt),
                  savefig_kwargs={"facecolor": BG})
        print(f"  GIF saved ({os.path.getsize(path)//1024} KB)")
    except Exception as exc:
        print(f"  GIF export failed: {exc}")
    finally:
        plt.close(fig)


if __name__ == "__main__":
    main()
