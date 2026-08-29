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

    # Geometry (halfLen, radius) matching worm.xml:
    #   seg0: fromto="-0.09 0 0 0.09 0 0"  → symmetric, center at frame origin
    #   seg1: fromto="0 0 0 0.15 0 0"      → one-sided; center 0.075m ahead of joint
    #   seg2: fromto="0 0 0 0.13 0 0"      → center 0.065m ahead of joint
    #   seg3: fromto="0 0 0 0.11 0 0"      → center 0.055m ahead of joint
    SEGS = [(0.090, 0.040), (0.075, 0.036), (0.065, 0.031), (0.055, 0.025)]
    SEG_COLORS = ["#59d1bc", "#4dbfaa", "#40ad98", "#339b87"]
    R_MOM = MUSCLE.r       # moment arm (m)
    ATTACH_D = 0.022       # muscle attachment offset along body from the pivot (m)
    VIS_RADII = [r * 0.60 for _, r in SEGS]

    # Hill force-length for visual weight (isometric, rigid tendon)
    def _force_frac(act, p0, p1):
        lmtu = float(np.hypot(p1[0]-p0[0], p1[1]-p0[1]))
        lCE  = max(lmtu - MUSCLE.l_slack, 0.01 * MUSCLE.l_opt)
        fl   = float(np.exp(-((lCE / MUSCLE.l_opt - 1.0) / MUSCLE.w) ** 2))
        return float(act) * fl

    def _capsule_xy(cx, cy, hl, r, angle, n=10):
        """Polygon for a capsule centred at (cx,cy), half-length hl, radius r, rotated by angle."""
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

        state  = states[frame_idx]
        q      = np.array(state.q)
        t_sim  = frame_idx * frame_dt

        # state.x.pos[k] = frame origin of body k in world:
        #   k=0 → seg0 center (symmetric geom)
        #   k=1 → j0 pivot  = right end of seg0 = left end of seg1
        #   k=2 → j1 pivot  = right end of seg1 = left end of seg2
        #   k=3 → j2 pivot  = right end of seg2 = left end of seg3
        jp = np.array(state.x.pos[:, :2])   # (4, 2)  joint/origin positions

        # World orientation of each segment (cumulative joint angles)
        angles = [float(q[2])]
        for k in range(3):
            angles.append(angles[-1] + float(q[3 + k]))

        # Body centre positions (for capsule drawing and camera)
        # seg0 is symmetric → COM = frame origin
        # seg k≥1 → COM is HALF_LEN ahead of the joint pivot along seg k's axis
        hl_arr = [s[0] for s in SEGS]  # [0.09, 0.075, 0.065, 0.055]
        centers = [jp[0].copy()]
        for k in range(1, N_SEGMENTS):
            a = angles[k]
            centers.append(jp[k] + hl_arr[k] * np.array([np.cos(a), np.sin(a)]))

        # Camera: follow midpoint between seg0 COM and far end of seg3
        a3 = angles[3]
        tail_end = jp[3] + 2*hl_arr[3] * np.array([np.cos(a3), np.sin(a3)])
        cx_mid   = float((centers[0][0] + tail_end[0]) / 2)
        ax.set_xlim(cx_mid - 0.55*window_w, cx_mid + 0.45*window_w)
        ax.set_ylim(-window_h/2, window_h/2)

        # CPG activations at this time
        acts = []
        for k in range(N_JOINTS):
            phase = 2.0*np.pi*FREQ*t_sim - k*PHASE_OFFSET
            acts.append(AMPLITUDE * max(0.0,  np.sin(phase)))   # flexor
            acts.append(AMPLITUDE * max(0.0, -np.sin(phase)))   # extensor

        # Skeleton through COMs
        ax.plot([c[0] for c in centers], [c[1] for c in centers],
                color="#1a3028", lw=2.0, zorder=2, solid_capstyle="round")

        # Draw muscles (behind bodies)
        # Muscle at joint k connects seg k (parent) and seg k+1 (child).
        # Pivot = jp[k+1].
        # Attachment on seg k:   ATTACH_D *behind* the pivot along seg k's axis
        # Attachment on seg k+1: ATTACH_D *ahead*  of the pivot along seg k+1's axis
        for k in range(N_JOINTS):
            pivot = jp[k+1]          # world position of hinge pivot
            a_k  = angles[k]         # seg k (parent) world angle
            a_k1 = angles[k+1]       # seg k+1 (child) world angle
            axis_k  = np.array([ np.cos(a_k),   np.sin(a_k)])
            axis_k1 = np.array([ np.cos(a_k1),  np.sin(a_k1)])
            perp_k  = np.array([-np.sin(a_k),   np.cos(a_k)])
            perp_k1 = np.array([-np.sin(a_k1),  np.cos(a_k1)])

            for side, color, a_act in [
                (+1, "#d94040", acts[2*k]),     # flexor  — red, +perp
                (-1, "#3070c8", acts[2*k+1]),   # extensor — blue, -perp
            ]:
                # prox: inside seg k, behind the pivot
                prox = pivot - ATTACH_D*axis_k  + side*R_MOM*perp_k
                # dist: inside seg k+1, ahead of the pivot
                dist = pivot + ATTACH_D*axis_k1 + side*R_MOM*perp_k1

                ff    = _force_frac(a_act, prox, dist)
                belly = 0.003 + 0.018*np.sqrt(max(0.0, ff))
                alpha = 0.35 + 0.65*a_act

                sx, sy = _spindle_xy(prox, dist, belly)
                ax.add_patch(Polygon(np.column_stack([sx, sy]), closed=True,
                                     facecolor=color, alpha=alpha,
                                     edgecolor=color, linewidth=0.3, zorder=3))

        # Draw body capsules (using COMs + visual radii)
        for j in range(N_SEGMENTS):
            hl = hl_arr[j]
            cx, cy = float(centers[j][0]), float(centers[j][1])
            xs, ys = _capsule_xy(cx, cy, hl, VIS_RADII[j], angles[j])
            ax.add_patch(Polygon(np.column_stack([xs, ys]), closed=True,
                                 facecolor=SEG_COLORS[j], alpha=0.80,
                                 edgecolor="#a0e8d4", linewidth=0.9, zorder=4))

        # Joint pivot markers at the actual boundary between segments
        for k in range(N_JOINTS):
            ax.plot(float(jp[k+1, 0]), float(jp[k+1, 1]),
                    'o', color='#ffffff', ms=5.0, zorder=6, alpha=0.95,
                    markeredgecolor="#88e8d6", markeredgewidth=0.7)

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
