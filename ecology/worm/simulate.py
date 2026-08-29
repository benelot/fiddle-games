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
    F_max   = 4.0,   # N  — moderate force for a ~50g worm
    l_opt   = 0.08,  # m  — optimal fibre length
    l_slack = 0.05,  # m  — tendon slack length
    v_max   = 0.6,   # m/s
    r       = 0.022, # m  — moment arm
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
    """Top-down 2D animated GIF via matplotlib."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    import matplotlib.patches as mpatches

    print(f"Rendering GIF ({len(states)} frames) → {path}")

    BG      = "#111a14"
    COLORS  = ["#4dd6c4", "#3dc2b0", "#2dae9c", "#1d9a88"]
    RADII   = [0.040, 0.036, 0.031, 0.025]
    CONNECT = "#2a9a8a"

    fig, ax = plt.subplots(figsize=(9, 4.5), facecolor=BG)
    ax.set_facecolor(BG)
    ax.set_aspect("equal")
    ax.axis("off")

    window_w = 2.0
    window_h = 1.0

    circles = [plt.Circle((0, 0), RADII[j], color=COLORS[j], zorder=5)
               for j in range(N_SEGMENTS)]
    for c in circles:
        ax.add_patch(c)

    conn_lines = [ax.plot([], [], color=CONNECT, lw=4, zorder=4, solid_capstyle="round")[0]
                  for _ in range(N_JOINTS)]

    time_txt = ax.text(0.02, 0.95, "", transform=ax.transAxes,
                       color="white", fontsize=9, va="top",
                       fontfamily="monospace")

    def get_xy(state):
        """Return (4, 2) array of XY positions for each body."""
        return np.array(state.x.pos[:, :2])  # (n_links, 3) → take XY

    def animate(frame_idx):
        state = states[frame_idx]
        xy = get_xy(state)                       # shape (4, 2)

        cx = float(xy[0, 0])                     # follow head
        ax.set_xlim(cx - 0.3 * window_w, cx + 0.7 * window_w)
        ax.set_ylim(-window_h / 2, window_h / 2)

        for j, circle in enumerate(circles):
            circle.center = (float(xy[j, 0]), float(xy[j, 1]))

        for k, line in enumerate(conn_lines):
            line.set_data(
                [float(xy[k, 0]), float(xy[k + 1, 0])],
                [float(xy[k, 1]), float(xy[k + 1, 1])],
            )

        t = frame_idx * frame_dt
        time_txt.set_text(f"t = {t:.2f}s")
        return circles + conn_lines + [time_txt]

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
