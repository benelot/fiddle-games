"""
Muscle-driven planar hopper — Thelen 2003 Hill-type muscles in Brax/JAX.

Physics
-------
  • Brax generalized pipeline: rigid-body dynamics + MuJoCo contact model.
  • Three DOF: torso slide_z (vertical) + rooty (sagittal tipping) + knee hinge.
  • Lateral motion and roll/yaw are blocked; sagittal plane is free.
  • X position is fixed so contact geometry stays stable under the body.
  • Ground contact handled by Brax through the foot sphere geom.

Muscles
-------
  One antagonistic pair at the knee:
    Extensor — straightens the leg → pushes torso upward (positive thrust).
    Flexor   — bends the knee → pulls foot up during flight (leg recovery).
  Both modelled with Thelen 2003 CE + PEE, rigid tendon (same as worm).

Controller
----------
  Simple CPG oscillator at FREQ Hz:
    u_ext(t) = A · max(0,  sin(2π·f·t))   — fires during first half-cycle
    u_flex(t) = A · max(0, −sin(2π·f·t))  — fires during second half-cycle
  Activation dynamics smooth the bang-bang into a physiological ramp.

  This open-loop CPG is not adaptive, but it naturally entrains to the
  mechanical resonance of the hopper through its force-velocity coupling:
  extension is stronger while the leg is slower (eccentric phase of fv).

Run
---
    python ecology/worm/hopper.py

Outputs
-------
    ecology/worm/hopper_sim.html  — interactive Brax 3-D viewer
    ecology/worm/hopper_sim.gif   — Geijtenbeek-style 2-D visualization
"""

import os
import sys
import time
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="brax")

import jax
import jax.numpy as jnp
import numpy as np

from brax.io import mjcf, html
import brax.generalized.pipeline as pipeline

_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _DIR)
from muscle import MuscleParams, joint_torque, activation_ode

# ── Simulation constants ──────────────────────────────────────────────────────

DT           = 0.002   # physics timestep (s) — matches hopper.xml
SIM_DURATION = 8.0     # total simulation time (s)
RENDER_FPS   = 40      # frames/s stored for rendering

# ── Muscle parameters ─────────────────────────────────────────────────────────
#
# Scaled for a ~6 kg hopper. The key design choice:
#   F_max = 2000 N with r = 0.06 m → peak torque 120 N·m.
#   Hopper weight ≈ 62 N; shin+foot weight ≈ 13 N.
#   At knee angle = −0.5 rad (crouched) the ground reaction creates ≈35 N·m
#   of flexion moment at the knee that the extensor must overcome and exceed
#   to produce net upward impulse.
#
MUSCLE = MuscleParams(
    F_max        = 2000.0,  # N   — strong enough for visible hopping
    l_opt        = 0.10,    # m   — optimal CE fibre length
    l_slack      = 0.06,    # m   — rigid-tendon slack
    v_max        = 1.0,     # m/s — max shortening speed
    r            = 0.06,    # m   — moment arm at knee
    KshapeActive = 0.45,    # Thelen default Gaussian σ
    Af           = 0.25,    # Hill concentric shape
    Flen         = 1.4,     # eccentric force ceiling
    kpe          = 5.0,     # Thelen default PEE curvature
    e0           = 0.6,     # Thelen default PEE strain at F_max
    tau_act      = 0.015,   # s — activation rise  (Thelen)
    tau_deact    = 0.050,   # s — activation decay (Thelen)
)

# ── CPG controller ────────────────────────────────────────────────────────────

FREQ      = 1.8    # Hz — CPG oscillator frequency
AMPLITUDE = 0.95   # peak neural drive (< 1 for numerical stability)

# Soft joint-limit spring — prevents knee from slamming into its hard limit.
# Spring natural frequency ≈ √(K/I_eff) ≈ √(8/0.05) ≈ 13 rad/s >> 1/DT safe.
_LIMIT_RAD = 1.4    # just inside XML range="-1.5  0.2"  (0 side)
_K_LIMIT   = 8.0   # N·m/rad
_D_LIMIT   = 0.2   # N·m·s/rad


def _cpg_drive(t: float) -> jnp.ndarray:
    """
    CPG neural drives for [extensor, flexor].

    Positive half of sin → extensor fires (leg extends, thrust phase).
    Negative half of sin → flexor fires (knee bends, recovery phase).
    Returns shape (2,) with values in [0, AMPLITUDE].
    """
    phase = 2.0 * jnp.pi * FREQ * t
    u_ext  = AMPLITUDE * jnp.maximum(0.0,  jnp.sin(phase))
    u_flex = AMPLITUDE * jnp.maximum(0.0, -jnp.sin(phase))
    return jnp.stack([u_ext, u_flex])


@jax.jit
def _update_activation(a: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
    """Thelen 2003 variable-τ activation dynamics (vectorised over muscles)."""
    a_c = jnp.clip(a, 0.01, 1.0)
    tau = jnp.where(
        u > a_c,
        MUSCLE.tau_act   * (0.5 + 1.5 * a_c),
        MUSCLE.tau_deact / (0.5 + 1.5 * a_c),
    )
    return jnp.clip(a + (u - a_c) / tau * DT, 0.0, 1.0)


@jax.jit
def _compute_torque(q: jnp.ndarray, qd: jnp.ndarray,
                    a: jnp.ndarray) -> jnp.ndarray:
    """
    Map muscle activations to a scalar knee torque.

    q[0] = slide_z (m),  q[1] = rooty (rad),  q[2] = knee (rad)
    qd[0] = vz (m/s),   qd[1] = omega_root,  qd[2] = omega_knee
    a[0] = a_ext,        a[1] = a_flex

    Returns shape (1,) — one actuator (knee_motor).
    """
    theta     = q[2]
    theta_dot = qd[2]

    tau = joint_torque(
        a_flex    = a[1],
        a_ext     = a[0],
        theta     = theta,
        theta_dot = theta_dot,
        p         = MUSCLE,
    )

    # Soft damped spring at joint limits (prevents hard-limit bouncing)
    excess = jnp.maximum(0.0, jnp.abs(theta) - _LIMIT_RAD)
    past   = jnp.float32(excess > 0.0)
    tau    = tau - _K_LIMIT * excess * jnp.sign(theta)
    tau    = tau - past * _D_LIMIT * theta_dot

    return jnp.array([tau])


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    xml_path = os.path.join(_DIR, "hopper.xml")
    print(f"Loading {xml_path} …")
    sys_brax = mjcf.load(xml_path)

    # Initial state: torso at rest height, knee slightly bent (-0.3 rad)
    q0    = jnp.array([0.0, 0.0, -0.3])   # [slide_z, rooty, knee]
    qd0   = jnp.zeros(3)
    state = pipeline.init(sys_brax, q0, qd0)

    @jax.jit
    def step_fn(state, act):
        return pipeline.step(sys_brax, state, act)

    # JIT warm-up
    print("Compiling … (one-time cost)")
    t0 = time.time()
    _ = step_fn(state, jnp.zeros(1)).q.block_until_ready()
    print(f"  compile done in {time.time()-t0:.1f}s")

    n_steps    = int(SIM_DURATION / DT)
    keep_every = max(1, int(1.0 / (RENDER_FPS * DT)))

    a_state = jnp.array([0.0, 0.0])  # [a_ext, a_flex]
    stored  = [(state, a_state)]

    print(f"Simulating {SIM_DURATION}s ({n_steps} steps, {n_steps//keep_every} frames) …")
    t0 = time.time()

    for i in range(1, n_steps + 1):
        t_sim   = jnp.float32(i * DT)
        u       = _cpg_drive(t_sim)
        a_state = _update_activation(a_state, u)
        act     = _compute_torque(state.q, state.qd, a_state)
        state   = step_fn(state, act)
        if i % keep_every == 0:
            stored.append((state, a_state))
        if i % (n_steps // 10) == 0:
            q_np = np.array(state.q)
            z = float(q_np[0])
            root_deg = float(np.degrees(q_np[1]))
            knee_deg = float(np.degrees(q_np[2]))
            print(f"  {100*i//n_steps}%  t={i*DT:.1f}s  "
                  f"z_torso={z:+.3f}m  rooty={root_deg:+.1f}°  knee={knee_deg:+.1f}°",
                  flush=True)

    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s  ({n_steps/elapsed:.0f} steps/s, {len(stored)} frames)")

    z_vals = [float(s.q[0]) for s, _ in stored]
    print(f"Torso z range: {min(z_vals):+.3f} … {max(z_vals):+.3f} m "
          f"(peak rise above init: {max(z_vals):.3f} m)")

    # ── Brax 3-D HTML viewer ──────────────────────────────────────────────────
    html_path = os.path.join(_DIR, "hopper_sim.html")
    print(f"Writing 3-D viewer → {html_path}")
    html_str = html.render(sys_brax, [s for s, _ in stored], height=540, colab=False)
    with open(html_path, "w") as f:
        f.write(html_str)

    # ── 2-D Geijtenbeek-style GIF ─────────────────────────────────────────────
    gif_path = os.path.join(_DIR, "hopper_sim.gif")
    _render_gif(stored, gif_path, DT * keep_every)

    print(f"\nDone.\n  3-D viewer : {html_path}\n  GIF        : {gif_path}")


# ── Visualization ─────────────────────────────────────────────────────────────

def _quat_to_angle_y(quat: np.ndarray) -> float:
    """Extract rotation around the Y axis from a [w, x, y, z] quaternion."""
    w, qx, qy, qz = float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])
    return np.arctan2(2.0 * (w * qy + qx * qz),
                      1.0 - 2.0 * (qx * qx + qy * qy))


def _rotate_by_quat_y(vec_local: np.ndarray, quat: np.ndarray) -> np.ndarray:
    """Rotate a 3-vector by a [w,x,y,z] quaternion (XZ-plane only for rendering)."""
    angle = _quat_to_angle_y(quat)
    ca, sa = np.cos(angle), np.sin(angle)
    # Y-axis rotation: (x,y,z) → (ca*x+sa*z, y, -sa*x+ca*z)
    x, y, z = float(vec_local[0]), float(vec_local[1]), float(vec_local[2])
    return np.array([ca * x + sa * z, y, -sa * x + ca * z])


def _render_gif(stored, path: str, frame_dt: float):
    """
    Side-view 2-D visualization using faithful Brax world-frame positions.

    state.x.pos[i] = [x, y, z] world position of body i (torso=0, shin=1, foot=2)
    state.x.rot[i] = [w, x, y, z] world-frame quaternion of body i
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    import matplotlib.cm as cm
    from matplotlib.patches import Polygon, Circle

    print(f"Rendering GIF ({len(stored)} frames) → {path}")

    BG       = "#f0f2f5"
    BODY_COL = "#7a8fb5"
    BODY_EDG = "#4a5a80"
    FOOT_COL = "#5a6a90"

    TORSO_HL = 0.15   # capsule half-length
    TORSO_R  = 0.05
    SHIN_HL  = 0.25
    SHIN_R   = 0.04
    FOOT_R   = 0.07

    def _capsule_pts(cx, cz, hl, r, angle, n=14):
        """Screen polygon for a 2-D capsule viewed from the side."""
        tr = np.linspace(-np.pi/2, np.pi/2, n)
        tl = np.linspace(np.pi/2, 3*np.pi/2, n)
        lx = np.concatenate([hl + r*np.cos(tr), -hl + r*np.cos(tl)])
        lz = np.concatenate([r*np.sin(tr), r*np.sin(tl)])
        ca, sa = np.cos(angle), np.sin(angle)
        return cx + ca*lx - sa*lz, cz + sa*lx + ca*lz

    def _spindle_pts(p0, p1, belly):
        """Geijtenbeek-style lozenge spindle for one muscle."""
        dx, dz = p1[0]-p0[0], p1[1]-p0[1]
        L = max(float(np.hypot(dx, dz)), 1e-6)
        nx, nz = -dz/L, dx/L
        mx, mz = (p0[0]+p1[0])/2, (p0[1]+p1[1])/2
        t  = np.linspace(0, 1, 20)
        tx = (1-t)**2*p0[0] + 2*(1-t)*t*(mx+nx*belly) + t**2*p1[0]
        tz = (1-t)**2*p0[1] + 2*(1-t)*t*(mz+nz*belly) + t**2*p1[1]
        bx = (1-t)**2*p0[0] + 2*(1-t)*t*(mx-nx*belly) + t**2*p1[0]
        bz = (1-t)**2*p0[1] + 2*(1-t)*t*(mz-nz*belly) + t**2*p1[1]
        return (np.concatenate([tx, bx[::-1]]),
                np.concatenate([tz, bz[::-1]]))

    # Compute x-range for a scrolling camera (torso=body 0 in 2-link system)
    all_torso_x = []
    for s, _ in stored:
        pos = np.array(s.x.pos)   # [2, 3]: torso, shin
        all_torso_x.append(float(pos[0, 0]))

    fig, ax = plt.subplots(figsize=(5, 7), facecolor=BG)
    ax.set_facecolor(BG)
    ax.set_aspect("equal")
    ax.axis("off")

    def animate(fi):
        ax.cla()
        ax.set_facecolor(BG)
        ax.set_aspect("equal")
        ax.axis("off")

        state, a_act = stored[fi]
        # ── Faithful world-frame positions from Brax ──────────────────────────
        # state.x.pos has 2 entries: [0]=torso, [1]=shin
        # (foot body has no joints, so it's folded into the 2-link system;
        #  foot geom IS active for contact, but we compute its visual pos manually)
        pos = np.array(state.x.pos)   # [2, 3]: torso, shin
        rot = np.array(state.x.rot)   # [2, 4]: quaternions [w,x,y,z]
        q   = np.array(state.q)       # [slide_x, slide_z, rooty, knee]
        a_np = np.array(a_act)        # [a_ext, a_flex]
        t_sim = fi * frame_dt

        # Body centres in the X-Z (sagittal) plane
        torso_x, torso_z = float(pos[0, 0]), float(pos[0, 2])
        shin_x,  shin_z  = float(pos[1, 0]), float(pos[1, 2])

        # Orientation angles around Y axis (for capsule tilt)
        torso_angle_y = _quat_to_angle_y(rot[0])   # sagittal tipping of torso
        shin_angle_y  = _quat_to_angle_y(rot[1])   # shin orientation

        # Foot: attached at (0, 0, -0.50) in shin body frame — rotate into world
        foot_offset_world = _rotate_by_quat_y(np.array([0.0, 0.0, -0.50]), rot[1])
        foot_x = shin_x + float(foot_offset_world[0])
        foot_z = shin_z + float(foot_offset_world[2])

        # Knee joint position = shin body origin (hinge is at shin body frame origin)
        knee_x = shin_x
        knee_z = shin_z

        # Camera: scroll to keep torso centred
        cam_x = torso_x
        def cx(x): return x - cam_x   # camera-relative x

        # ── Ground ──
        ax.axhline(0, color="#c0c0c0", lw=1.5, zorder=1)
        ax.fill_between([-0.55, 0.55], [0, 0], [-0.05, -0.05],
                        color="#d8d8d8", zorder=1)

        # ── Muscle spindles at knee (in camera frame) ──────────────────────────
        R_MOM = MUSCLE.r
        kn_ca, kn_sa = np.cos(shin_angle_y), np.sin(shin_angle_y)
        # Extensor: anterior (positive x side relative to shin)
        ext_prox = np.array([cx(knee_x) + R_MOM * kn_ca,   knee_z + 0.05])
        ext_dist = np.array([cx(knee_x) + R_MOM * kn_ca,   knee_z - 0.05])
        # Flexor: posterior (negative x side)
        flex_prox = np.array([cx(knee_x) - R_MOM * kn_ca,  knee_z + 0.05])
        flex_dist = np.array([cx(knee_x) - R_MOM * kn_ca,  knee_z - 0.05])

        for prox, dist, act_val in [(ext_prox, ext_dist, a_np[0]),
                                    (flex_prox, flex_dist, a_np[1])]:
            belly = 0.004 + 0.020 * float(act_val)
            sx, sz = _spindle_pts(prox, dist, belly)
            color = cm.jet(float(act_val))[:3]
            alpha = 0.5 + 0.5 * float(act_val)
            ax.add_patch(Polygon(np.column_stack([sx, sz]),
                                 facecolor=color, alpha=alpha,
                                 edgecolor=color, lw=0.3, zorder=3))

        # ── Skeleton line (camera frame) ──
        ax.plot([cx(torso_x), cx(knee_x), cx(foot_x)],
                [torso_z,      knee_z,      foot_z],
                color="#8090b0", lw=1.5, zorder=2)

        # ── Torso capsule — tilted by rooty ──
        # Capsule long-axis is vertical when torso_angle_y=0; np.pi/2 = vertical
        torso_cap_angle = np.pi/2 + torso_angle_y
        tx, tz = _capsule_pts(cx(torso_x), torso_z,
                              TORSO_HL, TORSO_R, torso_cap_angle)
        ax.add_patch(Polygon(np.column_stack([tx, tz]),
                             facecolor=BODY_COL, edgecolor=BODY_EDG, lw=0.8,
                             alpha=0.9, zorder=4))

        # ── Shin capsule — tilted by shin world angle ──
        shin_cap_angle = np.pi/2 + shin_angle_y
        sx2, sz2 = _capsule_pts(cx(shin_x), shin_z,
                                SHIN_HL, SHIN_R, shin_cap_angle)
        ax.add_patch(Polygon(np.column_stack([sx2, sz2]),
                             facecolor=BODY_COL, edgecolor=BODY_EDG, lw=0.8,
                             alpha=0.9, zorder=4))

        # ── Foot circle ──
        ax.add_patch(Circle((cx(foot_x), foot_z), FOOT_R,
                            color=FOOT_COL, zorder=5))

        # ── Knee pivot dot ──
        ax.plot(cx(knee_x), knee_z, 'o', color="#334060", ms=6, zorder=6)

        # ── HUD ──
        ax.set_xlim(-0.50, 0.50)
        ax.set_ylim(-0.05, 1.60)
        z_disp = float(q[0])
        ax.text(-0.47, 1.53, f"t = {t_sim:.2f} s", fontsize=8,
                color="#334060", fontfamily="monospace", va="top")
        ax.text(-0.47, 1.43, f"z = {z_disp:+.3f} m", fontsize=8,
                color="#334060", fontfamily="monospace", va="top")
        ax.text(-0.47, 1.33,
                f"a_ext={a_np[0]:.2f}  a_flex={a_np[1]:.2f}", fontsize=7,
                color="#334060", fontfamily="monospace", va="top")
        return []

    anim = animation.FuncAnimation(
        fig, animate, frames=len(stored),
        interval=int(frame_dt * 1000), blit=False,
    )

    try:
        anim.save(path, writer="pillow", fps=int(1.0/frame_dt),
                  savefig_kwargs={"facecolor": BG})
        print(f"  GIF saved ({os.path.getsize(path)//1024} KB)")
    except Exception as exc:
        print(f"  GIF export failed: {exc}")
    finally:
        plt.close(fig)


if __name__ == "__main__":
    main()
