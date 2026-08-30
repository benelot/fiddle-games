"""
Hill-type muscle-tendon model — Thelen 2003 formulation.

References:
  Thelen 2003 "Adjustment of Muscle Mechanics Model Parameters to Simulate
    Dynamic Contractions in Older Adults"
  Geijtenbeek 2013 "Flexible Muscle-Based Locomotion for Bipedal Creatures"
  OpenSim Thelen2003Muscle.cpp  (source for numerical constants)

Architecture per joint (used by joint_torque):
  Two antagonistic muscles — flexor and extensor — each with:
    CE : contractile element  (active force, gated by fl × fv)
    PEE: parallel elastic     (passive resistance when stretched)
  Rigid tendon: l_CE = l_MTU − l_slack  (no implicit solve needed, JIT-friendly)
  Net joint torque = moment_arm × (F_flex − F_ext)
"""

import jax.numpy as jnp
from typing import NamedTuple


class MuscleParams(NamedTuple):
    """Thelen 2003 parameter bundle for one antagonistic muscle pair at a joint."""
    F_max:   float = 20.0    # max isometric force (N)
    l_opt:   float = 0.10    # optimal CE fibre length (m)
    l_slack: float = 0.06    # tendon slack length (m); rigid: l_CE = MTU − l_slack
    v_max:   float = 1.0     # max shortening speed (m/s); Thelen: 10 × l_opt/s
    r:       float = 0.025   # moment arm (m) — used only by joint_torque

    # Active force-length: Gaussian σ (Thelen KshapeActive = 0.45)
    KshapeActive: float = 0.45

    # Force-velocity shape (Thelen / Hill)
    Af:   float = 0.25   # concentric Hill shape constant
    Flen: float = 1.4    # eccentric force ceiling (× F_iso)

    # Passive force-length: Thelen 2003 exponential
    kpe: float = 5.0    # exponential curvature
    e0:  float = 0.6    # PEE strain at F_iso (fraction of l_opt)

    # Activation dynamics (Thelen 2003 with variable-τ modulation)
    tau_act:   float = 0.015   # rise  time-constant (s)
    tau_deact: float = 0.050   # decay time-constant (s)


# ── Force-length ──────────────────────────────────────────────────────────────

def _fl(l_ce: jnp.ndarray, l_opt: float, KshapeActive: float) -> jnp.ndarray:
    """Active force-length: Gaussian centred at l_opt (Thelen 2003)."""
    lceN = l_ce / l_opt
    return jnp.exp(-((lceN - 1.0) / KshapeActive) ** 2)


# ── Force-velocity ────────────────────────────────────────────────────────────

def _fv(v_ce: jnp.ndarray, v_max: float, Af: float, Flen: float) -> jnp.ndarray:
    """
    Force-velocity relationship (Thelen 2003).

    Concentric (v_ce ≤ 0, shortening):
      fv = (1 + v̄) / (1 − v̄/Af)        v̄ = v_ce/v_max ∈ [−1, 0]

    Eccentric (v_ce > 0, lengthening) — derived from Thelen 2003 eq. 6:
      c  = (2 + 2/Af) / (Flen − 1)
      fv = (1 + c·Flen·v̄) / (1 + c·v̄)  → fv(0)=1, fv(∞)→Flen
    """
    v_bar = v_ce / v_max

    # Concentric
    f_conc = (1.0 + v_bar) / (1.0 - v_bar / Af)
    f_conc = jnp.clip(f_conc, 0.0, 1.0)

    # Eccentric (Thelen rational form)
    c = (2.0 + 2.0 / Af) / (Flen - 1.0)
    f_ecc = (1.0 + c * Flen * v_bar) / (1.0 + c * v_bar)
    f_ecc = jnp.clip(f_ecc, 1.0, Flen)

    return jnp.where(v_ce <= 0.0, f_conc, f_ecc)


# ── Passive force-length (PEE) ────────────────────────────────────────────────

def _pee(l_ce: jnp.ndarray, l_opt: float, kpe: float, e0: float) -> jnp.ndarray:
    """
    Passive force-length — Thelen 2003 exponential (eq. 3).
    fpe = (exp(kpe·(lceN−1)/e0) − 1) / (exp(kpe) − 1)   for lceN > 1, else 0

    Returns normalised force in [0, 1]; multiply by F_max for Newtons.
    kpe=5.0, e0=0.6 are the Thelen / OpenSim defaults.
    """
    lceN = l_ce / l_opt
    exp_kpe = jnp.exp(kpe)
    fpe = (jnp.exp(kpe * (lceN - 1.0) / e0) - 1.0) / (exp_kpe - 1.0)
    return jnp.maximum(0.0, fpe)


# ── Activation dynamics ───────────────────────────────────────────────────────

def activation_ode(
    u: jnp.ndarray,
    a: jnp.ndarray,
    p: MuscleParams,
) -> jnp.ndarray:
    """
    da/dt — Thelen 2003 variable-τ activation dynamics.

    τ = τ_act  × (0.5 + 1.5·a)   when u > a  (rising)
    τ = τ_deact / (0.5 + 1.5·a)  when u ≤ a  (falling)

    The (0.5+1.5a) factor makes rise faster at high activation and
    fall slower at high activation — matches OpenSim source exactly.
    """
    a_c = jnp.clip(a, 0.01, 1.0)
    tau = jnp.where(
        u > a_c,
        p.tau_act   * (0.5 + 1.5 * a_c),
        p.tau_deact / (0.5 + 1.5 * a_c),
    )
    return (u - a_c) / tau


# ── MTU kinematics ────────────────────────────────────────────────────────────

def _mtu_kinematics(
    theta: jnp.ndarray,
    theta_dot: jnp.ndarray,
    r: float,
    l_opt: float,
    l_slack: float,
) -> tuple:
    """
    MTU length and velocity for a via-point (wrapping) model.

    Flexor  (positive-θ side):  l_f = l_rest + r·θ,  v_f = r·θ̇
    Extensor (negative-θ side): l_e = l_rest − r·θ,  v_e = −r·θ̇

    Positive v_mtu = lengthening = eccentric; negative = shortening = concentric.
    """
    l_rest = l_opt + l_slack
    l_f = l_rest + r * theta
    l_e = l_rest - r * theta
    v_f =  r * theta_dot
    v_e = -r * theta_dot
    return l_f, l_e, v_f, v_e


# ── Single-muscle force ───────────────────────────────────────────────────────

def muscle_force(
    activation: jnp.ndarray,
    l_mtu: jnp.ndarray,
    v_mtu: jnp.ndarray,
    p: MuscleParams,
) -> jnp.ndarray:
    """
    Total force for one muscle (rigid tendon).

    Args:
        activation : neural drive / activation in [0, 1]
        l_mtu      : muscle-tendon unit length (m)
        v_mtu      : MTU velocity (m/s); positive = lengthening (eccentric)
        p          : MuscleParams

    Returns:
        Force in Newtons (≥ 0)
    """
    l_ce = jnp.maximum(l_mtu - p.l_slack, 0.01 * p.l_opt)
    v_ce = v_mtu  # rigid tendon: CE velocity == MTU velocity

    fl = _fl(l_ce, p.l_opt, p.KshapeActive)
    fv = _fv(v_ce, p.v_max, p.Af, p.Flen)
    fp = _pee(l_ce, p.l_opt, p.kpe, p.e0)

    f_active  = activation * fl * fv * p.F_max
    f_passive = fp * p.F_max
    return jnp.maximum(0.0, f_active + f_passive)


# ── Joint torque (antagonistic pair) ─────────────────────────────────────────

def joint_torque(
    a_flex: jnp.ndarray,
    a_ext: jnp.ndarray,
    theta: jnp.ndarray,
    theta_dot: jnp.ndarray,
    p: MuscleParams,
) -> jnp.ndarray:
    """
    Net torque at a joint from one antagonistic flexor–extensor pair.

    Positive torque → positive-θ direction (flexion).
    τ = r × (F_flex − F_ext)
    """
    l_f, l_e, v_f, v_e = _mtu_kinematics(theta, theta_dot, p.r, p.l_opt, p.l_slack)
    F_flex = muscle_force(a_flex, l_f, v_f, p)
    F_ext  = muscle_force(a_ext,  l_e, v_e, p)
    return p.r * (F_flex - F_ext)
