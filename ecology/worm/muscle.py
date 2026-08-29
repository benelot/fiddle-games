"""
Hill-type muscle-tendon model.

Based on Geijtenbeek et al. 2013 "Flexible Muscle-Based Locomotion for Bipedal
Creatures" — simplified to a rigid-tendon approximation so CE length can be
computed analytically (no implicit solve needed, making it JIT-friendly).

Architecture per joint:
  • Two antagonistic muscles: flexor (positive theta direction) and extensor
  • Each muscle has:
      – Contractile element (CE): active force, gated by force-length × force-velocity
      – Parallel elastic element (PEE): passive resistance at extremes
  • Rigid tendon assumption: l_CE = l_MTU – l_slack_tendon
  • Net joint torque = moment_arm × (F_flexor – F_extensor)
"""

import jax.numpy as jnp
from typing import NamedTuple


class MuscleParams(NamedTuple):
    """Immutable parameter bundle for one antagonistic muscle pair at a joint."""
    F_max: float = 5.0      # max isometric force per muscle (N)
    l_opt: float = 0.10     # optimal CE fibre length (m)
    l_slack: float = 0.06   # tendon slack length (m); CE = MTU – l_slack
    v_max: float = 0.8      # max shortening speed (m/s, i.e. l_opt/s ≈ 10)
    r: float = 0.025        # moment arm at the joint (m)
    w: float = 0.56         # width of the F-L Gaussian (fraction of l_opt)
    k_pee: float = 0.4      # PEE stiffness (× F_max)
    pee_slack: float = 1.0  # PEE slack as fraction of l_opt (rest at l_opt)


def _fl(l_ce: jnp.ndarray, l_opt: float, w: float) -> jnp.ndarray:
    """Active force-length relationship: Gaussian centred at l_opt."""
    return jnp.exp(-((l_ce / l_opt - 1.0) / w) ** 2)


def _fv(v_ce: jnp.ndarray, v_max: float) -> jnp.ndarray:
    """
    Force-velocity: Hill hyperbola for concentric (shortening, v_ce < 0)
    and a linear extension for eccentric (lengthening, v_ce > 0).

    Normalised so f_v(0)=1, f_v(–v_max)=0, f_v(+∞)→1.5.
    """
    # Concentric branch (v_ce ≤ 0)
    b = 0.25 * v_max  # shape constant
    f_conc = (v_max + v_ce) / (v_max - v_ce / b)
    f_conc = jnp.clip(f_conc, 0.0, 1.0)

    # Eccentric branch (v_ce > 0): linear ramp from 1 → 1.5
    f_ecc = 1.0 + 0.5 * v_ce / v_max
    f_ecc = jnp.clip(f_ecc, 1.0, 1.5)

    return jnp.where(v_ce <= 0.0, f_conc, f_ecc)


def _pee(l_ce: jnp.ndarray, l_opt: float, k: float, slack: float) -> jnp.ndarray:
    """Parallel elastic element: quadratic rise above pee_slack × l_opt."""
    stretch = jnp.maximum(0.0, l_ce / l_opt - slack)
    return k * stretch ** 2


def _mtu_kinematics(
    theta: jnp.ndarray,
    theta_dot: jnp.ndarray,
    r: float,
    l_opt: float,
    l_slack: float,
) -> tuple:
    """
    Muscle-tendon unit length and velocity for a simple via-point model.

    Flexor:  wraps over the joint on the positive-theta side.
      l_mtu_flex = l_rest + r·θ       (lengthens when joint extends)
      v_mtu_flex = r·θ̇
    Extensor: opposite side.
      l_mtu_ext  = l_rest – r·θ
      v_mtu_ext  = –r·θ̇
    """
    l_rest = l_opt + l_slack
    l_f = l_rest + r * theta
    l_e = l_rest - r * theta
    v_f = r * theta_dot
    v_e = -r * theta_dot
    return l_f, l_e, v_f, v_e


def muscle_force(
    activation: jnp.ndarray,
    l_mtu: jnp.ndarray,
    v_mtu: jnp.ndarray,
    p: MuscleParams,
) -> jnp.ndarray:
    """
    Total muscle force for one muscle (rigid tendon).

    Args:
        activation: neural drive in [0, 1]
        l_mtu:      muscle-tendon unit length (m)
        v_mtu:      MTU velocity (m/s); positive = lengthening
        p:          MuscleParams

    Returns:
        Force in Newtons (≥ 0)
    """
    l_ce = jnp.maximum(l_mtu - p.l_slack, 0.01 * p.l_opt)  # rigid tendon
    v_ce = v_mtu  # rigid tendon: CE velocity == MTU velocity

    f_active = activation * _fl(l_ce, p.l_opt, p.w) * _fv(v_ce, p.v_max) * p.F_max

    # Cap CE length for PEE to prevent runaway passive forces at extreme joint angles.
    # Real muscles have a finite functional range; beyond ~1.5 × l_opt the fibre
    # arrangement cannot sustain large passive forces.
    l_ce_pee = jnp.clip(l_ce, 0.01 * p.l_opt, 1.5 * p.l_opt)
    f_passive = _pee(l_ce_pee, p.l_opt, p.k_pee, p.pee_slack) * p.F_max
    return jnp.maximum(0.0, f_active + f_passive)


def joint_torque(
    a_flex: jnp.ndarray,
    a_ext: jnp.ndarray,
    theta: jnp.ndarray,
    theta_dot: jnp.ndarray,
    p: MuscleParams,
) -> jnp.ndarray:
    """
    Net torque at a joint from an antagonistic flexor–extensor pair.

    Positive torque → positive θ direction (flexion).
    tau = r × (F_flex – F_ext)
    """
    l_f, l_e, v_f, v_e = _mtu_kinematics(theta, theta_dot, p.r, p.l_opt, p.l_slack)
    F_flex = muscle_force(a_flex, l_f, v_f, p)
    F_ext = muscle_force(a_ext, l_e, v_e, p)
    return p.r * (F_flex - F_ext)
