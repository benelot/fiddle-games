"""
Unit tests for the Thelen 2003 Hill-type muscle model.

Run:  pytest ecology/worm/test_muscle.py -v
"""

import math
import pytest
import jax.numpy as jnp

from muscle import (
    MuscleParams,
    _fl,
    _fv,
    _pee,
    activation_ode,
    muscle_force,
    joint_torque,
)

P = MuscleParams()  # default Thelen 2003 parameters


# ── Active force-length ────────────────────────────────────────────────────────

class TestFL:
    def test_peak_at_l_opt(self):
        """fl = 1 exactly at l_opt."""
        assert float(_fl(P.l_opt, P.l_opt, P.KshapeActive)) == pytest.approx(1.0)

    def test_symmetric_around_l_opt(self):
        """Gaussian is symmetric: fl(l_opt + δ) == fl(l_opt − δ)."""
        delta = 0.02
        f_above = float(_fl(P.l_opt + delta, P.l_opt, P.KshapeActive))
        f_below = float(_fl(P.l_opt - delta, P.l_opt, P.KshapeActive))
        assert f_above == pytest.approx(f_below, rel=1e-6)

    def test_falls_off_steeply_at_double_l_opt(self):
        """At 2 × l_opt (lceN = 2) the force-length multiplier should be near zero."""
        f = float(_fl(2.0 * P.l_opt, P.l_opt, P.KshapeActive))
        assert f < 0.01

    def test_monotone_away_from_optimal(self):
        """Larger deviations from l_opt produce smaller fl values."""
        f1 = float(_fl(P.l_opt + 0.01, P.l_opt, P.KshapeActive))
        f2 = float(_fl(P.l_opt + 0.03, P.l_opt, P.KshapeActive))
        assert f1 > f2

    def test_positive_everywhere(self):
        """fl is non-negative for any length."""
        for lceN in [0.5, 0.8, 1.0, 1.5, 2.0]:
            assert float(_fl(lceN * P.l_opt, P.l_opt, P.KshapeActive)) >= 0.0

    def test_narrower_kshape_is_more_peaked(self):
        """A smaller KshapeActive narrows the Gaussian: less force off-peak."""
        delta = 0.05
        f_narrow = float(_fl(P.l_opt + delta, P.l_opt, 0.20))
        f_wide   = float(_fl(P.l_opt + delta, P.l_opt, 0.60))
        assert f_narrow < f_wide


# ── Force-velocity ─────────────────────────────────────────────────────────────

class TestFV:
    def test_unity_at_zero_velocity(self):
        """fv = 1 at v = 0 (isometric)."""
        assert float(_fv(0.0, P.v_max, P.Af, P.Flen)) == pytest.approx(1.0, rel=1e-5)

    def test_zero_at_max_shortening(self):
        """fv → 0 as v_ce → −v_max (concentric limit)."""
        assert float(_fv(-P.v_max, P.v_max, P.Af, P.Flen)) == pytest.approx(0.0, abs=1e-5)

    def test_eccentric_ceiling(self):
        """fv saturates at Flen for very fast lengthening."""
        f = float(_fv(100.0 * P.v_max, P.v_max, P.Af, P.Flen))
        assert f == pytest.approx(P.Flen, rel=1e-3)

    def test_continuity_at_zero(self):
        """Concentric and eccentric branches both return ≈1 at v = 0."""
        eps = 1e-6
        f_c = float(_fv(-eps, P.v_max, P.Af, P.Flen))
        f_e = float(_fv(+eps, P.v_max, P.Af, P.Flen))
        assert abs(f_c - f_e) < 1e-3

    def test_monotone_increasing(self):
        """fv is monotonically increasing: concentric < isometric < eccentric."""
        f_conc = float(_fv(-0.5 * P.v_max, P.v_max, P.Af, P.Flen))
        f_iso  = float(_fv(0.0,             P.v_max, P.Af, P.Flen))
        f_ecc  = float(_fv(+0.5 * P.v_max, P.v_max, P.Af, P.Flen))
        assert f_conc < f_iso < f_ecc

    def test_concentric_clipped_to_zero(self):
        """fv never goes negative during fast concentric contractions."""
        f = float(_fv(-10.0 * P.v_max, P.v_max, P.Af, P.Flen))
        assert f >= 0.0

    def test_eccentric_clipped_to_flen(self):
        """fv never exceeds Flen."""
        f = float(_fv(10.0 * P.v_max, P.v_max, P.Af, P.Flen))
        assert f <= P.Flen + 1e-6


# ── Passive force-length (PEE) ─────────────────────────────────────────────────

class TestPEE:
    def test_zero_at_l_opt(self):
        """PEE = 0 at l_opt (onset is exactly at l_opt)."""
        assert float(_pee(P.l_opt, P.l_opt, P.kpe, P.e0)) == pytest.approx(0.0, abs=1e-6)

    def test_zero_below_l_opt(self):
        """PEE = 0 for any CE length ≤ l_opt (no compressive passive force)."""
        assert float(_pee(0.5 * P.l_opt, P.l_opt, P.kpe, P.e0)) == pytest.approx(0.0, abs=1e-6)

    def test_one_at_e0_strain(self):
        """By definition fpe = 1 when lceN = 1 + e0 (normalised strain = e0)."""
        l_e0 = P.l_opt * (1.0 + P.e0)
        assert float(_pee(l_e0, P.l_opt, P.kpe, P.e0)) == pytest.approx(1.0, rel=1e-4)

    def test_monotone_beyond_l_opt(self):
        """PEE is strictly increasing beyond l_opt."""
        f1 = float(_pee(P.l_opt * 1.1, P.l_opt, P.kpe, P.e0))
        f2 = float(_pee(P.l_opt * 1.3, P.l_opt, P.kpe, P.e0))
        assert f1 < f2

    def test_positive_beyond_l_opt(self):
        """PEE is non-negative everywhere."""
        for lceN in [0.5, 1.0, 1.2, 1.5, 2.0]:
            assert float(_pee(lceN * P.l_opt, P.l_opt, P.kpe, P.e0)) >= 0.0


# ── Activation dynamics ────────────────────────────────────────────────────────

class TestActivationODE:
    def test_rising_direction(self):
        """With u > a, da/dt > 0."""
        assert float(activation_ode(1.0, 0.1, P)) > 0.0

    def test_falling_direction(self):
        """With u < a, da/dt < 0."""
        assert float(activation_ode(0.0, 0.9, P)) < 0.0

    def test_near_zero_at_steady_state(self):
        """da/dt ≈ 0 when u ≈ a."""
        assert abs(float(activation_ode(0.5, 0.5, P))) < 0.01

    def test_rising_tau_formula(self):
        """Rising τ = τ_act × (0.5 + 1.5·a)."""
        a = 0.4
        tau_expected = P.tau_act * (0.5 + 1.5 * a)
        rate_expected = (1.0 - a) / tau_expected
        assert float(activation_ode(1.0, a, P)) == pytest.approx(rate_expected, rel=1e-5)

    def test_falling_tau_formula(self):
        """Falling τ = τ_deact / (0.5 + 1.5·a)."""
        a = 0.4
        tau_expected = P.tau_deact / (0.5 + 1.5 * a)
        rate_expected = (0.0 - a) / tau_expected
        assert float(activation_ode(0.0, a, P)) == pytest.approx(rate_expected, rel=1e-5)

    def test_faster_rise_at_high_activation(self):
        """At high a the effective τ_act is larger (slower rise than at low a)."""
        rate_low_a  = float(activation_ode(1.0, 0.1, P))
        rate_high_a = float(activation_ode(1.0, 0.8, P))
        # At high a, remaining gap (1-a) is small and τ is larger → smaller da/dt
        assert rate_high_a < rate_low_a


# ── Single-muscle force ────────────────────────────────────────────────────────

class TestMuscleForce:
    def _mtu(self, lceN: float = 1.0) -> float:
        return lceN * P.l_opt + P.l_slack

    def test_zero_activation_zero_passive_at_optimal(self):
        """At l_opt, passive PEE = 0, so zero activation → zero force."""
        F = float(muscle_force(0.0, self._mtu(1.0), 0.0, P))
        assert F == pytest.approx(0.0, abs=1e-6)

    def test_max_force_at_optimal_isometric(self):
        """At l_opt, v=0, a=1: total force ≈ F_max (passive negligible)."""
        F = float(muscle_force(1.0, self._mtu(1.0), 0.0, P))
        assert F == pytest.approx(P.F_max, rel=0.02)

    def test_force_increases_with_activation(self):
        """More activation → more force (at fixed kinematics)."""
        l_mtu = self._mtu(1.0)
        F_lo = float(muscle_force(0.2, l_mtu, 0.0, P))
        F_hi = float(muscle_force(0.8, l_mtu, 0.0, P))
        assert F_lo < F_hi

    def test_force_non_negative(self):
        """Force is always ≥ 0 (muscle can only pull, not push)."""
        for a in [0.0, 0.5, 1.0]:
            for lceN in [0.5, 1.0, 1.5]:
                F = float(muscle_force(a, self._mtu(lceN), 0.0, P))
                assert F >= 0.0

    def test_passive_force_at_stretched(self):
        """At lceN > 1, zero activation still produces passive (PEE) force."""
        F = float(muscle_force(0.0, self._mtu(1.3), 0.0, P))
        assert F > 0.0

    def test_eccentric_exceeds_isometric(self):
        """Eccentric contraction produces more force than isometric at same length."""
        l_mtu = self._mtu(1.0)
        F_iso = float(muscle_force(1.0, l_mtu, 0.0,            P))
        F_ecc = float(muscle_force(1.0, l_mtu, 0.3 * P.v_max, P))
        assert F_ecc > F_iso

    def test_concentric_less_than_isometric(self):
        """Concentric contraction produces less force than isometric."""
        l_mtu = self._mtu(1.0)
        F_iso  = float(muscle_force(1.0, l_mtu, 0.0,            P))
        F_conc = float(muscle_force(1.0, l_mtu, -0.3 * P.v_max, P))
        assert F_conc < F_iso


# ── Joint torque (antagonistic pair) ──────────────────────────────────────────

class TestJointTorque:
    def test_zero_with_equal_activations_at_zero_angle(self):
        """Equal muscles at θ=0 → net torque ≈ 0."""
        tau = float(joint_torque(0.5, 0.5, 0.0, 0.0, P))
        assert abs(tau) < 1e-5

    def test_positive_torque_from_flexor(self):
        """Flexor > extensor → positive (flexion) torque."""
        tau = float(joint_torque(1.0, 0.0, 0.0, 0.0, P))
        assert tau > 0.0

    def test_negative_torque_from_extensor(self):
        """Extensor > flexor → negative (extension) torque."""
        tau = float(joint_torque(0.0, 1.0, 0.0, 0.0, P))
        assert tau < 0.0

    def test_torque_antisymmetric_in_activations(self):
        """Swapping activations negates the torque (at θ=0)."""
        tau_fe = float(joint_torque(0.8, 0.2, 0.0, 0.0, P))
        tau_ef = float(joint_torque(0.2, 0.8, 0.0, 0.0, P))
        assert tau_fe == pytest.approx(-tau_ef, rel=1e-5)

    def test_torque_scales_with_moment_arm(self):
        """Doubling r should roughly double the torque."""
        p2 = MuscleParams(r=P.r * 2)
        tau1 = float(joint_torque(1.0, 0.0, 0.0, 0.0, P))
        tau2 = float(joint_torque(1.0, 0.0, 0.0, 0.0, p2))
        assert tau2 / tau1 == pytest.approx(2.0, rel=0.05)

    def test_torque_scales_with_f_max(self):
        """Doubling F_max should double the torque."""
        p2 = MuscleParams(F_max=P.F_max * 2)
        tau1 = float(joint_torque(1.0, 0.0, 0.0, 0.0, P))
        tau2 = float(joint_torque(1.0, 0.0, 0.0, 0.0, p2))
        assert tau2 / tau1 == pytest.approx(2.0, rel=0.02)
