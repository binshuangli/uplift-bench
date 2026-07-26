"""Unit tests for WP2 metrics module.

Every test uses synthetic data with known closed-form answers so no
network access or large datasets are required.
"""

from __future__ import annotations

import numpy as np
import pytest

from uplift_bench.metrics import (
    auuc,
    bootstrap_ci,
    epsilon_ate,
    jobs_policy_risk,
    pehe,
    policy_value_at_k,
    qini_coefficient,
    uplift_at_k,
    uplift_calibration_error,
    uplift_reliability_curve,
    wilcoxon_paired,
)

RNG = np.random.default_rng(0)


# ---------------------------------------------------------------------------
# Helpers to build synthetic scenarios
# ---------------------------------------------------------------------------


def _perfect_scenario(n: int = 500, seed: int = 0):
    """All treated positives have score=1, all others score=0."""
    rng = np.random.default_rng(seed)
    treatment = rng.integers(0, 2, n).astype(float)
    outcome = np.where(treatment == 1, rng.binomial(1, 0.7, n), rng.binomial(1, 0.3, n)).astype(
        float
    )
    # Perfect score = outcome * treatment (ranks treated positives first)
    uplift_score = (outcome * treatment).astype(float) + rng.uniform(0, 1e-6, n)
    return uplift_score, treatment, outcome


def _random_scenario(n: int = 500, seed: int = 1):
    """Random scores — should yield metrics near zero / baseline."""
    rng = np.random.default_rng(seed)
    treatment = rng.integers(0, 2, n).astype(float)
    outcome = rng.binomial(1, 0.5, n).astype(float)
    uplift_score = rng.standard_normal(n)
    return uplift_score, treatment, outcome


def _constant_scenario(n: int = 500, seed: int = 2):
    """Constant scores — exactly random baseline."""
    rng = np.random.default_rng(seed)
    treatment = rng.integers(0, 2, n).astype(float)
    outcome = rng.binomial(1, 0.5, n).astype(float)
    uplift_score = np.ones(n)
    return uplift_score, treatment, outcome


# ---------------------------------------------------------------------------
# Qini coefficient
# ---------------------------------------------------------------------------


class TestQiniCoefficient:
    def test_perfect_is_close_to_one(self):
        score, t, y = _perfect_scenario()
        q = qini_coefficient(score, t, y, normalize=True)
        assert q > 0.5, f"Perfect scorer Qini should be > 0.5, got {q:.4f}"

    def test_random_near_zero(self):
        score, t, y = _random_scenario(n=2000)
        q = qini_coefficient(score, t, y, normalize=True)
        assert abs(q) < 0.15, f"Random scorer Qini should be near 0, got {q:.4f}"

    def test_constant_near_zero(self):
        # Constant scores give an arbitrary sort order (no signal), Qini ≈ 0.
        # Use large n so law-of-large-numbers kicks in; can't guarantee exactly 0.
        score, t, y = _constant_scenario(n=5000)
        q = qini_coefficient(score, t, y, normalize=True)
        assert abs(q) < 0.15, f"Constant scorer Qini should be near 0, got {q:.4f}"

    def test_unnormalized_returns_float(self):
        score, t, y = _perfect_scenario()
        q = qini_coefficient(score, t, y, normalize=False)
        assert isinstance(q, float)

    def test_requires_both_groups(self):
        score = np.ones(10)
        t_all_treated = np.ones(10)
        y = np.ones(10)
        with pytest.raises(ValueError):
            qini_coefficient(score, t_all_treated, y)

    def test_perfect_beats_random(self):
        q_perfect = qini_coefficient(*_perfect_scenario())
        q_random = qini_coefficient(*_random_scenario())
        assert q_perfect > q_random


# ---------------------------------------------------------------------------
# AUUC
# ---------------------------------------------------------------------------


class TestAUUC:
    def test_positive_for_good_model(self):
        score, t, y = _perfect_scenario()
        val = auuc(score, t, y)
        assert val > 0, f"Expected positive AUUC for good model, got {val:.4f}"

    def test_near_zero_for_random(self):
        score, t, y = _random_scenario(n=2000)
        val = auuc(score, t, y)
        assert abs(val) < 0.05, f"Random AUUC should be near 0, got {val:.4f}"

    def test_returns_float(self):
        score, t, y = _perfect_scenario()
        assert isinstance(auuc(score, t, y), float)


# ---------------------------------------------------------------------------
# Uplift @ k
# ---------------------------------------------------------------------------


class TestUpliftAtK:
    def test_positive_top_k_good_model(self):
        # Use a scenario with signal but mixed treatment in top-k (not all-treated).
        # Scores = true uplift per unit (continuous), both groups represented everywhere.
        rng = np.random.default_rng(3)
        n = 2000
        t = rng.integers(0, 2, n).astype(float)
        ite = rng.uniform(0, 1, n)  # everyone has positive uplift
        score = ite + rng.normal(0, 0.05, n)  # score ≈ true ITE + small noise
        y = np.where(t == 1, rng.binomial(1, 0.3 + ite * 0.4, n), rng.binomial(1, 0.3, n)).astype(
            float
        )
        val = uplift_at_k(score, t, y, k=0.3)
        assert val > 0, f"Good model top-30% uplift should be positive, got {val}"

    def test_nan_when_no_treated_in_top_k(self):
        # Construct scenario where no treated units land in top-k
        n = 100
        score = np.arange(n, dtype=float)  # ascending
        treatment = np.zeros(n)
        treatment[:10] = 1  # treated at bottom (lowest scores)
        outcome = np.zeros(n)
        val = uplift_at_k(score, treatment, outcome, k=0.1)
        assert np.isnan(val), "Should be NaN when no treated in top-k"

    def test_k_equals_one_is_ate(self):
        """uplift_at_k(k=1) should equal the naive ATE."""
        rng = np.random.default_rng(42)
        n = 500
        t = rng.integers(0, 2, n).astype(float)
        y = rng.binomial(1, 0.5, n).astype(float)
        score = rng.standard_normal(n)
        val = uplift_at_k(score, t, y, k=1.0)
        ate = float(y[t == 1].mean() - y[t == 0].mean())
        assert abs(val - ate) < 1e-10


# ---------------------------------------------------------------------------
# Policy value @ k
# ---------------------------------------------------------------------------


class TestPolicyValueAtK:
    def test_returns_float(self):
        score, t, y = _perfect_scenario()
        val = policy_value_at_k(score, t, y, k=0.5)
        assert isinstance(val, float)

    def test_with_propensity(self):
        score, t, y = _perfect_scenario()
        val_plain = policy_value_at_k(score, t, y, k=0.5)
        val_ipw = policy_value_at_k(score, t, y, k=0.5, propensity=0.5)
        # Both should be finite
        assert np.isfinite(val_plain)
        assert np.isfinite(val_ipw)

    def test_good_policy_beats_bad(self):
        """Top-k targeting by good scores > targeting by anti-scores."""
        score, t, y = _perfect_scenario(n=1000)
        val_good = policy_value_at_k(score, t, y, k=0.3)
        val_bad = policy_value_at_k(-score, t, y, k=0.3)
        assert val_good > val_bad


# ---------------------------------------------------------------------------
# PEHE
# ---------------------------------------------------------------------------


class TestPEHE:
    def test_perfect_is_zero(self):
        ite_true = np.array([1.0, 2.0, -1.0, 0.5])
        assert pehe(ite_true, ite_true) == pytest.approx(0.0, abs=1e-10)

    def test_known_value(self):
        pred = np.array([1.0, 1.0])
        true = np.array([0.0, 2.0])
        # errors = [1, -1], PEHE = sqrt(mean([1,1])) = 1.0
        assert pehe(pred, true) == pytest.approx(1.0, abs=1e-10)

    def test_always_nonneg(self):
        rng = np.random.default_rng(0)
        pred = rng.standard_normal(100)
        true = rng.standard_normal(100)
        assert pehe(pred, true) >= 0


# ---------------------------------------------------------------------------
# ε_ATE
# ---------------------------------------------------------------------------


class TestEpsilonATE:
    def test_perfect_is_zero(self):
        ite = np.array([1.0, 2.0, 3.0])
        assert epsilon_ate(ite, ite) == pytest.approx(0.0, abs=1e-10)

    def test_known_value(self):
        pred = np.array([2.0, 2.0])  # mean = 2
        true = np.array([1.0, 3.0])  # mean = 2
        assert epsilon_ate(pred, true) == pytest.approx(0.0, abs=1e-10)

    def test_nonzero(self):
        pred = np.array([1.0, 1.0])  # mean = 1
        true = np.array([0.0, 0.0])  # mean = 0
        assert epsilon_ate(pred, true) == pytest.approx(1.0, abs=1e-10)


# ---------------------------------------------------------------------------
# Jobs policy risk
# ---------------------------------------------------------------------------


class TestJobsPolicyRisk:
    def _synthetic_jobs(self, n: int = 200, seed: int = 7):
        rng = np.random.default_rng(seed)
        treatment = rng.integers(0, 2, n).astype(float)
        outcome = rng.binomial(1, 0.4, n).astype(float)
        experimental = np.zeros(n)
        experimental[: n // 2] = 1
        uplift_score = rng.standard_normal(n)
        return uplift_score, outcome, treatment, experimental

    def test_returns_float(self):
        score, y, t, e = self._synthetic_jobs()
        val = jobs_policy_risk(score, y, t, e)
        assert isinstance(val, float)

    def test_no_experimental_raises(self):
        score, y, t, e = self._synthetic_jobs()
        with pytest.raises(ValueError, match="No experimental"):
            jobs_policy_risk(score, y, t, np.zeros_like(e))

    def test_treat_all_is_finite(self):
        score, y, t, e = self._synthetic_jobs()
        val = jobs_policy_risk(np.ones_like(score), y, t, e, threshold=0.5)
        assert np.isfinite(val)


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


class TestCalibration:
    def _perfect_calibration_data(self, n: int = 2000, seed: int = 0):
        """Construct data where predicted uplift == true uplift per bin."""
        rng = np.random.default_rng(seed)
        treatment = rng.integers(0, 2, n).astype(float)
        # Uplift score drawn from [0,1]
        score = rng.uniform(0, 1, n)
        # Outcome: P(Y=1|T=1) = base + score, P(Y=1|T=0) = base
        base = 0.3
        p1 = np.clip(base + score, 0, 1)
        p0 = np.full(n, base)
        outcome = np.where(treatment == 1, rng.binomial(1, p1), rng.binomial(1, p0)).astype(float)
        return score, treatment, outcome

    def test_reliability_curve_shape(self):
        score, t, y = self._perfect_calibration_data()
        curve = uplift_reliability_curve(score, t, y, n_bins=5)
        for key in ("bin_centers", "observed_uplift", "bin_counts"):
            assert key in curve
        assert len(curve["bin_centers"]) == len(curve["observed_uplift"])

    def test_perfect_calibration_low_ece(self):
        score, t, y = self._perfect_calibration_data(n=5000)
        ece = uplift_calibration_error(score, t, y, n_bins=10)
        assert ece < 0.1, f"Well-calibrated model ECE should be < 0.1, got {ece:.4f}"

    def test_badly_calibrated_high_ece(self):
        """Predicted uplift = 1 everywhere but true uplift ≈ 0 → high ECE."""
        rng = np.random.default_rng(1)
        n = 1000
        t = rng.integers(0, 2, n).astype(float)
        y = rng.binomial(1, 0.5, n).astype(float)  # no treatment effect
        score = np.ones(n)  # always predict uplift=1
        ece = uplift_calibration_error(score, t, y)
        assert ece > 0.5, f"Badly calibrated ECE should be > 0.5, got {ece:.4f}"

    def test_ece_nonneg(self):
        score, t, y = self._perfect_calibration_data()
        assert uplift_calibration_error(score, t, y) >= 0

    def test_uniform_strategy(self):
        score, t, y = self._perfect_calibration_data()
        curve = uplift_reliability_curve(score, t, y, n_bins=5, strategy="uniform")
        assert len(curve["bin_centers"]) > 0

    def test_invalid_strategy_raises(self):
        score, t, y = self._perfect_calibration_data()
        with pytest.raises(ValueError, match="strategy"):
            uplift_reliability_curve(score, t, y, strategy="bad")


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------


class TestBootstrapCI:
    def test_point_estimate_matches_direct(self):
        score, t, y = _perfect_scenario()
        point, lo, hi = bootstrap_ci(qini_coefficient, score, t, y, n_bootstrap=200, seed=0)
        direct = qini_coefficient(score, t, y)
        assert abs(point - direct) < 1e-10

    def test_ci_contains_point(self):
        score, t, y = _perfect_scenario()
        point, lo, hi = bootstrap_ci(qini_coefficient, score, t, y, n_bootstrap=200, seed=0)
        assert lo <= point <= hi, f"CI [{lo:.4f}, {hi:.4f}] should contain point {point:.4f}"

    def test_ci_ordering(self):
        score, t, y = _perfect_scenario()
        _, lo, hi = bootstrap_ci(qini_coefficient, score, t, y, n_bootstrap=200, seed=0)
        assert lo <= hi

    def test_reproducible_under_fixed_seed(self):
        score, t, y = _perfect_scenario()
        r1 = bootstrap_ci(qini_coefficient, score, t, y, n_bootstrap=100, seed=99)
        r2 = bootstrap_ci(qini_coefficient, score, t, y, n_bootstrap=100, seed=99)
        assert r1 == r2

    def test_different_seeds_differ(self):
        score, t, y = _perfect_scenario(n=1000)
        r1 = bootstrap_ci(qini_coefficient, score, t, y, n_bootstrap=50, seed=0)
        r2 = bootstrap_ci(qini_coefficient, score, t, y, n_bootstrap=50, seed=1)
        assert r1 != r2


# ---------------------------------------------------------------------------
# Wilcoxon paired test
# ---------------------------------------------------------------------------


class TestWilcoxonPaired:
    def test_significantly_different_series(self):
        a = np.array([0.9, 0.85, 0.88, 0.91, 0.87, 0.90, 0.86, 0.89])
        b = np.array([0.5, 0.48, 0.51, 0.49, 0.50, 0.52, 0.47, 0.50])
        result = wilcoxon_paired(a, b)
        assert result["significant_05"] is True
        assert result["p_value"] < 0.05

    def test_identical_series_returns_nan(self):
        a = np.array([0.5, 0.5, 0.5, 0.5])
        result = wilcoxon_paired(a, a)
        assert np.isnan(result["p_value"])
        assert result["significant_05"] is False

    def test_returns_expected_keys(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([0.5, 1.5, 2.5])
        result = wilcoxon_paired(a, b)
        for key in ("statistic", "p_value", "significant_05", "significant_01"):
            assert key in result

    def test_alternative_greater(self):
        a = np.array([0.8, 0.9, 0.85, 0.88, 0.87, 0.86])
        b = np.array([0.5, 0.4, 0.45, 0.48, 0.47, 0.46])
        result = wilcoxon_paired(a, b, alternative="greater")
        assert result["significant_05"] is True
