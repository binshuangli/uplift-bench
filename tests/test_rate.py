"""Tests for the RATE metric (metrics/rate.py, R4-W3)."""

from __future__ import annotations

import numpy as np
import pytest

from uplift_bench.metrics.rate import ipw_scores, rate


def _sim(n=4000, seed=0):
    """RCT with heterogeneous effect tau(x) = x (x ~ N(0,1))."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    tau = x
    t = rng.binomial(1, 0.5, size=n)
    y = 0.5 * x + tau * t + rng.normal(scale=0.5, size=n)
    e = np.full(n, 0.5)
    return x, tau, t, y, e


class TestIpwScores:
    def test_unbiased_for_ate(self):
        _, tau, t, y, e = _sim()
        gamma = ipw_scores(t, y, e)
        assert abs(gamma.mean() - tau.mean()) < 0.1

    def test_clipping(self):
        t = np.array([1, 0])
        y = np.array([1.0, 1.0])
        e = np.array([0.0, 1.0])  # degenerate; must be clipped, not inf
        g = ipw_scores(t, y, e, clip=0.01)
        assert np.all(np.isfinite(g))


class TestRate:
    @pytest.mark.parametrize("weighting", ["autoc", "qini"])
    def test_oracle_positive_random_zero_antioracle_negative(self, weighting):
        x, tau, t, y, e = _sim()
        rng = np.random.default_rng(1)
        r_oracle = rate(tau, t, y, e, weighting=weighting)
        r_anti = rate(-tau, t, y, e, weighting=weighting)
        r_rand = np.mean(
            [rate(rng.normal(size=len(x)), t, y, e, weighting=weighting) for _ in range(20)]
        )
        assert r_oracle > 0.1
        assert r_anti < -0.1
        assert abs(r_rand) < 0.05
        assert r_oracle > r_rand > r_anti

    def test_tie_invariance(self):
        """Constant scores (all tied) must give exactly 0 and not depend on order."""
        _, _, t, y, e = _sim(n=500)
        r = rate(np.zeros(500), t, y, e)
        assert abs(r) < 1e-12

    def test_scale_invariance_of_ranking(self):
        """RATE depends on scores only through their ranking (affine-safe)."""
        x, tau, t, y, e = _sim()
        assert rate(tau, t, y, e) == pytest.approx(rate(10 * tau + 3, t, y, e))

    def test_bad_weighting_raises(self):
        _, tau, t, y, e = _sim(n=100)
        with pytest.raises(ValueError):
            rate(tau, t, y, e, weighting="uniform")

    def test_length_mismatch_raises(self):
        _, tau, t, y, e = _sim(n=100)
        with pytest.raises(ValueError):
            rate(tau[:50], t, y, e)
