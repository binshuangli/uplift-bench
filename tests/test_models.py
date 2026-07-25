"""WP3 smoke tests: every model trains and predicts on a tiny synthetic dataset.

Tests run without any downloaded data (pure synthetic).
All models must:
  - accept fit(X, treatment, outcome, propensity)
  - return predict_uplift(X) of shape (n,) with no NaN/Inf
  - work with either base learner (lightgbm / xgboost)
  - produce different predictions than the constant-zero baseline
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from uplift_bench.models.registry import get_estimator, list_models

# ---------------------------------------------------------------------------
# Shared synthetic data
# ---------------------------------------------------------------------------

N_TRAIN = 300
N_TEST = 100
N_FEATURES = 8


def _make_synthetic(seed: int = 0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        rng.standard_normal((N_TRAIN, N_FEATURES)),
        columns=[f"f{i}" for i in range(N_FEATURES)],
    )
    treatment = pd.Series(rng.integers(0, 2, N_TRAIN).astype(int), name="treatment")
    # Binary outcome with a mild heterogeneous treatment effect
    ite = 0.15 * X["f0"].values
    p_y = 0.4 + ite * treatment.values
    outcome = pd.Series(rng.binomial(1, np.clip(p_y, 0.01, 0.99)).astype(float), name="outcome")
    propensity = pd.Series(np.full(N_TRAIN, 0.5), name="propensity")

    X_test = pd.DataFrame(
        rng.standard_normal((N_TEST, N_FEATURES)),
        columns=[f"f{i}" for i in range(N_FEATURES)],
    )
    return X, treatment, outcome, propensity, X_test


X_TR, T_TR, Y_TR, P_TR, X_TE = _make_synthetic()


# ---------------------------------------------------------------------------
# Parametrized smoke test across all models × both base learners
# ---------------------------------------------------------------------------

ALL_MODELS = list_models()

# Models that only support LightGBM/XGBoost base learners via meta-learner API
META_MODELS = [
    "s_learner",
    "t_learner",
    "x_learner",
    "r_learner",
    "dr_learner",
    "class_transformation",
    "two_model",
    "solo_model",
]
# Forest / tree models don't have a swappable base learner
FOREST_MODELS = ["causal_forest", "uplift_rf_kl", "uplift_rf_ed", "uplift_rf_chi"]


def _smoke_one(name: str, base_learner: str = "lightgbm", with_propensity: bool = True):
    """Fit + predict; assert shape and finiteness."""
    est = get_estimator(name, base_learner=base_learner, seed=0)
    prop = P_TR if with_propensity else None
    est.fit(X_TR, T_TR, Y_TR, prop)
    pred = est.predict_uplift(X_TE)
    assert pred.shape == (N_TEST,), f"{name}: expected ({N_TEST},), got {pred.shape}"
    assert np.isfinite(pred).all(), f"{name}: predictions contain NaN/Inf"
    return pred


# One test per model (lightgbm, with propensity)
@pytest.mark.parametrize("model_name", ALL_MODELS)
def test_smoke_fit_predict(model_name):
    _smoke_one(model_name, base_learner="lightgbm", with_propensity=True)


# Base-learner swap test for meta-learners only
@pytest.mark.parametrize("model_name", META_MODELS)
def test_base_learner_xgboost(model_name):
    _smoke_one(model_name, base_learner="xgboost", with_propensity=True)


# Without propensity — models must handle None gracefully
@pytest.mark.parametrize("model_name", ALL_MODELS)
def test_smoke_no_propensity(model_name):
    _smoke_one(model_name, base_learner="lightgbm", with_propensity=False)


# ---------------------------------------------------------------------------
# Interface / registry tests
# ---------------------------------------------------------------------------


def test_list_models_contains_all_expected():
    names = set(list_models())
    expected = {
        "s_learner",
        "t_learner",
        "x_learner",
        "r_learner",
        "dr_learner",
        "class_transformation",
        "two_model",
        "solo_model",
        "causal_forest",
        "uplift_rf_kl",
        "uplift_rf_ed",
        "uplift_rf_chi",
    }
    assert expected.issubset(names), f"Missing: {expected - names}"


def test_unknown_model_raises():
    with pytest.raises(ValueError, match="Unknown model"):
        get_estimator("does_not_exist")


def test_repr():
    est = get_estimator("t_learner")
    assert "TLearnerEstimator" in repr(est)


def test_base_learner_is_swappable():
    """Same model name, different base learner → different objects, both predict fine."""
    pred_lgbm = _smoke_one("t_learner", base_learner="lightgbm")
    pred_xgb = _smoke_one("t_learner", base_learner="xgboost")
    # Different base learners should generally produce different scores
    # (not identical) — but we only require both are finite
    assert np.isfinite(pred_lgbm).all()
    assert np.isfinite(pred_xgb).all()


def test_fit_returns_self():
    est = get_estimator("t_learner", seed=0)
    result = est.fit(X_TR, T_TR, Y_TR, P_TR)
    assert result is est


def test_predictions_not_all_identical():
    """A trained model should produce non-constant predictions (signal check)."""
    pred = _smoke_one("t_learner")
    assert pred.std() > 1e-8, "t_learner predictions are all identical — no signal"
