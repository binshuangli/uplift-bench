"""Compute the applicable metric subset for one held-out test fold.

Which metrics run depends on the dataset:
  - All datasets: Qini, AUUC, uplift@k, policy value@k, calibration ECE.
  - Ground-truth ITE present ('ite' column): PEHE, ε_ATE.
  - Jobs ('experimental' column): policy risk on the experimental subset.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from uplift_bench.metrics.calibration import uplift_calibration_error
from uplift_bench.metrics.causal import epsilon_ate, jobs_policy_risk, pehe
from uplift_bench.metrics.ranking import auuc, policy_value_at_k, qini_coefficient, uplift_at_k

log = logging.getLogger(__name__)


def evaluate_fold(
    uplift_pred: np.ndarray,
    treatment: np.ndarray,
    outcome: np.ndarray,
    test_propensity: np.ndarray | None,
    ite_test: pd.DataFrame | None,
    budget_k: float = 0.3,
) -> dict[str, float]:
    """Return a dict of metric_name -> value for one test fold.

    Metrics that don't apply to this dataset are simply absent from the dict
    (the runner fills them as NaN in the results table).
    """
    t = np.asarray(treatment, dtype=float)
    y = np.asarray(outcome, dtype=float)
    pred = np.asarray(uplift_pred, dtype=float)

    out: dict[str, float] = {}

    def _safe(name, fn, *args, **kwargs):
        try:
            out[name] = float(fn(*args, **kwargs))
        except Exception as exc:
            log.debug("metric %s failed: %s", name, exc)
            out[name] = float("nan")

    # --- ranking / policy (all datasets) ---
    # Use normalize=False: the full-integral normalization breaks for datasets with
    # extreme treatment imbalance (n_t >> n_c), where the perfect Qini curve
    # dips below the random baseline in aggregate. Unnormalized Qini is still
    # valid for within-dataset model ranking; AUUC handles cross-dataset comparison.
    _safe("qini", qini_coefficient, pred, t, y, normalize=False)
    _safe("auuc", auuc, pred, t, y)
    _safe("uplift_at_k", uplift_at_k, pred, t, y, k=budget_k)
    _safe(
        "policy_value_at_k",
        policy_value_at_k,
        pred,
        t,
        y,
        k=budget_k,
        propensity=test_propensity,
    )
    _safe("calibration_ece", uplift_calibration_error, pred, t, y)

    # --- ground-truth CATE (IHDP / synthetic) ---
    if ite_test is not None and "ite" in ite_test.columns:
        ite_true = ite_test["ite"].to_numpy(dtype=float)
        _safe("pehe", pehe, pred, ite_true)
        _safe("epsilon_ate", epsilon_ate, pred, ite_true)

    # --- Jobs policy risk (experimental subset) ---
    if ite_test is not None and "experimental" in ite_test.columns:
        exp_flag = ite_test["experimental"].to_numpy(dtype=float)
        _safe("jobs_policy_risk", jobs_policy_risk, pred, y, t, exp_flag)

    return out


# Models that require a binary outcome (use classification base learners).
BINARY_OUTCOME_MODELS = frozenset(
    {
        "class_transformation",
        "two_model",
        "solo_model",
        "uplift_rf_kl",
        "uplift_rf_ed",
        "uplift_rf_chi",
    }
)


def outcome_is_binary(outcome) -> bool:
    vals = np.unique(np.asarray(outcome))
    return len(vals) <= 2 and set(np.round(vals).astype(int).tolist()).issubset({0, 1})
