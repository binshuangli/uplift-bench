"""Nested hyperparameter tuning via random search against validation Qini.

Design decisions (confirmed with project owner):
  - Objective: Qini coefficient on the inner validation fold (uplift-specific,
    uniform across all datasets, requires no ground-truth ITE).
  - Strategy: random search with a fixed number of samples (identical budget
    across models for fairness), seeded for reproducibility.

Leakage contract: tuning operates ONLY on rows passed in (the outer training
fold). Inner folds are drawn from those rows; test-fold rows never enter here.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from uplift_bench.experiments.splitting import make_strat_labels
from uplift_bench.metrics.ranking import qini_coefficient
from uplift_bench.models.base_learners import SHARED_HPARAM_SPACE
from uplift_bench.models.registry import get_estimator

log = logging.getLogger(__name__)


# Native search spaces for forest/tree models (distinct, non-shared hyperparameters).
# Meta-learners and baselines use the SHARED base-learner space so that differences
# reflect the meta-strategy, not tuning luck (per spec §5).
_FOREST_SPACES: dict[str, dict] = {
    "causal_forest": {"n_estimators": [100, 200, 300]},
    "uplift_rf_kl": {
        "n_estimators": [50, 100, 200],
        "max_depth": [4, 6, 8],
        "min_samples_leaf": [25, 50, 100],
    },
}
_FOREST_SPACES["uplift_rf_ed"] = _FOREST_SPACES["uplift_rf_kl"]
_FOREST_SPACES["uplift_rf_chi"] = _FOREST_SPACES["uplift_rf_kl"]


def get_search_space(model_name: str, base_learner: str) -> dict[str, list]:
    """Return the hyperparameter search space for a model."""
    if model_name in _FOREST_SPACES:
        return _FOREST_SPACES[model_name]
    return SHARED_HPARAM_SPACE[base_learner]


def _sample_hparams(space: dict[str, list], rng: np.random.Generator) -> dict[str, Any]:
    """Draw one random hyperparameter configuration from the space."""
    return {key: rng.choice(values).item() for key, values in space.items()}


def _inner_qini(
    model_name: str,
    base_learner: str,
    hparams: dict,
    X: pd.DataFrame,
    t: pd.Series,
    y: pd.Series,
    prop: np.ndarray,
    inner_folds: int,
    seed: int,
) -> float:
    """Mean validation Qini for one HP config via inner stratified CV."""
    from sklearn.model_selection import StratifiedKFold

    strat = make_strat_labels(t, y)
    skf = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=seed)
    scores = []
    for tr, va in skf.split(np.zeros(len(X)), strat):
        try:
            est = get_estimator(model_name, base_learner=base_learner, seed=seed, **hparams)
            est.fit(
                X.iloc[tr],
                t.iloc[tr],
                y.iloc[tr],
                pd.Series(prop[tr]) if prop is not None else None,
            )
            pred = est.predict_uplift(X.iloc[va])
            q = qini_coefficient(
                pred, t.iloc[va].to_numpy(), y.iloc[va].to_numpy(), normalize=False
            )
            scores.append(q)
        except Exception as exc:  # a bad HP combo shouldn't kill the search
            log.debug("inner fold failed for %s hparams=%s: %s", model_name, hparams, exc)
            scores.append(np.nan)
    if np.all(np.isnan(scores)):
        return -np.inf
    return float(np.nanmean(scores))


def tune_hyperparameters(
    model_name: str,
    base_learner: str,
    X: pd.DataFrame,
    treatment: pd.Series,
    outcome: pd.Series,
    propensity: np.ndarray | None,
    n_samples: int = 20,
    inner_folds: int = 3,
    seed: int = 42,
    enabled: bool = True,
) -> tuple[dict, float]:
    """Random-search tune against validation Qini on the given (training) rows.

    Returns (best_hparams, best_score). If tuning is disabled or only one config
    exists, returns the default config with score NaN.
    """
    X = X.reset_index(drop=True)
    treatment = treatment.reset_index(drop=True)
    outcome = outcome.reset_index(drop=True)

    space = get_search_space(model_name, base_learner)
    if not enabled or not space:
        return {}, float("nan")

    rng = np.random.default_rng(seed)
    # Deduplicate sampled configs to avoid wasted evaluations on small spaces.
    seen: set = set()
    candidates: list[dict] = []
    for _ in range(n_samples * 3):  # oversample then dedupe
        cfg = _sample_hparams(space, rng)
        key = tuple(sorted(cfg.items()))
        if key not in seen:
            seen.add(key)
            candidates.append(cfg)
        if len(candidates) >= n_samples:
            break

    best_hp: dict = {}
    best_score = -np.inf
    for cfg in candidates:
        score = _inner_qini(
            model_name,
            base_learner,
            cfg,
            X,
            treatment,
            outcome,
            propensity,
            inner_folds,
            seed,
        )
        if score > best_score:
            best_score = score
            best_hp = cfg

    log.info(
        "Tuned %s: best validation Qini=%.4f from %d configs",
        model_name,
        best_score,
        len(candidates),
    )
    return best_hp, (best_score if np.isfinite(best_score) else float("nan"))
