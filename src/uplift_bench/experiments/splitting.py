"""Repeated stratified K-fold splitting with explicit leakage guards.

Stratification is by treatment × outcome (binary) or treatment × outcome-quantile
(continuous), so every fold preserves both the treatment fraction and the outcome
distribution — a key requirement of the leakage-free protocol.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

log = logging.getLogger(__name__)


def make_strat_labels(treatment: pd.Series, outcome: pd.Series, n_bins: int = 5) -> np.ndarray:
    """Build stratification labels combining treatment and (binned) outcome."""
    t = np.asarray(treatment).astype(int)
    y = np.asarray(outcome, dtype=float)

    unique_y = np.unique(y)
    if len(unique_y) <= 2:
        # Binary / near-binary outcome: stratify by exact value
        y_bins = y.astype(int)
    else:
        # Continuous outcome: stratify by quantile bin
        try:
            y_bins = pd.qcut(y, q=min(n_bins, len(unique_y)), labels=False, duplicates="drop")
            y_bins = np.asarray(y_bins, dtype=int)
        except ValueError:
            y_bins = np.zeros_like(y, dtype=int)

    # Build combined labels with a single consistent unicode dtype
    t_lab = t.astype(str).astype("U")
    y_lab = y_bins.astype(str).astype("U")
    return np.char.add(np.char.add(t_lab, "_"), y_lab)


def assert_no_leakage(train_idx: np.ndarray, test_idx: np.ndarray, n_total: int) -> None:
    """Hard guard: train and test index sets must be disjoint and complete.

    Raises AssertionError on any overlap or missing coverage.
    """
    train_set = set(np.asarray(train_idx).tolist())
    test_set = set(np.asarray(test_idx).tolist())
    overlap = train_set & test_set
    assert not overlap, f"LEAKAGE: {len(overlap)} indices appear in both train and test"
    assert len(train_set) + len(test_set) == n_total, (
        f"Index coverage mismatch: |train|={len(train_set)} + |test|={len(test_set)} "
        f"!= n={n_total}"
    )
    assert len(train_set) == len(train_idx), "Duplicate indices in train fold"
    assert len(test_set) == len(test_idx), "Duplicate indices in test fold"


def repeated_stratified_kfold(
    treatment: pd.Series,
    outcome: pd.Series,
    n_folds: int = 5,
    n_seeds: int = 5,
    base_seed: int = 42,
):
    """Yield (seed_idx, fold_idx, train_idx, test_idx) for the full repeated CV.

    Each repeat uses a different shuffle seed; within a repeat, StratifiedKFold
    produces ``n_folds`` disjoint test folds. Every yielded split is checked by
    ``assert_no_leakage`` before being returned.
    """
    n = len(treatment)
    strat = make_strat_labels(treatment, outcome)

    for seed_idx in range(n_seeds):
        seed = base_seed + seed_idx
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(np.zeros(n), strat)):
            assert_no_leakage(train_idx, test_idx, n)
            log.debug(
                "split seed=%d fold=%d: n_train=%d n_test=%d (train_frac_t=%.3f)",
                seed,
                fold_idx,
                len(train_idx),
                len(test_idx),
                float(np.asarray(treatment)[train_idx].mean()),
            )
            yield seed_idx, fold_idx, train_idx, test_idx
