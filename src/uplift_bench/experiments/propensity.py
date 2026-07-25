"""Per-dataset propensity policy (a key leakage control).

Policy (per spec §6):
  - RCT datasets (known propensity present): use the known randomization
    propensity directly.
  - Observational / semi-synthetic (propensity is None): estimate P(T=1|X) by
    fitting a classifier on the TRAINING fold ONLY, then predict on both folds.

The estimator is never shown test-fold features during fitting — this is the
explicit no-leakage contract enforced here.
"""

from __future__ import annotations

import logging

import numpy as np

from uplift_bench.data.base import UpliftDataset
from uplift_bench.models.base_learners import make_base_learner

log = logging.getLogger(__name__)

_CLIP = 0.01  # keep propensity in [clip, 1-clip] to avoid IPW blow-up


def resolve_propensity(
    ds: UpliftDataset,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    base_learner: str = "lightgbm",
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Return (train_propensity, test_propensity, policy_name).

    policy_name is 'known' for RCTs or 'estimated_on_train' for observational.
    """
    if ds.propensity is not None:
        # Known randomization propensity (RCT) — use as-is.
        train_prop = ds.propensity.iloc[train_idx].to_numpy(dtype=float)
        test_prop = ds.propensity.iloc[test_idx].to_numpy(dtype=float)
        return train_prop, test_prop, "known"

    # Observational: estimate on TRAIN ONLY.
    X_train = ds.X.iloc[train_idx].to_numpy(dtype=float)
    t_train = ds.treatment.iloc[train_idx].to_numpy(dtype=int)
    X_test = ds.X.iloc[test_idx].to_numpy(dtype=float)

    clf = make_base_learner(base_learner, "classification", seed)
    clf.fit(X_train, t_train)  # <-- fit sees train features only

    train_prop = np.clip(clf.predict_proba(X_train)[:, 1], _CLIP, 1 - _CLIP)
    test_prop = np.clip(clf.predict_proba(X_test)[:, 1], _CLIP, 1 - _CLIP)
    log.debug(
        "Estimated propensity on train (n=%d); mean train=%.3f test=%.3f",
        len(train_idx),
        train_prop.mean(),
        test_prop.mean(),
    )
    return train_prop, test_prop, "estimated_on_train"
