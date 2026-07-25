"""IHDP 100-split loader (Infant Health and Development Program, semi-synthetic).

Standard ML benchmark for CATE estimation with known ground-truth effects.
Uses the 100 train/test splits from Shalit et al. (2017) / Johansson et al.

Sources (NPZ format, ~1 MB each):
  https://www.fredjo.com/files/ihdp_npci_1-100.train.npz
  https://www.fredjo.com/files/ihdp_npci_1-100.test.npz

Each split: X (747×25), t (747,), yf (747,), ycf (747,), mu0 (747,), mu1 (747,).
Potential outcomes: y0 = mu0, y1 = mu1.
ITE = mu1 - mu0.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from uplift_bench.data._cache import download, get_data_dir
from uplift_bench.data.base import DatasetMeta, UpliftDataset

log = logging.getLogger(__name__)

_TRAIN_URL = "https://www.fredjo.com/files/ihdp_npci_1-100.train.npz"
_TEST_URL = "https://www.fredjo.com/files/ihdp_npci_1-100.test.npz"

_N_FEATURES = 25
_FEATURE_NAMES = [f"x{i}" for i in range(_N_FEATURES)]


def load_ihdp(
    split_idx: int = 0,
    use_test: bool = False,
    data_dir: Optional[Path] = None,
) -> UpliftDataset:
    """Load one IHDP 100-split.

    Parameters
    ----------
    split_idx: which of the 100 splits to load (0-indexed, 0..99)
    use_test:  if True, return the test portion; else training portion
    data_dir:  override cache directory
    """
    if not (0 <= split_idx < 100):
        raise ValueError(f"split_idx must be 0..99, got {split_idx}")

    data_dir = Path(data_dir) if data_dir else get_data_dir() / "ihdp"
    train_path = data_dir / "ihdp_npci_1-100.train.npz"
    test_path = data_dir / "ihdp_npci_1-100.test.npz"

    if not train_path.exists():
        log.info("Downloading IHDP train splits...")
        download(_TRAIN_URL, train_path)
    if not test_path.exists():
        log.info("Downloading IHDP test splits...")
        download(_TEST_URL, test_path)

    path = test_path if use_test else train_path
    data = np.load(path)

    # Arrays are shaped (n, n_splits) except X which is (n, p, n_splits)
    i = split_idx
    X_arr = data["x"][:, :, i]  # (n, 25)
    t_arr = data["t"][:, i].astype(int)
    yf_arr = data["yf"][:, i]  # factual outcome
    mu0_arr = data["mu0"][:, i]
    mu1_arr = data["mu1"][:, i]
    ite_arr = mu1_arr - mu0_arr

    X = pd.DataFrame(X_arr, columns=_FEATURE_NAMES)
    treatment = pd.Series(t_arr, name="treatment")
    outcome = pd.Series(yf_arr, name="outcome")
    ite_df = pd.DataFrame({"ite": ite_arr, "mu0": mu0_arr, "mu1": mu1_arr})

    prop_value = float(treatment.mean())
    # IHDP is observational (not perfectly randomized); propensity is None
    # so that the runner estimates it during evaluation
    ds = UpliftDataset(
        X=X,
        treatment=treatment,
        outcome=outcome,
        propensity=None,
        ite=ite_df,
        meta=DatasetMeta(
            name=f"ihdp_split{split_idx}{'_test' if use_test else ''}",
            n=len(X),
            n_features=_N_FEATURES,
            treatment_fraction=prop_value,
            outcome_base_rate=float(outcome.mean()),
            has_ground_truth_effect=True,
            extras={"split_idx": split_idx, "use_test": use_test},
        ),
    )
    ds.validate()
    return ds


def load_ihdp_all_splits(
    use_test: bool = False,
    data_dir: Optional[Path] = None,
) -> list[UpliftDataset]:
    """Return all 100 IHDP splits as a list."""
    return [load_ihdp(i, use_test=use_test, data_dir=data_dir) for i in range(100)]
