"""Jobs (LaLonde) dataset loader.

The Jobs dataset is used for policy-value / ATT estimation with a known
experimental benchmark (LaLonde 1986 experimental group vs CPS/PSID controls).

We use the standard split from Smith & Todd (2005) as distributed by
Shalit et al. (ICML 2017) — same NPZ format as IHDP.

Source:
  https://www.fredjo.com/files/jobs_DW_bin.new.10.train.npz
  https://www.fredjo.com/files/jobs_DW_bin.new.10.test.npz

Fields: x (n,17,splits), t (n,splits), yf (n,splits), e (n,splits) [experimental flag].
Ground-truth: policy risk computed using experimental subset (e==1).
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

_TRAIN_URL = "https://www.fredjo.com/files/jobs_DW_bin.new.10.train.npz"
_TEST_URL = "https://www.fredjo.com/files/jobs_DW_bin.new.10.test.npz"

_N_FEATURES = 17
_FEATURE_NAMES = [f"x{i}" for i in range(_N_FEATURES)]
_N_SPLITS = 10


def load_jobs(
    split_idx: int = 0,
    use_test: bool = False,
    data_dir: Optional[Path] = None,
) -> UpliftDataset:
    """Load one Jobs 10-split.

    Parameters
    ----------
    split_idx: which of the 10 splits (0-indexed, 0..9)
    use_test:  if True, return the test portion
    data_dir:  override cache directory
    """
    if not (0 <= split_idx < _N_SPLITS):
        raise ValueError(f"split_idx must be 0..{_N_SPLITS - 1}, got {split_idx}")

    data_dir = Path(data_dir) if data_dir else get_data_dir() / "jobs"
    train_path = data_dir / "jobs_DW_bin.new.10.train.npz"
    test_path = data_dir / "jobs_DW_bin.new.10.test.npz"

    if not train_path.exists():
        log.info("Downloading Jobs train splits...")
        download(_TRAIN_URL, train_path)
    if not test_path.exists():
        log.info("Downloading Jobs test splits...")
        download(_TEST_URL, test_path)

    path = test_path if use_test else train_path
    data = np.load(path)

    i = split_idx
    X_arr = data["x"][:, :, i]
    t_arr = data["t"][:, i].astype(int)
    yf_arr = data["yf"][:, i]
    # 'e' flag: 1 = experimental subset (LaLonde RCT), 0 = observational (CPS/PSID)
    e_arr = data["e"][:, i].astype(int) if "e" in data else np.ones(len(t_arr), dtype=int)

    X = pd.DataFrame(X_arr, columns=_FEATURE_NAMES)
    treatment = pd.Series(t_arr, name="treatment")
    outcome = pd.Series(yf_arr, name="outcome")

    # ITE not directly available; store the experimental flag so the metrics
    # module can compute policy risk correctly
    ite_df = pd.DataFrame({"experimental": e_arr})

    # Jobs mixes experimental and observational; propensity is None (estimated per policy)
    ds = UpliftDataset(
        X=X,
        treatment=treatment,
        outcome=outcome,
        propensity=None,
        ite=ite_df,
        meta=DatasetMeta(
            name=f"jobs_split{split_idx}{'_test' if use_test else ''}",
            n=len(X),
            n_features=_N_FEATURES,
            treatment_fraction=float(treatment.mean()),
            outcome_base_rate=float(outcome.mean()),
            # Jobs has NO individual treatment effects: the `ite` frame carries only the
            # `experimental` flag, which identifies the randomized subset on which the
            # RCT-estimated reference policy risk (1 - E[Y(pi)]) is computed. It is a
            # reference *policy* objective, not an oracle effect or optimal policy.
            has_ground_truth_effect=False,
            extras={"split_idx": split_idx, "use_test": use_test, "n_splits": _N_SPLITS},
        ),
    )
    ds.validate()
    return ds


def load_jobs_all_splits(
    use_test: bool = False,
    data_dir: Optional[Path] = None,
) -> list[UpliftDataset]:
    """Return all 10 Jobs splits."""
    return [load_jobs(i, use_test=use_test, data_dir=data_dir) for i in range(_N_SPLITS)]
