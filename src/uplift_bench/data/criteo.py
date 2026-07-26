"""Criteo Uplift v2.1 loader with subsampling tiers.

Source: http://go.criteo.net/criteo-research-uplift-v2.1.csv.gz
~13.98M rows, binary treatment, visit + conversion outcomes.
Known (uniform) propensity from RCT design.
"""

from __future__ import annotations

import gzip
import logging
from pathlib import Path
from typing import Literal, Optional

import pandas as pd

from uplift_bench.data._cache import download, get_data_dir
from uplift_bench.data.base import DatasetMeta, UpliftDataset

log = logging.getLogger(__name__)

_URL = "http://go.criteo.net/criteo-research-uplift-v2.1.csv.gz"
_FILENAME = "criteo-uplift-v2.1.csv.gz"

# Feature columns (all numeric; treatment and outcome are separate)
_FEATURE_COLS = [f"f{i}" for i in range(12)]

# Subsampling sizes
CRITEO_TIERS: dict[str, Optional[int]] = {
    "1M": 1_000_000,
    "5M": 5_000_000,
    "full": None,
}


def load_criteo(
    outcome: Literal["visit", "conversion"] = "visit",
    subsample_tier: Literal["1M", "5M", "full"] = "1M",
    seed: int = 42,
    data_dir: Optional[Path] = None,
) -> UpliftDataset:
    """Load Criteo Uplift v2.1.

    Parameters
    ----------
    outcome:        'visit' or 'conversion'
    subsample_tier: '1M', '5M', or 'full'
    seed:           random seed for stratified subsampling
    data_dir:       override cache directory
    """
    data_dir = Path(data_dir) if data_dir else get_data_dir() / "criteo"
    dest = data_dir / _FILENAME

    if not dest.exists():
        log.info("Criteo not found locally; downloading (~1 GB compressed)...")
        download(_URL, dest)

    log.info("Reading Criteo from %s ...", dest)
    with gzip.open(dest, "rt") as fh:
        df = pd.read_csv(fh)

    # Rename to canonical columns
    df = df.rename(columns={"treatment": "treatment", "visit": "visit", "conversion": "conversion"})

    treatment = df["treatment"].astype(int)
    out = df[outcome].astype(int)
    # Criteo is a randomized experiment; propensity is the empirical treatment fraction
    # (constant by design — report the empirical estimate)
    prop_value = float(treatment.mean())
    propensity = pd.Series([prop_value] * len(df), name="propensity")

    X = df[_FEATURE_COLS].copy()

    n_full = len(X)
    ds = UpliftDataset(
        X=X.reset_index(drop=True),
        treatment=treatment.reset_index(drop=True),
        outcome=out.reset_index(drop=True),
        propensity=propensity.reset_index(drop=True),
        ite=None,
        meta=DatasetMeta(
            name=f"criteo_{outcome}",
            n=n_full,
            n_features=len(_FEATURE_COLS),
            treatment_fraction=prop_value,
            outcome_base_rate=float(out.mean()),
            has_ground_truth_effect=False,
            extras={"outcome_col": outcome, "tier": subsample_tier},
        ),
    )

    target_n = CRITEO_TIERS[subsample_tier]
    if target_n is not None and target_n < n_full:
        ds = ds.subsample(target_n, seed=seed)

    ds.validate()
    return ds
