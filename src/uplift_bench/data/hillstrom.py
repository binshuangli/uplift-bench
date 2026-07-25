"""Hillstrom MineThatData email marketing dataset.

Source: Kevin Hillstrom's blog; ~64 k rows.
Two treatment arms (Men's email, Women's email) + control.
Standard setup: collapse to a single binary treatment (any email vs no email),
predict visit or spend.

URL: https://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal, Optional

import pandas as pd

from uplift_bench.data._cache import download, get_data_dir
from uplift_bench.data.base import DatasetMeta, UpliftDataset

log = logging.getLogger(__name__)

# HTTP (not HTTPS) — the minethatdata.com server cert doesn't cover this hostname
_URL = (
    "http://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_"
    "DataMiningChallenge_2008.03.20.csv"
)
_FILENAME = "hillstrom.csv"

_FEATURE_COLS = [
    "recency",
    "history_segment",
    "history",
    "mens",
    "womens",
    "zip_code",
    "newbie",
    "channel",
]


def load_hillstrom(
    outcome: Literal["visit", "conversion", "spend"] = "visit",
    treatment_arm: Literal["any", "mens", "womens"] = "any",
    data_dir: Optional[Path] = None,
) -> UpliftDataset:
    """Load Hillstrom MineThatData email RCT.

    Parameters
    ----------
    outcome:       'visit', 'conversion', or 'spend'
    treatment_arm: 'any'   — any email vs no email (binary)
                   'mens'  — Men's email vs no email (drops Women's rows)
                   'womens'— Women's email vs no email (drops Men's rows)
    """
    data_dir = Path(data_dir) if data_dir else get_data_dir() / "hillstrom"
    dest = data_dir / _FILENAME

    if not dest.exists():
        log.info("Downloading Hillstrom dataset...")
        download(_URL, dest)

    df = pd.read_csv(dest)
    df.columns = [c.lower().strip() for c in df.columns]

    # 'segment' column: 'Mens E-Mail', 'Womens E-Mail', 'No E-Mail'
    if treatment_arm == "any":
        df = df.copy()
        df["treatment"] = (df["segment"] != "No E-Mail").astype(int)
    elif treatment_arm == "mens":
        df = df[df["segment"].isin(["Mens E-Mail", "No E-Mail"])].copy()
        df["treatment"] = (df["segment"] == "Mens E-Mail").astype(int)
    elif treatment_arm == "womens":
        df = df[df["segment"].isin(["Womens E-Mail", "No E-Mail"])].copy()
        df["treatment"] = (df["segment"] == "Womens E-Mail").astype(int)

    treatment = df["treatment"].astype(int)
    out_col = {"visit": "visit", "conversion": "conversion", "spend": "spend"}[outcome]
    outcome_s = df[out_col].astype(float)

    feature_df = df[[c for c in _FEATURE_COLS if c in df.columns]].copy()
    feature_df = pd.get_dummies(feature_df, drop_first=True)
    feature_df = feature_df.astype(float)

    prop_value = float(treatment.mean())
    propensity = pd.Series([prop_value] * len(feature_df), name="propensity")

    ds = UpliftDataset(
        X=feature_df.reset_index(drop=True),
        treatment=treatment.reset_index(drop=True),
        outcome=outcome_s.reset_index(drop=True),
        propensity=propensity,
        ite=None,
        meta=DatasetMeta(
            name=f"hillstrom_{outcome}_{treatment_arm}",
            n=len(feature_df),
            n_features=feature_df.shape[1],
            treatment_fraction=prop_value,
            outcome_base_rate=float(outcome_s.mean()),
            has_ground_truth_effect=False,
            extras={"outcome_col": outcome, "treatment_arm": treatment_arm},
        ),
    )
    ds.validate()
    return ds
