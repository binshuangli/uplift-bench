"""Loaders for scikit-uplift built-in datasets: Lenta, X5 RetailHero, MegaFon.

All three are downloaded and cached by scikit-uplift itself.
Lenta and X5 are marketing RCT datasets (known propensity = empirical fraction).
MegaFon is semi-synthetic (no ground-truth ITE; still RCT structure).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from uplift_bench.data.base import DatasetMeta, UpliftDataset

log = logging.getLogger(__name__)

# Known string→binary mappings for scikit-uplift treatment columns
_TREATMENT_MAP = {
    "test": 1,
    "treatment": 1,
    "control": 0,
    "1": 1,
    "0": 0,
}


def _coerce_treatment(s: pd.Series) -> pd.Series:
    """Map string or numeric treatment labels to binary int {0, 1}."""
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(int)
    mapped = s.map(_TREATMENT_MAP)
    if mapped.isna().any():
        raise ValueError(f"Unexpected treatment values: {s.unique().tolist()}")
    return mapped.astype(int)


# ------------------------------------------------------------------
# Shared helper
# ------------------------------------------------------------------


def _build_dataset(
    X: pd.DataFrame,
    treatment: pd.Series,
    outcome: pd.Series,
    name: str,
    has_ground_truth_effect: bool = False,
    extras: Optional[dict] = None,
) -> UpliftDataset:
    prop_value = float(treatment.mean())
    propensity = pd.Series([prop_value] * len(X), name="propensity")
    ds = UpliftDataset(
        X=X.reset_index(drop=True),
        treatment=treatment.astype(int).reset_index(drop=True),
        outcome=outcome.astype(float).reset_index(drop=True),
        propensity=propensity,
        ite=None,
        meta=DatasetMeta(
            name=name,
            n=len(X),
            n_features=X.shape[1],
            treatment_fraction=prop_value,
            outcome_base_rate=float(outcome.mean()),
            has_ground_truth_effect=has_ground_truth_effect,
            extras=extras or {},
        ),
    )
    ds.validate()
    return ds


# ------------------------------------------------------------------
# Lenta
# ------------------------------------------------------------------


_LENTA_GENDER_MAP = {"Ж": 1.0, "М": 0.0, "Не определен": float("nan")}


def load_lenta(data_dir: Optional[Path] = None) -> UpliftDataset:
    """Retail RCT from Lenta supermarket chain (mid-size, ~700 k rows)."""
    from sklift.datasets import fetch_lenta

    log.info("Loading Lenta via scikit-uplift...")
    bunch = fetch_lenta()
    df: pd.DataFrame = bunch.data.copy()
    # Encode Cyrillic gender column to numeric (Ж=female=1, М=male=0, unknown=NaN)
    if "gender" in df.columns:
        df["gender"] = df["gender"].map(_LENTA_GENDER_MAP).astype("float32")
    treatment = _coerce_treatment(bunch.treatment).rename("treatment")
    outcome = bunch.target.rename("outcome")
    return _build_dataset(df, treatment, outcome, name="lenta")


# ------------------------------------------------------------------
# X5 RetailHero
# ------------------------------------------------------------------


def load_x5(data_dir: Optional[Path] = None) -> UpliftDataset:
    """Retail loyalty-card RCT from X5 Group (~200 k rows in train split).

    Scikit-uplift exposes b.treatment (binary) and b.target at the top level,
    with b.data['clients'] holding per-client demographic features.
    b.data['train'] contains only client_id (no treatment/target columns).
    We join client features onto the train-set IDs, then attach treatment/target.
    """
    from sklift.datasets import fetch_x5

    log.info("Loading X5 RetailHero via scikit-uplift...")
    bunch = fetch_x5()
    clients: pd.DataFrame = bunch.data["clients"]
    train_ids: pd.DataFrame = bunch.data["train"]  # only column: client_id
    treatment_s: pd.Series = bunch.treatment  # binary 0/1, index aligns with train_ids
    target_s: pd.Series = bunch.target  # binary 0/1

    # Align train client features with treatment/target by position
    # (train_ids and bunch.treatment share the same positional index)
    train_client_ids = train_ids["client_id"].values
    clients_indexed = clients.set_index("client_id")

    # Keep only clients that appear in train set (inner join by position)
    df = clients_indexed.reindex(train_client_ids).reset_index(drop=True)

    # Drop date columns (high-cardinality strings, not usable as-is)
    drop_cols = [c for c in df.columns if "date" in c.lower() or "id" in c.lower()]
    df = df.drop(columns=drop_cols, errors="ignore")
    df = pd.get_dummies(df, drop_first=True).astype(float)

    treatment = treatment_s.reset_index(drop=True).rename("treatment").astype(int)
    outcome = target_s.reset_index(drop=True).rename("outcome").astype(float)

    return _build_dataset(df, treatment, outcome, name="x5")


# ------------------------------------------------------------------
# MegaFon
# ------------------------------------------------------------------


def load_megafon(data_dir: Optional[Path] = None) -> UpliftDataset:
    """Telecom semi-synthetic dataset from MegaFon (~600 k rows).

    bunch.treatment is a string Series: 'control' | 'treatment'.
    """
    from sklift.datasets import fetch_megafon

    log.info("Loading MegaFon via scikit-uplift...")
    bunch = fetch_megafon()
    df: pd.DataFrame = bunch.data
    treatment = _coerce_treatment(bunch.treatment).rename("treatment")
    outcome = bunch.target.rename("outcome")
    return _build_dataset(df, treatment, outcome, name="megafon")
