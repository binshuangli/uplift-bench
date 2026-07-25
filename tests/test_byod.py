"""Tests for the bring-your-own-data adapter UpliftDataset.from_frame."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from uplift_bench.data.base import UpliftDataset


def _frame(n=200, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "x0": rng.normal(size=n),
            "x1": rng.normal(size=n),
            "w": rng.binomial(1, 0.5, size=n),
            "y": rng.normal(size=n),
            "e": np.full(n, 0.5),
            "tau": rng.normal(size=n),
        }
    )


def test_minimal_adapter():
    df = _frame()
    ds = UpliftDataset.from_frame(df, treatment_col="w", outcome_col="y")
    assert list(ds.X.columns) == ["x0", "x1", "e", "tau"]  # all non-named cols become features
    assert ds.meta.n == len(df)
    assert ds.propensity is None and ds.ite is None
    assert not ds.meta.has_ground_truth_effect


def test_full_adapter_with_propensity_and_ite():
    df = _frame()
    ds = UpliftDataset.from_frame(
        df,
        treatment_col="w",
        outcome_col="y",
        feature_cols=["x0", "x1"],
        propensity_col="e",
        ite_col="tau",
        name="my_data",
    )
    assert list(ds.X.columns) == ["x0", "x1"]
    assert ds.propensity is not None and (ds.propensity == 0.5).all()
    assert ds.ite is not None and "ite" in ds.ite.columns
    assert ds.meta.has_ground_truth_effect
    assert ds.meta.name == "my_data"


def test_validation_still_runs():
    df = _frame()
    df["w"] = 2  # invalid treatment
    with pytest.raises(AssertionError):
        UpliftDataset.from_frame(df, treatment_col="w", outcome_col="y")
