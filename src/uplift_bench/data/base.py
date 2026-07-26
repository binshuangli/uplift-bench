"""Standardized container returned by every dataset loader."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class DatasetMeta:
    name: str
    n: int
    n_features: int
    treatment_fraction: float
    outcome_base_rate: float
    # True only where per-unit effects are known (IHDP, synthetic, ACIC, revenue-synthetic).
    # Jobs is False: it supplies an RCT-estimated reference *policy* objective, not effects.
    has_ground_truth_effect: bool
    # Machine-readable outcome type. The continuous-vs-binary split is the paper's central
    # axis, so it must not be re-derived by hand in each analysis script: doing that once
    # put the binary `synthetic` dataset into the "continuous" panel. Default None means
    # "not declared by the loader"; `UpliftDataset.validate()` fills it in from the data.
    outcome_type: str | None = None
    # extra dataset-specific fields stored here
    extras: dict = field(default_factory=dict)


@dataclass
class UpliftDataset:
    """
    Standardized container for uplift/CATE datasets.

    X            : pd.DataFrame, shape (n, p)  — features only (no treatment/outcome)
    treatment    : pd.Series, dtype int {0,1}
    outcome      : pd.Series, dtype float or int
    propensity   : pd.Series or None  — P(T=1|X); known for RCTs, None otherwise
    ite          : pd.DataFrame or None — ground-truth individual effects (IHDP/Jobs)
    meta         : DatasetMeta
    """

    X: pd.DataFrame
    treatment: pd.Series
    outcome: pd.Series
    propensity: Optional[pd.Series]
    ite: Optional[pd.DataFrame]
    meta: DatasetMeta

    # ------------------------------------------------------------------
    @classmethod
    def from_frame(
        cls,
        df: pd.DataFrame,
        *,
        treatment_col: str,
        outcome_col: str,
        feature_cols: Optional[list[str]] = None,
        propensity_col: Optional[str] = None,
        ite_col: Optional[str] = None,
        name: str = "user_dataset",
    ) -> "UpliftDataset":
        """Bring-your-own-data adapter: build an :class:`UpliftDataset` from a DataFrame.

        Specify the covariates, treatment, and outcome columns (plus optional propensity
        and reference-effect columns); every benchmark estimator and metric then runs on
        the result exactly as on the built-in datasets. This is the documented entry point
        for evaluating user datasets against the released protocol.

        Parameters
        ----------
        df             : the source table.
        treatment_col  : binary {0,1} treatment column name.
        outcome_col    : outcome column name (float or {0,1}).
        feature_cols   : covariate columns (default: all columns except the ones named above).
        propensity_col : optional P(T=1|X) column; enables the known-propensity policy.
        ite_col        : optional ground-truth individual effect column; enables sqrt(PEHE).
        name           : label recorded in the dataset metadata.
        """
        named = {treatment_col, outcome_col}
        if propensity_col:
            named.add(propensity_col)
        if ite_col:
            named.add(ite_col)
        if feature_cols is None:
            feature_cols = [c for c in df.columns if c not in named]

        X = df[feature_cols].reset_index(drop=True)
        treatment = pd.Series(df[treatment_col].astype(int).to_numpy(), name="treatment")
        outcome = pd.Series(df[outcome_col].to_numpy(), name="outcome")
        propensity = (
            pd.Series(df[propensity_col].astype(float).to_numpy(), name="propensity")
            if propensity_col
            else None
        )
        ite = pd.DataFrame({"ite": df[ite_col].astype(float).to_numpy()}) if ite_col else None
        ds = cls(
            X=X,
            treatment=treatment,
            outcome=outcome,
            propensity=propensity,
            ite=ite,
            meta=DatasetMeta(
                name=name,
                n=len(X),
                n_features=X.shape[1],
                treatment_fraction=float(treatment.mean()),
                outcome_base_rate=float(outcome.mean()),
                has_ground_truth_effect=ite is not None,
                extras={"source": "from_frame"},
            ),
        )
        ds.validate()
        return ds

    # ------------------------------------------------------------------
    def validate(self) -> None:
        """Raise ValueError on obvious integrity violations."""
        n = len(self.X)
        assert len(self.treatment) == n, "treatment length mismatch"
        assert len(self.outcome) == n, "outcome length mismatch"
        assert set(self.treatment.unique()).issubset({0, 1}), "treatment must be binary {0,1}"
        if self.propensity is not None:
            assert len(self.propensity) == n, "propensity length mismatch"
            assert (self.propensity > 0).all() and (
                self.propensity < 1
            ).all(), "propensity must be in (0,1)"
        # No column in X should be treatment or outcome (leakage guard)
        bad = set(self.X.columns) & {"treatment", "outcome", "y", "t", "converted", "visit"}
        if bad:
            raise ValueError(f"Feature matrix X contains reserved column names: {bad}")

        # Derive the outcome type from the data, and hold declaring loaders to it.
        observed = "binary" if set(self.outcome.unique()).issubset({0, 1}) else "continuous"
        if self.meta.outcome_type is None:
            self.meta.outcome_type = observed
        elif self.meta.outcome_type != observed:
            raise ValueError(
                f"{self.meta.name}: loader declares outcome_type="
                f"{self.meta.outcome_type!r} but the outcome column is {observed}"
            )

    def subsample(self, n: int, seed: int = 42) -> "UpliftDataset":
        """Return a stratified subsample of size n (stratified by treatment × outcome)."""
        from sklearn.model_selection import StratifiedShuffleSplit

        if n >= len(self.X):
            return self

        # Stratify by treatment x outcome only where the outcome is discrete. Casting a
        # continuous outcome to int creates near-singleton strata and can make
        # StratifiedShuffleSplit fail outright, so continuous outcomes stratify by
        # treatment alone (external audit finding).
        outcome_vals = self.outcome.to_numpy()
        is_binary = set(np.unique(outcome_vals[~np.isnan(outcome_vals)])) <= {0, 1, 0.0, 1.0}
        if is_binary:
            strat_key = self.treatment.astype(str) + "_" + self.outcome.astype(int).astype(str)
        else:
            strat_key = self.treatment.astype(str)
        sss = StratifiedShuffleSplit(n_splits=1, train_size=n, random_state=seed)
        idx, _ = next(sss.split(self.X, strat_key))
        idx = np.sort(idx)

        new_meta = DatasetMeta(
            name=self.meta.name,
            n=n,
            n_features=self.meta.n_features,
            treatment_fraction=float(self.treatment.iloc[idx].mean()),
            outcome_base_rate=float(self.outcome.iloc[idx].mean()),
            has_ground_truth_effect=self.meta.has_ground_truth_effect,
            outcome_type=self.meta.outcome_type,
            extras=self.meta.extras,
        )
        return UpliftDataset(
            X=self.X.iloc[idx].reset_index(drop=True),
            treatment=self.treatment.iloc[idx].reset_index(drop=True),
            outcome=self.outcome.iloc[idx].reset_index(drop=True),
            propensity=(
                self.propensity.iloc[idx].reset_index(drop=True)
                if self.propensity is not None
                else None
            ),
            ite=(self.ite.iloc[idx].reset_index(drop=True) if self.ite is not None else None),
            meta=new_meta,
        )
