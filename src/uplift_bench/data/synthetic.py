"""In-memory synthetic dataset — no download, used for smoke runs and tests.

Generates a dataset with a known heterogeneous treatment effect so that both
ranking metrics (Qini/AUUC) and ground-truth metrics (PEHE/ε_ATE) can be
exercised end-to-end without any network access.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from uplift_bench.data.base import DatasetMeta, UpliftDataset


def load_synthetic(
    n: int = 2000,
    n_features: int = 8,
    treatment_fraction: float = 0.5,
    seed: int = 42,
    binary_outcome: bool = True,
    has_ground_truth: bool = True,
    known_propensity: bool = True,
    data_dir=None,  # accepted for interface symmetry; unused
) -> UpliftDataset:
    """Generate a synthetic uplift dataset.

    Parameters
    ----------
    n                  : number of rows
    n_features         : number of features
    treatment_fraction : P(T=1) (constant — RCT-style)
    seed               : RNG seed
    binary_outcome     : if True, outcome is {0,1}; else continuous
    has_ground_truth   : if True, attach per-row ITE (enables PEHE/ε_ATE)
    known_propensity   : if True, attach the known constant propensity
    """
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        rng.standard_normal((n, n_features)),
        columns=[f"f{i}" for i in range(n_features)],
    )
    treatment = pd.Series(rng.binomial(1, treatment_fraction, n).astype(int), name="treatment")

    # Heterogeneous treatment effect driven by the first two features
    tau = 0.5 * X["f0"].values + 0.3 * X["f1"].values  # true ITE
    base = 0.2 * X["f2"].values  # baseline outcome signal

    if binary_outcome:
        p0 = np.clip(0.4 + 0.1 * base, 0.01, 0.99)
        p1 = np.clip(p0 + 0.15 * tau, 0.01, 0.99)
        y0 = rng.binomial(1, p0)
        y1 = rng.binomial(1, p1)
        ite = (p1 - p0).astype(float)  # ground-truth CATE on the probability scale
    else:
        noise = rng.standard_normal(n) * 0.5
        y0 = base + noise
        y1 = base + tau + noise
        ite = tau.astype(float)

    outcome_arr = np.where(treatment.values == 1, y1, y0).astype(float)
    outcome = pd.Series(outcome_arr, name="outcome")

    propensity = (
        pd.Series(np.full(n, treatment_fraction), name="propensity") if known_propensity else None
    )
    ite_df = pd.DataFrame({"ite": ite}) if has_ground_truth else None

    ds = UpliftDataset(
        X=X,
        treatment=treatment,
        outcome=outcome,
        propensity=propensity,
        ite=ite_df,
        meta=DatasetMeta(
            name="synthetic",
            n=n,
            n_features=n_features,
            treatment_fraction=float(treatment.mean()),
            outcome_base_rate=float(outcome.mean()),
            has_ground_truth_effect=has_ground_truth,
            extras={"binary_outcome": binary_outcome, "seed": seed},
        ),
    )
    ds.validate()
    return ds
