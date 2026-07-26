"""Simulated revenue-uplift dataset (continuous, heavy-tailed outcome).

Models the canonical online-marketing revenue setting: baseline spend is lognormal
(many small spenders, a few very large ones) and the treatment effect is multiplicative,
so absolute uplift is largest for the biggest spenders and the row-level ITE is itself
heavy-tailed. Ground-truth ITE is attached, enabling PEHE evaluation. This is the
regime where a cumulative-sum statistic is most exposed to single large outcomes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from uplift_bench.data.base import DatasetMeta, UpliftDataset


def load_revenue_synthetic(
    n: int = 4000,
    n_features: int = 10,
    treatment_fraction: float = 0.5,
    sigma: float = 1.0,
    seed: int = 42,
    data_dir=None,  # accepted for interface symmetry; unused
) -> UpliftDataset:
    """Generate a revenue-uplift dataset.

    Parameters
    ----------
    n                  : number of rows
    n_features         : number of features
    treatment_fraction : P(T=1) (constant — RCT-style)
    sigma              : lognormal shape of baseline spend (1.0 → heavy right tail)
    seed               : RNG seed
    """
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        rng.standard_normal((n, n_features)),
        columns=[f"f{i}" for i in range(n_features)],
    )
    treatment = pd.Series(rng.binomial(1, treatment_fraction, n).astype(int), name="treatment")

    # Baseline spend: lognormal with mean driven by f2 (spend propensity).
    mu = 1.0 + 0.3 * X["f2"].values
    y0 = np.exp(mu + sigma * rng.standard_normal(n))

    # Multiplicative treatment effect in (-0.1, +0.3), heterogeneous in f0/f1:
    # large spenders who respond gain the most in absolute terms; some units are harmed.
    logit = 2.0 * X["f0"].values + 0.5 * X["f1"].values
    delta = 0.4 / (1.0 + np.exp(-logit)) - 0.1
    y1 = y0 * (1.0 + delta)
    ite = (y1 - y0).astype(float)  # row-level true ITE (heavy-tailed by construction)

    outcome_arr = np.where(treatment.values == 1, y1, y0).astype(float)
    outcome = pd.Series(outcome_arr, name="outcome")
    propensity = pd.Series(np.full(n, treatment_fraction), name="propensity")

    return UpliftDataset(
        X=X,
        treatment=treatment,
        outcome=outcome,
        propensity=propensity,
        ite=pd.DataFrame({"ite": ite}),
        meta=DatasetMeta(
            name="revenue_synthetic",
            n=n,
            n_features=n_features,
            treatment_fraction=float(treatment.mean()),
            outcome_base_rate=float(outcome.mean()),
            has_ground_truth_effect=True,
            extras={"outcome_type": "continuous", "dgp": "lognormal spend, multiplicative effect"},
        ),
    )
