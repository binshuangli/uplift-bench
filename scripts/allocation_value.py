"""Empirical calibration->value demonstration (clean-Claude, reviewer comment B).

Tests, rather than asserts, the claim behind the budget-allocation example: under
*proportional* allocation (treatment probability proportional to predicted uplift, which
uses prediction magnitude and is therefore calibration-sensitive), does the best-CALIBRATED
model deliver higher realised policy value than the best-QINI model?

For each marketing RCT: leakage-safe split, fit fast estimators, on the test fold compute
Qini, calibration ECE, and the IPW value of the proportional-allocation policy at a budget b.
We then compare the model selected by Qini against the model selected by ECE.

Value estimator (RCT propensity e known):
    V(pi) = mean_i [ 1{t_i=1} pi_i / e + 1{t_i=0} (1-pi_i)/(1-e) ] * y_i
with pi_i proportional to max(0, uplift_i), scaled so mean(pi)=b.

Usage: python scripts/allocation_value.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit

from uplift_bench.data.registry import load_dataset
from uplift_bench.metrics.calibration import uplift_calibration_error
from uplift_bench.metrics.ranking import qini_coefficient
from uplift_bench.models.registry import get_estimator

DATASETS = [("Hillstrom", "hillstrom", {}), ("MegaFon", "megafon", {}), ("X5", "x5", {})]
MODELS = ["s_learner", "t_learner", "x_learner", "class_transformation", "two_model", "solo_model"]
BENCH_N, SEED, BUDGET = 10_000, 42, 0.3


def proportional_policy_value(uplift, t, y, e, budget):
    """IPW value of the stochastic policy pi_i propto max(0, uplift_i), mean(pi)=budget."""
    u = np.maximum(uplift, 0.0)
    if u.sum() <= 0:
        pi = np.full_like(u, budget, dtype=float)
    else:
        pi = u / u.mean() * budget  # mean(pi) = budget
        pi = np.clip(pi, 0.0, 1.0)
    w = np.where(t == 1, pi / e, (1.0 - pi) / (1.0 - e))
    return float(np.mean(w * y))


def run_dataset(loader, kwargs):
    ds = load_dataset(loader, **kwargs)
    if ds.meta.n > BENCH_N:
        ds = ds.subsample(BENCH_N, seed=SEED)
    strat = ds.treatment.astype(str) + "_" + ds.outcome.astype(int).astype(str)
    tr, te = next(StratifiedShuffleSplit(1, test_size=0.3, random_state=SEED).split(ds.X, strat))
    Xtr, Xte = ds.X.iloc[tr].copy(), ds.X.iloc[te].copy()
    ttr, tte = ds.treatment.iloc[tr], ds.treatment.iloc[te].to_numpy()
    ytr, yte = ds.outcome.iloc[tr], ds.outcome.iloc[te].to_numpy()
    if Xtr.isnull().any().any():
        imp = SimpleImputer(strategy="median")
        Xtr[:] = imp.fit_transform(Xtr)
        Xte[:] = imp.transform(Xte)
    e = float(ttr.mean())
    rows = []
    for m in MODELS:
        try:
            est = get_estimator(m, base_learner="lightgbm", seed=SEED)
            est.fit(Xtr, ttr, ytr)
            pred = est.predict_uplift(Xte)
            rows.append(
                {
                    "model": m,
                    "qini": qini_coefficient(pred, tte, yte, normalize=False),
                    "ece": uplift_calibration_error(pred, tte, yte),
                    "alloc_value": proportional_policy_value(pred, tte, yte, e, BUDGET),
                }
            )
        except Exception as exc:
            print(f"    [{loader}/{m}] skipped: {exc}")
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()
    out = Path(args.out_dir)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    summary = []
    for label, loader, kw in DATASETS:
        print(f"=== {label} ===")
        df = run_dataset(loader, kw)
        best_qini = df.loc[df.qini.idxmax()]
        best_ece = df.loc[df.ece.idxmin()]
        print(df.round(4).to_string(index=False))
        print(
            f"  best-Qini model: {best_qini['model']:20s} alloc_value={best_qini['alloc_value']:.4f}"
        )
        print(
            f"  best-ECE  model: {best_ece['model']:20s} alloc_value={best_ece['alloc_value']:.4f}"
        )
        winner = (
            "ECE-selected >= Qini-selected"
            if best_ece["alloc_value"] >= best_qini["alloc_value"]
            else "Qini-selected wins"
        )
        print(f"  -> realised proportional-allocation value: {winner}\n")
        summary.append(
            {
                "dataset": label,
                "best_qini_model": best_qini["model"],
                "qini_alloc_value": best_qini["alloc_value"],
                "best_ece_model": best_ece["model"],
                "ece_alloc_value": best_ece["alloc_value"],
                "ece_minus_qini_value": best_ece["alloc_value"] - best_qini["alloc_value"],
                "same_model": best_qini["model"] == best_ece["model"],
            }
        )
    sdf = pd.DataFrame(summary)
    sdf.to_csv(out / "tables" / "allocation_value_comparison.csv", index=False)
    print("Summary:\n" + sdf.round(4).to_string(index=False))
    print(f"\nSaved: {out/'tables'/'allocation_value_comparison.csv'}")


if __name__ == "__main__":
    main()
