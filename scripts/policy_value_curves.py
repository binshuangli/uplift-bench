"""Policy-value-vs-budget curves (clean-Claude, reviewer figure for F2).

The benchmark stores policy value only at a single budget (k=0.3), so a budget *curve*
must be recomputed from predictions. This script does a single leakage-safe train/test
split per dataset, fits a representative set of fast estimators, and sweeps the targeting
budget k, computing policy value at each k via the same metric the benchmark uses
(`policy_value_at_k`). It illustrates Finding 2: the model that maximises policy value
depends on the budget and need not be the Qini-ranking winner.

Reuses only first-party modules (data loaders, model registry, ranking metric).

Usage:
    python scripts/policy_value_curves.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit

from uplift_bench.data.registry import load_dataset
from uplift_bench.metrics.ranking import policy_value_at_k
from uplift_bench.models.registry import get_estimator

warnings.filterwarnings("ignore")

# Marketing RCTs (binary outcome, deployment-relevant) — the regime where budgeted
# targeting actually happens and ground-truth effects are unavailable.
DATASETS = [
    ("Hillstrom", "hillstrom", {}),
    ("Lenta", "lenta", {}),
    ("MegaFon", "megafon", {}),
    ("X5", "x5", {}),
]
MODELS = ["s_learner", "t_learner", "x_learner", "class_transformation", "two_model", "solo_model"]
MODEL_LABELS = {
    "s_learner": "S-Learner",
    "t_learner": "T-Learner",
    "x_learner": "X-Learner",
    "class_transformation": "ClassTrans",
    "two_model": "TwoModel",
    "solo_model": "SoloModel",
}
BENCH_N = 10_000
SEED = 42
K_GRID = np.round(np.arange(0.05, 1.0001, 0.05), 2)
PALETTE = sns.color_palette("tab10")

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7.5,
        "figure.dpi": 150,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def curves_for_dataset(loader_name: str, loader_kwargs: dict) -> pd.DataFrame:
    ds = load_dataset(loader_name, **loader_kwargs)
    if ds.meta.n > BENCH_N:
        ds = ds.subsample(BENCH_N, seed=SEED)

    X, t, y = ds.X, ds.treatment.to_numpy(), ds.outcome.to_numpy()
    # Leakage-safe split: stratify on treatment x outcome.
    strat = ds.treatment.astype(str) + "_" + ds.outcome.astype(int).astype(str)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=SEED)
    tr, te = next(sss.split(X, strat))
    Xtr, Xte = X.iloc[tr].copy(), X.iloc[te].copy()
    ttr, tte = ds.treatment.iloc[tr], ds.treatment.iloc[te]
    ytr, yte = ds.outcome.iloc[tr], ds.outcome.iloc[te]

    # Train-only median imputation (no leakage).
    if Xtr.isnull().any().any():
        imp = SimpleImputer(strategy="median")
        Xtr[:] = imp.fit_transform(Xtr)
        Xte[:] = imp.transform(Xte)

    prop = float(ttr.mean())  # RCT: known propensity = treatment fraction
    rows = []
    for m in MODELS:
        try:
            est = get_estimator(m, base_learner="lightgbm", seed=SEED)
            est.fit(Xtr, ttr, ytr)
            pred = est.predict_uplift(Xte)
            for k in K_GRID:
                pv = policy_value_at_k(
                    pred, tte.to_numpy(), yte.to_numpy(), k=float(k), propensity=prop
                )
                rows.append({"model": m, "k": k, "policy_value": pv})
        except Exception as exc:  # keep the figure robust to one model failing
            print(f"    [{loader_name}/{m}] skipped: {exc}")
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()
    out = Path(args.out_dir)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "tables").mkdir(parents=True, exist_ok=True)

    # 2x2 grid sized to render ~1:1 at full column width (was 1xN at 5in/panel,
    # which downscaled ~5.5x in the two-column layout and became illegible).
    ncol = 2
    nrow = (len(DATASETS) + ncol - 1) // ncol
    fig, axes_grid = plt.subplots(nrow, ncol, figsize=(7.0, 2.5 * nrow), sharex=True)
    axes = axes_grid.ravel()
    if len(DATASETS) == 1:
        axes = [axes]
    all_rows = []
    for ax, (label, loader, kw) in zip(axes, DATASETS):
        print(f"Computing budget curves: {label} ...")
        df = curves_for_dataset(loader, kw)
        df["dataset"] = label
        all_rows.append(df)
        for i, m in enumerate(MODELS):
            sub = df[df.model == m]
            if sub.empty:
                continue
            ax.plot(
                sub["k"],
                sub["policy_value"],
                marker="o",
                ms=2.5,
                lw=1.3,
                color=PALETTE[i % 10],
                label=MODEL_LABELS[m],
            )
        ax.set_title(label)
        ax.set_xlabel("Targeting budget $k$ (fraction treated)")
        ax.set_ylabel("Policy value")
        ax.grid(True, alpha=0.3)
    axes[0].legend(fontsize=7, loc="best", ncol=2)
    # No suptitle: the LaTeX caption carries this text, and a long single-line title
    # inflates the tight bounding box (and hence the downscale factor) in the paper.
    fig.tight_layout()
    fig.savefig(out / "figures" / "fig7_policy_value_budget.png", bbox_inches="tight", dpi=150)
    fig.savefig(out / "figures" / "fig7_policy_value_budget.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out/'figures'/'fig7_policy_value_budget.pdf'}")

    full = pd.concat(all_rows, ignore_index=True)
    full.to_csv(out / "tables" / "policy_value_budget_curves.csv", index=False)
    print(f"  Saved: {out/'tables'/'policy_value_budget_curves.csv'}")


if __name__ == "__main__":
    main()
