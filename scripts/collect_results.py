"""WP5 results collection and master-table builder.

Reads all parquet files in results/, merges them into master_raw.parquet
and master_summary.parquet, and prints a findings summary for RQ1-RQ4.

Usage:
    python scripts/collect_results.py
    python scripts/collect_results.py --results-dir results/
    python scripts/collect_results.py --print-rqs
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).parent.parent.resolve()

METRIC_NAMES = [
    "qini",
    "auuc",
    "uplift_at_k",
    "policy_value_at_k",
    "calibration_ece",
    "pehe",
    "epsilon_ate",
    "jobs_policy_risk",
]

RANKING_METRICS = ["qini", "auuc", "uplift_at_k", "policy_value_at_k"]
CAUSAL_METRICS = ["pehe", "epsilon_ate"]
CALIB_METRICS = ["calibration_ece"]
POLICY_METRICS = ["jobs_policy_risk"]

# Dataset regime classification
MARKETING_DATASETS = {"hillstrom", "lenta", "x5", "megafon", "criteo_1m", "criteo_5m", "criteo_full"}
SEMI_SYNTHETIC_DATASETS = {"ihdp", "jobs"}  # and ihdp_sN, jobs_sN patterns

# Model families
META_LEARNERS = {"s_learner", "t_learner", "x_learner", "r_learner", "dr_learner"}
FOREST_LEARNERS = {"causal_forest", "uplift_rf_kl", "uplift_rf_ed", "uplift_rf_chi"}
BASELINES = {"class_transformation", "two_model", "solo_model"}


def _dataset_regime(ds_name: str) -> str:
    if any(ds_name.startswith(p) for p in ("ihdp", "jobs")):
        return "semi_synthetic"
    if ds_name == "synthetic":
        return "synthetic"
    return "marketing_rct"


def load_all_raw(results_dir: Path) -> pd.DataFrame:
    """Load every *__*.parquet (excluding __summary__ files) into one DataFrame."""
    parquets = [
        p for p in results_dir.glob("*.parquet")
        if "__summary" not in p.name and p.name not in ("master_raw.parquet", "master_summary.parquet")
    ]
    if not parquets:
        print(f"No result parquets found in {results_dir}", file=sys.stderr)
        return pd.DataFrame()
    frames = [pd.read_parquet(p) for p in sorted(parquets)]
    df = pd.concat(frames, ignore_index=True)
    df["regime"] = df["dataset"].apply(_dataset_regime)
    return df


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    """Mean + 95% CI across ok folds for each (dataset, model, base_learner)."""
    group_cols = ["dataset", "model", "base_learner", "regime"]
    metric_cols = [m for m in METRIC_NAMES if m in raw.columns]
    ok = raw[raw["status"] == "ok"]
    records = []
    for keys, g in ok.groupby(group_cols):
        rec = dict(zip(group_cols, keys))
        rec["n_folds_ok"] = len(g)
        rec["total_runtime_sec"] = g["runtime_sec"].sum()
        for m in metric_cols:
            vals = g[m].dropna().to_numpy()
            if len(vals) == 0:
                rec[f"{m}_mean"] = np.nan
                rec[f"{m}_ci_low"] = np.nan
                rec[f"{m}_ci_high"] = np.nan
            else:
                mean = float(vals.mean())
                sem = float(vals.std(ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
                rec[f"{m}_mean"] = mean
                rec[f"{m}_ci_low"] = mean - 1.96 * sem
                rec[f"{m}_ci_high"] = mean + 1.96 * sem
            rec[f"{m}_n"] = int(len(vals))
        records.append(rec)
    return pd.DataFrame(records)


def leaderboard(summary: pd.DataFrame, metric: str = "qini") -> pd.DataFrame:
    """Per-dataset leaderboard ranked by a metric (descending)."""
    col = f"{metric}_mean"
    if col not in summary.columns:
        return pd.DataFrame()
    out = []
    for ds, g in summary.groupby("dataset"):
        ranked = g.sort_values(col, ascending=False).reset_index(drop=True)
        ranked["rank"] = ranked.index + 1
        out.append(ranked)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def rank_correlation_across_datasets(summary: pd.DataFrame, metric: str = "qini") -> pd.DataFrame:
    """Spearman rank-correlation matrix across datasets (RQ2: cross-domain stability)."""
    from scipy.stats import spearmanr

    col = f"{metric}_mean"
    pivot = summary.pivot_table(index="model", columns="dataset", values=col)
    pivot = pivot.dropna(axis=1, how="all").dropna(axis=0, how="all")
    datasets = pivot.columns.tolist()
    models = pivot.index.tolist()
    n = len(datasets)
    corr_mat = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            x = pivot.iloc[:, i].dropna()
            y = pivot.iloc[:, j].dropna()
            common = x.index.intersection(y.index)
            if len(common) >= 3:
                corr_mat[i, j] = spearmanr(x[common], y[common]).statistic
    return pd.DataFrame(corr_mat, index=datasets, columns=datasets)


def print_findings(raw: pd.DataFrame, summary: pd.DataFrame) -> None:
    """Print preliminary answers to RQ1–RQ4."""
    print("\n" + "=" * 70)
    print("WP5 FINDINGS SUMMARY")
    print("=" * 70)

    if raw.empty:
        print("No results yet.")
        return

    # Status overview
    status_counts = raw["status"].value_counts()
    print(f"\nRun status: {dict(status_counts)}")
    print(f"Total folds attempted: {len(raw)}")
    ok = raw[raw["status"] == "ok"]
    print(f"OK folds: {len(ok)}")
    total_runtime_h = raw["runtime_sec"].sum() / 3600
    print(f"Total runtime: {total_runtime_h:.1f} hours (wall clock per fold, sequential)")

    if summary.empty:
        print("Not enough OK results to summarize.")
        return

    print("\n" + "-" * 70)
    print("RQ1: Does naive Qini ranking agree with leakage-free ranking?")
    print("(Metric: Qini coefficient mean ± CI across folds)")
    if "qini_mean" in summary.columns:
        lb = leaderboard(summary, "qini")
        for ds, g in lb.groupby("dataset"):
            top3 = g.head(3)[["rank", "model", "qini_mean", "qini_ci_low", "qini_ci_high"]]
            print(f"\n  [{ds}] top-3 by Qini:")
            for _, row in top3.iterrows():
                print(
                    f"    #{int(row['rank'])} {row['model']:<25} "
                    f"{row['qini_mean']:.4f} [{row['qini_ci_low']:.4f}, {row['qini_ci_high']:.4f}]"
                )

    print("\n" + "-" * 70)
    print("RQ2: Are rankings stable across datasets (cross-domain)?")
    if "qini_mean" in summary.columns and summary["dataset"].nunique() >= 2:
        try:
            corr = rank_correlation_across_datasets(summary, "qini")
            if not corr.empty:
                off_diag = corr.values[np.triu_indices_from(corr.values, k=1)]
                off_diag = off_diag[~np.isnan(off_diag)]
                print(
                    f"  Mean pairwise Spearman rank-correlation across datasets: "
                    f"{np.mean(off_diag):.3f} (n={len(off_diag)} pairs)"
                )
                print("  (High value → stable rankings; low → regime matters)")
        except Exception as e:
            print(f"  Rank correlation not computed: {e}")
    else:
        print("  (Need ≥2 datasets with results)")

    print("\n" + "-" * 70)
    print("RQ3: Does the data regime (marketing vs. semi-synthetic) change which model wins?")
    if "qini_mean" in summary.columns:
        for regime, g in summary.groupby("regime"):
            if g["qini_mean"].notna().any():
                winner = g.loc[g["qini_mean"].idxmax(), "model"]
                best_q = g["qini_mean"].max()
                print(f"  [{regime}] best model by avg Qini: {winner} ({best_q:.4f})")

    print("\n" + "-" * 70)
    print("RQ4: Do models with best ranking metrics also have best calibration and policy value?")
    calib_col = "calibration_ece_mean"
    pv_col = "policy_value_at_k_mean"
    if calib_col in summary.columns and pv_col in summary.columns:
        for ds, g in summary.groupby("dataset"):
            if g["qini_mean"].notna().any() and g[calib_col].notna().any():
                best_rank = g.loc[g["qini_mean"].idxmax(), "model"]
                best_calib = g.loc[g[calib_col].idxmin(), "model"]  # lower ECE = better
                print(
                    f"  [{ds}] best Qini: {best_rank}, "
                    f"best calibration (low ECE): {best_calib} "
                    f"({'same' if best_rank == best_calib else 'different'})"
                )

    print("\n" + "=" * 70)

    # Flag any failures
    failed = raw[raw["status"] != "ok"]
    if not failed.empty:
        print(f"\nFAILED / SKIPPED CELLS ({len(failed)}):")
        for _, row in failed[["dataset", "model", "status", "error_msg"]].drop_duplicates().iterrows():
            msg = str(row["error_msg"])[:80] if row["error_msg"] else ""
            print(f"  {row['dataset']:<20} {row['model']:<25} {row['status']}  {msg}")

    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=str(REPO_ROOT / "results"))
    ap.add_argument("--print-rqs", action="store_true", default=True)
    ap.add_argument("--no-print-rqs", dest="print_rqs", action="store_false")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    raw = load_all_raw(results_dir)

    if raw.empty:
        print("No results found.")
        return

    # Write master files
    master_raw = results_dir / "master_raw.parquet"
    master_sum = results_dir / "master_summary.parquet"
    raw.to_parquet(master_raw, index=False)
    print(f"Wrote master raw ({len(raw)} rows) → {master_raw}")

    summary = summarize(raw)
    summary.to_parquet(master_sum, index=False)
    print(f"Wrote master summary ({len(summary)} rows) → {master_sum}")

    if args.print_rqs:
        print_findings(raw, summary)

    # Print master table (Qini mean per dataset × model)
    if "qini_mean" in summary.columns:
        pivot = summary.pivot_table(index="model", columns="dataset", values="qini_mean")
        print("\nQini coefficient (mean across folds) — master table:")
        print(pivot.to_string(float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
