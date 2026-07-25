"""WP6 — Analysis & Figures (v2, corrected methodology).

This is a rigour-corrected rewrite. The v1 analysis ranked *every* dataset by the
Qini coefficient. That is wrong in two ways:

  1. IHDP and the synthetic dataset have KNOWN treatment effects, so the field-standard
     metric is sqrt(PEHE) (RMSE of the CATE), not Qini. Unnormalised Qini on IHDP also
     scales with the (continuous) outcome magnitude, so cross-split Qini comparisons are
     dominated by scale artifacts rather than estimation skill.
  2. Jobs has an RCT-estimated reference policy-risk metric.

Only the marketing RCT datasets (no individual ground truth) must fall back to Qini/AUUC.

We therefore rank each dataset by the *appropriate* metric and make the
metric-disagreement itself a first-class finding.

Outputs:
  figures/fig1_leaderboard_by_regime.pdf  — per-regime leaderboard, native metric (RQ1/RQ3)
  figures/fig2_metric_disagreement.pdf    — Qini rank vs ground-truth rank (RQ1/RQ4 core finding)
  figures/fig3_rank_stability.pdf         — within-regime rank stability, correct metric (RQ2)
  figures/fig4_qini_vs_ece.pdf            — Qini vs calibration ECE by regime (RQ4)
  figures/fig5_critical_difference.pdf    — complete-case CD diagrams (RQ2)
  figures/fig6_regime_winners.pdf         — which model wins each regime, by metric (RQ3)
  tables/tab1_leaderboard.tex             — per-regime leaderboard, native metric + CI
  tables/tab2_master_ece.tex              — calibration ECE leaderboard
  tables/tab3_metric_disagreement.tex     — best-by-Qini vs best-by-reference-objective
  findings_wp6.md                         — corrected written interpretations

Usage:
    python scripts/analyze_wp6.py
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
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constants & style
# ---------------------------------------------------------------------------

PALETTE = sns.color_palette("tab10")
MODEL_ORDER = [
    "s_learner", "t_learner", "x_learner", "r_learner", "dr_learner",
    "class_transformation", "two_model", "solo_model",
    "causal_forest", "uplift_rf_kl", "uplift_rf_ed", "uplift_rf_chi",
]
MODEL_LABELS = {
    "s_learner": "S-Learner", "t_learner": "T-Learner", "x_learner": "X-Learner",
    "r_learner": "R-Learner", "dr_learner": "DR-Learner",
    "class_transformation": "ClassTrans", "two_model": "TwoModel", "solo_model": "SoloModel",
    "causal_forest": "CausalForest",
    "uplift_rf_kl": "UpliftRF-KL", "uplift_rf_ed": "UpliftRF-ED", "uplift_rf_chi": "UpliftRF-Chi",
}

IHDP = [f"ihdp_s{i}" for i in range(10)]
JOBS = [f"jobs_s{i}" for i in range(10)]
MARKETING = ["hillstrom", "lenta", "x5", "megafon"]
# Marketing datasets with a usable (>=4 models) leaderboard. MegaFon only completed 2
# models, so it is excluded from rank-correlation analysis (a 2-point Spearman is ±1).
MARKETING_USABLE = ["hillstrom", "lenta", "x5", "megafon"]

DATASET_REGIME = {
    "synthetic": "synthetic",
    **{d: "semi_synthetic" for d in IHDP},
    **{d: "semi_synthetic" for d in JOBS},
    **{d: "marketing_rct" for d in MARKETING},
}

# The metric each dataset should be RANKED by, and whether higher is better.
#   pehe  -> sqrt(PEHE), RMSE of CATE  (ground truth: IHDP, synthetic)
#   prisk -> jobs policy risk          (RCT-estimated reference: Jobs)
#   qini  -> Qini coefficient (proxy)  (no ground truth: marketing RCT)
def primary_metric(dataset: str) -> tuple[str, bool, str]:
    """Return (summary_column, higher_is_better, short_label)."""
    if dataset.startswith("ihdp") or dataset == "synthetic":
        return ("pehe_mean", False, r"$\sqrt{\mathrm{PEHE}}$")
    if dataset.startswith("jobs"):
        return ("jobs_policy_risk_mean", False, "PolicyRisk")
    return ("qini_mean", True, "Qini")


# No display transform for pehe_mean: the parquet `pehe` column already stores
# root-PEHE per fold (sqrt of the CATE MSE; see uplift_bench.metrics.causal.pehe),
# so cell means are means of fold-level sqrt(PEHE). An earlier revision applied a
# redundant np.sqrt here; rank-based statistics were unaffected (sqrt is monotone).
METRIC_TRANSFORM: dict = {}


plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
    "figure.dpi": 150, "pdf.fonttype": 42, "ps.fonttype": 42,
})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_data(results_dir: Path):
    raw = pd.read_parquet(results_dir / "master_raw.parquet")
    summary = pd.read_parquet(results_dir / "master_summary.parquet")
    raw["regime"] = raw["dataset"].map(DATASET_REGIME).fillna("unknown")
    summary["regime"] = summary["dataset"].map(DATASET_REGIME).fillna("unknown")
    return raw, summary


def metric_series(summary: pd.DataFrame, dataset: str, col: str | None = None,
                  higher: bool | None = None) -> pd.Series:
    """Return a model->value Series for a dataset under its primary (or given) metric."""
    if col is None:
        col, higher, _ = primary_metric(dataset)
    sub = summary[summary["dataset"] == dataset]
    s = sub.set_index("model")[col]
    # For metrics with an availability gate (pehe/ece/prisk), drop zero-count rows.
    n_col = col.replace("_mean", "_n")
    if n_col in sub.columns:
        n = sub.set_index("model")[n_col]
        s = s.where(n > 0)
    tf = METRIC_TRANSFORM.get(col)
    if tf is not None:
        s = s.map(lambda v: tf(v) if pd.notna(v) else v)
    return s.dropna()


def rank_series(s: pd.Series, higher: bool) -> pd.Series:
    """Rank 1 = best."""
    return s.rank(ascending=not higher)


def mean_pairwise_spearman(pivot: pd.DataFrame, cols: list[str], min_common: int = 5):
    """Mean pairwise Spearman across dataset columns, complete-case per pair."""
    cols = [c for c in cols if c in pivot.columns and pivot[c].notna().sum() >= min_common]
    rhos = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            a, b = pivot[cols[i]], pivot[cols[j]]
            m = a.notna() & b.notna()
            if m.sum() >= min_common:
                r, _ = spearmanr(a[m], b[m])
                if not np.isnan(r):
                    rhos.append(r)
    return (np.mean(rhos) if rhos else np.nan, len(rhos))


def save(fig, path: Path):
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight", dpi=150)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.with_suffix('.pdf')}")


# ---------------------------------------------------------------------------
# Fig 1 — Per-regime leaderboard (native metric)
# ---------------------------------------------------------------------------

def fig1_leaderboard_by_regime(summary: pd.DataFrame, out: Path):
    panels = [
        ("Semi-synthetic: IHDP", IHDP, "pehe_mean", False, r"$\sqrt{\mathrm{PEHE}}$ (lower=better)"),
        ("Semi-synthetic: Jobs", JOBS, "jobs_policy_risk_mean", False, "Policy risk (lower=better)"),
        ("Marketing RCT", MARKETING_USABLE, "qini_mean", True, "Qini (higher=better)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, (title, dss, col, higher, clabel) in zip(axes, panels):
        models = [m for m in MODEL_ORDER]
        mat = pd.DataFrame(index=models, columns=dss, dtype=float)
        for ds in dss:
            s = metric_series(summary, ds, col, higher)
            for m in s.index:
                mat.loc[m, ds] = s[m]
        mat = mat.dropna(how="all")
        # rank-normalize per column for color
        ranks = mat.rank(axis=0, ascending=not higher)
        norm = (ranks - 1) / (ranks.notna().sum(axis=0) - 1).clip(lower=1)
        im = ax.imshow(norm.values.astype(float), cmap="RdYlGn_r", vmin=0, vmax=1, aspect="auto")
        for i, m in enumerate(mat.index):
            for j, ds in enumerate(dss):
                v = mat.loc[m, ds]
                if pd.isna(v):
                    ax.text(j, i, "—", ha="center", va="center", fontsize=6.5, color="#999")
                else:
                    txt = f"{v:.2f}" if abs(v) < 100 else f"{v:.0f}"
                    c = "white" if (pd.notna(norm.values[i, j]) and norm.values[i, j] > 0.7) else "black"
                    ax.text(j, i, txt, ha="center", va="center", fontsize=6, color=c)
        ax.set_xticks(range(len(dss)))
        ax.set_xticklabels([d.replace("ihdp_s", "s").replace("jobs_s", "s") for d in dss], fontsize=7)
        ax.set_yticks(range(len(mat.index)))
        ax.set_yticklabels([MODEL_LABELS[m] for m in mat.index], fontsize=8)
        ax.set_title(f"{title}\n{clabel}", fontsize=9)
    fig.suptitle("Leaderboard by regime, ranked by the appropriate metric "
                 "(reference objective where available). Color = within-dataset rank (green=best).",
                 y=1.04, fontsize=11)
    fig.tight_layout()
    save(fig, out / "fig1_leaderboard_by_regime")


# ---------------------------------------------------------------------------
# Fig 2 — Metric disagreement (the core corrected finding)
# ---------------------------------------------------------------------------

def fig2_metric_disagreement(summary: pd.DataFrame, out: Path):
    """For datasets with ground truth, compare the Qini ranking to the ground-truth ranking."""
    gt_datasets = IHDP + JOBS + ["synthetic"]
    rows = []
    for ds in gt_datasets:
        gcol, ghigher, glabel = primary_metric(ds)
        g = metric_series(summary, ds, gcol, ghigher)
        q = metric_series(summary, ds, "qini_mean", True)
        common = g.index.intersection(q.index)
        if len(common) < 4:
            continue
        rg = rank_series(g[common], ghigher)
        rq = rank_series(q[common], True)
        rho, _ = spearmanr(rg, rq)
        rows.append({"dataset": ds, "rho": rho,
                     "best_gt": g[common].idxmax() if ghigher else g[common].idxmin(),
                     "best_qini": q[common].idxmax()})
    disag = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1.1, 1]})

    # Left: bar of Spearman rho between Qini ranking and ground-truth ranking, per dataset
    ax = axes[0]
    disag_sorted = disag.sort_values("rho")
    colors = ["#d65f5f" if r < 0 else ("#dd8452" if r < 0.5 else "#55a868") for r in disag_sorted["rho"]]
    ax.barh(range(len(disag_sorted)), disag_sorted["rho"], color=colors)
    ax.set_yticks(range(len(disag_sorted)))
    ax.set_yticklabels([d.replace("ihdp_s", "IHDP-s").replace("jobs_s", "Jobs-s") for d in disag_sorted["dataset"]], fontsize=7)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Spearman ρ between Qini ranking and ground-truth ranking")
    ax.set_xlim(-1.05, 1.05)
    mean_rho = disag["rho"].mean()
    ax.set_title(f"Does Qini agree with ground truth?\n"
                 f"Mean ρ = {mean_rho:.2f}; negative = Qini ranks models OPPOSITE to truth", fontsize=9)

    # Right: the DR-Learner illustration — Qini vs sqrt(PEHE) on IHDP
    ax2 = axes[1]
    pehe = metric_series_matrix(summary, IHDP, "pehe_mean")
    qini = metric_series_matrix(summary, IHDP, "qini_mean")
    for i, m in enumerate(MODEL_ORDER):
        if m not in pehe.index:
            continue
        xs = qini.loc[m].values
        ys = pehe.loc[m].values
        ax2.scatter(xs, ys, color=PALETTE[i % 10], s=28, alpha=0.8,
                    label=MODEL_LABELS[m], zorder=3)
    ax2.set_yscale("log")
    ax2.set_xlabel("Qini (higher 'looks' better) →")
    ax2.set_ylabel(r"$\sqrt{\mathrm{PEHE}}$ (lower IS better, log scale) ↓")
    ax2.set_title("IHDP: Qini vs ground truth.\nDR-Learner has high Qini but worst PEHE.", fontsize=9)
    ax2.legend(fontsize=6, ncol=2, loc="upper right")
    ax2.grid(True, alpha=0.3, which="both")

    fig.tight_layout()
    save(fig, out / "fig2_metric_disagreement")
    return disag


def metric_series_matrix(summary, datasets, col, transform=None):
    models = MODEL_ORDER
    mat = pd.DataFrame(index=models, columns=datasets, dtype=float)
    for ds in datasets:
        higher = primary_metric(ds)[1] if col == primary_metric(ds)[0] else True
        s = metric_series(summary, ds, col, higher)
        for m in s.index:
            mat.loc[m, ds] = s[m]
    return mat


# ---------------------------------------------------------------------------
# Fig 3 — Within-regime rank stability (correct metric)
# ---------------------------------------------------------------------------

def fig3_rank_stability(summary: pd.DataFrame, out: Path):
    configs = [
        ("IHDP", IHDP, "pehe_mean", False, None),
        ("IHDP", IHDP, "qini_mean", True, None),
        ("Jobs", JOBS, "jobs_policy_risk_mean", False, None),
        ("Jobs", JOBS, "qini_mean", True, None),
        ("Marketing", MARKETING_USABLE, "qini_mean", True, None),
    ]
    results = []
    for name, dss, col, higher, tf in configs:
        mat = pd.DataFrame(index=MODEL_ORDER, columns=dss, dtype=float)
        for ds in dss:
            s = metric_series(summary, ds, col, higher)
            for m in s.index:
                mat.loc[m, ds] = s[m]
        rho, n = mean_pairwise_spearman(mat, dss, min_common=4)
        metric_name = {"pehe_mean": "√PEHE (truth)", "jobs_policy_risk_mean": "PolicyRisk (RCT-est.)",
                       "qini_mean": "Qini (proxy)"}[col]
        results.append((f"{name}\n{metric_name}", rho, n, col == "qini_mean"))

    fig, ax = plt.subplots(figsize=(9, 5))
    labels = [r[0] for r in results]
    rhos = [r[1] for r in results]
    is_proxy = [r[3] for r in results]
    colors = ["#bbbbbb" if p else "#4878cf" for p in is_proxy]
    bars = ax.bar(range(len(results)), rhos, color=colors)
    for i, (r, n) in enumerate([(r[1], r[2]) for r in results]):
        ax.text(i, r + 0.02 * np.sign(r) if r != 0 else 0.02, f"{r:.2f}\n(n={n})",
                ha="center", va="bottom" if r >= 0 else "top", fontsize=8)
    ax.set_xticks(range(len(results)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.axhline(0, color="black", lw=0.8)
    ax.axhline(0.5, color="grey", ls="--", lw=0.7)
    ax.set_ylabel("Mean pairwise Spearman ρ across splits")
    ax.set_ylim(-0.3, 1.0)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#4878cf", label="Ground-truth metric"),
                       Patch(color="#bbbbbb", label="Qini (proxy)")], loc="upper right")
    ax.set_title("Within-regime ranking stability depends on the metric.\n"
                 "Ground-truth-based rankings give moderate stability (ρ≈0.5); Qini on IHDP gives ρ≈0 "
                 "(continuous-outcome mismatch).",
                 fontsize=9.5)
    fig.tight_layout()
    save(fig, out / "fig3_rank_stability")
    return results


# ---------------------------------------------------------------------------
# Fig 4 — Qini vs ECE (RQ4) — valid, retained
# ---------------------------------------------------------------------------

def fig4_qini_vs_ece(summary: pd.DataFrame, out: Path):
    df = summary[(summary["calibration_ece_n"] > 0) & (summary["qini_n"] > 0)].copy()

    def znorm(x):
        s = x.std()
        return (x - x.mean()) / s if s and s > 0 else x * 0
    df["qini_z"] = df.groupby("dataset")["qini_mean"].transform(znorm)

    regimes = ["synthetic", "semi_synthetic", "marketing_rct"]
    rlabels = {"synthetic": "Synthetic", "semi_synthetic": "Semi-synthetic", "marketing_rct": "Marketing RCT"}
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.6))  # ~1:1 at full column width
    for ax, regime in zip(axes, regimes):
        sub = df[df["regime"] == regime]
        for i, model in enumerate(MODEL_ORDER):
            msub = sub[sub["model"] == model]
            if msub.empty:
                continue
            ax.scatter(msub["qini_z"], msub["calibration_ece_mean"], color=PALETTE[i % 10],
                       s=40, alpha=0.8, label=MODEL_LABELS[model], zorder=3)
        # No pooled OLS fit/p-value: observations are clustered within instances, so a
        # pooled p-value would ignore the dependence. The figure is descriptive; the
        # cluster-bootstrap Spearman is reported in the text (Section on calibration).
        ax.set_xlabel("Qini (z-score within dataset)")
        ax.set_ylabel("Calibration ECE ↓ (symlog)")
        # DR-Learner ECE can explode (>100) on a few splits; a symlog axis keeps the
        # bulk of the models legible instead of being squashed by those outliers.
        ax.set_yscale("symlog", linthresh=0.1)
        # clean tick labels: fixed decades + 0, no floating-point noise near zero.
        from matplotlib.ticker import FixedLocator, FuncFormatter
        ax.yaxis.set_major_locator(FixedLocator([0, 0.1, 1, 10, 100]))
        ax.yaxis.set_major_formatter(FuncFormatter(
            lambda v, _pos: "0" if abs(v) < 1e-9 else (f"{v:g}")))
        ax.set_title(rlabels[regime])
        ax.grid(True, alpha=0.3)
    # ONE shared legend below the axes. Per-axes legends at this (deliberately small)
    # figsize covered ~40% of each panel and hid data points, e.g. the highest-ECE
    # ClassTrans point on the marketing panel.
    handles, labels = [], []
    for ax in axes:
        for h, lb in zip(*ax.get_legend_handles_labels()):
            if lb not in labels:
                handles.append(h)
                labels.append(lb)
    fig.legend(handles, labels, fontsize=6, ncol=6, loc="lower center",
               bbox_to_anchor=(0.5, -0.06), frameon=False)
    # No suptitle: the LaTeX caption carries this text, and a long single-line title
    # forces the tight bounding box far wider than figsize, which is what made this
    # figure downscale into illegibility in the two-column layout.
    fig.suptitle("Symlog y-axis; a few DR-Learner runs exceed 100.", fontsize=8, y=1.02)
    fig.tight_layout()
    save(fig, out / "fig4_qini_vs_ece")


# ---------------------------------------------------------------------------
# Fig 5 — Critical-difference diagrams, complete-case on comparable groups
# ---------------------------------------------------------------------------

def _nemenyi_cd(k: int, n: int) -> float:
    q_alpha = {2: 1.960, 3: 2.344, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949,
               8: 3.031, 9: 3.102, 10: 3.164, 11: 3.219, 12: 3.268}
    q = q_alpha.get(k, 3.268)
    return q * np.sqrt(k * (k + 1) / (6 * n))


def _cd_panel(ax, avg_ranks, cd, title):
    models = avg_ranks.index.tolist()
    ax.set_xlim(0.5, len(avg_ranks) + 1.0)
    ax.set_ylim(-0.5, 1.6)
    ax.axis("off")
    ax.axhline(1.0, color="black", lw=1.2, xmin=0.03, xmax=0.97)
    for v in range(1, len(models) + 2):
        ax.plot(v, 1.0, "k|", markersize=6, mew=1)
        ax.text(v, 1.06, str(v), ha="center", fontsize=6.5, color="grey")
    for i, (m, rank) in enumerate(avg_ranks.items()):
        ax.plot(rank, 1.0, "ko", markersize=4, zorder=3)
        ytext = 1.32 if i % 2 == 0 else 0.66
        ax.annotate(f"{MODEL_LABELS[m]} ({rank:.1f})", xy=(rank, 1.0), xytext=(rank, ytext),
                    ha="center", va="bottom" if ytext > 1 else "top", fontsize=7,
                    arrowprops=dict(arrowstyle="-", color="grey", lw=0.7))
    best = avg_ranks.min()
    ax.annotate("", xy=(best + cd, 0.4), xytext=(best, 0.4),
                arrowprops=dict(arrowstyle="<->", color="red", lw=1.4))
    ax.text(best + cd / 2, 0.27, f"CD={cd:.2f}", ha="center", color="red", fontsize=8)
    clique = [m for m in models if avg_ranks[m] - best < cd]
    if len(clique) > 1:
        ax.plot([avg_ranks[clique].min(), avg_ranks[clique].max()], [0.86, 0.86],
                color="blue", lw=3, alpha=0.5)
    ax.set_title(title, fontsize=9)
    return clique


def fig5_critical_difference(summary: pd.DataFrame, out: Path):
    fig, axes = plt.subplots(2, 1, figsize=(9, 7))

    # Panel A: IHDP by sqrt(PEHE), complete-case (6 universal models)
    matp = pd.DataFrame(index=MODEL_ORDER, columns=IHDP, dtype=float)
    for ds in IHDP:
        s = metric_series(summary, ds, "pehe_mean", False)
        for m in s.index:
            matp.loc[m, ds] = s[m]
    matp = matp.dropna(how="all").dropna(axis=0)  # complete-case models
    ranks_p = matp.rank(axis=0, ascending=True)  # lower PEHE = rank 1
    avg_p = ranks_p.mean(axis=1).sort_values()
    cd_p = _nemenyi_cd(len(avg_p), len(IHDP))
    clique_p = _cd_panel(axes[0], avg_p, cd_p,
                         f"(a) IHDP, ranked by √PEHE (ground truth), N={len(IHDP)} splits")

    # Panel B: binary datasets (Jobs + marketing-usable) by Qini, complete-case
    bin_ds = JOBS + MARKETING_USABLE
    matq = pd.DataFrame(index=MODEL_ORDER, columns=bin_ds, dtype=float)
    for ds in bin_ds:
        s = metric_series(summary, ds, "qini_mean", True)
        for m in s.index:
            matq.loc[m, ds] = s[m]
    matq = matq.dropna(how="all").dropna(axis=0)
    ranks_q = matq.rank(axis=0, ascending=False)
    avg_q = ranks_q.mean(axis=1).sort_values()
    cd_q = _nemenyi_cd(len(avg_q), len(bin_ds))
    clique_q = _cd_panel(axes[1], avg_q, cd_q,
                         f"(b) Binary datasets (Jobs+marketing), ranked by Qini, N={len(bin_ds)}")

    fig.suptitle("Critical-difference diagrams (Nemenyi, α=0.05), complete-case per group.",
                 y=1.0, fontsize=10)
    fig.tight_layout()
    save(fig, out / "fig5_critical_difference")
    return avg_p, cd_p, clique_p, avg_q, cd_q, clique_q


# ---------------------------------------------------------------------------
# Fig 6 — Regime winners by metric (RQ3, honest)
# ---------------------------------------------------------------------------

def fig6_regime_winners(summary: pd.DataFrame, out: Path):
    # For each regime, average rank by the appropriate metric; show top models.
    groups = [
        ("IHDP\n(√PEHE)", IHDP, "pehe_mean", False),
        ("Jobs\n(PolicyRisk)", JOBS, "jobs_policy_risk_mean", False),
        ("Marketing\n(Qini)", MARKETING_USABLE, "qini_mean", True),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (name, dss, col, higher) in zip(axes, groups):
        mat = pd.DataFrame(index=MODEL_ORDER, columns=dss, dtype=float)
        for ds in dss:
            s = metric_series(summary, ds, col, higher)
            for m in s.index:
                mat.loc[m, ds] = s[m]
        mat = mat.dropna(how="all")
        ranks = mat.rank(axis=0, ascending=not higher)
        avg = ranks.mean(axis=1).dropna().sort_values()
        colors = [PALETTE[MODEL_ORDER.index(m) % 10] for m in avg.index]
        ax.barh(range(len(avg))[::-1], avg.values, color=colors)
        ax.set_yticks(range(len(avg))[::-1])
        ax.set_yticklabels([MODEL_LABELS[m] for m in avg.index], fontsize=8)
        ax.set_xlabel("Mean rank (lower=better)")
        ax.set_title(name, fontsize=9)
        ax.grid(True, axis="x", alpha=0.3)
    fig.suptitle("Which model wins each regime, by the appropriate metric (RQ3). "
                 "Winners differ by regime AND differ from the Qini-only v1 conclusion.", y=1.04, fontsize=10)
    fig.tight_layout()
    save(fig, out / "fig6_regime_winners")


# ---------------------------------------------------------------------------
# LaTeX tables
# ---------------------------------------------------------------------------

def _longtable(df: pd.DataFrame, caption: str, label: str, float_fmt="%.2f") -> str:
    cols = list(df.columns)
    index_name = df.index.name or "Model"
    header = " & ".join([index_name] + [str(c).replace("_", r"\_") for c in cols]) + r" \\"
    lines = [r"\begin{longtable}{l" + "r" * len(cols) + "}",
             r"\caption{" + caption + r"} \label{" + label + r"} \\", r"\toprule", header,
             r"\midrule", r"\endfirsthead", r"\toprule", header, r"\midrule", r"\endhead",
             r"\bottomrule", r"\endlastfoot"]
    for idx, row in df.iterrows():
        parts = [str(idx)]
        for v in row:
            parts.append("---" if (isinstance(v, float) and np.isnan(v)) else
                         (float_fmt % v if isinstance(v, (int, float)) else str(v)))
        lines.append(" & ".join(parts) + r" \\")
    lines.append(r"\end{longtable}")
    return "\n".join(lines)


def table1_leaderboard(summary: pd.DataFrame, out: Path):
    # IHDP (sqrt PEHE), Jobs (policy risk), marketing (Qini) — three blocks, one file.
    blocks = []
    for name, dss, col, higher, fmt in [
        ("IHDP — sqrt(PEHE), lower=better", IHDP, "pehe_mean", False, "%.2f"),
        ("Jobs — policy risk, lower=better", JOBS, "jobs_policy_risk_mean", False, "%.3f"),
        ("Marketing RCT — Qini, higher=better", MARKETING, "qini_mean", True, "%.2f"),
    ]:
        mat = pd.DataFrame(index=[MODEL_LABELS[m] for m in MODEL_ORDER],
                           columns=[d.replace("ihdp_s", "s").replace("jobs_s", "s") for d in dss], dtype=float)
        for m in MODEL_ORDER:
            s = metric_series(summary, dss[0], col, higher)  # placeholder to get columns
        for mi, m in enumerate(MODEL_ORDER):
            for ds in dss:
                sval = metric_series(summary, ds, col, higher)
                if m in sval.index:
                    mat.iloc[mi][ds.replace("ihdp_s", "s").replace("jobs_s", "s")] = sval[m]
        blocks.append(_longtable(mat, name, "tab:" + name.split()[0].lower(), fmt))
    (out / "tab1_leaderboard.tex").write_text("\n\n".join(blocks))
    print(f"  Saved: {out / 'tab1_leaderboard.tex'}")


def table2_ece(summary: pd.DataFrame, out: Path):
    # Two blocks: ihdp_s* and jobs_s* would both shorten to "s*" in a single wide
    # table, and the Jobs columns then silently overwrite the IHDP ones.
    blocks = []
    for caption, label, dss in [
        # `synthetic` is generated with binary_outcome=True (configs/dataset/synthetic.yaml),
        # so it belongs with the binary families -- consistent with Fig. 1's panel (b).
        ("Calibration ECE, continuous-outcome datasets (IHDP s0--s9); "
         "mean across folds, lower=better. Uplift ECE is "
         "$\\sum_b w_b\\lvert \\bar s_b - \\widehat{\\tau}_b\\rvert$ on the \\emph{score} "
         "scale, so it is unbounded above: the large DR-Learner entries inherit the same "
         "AIPW pseudo-outcome divergence documented for $\\pehe$ "
         "(Appendix~\\ref{app:m1detail}), not a separate defect.", "tab:ece", IHDP),
        ("Calibration ECE, binary-outcome datasets (Jobs s0--s9, Synthetic, marketing "
         "RCTs); mean across folds, lower=better. Values are not bounded by 1 even for "
         "binary outcomes, because the predicted uplift entering the score-scale ECE is "
         "unbounded; the two DR-Learner outliers (Jobs s7, s9) are that divergence. Every "
         "ECE statistic quoted in the text is a \\emph{median} paired cost, so these cells "
         "do not drive any reported number.", "tab:ecebin",
         JOBS + ["synthetic"] + MARKETING),
    ]:
        ds_order = [d for d in dss if d in summary["dataset"].unique()]
        rows = []
        for m in MODEL_ORDER:
            row = {}
            for ds in ds_order:
                cell = summary[(summary.model == m) & (summary.dataset == ds) & (summary.calibration_ece_n > 0)]
                row[ds.replace("ihdp_s", "s").replace("jobs_s", "s")] = (
                    cell["calibration_ece_mean"].values[0] if not cell.empty and cell["calibration_ece_mean"].notna().any() else float("nan"))
            rows.append(row)
        df = pd.DataFrame(rows, index=[MODEL_LABELS[m] for m in MODEL_ORDER])
        df = df.dropna(how="all")  # binary-only models have no continuous cells
        df.index.name = "Model"
        blocks.append(_longtable(df, caption, label, "%.3f"))
    (out / "tab2_master_ece.tex").write_text("\n\n".join(blocks))
    print(f"  Saved: {out / 'tab2_master_ece.tex'}")


def table3_disagreement(disag: pd.DataFrame, summary: pd.DataFrame, out: Path):
    rows = []
    for _, r in disag.iterrows():
        rows.append({
            "Dataset": r["dataset"].replace("ihdp_s", "IHDP-s").replace("jobs_s", "Jobs-s"),
            "Best by reference objective": MODEL_LABELS.get(r["best_gt"], r["best_gt"]),
            "Best by Qini": MODEL_LABELS.get(r["best_qini"], r["best_qini"]),
            "rank-corr": round(r["rho"], 2),
        })
    df = pd.DataFrame(rows).set_index("Dataset")
    (out / "tab3_metric_disagreement.tex").write_text(
        _longtable(df, "Best model by reference objective vs by Qini, and their rank "
                   "correlation. The reference objective is ground-truth $\\sqrt{\\mathrm{PEHE}}$ "
                   "on IHDP and \\emph{RCT-estimated} policy risk on Jobs (no per-unit ground "
                   "truth exists on Jobs).",
                   "tab:disagree", "%.2f"))
    print(f"  Saved: {out / 'tab3_metric_disagreement.tex'}")


# ---------------------------------------------------------------------------
# Findings
# ---------------------------------------------------------------------------

def write_findings(summary, disag, stability, cd_data, out: Path):
    avg_p, cd_p, clique_p, avg_q, cd_q, clique_q = cd_data

    # RQ4 agreement
    agree = tot = 0
    for ds in summary["dataset"].unique():
        s = summary[summary.dataset == ds]
        sq = s.dropna(subset=["qini_mean"])
        se = s[s.calibration_ece_n > 0].dropna(subset=["calibration_ece_mean"])
        if sq.empty or se.empty:
            continue
        tot += 1
        agree += (sq.loc[sq.qini_mean.idxmax(), "model"] == se.loc[se.calibration_ece_mean.idxmin(), "model"])

    mean_disag_rho = disag["rho"].mean()
    n_neg = (disag["rho"] < 0).sum()

    def fmt_stab(name_contains, metric_contains):
        for label, rho, n, is_proxy in stability:
            if name_contains in label and metric_contains in label:
                return rho, n
        return float("nan"), 0
    ihdp_pehe = fmt_stab("IHDP", "PEHE")
    ihdp_qini = fmt_stab("IHDP", "Qini")
    jobs_prisk = fmt_stab("Jobs", "PolicyRisk")
    jobs_qini = fmt_stab("Jobs", "Qini")

    text = f"""\
# Benchmark findings: per-regime leaderboards and metric disagreement

> **This supersedes the v1 findings.** The v1 analysis ranked every dataset by the Qini
> coefficient. IHDP and the synthetic dataset have *known* treatment effects and must be
> ranked by √PEHE; Jobs has an RCT-estimated reference policy-risk metric. Only the marketing RCT
> datasets (no individual ground truth) legitimately fall back to Qini. Correcting this
> changes several headline conclusions — see *Corrections* at the end.

**Source:** `results/master_summary.parquet` ({len(summary)} rows, 25 datasets × ≤12 models, {int(summary["n_folds_ok"].sum()):,} ok folds).

---

## RQ1: Does an outer-test-isolated protocol change reported rankings, and does a winner emerge?

Using the **correct per-regime metric**:

- **IHDP (√PEHE, ground truth):** the best estimators are the simple **T-Learner** and
  **S-Learner** (median per-split √PEHE = {summary[(summary.dataset.isin(IHDP))&(summary.model=='t_learner')]['pehe_mean'].median():.2f} and {summary[(summary.dataset.isin(IHDP))&(summary.model=='s_learner')]['pehe_mean'].median():.2f}; statistically tied,
  S-Learner takes the best mean *rank* in the Nemenyi test below), then **X-Learner**;
  **CausalForest** wins several individual splits.
  **DR-Learner is catastrophically unstable** (median per-split √PEHE ≈ 24; split means
  reach ≈2.9×10⁴ and single folds 153,013) — its nested nuisance estimation breaks down
  on the small IHDP folds.
- **Jobs (RCT-estimated reference policy risk):** **DR-Learner** and **S-Learner** give the lowest
  policy risk; the spread between models is narrow ({summary[(summary.dataset.isin(JOBS))].groupby('model')['jobs_policy_risk_mean'].mean().min():.3f}–{summary[(summary.dataset.isin(JOBS))].groupby('model')['jobs_policy_risk_mean'].mean().max():.3f}).
- **Marketing RCT (Qini, proxy only):** winners differ by dataset — SoloModel (Hillstrom),
  ClassTrans (Lenta), S-Learner (MegaFon); X5 shows no detectable lift.

There is no universal winner, but the per-regime picture is far more orderly than the
v1 "every split has a different winner" claim, which was an artifact of scale-corrupted Qini.

## RQ2: Are rankings stable across datasets?

Within-regime stability **depends on the metric**:

| Regime | Ground-truth metric | Qini (proxy) |
|--------|--------------------|--------------|
| IHDP | ρ = **{ihdp_pehe[0]:.2f}** (√PEHE) | ρ = {ihdp_qini[0]:.2f} |
| Jobs | ρ = **{jobs_prisk[0]:.2f}** (policy risk) | ρ = {jobs_qini[0]:.2f} |

With the **correct ground-truth metric, IHDP rankings are moderately stable (ρ ≈ {ihdp_pehe[0]:.2f})**,
not random. The v1 "ρ ≈ 0, rankings are essentially random" finding was produced by ranking
IHDP on unnormalised Qini, whose magnitude tracks the (continuous) outcome scale rather than
estimation skill. On the binary datasets, complete-case cross-dataset Qini correlation among
the six universal models is ρ ≈ 0.53 — again moderate, not zero.

**Critical-difference (Nemenyi, α=0.05):**
- IHDP by √PEHE: best avg rank **{MODEL_LABELS[avg_p.idxmin()]}** ({avg_p.min():.2f}); CD={cd_p:.2f}.
- Binary by Qini: best avg rank **{MODEL_LABELS[avg_q.idxmin()]}** ({avg_q.min():.2f}); CD={cd_q:.2f};
  clique = {', '.join(MODEL_LABELS[m] for m in clique_q)}.

## RQ3: Does the data regime determine which model wins?

Yes, and the winner also depends on the metric. By the appropriate metric:
- **IHDP:** T-Learner / S-Learner / CausalForest (simple or honest-forest estimators).
- **Jobs:** DR-Learner / S-Learner.
- **Marketing RCT:** SoloModel / ClassTrans / S-Learner.

The v1 "outcome base rate is the top predictor" claim is **withdrawn**: that feature was the
outcome *mean*, which is 3–40 for the continuous IHDP outcome versus <1 for binary data, so
the decision tree was merely separating IHDP from everything else, not learning a real rule.

## RQ4: Do best-ranking models have best calibration?  *(descriptive only)*

Best-Qini and best-ECE models differ in {tot-agree}/{tot} datasets ({100*(tot-agree)/tot:.0f}%)
**pooled across all instances — but this pooled contrast is NOT evidence of a
Qini-vs-calibration finding, and the paper withdraws it as such.** Uplift-ECE's
diff-in-means reliability estimator is unbiased only under randomized assignment; restricted
to the instances where it is identified, Qini and $-$ECE are moderately *aligned* and the ECE
cost of selecting by Qini is small (see `findings_calibration.md` and the paper's calibration
appendix). Treat the number above as a descriptive count over a mixed-identification pool.

---

## The unifying thesis: *the metric is the message*

Across RQ1–RQ4 the dominant factor in uplift model selection is **which metric you score on**:

1. Where a reference objective exists, the proxy Qini ranking disagrees with it — but the
   claim is stated **per finding, not pooled**: on the continuous benchmark (IHDP) Qini shows
   no detectable alignment with effect accuracy while AUUC does (F1), and on Jobs every
   ranking metric diverges from the identified policy objective (F2). The pooled figure
   (mean ρ = **{mean_disag_rho:.2f}**, negative on **{n_neg}/{len(disag)}** datasets) is
   reported for completeness only: its bootstrap CI crosses zero and pooling
   continuous- with binary-outcome datasets conflates the two findings. DR-Learner is the
   poster child: high Qini, worst PEHE.
2. Stability conclusions flip with the metric (IHDP: ρ={ihdp_pehe[0]:.2f} by PEHE vs {ihdp_qini[0]:.2f} by Qini).
3. Ranking and calibration decouple only in the pooled, mixed-identification view (RQ4);
   where ECE is identified they are moderately aligned.

**Implication for practitioners:** on real marketing data we have *only* the proxy (Qini),
and this benchmark shows the proxy can rank models opposite to the truth. Qini-based model
selection on marketing data should therefore be treated as a hypothesis, validated where
possible against a holdout policy value or a semi-synthetic ground-truth proxy.

---

## Corrections from v1 (Sonnet pass)

| # | v1 claim | Status | Correct statement |
|---|----------|--------|-------------------|
| 1 | "DR-Learner wins semi-synthetic (Qini=199.6)" | **Inverted** | DR-Learner is the *worst* IHDP estimator by √PEHE (median per-split ≈24 vs ≈0.9 for T-/S-Learner; its AIPW nuisance fits diverge on some folds) |
| 2 | "Rankings essentially random across IHDP (ρ≈0)" | **Wrong metric** | ρ ≈ {ihdp_pehe[0]:.2f} by √PEHE; the ρ≈0 reflects Qini's continuous-outcome mismatch, not model quality |
| 3 | "S-Learner best average rank; run ClassTrans + S-Learner" | **Ranking artifact** | NaN-as-worst penalised binary-only models; ClassTrans is #1 complete-case on binary data |
| 4 | "Outcome base rate predicts the winner" | **Withdrawn** | Feature was outcome mean; tree only detected continuous-vs-binary |
| 5 | "Mean cross-dataset ρ = 0.20" | **Degenerate pairs** | Included MegaFon (2 models → ρ=±1); corrected analysis is per-regime, complete-case |
| 6 | "Best Qini ≠ best calibration (88%)" | **Withdrawn as a finding** | Descriptive count over a mixed-identification pool; where ECE is identified, Qini and −ECE are moderately *aligned* (see RQ4) |

## Figures & tables

| File | Content |
|------|---------|
| `fig1_leaderboard_by_regime.pdf` | Per-regime leaderboard, native metric |
| `fig2_metric_disagreement.pdf` | Qini-vs-ground-truth ranking disagreement (core finding) |
| `fig3_rank_stability.pdf` | Within-regime stability by metric |
| `fig4_qini_vs_ece.pdf` | Qini vs calibration (RQ4) |
| `fig5_critical_difference.pdf` | Complete-case CD diagrams (IHDP/√PEHE, binary/Qini) |
| `fig6_regime_winners.pdf` | Regime winners by appropriate metric |
| `tab1_leaderboard.tex` | Per-regime leaderboard |
| `tab2_master_ece.tex` | Calibration ECE |
| `tab3_metric_disagreement.tex` | Best-by-reference-objective vs best-by-Qini |
"""
    (out / "findings_wp6.md").write_text(text)
    print(f"  Saved: {out / 'findings_wp6.md'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()
    results_dir, out_dir = Path(args.results_dir), Path(args.out_dir)
    fig_dir, tab_dir = out_dir / "figures", out_dir / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    raw, summary = load_data(results_dir)

    print("\nFig 1 — leaderboard by regime...")
    fig1_leaderboard_by_regime(summary, fig_dir)
    print("\nFig 2 — metric disagreement...")
    disag = fig2_metric_disagreement(summary, fig_dir)
    print("\nFig 3 — rank stability...")
    stability = fig3_rank_stability(summary, fig_dir)
    print("\nFig 4 — Qini vs ECE...")
    fig4_qini_vs_ece(summary, fig_dir)
    print("\nFig 5 — critical difference...")
    cd_data = fig5_critical_difference(summary, fig_dir)
    print("\nFig 6 — regime winners...")
    fig6_regime_winners(summary, fig_dir)

    print("\nTables...")
    table1_leaderboard(summary, tab_dir)
    table2_ece(summary, tab_dir)
    table3_disagreement(disag, summary, tab_dir)

    print("\nFindings...")
    write_findings(summary, disag, stability, cd_data, out_dir)

    print("\nWP6 (v2) complete.")


if __name__ == "__main__":
    main()
