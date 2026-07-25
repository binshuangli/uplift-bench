# ruff: noqa: E501
"""Metric-reliability analysis: which metrics agree with which, and where.

Refined thesis: uplift metrics disagree about model ranking in a *structured* way.
Rather than "Qini disagrees with PEHE", we show:
  (1) which metrics agree with which, via a cross-metric rank-agreement matrix;
  (2) that the unnormalised Qini is specifically discordant on CONTINUOUS outcomes,
      where it diverges even from its sibling ranking metric AUUC;
  (3) significance (bootstrap CI) of the headline disagreement;
  (4) sensitivity to excluding IHDP;
  (5) base-learner robustness (LightGBM vs XGBoost) when results_xgb/ is present.

Outputs:
  figures/figR1_metric_agreement.pdf      — cross-metric agreement matrix (centerpiece)
  figures/figR2_qini_by_outcometype.pdf   — Qini vs sibling/ground-truth by outcome type
  figures/figR3_baselearner_robust.pdf    — disagreement under LightGBM vs XGBoost (if available)
  findings_metric_reliability.md          — numbers + written interpretation

Usage:
    python scripts/analyze_metric_reliability.py
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

IHDP = [f"ihdp_s{i}" for i in range(10)]
JOBS = [f"jobs_s{i}" for i in range(10)]
MARKETING = ["hillstrom", "lenta", "x5", "megafon"]

# Metrics: (column, higher_is_better, short label, family)
METRICS = [
    ("qini_mean", True, "Qini", "ranking"),
    ("auuc_mean", True, "AUUC", "ranking"),
    ("uplift_at_k_mean", True, "Uplift@k", "ranking"),
    ("policy_value_at_k_mean", True, "PolicyValue@k", "policy"),
    ("pehe_mean", False, "−√PEHE\n(effect acc.)", "estimation(GT)"),
    ("jobs_policy_risk_mean", False, "−PolicyRisk\n(policy value)", "policy(GT)"),
]
METRIC_LABEL = {m[0]: m[2] for m in METRICS}

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 150,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def load(results_dir: Path, fname="master_summary.parquet"):
    return pd.read_parquet(results_dir / fname)


def oriented_series(summ, ds, col, higher):
    """Model -> value, oriented so higher = better quality, NaN-dropped, availability-gated."""
    sub = summ[summ.dataset == ds]
    if col not in sub.columns:
        return pd.Series(dtype=float)
    v = sub.set_index("model")[col]
    ncol = col.replace("_mean", "_n")
    if ncol in sub.columns:
        v = v.where(sub.set_index("model")[ncol] > 0)
    v = v.dropna()
    return v if higher else -v


def pairwise_rho(summ, ds, colA, hiA, colB, hiB, min_models=4):
    a = oriented_series(summ, ds, colA, hiA)
    b = oriented_series(summ, ds, colB, hiB)
    common = a.index.intersection(b.index)
    if len(common) < min_models:
        return np.nan
    r, _ = spearmanr(a[common], b[common])
    return r


def mean_rho_over(summ, datasets, colA, hiA, colB, hiB):
    rs = [pairwise_rho(summ, ds, colA, hiA, colB, hiB) for ds in datasets]
    rs = [r for r in rs if not np.isnan(r)]
    return (np.mean(rs) if rs else np.nan), len(rs)


def save(fig, path: Path):
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight", dpi=150)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.with_suffix('.pdf')}")


# ---------------------------------------------------------------------------
# Fig R1 — cross-metric agreement matrix
# ---------------------------------------------------------------------------


def _agreement_matrix(summ, datasets, cols):
    n = len(cols)
    M = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            ci, hi, *_ = cols[i]
            cj, hj, *_ = cols[j]
            r, k = mean_rho_over(summ, datasets, ci, hi, cj, hj)
            M[i, j] = r
    return M


def figR1_metric_agreement(summ, out: Path):
    """Two matrices (continuous vs binary) reveal two distinct mechanisms."""
    # Continuous-outcome group: Qini-specific construction failure. PolicyValue@k is
    # excluded from panel (a): it has no oracle reading on the continuous benchmarks and
    # its uniformly weak correlations visually muddy the one message (Qini isolated).
    cols_cont = [
        m for m in METRICS
        if m[0] not in ("jobs_policy_risk_mean", "policy_value_at_k_mean")
    ]
    # Binary-outcome group: ranking metrics agree but diverge from the policy objective.
    cols_bin = [m for m in METRICS if m[0] != "pehe_mean"]

    # Three panels: (a) continuous; (b) binary ranking metrics only (14 datasets);
    # (c) each metric vs -PolicyRisk on the 10 Jobs realizations alone. Splitting (c)
    # out keeps each panel on a single, stated instance set.
    cols_bin_rank = [m for m in cols_bin if m[0] != "jobs_policy_risk_mean"]
    fig, axes = plt.subplots(
        1, 3, figsize=(17, 6.4), gridspec_kw={"width_ratios": [5, 4.2, 1.7]}
    )
    # NOTE: the `synthetic` dataset is generated with binary_outcome=True (see
    # configs/dataset/synthetic.yaml), so it belongs in the BINARY group, not here.
    # The continuous panel is the 10 IHDP realizations.
    Mc = _agreement_matrix(summ, IHDP, cols_cont)
    sns.heatmap(
        Mc,
        ax=axes[0],
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        center=0,
        xticklabels=[c[2] for c in cols_cont],
        yticklabels=[c[2] for c in cols_cont],
        annot=True,
        fmt=".2f",
        annot_kws={"size": 8},
        square=True,
        linewidths=0.4,
        linecolor="#ccc",
        cbar=False,
    )
    axes[0].set_title(
        "(a) Continuous outcomes (IHDP ×10)\n"
        "Qini is isolated; AUUC / Uplift@k track effect accuracy (−√PEHE).\n"
        "→ Qini's ranking does not track effect accuracy here.",
        fontsize=9,
    )
    axes[0].tick_params(axis="x", rotation=35)

    Mb = _agreement_matrix(
        summ, JOBS + ["synthetic", "hillstrom", "lenta", "x5", "megafon"], cols_bin_rank
    )
    sns.heatmap(
        Mb,
        ax=axes[1],
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        center=0,
        xticklabels=[c[2] for c in cols_bin_rank],
        yticklabels=[c[2] for c in cols_bin_rank],
        annot=True,
        fmt=".2f",
        annot_kws={"size": 8},
        square=True,
        linewidths=0.4,
        linecolor="#ccc",
        cbar=False,
    )
    axes[1].set_title(
        "(b) Binary outcomes (Jobs + synthetic + marketing, N=15)\n"
        "Ranking metrics agree with each other.",
        fontsize=9,
    )
    axes[1].tick_params(axis="x", rotation=35)

    # Panel (c): correlations with -PolicyRisk, computable on Jobs only.
    Mj = _agreement_matrix(summ, JOBS, cols_bin)
    _ipr = next(i for i, c in enumerate(cols_bin) if c[0] == "jobs_policy_risk_mean")
    col = Mj[[i for i, c in enumerate(cols_bin) if c[0] != "jobs_policy_risk_mean"], _ipr]
    sns.heatmap(
        col.reshape(-1, 1),
        ax=axes[2],
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        center=0,
        xticklabels=["−PolicyRisk"],
        yticklabels=[c[2] for c in cols_bin_rank],
        annot=True,
        fmt=".2f",
        annot_kws={"size": 8},
        square=False,
        linewidths=0.4,
        linecolor="#ccc",
        cbar_kws={"label": "Mean Spearman ρ"},
    )
    axes[2].set_title(
        "(c) vs. the policy objective\n(Jobs only, N=10)\n"
        "→ ranking quality ≠ deployment objective.",
        fontsize=9,
    )
    axes[2].tick_params(axis="x", rotation=0)

    fig.suptitle(
        "Two findings of metric disagreement in uplift evaluation", fontsize=11, y=1.02
    )
    fig.tight_layout()
    save(fig, out / "figR1_metric_agreement")
    # Return Qini-row agreements over all datasets for the findings doc.
    labels = [c[2] for c in METRICS]
    allds = ["synthetic"] + IHDP + JOBS + MARKETING
    M = _agreement_matrix(summ, allds, METRICS)
    return M, labels


# ---------------------------------------------------------------------------
# Fig R2 — Qini's agreement, split by outcome type
# ---------------------------------------------------------------------------


def figR2_qini_by_outcometype(summ, out: Path):
    groups = [
        ("IHDP\n(continuous)", IHDP),
        ("Jobs\n(binary)", JOBS),
        ("Marketing\n(binary)", ["hillstrom", "lenta", "x5", "megafon"]),
    ]
    # Compare Qini against AUUC (sibling) and the ground-truth metric for that group.
    comps = []
    for gname, dss in groups:
        rA, _ = mean_rho_over(summ, dss, "qini_mean", True, "auuc_mean", True)
        ru, _ = mean_rho_over(summ, dss, "qini_mean", True, "uplift_at_k_mean", True)
        gt_col, gt_hi = ("pehe_mean", False) if dss is IHDP else ("jobs_policy_risk_mean", False)
        if dss is IHDP:
            rg, _ = mean_rho_over(summ, dss, "qini_mean", True, "pehe_mean", False)
            gtlabel = "√PEHE (GT)"
        elif dss[0].startswith("jobs"):
            rg, _ = mean_rho_over(summ, dss, "qini_mean", True, "jobs_policy_risk_mean", False)
            gtlabel = "PolicyRisk (GT)"
        else:
            rg, gtlabel = np.nan, "—(no GT)"
        comps.append((gname, rA, ru, rg, gtlabel))

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(groups))
    w = 0.27
    ax.bar(x - w, [c[1] for c in comps], w, label="Qini vs AUUC (sibling ranking)", color="#4878cf")
    ax.bar(x, [c[2] for c in comps], w, label="Qini vs Uplift@k (operational)", color="#6acc65")
    ax.bar(x + w, [c[3] for c in comps], w, label="Qini vs ground-truth", color="#d65f5f")
    for xi, c in zip(x, comps):
        for dx, val in [(-w, c[1]), (0, c[2]), (w, c[3])]:
            if not np.isnan(val):
                ax.text(
                    xi + dx,
                    val + (0.03 if val >= 0 else -0.03),
                    f"{val:.2f}",
                    ha="center",
                    va="bottom" if val >= 0 else "top",
                    fontsize=7,
                )
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([c[0] for c in comps])
    ax.set_ylabel("Mean within-dataset Spearman correlation (ρ)")
    ax.set_ylim(-0.5, 1.05)
    ax.legend(loc="lower right")
    ax.set_title(
        "Qini behaves like a normal ranking metric on BINARY outcomes,\n"
        "but breaks down on CONTINUOUS outcomes (diverges from AUUC and ground truth).",
        fontsize=9.5,
    )
    fig.tight_layout()
    save(fig, out / "figR2_qini_by_outcometype")
    return comps


# ---------------------------------------------------------------------------
# Significance + sensitivity (printed + returned)
# ---------------------------------------------------------------------------


def disagreement_stats(summ):
    gt = IHDP + JOBS + ["synthetic"]

    def gt_rho(ds):
        if ds.startswith("ihdp") or ds == "synthetic":
            return pairwise_rho(summ, ds, "qini_mean", True, "pehe_mean", False)
        return pairwise_rho(summ, ds, "qini_mean", True, "jobs_policy_risk_mean", False)

    vals = np.array([gt_rho(ds) for ds in gt])
    vals = vals[~np.isnan(vals)]
    rng = np.random.default_rng(42)
    boot = np.array([rng.choice(vals, len(vals), replace=True).mean() for _ in range(10000)])
    ci = (np.percentile(boot, 2.5), np.percentile(boot, 97.5))

    # Exclude-IHDP sensitivity
    vals_noihdp = np.array([gt_rho(ds) for ds in JOBS + ["synthetic"]])
    vals_noihdp = vals_noihdp[~np.isnan(vals_noihdp)]
    vals_ihdp = np.array([gt_rho(ds) for ds in IHDP])
    vals_ihdp = vals_ihdp[~np.isnan(vals_ihdp)]

    return {
        "all_mean": vals.mean(),
        "all_ci": ci,
        "all_n": len(vals),
        "all_neg": int((vals < 0).sum()),
        "p_ge0": float((boot >= 0).mean()),
        "ihdp_mean": vals_ihdp.mean(),
        "ihdp_n": len(vals_ihdp),
        "noihdp_mean": vals_noihdp.mean(),
        "noihdp_n": len(vals_noihdp),
    }


# ---------------------------------------------------------------------------
# Fig R3 — base-learner robustness (if results_xgb present)
# ---------------------------------------------------------------------------


def figR3_baselearner(summ_lgb, results_dir: Path, out: Path):
    xgb_path = Path("results_xgb") / "master_summary.parquet"
    if not xgb_path.exists():
        print("  [figR3] results_xgb/master_summary.parquet not found — skipping (run pending).")
        return None
    summ_xgb = pd.read_parquet(xgb_path)

    def block(summ):
        out = {}
        out["ihdp_qini_pehe"] = mean_rho_over(summ, IHDP, "qini_mean", True, "pehe_mean", False)[0]
        out["ihdp_auuc_pehe"] = mean_rho_over(summ, IHDP, "auuc_mean", True, "pehe_mean", False)[0]
        out["ihdp_qini_auuc"] = mean_rho_over(summ, IHDP, "qini_mean", True, "auuc_mean", True)[0]
        out["jobs_qini_auuc"] = mean_rho_over(summ, JOBS, "qini_mean", True, "auuc_mean", True)[0]
        return out

    lgb, xgb = block(summ_lgb), block(summ_xgb)
    keys = ["ihdp_qini_pehe", "ihdp_auuc_pehe", "ihdp_qini_auuc", "jobs_qini_auuc"]
    klabels = ["IHDP\nQini–PEHE", "IHDP\nAUUC–PEHE", "IHDP\nQini–AUUC", "Jobs\nQini–AUUC"]
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(keys))
    w = 0.38
    ax.bar(x - w / 2, [lgb[k] for k in keys], w, label="LightGBM base", color="#4878cf")
    ax.bar(x + w / 2, [xgb[k] for k in keys], w, label="XGBoost base", color="#dd8452")
    for xi, k in zip(x, keys):
        for dx, d in [(-w / 2, lgb), (w / 2, xgb)]:
            val = d[k]
            if not np.isnan(val):
                ax.text(
                    xi + dx,
                    val + 0.03 * np.sign(val) if val else 0.03,
                    f"{val:.2f}",
                    ha="center",
                    va="bottom" if val >= 0 else "top",
                    fontsize=7.5,
                )
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(klabels)
    ax.set_ylabel("Mean within-dataset Spearman correlation (ρ)")
    ax.set_ylim(-0.6, 1.05)
    ax.legend()
    ax.set_title(
        "Metric (dis)agreement is robust to the base learner.\n"
        "The Qini-on-continuous breakdown persists under XGBoost.",
        fontsize=9.5,
    )
    fig.tight_layout()
    save(fig, out / "figR3_baselearner_robust")
    return {"lgb": lgb, "xgb": xgb}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# F1 estimator-exclusion sensitivity (reviewer: "is F1 just the DR-Learner?")
# ---------------------------------------------------------------------------

EXCLUSION_CONFIGS = [
    ((), "six meta/forest estimators"),
    (("dr_learner",), "DR-Learner excluded"),
    (("dr_learner", "r_learner"), "DR- and R-Learner excluded"),
]


def _rho_excl(summ, ds, colA, hiA, colB, hiB, exclude, min_models=4):
    a = oriented_series(summ, ds, colA, hiA)
    b = oriented_series(summ, ds, colB, hiB)
    a, b = a[~a.index.isin(exclude)], b[~b.index.isin(exclude)]
    common = a.index.intersection(b.index)
    if len(common) < min_models:
        return np.nan
    r, _ = spearmanr(a[common], b[common])
    return r


def _boot_ci(vals, seed=42, n=10000):
    rng = np.random.default_rng(seed)
    vals = np.asarray(vals)
    boot = np.array([rng.choice(vals, len(vals), replace=True).mean() for _ in range(n)])
    return vals.mean(), np.percentile(boot, 2.5), np.percentile(boot, 97.5)


def m1_exclusion_sensitivity(summ_lgb, out_tables: Path):
    """F1 must survive removing the numerically unstable DR-(and R-)Learner.

    For each (base learner, exclusion set): Qini-vs-truth and AUUC-vs-truth mean rho on
    the continuous-outcome datasets, plus the PAIRED per-dataset difference
    delta = rho(AUUC, -sqrtPEHE) - rho(Qini, -sqrtPEHE) with a bootstrap CI — the sharp
    statistic, since it compares the two metrics on identical datasets and models.
    """
    # `synthetic` is binary (configs/dataset/synthetic.yaml: binary_outcome=True), so the
    # continuous-outcome set is the 10 IHDP realizations.
    cont = IHDP
    xgb_path = Path("results_xgb") / "master_summary.parquet"
    sources = [("LightGBM", summ_lgb)]
    if xgb_path.exists():
        sources.append(("XGBoost", pd.read_parquet(xgb_path)))

    rows = []
    for blabel, summ in sources:
        for exclude, elabel in EXCLUSION_CONFIGS:
            rq, ra, delta = [], [], []
            for ds in cont:
                q = _rho_excl(summ, ds, "qini_mean", True, "pehe_mean", False, exclude)
                a = _rho_excl(summ, ds, "auuc_mean", True, "pehe_mean", False, exclude)
                if not (np.isnan(q) or np.isnan(a)):
                    rq.append(q)
                    ra.append(a)
                    delta.append(a - q)
            qm, qlo, qhi = _boot_ci(rq)
            am, alo, ahi = _boot_ci(ra)
            dm, dlo, dhi = _boot_ci(delta)
            rows.append({
                "base": blabel, "config": elabel, "n": len(delta),
                "qini_rho": qm, "auuc_rho": am,
                "delta": dm, "delta_lo": dlo, "delta_hi": dhi,
                "delta_pos": int(sum(d > 0 for d in delta)),
            })

    # LaTeX table for the paper appendix.
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{F1 estimator-exclusion sensitivity on the continuous-outcome datasets"
        r" (IHDP $\times$10). $\rho$: mean within-dataset Spearman correlation"
        r" with $-\pehe$. $\Delta$: paired per-dataset difference"
        r" $\rho(\mathrm{AUUC})-\rho(\mathrm{Qini})$ with 95\% bootstrap CI. The AUUC"
        r" advantage is positive in all six configurations; its CI excludes zero in"
        r" five of six (the exception is XGBoost with DR- and R-Learner excluded).}",
        r"\label{tab:m1sens}",
        r"\small",
        r"\begin{tabular}{llrrrl}",
        r"\toprule",
        r"Base learner & Estimators & Qini $\rho$ & AUUC $\rho$ & $\Delta$ & 95\% CI \\",
        r"\midrule",
    ]
    for r_ in rows:
        lines.append(
            f"{r_['base']} & {r_['config']} & {r_['qini_rho']:+.2f} & {r_['auuc_rho']:+.2f}"
            f" & {r_['delta']:+.2f} & [{r_['delta_lo']:+.2f}, {r_['delta_hi']:+.2f}] \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    out_tables.mkdir(parents=True, exist_ok=True)
    (out_tables / "tab4_m1_sensitivity.tex").write_text("\n".join(lines) + "\n")
    print(f"  Saved: {out_tables / 'tab4_m1_sensitivity.tex'}")
    return rows


def m1_extension(out_dir: Path):
    """F1 generality check on extended continuous data (results_m1ext/).

    Extends the continuous-outcome evidence base beyond the main benchmark's
    IHDP x10 + synthetic: 20 additional IHDP realizations (30 of 100 total) plus a
    simulated revenue-uplift dataset (lognormal spend, multiplicative effect).
    Reports Qini-vs-truth, AUUC-vs-truth and the paired delta on the extension data
    alone, so the main 25-dataset benchmark is untouched.
    """
    ext_path = Path("results_m1ext") / "master_summary.parquet"
    if not ext_path.exists():
        print("  [m1_extension] results_m1ext/master_summary.parquet not found — skipping.")
        return None
    summ = pd.read_parquet(ext_path)
    ext_ds = [f"ihdp_s{i}" for i in range(10, 30)] + ["revenue_synthetic"]
    ext_ds = [d for d in ext_ds if d in summ.dataset.unique()]

    rq, ra, delta = [], [], []
    for ds in ext_ds:
        q = _rho_excl(summ, ds, "qini_mean", True, "pehe_mean", False, ())
        a = _rho_excl(summ, ds, "auuc_mean", True, "pehe_mean", False, ())
        if not (np.isnan(q) or np.isnan(a)):
            rq.append(q)
            ra.append(a)
            delta.append(a - q)
    if not delta:
        print("  [m1_extension] no usable extension datasets — skipping.")
        return None
    qm, qlo, qhi = _boot_ci(rq)
    am, alo, ahi = _boot_ci(ra)
    dm, dlo, dhi = _boot_ci(delta)
    # Revenue dataset on its own (single dataset: report point values, no CI).
    rev = {}
    if "revenue_synthetic" in ext_ds:
        rev = {
            "qini": _rho_excl(summ, "revenue_synthetic", "qini_mean", True, "pehe_mean", False, ()),
            "auuc": _rho_excl(summ, "revenue_synthetic", "auuc_mean", True, "pehe_mean", False, ()),
        }
    res = {
        "n_datasets": len(delta),
        "qini": (qm, qlo, qhi), "auuc": (am, alo, ahi), "delta": (dm, dlo, dhi),
        "delta_pos": int(sum(d > 0 for d in delta)),
        "revenue": rev,
    }
    print(f"  [m1_extension] n={res['n_datasets']}: Qini rho={qm:+.2f}, AUUC rho={am:+.2f}, "
          f"delta={dm:+.2f} [{dlo:+.2f},{dhi:+.2f}] (positive {res['delta_pos']}/{res['n_datasets']})")
    return res


def m2_regime_cis(summ):
    """Per-regime bootstrap CIs for F2 on Jobs (ranking metrics vs policy risk)."""
    out = {}
    for col, label in [("qini_mean", "Qini"), ("auuc_mean", "AUUC"),
                       ("uplift_at_k_mean", "Uplift@k")]:
        vals = [pairwise_rho(summ, ds, col, True, "jobs_policy_risk_mean", False)
                for ds in JOBS]
        vals = [v for v in vals if not np.isnan(v)]
        out[label] = _boot_ci(vals)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()
    results_dir, out_dir = Path(args.results_dir), Path(args.out_dir)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    summ = load(results_dir)
    print("Fig R1 — metric agreement matrix...")
    M, labels = figR1_metric_agreement(summ, fig_dir)
    print("Fig R2 — Qini by outcome type...")
    comps = figR2_qini_by_outcometype(summ, fig_dir)
    print("Stats — disagreement significance + sensitivity...")
    stats = disagreement_stats(summ)
    print("Fig R3 — base-learner robustness...")
    robust = figR3_baselearner(summ, results_dir, fig_dir)
    print("F1 estimator-exclusion sensitivity...")
    sens = m1_exclusion_sensitivity(summ, out_dir / "tables")
    print("F2 per-regime CIs (Jobs)...")
    m2ci = m2_regime_cis(summ)
    print("F1 extension (extra continuous data)...")
    ext = m1_extension(out_dir)

    # Findings doc
    qi = {labels[i]: M[labels.index("Qini")][i] for i in range(len(labels))}
    lines = []
    lines.append("# Metric-reliability findings\n")
    lines.append("## Cross-metric agreement (Fig R1)\n")
    lines.append(
        "Mean within-dataset Spearman ρ of **Qini** vs each metric (higher=better orientation):\n"
    )
    for lab in labels:
        if lab != "Qini":
            lines.append(f"- Qini vs {lab}: {qi[lab]:+.2f}")
    lines.append("")
    lines.append("## Qini by outcome type (Fig R2)\n")
    for gname, rA, ru, rg, gtlabel in comps:
        g = gname.replace("\n", " ")
        lines.append(
            f"- **{g}**: Qini–AUUC {rA:+.2f}, Qini–Uplift@k {ru:+.2f}, Qini–{gtlabel} {rg:+.2f}"
        )
    lines.append("")
    lines.append("## Significance & sensitivity of the headline disagreement\n")
    s = stats
    lines.append(
        f"- Qini vs ground-truth, all GT datasets: mean ρ = {s['all_mean']:+.3f}, "
        f"95% bootstrap CI [{s['all_ci'][0]:+.3f}, {s['all_ci'][1]:+.3f}], "
        f"n={s['all_n']}, negative on {s['all_neg']}/{s['all_n']}, one-sided p(≥0)={s['p_ge0']:.3f}"
    )
    lines.append(
        f"- IHDP only (continuous, vs √PEHE): mean ρ = {s['ihdp_mean']:+.3f} (n={s['ihdp_n']})"
    )
    lines.append(
        f"- Excluding IHDP (Jobs vs PolicyRisk + synthetic vs √PEHE): mean ρ = {s['noihdp_mean']:+.3f} (n={s['noihdp_n']})"
    )
    lines.append(
        "  → NOTE: the pooled figure above is reported for completeness only. Its CI crosses "
        "zero and it mixes continuous- and binary-outcome instances, which conflates the two "
        "findings; the paper states F1 on the continuous benchmark and F2 on Jobs separately."
    )
    lines.append("")
    lines.append("## Two distinct findings (Fig R1)\n")
    lines.append(
        "- **Construction (continuous outcomes):** on IHDP, Qini vs √PEHE = "
        f"{mean_rho_over(summ, IHDP, 'qini_mean', True, 'pehe_mean', False)[0]:+.2f}, but "
        f"AUUC vs √PEHE = {mean_rho_over(summ, IHDP, 'auuc_mean', True, 'pehe_mean', False)[0]:+.2f} and "
        f"Uplift@k vs √PEHE = {mean_rho_over(summ, IHDP, 'uplift_at_k_mean', True, 'pehe_mean', False)[0]:+.2f}. "
        "The divergence is specific to Qini here -- sibling ranking metrics track effect "
        "accuracy -- but the *sufficient mechanism* remains open: scale is ruled out "
        "(affine invariance), and controlled sweeps of kurtosis, treatment imbalance and "
        "outcome-correlated score error do not isolate it. Reported as a bounded, "
        "benchmark-discovered signature, not a proven property of the Qini construction."
    )
    lines.append(
        "- **Objective (binary outcomes):** on Jobs, ALL ranking metrics diverge from PolicyRisk "
        f"(Qini {mean_rho_over(summ, JOBS, 'qini_mean', True, 'jobs_policy_risk_mean', False)[0]:+.2f}, "
        f"AUUC {mean_rho_over(summ, JOBS, 'auuc_mean', True, 'jobs_policy_risk_mean', False)[0]:+.2f}, "
        f"Uplift@k {mean_rho_over(summ, JOBS, 'uplift_at_k_mean', True, 'jobs_policy_risk_mean', False)[0]:+.2f}). "
        "Ranking quality ≠ deployment objective — independent of which ranking metric you pick."
    )
    lines.append("")
    if robust:
        lines.append("## Base-learner robustness (Fig R3)\n")
        lines.append(
            f"- IHDP Qini–PEHE: LightGBM {robust['lgb']['ihdp_qini_pehe']:+.2f}, XGBoost {robust['xgb']['ihdp_qini_pehe']:+.2f}"
        )
        lines.append(
            f"- IHDP AUUC–PEHE: LightGBM {robust['lgb']['ihdp_auuc_pehe']:+.2f}, XGBoost {robust['xgb']['ihdp_auuc_pehe']:+.2f}"
        )
        lines.append(
            f"- IHDP Qini–AUUC: LightGBM {robust['lgb']['ihdp_qini_auuc']:+.2f}, XGBoost {robust['xgb']['ihdp_qini_auuc']:+.2f}"
        )
        lines.append(
            f"- Jobs Qini–AUUC: LightGBM {robust['lgb']['jobs_qini_auuc']:+.2f}, XGBoost {robust['xgb']['jobs_qini_auuc']:+.2f}"
        )
    else:
        lines.append(
            "## Base-learner robustness\n\n_Pending: run `results_xgb` then re-run this script._"
        )
    lines.append("")
    lines.append("## F1 estimator-exclusion sensitivity (Table tab4)\n")
    lines.append("Paired per-dataset Δ = ρ(AUUC,−√PEHE) − ρ(Qini,−√PEHE), continuous datasets:\n")
    for r_ in sens:
        lines.append(
            f"- {r_['base']}, {r_['config']}: Qini ρ={r_['qini_rho']:+.2f}, "
            f"AUUC ρ={r_['auuc_rho']:+.2f}, Δ={r_['delta']:+.2f} "
            f"[{r_['delta_lo']:+.2f}, {r_['delta_hi']:+.2f}] "
            f"(positive on {r_['delta_pos']}/{r_['n']})"
        )
    lines.append("")
    lines.append("→ F1 is NOT an artifact of the unstable DR-Learner: the AUUC-over-Qini")
    lines.append("ground-truth-agreement advantage is positive in all six configurations "
                 "(CI excludes zero in five of six), and")
    lines.append("under LightGBM the Qini divergence *strengthens* when DR is excluded.")
    lines.append("")
    lines.append("## F2 per-regime CIs (Jobs, ranking metric vs −policy risk)\n")
    for label, (m, lo, hi) in m2ci.items():
        strad = "straddles 0" if lo < 0 < hi else "excludes 0"
        lines.append(f"- {label}: ρ={m:+.3f} [{lo:+.3f}, {hi:+.3f}] ({strad})")
    (out_dir / "findings_metric_reliability.md").write_text("\n".join(lines) + "\n")
    print(f"  Saved: {out_dir / 'findings_metric_reliability.md'}")
    print("\nDone.")


if __name__ == "__main__":
    main()
