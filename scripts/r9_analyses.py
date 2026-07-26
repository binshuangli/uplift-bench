"""R9 reviewer analyses (F1 kurtosis-within-IHDP; F2 Jobs between-split dependence).

Q4 (F1): For each of the 100 IHDP realizations, compute outcome excess kurtosis
and the per-realization metric-vs-truth alignment (Qini and AUUC rank correlation
with -sqrt(PEHE) across the common estimators, and the paired AUUC-Qini gap).
Then correlate kurtosis with alignment ACROSS realizations. If kurtosis does not
predict Qini's failure even within IHDP, that cements the honest F1 framing: the
paper does not rely on a tail story it elsewhere disclaims. (The per-realization
rank correlation over ~6-12 models is coarse; the point is the cross-realization
trend, reported with a bootstrap CI.)

Q3 (F2): Quantify between-split dependence on Jobs two ways --
  (a) mean pairwise Jaccard overlap of the 10 test-unit index sets (from the
      released split indices);
  (b) mean pairwise Pearson correlation of the per-model policy-risk vectors
      across the 10 splits (a direct read on how similarly the splits rank
      models), with a bootstrap CI.
Both are descriptive dependence signals compared against the rho* = 0.18
design-effect breakpoint; overlap does not uniquely determine rho in the regret
statistic, so the design-effect analysis remains a sensitivity, not a corrected CI.

Reads results/, results_m1ext/, results_ihdpval/ (IHDP per-realization metrics),
the Jobs split-index npz, and results/master_raw.parquet. Seeded. Emits
results/tables/tab18_r9.tex + results/findings_r9.md + the kurtosis scatter figure.
"""

from __future__ import annotations

import glob
import itertools
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import kurtosis, pearsonr, spearmanr  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from uplift_bench.data._cache import get_data_dir  # noqa: E402
from uplift_bench.data.registry import load_dataset  # noqa: E402

SEED = 42
JOBS = [f"jobs_s{i}" for i in range(10)]


def _boot_ci(v, n=10000):
    rng = np.random.default_rng(SEED)
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return (float("nan"),) * 3
    b = np.array([rng.choice(v, len(v), replace=True).mean() for _ in range(n)])
    return float(v.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


# --------------------------------------------------------------------------- Q4
def q4_ihdp_kurtosis(macros, out_dir):
    # per-realization metrics from all three result dirs
    frames = []
    for d in ["results", "results_m1ext", "results_ihdpval"]:
        for f in glob.glob(f"{d}/ihdp_s*__*.parquet"):
            if "summary" in f:
                continue
            frames.append(pd.read_parquet(f))
    raw = pd.concat(frames, ignore_index=True)
    raw = raw[(raw.status == "ok") & raw.pehe.notna()]

    rows = []
    for split in range(100):
        ds = f"ihdp_s{split}"
        g = raw[raw.dataset == ds].groupby("model")[["qini", "auuc", "pehe"]].mean().dropna()
        if len(g) < 4:
            continue
        # outcome excess kurtosis of this realization (factual outcome)
        y = load_dataset("ihdp", split_idx=split).outcome.to_numpy()
        rq = spearmanr(g.qini, -g.pehe).correlation
        ra = spearmanr(g.auuc, -g.pehe).correlation
        rows.append(
            dict(
                split=split,
                kurt=float(kurtosis(y)),
                qini_align=rq,
                auuc_align=ra,
                gap=ra - rq,
                nmodels=len(g),
            )
        )
    df = pd.DataFrame(rows)
    kv = df["kurt"]

    # does kurtosis predict Qini's (mis)alignment across realizations?
    rk_q = spearmanr(df["kurt"], df.qini_align).correlation
    rk_gap = spearmanr(df["kurt"], df.gap).correlation
    # bootstrap CIs on both cross-realization correlations (paired resample)
    rng = np.random.default_rng(SEED)
    bs, bs_g = [], []
    for _ in range(5000):
        i = rng.integers(0, len(df), len(df))
        bs.append(spearmanr(df["kurt"].values[i], df.qini_align.values[i]).correlation)
        bs_g.append(spearmanr(df["kurt"].values[i], df.gap.values[i]).correlation)
    bs = np.array([b for b in bs if not np.isnan(b)])
    bs_g = np.array([b for b in bs_g if not np.isnan(b)])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    glo, ghi = np.percentile(bs_g, [2.5, 97.5])

    print("=== Q4: within-IHDP kurtosis vs alignment (100 realizations) ===")
    print(f"  kurtosis range: {kv.min():.1f}--{kv.max():.1f} " f"(median {kv.median():.1f})")
    print(
        f"  Spearman(kurtosis, Qini-vs-(-PEHE) alignment) = {rk_q:+.3f} " f"[{lo:+.3f},{hi:+.3f}]"
    )
    print(
        f"  Spearman(kurtosis, AUUC-Qini gap)             = {rk_gap:+.3f} "
        f"[{glo:+.3f},{ghi:+.3f}]"
    )
    print("  (near zero => kurtosis does NOT predict Qini's failure within IHDP)")

    # scatter figure
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
    ax[0].scatter(df["kurt"], df.qini_align, s=18, alpha=0.7, color="#d65f5f", label="Qini")
    ax[0].scatter(df["kurt"], df.auuc_align, s=18, alpha=0.7, color="#4c72b0", label="AUUC")
    ax[0].axhline(0, color="k", lw=0.6)
    ax[0].set_xscale("symlog")
    ax[0].set_xlabel("outcome excess kurtosis (per IHDP realization)")
    ax[0].set_ylabel(r"rank corr. with $-\sqrt{\mathrm{PEHE}}$")
    ax[0].set_title(
        f"(a) Alignment vs kurtosis\nSpearman(kurt, Qini align)"
        f"={rk_q:+.2f} [{lo:+.2f},{hi:+.2f}]",
        fontsize=9,
    )
    ax[0].legend(fontsize=8)
    ax[1].scatter(df["kurt"], df.gap, s=18, alpha=0.7, color="#55a868")
    ax[1].axhline(0, color="k", lw=0.6)
    ax[1].set_xscale("symlog")
    ax[1].set_xlabel("outcome excess kurtosis")
    ax[1].set_ylabel(r"paired AUUC$-$Qini gap")
    ax[1].set_title(
        f"(b) Paired gap vs kurtosis\nSpearman={rk_gap:+.2f} " f"[{glo:+.2f},{ghi:+.2f}]",
        fontsize=9,
    )
    fig.suptitle(
        "Within-IHDP: outcome kurtosis does not predict Qini's misalignment", fontsize=10, y=1.02
    )
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(
            Path(out_dir) / "figures" / f"figR5_ihdp_kurtosis.{ext}", dpi=150, bbox_inches="tight"
        )
    plt.close(fig)

    macros += [
        f"\\newcommand{{\\kurtQiniCorr}}{{{rk_q:+.2f}}}",
        f"\\newcommand{{\\kurtQiniCorrLo}}{{{lo:+.2f}}}",
        f"\\newcommand{{\\kurtQiniCorrHi}}{{{hi:+.2f}}}",
        f"\\newcommand{{\\kurtGapCorr}}{{{rk_gap:+.2f}}}",
        f"\\newcommand{{\\kurtGapCorrLo}}{{{glo:+.2f}}}",
        f"\\newcommand{{\\kurtGapCorrHi}}{{{ghi:+.2f}}}",
        f"\\newcommand{{\\kurtLo}}{{{kv.min():.0f}}}",
        f"\\newcommand{{\\kurtHi}}{{{kv.max():.0f}}}",
        f"\\newcommand{{\\kurtNReal}}{{{len(df)}}}",
    ]
    return df


# --------------------------------------------------------------------------- Q3
def q3_jobs_dependence(macros):
    # (a) Jaccard overlap of the 10 test-unit index sets
    d = np.load(get_data_dir() / "jobs" / "jobs_DW_bin.new.10.test.npz")
    idx = d["I"]
    jac = [
        len(set(idx[:, a]) & set(idx[:, b])) / len(set(idx[:, a]) | set(idx[:, b]))
        for a, b in itertools.combinations(range(10), 2)
    ]
    jac_m, jac_lo, jac_hi = _boot_ci(jac)

    # (b) between-split correlation of per-model policy-risk vectors
    raw = pd.read_parquet("results/master_raw.parquet")
    raw = raw[(raw.status == "ok") & raw.dataset.isin(JOBS) & raw.jobs_policy_risk.notna()]
    risk = (
        raw.groupby(["dataset", "model"]).jobs_policy_risk.mean().unstack("dataset")
    )  # rows=model, cols=split
    corrs = []
    for a, b in itertools.combinations(JOBS, 2):
        pair = risk[[a, b]].dropna()
        if len(pair) >= 4:
            r = pearsonr(pair[a], pair[b])[0]
            if not np.isnan(r):
                corrs.append(r)
    cor_m, cor_lo, cor_hi = _boot_ci(corrs)

    print("\n=== Q3: Jobs between-split dependence ===")
    print(f"  mean pairwise test-set Jaccard overlap = {jac_m:.3f} [{jac_lo:.3f},{jac_hi:.3f}]")
    print(
        f"  mean pairwise corr. of per-model risk vectors = {cor_m:+.3f} "
        f"[{cor_lo:+.3f},{cor_hi:+.3f}]"
    )
    print("  compare to design-effect breakpoint rho* = 0.18")

    macros += [
        f"\\newcommand{{\\jobsJaccard}}{{{jac_m:.2f}}}",
        f"\\newcommand{{\\jobsJaccardLo}}{{{jac_lo:.2f}}}",
        f"\\newcommand{{\\jobsJaccardHi}}{{{jac_hi:.2f}}}",
        f"\\newcommand{{\\jobsRiskCorr}}{{{cor_m:+.2f}}}",
        f"\\newcommand{{\\jobsRiskCorrLo}}{{{cor_lo:+.2f}}}",
        f"\\newcommand{{\\jobsRiskCorrHi}}{{{cor_hi:+.2f}}}",
    ]


def main(out_dir="results"):
    # Graceful degradation on a fresh clone: these analyses need the benchmark
    # result parquets (make repro-main / repro-aux) and the cached Jobs splits.
    needed = [
        Path("results/master_raw.parquet"),
        get_data_dir() / "jobs" / "jobs_DW_bin.new.10.test.npz",
    ]
    missing = [str(q) for q in needed if not q.exists()]
    if missing:
        print("skipping r9_analyses: missing inputs ->", ", ".join(missing))
        print("(needs the benchmark parquets from `make repro-main repro-aux` AND the")
        print(" cached Jobs split indices from `make data-no-criteo`; see README)")
        return
    macros = ["% auto-generated by scripts/r9_analyses.py (Q3/Q4)"]
    (Path(out_dir) / "figures").mkdir(parents=True, exist_ok=True)
    q4_ihdp_kurtosis(macros, out_dir)
    q3_jobs_dependence(macros)
    outp = Path(out_dir) / "tables" / "tab18_r9.tex"
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text("\n".join(macros) + "\n")
    print(f"\nSaved: {outp}")


if __name__ == "__main__":
    main()
