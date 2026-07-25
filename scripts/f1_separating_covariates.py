"""Which realization-level covariate, if any, separates the IHDP realizations where
Qini's ranking tracks effect accuracy from those where it does not?

`r9_analyses.py` already tested outcome excess **kurtosis** and found it does not
predict Qini's misalignment across the 100 IHDP realizations. Kurtosis may simply be
the wrong summary: the realizations differ enormously in outcome *level* (mean outcome
ranges ~2-41) and the largest sqrt(PEHE) blow-ups sit on particular splits. This script
tests three further candidates, all computable ex ante from the observed data alone
(no ground-truth effects), against the same alignment outcomes:

  ybar        mean outcome (the level)
  cv          coefficient of variation, sd(y)/|mean(y)| (scale-free dispersion)
  arm_ratio   mean outcome ratio between arms, mean(y|T=1)/mean(y|T=0)

IMPORTANT (affine invariance). Proposition 5.1 proves every metric's within-split model
ranking is invariant to an affine transform of the outcome, so outcome level or scale
*per se* cannot drive F1: rescaling one dataset changes nothing. These covariates vary
across realizations because the response surfaces differ, not because one surface has
been rescaled, so a correlation here identifies a property the covariate *proxies*
(surface shape/heterogeneity) -- it is not evidence that scale matters. `arm_ratio` is
the most interesting of the three precisely because it is arm-asymmetric and therefore
NOT affine-invariant.

Alignment outcomes per realization (across the common estimators, as in r9):
  qini_align  Spearman(Qini, -sqrt(PEHE))
  auuc_align  Spearman(AUUC, -sqrt(PEHE))
  gap         auuc_align - qini_align                    (the load-bearing F1 statistic)

Reporting all three is the point. A covariate that predicts *both* alignments is a
realization-difficulty covariate; only a covariate that predicts the **gap** would be a
candidate mechanism for F1, which is a claim about one metric relative to another.

Emits results/tables/tab22_sepcov.tex + prints a summary. Needs the IHDP result
parquets (`make repro-main repro-aux`) and the cached IHDP data.

Usage: python scripts/f1_separating_covariates.py
"""

from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from uplift_bench.data.registry import load_dataset  # noqa: E402

SEED = 42
OUT = Path("results/tables/tab22_sepcov.tex")        # \newcommand macros (preamble)
OUT_BODY = Path("results/tables/tab22_sepcov_body.tex")  # the table float (appendix)
# Candidate separating covariates, in the order reported.
COVARIATES = ["kurt", "ybar", "cv", "arm_ratio"]
LABELS = {
    "kurt": r"excess kurtosis",
    "ybar": r"mean outcome $\bar y$",
    "cv": r"coef.\ of variation",
    "arm_ratio": r"arm outcome ratio",
}


def _boot_corr(x, y, n=5000):
    """Paired-resample bootstrap CI for a Spearman correlation."""
    rng = np.random.default_rng(SEED)
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok], y[ok]
    point = spearmanr(x, y).correlation
    boot = []
    for _ in range(n):
        i = rng.integers(0, len(x), len(x))
        r = spearmanr(x[i], y[i]).correlation
        if not np.isnan(r):
            boot.append(r)
    boot = np.asarray(boot)
    return float(point), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def build_frame() -> pd.DataFrame:
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
        g = (raw[raw.dataset == f"ihdp_s{split}"]
             .groupby("model")[["qini", "auuc", "pehe"]].mean().dropna())
        if len(g) < 4:
            continue
        ds = load_dataset("ihdp", split_idx=split)
        y = ds.outcome.to_numpy(dtype=float)
        t = ds.treatment.to_numpy()
        m1, m0 = y[t == 1].mean(), y[t == 0].mean()
        rq = spearmanr(g.qini, -g.pehe).correlation
        ra = spearmanr(g.auuc, -g.pehe).correlation
        rows.append(dict(
            split=split,
            kurt=float(kurtosis(y)),
            ybar=float(y.mean()),
            cv=float(y.std() / abs(y.mean())) if y.mean() != 0 else np.nan,
            arm_ratio=float(m1 / m0) if m0 != 0 else np.nan,
            qini_align=rq,
            auuc_align=ra,
            gap=ra - rq,
        ))
    return pd.DataFrame(rows)


def main():
    needed = Path("results/master_raw.parquet")
    if not needed.exists():
        print(f"skipping f1_separating_covariates: {needed} missing "
              "(needs `make repro-main repro-aux`)")
        return
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df = build_frame()

    print(f"=== F1 separating-covariate search ({len(df)} IHDP realizations) ===")
    print("Spearman(covariate, alignment) with 95% paired bootstrap CI\n")
    def fmt(r):
        return f"{r[0]:+.2f} [{r[1]:+.2f},{r[2]:+.2f}]"

    print(f"{'covariate':<16}{'range':<16}{'vs Qini':<22}{'vs AUUC':<22}{'vs GAP'}")
    lines = []
    for cov in COVARIATES:
        v = df[cov]
        rq = _boot_corr(v, df.qini_align)
        ra = _boot_corr(v, df.auuc_align)
        rg = _boot_corr(v, df.gap)
        rng_s = f"{v.min():.1f}--{v.max():.1f}"
        print(f"{cov:<16}{rng_s:<16}{fmt(rq):<22}{fmt(ra):<22}{fmt(rg)}")
        lines.append((cov, rng_s, rq, ra, rg))

    gap_hits = [c for c, _, _, _, rg in lines if rg[1] > 0 or rg[2] < 0]
    both_hits = [c for c, _, rq, ra, _ in lines
                 if (rq[1] > 0 or rq[2] < 0) and (ra[1] > 0 or ra[2] < 0)]
    print(f"\npredict BOTH alignments (difficulty covariates): {both_hits or 'none'}")
    print(f"predict the GAP (candidate F1 mechanisms):       {gap_hits or 'none'}")
    print("\nNote: 4 covariates x 3 outcomes = 12 tests; treat marginal CIs accordingly.")

    table = [
        r"\begin{table}[ht]", r"\centering",
        r"\caption{Searching for a realization-level covariate that separates the IHDP "
        r"realizations where Qini's ranking tracks effect accuracy from those where it "
        r"does not. Spearman correlation across the $\sepcovN$ realizations between each "
        r"ex-ante covariate (computable from observed data alone) and two alignment "
        r"outcomes plus AUUC's alignment, with 95\% paired bootstrap CIs. Only a covariate "
        r"predicting the \emph{gap} would be a candidate mechanism for F1; covariates "
        r"predicting both alignments merely mark harder realizations. By Prop.~\ref{prop:m1} an affine "
        r"transform of the outcome leaves every ranking unchanged, so outcome level or "
        r"scale cannot itself drive F1; these covariates vary because the response "
        r"surfaces differ, and the arm ratio is the one candidate that is not "
        r"affine-invariant.}",
        r"\label{tab:sepcov}", r"\small",
        r"\begin{tabular}{llrrr}", r"\toprule",
        r"Covariate & Range & vs.\ Qini align. & vs.\ AUUC align. & vs.\ gap \\",
        r"\midrule",
    ]
    def cell(r):
        return f"${r[0]:+.2f}$ [${r[1]:+.2f},{r[2]:+.2f}$]"

    for cov, rng_s, rq, ra, rg in lines:
        table.append(f"{LABELS[cov]} & {rng_s} & {cell(rq)} & {cell(ra)} & {cell(rg)} \\\\")
    table += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    OUT_BODY.write_text("\n".join(table) + "\n")

    body = ["% auto-generated by scripts/f1_separating_covariates.py"]
    body.append(f"\\newcommand{{\\sepcovN}}{{{len(df)}}}")
    body.append(f"\\newcommand{{\\sepcovNGapHits}}{{{len(gap_hits)}}}")
    for cov, _, rq, ra, rg in lines:
        cc = cov.replace("_", "")
        for tag, r in (("Qini", rq), ("Auuc", ra), ("Gap", rg)):
            body.append(f"\\newcommand{{\\sepcov{cc}{tag}}}{{{r[0]:+.2f}}}")
            body.append(f"\\newcommand{{\\sepcov{cc}{tag}Lo}}{{{r[1]:+.2f}}}")
            body.append(f"\\newcommand{{\\sepcov{cc}{tag}Hi}}{{{r[2]:+.2f}}}")
    OUT.write_text("\n".join(body) + "\n")
    print(f"\nSaved: {OUT} and {OUT_BODY}")


if __name__ == "__main__":
    main()
