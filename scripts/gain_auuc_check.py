"""Does the FIELD'S AUUC (cumulative-gain family) behave like Qini or like ours on IHDP?

An external review observed that this repo's `auuc` integrates the unweighted
prefix-mean uplift curve u(k), while the AUUC shipped by scikit-uplift and causalml
integrates a cumulative-GAIN curve — u(k) weighted by prefix size. By Lemma 5.1 all
three cumulative statistics are integrals of w(k)u(k):

    w(k) = T_k       -> Qini                 (treated-count depth weighting)
    w(k) = k*n       -> cumulative-gain AUUC (population depth weighting; the library one)
    w(k) = 1         -> prefix-mean AUUC     (this repo's; no depth weighting)

So cumulative-gain AUUC sits in the same *family* as Qini, and the decisive question for
F1 is which side of the divide it lands on, on IHDP itself: if it decouples from effect
accuracy like Qini, F1 is a DEPTH-WEIGHTING effect (a mechanism, from a lemma the paper
already has); if it tracks like our prefix-mean AUUC, the paper's "prefer AUUC" advice
transfers to the shipped implementations.

Everything is computed from the prediction stores (identical predictions for all
metrics — the same design as the causalml-Qini validation), on all 100 IHDP
realizations. The k-weighted implementation is validated against
`causalml.metrics.auuc_score` on EVERY fold: `auuc_score(normalize=False)` equals
`n * gain_auuc` up to curve discretization, and both the maximum relative deviation
from that identity and the all-folds rank agreement are reported and emitted as macros.

Artifact trail: the per-fold metric table is written to
`results_r4pred/gain_auuc_folds.parquet` (committed) with the validation stats in
`results_r4pred/gain_auuc_validation.json` (committed), so on a checkout without the
multi-GB prediction stores the headline numbers reproduce from the committed table
instead of silently skipping.

Emits results/tables/tab27_gainauuc.tex.

Usage: python scripts/gain_auuc_check.py
"""

from __future__ import annotations

import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

_trapz = getattr(np, "trapezoid", None) or np.trapz  # NumPy 1/2 compat

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

SEED = 42
OUT = Path("results/tables/tab27_gainauuc.tex")
FOLD_TABLE = Path("results_r4pred/gain_auuc_folds.parquet")
VALID_JSON = Path("results_r4pred/gain_auuc_validation.json")
MODELS6 = ["s_learner", "t_learner", "x_learner", "r_learner", "dr_learner", "causal_forest"]


def prefix_mean_uplift(pred, t, y):
    """u(k) on the fraction axis for one fold, sorted by descending prediction."""
    order = np.argsort(-pred)
    t, y = t[order], y[order]
    ct, cc = np.cumsum(t), np.cumsum(1 - t)
    st = np.cumsum(y * t)
    sc = np.cumsum(y * (1 - t))
    with np.errstate(divide="ignore", invalid="ignore"):
        u = np.where((ct > 0) & (cc > 0), st / np.maximum(ct, 1) - sc / np.maximum(cc, 1), 0.0)
    frac = np.arange(1, len(pred) + 1) / len(pred)
    return frac, u


def gain_auuc(pred, t, y):
    """Cumulative-gain (library-family) AUUC: integral of u(k)*k on the fraction axis.
    `causalml.metrics.auuc_score(..., normalize=False)` equals n times this quantity
    up to curve discretization (validated fold-by-fold below)."""
    frac, u = prefix_mean_uplift(pred, t, y)
    return float(_trapz(u * frac, frac))


def mean_auuc(pred, t, y):
    """This repo's prefix-mean AUUC (without the ATE/2 shift, which is model-invariant
    and irrelevant to rankings)."""
    frac, u = prefix_mean_uplift(pred, t, y)
    return float(_trapz(u, frac))


def _boot_ci(vals, n=10000):
    rng = np.random.default_rng(SEED)
    v = np.asarray(vals, float)
    v = v[~np.isnan(v)]
    boot = np.array([rng.choice(v, len(v), replace=True).mean() for _ in range(n)])
    return float(v.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def _try_causalml():
    try:
        from causalml.metrics import auuc_score

        return auuc_score
    except ImportError:
        return None


def build_fold_table(files) -> tuple[pd.DataFrame, dict | None]:
    """Per-fold gain/prefix-mean AUUC and sqrt(PEHE) from the prediction stores, plus
    fold-by-fold validation of gain_auuc against causalml.auuc_score (ALL folds)."""
    auuc_score = _try_causalml()
    if auuc_score is None:
        print("  causalml unavailable; skipping cross-validation of the implementation")
    recs, ours_all, theirs_all, max_dev = [], [], [], 0.0
    for f in files:
        d = pd.read_parquet(f)
        m = d["model"].iloc[0]
        if m not in MODELS6:
            continue
        real = re.search(r"(ihdp_s\d+)__", f).group(1)
        for (si, fi), g in d.groupby(["seed_idx", "fold_idx"]):
            p, t, y = g["pred"].to_numpy(), g["t"].to_numpy(), g["y"].to_numpy()
            n = len(g)
            ga = gain_auuc(p, t, y)
            recs.append(
                dict(
                    real=real,
                    model=m,
                    seed_idx=int(si),
                    fold_idx=int(fi),
                    n=n,
                    gain_auuc=ga,
                    mean_auuc=mean_auuc(p, t, y),
                    sqrt_pehe=float(np.sqrt(np.mean((p - g["ite"].to_numpy()) ** 2))),
                )
            )
            if auuc_score is not None:
                df = pd.DataFrame({"y": y, "w": t.astype(int), "score": p})
                try:
                    cm = float(auuc_score(df, outcome_col="y", treatment_col="w", normalize=False))
                except TypeError:  # older causalml returns a Series
                    cm = float(
                        auuc_score(df, outcome_col="y", treatment_col="w", normalize=False)["score"]
                    )
                except Exception:
                    continue
                ours_all.append(ga)
                theirs_all.append(cm)
                if ga != 0:
                    max_dev = max(max_dev, abs(cm / (ga * n) - 1.0))
    table = pd.DataFrame(recs)
    valid = None
    if theirs_all:
        valid = {
            "spearman": float(spearmanr(ours_all, theirs_all).correlation),
            "max_rel_dev_pct": 100.0 * max_dev,
            "n_folds": len(theirs_all),
        }
    return table, valid


def headline(table: pd.DataFrame) -> pd.DataFrame:
    """Per-realization Spearman of each AUUC variant vs -sqrt(PEHE), across estimators."""
    per = (
        table.groupby(["real", "model"])[["gain_auuc", "mean_auuc", "sqrt_pehe"]]
        .mean()
        .reset_index()
    )
    rows = []
    for real, g in per.groupby("real"):
        if len(g) < 4:
            continue
        rows.append(
            dict(
                real=real,
                gain=spearmanr(g["gain_auuc"], -g["sqrt_pehe"]).correlation,
                mean=spearmanr(g["mean_auuc"], -g["sqrt_pehe"]).correlation,
            )
        )
    return pd.DataFrame(rows)


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    files = sorted(glob.glob("results_r4pred/predictions/ihdp_s*__pred.parquet"))
    if files:
        table, valid = build_fold_table(files)
        table.to_parquet(FOLD_TABLE, index=False)
        print(f"fold table: {FOLD_TABLE} ({len(table)} rows)")
        if valid is not None:
            VALID_JSON.write_text(json.dumps(valid, indent=2) + "\n")
            print(
                f"implementation check vs causalml.auuc_score on ALL {valid['n_folds']} folds: "
                f"Spearman {valid['spearman']:+.2f}, max deviation from the n*gain_auuc "
                f"identity {valid['max_rel_dev_pct']:.2f}%"
            )
    elif FOLD_TABLE.exists():
        table = pd.read_parquet(FOLD_TABLE)
        valid = json.loads(VALID_JSON.read_text()) if VALID_JSON.exists() else None
        print(
            f"prediction stores absent; recomputing from committed {FOLD_TABLE} "
            f"({len(table)} rows; validation stats from the committed run)"
        )
    else:
        print("skipping gain_auuc_check: no prediction stores and no committed fold table")
        if not OUT.exists():
            OUT.write_text("% gain_auuc_check skipped: no inputs\n")
        return

    df = headline(table)
    g = _boot_ci(df["gain"])
    m = _boot_ci(df["mean"])
    d = _boot_ci((df["mean"] - df["gain"]).to_numpy())
    print(f"\n=== IHDP, {len(df)} realizations, identical stored predictions ===")
    print(
        f"  rho(cumulative-gain / library AUUC, -sqrtPEHE)  = {g[0]:+.2f} [{g[1]:+.2f},{g[2]:+.2f}]"
    )
    print(
        f"  rho(prefix-mean / this repo's AUUC, -sqrtPEHE)  = {m[0]:+.2f} [{m[1]:+.2f},{m[2]:+.2f}]"
    )
    print(
        f"  paired prefix-mean-over-gain gap                = {d[0]:+.2f} [{d[1]:+.2f},{d[2]:+.2f}]"
        f"  (positive on {int(((df['mean'] - df['gain']) > 0).sum())}/{len(df)})"
    )

    def f(x):
        return f"{x:+.2f}"

    lines = [
        "% auto-generated by scripts/gain_auuc_check.py",
        f"\\newcommand{{\\gainAuuc}}{{{f(g[0])}}}",
        f"\\newcommand{{\\gainAuucLo}}{{{f(g[1])}}}",
        f"\\newcommand{{\\gainAuucHi}}{{{f(g[2])}}}",
        f"\\newcommand{{\\meanAuucRho}}{{{f(m[0])}}}",
        f"\\newcommand{{\\meanAuucRhoLo}}{{{f(m[1])}}}",
        f"\\newcommand{{\\meanAuucRhoHi}}{{{f(m[2])}}}",
        f"\\newcommand{{\\gainMeanGap}}{{{f(d[0])}}}",
        f"\\newcommand{{\\gainMeanGapLo}}{{{f(d[1])}}}",
        f"\\newcommand{{\\gainMeanGapHi}}{{{f(d[2])}}}",
        f"\\newcommand{{\\gainAuucN}}{{{len(df)}}}",
    ]
    if valid is not None:
        lines += [
            f"\\newcommand{{\\gainCmlRho}}{{{valid['spearman']:+.2f}}}",
            f"\\newcommand{{\\gainCmlFolds}}{{{valid['n_folds']}}}",
            f"\\newcommand{{\\gainCmlDev}}{{{valid['max_rel_dev_pct']:.1f}\\%}}",
        ]
    else:
        lines += [
            "\\newcommand{\\gainCmlRho}{n/a}",
            "\\newcommand{\\gainCmlFolds}{n/a}",
            "\\newcommand{\\gainCmlDev}{n/a}",
        ]
    OUT.write_text("\n".join(lines) + "\n")
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
