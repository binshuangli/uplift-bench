"""Does the FIELD'S AUUC (cumulative-gain family) behave like Qini or like ours on IHDP?

An external review observed that this repo's `auuc` integrates the unweighted
prefix-mean uplift curve u(k), while the AUUC shipped by scikit-uplift and causalml
integrates a cumulative-GAIN curve — u(k) weighted by prefix size. By Lemma 5.1 all
three cumulative statistics are integrals of w(k)u(k):

    w(k) = T_k       -> Qini            (treated-count depth weighting)
    w(k) = k*n       -> library AUUC    (population depth weighting)
    w(k) = 1         -> this repo's AUUC (no depth weighting)

So library AUUC sits in the same *family* as Qini, and the decisive question for F1 is
which side of the divide it lands on, on IHDP itself: if it decouples from effect
accuracy like Qini, F1 is a DEPTH-WEIGHTING effect (a mechanism, from a lemma the paper
already has); if it tracks like our unweighted AUUC, the paper's "prefer AUUC" advice
transfers to the shipped implementations.

Everything is computed from the committed prediction stores (identical predictions for
all metrics — the same design as the causalml-Qini validation), on all 100 IHDP
realizations. The k-weighted implementation is validated against
`causalml.metrics.auuc_score` on identical fold predictions before being trusted.

Emits results/tables/tab27_gainauuc.tex.

Usage: python scripts/gain_auuc_check.py
"""

from __future__ import annotations

import glob
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
    """Library-family AUUC: integral of the cumulative-gain curve u(k)*k on the
    fraction axis (matches scikit-uplift's uplift curve up to the constant n)."""
    frac, u = prefix_mean_uplift(pred, t, y)
    return float(_trapz(u * frac, frac))


def mean_auuc(pred, t, y):
    """This repo's unweighted prefix-mean AUUC (without the ATE/2 shift, which is
    model-invariant and irrelevant to rankings)."""
    frac, u = prefix_mean_uplift(pred, t, y)
    return float(_trapz(u, frac))


def _boot_ci(vals, n=10000):
    rng = np.random.default_rng(SEED)
    v = np.asarray(vals, float)
    v = v[~np.isnan(v)]
    boot = np.array([rng.choice(v, len(v), replace=True).mean() for _ in range(n)])
    return float(v.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def validate_against_causalml(files, n_folds=6):
    """Rank-agreement of gain_auuc with causalml.metrics.auuc_score on identical folds."""
    try:
        from causalml.metrics import auuc_score
    except ImportError:
        print("  causalml unavailable; skipping cross-validation of the implementation")
        return None
    ours, theirs = [], []
    done = 0
    for f in files:
        d = pd.read_parquet(f)
        for (_, _), g in d.groupby(["seed_idx", "fold_idx"]):
            df = pd.DataFrame(
                {
                    "y": g["y"].to_numpy(),
                    "w": g["t"].to_numpy().astype(int),
                    "score": g["pred"].to_numpy(),
                }
            )
            try:
                cm = auuc_score(df, outcome_col="y", treatment_col="w", normalize=False)
                theirs.append(float(cm["score"]))
            except Exception:
                continue
            ours.append(gain_auuc(g["pred"].to_numpy(), g["t"].to_numpy(), g["y"].to_numpy()))
            done += 1
            if done >= n_folds:
                r = spearmanr(ours, theirs).correlation
                return float(r)
    return None


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    files = sorted(glob.glob("results_r4pred/predictions/ihdp_s*__pred.parquet"))
    if not files:
        print("skipping gain_auuc_check: prediction stores absent (make repro-r4pred)")
        if not OUT.exists():
            OUT.write_text("% gain_auuc_check skipped: prediction stores absent\n")
        return

    rho_check = validate_against_causalml(files)
    if rho_check is not None:
        print(
            f"implementation check vs causalml.auuc_score on identical folds: "
            f"Spearman {rho_check:+.2f}"
        )

    # Per (realization, model): fold-mean of each metric + fold-mean sqrt(PEHE).
    reals = sorted({re.search(r"(ihdp_s\d+)__", f).group(1) for f in files})
    rows = []
    for nm in reals:
        per_model = {}
        for f in glob.glob(f"results_r4pred/predictions/{nm}__*__pred.parquet"):
            d = pd.read_parquet(f)
            m = d["model"].iloc[0]
            if m not in MODELS6:
                continue
            ga, ma, pe = [], [], []
            for (_, _), g in d.groupby(["seed_idx", "fold_idx"]):
                p, t, y = (g["pred"].to_numpy(), g["t"].to_numpy(), g["y"].to_numpy())
                ga.append(gain_auuc(p, t, y))
                ma.append(mean_auuc(p, t, y))
                pe.append(float(np.sqrt(np.mean((p - g["ite"].to_numpy()) ** 2))))
            per_model[m] = (np.mean(ga), np.mean(ma), np.mean(pe))
        if len(per_model) < 4:
            continue
        mods = sorted(per_model)
        ga = [per_model[m][0] for m in mods]
        ma = [per_model[m][1] for m in mods]
        pe = [per_model[m][2] for m in mods]
        rows.append(
            dict(
                real=nm,
                gain=spearmanr(ga, [-x for x in pe]).correlation,
                mean=spearmanr(ma, [-x for x in pe]).correlation,
            )
        )
    df = pd.DataFrame(rows)
    g = _boot_ci(df["gain"])
    m = _boot_ci(df["mean"])
    d = _boot_ci((df["mean"] - df["gain"]).to_numpy())
    print(f"\n=== IHDP, {len(df)} realizations, identical stored predictions ===")
    print(
        f"  rho(k-weighted / library-family AUUC, -sqrtPEHE) = {g[0]:+.2f} [{g[1]:+.2f},{g[2]:+.2f}]"
    )
    print(
        f"  rho(unweighted / this repo's AUUC,   -sqrtPEHE) = {m[0]:+.2f} [{m[1]:+.2f},{m[2]:+.2f}]"
    )
    print(
        f"  paired unweighted-over-k-weighted gap           = {d[0]:+.2f} [{d[1]:+.2f},{d[2]:+.2f}]"
        f"  (positive on {int(((df['mean'] - df['gain']) > 0).sum())}/{len(df)})"
    )

    def f(x):
        return f"{x:+.2f}"

    OUT.write_text(
        "\n".join(
            [
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
                (
                    f"\\newcommand{{\\gainCmlRho}}{{{rho_check:+.2f}}}"
                    if rho_check is not None
                    else "\\newcommand{\\gainCmlRho}{n/a}"
                ),
            ]
        )
        + "\n"
    )
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
