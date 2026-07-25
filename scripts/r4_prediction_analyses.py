"""R4 prediction-level analyses (reviewer W3/W4/W6, questions Q1/Q3 + budget grid).

Consumes the per-unit prediction store written by `run_bench.py --save-predictions`
(results_r4pred/predictions/, optionally results_acic/predictions/) and produces
three analyses on IDENTICAL predictions:

A. Implementation-variant comparison (W4/Q3): the paper's canonical cumulative-gain
   Qini vs `causalml.metrics.qini_score` (v0.16.0, default normalize=True — what a
   practitioner gets). Reports per-fold score correlation, per-dataset model-rank
   correlation, winner agreement, and each variant's per-dataset Spearman rho of the
   model ranking against effect accuracy (-sqrt(PEHE)) on the continuous datasets.
   Verdict: does the F1 conclusion change under causalml's shipped implementation?

B. RATE (W3/Q1): rank-weighted ATE (Yadlowsky et al.), AUTOC and Qini weightings,
   with IPW effect scores from the stored fold propensities. Same rank-vs-(-PEHE)
   aggregation as the F1 analysis. Verdict: does RATE fix F1?

C. Budget grid (W6): uplift-at-k, policy-value-at-k for k in {0.1,0.2,0.3,0.5} plus
   the area under the policy-value-vs-budget curve (PV-AUC); per-dataset ranking
   stability across budgets, and the F2 cross-repeat selection/evaluation regret on
   Jobs re-run with each budget's uplift-at-k as the selector (same rotation design
   as scripts/m2_regret.py). Verdict: does F2 persist across budgets?

Outputs: printed report + results/tables/tab13_r4_analyses.tex (macros).
Usage: python scripts/r4_prediction_analyses.py [--pred-dirs results_r4pred/predictions ...]
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from uplift_bench.metrics.causal import jobs_policy_risk, pehe  # noqa: E402
from uplift_bench.metrics.ranking import (  # noqa: E402
    auuc,
    policy_value_at_k,
    qini_coefficient,
    uplift_at_k,
)
from uplift_bench.metrics.rate import rate  # noqa: E402

SEED = 42
KGRID = (0.1, 0.2, 0.3, 0.5)
SEEDS = (0, 1, 2)


def _boot_ci(v, n=10000):
    rng = np.random.default_rng(SEED)
    v = np.asarray(v, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return float("nan"), float("nan"), float("nan")
    b = np.array([rng.choice(v, len(v), replace=True).mean() for _ in range(n)])
    return float(v.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def _causalml_qini(pred, t, y):
    from causalml.metrics import qini_score

    df = pd.DataFrame({"y": y, "w": t, "model": pred})
    # default normalize=True: the out-of-the-box behaviour a practitioner gets
    return float(qini_score(df, outcome_col="y", treatment_col="w").loc["model"])


def load_fold_metrics(pred_dirs: list[str]) -> pd.DataFrame:
    """Per-(dataset, model, seed, fold) metric table from stored predictions."""
    rows = []
    files = sorted(f for d in pred_dirs for f in glob.glob(str(Path(d) / "*__pred.parquet")))
    print(f"loading {len(files)} prediction files from {len(pred_dirs)} dir(s)")
    for f in files:
        df = pd.read_parquet(f)
        for (ds, model, seed, fold), g in df.groupby(
            ["dataset", "model", "seed_idx", "fold_idx"]
        ):
            pred = g["pred"].to_numpy()
            t = g["t"].to_numpy()
            y = g["y"].to_numpy()
            e = g["propensity"].to_numpy()
            row = {
                "dataset": ds,
                "model": model,
                "seed_idx": seed,
                "fold_idx": fold,
                "qini_canonical": qini_coefficient(pred, t, y, normalize=False),
                # AUUC on the identical frame, so the RATE-vs-AUUC appendix comparison is
                # computed from the same stored predictions rather than a separate run.
                "auuc": auuc(pred, t, y),
                "rate_autoc": rate(pred, t, y, e, weighting="autoc"),
                "rate_qini": rate(pred, t, y, e, weighting="qini"),
            }
            try:
                row["qini_causalml"] = _causalml_qini(pred, t, y)
            except Exception:
                row["qini_causalml"] = np.nan
            for k in KGRID:
                row[f"uplift_at_{k}"] = uplift_at_k(pred, t, y, k=k)
                row[f"pv_at_{k}"] = policy_value_at_k(pred, t, y, k=k)
            pvs = [row[f"pv_at_{k}"] for k in KGRID]
            row["pv_auc"] = float(np.trapz(pvs, KGRID)) / (KGRID[-1] - KGRID[0])
            if g["ite"].notna().all():
                row["pehe"] = pehe(pred, g["ite"].to_numpy())
            if "experimental" in g.columns and g["experimental"].notna().all():
                row["jobs_policy_risk"] = jobs_policy_risk(
                    pred, y, t, g["experimental"].to_numpy()
                )
            rows.append(row)
    return pd.DataFrame(rows)


def rank_vs_pehe(fold_df: pd.DataFrame, col: str, datasets: list[str]) -> tuple:
    """Mean per-dataset Spearman rho of the model ranking by `col` vs -sqrt(PEHE)."""
    rhos = []
    for ds in datasets:
        g = fold_df[fold_df.dataset == ds].groupby("model")[[col, "pehe"]].mean().dropna()
        if len(g) >= 4:
            r = spearmanr(g[col], -g["pehe"]).correlation
            if not np.isnan(r):
                rhos.append(r)
    return _boot_ci(rhos) + (len(rhos),)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--pred-dirs",
        nargs="+",
        default=["results_r4pred/predictions"],
        help="prediction store directories",
    )
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--cache", default=None, help="optional parquet cache for fold metrics")
    args = ap.parse_args()

    if args.cache and Path(args.cache).exists():
        fold = pd.read_parquet(args.cache)
        print(f"loaded cached fold metrics: {fold.shape}")
    else:
        fold = load_fold_metrics(args.pred_dirs)
        if args.cache:
            fold.to_parquet(args.cache, index=False)
    print(f"fold metrics: {fold.shape}; datasets: {fold.dataset.nunique()}")

    cont = sorted(
        d
        for d in fold.dataset.unique()
        if d.startswith(("ihdp", "synthetic", "revenue", "acic"))
        and fold.loc[fold.dataset == d, "pehe"].notna().any()
    )
    # `synthetic` is binary (configs/dataset/synthetic.yaml), so the continuous-outcome
    # group is IHDP only; synthetic is reported separately where relevant.
    ihdp_synth = [d for d in cont if d.startswith("ihdp")]
    acic = [d for d in cont if d.startswith("acic")]
    jobs = sorted(d for d in fold.dataset.unique() if d.startswith("jobs"))
    macros = ["% auto-generated by scripts/r4_prediction_analyses.py"]

    # ---------------- A. causalml implementation variant (W4/Q3) ----------------
    print("\n=== A. Canonical Qini vs causalml qini_score (identical predictions) ===")
    ab = fold.dropna(subset=["qini_causalml"])
    if len(ab):
        sc = spearmanr(ab.qini_canonical, ab.qini_causalml).correlation
        print(f"  per-fold score Spearman (all folds pooled): {sc:+.3f} (n={len(ab)})")
        rank_rhos, winner_agree = [], []
        for ds in ihdp_synth:
            g = (
                ab[ab.dataset == ds]
                .groupby("model")[["qini_canonical", "qini_causalml"]]
                .mean()
                .dropna()
            )
            if len(g) >= 4:
                rank_rhos.append(
                    spearmanr(g.qini_canonical, g.qini_causalml).correlation
                )
                winner_agree.append(
                    int(g.qini_canonical.idxmax() == g.qini_causalml.idxmax())
                )
        rr, rlo, rhi = _boot_ci(rank_rhos)
        wa = float(np.mean(winner_agree)) if winner_agree else float("nan")
        print(f"  per-dataset model-rank Spearman: {rr:+.3f} [{rlo:+.3f},{rhi:+.3f}] "
              f"(n={len(rank_rhos)})")
        print(f"  winner agreement: {wa:.0%}")
        m_can = rank_vs_pehe(fold, "qini_canonical", ihdp_synth)
        m_cml = rank_vs_pehe(ab, "qini_causalml", ihdp_synth)
        print("  F1 check, rho(rank, -PEHE) on IHDP:")
        print(f"    canonical Qini: {m_can[0]:+.3f} [{m_can[1]:+.3f},{m_can[2]:+.3f}] (n={m_can[3]})")
        print(f"    causalml Qini : {m_cml[0]:+.3f} [{m_cml[1]:+.3f},{m_cml[2]:+.3f}] (n={m_cml[3]})")
        macros += [
            f"\\newcommand{{\\cmlScoreRho}}{{{sc:+.2f}}}",
            f"\\newcommand{{\\cmlRankRho}}{{{rr:+.2f}}}",
            f"\\newcommand{{\\cmlRankRhoLo}}{{{rlo:+.2f}}}",
            f"\\newcommand{{\\cmlRankRhoHi}}{{{rhi:+.2f}}}",
            f"\\newcommand{{\\cmlWinnerAgree}}{{{wa*100:.0f}}}",
            f"\\newcommand{{\\cmlMOne}}{{{m_cml[0]:+.2f}}}",
            f"\\newcommand{{\\cmlMOneLo}}{{{m_cml[1]:+.2f}}}",
            f"\\newcommand{{\\cmlMOneHi}}{{{m_cml[2]:+.2f}}}",
            f"\\newcommand{{\\canMOne}}{{{m_can[0]:+.2f}}}",
            f"\\newcommand{{\\canMOneLo}}{{{m_can[1]:+.2f}}}",
            f"\\newcommand{{\\canMOneHi}}{{{m_can[2]:+.2f}}}",
        ]

    # ---------------- B. RATE (W3/Q1) ----------------
    print("\n=== B. Does RATE fix F1? rho(model rank by metric, -PEHE) ===")
    table = {}
    for col, label in [
        ("qini_canonical", "Qini"),
        ("auuc", "AUUC"),
        ("rate_autoc", "RATE-AUTOC"),
        ("rate_qini", "RATE-Qini"),
    ]:
        table[label] = {}
        for name, dss in [("IHDP", ihdp_synth), ("ACIC", acic)]:
            if dss:
                m, lo, hi, n = rank_vs_pehe(fold, col, dss)
                table[label][name] = (m, lo, hi, n)
                print(f"  {label:11s} on {name:15s}: {m:+.3f} [{lo:+.3f},{hi:+.3f}] (n={n})")
    if "RATE-AUTOC" in table and "IHDP" in table["RATE-AUTOC"]:
        ra = table["RATE-AUTOC"]["IHDP"]
        rq = table["RATE-Qini"]["IHDP"]
        macros += [
            f"\\newcommand{{\\rateAutocMOne}}{{{ra[0]:+.2f}}}",
            f"\\newcommand{{\\rateAutocMOneLo}}{{{ra[1]:+.2f}}}",
            f"\\newcommand{{\\rateAutocMOneHi}}{{{ra[2]:+.2f}}}",
            f"\\newcommand{{\\rateQiniMOne}}{{{rq[0]:+.2f}}}",
            f"\\newcommand{{\\rateQiniMOneLo}}{{{rq[1]:+.2f}}}",
            f"\\newcommand{{\\rateQiniMOneHi}}{{{rq[2]:+.2f}}}",
        ]
        if "AUUC" in table and "IHDP" in table["AUUC"]:
            au = table["AUUC"]["IHDP"]
            macros += [
                f"\\newcommand{{\\auucMOne}}{{{au[0]:+.2f}}}",
                f"\\newcommand{{\\auucMOneLo}}{{{au[1]:+.2f}}}",
                f"\\newcommand{{\\auucMOneHi}}{{{au[2]:+.2f}}}",
            ]
    if acic:
        for col, label, mac in [
            ("qini_canonical", "Qini", "acicQini"),
            ("rate_autoc", "RATE-AUTOC", "acicRateAutoc"),
        ]:
            if label in table and "ACIC" in table[label]:
                m, lo, hi, n = table[label]["ACIC"]
                macros += [
                    f"\\newcommand{{\\{mac}MOne}}{{{m:+.2f}}}",
                    f"\\newcommand{{\\{mac}MOneLo}}{{{lo:+.2f}}}",
                    f"\\newcommand{{\\{mac}MOneHi}}{{{hi:+.2f}}}",
                ]
        # AUUC-analogue on ACIC via uplift_at_0.3 for the F1-replication statement
        m_u = rank_vs_pehe(fold, "uplift_at_0.3", acic)
        print(f"  {'Uplift@0.3':11s} on {'ACIC':15s}: {m_u[0]:+.3f} [{m_u[1]:+.3f},{m_u[2]:+.3f}] (n={m_u[3]})")
        macros += [
            f"\\newcommand{{\\acicUpliftMOne}}{{{m_u[0]:+.2f}}}",
            f"\\newcommand{{\\acicUpliftMOneLo}}{{{m_u[1]:+.2f}}}",
            f"\\newcommand{{\\acicUpliftMOneHi}}{{{m_u[2]:+.2f}}}",
            f"\\newcommand{{\\acicN}}{{{m_u[3]}}}",
        ]

    # ---------------- C. Budget grid (W6) ----------------
    print("\n=== C. Budget sensitivity: k in", KGRID, "===")
    # C1: within-dataset ranking stability of uplift@k across budgets
    stab = []
    for ds in ihdp_synth + jobs:
        g = fold[fold.dataset == ds].groupby("model")[
            [f"uplift_at_{k}" for k in KGRID]
        ].mean().dropna()
        if len(g) >= 4:
            for i, ka in enumerate(KGRID):
                for kb in KGRID[i + 1:]:
                    stab.append(
                        spearmanr(g[f"uplift_at_{ka}"], g[f"uplift_at_{kb}"]).correlation
                    )
    st, stlo, sthi = _boot_ci(stab)
    print(f"  uplift@k ranking stability across budgets (pairwise rho): "
          f"{st:+.3f} [{stlo:+.3f},{sthi:+.3f}] (n={len(stab)})")
    macros += [
        f"\\newcommand{{\\kgridStab}}{{{st:+.2f}}}",
        f"\\newcommand{{\\kgridStabLo}}{{{stlo:+.2f}}}",
        f"\\newcommand{{\\kgridStabHi}}{{{sthi:+.2f}}}",
    ]

    # C2: F2 regret on Jobs at each budget (same rotation as m2_regret.py),
    # selector = uplift@k (and PV-AUC), objective = policy risk.
    if jobs:
        have_risk = fold.dropna(subset=["jobs_policy_risk"])
        selectors = [(f"uplift_at_{k}", f"Uplift@{k}") for k in KGRID] + [
            ("pv_auc", "PV-AUC")
        ]
        print("  F2 rotation regret on Jobs by selector:")
        for col, label in selectors:
            per_ds = []
            for ds in jobs:
                d = have_risk[have_risk.dataset == ds]
                rot = []
                for ev_seed in SEEDS:
                    sel = d[d.seed_idx != ev_seed].groupby("model").mean(numeric_only=True)
                    ev = d[d.seed_idx == ev_seed].groupby("model").mean(numeric_only=True)
                    common = sel.index.intersection(ev.index)
                    if len(common) < 4:
                        continue
                    r_star = sel.loc[common, "jobs_policy_risk"].idxmin()
                    m_star = sel.loc[common, col].idxmax()
                    rot.append(
                        float(
                            ev.loc[m_star, "jobs_policy_risk"]
                            - ev.loc[r_star, "jobs_policy_risk"]
                        )
                    )
                if rot:
                    per_ds.append(np.mean(rot))
            m, lo, hi = _boot_ci(per_ds)
            flag = " (CI excludes 0)" if lo > 0 else ""
            print(f"    {label:11s}: regret={m:+.4f} [{lo:+.4f},{hi:+.4f}] n={len(per_ds)}{flag}")
            # LaTeX control sequences must be letters-only: spell out digits.
            safe = (
                label.replace("@", "").replace(".", "").replace("-", "")
                .replace("0", "Zero").replace("1", "One").replace("2", "Two")
                .replace("3", "Three").replace("5", "Five")
            )
            macros += [
                f"\\newcommand{{\\kreg{safe}}}{{{m:+.3f}}}",
                f"\\newcommand{{\\kreg{safe}Lo}}{{{lo:+.3f}}}",
                f"\\newcommand{{\\kreg{safe}Hi}}{{{hi:+.3f}}}",
            ]

    outp = Path(args.out_dir) / "tables" / "tab13_r4_analyses.tex"
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text("\n".join(macros) + "\n")
    print(f"\nSaved: {outp}")


if __name__ == "__main__":
    main()
