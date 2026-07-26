"""Direct policy-selection regret for F2 (reviewer #8), with an HONEST-SPLIT design
that removes the winner's curse (reviewer round 2, #1).

A model-rank correlation is one step removed from what a practitioner loses. We therefore
measure the *policy regret* of selecting a model by each ranking metric. The naive version
(select the metric-winner and the risk-winner from the same averaged test results) is
optimistically biased: the minimum of several noisy policy-risk estimates is a
winner's-curse quantity, so even an uninformative metric shows positive regret.

We mitigate this with a cross-repeat selection/evaluation rotation. Jobs is evaluated over
3 repeated-CV seeds; we rotate the held-out repetition. For each realization and each
held-out repetition:

    selection repetitions = the other two seeds
    m* = argmax_model  metric(model)          (chosen on SELECTION repetitions)
    r* = argmin_model  R_policy(model)         (chosen on SELECTION repetitions)
    Regret_metric = R_policy^eval(m*) - R_policy^eval(r*)   (both evaluated on HELD-OUT rep.)

Both the metric-winner and the reference (risk) winner are chosen on the selection
repetitions and both evaluated on the held-out repetition, so neither can benefit from
reusing the same partition-level estimates. IMPORTANT LIMITATION (stated in the paper):
repeated-CV seeds re-partition the SAME underlying Jobs observations, so selection and
evaluation estimates are not statistically independent; this is cross-repeat
benchmark-selection regret, NOT evaluation on a new deployment sample. We average the three
rotations within a realization, then cluster-bootstrap over the 10 realizations. We also
report the random-selection baseline (expected regret of picking a model uniformly at
random, same rotation) AND the paired per-realization difference (metric regret minus
random-baseline regret) with a cluster-bootstrap CI, so "comparable to random selection" is
a tested statement, not a numerical resemblance. Within-margin rates are reported at three
margins (0.005/0.01/0.02 risk = 0.5/1/2 percentage points of the binary employment outcome)
to avoid a favourable-cutoff appearance. Seed fixed (42). Emits tables/tab10_regret.tex.

R_policy here is the RCT-estimated policy risk (IPW value on the randomized experimental
subset; Sec.~\\ref{sec:methods}), not a literal oracle.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

JOBS = [f"jobs_s{i}" for i in range(10)]
METRICS = [("qini", "Qini"), ("auuc", "AUUC"), ("uplift_at_k", "Uplift@k")]
SEEDS = [0, 1, 2]
MARGIN = 0.01  # practical-equivalence margin on policy risk
SEED = 42


def _boot_ci(v, n=10000):
    rng = np.random.default_rng(SEED)
    v = np.asarray(v, dtype=float)
    b = np.array([rng.choice(v, len(v), replace=True).mean() for _ in range(n)])
    return float(v.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def _agg(df):
    return df.groupby("model").mean(numeric_only=True)


def main(results_dir="results", out_dir="results"):
    raw = pd.read_parquet(Path(results_dir) / "master_raw.parquet")
    raw = raw[(raw.status == "ok") & (raw.dataset.isin(JOBS)) & raw.jobs_policy_risk.notna()]

    MARGINS = (0.005, 0.01, 0.02)
    # per-realization regret for each metric (averaged over the 3 repetition rotations)
    per_ds = {label: [] for _, label in METRICS}
    base_ds = []
    within = {label: {mg: [] for mg in MARGINS} for _, label in METRICS}
    best_range = []
    for ds in JOBS:
        d = raw[raw.dataset == ds]
        rot = {label: [] for _, label in METRICS}
        rot_base = []
        rot_within = {label: {mg: [] for mg in MARGINS} for _, label in METRICS}
        for ev_seed in SEEDS:
            sel = _agg(d[d.seed_idx != ev_seed])
            ev = _agg(d[d.seed_idx == ev_seed])
            common = sel.index.intersection(ev.index)
            if len(common) < 4:
                continue
            risk_sel = sel.loc[common, "jobs_policy_risk"]
            risk_ev = ev.loc[common, "jobs_policy_risk"]
            r_star = risk_sel.idxmin()  # reference winner (selection repetitions)
            best_range.append(float(risk_ev.min()))
            # random-selection baseline: expected regret of a uniformly random model
            rot_base.append(float(risk_ev.mean() - risk_ev[r_star]))
            for col, label in METRICS:
                m_star = sel.loc[common, col].idxmax()  # metric winner (selection reps)
                reg = float(risk_ev[m_star] - risk_ev[r_star])
                rot[label].append(reg)
                for mg in MARGINS:
                    rot_within[label][mg].append(int(reg <= mg))
        base_ds.append(np.mean(rot_base))
        for _, label in METRICS:
            if rot[label]:
                per_ds[label].append(np.mean(rot[label]))
                for mg in MARGINS:
                    within[label][mg].append(np.mean(rot_within[label][mg]))

    rows = {}
    for _, label in METRICS:
        v = np.array(per_ds[label])
        m, lo, hi = _boot_ci(v)
        # paired per-realization difference vs the random-selection baseline
        diff = v - np.array(base_ds)
        dm, dlo, dhi = _boot_ci(diff)
        rows[label] = dict(
            mean=m,
            lo=lo,
            hi=hi,
            median=float(np.median(v)),
            n=len(v),
            within={mg: float(np.mean(within[label][mg])) for mg in MARGINS},
            dmean=dm,
            dlo=dlo,
            dhi=dhi,
        )
    br = (min(best_range), max(best_range))
    base_mean = float(np.mean(base_ds))

    print("=== F2 cross-repeat selection/evaluation regret on Jobs (RCT-estimated risk) ===")
    for label, r in rows.items():
        w = "  ".join(f"@{mg}:{r['within'][mg]:.0%}" for mg in MARGINS)
        print(
            f"  {label:9s}: mean={r['mean']:+.3f} [{r['lo']:+.3f},{r['hi']:+.3f}] "
            f"median={r['median']:+.3f} | vs-random diff={r['dmean']:+.4f} "
            f"[{r['dlo']:+.4f},{r['dhi']:+.4f}] | within {w} | n={r['n']}"
        )
    print(f"  random-selection baseline regret: {base_mean:.3f}")
    print(f"  best-achievable risk range: {br[0]:.3f}-{br[1]:.3f}")

    q, a, u = rows["Qini"], rows["AUUC"], rows["Uplift@k"]
    # relative regret vs best-achievable risk midpoint, for the abstract range
    mid = 0.5 * (br[0] + br[1])
    rel_lo = min(q["mean"], a["mean"], u["mean"]) / mid * 100
    rel_hi = max(q["mean"], a["mean"], u["mean"]) / mid * 100
    macros = [
        r"% auto-generated by scripts/m2_regret.py (cross-repeat selection/evaluation rotation)",
        f"\\newcommand{{\\regQini}}{{{q['mean']:.3f}}}",
        f"\\newcommand{{\\regQiniLo}}{{{q['lo']:.3f}}}",
        f"\\newcommand{{\\regQiniHi}}{{{q['hi']:.3f}}}",
        f"\\newcommand{{\\regAuuc}}{{{a['mean']:.3f}}}",
        f"\\newcommand{{\\regAuucLo}}{{{a['lo']:.3f}}}",
        f"\\newcommand{{\\regAuucHi}}{{{a['hi']:.3f}}}",
        f"\\newcommand{{\\regUplift}}{{{u['mean']:.3f}}}",
        f"\\newcommand{{\\regUpliftLo}}{{{u['lo']:.3f}}}",
        f"\\newcommand{{\\regUpliftHi}}{{{u['hi']:.3f}}}",
        f"\\newcommand{{\\regLo}}{{{min(q['mean'],a['mean'],u['mean']):.3f}}}",
        f"\\newcommand{{\\regHi}}{{{max(q['mean'],a['mean'],u['mean']):.3f}}}",
        f"\\newcommand{{\\regRelLo}}{{{rel_lo:.0f}}}",
        f"\\newcommand{{\\regRelHi}}{{{rel_hi:.0f}}}",
        f"\\newcommand{{\\regBase}}{{{base_mean:.3f}}}",
        f"\\newcommand{{\\regDiffLo}}{{{q['dlo']:+.3f}}}",
        f"\\newcommand{{\\regDiffHi}}{{{q['dhi']:+.3f}}}",
        f"\\newcommand{{\\regDiffAuucLo}}{{{a['dlo']:+.3f}}}",
        f"\\newcommand{{\\regDiffAuucHi}}{{{a['dhi']:+.3f}}}",
        f"\\newcommand{{\\regDiffUpliftLo}}{{{u['dlo']:+.3f}}}",
        f"\\newcommand{{\\regDiffUpliftHi}}{{{u['dhi']:+.3f}}}",
        f"\\newcommand{{\\regBestLo}}{{{br[0]:.2f}}}",
        f"\\newcommand{{\\regBestHi}}{{{br[1]:.2f}}}",
        f"\\newcommand{{\\regN}}{{{q['n']}}}",
        f"\\newcommand{{\\regMedian}}{{{q['median']:.3f}}}",
        f"\\newcommand{{\\regWithinA}}{{{q['within'][0.005]*100:.0f}}}",
        f"\\newcommand{{\\regWithinB}}{{{q['within'][0.01]*100:.0f}}}",
        f"\\newcommand{{\\regWithinC}}{{{q['within'][0.02]*100:.0f}}}",
    ]
    # ---- XGBoost replication of the FULL F2 (reviewer: robustness must cover the
    # policy mismatch, not only ranking-metric mutual agreement) ----
    xgb_path = Path("results_xgb") / "master_raw.parquet"
    if xgb_path.exists():
        from scipy.stats import spearmanr

        xr = pd.read_parquet(xgb_path)
        xr = xr[(xr.status == "ok") & (xr.dataset.isin(JOBS)) & xr.jobs_policy_risk.notna()]
        xrho, xreg = {}, {}
        for col, label in METRICS:
            rhos = []
            for ds in JOBS:
                g = _agg(xr[xr.dataset == ds])
                d2 = g[[col, "jobs_policy_risk"]].dropna()
                if len(d2) >= 4:
                    r = spearmanr(d2[col], -d2["jobs_policy_risk"]).correlation
                    if not np.isnan(r):
                        rhos.append(r)
            xrho[label] = _boot_ci(rhos)
            perx = []
            for ds in JOBS:
                d2 = xr[xr.dataset == ds]
                rot = []
                for ev_seed in SEEDS:
                    sel = _agg(d2[d2.seed_idx != ev_seed])
                    ev = _agg(d2[d2.seed_idx == ev_seed])
                    c = sel.index.intersection(ev.index)
                    if len(c) < 4:
                        continue
                    rev = ev.loc[c, "jobs_policy_risk"]
                    rot.append(
                        float(
                            rev[sel.loc[c, col].idxmax()]
                            - rev[sel.loc[c, "jobs_policy_risk"].idxmin()]
                        )
                    )
                if rot:
                    perx.append(np.mean(rot))
            xreg[label] = _boot_ci(perx)
        print("=== XGBoost F2 replication ===")
        for label in xrho:
            print(
                f"  {label:9s}: rho(metric,-risk)={xrho[label][0]:+.2f} "
                f"[{xrho[label][1]:+.2f},{xrho[label][2]:+.2f}] | "
                f"regret={xreg[label][0]:+.3f} [{xreg[label][1]:+.3f},{xreg[label][2]:+.3f}]"
            )
        regs = [xreg[label] for label in ("Qini", "AUUC", "Uplift@k")]
        macros += [
            f"\\newcommand{{\\xregQiniRho}}{{{xrho['Qini'][0]:+.2f}}}",
            f"\\newcommand{{\\xregQiniRhoLo}}{{{xrho['Qini'][1]:+.2f}}}",
            f"\\newcommand{{\\xregQiniRhoHi}}{{{xrho['Qini'][2]:+.2f}}}",
            f"\\newcommand{{\\xregLo}}{{{min(r[0] for r in regs):.3f}}}",
            f"\\newcommand{{\\xregHi}}{{{max(r[0] for r in regs):.3f}}}",
            f"\\newcommand{{\\xregCiLo}}{{{min(r[1] for r in regs):.3f}}}",
        ]

        # ---- reviewer-requested LightGBM-vs-XGBoost audit table ----
        def _counts(rr):
            j = rr[rr.dataset.isin(JOBS)]
            jok = j[(j.status == "ok") & j.jobs_policy_risk.notna()]
            comp = [jok[jok.dataset == ds].model.nunique() for ds in JOBS]
            return j.dataset.nunique(), j.model.nunique(), min(comp), max(comp)

        def _qini_rho(rr):  # LightGBM per-realization rho(Qini, -policy risk)
            rr = rr[(rr.status == "ok") & (rr.dataset.isin(JOBS)) & rr.jobs_policy_risk.notna()]
            rhos = []
            for ds in JOBS:
                g = _agg(rr[rr.dataset == ds])
                dd = g[["qini", "jobs_policy_risk"]].dropna()
                if len(dd) >= 4:
                    r = spearmanr(dd["qini"], -dd["jobs_policy_risk"]).correlation
                    if not np.isnan(r):
                        rhos.append(r)
            return _boot_ci(rhos)

        lgb_c, xgb_c = _counts(raw), _counts(xr)
        lgb_qrho = _qini_rho(raw)

        def cell(t):  # (mean, lo, hi) -> "0.028 [0.012, 0.045]"
            return f"{t[0]:+.3f} [{t[1]:+.3f}, {t[2]:+.3f}]"

        def tup(d):  # rows[label] dict -> (mean, lo, hi)
            return (d["mean"], d["lo"], d["hi"])

        doc = [
            r"\begin{table}[ht]",
            r"\centering",
            r"\caption{Audit of the F2 Jobs policy-regret experiment under both base learners "
            r"(Section~\ref{sec:m2}). Regret is the cross-repeat selection/evaluation rotation "
            r"regret (mean over the ten Jobs splits, 95\% cluster bootstrap); "
            r"$\rho(\text{Qini},-R_{\mathrm{policy}})$ is the mean per-split Spearman "
            r"correlation with policy value. Both halves of F2 --- the ranking metrics' negative "
            r"association with policy value and the positive selection regret --- replicate "
            r"under XGBoost (indeed Qini's correlation is no longer borderline there).}",
            r"\label{tab:regretaudit}",
            r"\small",
            r"\begin{tabular}{lll}",
            r"\toprule",
            r"Item & LightGBM & XGBoost \\",
            r"\midrule",
            f"Jobs splits & {lgb_c[0]} & {xgb_c[0]} \\\\",
            f"Candidate models & {lgb_c[1]} & {xgb_c[1]} \\\\",
            f"Completed models / split & {lgb_c[2]}--{lgb_c[3]} & {xgb_c[2]}--{xgb_c[3]} \\\\",
            f"Qini regret & {cell(tup(rows['Qini']))} & {cell(xreg['Qini'])} \\\\",
            f"AUUC regret & {cell(tup(rows['AUUC']))} & {cell(xreg['AUUC'])} \\\\",
            f"Uplift-at-$k$ regret & {cell(tup(rows['Uplift@k']))} & {cell(xreg['Uplift@k'])} \\\\",
            (
                f"$\\rho(\\text{{Qini}},-R_{{\\mathrm{{policy}}}})$ & "
                f"{lgb_qrho[0]:+.2f} [{lgb_qrho[1]:+.2f}, {lgb_qrho[2]:+.2f}] & "
                f"{xrho['Qini'][0]:+.2f} [{xrho['Qini'][1]:+.2f}, {xrho['Qini'][2]:+.2f}] \\\\"
            ),
            r"Bootstrap unit & split & split \\",
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
        (Path(out_dir) / "tables" / "tab11_regret_doc.tex").write_text("\n".join(doc))
        print(f"  Saved: {Path(out_dir)/'tables'/'tab11_regret_doc.tex'}")

    outp = Path(out_dir) / "tables" / "tab10_regret.tex"
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text("\n".join(macros) + "\n")
    print(f"  Saved: {outp}")


if __name__ == "__main__":
    main()
