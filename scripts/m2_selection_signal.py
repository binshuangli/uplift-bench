"""Selection-signal analysis for F2 on Jobs (reviewer R4, W2/Q2).

The F2 regret result ("metric-based model selection is not statistically
distinguishable from random selection") admits two readings:

  (a) ranking metrics specifically miss the deployment objective, or
  (b) Jobs simply lacks cross-repetition selection signal under the
      sign-threshold objective, in which case NO selector could beat random
      and the finding says little about ranking metrics per se.

This script separates them with two analyses on the SAME cross-repeat
rotation design as scripts/m2_regret.py (select on two repeated-CV seeds,
evaluate on the held-out seed; cluster bootstrap over the 10 realizations):

1. SELECTOR LADDER. Every selector is scored identically by its evaluated
   policy-risk advantage over random selection on the held-out repetition,
   gain = mean_models(risk_ev) - risk_ev(selected):
     - risk selector (semi-oracle): argmin policy risk on the SELECTION
       repetitions -- uses the deployment objective itself, out-of-sample.
       If even this cannot beat random, Jobs has no cross-repetition
       selection signal (reading b).
     - metric selectors (Qini / AUUC / Uplift@k) for comparison.
     - oracle bound: min over models of risk_ev (selects on the evaluation
       repetition itself; winner's-curse upper bound, for scale only).

2. REFERENCE-SELECTION STABILITY. Across the 3 rotations within each
   realization: how often the risk-based reference winner r* is the same
   model (modal agreement), how many distinct winners appear, and the
   spread of its evaluated risk. High instability = the reference itself
   is noisy, tempering per-metric regret interpretation.

Reads results/master_raw.parquet (existing artifacts; no model re-runs).
Emits results/tables/tab12_selection_signal.tex (macros) and prints a
summary. Seed fixed (42).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

JOBS = [f"jobs_s{i}" for i in range(10)]
METRICS = [("qini", "Qini"), ("auuc", "AUUC"), ("uplift_at_k", "Uplift@k")]
SEEDS = [0, 1, 2]
SEED = 42


def _boot_ci(v, n=10000):
    rng = np.random.default_rng(SEED)
    v = np.asarray(v, dtype=float)
    b = np.array([rng.choice(v, len(v), replace=True).mean() for _ in range(n)])
    return float(v.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def _agg(df):
    return df.groupby("model").mean(numeric_only=True)


def main(results_dir="results", out_dir="results"):
    from pathlib import Path as _P
    if not _P("results/master_raw.parquet").exists():
        print("skipping m2_selection_signal: results/master_raw.parquet missing "
              "(run `make repro-main` first; see README)")
        return
    raw = pd.read_parquet(Path(results_dir) / "master_raw.parquet")
    raw = raw[(raw.status == "ok") & (raw.dataset.isin(JOBS)) & raw.jobs_policy_risk.notna()]

    selectors = ["Risk (semi-oracle)"] + [label for _, label in METRICS]
    gains = {s: [] for s in selectors}          # per-realization gain vs random
    oracle_gain = []                            # winner's-curse upper bound
    ref_modal_agree = []                        # rotation-stability of r*
    ref_n_distinct = []
    ref_risk_spread = []

    for ds in JOBS:
        d = raw[raw.dataset == ds]
        rot_gain = {s: [] for s in selectors}
        rot_oracle = []
        r_stars, r_star_risks = [], []
        for ev_seed in SEEDS:
            sel = _agg(d[d.seed_idx != ev_seed])
            ev = _agg(d[d.seed_idx == ev_seed])
            common = sel.index.intersection(ev.index)
            if len(common) < 4:
                continue
            risk_sel = sel.loc[common, "jobs_policy_risk"]
            risk_ev = ev.loc[common, "jobs_policy_risk"]
            random_risk = float(risk_ev.mean())

            r_star = risk_sel.idxmin()
            r_stars.append(r_star)
            r_star_risks.append(float(risk_ev[r_star]))
            rot_gain["Risk (semi-oracle)"].append(random_risk - float(risk_ev[r_star]))
            for col, label in METRICS:
                m_star = sel.loc[common, col].idxmax()
                rot_gain[label].append(random_risk - float(risk_ev[m_star]))
            rot_oracle.append(random_risk - float(risk_ev.min()))

        for s in selectors:
            if rot_gain[s]:
                gains[s].append(np.mean(rot_gain[s]))
        if rot_oracle:
            oracle_gain.append(np.mean(rot_oracle))
        if r_stars:
            vals, counts = np.unique(r_stars, return_counts=True)
            ref_modal_agree.append(counts.max() / len(r_stars))
            ref_n_distinct.append(len(vals))
            ref_risk_spread.append(float(np.max(r_star_risks) - np.min(r_star_risks)))

    print("=== Selector ladder: evaluated policy-risk gain over random selection ===")
    print("    (positive = beats random; cluster bootstrap over realizations)")
    res = {}
    for s in selectors:
        m, lo, hi = _boot_ci(gains[s])
        res[s] = (m, lo, hi)
        star = " <-- selection signal exists" if lo > 0 else ""
        print(f"  {s:20s}: gain={m:+.4f} [{lo:+.4f},{hi:+.4f}] n={len(gains[s])}{star}")
    om, olo, ohi = _boot_ci(oracle_gain)
    print(f"  {'Oracle bound':20s}: gain={om:+.4f} [{olo:+.4f},{ohi:+.4f}] "
          "(selects on eval seed; winner's-curse scale reference)")

    print("=== Reference-selection stability across the 3 rotations ===")
    ma, malo, mahi = _boot_ci(ref_modal_agree)
    nd = float(np.mean(ref_n_distinct))
    sp, splo, sphi = _boot_ci(ref_risk_spread)
    print(f"  modal agreement of r*: {ma:.0%} [{malo:.0%},{mahi:.0%}]")
    print(f"  distinct r* per realization (of 3 rotations): mean {nd:.2f}")
    print(f"  spread of evaluated risk of r*: {sp:.4f} [{splo:.4f},{sphi:.4f}]")

    risk_gain = res["Risk (semi-oracle)"]
    verdict = (
        "signal_exists"
        if risk_gain[1] > 0
        else ("no_signal" if risk_gain[2] < 0 else "indeterminate")
    )
    print(f"  VERDICT: cross-repetition selection signal on Jobs: {verdict}")

    macros = [
        r"% auto-generated by scripts/m2_selection_signal.py (R4 W2/Q2)",
        f"\\newcommand{{\\selRiskGain}}{{{risk_gain[0]:+.4f}}}",
        f"\\newcommand{{\\selRiskGainLo}}{{{risk_gain[1]:+.4f}}}",
        f"\\newcommand{{\\selRiskGainHi}}{{{risk_gain[2]:+.4f}}}",
        f"\\newcommand{{\\selQiniGain}}{{{res['Qini'][0]:+.4f}}}",
        f"\\newcommand{{\\selQiniGainLo}}{{{res['Qini'][1]:+.4f}}}",
        f"\\newcommand{{\\selQiniGainHi}}{{{res['Qini'][2]:+.4f}}}",
        f"\\newcommand{{\\selAuucGain}}{{{res['AUUC'][0]:+.4f}}}",
        f"\\newcommand{{\\selAuucGainLo}}{{{res['AUUC'][1]:+.4f}}}",
        f"\\newcommand{{\\selAuucGainHi}}{{{res['AUUC'][2]:+.4f}}}",
        f"\\newcommand{{\\selUpliftGain}}{{{res['Uplift@k'][0]:+.4f}}}",
        f"\\newcommand{{\\selUpliftGainLo}}{{{res['Uplift@k'][1]:+.4f}}}",
        f"\\newcommand{{\\selUpliftGainHi}}{{{res['Uplift@k'][2]:+.4f}}}",
        f"\\newcommand{{\\selOracleGain}}{{{om:+.4f}}}",
        f"\\newcommand{{\\selRefModalAgree}}{{{ma*100:.0f}}}",
        f"\\newcommand{{\\selRefDistinct}}{{{nd:.2f}}}",
        f"\\newcommand{{\\selRefSpread}}{{{sp:.4f}}}",
        f"\\newcommand{{\\selN}}{{{len(gains['Risk (semi-oracle)'])}}}",
    ]
    outp = Path(out_dir) / "tables" / "tab12_selection_signal.tex"
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text("\n".join(macros) + "\n")
    print(f"  Saved: {outp}")


if __name__ == "__main__":
    main()
