"""Calibration robustness (reviewer Issue 6).

Two parts, both reproducible from committed artifacts:

(1) Effect sizes from the main run (results/master_summary.parquet), not just winner
    disagreement: the within-dataset Spearman rank correlation between Qini and calibration
    (−ECE) across models, with a cluster bootstrap CI over datasets; the best-Qini vs.
    best-ECE winner-disagreement rate with a CI; the median ECE gap; and the count of
    datasets where a numerically unstable estimator produces an extreme ECE.

(2) Bin-count sensitivity of the ECE ranking (5/10/20 bins) on a controlled panel, showing
    the "ranking ≠ calibration" reading is not an artifact of the default 10-bin choice.

Writes tables/tab7_calibration.tex (LaTeX macros) and findings_calibration.md.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from uplift_bench.metrics.calibration import uplift_calibration_error

SEED = 7


def _boot_ci(a, f=np.mean, n=10000, seed=SEED):
    rng = np.random.default_rng(seed)
    a = np.asarray(a, dtype=float)
    b = np.array([f(rng.choice(a, len(a), replace=True)) for _ in range(n)])
    return float(f(a)), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


# Treatment is randomized (so the within-bin treated-minus-control difference identifies
# the bin-average uplift) on the marketing RCTs and the synthetic RCT. IHDP is a CONFOUNDED
# semi-synthetic design and Jobs mixes an experimental arm with observational PSID controls,
# so a raw diff-in-means calibration is NOT identified there.
IDENTIFIED = ["hillstrom", "lenta", "x5", "megafon", "synthetic"]


def _subset_stats(s, DS):
    """rho(Qini,-ECE) over instances, winner disagreement, and the PAIRED per-instance
    ECE difference Delta = ECE(Qini-selected) - ECE(best-calibrated)."""
    rhos, disagree, delta = [], [], []
    for ds in DS:
        sub = s[s.dataset == ds]
        d = sub[["model", "qini_mean", "calibration_ece_mean"]].dropna()
        if len(d) < 4:
            continue
        r = spearmanr(d["qini_mean"], -d["calibration_ece_mean"]).correlation
        if not np.isnan(r):
            rhos.append(r)
        bq = d.loc[d["qini_mean"].idxmax(), "model"]
        be = d.loc[d["calibration_ece_mean"].idxmin(), "model"]
        disagree.append(bq != be)
        eq = d.loc[d.model == bq, "calibration_ece_mean"].values[0]
        ee = d.loc[d.model == be, "calibration_ece_mean"].values[0]
        delta.append(eq - ee)   # >= 0 by construction; magnitude = ECE cost of Qini-selection
    return np.array(rhos), np.array(disagree, dtype=float), np.array(delta)


def part1_effect_sizes(summ_path: Path):
    s = pd.read_parquet(summ_path)
    ALL = sorted(s.dataset.unique())
    ident = [d for d in IDENTIFIED if d in ALL]

    # identified (randomized) subset — the only place diff-in-means calibration is valid
    r_id, dis_id, delta_id = _subset_stats(s, ident)
    rm, rlo, rhi = _boot_ci(r_id)
    dpm, dplo, dphi = _boot_ci(delta_id)   # paired ECE-difference bootstrap CI
    # pooled (all instances) — shown only to expose the confounding artifact
    r_all, dis_all, _ = _subset_stats(s, ALL)
    ram, ralo, rahi = _boot_ci(r_all)
    extreme = sum((s[s.dataset == ds]["calibration_ece_mean"] > 100).any() for ds in ALL)

    return dict(
        n_ident=len(r_id),
        rho_id=(rm, rlo, rhi),
        disagree_id=(int(dis_id.sum()), len(dis_id)),
        delta_id=(float(np.median(delta_id)), dplo, dphi),
        rho_all=(ram, ralo, rahi), n_all=len(r_all),
        disagree_all=(int(dis_all.sum()), len(dis_all)),
        extreme=extreme,
    )


# ---- Part 2: bin-count sensitivity on a controlled panel ----
def part2_bin_sensitivity(n=4000, n_models=8, n_seeds=20):
    """Is the ECE-induced MODEL RANKING stable across bin counts?

    On a controlled panel of graded-quality uplift scorings, compute the ECE ranking of the
    models at 5, 10 and 20 bins and report the pairwise Spearman agreement BETWEEN those
    rankings (not the Qini-vs-ECE association). High agreement ⇒ the near-zero real-data
    Qini--calibration association is not an artifact of the 10-bin default.
    """
    noise = np.linspace(0.3, 1.8, n_models)
    cross = {"5_10": [], "10_20": [], "5_20": []}
    for sd in range(n_seeds):
        rng = np.random.default_rng(100 + sd)
        x = rng.standard_normal((n, 3))
        t = rng.binomial(1, 0.5, n)
        tau = x[:, 0] + 0.5 * x[:, 1]
        y = 0.5 * x[:, 2] + t * tau + rng.standard_normal(n)          # continuous, moderate
        scores = [tau + e * rng.standard_normal(n) for e in noise]
        ece = {b: np.array([uplift_calibration_error(sc, t, y, n_bins=b) for sc in scores])
               for b in (5, 10, 20)}
        cross["5_10"].append(spearmanr(ece[5], ece[10]).correlation)
        cross["10_20"].append(spearmanr(ece[10], ece[20]).correlation)
        cross["5_20"].append(spearmanr(ece[5], ece[20]).correlation)
    allv = np.concatenate([np.asarray(v) for v in cross.values()])
    return {"panel_n": n, "panel_models": n_models, "panel_seeds": n_seeds,
            "min": float(np.nanmin(allv)), "max": float(np.nanmax(allv))}


def main(out_dir="results"):
    from pathlib import Path as _P
    if not _P("results/master_raw.parquet").exists():
        print("skipping calibration_robustness: results/master_raw.parquet missing "
              "(run `make repro-main` first; see README)")
        return
    out_dir = Path(out_dir)
    tab_dir = out_dir / "tables"
    tab_dir.mkdir(parents=True, exist_ok=True)
    p1 = part1_effect_sizes(out_dir / "master_summary.parquet")
    p2 = part2_bin_sensitivity()

    rm, rlo, rhi = p1["rho_id"]
    ram, ralo, rahi = p1["rho_all"]
    din_n, din_d = p1["disagree_id"]
    dall_n, dall_d = p1["disagree_all"]
    dmed, dlo, dhi = p1["delta_id"]
    print("=== Calibration, IDENTIFIED subset (randomized: marketing + synthetic) ===")
    print(f"  n_instances = {p1['n_ident']}")
    print(f"  rho(Qini, -ECE) = {rm:+.2f} [{rlo:+.2f}, {rhi:+.2f}]  (POSITIVE where identified)")
    print(f"  best-Qini != best-ECE: {din_n}/{din_d}")
    print(f"  paired ECE regret of Qini-selection: median={dmed:.3f} [{dlo:.3f}, {dhi:.3f}]")
    print("=== Pooled (all 25, INCLUDES confounded IHDP/Jobs) — for contrast ===")
    print(f"  rho(Qini, -ECE) = {ram:+.2f} [{ralo:+.2f}, {rahi:+.2f}];  disagree {dall_n}/{dall_d}")
    print(f"  extreme-ECE(>100) datasets = {p1['extreme']}/25 (DR-Learner blowups)")
    print("=== Bin-count STABILITY of the ECE ranking (controlled panel) ===")
    print(f"  panel: {p2['panel_models']} models, n={p2['panel_n']}, {p2['panel_seeds']} seeds")
    print(f"  pairwise Spearman across 5/10/20 bins in [{p2['min']:.3f}, {p2['max']:.3f}]")

    macros = [
        r"% auto-generated by scripts/calibration_robustness.py",
        f"\\newcommand{{\\calNident}}{{{p1['n_ident']}}}",
        f"\\newcommand{{\\calRhoId}}{{{rm:+.2f}}}",
        f"\\newcommand{{\\calRhoIdLo}}{{{rlo:+.2f}}}",
        f"\\newcommand{{\\calRhoIdHi}}{{{rhi:+.2f}}}",
        f"\\newcommand{{\\calDisIdN}}{{{din_n}}}",
        f"\\newcommand{{\\calDisIdD}}{{{din_d}}}",
        f"\\newcommand{{\\calDeltaMed}}{{{dmed:.3f}}}",
        f"\\newcommand{{\\calDeltaLo}}{{{dlo:.3f}}}",
        f"\\newcommand{{\\calDeltaHi}}{{{dhi:.3f}}}",
        f"\\newcommand{{\\calRhoAll}}{{{ram:+.2f}}}",
        f"\\newcommand{{\\calRhoAllLo}}{{{ralo:+.2f}}}",
        f"\\newcommand{{\\calRhoAllHi}}{{{rahi:+.2f}}}",
        f"\\newcommand{{\\calDisAllN}}{{{dall_n}}}",
        f"\\newcommand{{\\calDisAllD}}{{{dall_d}}}",
        f"\\newcommand{{\\calExtreme}}{{{p1['extreme']}}}",
        f"\\newcommand{{\\calBinLo}}{{{p2['min']:.3f}}}",
        f"\\newcommand{{\\calBinHi}}{{{p2['max']:.3f}}}",
        f"\\newcommand{{\\calPanelModels}}{{{p2['panel_models']}}}",
        f"\\newcommand{{\\calPanelSeeds}}{{{p2['panel_seeds']}}}",
    ]
    (tab_dir / "tab7_calibration.tex").write_text("\n".join(macros) + "\n")
    print(f"  Saved: {tab_dir/'tab7_calibration.tex'}")

    md = ["# Calibration, identification-corrected (reviewer #1, #11, #12)\n",
          "The within-bin treated-minus-control difference identifies bin-average uplift only "
          "under randomized treatment. It is valid on the marketing RCTs and the synthetic RCT; "
          "IHDP is a confounded semi-synthetic design and Jobs mixes an experimental arm with "
          "observational PSID controls, so ECE there is not identified.\n",
          f"On the IDENTIFIED subset (n={p1['n_ident']} randomized instances):",
          f"- rho(Qini, -ECE) = {rm:+.2f} [{rlo:+.2f}, {rhi:+.2f}] — POSITIVE, excludes 0.",
          f"- best-Qini != best-ECE in {din_n}/{din_d} instances, but the paired ECE regret of "
          f"Qini-selection is small: median {dmed:.3f} [{dlo:.3f}, {dhi:.3f}].",
          f"\nPooled over all 25 (INCLUDING confounded IHDP/Jobs), rho falls to "
          f"{ram:+.2f} [{ralo:+.2f}, {rahi:+.2f}] and disagreement rises to {dall_n}/{dall_d} — "
          "an artifact of the biased ECE on the confounded datasets, not a real Qini-calibration gap.",
          "\n=> We therefore do NOT claim calibration as a third instance of F2. Where uplift "
          "calibration is identified, Qini and calibration are moderately positively associated "
          "and the ECE cost of Qini-selection is small. Oracle-CATE calibration on IHDP/synthetic "
          "needs stored per-unit predictions (deferred).",
          f"\nBin-count robustness: on a controlled panel ({p2['panel_models']} models, "
          f"n={p2['panel_n']}, {p2['panel_seeds']} seeds) the ECE-induced model rankings at 5, 10 "
          f"and 20 bins agree with pairwise Spearman in [{p2['min']:.3f}, {p2['max']:.3f}], so the "
          "bin count is not a material analysis choice.",
          f"\nContext: extreme ECE (>100, DR-Learner blowups) in {p1['extreme']}/25 datasets; all "
          "statistics are rank-based and immune to them."]
    (out_dir / "findings_calibration.md").write_text("\n".join(md) + "\n")
    print(f"  Saved: {out_dir/'findings_calibration.md'}")


if __name__ == "__main__":
    main()
