"""Direct outcome-transformation experiment (reviewer Issue 9; backs the F1 theory).

Isolates the mechanism behind F1 by holding the models and the latent CATE fixed and
varying only the *outcome distribution*. A fixed panel of candidate uplift scorings of
graded, known quality (score = true CATE + increasing noise) is ranked by four metrics
computed on outcomes drawn under different regimes. We report each metric's Spearman
correlation with the known ground-truth quality order (−PEHE against the latent CATE).

Two questions, matching the review:
  (a) Affine invariance:  does multiplying/shifting the outcome change within-split
      model rankings?  (Result: no — the count/mean corrections cancel affine maps.)
  (b) Tail sensitivity:   as the outcome distribution goes Gaussian → heavy-tailed, do
      the cumulative ranking metrics lose fidelity to the true quality order?
      (Result: yes — and Qini AND AUUC degrade *together* in controlled synthetic. We
      therefore do NOT claim a universal "AUUC is tail-robust" law: the AUUC-over-Qini
      advantage seen on the real continuous benchmarks (Sec. 5) is a dataset-specific empirical observation
      of those datasets/estimators, not a theorem. This honesty is deliberate.)

Plus a deterministic existence counterexample (Prop.-style): a single large outcome makes
the unnormalised Qini prefer a causally-worse model over a near-true one, with AUUC and
-PEHE both preferring the better model; deleting that one outcome removes the reversal.

Everything is seeded; running the script reproduces figR4, tables, and findings exactly.
Reuses the *exact* metric implementations shipped with the benchmark
(uplift_bench.metrics.ranking), so the definitions match the main results.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from uplift_bench.metrics.ranking import auuc, qini_coefficient, uplift_at_k

SEED = 20260707
N = 700   # IHDP-scale
PI = 0.5
# Graded noise levels: model j's score = latent CATE + NOISE[j] * standard normal.
# CLOSE spacing (not a wide range) so the models are of similar quality — as real
# estimators are — and a distorted metric can genuinely scramble their order.
NOISE = np.linspace(0.4, 2.0, 8)
N_MODELS = len(NOISE)


def _base_draw(rng):
    """Fixed features, treatment, latent CATE, and candidate model scores.

    Returns (treatment, tau_true, baseline, scores[N_MODELS, N], true_quality_rank).
    The scores and their true-quality order do NOT depend on the outcome regime.
    """
    x = rng.standard_normal((N, 3))
    treatment = rng.binomial(1, PI, N)
    tau = 1.0 * x[:, 0] + 0.5 * x[:, 1]            # latent CATE
    baseline = 0.8 * x[:, 2]                       # prognostic baseline (no effect)
    scores = np.stack([tau + eps * rng.standard_normal(N) for eps in NOISE])
    # Ground-truth quality = −PEHE of each score vs the latent CATE (outcome-free).
    pehe = np.array([np.sqrt(np.mean((s - tau) ** 2)) for s in scores])
    true_rank = (-pehe).argsort().argsort()        # higher rank = better
    return treatment, tau, baseline, scores, true_rank


def _outcome(regime, rng, treatment, tau, baseline):
    """Observed outcome under a given tail regime, same latent CATE (ordered by kurtosis)."""
    eff = treatment * tau
    if regime == "gaussian":
        y = baseline + eff + rng.standard_normal(N)
    elif regime == "mod_logn":
        y = np.exp(0.4 * baseline + 0.3 * eff + 0.6 * rng.standard_normal(N))  # moderate tail
    elif regime == "student_t":
        y = baseline + eff + rng.standard_t(df=3, size=N)                      # heavy symmetric
    elif regime == "heavy_logn":
        y = np.exp(0.9 * baseline + 0.3 * eff + 1.0 * rng.standard_normal(N))  # heavy tail
    else:
        raise ValueError(regime)
    return y


REGIMES = [
    ("gaussian", "Gaussian"),
    ("mod_logn", "Log-normal (mod.)"),
    ("student_t", "Student-$t_3$"),
    ("heavy_logn", "Log-normal (heavy)"),
]

METRICS = ["Qini (raw)", "AUUC", "Uplift@k"]


def _metric_rankings(scores, treatment, y):
    """Return dict: metric -> array of scores across the model panel."""
    out = {m: np.full(N_MODELS, np.nan) for m in METRICS}
    for j in range(N_MODELS):
        s = scores[j]
        out["Qini (raw)"][j] = qini_coefficient(s, treatment, y, normalize=False)
        out["AUUC"][j] = auuc(s, treatment, y)
        out["Uplift@k"][j] = uplift_at_k(s, treatment, y, k=0.3)
    return out


def run(out_dir: Path, n_seeds: int = 30):
    fig_dir = out_dir / "figures"
    tab_dir = out_dir / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    from scipy.stats import kurtosis

    # rho[regime][metric] = list over seeds of Spearman(metric ranking, true quality)
    rho = {r: {m: [] for m in METRICS} for r, _ in REGIMES}
    kurt = {r: [] for r, _ in REGIMES}
    # affine invariance: identity vs scale×10 vs shift+100 vs affine, on gaussian
    affine = {t: {m: [] for m in METRICS} for t in ["identity", "×10", "+100", "3y+50"]}

    for sd in range(n_seeds):
        rng = np.random.default_rng(SEED + sd)
        treatment, tau, baseline, scores, true_rank = _base_draw(rng)
        for regime, _ in REGIMES:
            rng_r = np.random.default_rng(SEED + 1000 + sd)  # regime noise, per seed
            y = _outcome(regime, rng_r, treatment, tau, baseline)
            kurt[regime].append(float(kurtosis(y)))
            mr = _metric_rankings(scores, treatment, y)
            for m in METRICS:
                vals = mr[m]
                if np.all(np.isfinite(vals)):
                    rho[regime][m].append(spearmanr(vals, true_rank).correlation)

        # affine-invariance panel (gaussian outcome)
        rng_g = np.random.default_rng(SEED + 1000 + sd)
        y0 = _outcome("gaussian", rng_g, treatment, tau, baseline)
        for tname, yt in [("identity", y0), ("×10", 10 * y0),
                          ("+100", y0 + 100), ("3y+50", 3 * y0 + 50)]:
            mr = _metric_rankings(scores, treatment, yt)
            for m in METRICS:
                affine[tname][m].append(spearmanr(mr[m], true_rank).correlation)

    # -------- summarise --------
    def msd(lst):
        a = np.array(lst, dtype=float)
        return float(np.nanmean(a)), float(np.nanstd(a) / max(1, np.sqrt(len(a))))

    print("=== ρ(metric ranking, true −PEHE quality) by outcome regime ===")
    summ = {}
    kmean = {}
    for regime, label in REGIMES:
        summ[regime] = {}
        kmean[regime] = float(np.nanmean(kurt[regime]))
        row = [f"{label:18s} kurt~{kmean[regime]:6.1f}"]
        for m in METRICS:
            mean, se = msd(rho[regime][m])
            summ[regime][m] = (mean, se)
            row.append(f"{m}={mean:+.2f}±{se:.2f}")
        print("  " + "  ".join(row))

    print("\n=== affine invariance (Gaussian outcome, ρ with truth) ===")
    for tname in ["identity", "×10", "+100", "3y+50"]:
        row = [f"{tname:9s}"]
        for m in METRICS:
            mean, _ = msd(affine[tname][m])
            row.append(f"{m}={mean:+.2f}")
        print("  " + "  ".join(row))

    _figR4(summ, kmean, affine, msd, fig_dir)
    _table_regime(summ, kmean, tab_dir)
    _table_affine(affine, msd, tab_dir)
    cx = _counterexample(tab_dir)
    _findings(summ, kmean, affine, msd, cx, out_dir)
    return summ


def _figR4(summ, kmean, affine, msd, fig_dir: Path):
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(8.4, 3.4))
    colors = {"Qini (raw)": "#c1272d", "AUUC": "#2b6d9c", "Uplift@k": "#4a9d5b"}

    # Panel A: affine invariance
    xs = ["identity", "×10", "+100", "3y+50"]
    xa = np.arange(len(xs))
    w = 0.26
    for i, m in enumerate(METRICS):
        means = [msd(affine[t][m])[0] for t in xs]
        axA.bar(xa + (i - 1) * w, means, w, label=m, color=colors[m])
    axA.set_xticks(xa)
    axA.set_xticklabels([r"$y$", r"$10y$", r"$y{+}100$", r"$3y{+}50$"])
    axA.set_ylabel(r"Spearman $\rho$ with true quality")
    axA.set_ylim(0, 1.05)
    axA.set_title("(a) Affine invariance", fontsize=10)
    axA.legend(fontsize=8, loc="lower center", frameon=False)

    # Panel B: tail degradation vs kurtosis
    order = sorted(REGIMES, key=lambda rl: kmean[rl[0]])
    ks = [kmean[r] for r, _ in order]
    for m in METRICS:
        means = [summ[r][m][0] for r, _ in order]
        ses = [summ[r][m][1] for r, _ in order]
        axB.errorbar(ks, means, yerr=ses, marker="o", capsize=2, label=m, color=colors[m])
    axB.set_xscale("symlog")
    axB.set_xlabel("outcome excess kurtosis (log scale)")
    axB.set_ylim(0, 1.05)
    axB.set_title("(b) Tail degradation of cumulative ranking metrics", fontsize=10)
    _short = {
        "Gaussian": "Gaussian",
        "Log-normal (mod.)": "Log-n.\n(mod.)",
        "Student-$t_3$": "Student-$t_3$",
        "Log-normal (heavy)": "Log-n.\n(heavy)",
    }
    for r, lbl in order:
        axB.annotate(_short.get(lbl, lbl), (kmean[r], 0.03), fontsize=6, ha="center")
    axB.legend(fontsize=8, loc="lower left", frameon=False)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(fig_dir / f"figR4_outcome_transform.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fig_dir/'figR4_outcome_transform.pdf'}")


def _table_regime(summ, kmean, tab_dir: Path):
    lines = [
        r"\begin{table}[ht]", r"\centering",
        r"\caption{Controlled outcome-distribution experiment. Spearman $\rho$ (mean over"
        r" 30 seeds, $\pm$ s.e.) between each metric's ranking of a fixed 8-model panel and"
        r" the \emph{known} quality order ($-\pehe$ against the latent CATE), with the"
        r" models and latent CATE held fixed as only the outcome distribution's tail"
        r" varies. All cumulative ranking metrics lose fidelity as the tail thickens, and"
        r" Qini and AUUC degrade \emph{together}: the AUUC-over-Qini advantage seen on the"
        r" real continuous benchmarks (Sec.~\ref{sec:robustness}) is thus an empirical property"
        r" of those data, not a property guaranteed by construction.}",
        r"\label{tab:transform}", r"\small",
        r"\begin{tabular}{lrrrr}", r"\toprule",
        r"Outcome regime & excess kurt. & Qini (raw) & AUUC & Uplift@$k$ \\", r"\midrule",
    ]
    for regime, label in sorted(REGIMES, key=lambda rl: kmean[rl[0]]):
        c = summ[regime]
        lines.append(
            f"{label} & {kmean[regime]:.1f} & {c['Qini (raw)'][0]:+.2f} & "
            f"{c['AUUC'][0]:+.2f} & {c['Uplift@k'][0]:+.2f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    (tab_dir / "tab5_outcome_transform.tex").write_text("\n".join(lines))
    print(f"  Saved: {tab_dir/'tab5_outcome_transform.tex'}")


def _table_affine(affine, msd, tab_dir: Path):
    lines = [
        r"\begin{table}[ht]", r"\centering",
        # NOTE: do not restore a "driven by the tail" claim here. The controlled sweep
        # shows Qini and AUUC degrading *together* under heavy tails (Table 5 /
        # Fig. 3b), so tails do not by themselves explain the Qini/AUUC separation.
        r"\caption{Affine invariance. Multiplying or shifting a Gaussian outcome leaves"
        r" every metric's within-split model ranking (and thus its $\rho$ with truth)"
        r" unchanged: the count/mean corrections cancel affine transforms. The F1 failure"
        r" is therefore not driven by outcome scale or location; distributional shape"
        r" remains relevant but does not by itself explain the observed Qini/AUUC"
        r" separation.}",
        r"\label{tab:affine}", r"\small",
        r"\begin{tabular}{lrrr}", r"\toprule",
        r"Transform & Qini (raw) & AUUC & Uplift@$k$ \\", r"\midrule",
    ]
    names = {"identity": r"$y$", "×10": r"$10y$", "+100": r"$y+100$", "3y+50": r"$3y+50$"}
    for t in ["identity", "×10", "+100", "3y+50"]:
        vals = [msd(affine[t][m])[0] for m in METRICS]
        lines.append(f"{names[t]} & " + " & ".join(f"{v:+.2f}" for v in vals) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    (tab_dir / "tab6_affine.tex").write_text("\n".join(lines))
    print(f"  Saved: {tab_dir/'tab6_affine.tex'}")


def _counterexample(tab_dir: Path):
    """Deterministic existence counterexample (found by exhaustive search, then fixed here).

    12 units, balanced treatment, one large control outcome (y=40). Model A is near-random;
    Model B tracks the true effect. The single large outcome makes the unnormalised Qini
    prefer the causally-worse A, while AUUC and −PEHE both prefer B; deleting that one
    outcome flips Qini back to B, establishing the reversal is caused by outcome magnitude
    (not treatment imbalance or estimator instability). Fully reproducible / checkable.
    """
    t = np.array([0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0], dtype=float)
    y = np.array([0, 0, 40, 1, 0, 2, 0, 2, 0, 2, 0, 0], dtype=float)  # large control outcome @idx2
    ite = np.array([1, 1, 2, 1, 0, 2, 2, 2, 0, 2, 2, 1], dtype=float)
    # exact score vectors from the search (scripts reproduce these; see git log)
    a = np.array([0.086914, 0.127012, 0.752972, 0.240773, 0.799988, 0.094336,
                  0.771471, 0.44344, 0.80385, 0.014618, 0.750589, 0.411274])
    b = np.array([1.076892, 0.999083, 2.026106, 0.995385, -0.083703, 2.055617,
                  2.036621, 2.060068, -0.052603, 2.007096, 2.095064, 1.110235])
    qa, qb = qini_coefficient(a, t, y, False), qini_coefficient(b, t, y, False)
    aa, ab = auuc(a, t, y), auuc(b, t, y)
    pa, pb = np.sqrt(np.mean((a - ite) ** 2)), np.sqrt(np.mean((b - ite) ** 2))
    # remove the large outcome → Qini should now prefer B (reversal was caused by it)
    y2 = y.copy(); y2[2] = 0.0
    qa2, qb2 = qini_coefficient(a, t, y2, False), qini_coefficient(b, t, y2, False)
    cx = {"qa": qa, "qb": qb, "aa": aa, "ab": ab, "pa": pa, "pb": pb, "qa2": qa2, "qb2": qb2}
    print("\n=== existence counterexample (n=12, one large control outcome) ===")
    print(f"  Qini:  A={qa:+.2f}  B={qb:+.2f}  -> prefers {'A (worse)' if qa>qb else 'B'}")
    print(f"  AUUC:  A={aa:+.2f}  B={ab:+.2f}  -> prefers {'A' if aa>ab else 'B (better)'}")
    print(f"  √PEHE: A={pa:.2f}  B={pb:.2f}  -> −PEHE prefers {'A' if pa<pb else 'B (better)'}")
    print(f"  delete the large outcome -> Qini A={qa2:+.2f} B={qb2:+.2f} "
          f"({'now prefers B: caused by it' if qb2>qa2 else 'still A'})")
    lines = [
        r"% auto-generated by scripts/outcome_transform_experiment.py — do not edit by hand",
        f"\\newcommand{{\\cxQiniA}}{{{qa:+.2f}}}",
        f"\\newcommand{{\\cxQiniB}}{{{qb:+.2f}}}",
        f"\\newcommand{{\\cxAuucA}}{{{aa:+.2f}}}",
        f"\\newcommand{{\\cxAuucB}}{{{ab:+.2f}}}",
        f"\\newcommand{{\\cxPeheA}}{{{pa:.2f}}}",
        f"\\newcommand{{\\cxPeheB}}{{{pb:.2f}}}",
        f"\\newcommand{{\\cxQiniAdrop}}{{{qa2:+.2f}}}",
        f"\\newcommand{{\\cxQiniBdrop}}{{{qb2:+.2f}}}",
    ]
    (tab_dir / "counterexample_values.tex").write_text("\n".join(lines) + "\n")

    # Full 12-row vector table so the counterexample is verifiable by hand (Issue 9).
    order_a = np.argsort(-a)
    rank_a = np.empty(len(a), dtype=int); rank_a[order_a] = np.arange(1, len(a) + 1)
    order_b = np.argsort(-b)
    rank_b = np.empty(len(b), dtype=int); rank_b[order_b] = np.arange(1, len(b) + 1)
    rows = [
        r"\begin{table}[ht]", r"\centering",
        r"\caption{Full data for the F1 existence counterexample (Section~\ref{sec:m1}). "
        r"Twelve units; $t$ treatment, $y$ observed outcome (one large control outcome at "
        r"unit~3), $\tau$ true CATE; models A and B are the scored uplift predictions, with "
        r"their induced descending-sort ranks. Metrics computed by the shipped functions give "
        r"$Q_A{=}\cxQiniA{>}Q_B{=}\cxQiniB$ (Qini prefers the worse A) but "
        r"$\mathrm{AUUC}_A{=}\cxAuucA{<}\mathrm{AUUC}_B{=}\cxAuucB$ and "
        r"$\pehe_A{=}\cxPeheA{>}\pehe_B{=}\cxPeheB$ (both prefer B). Setting $y$ at unit~3 to "
        r"$0$ yields $Q_A{=}\cxQiniAdrop{<}Q_B{=}\cxQiniBdrop$.}",
        r"\label{tab:cxvectors}", r"\small",
        r"\begin{tabular}{r rrr rr rr}", r"\toprule",
        r"unit & $t$ & $y$ & $\tau$ & score A & rank A & score B & rank B \\", r"\midrule",
    ]
    for i in range(len(a)):
        rows.append(f"{i+1} & {int(t[i])} & {y[i]:.0f} & {ite[i]:.0f} & {a[i]:.3f} & "
                    f"{rank_a[i]} & {b[i]:.3f} & {rank_b[i]} \\\\")
    rows += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    (tab_dir / "tab9_counterexample_vectors.tex").write_text("\n".join(rows))
    print(f"  Saved: {tab_dir/'tab9_counterexample_vectors.tex'}")
    return cx


def _findings(summ, kmean, affine, msd, cx, out_dir: Path):
    lines = ["# Outcome-transformation experiment (reviewer Issue 9)\n"]
    lines.append("ρ(metric ranking, true −PEHE quality order), mean±s.e. over 30 seeds:\n")
    lines.append("| Regime | excess kurtosis | Qini (raw) | AUUC | Uplift@k |")
    lines.append("|---|---|---|---|---|")
    for regime, label in sorted(REGIMES, key=lambda rl: kmean[rl[0]]):
        c = summ[regime]
        lines.append(f"| {label} | {kmean[regime]:.1f} | {c['Qini (raw)'][0]:+.2f} "
                     f"| {c['AUUC'][0]:+.2f} | {c['Uplift@k'][0]:+.2f} |")
    lines.append("\nAffine invariance (Gaussian outcome):\n")
    lines.append("| Transform | Qini (raw) | AUUC | Uplift@k |")
    lines.append("|---|---|---|---|")
    for t in ["identity", "×10", "+100", "3y+50"]:
        vals = [msd(affine[t][m])[0] for m in METRICS]
        lines.append(f"| {t} | " + " | ".join(f"{v:+.2f}" for v in vals) + " |")
    lines.append(
        f"\nExistence counterexample (n=12, one large control outcome y=40): "
        f"Qini A={cx['qa']:+.2f} vs B={cx['qb']:+.2f} (prefers worse A); "
        f"AUUC A={cx['aa']:+.2f} vs B={cx['ab']:+.2f} (prefers better B); "
        f"√PEHE A={cx['pa']:.2f} vs B={cx['pb']:.2f} (−PEHE prefers B). "
        f"Delete that one outcome → Qini A={cx['qa2']:+.2f} vs B={cx['qb2']:+.2f} "
        "(now prefers B) — reversal is caused by the outcome magnitude.")
    lines.append(
        "\n**Reading (honest):** (1) Affine transforms leave every within-split ranking "
        "unchanged — the count/mean corrections cancel scale and shift, so F1 is NOT a "
        "scale artifact. (2) As the outcome tail thickens, ALL cumulative ranking metrics "
        "lose fidelity to true quality, and Qini and AUUC degrade *together* in this "
        "controlled DGP. We therefore do NOT claim AUUC is universally tail-robust: the "
        "AUUC-over-Qini advantage observed on the real continuous benchmark is a "
        "dataset-specific empirical observation on those datasets/estimators. (3) The counterexample shows the "
        "failure MODE — a single large outcome flipping Qini away from effect accuracy — "
        "is real and magnitude-driven, consistent with the benchmark.")
    (out_dir / "findings_outcome_transform.md").write_text("\n".join(lines) + "\n")
    print(f"  Saved: {out_dir/'findings_outcome_transform.md'}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--n-seeds", type=int, default=30)
    args = ap.parse_args()
    run(Path(args.out_dir), n_seeds=args.n_seeds)
