"""Treatment-imbalance x kurtosis sweep for the F1 mechanism (reviewer R5, W4/Q2).

The controlled outcome-transformation experiment (outcome_transform_experiment.py)
holds pi1 = 0.5 and finds that heavy tails degrade Qini and AUUC *together* -- it
does not reproduce the benchmark's signature (Qini isolated, AUUC surviving). The
reviewer proposes a concrete candidate rooted in Lemma 5.2: the interleaving term
proportional to (T_k - (k/n) T_n) -- the fluctuation of the cumulative treated
count around its share -- multiplies the full-sample ATE estimate u(n), whose
noise is heavy-tailed exactly in the regime where F1 appears. Treatment
IMBALANCE (IHDP has pi1 = 0.19) inflates those count fluctuations relative to
balanced assignment.

This script crosses pi1 in {0.50, 0.35, 0.19, 0.10} with the same outcome
regimes (Gaussian -> heavy log-normal), holding the latent CATE, the model
panel, and everything else identical to the original experiment. Per cell we
report each metric's mean Spearman correlation with the known quality order and
the paired Qini-minus-AUUC gap Delta with a bootstrap CI over seeds.

Interpretation:
  - If Delta turns significantly negative only under (low pi1) x (heavy tails),
    the interleaving term is a supported mechanism for the Qini/AUUC separation.
  - If Delta stays ~0 everywhere, imbalance does not explain the separation and
    the paper's "characterized, not explained" boundary stands.

Seeded; emits results/tables/tab14_pi1_sweep.tex (macros + a small table) and
results/findings_pi1_sweep.md.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.stats import kurtosis, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from uplift_bench.metrics.ranking import auuc, qini_coefficient, uplift_at_k  # noqa: E402

SEED = 20260717
N = 700
NOISE = np.linspace(0.4, 2.0, 8)
N_MODELS = len(NOISE)
PI_GRID = (0.50, 0.35, 0.19, 0.10)
REGIMES = [
    ("gaussian", "Gaussian"),
    ("mod_logn", "Log-normal (mod.)"),
    ("student_t", "Student-$t_3$"),
    ("heavy_logn", "Log-normal (heavy)"),
]
N_SEEDS = 200


def _draw(rng, pi1):
    x = rng.standard_normal((N, 3))
    treatment = rng.binomial(1, pi1, N)
    tau = 1.0 * x[:, 0] + 0.5 * x[:, 1]
    baseline = 0.8 * x[:, 2]
    scores = np.stack([tau + eps * rng.standard_normal(N) for eps in NOISE])
    pehe = np.array([np.sqrt(np.mean((s - tau) ** 2)) for s in scores])
    return treatment, tau, baseline, scores, -pehe  # higher = better


def _outcome(regime, rng, treatment, tau, baseline):
    eff = treatment * tau
    if regime == "gaussian":
        return baseline + eff + rng.standard_normal(N)
    if regime == "mod_logn":
        return np.exp(0.4 * baseline + 0.3 * eff + 0.6 * rng.standard_normal(N))
    if regime == "student_t":
        return baseline + eff + rng.standard_t(df=3, size=N)
    if regime == "heavy_logn":
        return np.exp(0.9 * baseline + 0.3 * eff + 1.0 * rng.standard_normal(N))
    raise ValueError(regime)


def main(out_dir="results"):
    rng = np.random.default_rng(SEED)
    boot = np.random.default_rng(SEED + 1)
    results = {}  # (pi1, regime) -> dict of per-seed rho lists + kurtosis
    for pi1 in PI_GRID:
        for regime, _label in REGIMES:
            rq, ra, ru, kurt = [], [], [], []
            for _ in range(N_SEEDS):
                t, tau, base, scores, quality = _draw(rng, pi1)
                # degenerate treated counts can occur at pi1=0.10; redraw-safe guard
                if t.sum() < 10 or (1 - t).sum() < 10:
                    continue
                y = _outcome(regime, rng, t, tau, base)
                kurt.append(kurtosis(y))
                mq = [qini_coefficient(s, t, y, normalize=False) for s in scores]
                ma = [auuc(s, t, y) for s in scores]
                mu = [uplift_at_k(s, t, y, k=0.3) for s in scores]
                rq.append(spearmanr(mq, quality).correlation)
                ra.append(spearmanr(ma, quality).correlation)
                ru.append(spearmanr(mu, quality).correlation)
            rq, ra, ru = map(np.asarray, (rq, ra, ru))
            delta = rq - ra
            db = np.array(
                [boot.choice(delta, len(delta), replace=True).mean() for _ in range(4000)]
            )
            results[(pi1, regime)] = dict(
                rho_qini=rq.mean(),
                rho_auuc=ra.mean(),
                rho_uplift=ru.mean(),
                delta=delta.mean(),
                dlo=float(np.percentile(db, 2.5)),
                dhi=float(np.percentile(db, 97.5)),
                kurt=float(np.mean(kurt)),
                n=len(rq),
            )

    print("=== pi1 x kurtosis sweep: rho(metric, true quality); Delta = Qini - AUUC ===")
    print(
        f"{'pi1':>5} {'regime':<18} {'kurt':>6} {'Qini':>7} {'AUUC':>7} {'Up@k':>7} "
        f"{'Delta':>7} {'95% CI':>18}"
    )
    sig_cells = []
    for pi1 in PI_GRID:
        for regime, label in REGIMES:
            r = results[(pi1, regime)]
            sig = r["dhi"] < 0
            if sig:
                sig_cells.append((pi1, label))
            print(
                f"{pi1:>5.2f} {label:<18} {r['kurt']:>6.1f} {r['rho_qini']:>+7.3f} "
                f"{r['rho_auuc']:>+7.3f} {r['rho_uplift']:>+7.3f} {r['delta']:>+7.3f} "
                f"[{r['dlo']:+.3f},{r['dhi']:+.3f}]{' <-- Qini<AUUC' if sig else ''}"
            )
    print(f"\ncells with significantly negative Delta (Qini worse than AUUC): {sig_cells}")

    # Macros for the paper: the IHDP-matched cell (pi1=0.19, heavy_logn) and the
    # balanced heavy cell (pi1=0.50, heavy_logn) as the contrast, plus verdict.
    key_imb = results[(0.19, "heavy_logn")]
    key_bal = results[(0.50, "heavy_logn")]
    verdict = "supported" if (0.19, "Log-normal (heavy)") in sig_cells else "not supported"
    macros = [
        r"% auto-generated by scripts/pi1_kurtosis_sweep.py (R5 W4/Q2)",
        f"\\newcommand{{\\piSweepImbDelta}}{{{key_imb['delta']:+.3f}}}",
        f"\\newcommand{{\\piSweepImbDeltaLo}}{{{key_imb['dlo']:+.3f}}}",
        f"\\newcommand{{\\piSweepImbDeltaHi}}{{{key_imb['dhi']:+.3f}}}",
        f"\\newcommand{{\\piSweepBalDelta}}{{{key_bal['delta']:+.3f}}}",
        f"\\newcommand{{\\piSweepBalDeltaLo}}{{{key_bal['dlo']:+.3f}}}",
        f"\\newcommand{{\\piSweepBalDeltaHi}}{{{key_bal['dhi']:+.3f}}}",
        f"\\newcommand{{\\piSweepImbQini}}{{{key_imb['rho_qini']:+.2f}}}",
        f"\\newcommand{{\\piSweepImbAuuc}}{{{key_imb['rho_auuc']:+.2f}}}",
        f"\\newcommand{{\\piSweepNSeeds}}{{{key_imb['n']}}}",
        f"\\newcommand{{\\piSweepNCellsSig}}{{{len(sig_cells)}}}",
    ]
    outp = Path(out_dir) / "tables" / "tab14_pi1_sweep.tex"
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text("\n".join(macros) + "\n")

    lines = [
        "# pi1 x kurtosis sweep (R5)",
        "",
        f"Interleaving-term mechanism (Lemma 5.2) for the Qini/AUUC separation: "
        f"**{verdict}** at the IHDP-matched cell (pi1=0.19, heavy log-normal).",
        "",
        "| pi1 | regime | kurt | rho_Qini | rho_AUUC | Delta [95% CI] |",
        "|---|---|---|---|---|---|",
    ]
    for pi1 in PI_GRID:
        for regime, label in REGIMES:
            r = results[(pi1, regime)]
            lines.append(
                f"| {pi1:.2f} | {label} | {r['kurt']:.1f} | {r['rho_qini']:+.3f} "
                f"| {r['rho_auuc']:+.3f} | {r['delta']:+.3f} "
                f"[{r['dlo']:+.3f},{r['dhi']:+.3f}] |"
            )
    (Path(out_dir) / "findings_pi1_sweep.md").write_text("\n".join(lines) + "\n")
    print(f"Saved: {outp} and findings_pi1_sweep.md")


if __name__ == "__main__":
    main()
