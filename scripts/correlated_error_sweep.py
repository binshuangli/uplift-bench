"""Structured score-error x tail experiment for the F1 mechanism (R6, B3).

Both prior controlled experiments (kurtosis ladder; pi1 x kurtosis) used a fixed
score panel with INDEPENDENT noise (score = tau + sigma_m * eps) and found Qini
and AUUC degrade together -- never the benchmark's signature (Qini isolated,
AUUC surviving). The remaining candidate is the interaction of heavy tails with
MODEL-CORRELATED score errors: real estimators' errors are heteroskedastic,
feature-dependent, and correlated with the outcome model.

Semi-controlled design: the model panel's score errors mix an
outcome-model-correlated component h with independent noise,

    tau_hat_m(x) = tau(x) + sigma_m * [ lambda * h_i + sqrt(1-lambda^2) * eps_mi ],

where h_i = standardized (y_i(0)+y_i(1))/2 (the unit's realized potential-outcome
magnitude -- large exactly where the heavy tail lives) is SHARED across models,
eps_mi is per-model independent noise, and lambda sweeps 0 -> 1, crossed with
the kurtosis ladder. sigma_m grades panel quality as before; "quality" remains
the outcome-free -PEHE of each score against the latent tau.

Readout per (lambda, regime) cell: rho(Qini, quality), rho(AUUC, quality), and
the PAIRED Qini-AUUC gap Delta with a bootstrap CI over seeds -- the gap is the
finding; level changes alone are not (both prior experiments already showed
joint degradation).

Seeded. Emits results/tables/tab17_correrr.tex + results/findings_correrr.md.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.stats import kurtosis, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from uplift_bench.metrics.ranking import auuc, qini_coefficient  # noqa: E402

SEED = 20260718
N = 700
PI = 0.5
NOISE = np.linspace(0.4, 2.0, 8)
N_MODELS = len(NOISE)
LAMBDAS = (0.0, 0.25, 0.5, 0.75, 1.0)
REGIMES = [
    ("gaussian", "Gaussian"),
    ("student_t", "Student-$t_3$"),
    ("heavy_logn", "Log-normal (heavy)"),
]
N_SEEDS = 200


def _potential_outcomes(regime, rng, tau, baseline):
    """Return (y0, y1) with SHARED unit-level noise so h reflects the realized tail."""
    if regime == "gaussian":
        e = rng.standard_normal(N)
        return baseline + e, baseline + tau + e
    if regime == "student_t":
        e = rng.standard_t(df=3, size=N)
        return baseline + e, baseline + tau + e
    if regime == "heavy_logn":
        e = rng.standard_normal(N)
        return (np.exp(0.9 * baseline + 1.0 * e), np.exp(0.9 * baseline + 0.3 * tau + 1.0 * e))
    raise ValueError(regime)


def main(out_dir="results"):
    rng = np.random.default_rng(SEED)
    boot = np.random.default_rng(SEED + 1)
    results = {}
    for lam in LAMBDAS:
        for regime, _label in REGIMES:
            rq, ra, kurt = [], [], []
            for _ in range(N_SEEDS):
                x = rng.standard_normal((N, 3))
                t = rng.binomial(1, PI, N)
                tau = 1.0 * x[:, 0] + 0.5 * x[:, 1]
                baseline = 0.8 * x[:, 2]
                y0, y1 = _potential_outcomes(regime, rng, tau, baseline)
                y = np.where(t == 1, y1, y0)
                kurt.append(kurtosis(y))
                h = (y0 + y1) / 2.0
                h = (h - h.mean()) / (h.std() + 1e-12)
                scores, pehe = [], []
                for eps_m in NOISE:
                    err = lam * h + np.sqrt(1 - lam**2) * rng.standard_normal(N)
                    s = tau + eps_m * err
                    scores.append(s)
                    pehe.append(np.sqrt(np.mean((s - tau) ** 2)))
                quality = -np.asarray(pehe)
                mq = [qini_coefficient(s, t, y, normalize=False) for s in scores]
                ma = [auuc(s, t, y) for s in scores]
                rq.append(spearmanr(mq, quality).correlation)
                ra.append(spearmanr(ma, quality).correlation)
            rq, ra = np.asarray(rq), np.asarray(ra)
            delta = rq - ra
            db = np.array(
                [boot.choice(delta, len(delta), replace=True).mean() for _ in range(4000)]
            )
            results[(lam, regime)] = dict(
                rho_qini=float(np.nanmean(rq)),
                rho_auuc=float(np.nanmean(ra)),
                delta=float(np.nanmean(delta)),
                dlo=float(np.percentile(db, 2.5)),
                dhi=float(np.percentile(db, 97.5)),
                kurt=float(np.mean(kurt)),
            )

    print("=== B3: correlated score error x tails; Delta = Qini - AUUC (paired) ===")
    print(
        f"{'lam':>5} {'regime':<20} {'kurt':>7} {'Qini':>7} {'AUUC':>7} "
        f"{'Delta':>7} {'95% CI':>18}"
    )
    sig = []
    for lam in LAMBDAS:
        for regime, label in REGIMES:
            r = results[(lam, regime)]
            neg = r["dhi"] < 0
            if neg:
                sig.append((lam, label))
            print(
                f"{lam:>5.2f} {label:<20} {r['kurt']:>7.1f} {r['rho_qini']:>+7.3f} "
                f"{r['rho_auuc']:>+7.3f} {r['delta']:>+7.3f} "
                f"[{r['dlo']:+.3f},{r['dhi']:+.3f}]{' <-- Qini isolated' if neg else ''}"
            )
    verdict = "REPRODUCED" if sig else "not reproduced"
    print(f"\ncells with significantly negative Delta: {sig} -> signature {verdict}")

    key = results[(0.75, "heavy_logn")]
    base = results[(0.0, "heavy_logn")]
    macros = [
        r"% auto-generated by scripts/correlated_error_sweep.py (R6 B3)",
        f"\\newcommand{{\\correrrKeyDelta}}{{{key['delta']:+.3f}}}",
        f"\\newcommand{{\\correrrKeyDeltaLo}}{{{key['dlo']:+.3f}}}",
        f"\\newcommand{{\\correrrKeyDeltaHi}}{{{key['dhi']:+.3f}}}",
        f"\\newcommand{{\\correrrBaseDelta}}{{{base['delta']:+.3f}}}",
        f"\\newcommand{{\\correrrNSig}}{{{len(sig)}}}",
        f"\\newcommand{{\\correrrNSeeds}}{{{N_SEEDS}}}",
    ]
    outp = Path(out_dir) / "tables" / "tab17_correrr.tex"
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text("\n".join(macros) + "\n")

    lines = [
        "# Correlated score-error x tails sweep (R6 B3)",
        "",
        f"Signature (Qini isolated, AUUC surviving): **{verdict}**",
        "",
        "| lambda | regime | kurt | rho_Qini | rho_AUUC | Delta [95% CI] |",
        "|---|---|---|---|---|---|",
    ]
    for lam in LAMBDAS:
        for regime, label in REGIMES:
            r = results[(lam, regime)]
            lines.append(
                f"| {lam:.2f} | {label} | {r['kurt']:.1f} | {r['rho_qini']:+.3f} "
                f"| {r['rho_auuc']:+.3f} | {r['delta']:+.3f} "
                f"[{r['dlo']:+.3f},{r['dhi']:+.3f}] |"
            )
    (Path(out_dir) / "findings_correrr.md").write_text("\n".join(lines) + "\n")
    print(f"Saved: {outp} and findings_correrr.md")


if __name__ == "__main__":
    main()
