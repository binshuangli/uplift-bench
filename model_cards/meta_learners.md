# Model Card: Meta-Learners (S / T / X / R / DR)

## Model details

| Field | Value |
|-------|-------|
| Method family | Meta-learner |
| Outcome type | Binary and Continuous |
| Requires propensity | X-Learner: yes; R/DR-Learner: yes (nuisance); S/T: no |
| Base learner | LightGBM (shared HP space) |
| Reference | Künzel et al. (2019), Nie & Wager (2021), Kennedy (2020) |
| Code | `src/uplift_bench/models/meta_learners.py` |

## Description

Meta-learners estimate CATE by combining one or more outcome models:

- **S-Learner**: Single outcome model `μ(x, t)`. CATE = `μ(x,1) − μ(x,0)`. Low variance
  but potentially high bias (treatment indicator may be down-weighted).
- **T-Learner**: Separate models `μ₀(x)`, `μ₁(x)` per arm. CATE = `μ₁(x) − μ₀(x)`.
  High variance when one arm is small.
- **X-Learner**: Two-stage. Stage 1: T-Learner imputed effects. Stage 2: Cross-fitted
  pseudo-outcomes, propensity-weighted combination. Improves on T-Learner for imbalanced
  treatment.
- **R-Learner**: Robinson decomposition. Fits residual outcomes after partialling out
  `E[Y|X]` and `E[T|X]`. Requires reliable nuisance estimation (slow on large datasets).
- **DR-Learner**: Doubly-robust pseudo-outcomes combining outcome and propensity models.
  Consistent if either nuisance model is correct.

All are implemented via the `econml` library with LightGBM nuisance models.

## Known strengths

- S/T-Learner: fast, interpretable, no propensity required.
- DR/R-Learner: doubly-robust; benefit observational confounding correction.
- X-Learner: robust to treatment imbalance (high π₁ datasets).

## Known weaknesses / failure modes

- R/DR-Learner: slow when nuisance models are expensive (timeout on Lenta@10K, 3600s).
- R/DR-Learner: underperform on simple RCT data where confounding correction is unnecessary.
- S-Learner: treatment effect may be shrunk to zero if the base learner ignores the
  treatment indicator as a low-importance feature.

## Benchmark performance (summary — paper run)

Scored by the **appropriate metric per regime** (√PEHE / policy risk where ground truth
exists; Qini only on marketing RCTs).

| Dataset | Best meta-learner | Metric | Notes |
|---------|-------------------|--------|-------|
| IHDP (avg, √PEHE) | T-Learner / S-Learner (√PEHE ≈ 1.25–1.32) | √PEHE ↓ | **DR-Learner is *worst* (√PEHE ≈ 40)** — high Qini but catastrophic effect error |
| Jobs (avg, policy risk) | DR-Learner / S-Learner (≈ 0.21) | PolicyRisk ↓ | narrow spread 0.21–0.25 |
| Hillstrom | R-Learner (Qini 45.1) | Qini (proxy) | no ground truth |
| Lenta | R/DR timed out | — | nuisance estimation >3600s |
| Synthetic (√PEHE) | S-Learner (≈ 0.30) | √PEHE ↓ | DR-Learner degenerates (√PEHE ≈ 27) |

> ⚠️ By Qini alone, the DR-Learner appears to win IHDP/semi-synthetic — this is an
> outcome-scale artifact. By √PEHE (ground truth) it is the worst meta-learner here. See the
> paper's metric-disagreement finding.

## Citation

```bibtex
@article{kunzel2019metalearners,
  title={Metalearners for estimating heterogeneous treatment effects using machine learning},
  author={K{\"u}nzel, S{\"o}ren R and others},
  journal={PNAS},
  year={2019}
}
@article{nie2021quasi,
  title={Quasi-oracle estimation of heterogeneous treatment effects},
  author={Nie, Xinkun and Wager, Stefan},
  journal={Biometrika},
  year={2021}
}
@article{kennedy2020optimal,
  title={Optimal doubly robust estimation of heterogeneous causal effects},
  author={Kennedy, Edward H},
  year={2020}
}
```
