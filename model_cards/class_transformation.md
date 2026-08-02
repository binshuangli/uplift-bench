# Model Card: Class-Transformation Models (ClassTrans / TwoModel / SoloModel)

## Model details

| Field | Value |
|-------|-------|
| Method family | Class-transformation |
| Outcome type | Binary only |
| Requires propensity | No (balanced treatment assumed or explicitly handled) |
| Base learner | LightGBM (shared HP space) |
| Reference | Jaskowski & Jaroszewicz (2012); Kane et al. (2014) |
| Code | `src/uplift_bench/models/baselines.py` |

## Description

- **ClassTrans** (Jaskowski & Jaroszewicz 2012): Reformulates binary uplift as
  classification on a derived target `Z = Y·T + (1−Y)·(1−T)` under balanced treatment.
  The classical derivation assumes `π₁ ≈ 0.5`; in practice it leads on Criteo at
  `π₁ = 0.85` (v1.4.2 supplement), so imbalance alone does not break it.
- **TwoModel**: Direct difference of two LightGBM classifiers trained separately on
  treated and control subsamples. Simple, fast, no assumptions on treatment balance.
- **SoloModel** (Kane et al. 2014): An S-learner — a single classifier trained on both
  groups with the treatment indicator as a feature (scikit-uplift's `SoloModel`); uplift
  is the difference between the model's predictions with the indicator set to 1 vs 0.

## Known strengths

- **ClassTrans**: consistently wins on Jobs (all 10 splits), likely due to binary outcome
  and near-balanced treatment in that dataset. It also leads on Criteo (v1.4.2 supplement)
  at treated fraction 0.85, so near-balanced treatment is evidently not *necessary* for its
  strong performance — the balance rationale above should be read as a hypothesis, not an
  established mechanism.
- **SoloModel**: best or near-best calibration (ECE) across most datasets. Produces
  scores that correlate well with individual treatment response.
- Fast inference: single pass at prediction time.

## Known weaknesses / failure modes

- All three require binary outcome. Not applicable to IHDP (continuous).
- ClassTrans degrades with treatment imbalance (Lenta π₁=0.75 makes `Z` noisy).
- SoloModel ignores control outcomes entirely — cannot detect negative treatments.

## Benchmark performance (summary — paper run)

| Dataset | ClassTrans | TwoModel | SoloModel |
|---------|-----------|---------|-----------|
| Jobs (avg 10 splits) | **6.4** (1st) | 5.2 | 4.5 |
| Hillstrom | 40.0 | 40.9 | **49.1** (1st) |
| Lenta | **20.3** (1st) | 3.9 | 1.8 |
| X5 | −6.1 | −4.3 | **−0.3** (1st) |
| Synthetic | 1.79 | **2.70** (1st) | −0.35 |

## Citation

```bibtex
@article{jaskowski2012uplift,
  title={Uplift modeling for clinical trial data},
  author={Jaskowski, Maciej and Jaroszewicz, Szymon},
  journal={ICML Workshop on Clinical Data Analysis},
  year={2012}
}
@article{kane2014mining,
  title={Mining for the truly responsive customers and prospects},
  author={Kane, Kevin and Lo, Victor SY and Zheng, Jianying},
  journal={Journal of Marketing Analytics},
  year={2014}
}
```
