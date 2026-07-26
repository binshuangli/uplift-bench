# Model Card: Causal Forest

## Model details

| Field | Value |
|-------|-------|
| Method family | Non-parametric / Forest |
| Outcome type | Binary and Continuous |
| Requires propensity | Yes (estimated internally) |
| Base learner | Honest forest (econml) |
| Reference | Wager & Athey (2018) |
| Code | `src/uplift_bench/models/causal_forest_model.py` |

## Description

Causal Forest (Wager & Athey 2018) adapts random forests for CATE estimation using
*honest splitting*: the sample is split into a build set (for splits) and an estimation
set (for leaf CATE estimates). This reduces overfitting of the treatment effect estimates.

Implemented via `econml.dml.CausalForestDML` (DML orchestration around the GRF forest). Internally fits propensity and outcome nuisance
models.

## Known strengths

- Asymptotically normal pointwise CIs available (not exploited in this benchmark).
- Handles both binary and continuous outcomes.
- Wins on IHDP-s9 (Qini = 72.4) — best on that particular split.

## Known weaknesses / failure modes

- Slow on large datasets due to multiprocessing overhead (semaphore leak warning on
  Hillstrom; run errored).
- HP tuning is less impactful than for LightGBM-based models.
- Can underperform simple meta-learners when n is small (high variance honest splits).

## Benchmark performance (summary — paper run)

| Dataset | Qini | Notes |
|---------|------|-------|
| IHDP (avg 10 splits) | varies | Wins on s9 (72.4), poor on s1 (−6.0) |
| Jobs (avg 10 splits) | ~1.8 | Worst-performing model on Jobs |
| Hillstrom | 14.3 (error noted) | Run completed but with semaphore warning |
| Lenta | 1.6 | |

## Citation

```bibtex
@article{wager2018estimation,
  title={Estimation and inference of heterogeneous treatment effects using random forests},
  author={Wager, Stefan and Athey, Susan},
  journal={Journal of the American Statistical Association},
  year={2018}
}
```
