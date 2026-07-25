# Model Card: Uplift Random Forests (UpliftRF-KL / ED / Chi)

## Model details

| Field | Value |
|-------|-------|
| Method family | Tree-based uplift |
| Outcome type | Binary only |
| Requires propensity | No |
| Base learner | Random forest (scikit-uplift) |
| Reference | Rzepakowski & Jaroszewicz (2012) |
| Code | `src/uplift_bench/models/uplift_forest.py` |

## Description

Uplift random forests modify the splitting criterion of a decision tree to maximise
the divergence between treatment and control outcome distributions in each leaf:

- **UpliftRF-KL**: KL-divergence splitting criterion.
- **UpliftRF-ED**: Euclidean distance (squared difference of means) splitting criterion.
- **UpliftRF-Chi**: χ² statistic splitting criterion.

All three are implemented via `scikit-uplift`. They require binary outcomes and do not
use a LightGBM base learner (they are native tree forests).

## Known strengths

- No propensity model required; splits directly on uplift divergence.
- Competitive on Jobs (2nd–3rd behind ClassTrans on all splits).
- Interpretable tree structure.

## Known weaknesses / failure modes

- **Slow on large datasets**: timed out (>3600s) on Hillstrom (n=64K).
  Runtime scales poorly with n.
- Not applicable to continuous outcomes.
- HP space is different from the shared LightGBM space; HP tuning uses forest-specific
  parameters (n_estimators, max_depth, min_samples_leaf).

## Benchmark performance (summary — paper run)

| Dataset | UpliftRF-KL | UpliftRF-ED | UpliftRF-Chi |
|---------|------------|------------|-------------|
| Jobs (avg 10 splits) | 5.3 | 5.3 | 5.3 |
| Hillstrom | TO | TO | TO |
| Lenta | 5.9 | −2.0 | 0.4 |
| X5 | −1.1 | −2.5 | −1.9 |

`TO` = timed out (>3600s). Results on Hillstrom pending cluster re-run.

## Citation

```bibtex
@article{rzepakowski2012decision,
  title={Decision trees for uplift modeling with single and multiple treatments},
  author={Rzepakowski, Piotr and Jaroszewicz, Szymon},
  journal={Knowledge and Information Systems},
  year={2012}
}
```
