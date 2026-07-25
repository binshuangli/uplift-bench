# Model Card: [Method Name]

## Model details

| Field | Value |
|-------|-------|
| Method family | Meta-learner / Class-transformation / Tree-based / Other |
| Outcome type | Binary / Continuous / Both |
| Requires propensity | Yes / No |
| Requires ground-truth effects | No |
| Base learner | LightGBM / XGBoost / Custom |
| Reference | [citation] |
| Code | [link or `src/uplift_bench/models/your_model.py`] |

## Description

Brief description of how the model estimates uplift, including any key assumptions
(e.g., unconfoundedness, positivity, binary outcome).

## Known strengths

- ...

## Known weaknesses / failure modes

- ...

## Leaderboard performance (summary)

| Dataset | Qini (mean) | 95% CI | Rank |
|---------|-------------|--------|------|
| ... | | | |

## Hyperparameter space

If using a non-standard HP space (i.e., not the shared LightGBM space), list it here.

## Citation

```bibtex
@article{...}
```
