# Draft GitHub issue for uber/causalml — ready to file

> **Who files it / when:** file this from your own GitHub account. Note the double-blind
> interaction: do NOT cite "our issue" in the anonymized submission (the paper's claim
> stands on the reproducible behaviour itself), and avoid linking the issue back to the
> paper until after review. Filing the issue itself is fine — it does not reference the
> manuscript.

---

**Title:** `qini_score` silently computes on continuous outcomes, where the Qini
coefficient is not well-defined

**Labels:** bug / documentation

## Description

`causalml.metrics.qini_score` (and the underlying `get_qini` cumulative-gain computation)
accepts a **continuous** outcome column and returns a number with no warning. The Qini
coefficient is defined for binary response (Radcliffe, 2007): it accumulates outcome
*sums* weighted by the treated/control ratio, so on a continuous outcome a few
large-magnitude observations can dominate the curve and the resulting model ranking can
diverge from ground-truth effect accuracy — silently.

For contrast, `scikit-uplift` guards against this: `sklift.metrics.qini_auc_score` calls
`check_is_binary(y_true)` and raises on non-binary outcomes.

## Minimal reproduction

```python
import numpy as np
import pandas as pd
from causalml.metrics import qini_score

rng = np.random.default_rng(0)
n = 2000
df = pd.DataFrame({
    "y": rng.normal(5, 3, n),        # CONTINUOUS outcome
    "w": rng.integers(0, 2, n),      # binary treatment
    "model": rng.normal(size=n),     # some uplift score
})
print(qini_score(df, outcome_col="y", treatment_col="w"))
# -> returns a float (e.g. {'model': -7.74}) with no warning or error
```

Expected: either a `ValueError` (like scikit-uplift) or at minimum a `UserWarning` that
Qini is being computed outside its binary-response design regime, with a pointer to
AUUC-style average-based metrics for continuous outcomes.

## Why it matters

In a controlled comparison on continuous-outcome benchmark data (IHDP semi-synthetic +
synthetic, where true effects are known), the Qini-based model ranking was essentially
uncorrelated with ground-truth CATE accuracy (mean per-dataset Spearman ρ ≈ −0.15 vs
−√PEHE), while the mean-based AUUC ranking remained substantially positively correlated
(ρ ≈ +0.6) on the same models and datasets. The gap persists when numerically unstable
estimators (DR-/R-Learner) are excluded and under a different base learner. Continuous
(e.g. revenue) outcomes are an active uplift setting, so the silent path is easy to hit.

## Suggested fix

1. Add an outcome-type check in `qini_score`/`get_qini`: raise or warn when
   `y` is not binary {0,1}.
2. Document the binary-response assumption in the metric docstrings, and point users to
   AUUC for continuous outcomes.

Happy to open a PR implementing option 1 (warning by default, `strict=True` to raise) if
maintainers agree on the desired behaviour.

---

## Ready-to-submit patch

A minimal patch implementing the warning is included alongside this draft
(`docs/causalml_qini_warning.patch`, +11 lines in `causalml/metrics/visualize.py`).
Verified behaviour: `get_qini`/`qini_score` emit a `UserWarning` on a non-binary outcome
column and remain silent on binary outcomes. To turn it into a PR:

```bash
git clone https://github.com/uber/causalml && cd causalml
git checkout -b qini-continuous-outcome-warning
git apply /path/to/causalml_qini_warning.patch
# add a small test, run their test suite, push, open PR referencing the issue
```
