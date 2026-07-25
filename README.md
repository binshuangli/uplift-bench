# UpliftBench

**UpliftBench: Revealing Outcome-Regime and Objective Mismatch in Uplift Evaluation**

> Paper under review at KDD 2027 (Datasets & Benchmarks Track). This repository is the
> reproducibility package (code, data loaders, and results); the manuscript PDF is not
> included here and will be linked on acceptance.

---

## TL;DR

Published uplift benchmarks disagree on which estimator is best. We show that a substantial
part of the disagreement is not about *models* — it is about *metrics* — and separate it into
two findings, each identified only where a reference objective exists. We evaluate **12
estimators** under one **outer-test-isolated** protocol (repeated stratified CV with
nested tuning; no test-fold information enters any reported number)
across **seven dataset families** (synthetic, IHDP ×10, Jobs ×10, Hillstrom, Lenta, X5,
MegaFon), with **ACIC 2016** as an independent boundary-case validation family.

- **(F1) Outcome-regime sensitivity of cumulative-gain Qini.** On the evaluated heavy-tailed
  *continuous* benchmark (IHDP), Qini's model ranking shows **no detectable alignment** with effect
  accuracy (all-100 IHDP mean Spearman ρ = **+0.00**, 95% CI [−0.10, +0.10]; 10-split ρ ≈
  −0.15), while AUUC and uplift-at-*k* track effect accuracy (paired AUUC-over-Qini gap
  **+0.56** [+0.45, +0.66]). The divergence is **not** a scale artifact (rankings are
  affine-invariant), not attributable to a single unstable estimator, and persists under a
  second base learner (XGBoost). "Regime" is used **descriptively** — no scalar statistic
  (kurtosis included) separates failing from non-failing continuous settings — and the failure
  is **not** seen on the mild-tailed ACIC 2016 family.
- **(F2) Objective mismatch.** F2 has a **structural** half — ranking metrics depend on the
  scores only through their induced ranking, so they *provably cannot* identify the
  sign-threshold-optimal model (which needs the score *level*) — and an **empirical** half:
  in a Jobs case study, direct policy-risk selection beats random model selection while the
  Qini, AUUC, and uplift-at-*k* selectors do not (14–15% regret). **Calibrating the decision
  threshold closes most of the gap**, and the mismatch **vanishes** under a budgeted-value
  objective where rank suffices. This replicates under XGBoost.
- **Calibration is *not* a third finding.** Where uplift calibration is identified (randomized
  regimes), Qini and calibration are moderately *aligned*; the earlier pooled "disagreement"
  was an unidentified within-bin contrast on confounded data (see the paper's calibration
  appendix).

See the auto-generated [findings notes](results/) (produced by `make analyze`) for the full
numbers; the paper is under review.

---

## Quick start

```bash
# 0. Python 3.12 recommended — tested on CPython 3.12.13; requires >=3.11,<3.13.
python3.12 -m venv .venv && source .venv/bin/activate

# 1a. Install the EXACT frozen environment used for the paper (recommended):
make install-frozen
# 1b. ...or install with loose, upper-bounded deps:
#     make install

# 2. Download the public datasets (~2 GB; Criteo skipped by default):
make data-no-criteo

# 3. Smoke test — verify the pipeline end-to-end (~5 min):
make repro-smoke

# 4. Reproduce the CORE benchmark (~20 h, 8-core Apple Silicon, WORKERS=4):
#    four base-learner/extension runs -> analysis -> tables/figures/macros in results/
make repro
# ...or additionally run the prediction-saving re-run + ACIC 2016 (~6 h more), which the
# prediction-level auxiliary analyses (causalml variant, RATE, budget grid, risk
# uncertainty, threshold calibration) require — regenerates EVERY reported artifact:
make repro-full
```

Every command respects `make PY=python3.12 ...` and `make WORKERS=8 ...` overrides; nothing is
hardcoded to a specific interpreter. No LaTeX install is required to reproduce the results.

---

## Reproducing the results

`make repro` chains the core pipeline; `make repro-full` adds the prediction-level runs.
The sub-targets let you run it in pieces or on a cluster:

| Target | What it does | Output | ~Runtime |
|--------|--------------|--------|----------|
| `make repro-main` | LightGBM medium-tier run + collect | `results/` | ~9 h |
| `make repro-xgb` | XGBoost robustness run | `results_xgb/` | ~9 h |
| `make repro-m1ext` | Revenue-uplift + IHDP splits 10–29 | `results_m1ext/` | ~40 min |
| `make repro-ihdpval` | IHDP splits 30–99 (meta/forest learners) | `results_ihdpval/` | ~30 min |
| `make repro-r4pred` | prediction-saving re-run + ACIC 2016 (causalml-variant, RATE, budget grid, risk noise, threshold calibration) | `results_r4pred/`, `results_acic/` | ~6 h |
| `make repro-aux` | the three auxiliary runs above | — | ~10 h |
| `make repro` | core: main + aux runs, then analyze | `results/` | ~20 h |
| `make repro-full` | core + `repro-r4pred`, then analyze — every reported artifact | `results/` | ~26 h |
| `make repro-f1-notune` | fixed-hyperparameter F1 audit: the primary continuous panel with tuning **off** (is F1 an artifact of Qini tuning?) | `results_notune/` | ~25 min |
| `make repro-f1-auuctune` | the other tuning arm: same panel and budget, inner objective **AUUC** instead of Qini | `results_auuctune/` | ~1 h |
| `make repro-f1-revsynth` | third continuous family (Revenue-Synthetic) — tests whether F1 generalises beyond IHDP (it does not) | `results_revsynth/` | ~1 h |
| `make analyze` | run every analysis script | `results/{tables,figures}` | ~2 min |

`make analyze` runs the full analysis pipeline, regenerating the LaTeX tables, figures and
numeric macros into `results/{tables,figures}`:

| Script | Produces |
|--------|----------|
| `analyze_wp6.py` | per-regime leaderboards, CD diagrams, ECE tables (fig1–6, tab1–3) |
| `analyze_metric_reliability.py` | cross-metric agreement (figR1–R3), F1 exclusion sensitivity (tab4) |
| `outcome_transform_experiment.py` | F1 controlled probes (figR4, tab5/6/9) |
| `calibration_robustness.py` | identification-stratified calibration (tab7) |
| `ihdp_all100_validation.py` | all-100 IHDP validation (tab8, tab8b) |
| `m2_regret.py` | F2 cross-repeat policy regret + XGBoost replication (tab10, tab11) |
| `m2_selection_signal.py` | F2 selector ladder (tab12) |
| `pi1_kurtosis_sweep.py` | treatment-imbalance × kurtosis sweep (tab14) |
| `correlated_error_sweep.py` | correlated-error × tails sweep (tab17) |
| `r4_prediction_analyses.py`* | causalml variant, RATE, budget grid (tab13); builds the fold-metric cache |
| `r6_analyses.py`* | value-reference selectors, oracle-adjusted Qini (tab16) |
| `m2_risk_uncertainty.py`* | IPW risk noise, design-effect sensitivity (tab15) |
| `r9_analyses.py` | within-IHDP kurtosis (figR5), Jobs dependence (tab18) |
| `f2_threshold.py`* | calibrated-threshold experiment (tab19) |
| `policy_value_curves.py` | policy-value-vs-budget curves (fig7) |
| `dr_audit.py` | DR-Learner instability macros (tab20) |
| `notune_sensitivity.py` | F1 tuning-sensitivity macros: no-tune arm (tab21) + AUUC-tuned arm (tab23) |
| `f1_separating_covariates.py` | searches four ex-ante covariates for one that predicts the AUUC-over-Qini gap (tab22) |
| `revsynth_family.py` | third-family F1 replication test + the kurtosis comparison that rules out heavy tails (tab25) |

\* needs the prediction stores from `make repro-r4pred`; skipped with a clear message when
absent (that is what `make repro` vs `make repro-full` selects between). `r9_analyses.py`,
`dr_audit.py` and `notune_sensitivity.py` additionally need downloaded data
(`make data-no-criteo`) or a small dedicated run (`make repro-dr-audit`,
`make repro-f1-notune`) for *part* of their output; each still emits everything the
committed parquets determine and prints what is missing.

Two scripts sit outside `make analyze`: `check_results_md.py` (run by `make check-results`,
and also by `make test` and CI) verifies `RESULTS.md` against the committed parquets, and
`allocation_value.py` is an **exploratory** calibration-to-value probe that no reported
artifact depends on.

Every numeric artifact is **deterministic** (fixed seeds): regenerating from the committed
result parquets reproduces them byte-for-byte. Analysis
scripts skip any auxiliary result directory that is absent, so `make repro-main && make
analyze` already regenerates every artifact reachable from the main run alone
(auxiliary-dependent macros retain their committed values until the corresponding `repro-*`
target has been run).

Every result parquet is stamped with `git_hash`, the full Hydra `config_json`, and a UTC
`timestamp`. The exact environment is pinned in [`requirements.txt`](requirements.txt).

---

## Datasets

| Dataset | Regime | Outcome | n (bench) | p | Treatment % | Reference objective |
|---------|--------|---------|-----------|---|-------------|---------------------|
| Synthetic | Synthetic | binary | 2,000 | 8 | 50% | τᵢ known (√PEHE) |
| IHDP ×10 splits | Semi-synthetic | **continuous** | 672 | 25 | 19% | τᵢ known (√PEHE) |
| Jobs ×10 splits | Policy benchmark | binary | 2,570 | 17 | 9% | RCT-estimated policy risk |
| Hillstrom | Marketing RCT | binary | 64,000 | 15 | 67% | — (Qini proxy) |
| Lenta | Marketing RCT | binary | 10,000 | 193 | 75% | — (Qini proxy) |
| X5 | Marketing RCT | binary | 10,000 | 3 | 50% | — (Qini proxy) |
| MegaFon | Marketing RCT | binary | 10,000 | 50 | 50% | — (Qini proxy) |

Per-dataset datasheets — provenance, licensing, preprocessing, caveats — are in
[`docs/DATASETS.md`](docs/DATASETS.md).

Lenta / X5 / MegaFon are subsampled to 10K for the benchmark run. **IHDP is the only
continuous-outcome family** — the axis F1 is about; every loader also reports this as
`meta.outcome_type`, checked against the data by `UpliftDataset.validate()`.

### Bring your own data

Evaluate any dataset under the released protocol with the `from_frame` adapter — specify the
covariate, treatment, and outcome columns (plus optional propensity / reference-effect
columns); every estimator and metric then runs on it exactly as on the built-in datasets.

```python
import pandas as pd
from uplift_bench.data.base import UpliftDataset

df = pd.read_csv("my_experiment.csv")
ds = UpliftDataset.from_frame(
    df,
    treatment_col="treated",     # binary {0,1}
    outcome_col="revenue",       # continuous or binary
    propensity_col="p_treat",    # optional: enables the known-propensity policy
    ite_col="true_cate",         # optional: enables sqrt(PEHE)
    name="my_experiment",
)
# ds now plugs into the same runner, estimators, and metrics as the built-in loaders.
```

---

## Models

| Model | Family | Outcome | Reference |
|-------|--------|---------|-----------|
| S/T/X-Learner | Meta-learner | Any | Künzel et al. 2019 |
| R-Learner | Meta-learner | Any | Nie & Wager 2021 |
| DR-Learner | Meta-learner | Any | Kennedy 2020 |
| ClassTrans / TwoModel / SoloModel | Class-transformation | Binary | Jaskowski & Jaroszewicz 2012; Kane et al. 2014 |
| CausalForest | Non-parametric | Any | Wager & Athey 2018 |
| UpliftRF-KL/ED/Chi | Tree-based uplift | Binary | Rzepakowski & Jaroszewicz 2012 |

All meta-learners and class-transformation models share a **LightGBM base learner** with a
common hyperparameter search space (10-config random search, 2-fold inner CV). The XGBoost
robustness run (`make repro-xgb`) swaps in an XGBoost base learner.

---

## Evaluation protocol (outer-test-isolated)

1. Stratified 3-fold CV, repeated 3× with different seeds → 9 fold evaluations per cell.
2. All preprocessing (imputation, propensity estimation) fitted on the training fold only.
3. HP tuning via inner CV within the training fold only.
4. Six metrics computed per fold: √PEHE, policy risk, Qini (unnormalized), AUUC, policy value
   at k=30%, calibration ECE.

Each dataset is scored by the metric that is actually knowable: **√PEHE** (Synthetic, IHDP),
**policy risk** (Jobs), and the **Qini** proxy on the marketing RCTs. The paper's contribution
is precisely what goes wrong when the standard cumulative-gain Qini is used as the sole metric
outside its appropriate outcome regime (F1), and when any ranking metric is used as a stand-in
for the level-dependent deployment objective (F2).

---

## Project layout

```
src/uplift_bench/       # library code (data loaders, estimators, metrics, experiment runner)
configs/                # Hydra configs (datasets, models, bench, smoke)
scripts/
  run_bench.py                    # parallel benchmark orchestrator (tiers: medium, m1ext, ihdpval, ...)
  collect_results.py              # merge per-cell parquets -> master tables
  analyze_wp6.py                  # per-regime leaderboards, CD diagrams, fig1-7, tab1-3
  analyze_metric_reliability.py   # figR1-R3, F1 exclusion sensitivity (tab4)
  outcome_transform_experiment.py # F1 controlled experiment (figR4, tab5/6/9, counterexample)
  calibration_robustness.py       # calibration identification analysis (tab7)
  ihdp_all100_validation.py       # all-100 IHDP validation (tab8, tab8b)
  m2_regret.py                    # F2 cross-repeat policy regret + XGBoost replication (tab10, tab11)
results/                          # main LightGBM run — canonical result parquets (committed)
results_xgb/                      # XGBoost robustness run (parquets committed)
results_m1ext/, results_ihdpval/  # F1 extension + all-100 IHDP (parquets committed)
results_r4pred/, results_acic/    # prediction stores — large; regenerate via `make repro-r4pred`
model_cards/                      # one markdown card per estimator family
tests/                            # pytest unit + smoke tests
```

---

## Contributing / leaderboard

See [RESULTS.md](RESULTS.md) for the leaderboard and submission template, and
[CONTRIBUTING.md](CONTRIBUTING.md) for how to add a model or dataset. Submitted results are
independently re-run before merging.

---

## Citation

```bibtex
@article{li2026upliftbench,
  title  = {UpliftBench: Revealing Outcome-Regime and Objective Mismatch
            in Uplift Evaluation},
  author = {Li, Binshuang},
  year   = {2026},
  note   = {Under review}
}
```

Or use the [CITATION.cff](CITATION.cff) file.

---

## License

MIT — see [LICENSE](LICENSE).
