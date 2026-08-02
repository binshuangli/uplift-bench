# UpliftBench

**UpliftBench: Revealing Outcome-Regime and Objective Mismatch in Uplift Evaluation**

> This repository is the reproducibility package (code, data loaders, and results) for
> the paper *UpliftBench: Revealing Outcome-Regime and Objective Mismatch in Uplift
> Evaluation* (preprint forthcoming on arXiv; the link and citation entry will be added
> here once it is announced). The archival venue version will be linked when available.

---

## TL;DR

Published uplift benchmarks disagree on which estimator is best. We show that a substantial
part of the disagreement is not about *models* — it is about *metrics* — and separate it into
two findings, each identified only where a reference objective exists. We evaluate **12
estimators** under one **outer-test-isolated** protocol (repeated stratified CV with
nested tuning; no test-fold information enters any reported number)
across **seven dataset families** (synthetic, IHDP ×10, Jobs ×10, Hillstrom, Lenta, X5,
MegaFon), with **ACIC 2016** as an independent boundary-case validation family, plus a
**Criteo Uplift v2.1 supplement** (the canonical industry RCT, 1M tier; leaderboard rows in
[RESULTS.md](RESULTS.md), not used by the paper).

- **(F1) Outcome-regime sensitivity of Qini.** On the standard continuous
  benchmark (IHDP, the *primary* continuous panel), Qini's model ranking shows **no detectable
  alignment** with effect accuracy (all-100 IHDP mean Spearman ρ = **+0.07**, 95% CI
  [−0.03, +0.16]; 10-split ρ ≈ +0.02), while AUUC and uplift-at-*k* track effect accuracy
  (paired prefix-mean-AUUC-over-Qini gap **+0.49** [+0.40, +0.59]; the field's shipped
  cumulative-gain AUUC aligns better still, ρ ≈ +0.73). The divergence is **not** a scale
  artifact (rankings are affine-invariant), not attributable to a single unstable estimator,
  persists under a second base learner (XGBoost) and under three tuning arms (Qini-tuned,
  untuned, AUUC-tuned). **It is not a tail effect and it is one of three evaluated continuous
  families**: IHDP's primary panel is *mild*-tailed (median excess kurtosis −0.4) and fails;
  ACIC 2016 (gap +0.11, CI covering zero) and the genuinely heavy-tailed Revenue-Synthetic
  family (kurtosis ≈ 44; gap **−0.02**, indistinguishable from zero) do not. The
  three families form a gradient, and no ex-ante quantity tested predicts the metric-specific
  gap within a family — so compare ranking metrics on your own data rather than trusting
  either one.
- **(F2) Objective mismatch.** F2 has a **structural** half — ranking metrics depend on the
  scores only through their induced ranking, so they *provably cannot* identify the
  sign-threshold-optimal model (which needs the score *level*) — and an **empirical** half:
  in a *within-sample* Jobs case study (released split rotation), direct policy-risk
  selection yields lower benchmark regret than random model selection while the Qini, AUUC,
  and uplift-at-*k* selectors do not (14–15% regret). **Calibrating the decision
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
#    FORCE=1 is REQUIRED for actual recomputation: the released parquets are committed,
#    and without it every existing job is skipped (resume semantics). Note the auxiliary
#    run directories (results_xgb/, results_m1ext/, ...) are fixed paths, so a forced
#    rerun overwrites the released auxiliary outputs in place -- work in a fresh clone
#    (or fresh branch) if you want to diff regenerated outputs against the released ones.
make repro FORCE=1
# ...or additionally run the prediction-saving re-run + ACIC 2016 (~6 h more), which the
# prediction-level auxiliary analyses (causalml variant, RATE, budget grid, risk
# uncertainty, threshold calibration) require — regenerates EVERY reported artifact
# (FORCE=1 required for the same reason as above):
make repro-full FORCE=1
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
| `r19_analyses.py` | metric-artifact consistency audit (tab26) |
| `gain_auuc_check.py`* | cumulative-gain (library) vs prefix-mean AUUC weighting decomposition (tab27); validated fold-by-fold against `causalml.auuc_score`; falls back to the committed `results_r4pred/gain_auuc_folds.parquet` when prediction stores are absent |
| `nuisance_fix_check.py` | F1 across the EconML treatment-nuisance fix, vs the archived pre-fix cells in `results_pre141/` (tab28) |

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
`timestamp`. Direct dependencies are pinned in [`requirements.txt`](requirements.txt); the FULL transitive environment (68 packages) is frozen in [`requirements-lock.txt`](requirements-lock.txt), which is what CI installs. Result provenance (git-hash stamps, the dirty flag, and how to verify released results by regeneration) is documented in [`PROVENANCE.md`](PROVENANCE.md).

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
| Criteo† *(supplement)* | Marketing RCT | binary | 10,000 (+100K probe) | 12 | 85% | — (Qini proxy) |
| Revenue-Synthetic ×10 *(auxiliary)* | Synthetic | **continuous** | 4,000 | 10 | 49% | τᵢ known (√PEHE) |
| ACIC 2016 ×18 *(auxiliary)* | Semi-synthetic | **continuous** | 4,802 | 79 | ~37% | τᵢ known (√PEHE) |

†Criteo (13.9M rows; 1M tier subsampled to the released 10K cap, plus a nine-estimator
100K probe) is a **v1.4.3 supplementary release**: leaderboard in
[RESULTS.md](RESULTS.md), outside the primary 25-instance accounting, not used by the
paper's findings. The auxiliary continuous families serve as F1's boundary cases.

Per-dataset datasheets — provenance, licensing, preprocessing, caveats — are in
[`docs/DATASETS.md`](docs/DATASETS.md).

Lenta / X5 / MegaFon are subsampled to 10K for the benchmark run. **IHDP is the primary
continuous-outcome panel**; the auxiliary Revenue-Synthetic and ACIC 2016 families are also
continuous (with known effects) and serve as F1's boundary cases. Every loader reports the
regime as `meta.outcome_type`, checked against the data by `UpliftDataset.validate()`.

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
# Route into the benchmark: register your loader with the PUBLIC API. Same-process use:
from uplift_bench.data.registry import register_loader
register_loader("my_experiment", "my_module:load_my_experiment")  # returns `ds` above
```

The Hydra runner executes in its own process, so pass the registration through the
environment instead (the registry reads `UPLIFT_BENCH_LOADERS` at import):

```bash
UPLIFT_BENCH_LOADERS="my_experiment=my_module:load_my_experiment" \
python -m uplift_bench.experiments.runner dataset=synthetic model=t_learner \
  dataset.name=my_experiment +dataset.loader_name=my_experiment \
  results_dir=results_mine/
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
