# Result provenance

Every result parquet row is stamped with `git_hash` and `git_dirty` at run time. Two
facts an auditor needs to interpret those stamps:

## The stamped hashes do not resolve on GitHub — by design, not by accident

This public repository's history is a **single squashed release commit** per version tag:
the development history (which contains reviewer correspondence and draft manuscripts) is
private, and each release is published as one clean tree via `git commit-tree`. The
`git_hash` values inside the released parquets were stamped during development and refer
to that private history, so `git cat-file` against this repository will not find them.

What the stamps still give you: **within** the released results, rows sharing a hash were
produced by the same source state, and the run-to-run boundaries (main run vs. extension
runs vs. audit arms) are recoverable from the hash groups.

**How to verify the released results** — by regeneration, not by hash checkout:

```bash
make repro FORCE=1 RESULTS_DIR=results_fresh   # rerun the core benchmark into a new dir
python scripts/check_results_md.py             # released leaderboards vs released parquets
# then compare results_fresh/ against results/ (rank statistics should agree; fold-level
# values match up to library/BLAS nondeterminism documented in the paper's Appendix)
```

`FORCE=1` matters: the released parquets are committed, and without it the orchestrator
resumes (skips) every job whose output already exists.

## `git_dirty` in released parquets

Rows with `git_dirty=True` predate a fix to the dirtiness check: the old check counted
**newly written result files themselves** as working-tree changes, so any run that wrote
into a committed results directory was flagged dirty by its own output. The current
`git_utils.is_dirty()` considers source files only (output directories are excluded), so
the flag now records what it was always meant to: whether *source* was modified relative
to the stamped commit. The affected released rows were produced by unmodified source
plus in-flight outputs; the science is unaffected, and the flag is retained rather than
rewritten because editing released parquets would be worse provenance than documenting
them.

## Known implementation deltas in released parquets

* **AUUC convention** (all releases): `metrics/ranking.py::auuc` subtracts an ATE/2 chord
  from the prefix-mean curve; a random ranking scores ~ +ATE/2, not 0. The shift is
  model-invariant per fold, so every reported statistic is unaffected (pinned by tests).
* **R-Learner / Causal Forest treatment nuisance** (releases <= v1.4.0): `model_t` was a
  regressor although `discrete_treatment=True` requires a classifier (EconML); caught by
  an external audit. The code was fixed in v1.4.0, and **as of v1.4.1 every released
  parquet is regenerated under the corrected nuisance** -- the R-Learner and Causal
  Forest cells were rerun across all result directories (they are the only estimators
  that construct a `model_t`; every other estimator's code path is untouched by the
  fix), so the released artifacts and the released code are one coherent execution.
  The pre-fix main-panel cells are archived in `results_pre141/`, and
  `scripts/nuisance_fix_check.py` reproduces the before/after comparison from them:
  Causal Forest cells essentially unchanged across the fix (cell-level Spearman +1.00),
  R-Learner mildly shifted, F1 intact (values in `results/tables/tab28_nuisancefix.tex`).

## Derived tables committed for fresh-checkout verification

The multi-GB prediction stores (`results_r4pred/predictions/`, `results_acic/predictions/`)
are not committed; two compact derived artifacts are, so the prediction-level headline
numbers verify on a fresh checkout without the ~6 h `make repro-r4pred`:

* `results_r4pred/fold_metrics_cache.parquet` — per-fold metric cache read by
  `r6_analyses.py`, `m2_risk_uncertainty.py`, and `f2_threshold.py`.
* `results_r4pred/gain_auuc_folds.parquet` + `gain_auuc_validation.json` — per-fold
  cumulative-gain/prefix-mean AUUC and √PEHE for the weighting decomposition (tab27),
  plus the all-folds validation statistics against `causalml.auuc_score`;
  `scripts/gain_auuc_check.py` recomputes the headline correlations from this table
  when the prediction stores are absent.

Regenerating the stores and rerunning the scripts reproduces both artifacts exactly.
