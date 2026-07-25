.PHONY: install install-frozen data data-no-criteo \
        repro repro-full repro-main repro-aux repro-xgb repro-m1ext repro-ihdpval \
        repro-r4pred analyze \
        repro-smoke smoke test test-fast lint fmt clean check-install check-results

# Portable Python: defaults to whatever `python` resolves to in the active environment
# (create a venv first — see README). Override with `make PY=python3.12 ...` if needed.
PY ?= python
WORKERS ?= 4

RESULTS_DIR    := results
DATA_DIR       := data
XGB_DIR        := results_xgb
M1EXT_DIR      := results_m1ext
IHDPVAL_DIR    := results_ihdpval
DRAUDIT_DIR    := results_drB50
NOTUNE_DIR     := results_notune
AUUCTUNE_DIR ?= results_auuctune
META_LEARNERS  := s_learner,t_learner,x_learner,r_learner,dr_learner,causal_forest

# ── Installation ──────────────────────────────────────────────────────────────

# Editable install (loose, upper-bounded deps). Requires Python >=3.11,<3.13.
install:
	$(PY) -m pip install -e ".[dev]"

# Exact frozen environment used for the paper (recommended for close reproduction).
# requirements.txt pins the dev/test tools too, so `make test` / `make repro-smoke`
# work from this path.
install-frozen:
	$(PY) -m pip install -r requirements.txt
	$(PY) -m pip install -e . --no-deps

# ── Data download ─────────────────────────────────────────────────────────────

data:
	$(PY) -m uplift_bench.data.download_all --data-dir $(DATA_DIR)

data-no-criteo:
	$(PY) -m uplift_bench.data.download_all --data-dir $(DATA_DIR) --skip-criteo

# ── Full reproduction ─────────────────────────────────────────────────────────
# `make repro` reproduces the CORE benchmark (the four base-learner/extension runs) and
# regenerates every artifact derivable from them into results/ — no LaTeX install
# required. ~20 h on an 8-core Apple Silicon machine (WORKERS=4). For a partial run use
# the sub-targets below (repro-main, repro-aux, analyze).
repro: repro-main repro-aux analyze
	@echo "=== Reproduced: core benchmark results + reported tables/figures/macros in results/ ==="

# `make repro-full` additionally runs the prediction-saving re-run + ACIC 2016
# (repro-r4pred, ~6 h), which the prediction-level auxiliary analyses (causalml variant,
# RATE, budget grid, risk uncertainty, threshold calibration) require; the final analyze
# pass then regenerates ALL reported artifacts.
repro-full: repro-main repro-aux repro-r4pred analyze
	@echo "=== Reproduced: ALL benchmark results + every reported artifact in results/ ==="

# Main leakage-free run: LightGBM, medium tier (synthetic + IHDP x10 + Jobs x10 + 4 RCTs).
# ~9 h. Produces $(RESULTS_DIR)/master_summary.parquet.
repro-main: data-no-criteo
	$(PY) scripts/run_bench.py --tier medium --workers $(WORKERS) \
		--results-dir $(RESULTS_DIR) --data-dir $(DATA_DIR)
	$(PY) scripts/collect_results.py --results-dir $(RESULTS_DIR)

# Auxiliary runs the paper depends on (XGBoost robustness, M1 extension, all-100 IHDP).
repro-aux: repro-xgb repro-m1ext repro-ihdpval

repro-xgb: data-no-criteo
	$(PY) scripts/run_bench.py --tier medium --base-learner xgboost --workers $(WORKERS) \
		--results-dir $(XGB_DIR) --data-dir $(DATA_DIR)
	$(PY) scripts/collect_results.py --results-dir $(XGB_DIR)

repro-m1ext: data-no-criteo
	$(PY) scripts/run_bench.py --tier m1ext --workers $(WORKERS) \
		--results-dir $(M1EXT_DIR) --data-dir $(DATA_DIR)
	$(PY) scripts/collect_results.py --results-dir $(M1EXT_DIR)

repro-ihdpval: data-no-criteo
	$(PY) scripts/run_bench.py --tier ihdpval --models $(META_LEARNERS) --n-samples 6 \
		--workers $(WORKERS) --results-dir $(IHDPVAL_DIR) --data-dir $(DATA_DIR)
	$(PY) scripts/collect_results.py --results-dir $(IHDPVAL_DIR)

# R4 reviewer analyses: prediction-saving re-run (identical protocol; metrics verified
# to reproduce the canonical parquets exactly) + the ACIC 2016 auxiliary family.
# Feeds scripts/r4_prediction_analyses.py (causalml-variant / RATE / budget grid).
repro-r4pred: data-no-criteo
	$(PY) scripts/run_bench.py --tier small --save-predictions \
		--workers $(WORKERS) --results-dir results_r4pred --data-dir $(DATA_DIR)
	$(PY) scripts/run_bench.py --tier m1ext --save-predictions \
		--workers $(WORKERS) --results-dir results_r4pred --data-dir $(DATA_DIR)
	$(PY) scripts/run_bench.py --tier ihdpval --models $(META_LEARNERS) --n-samples 6 \
		--save-predictions --workers $(WORKERS) --results-dir results_r4pred --data-dir $(DATA_DIR)
	$(PY) scripts/run_bench.py --tier acic --models $(META_LEARNERS) --save-predictions \
		--workers $(WORKERS) --results-dir results_acic --data-dir $(DATA_DIR)

# DR-Learner tuning-budget audit (Appendix D): is the AIPW blow-up a budget artifact?
# Two IHDP splits x two tuning budgets, DR-Learner only (~minutes). Dataset names are
# tagged so scripts/dr_audit.py can tell the two budgets apart.
.PHONY: repro-dr-audit
repro-dr-audit: data-no-criteo
	@for s in 0 8; do \
	  $(PY) -m uplift_bench.experiments.runner dataset=ihdp model=dr_learner \
	    dataset.loader_kwargs.split_idx=$$s dataset.name=ihdp_B10_s$$s \
	    tuning.n_samples=10 results_dir=$(DRAUDIT_DIR)/ || exit 1; \
	  $(PY) -m uplift_bench.experiments.runner dataset=ihdp model=dr_learner \
	    dataset.loader_kwargs.split_idx=$$s dataset.name=ihdp_bigB_s$$s \
	    tuning.n_samples=50 results_dir=$(DRAUDIT_DIR)/ || exit 1; \
	done
	$(PY) scripts/dr_audit.py
	@echo "=== DR budget audit: $(DRAUDIT_DIR)/ + results/tables/tab20_dr.tex ==="

# Fixed-hyperparameter F1 sensitivity (Appendix: app:notune): rerun the primary continuous
# panel with tuning switched OFF, to test whether F1 is an artifact of the Qini-tuned
# candidate panel. 6 estimators x 10 IHDP splits at library defaults (~30 min).
.PHONY: repro-f1-notune
repro-f1-notune: data-no-criteo
	@for s in 0 1 2 3 4 5 6 7 8 9; do \
	  for m in s_learner t_learner x_learner r_learner dr_learner causal_forest; do \
	    $(PY) -m uplift_bench.experiments.runner dataset=ihdp model=$$m \
	      dataset.loader_kwargs.split_idx=$$s dataset.name=ihdp_s$$s \
	      n_folds=3 n_seeds=3 tuning.enabled=false \
	      results_dir=$(NOTUNE_DIR)/ || exit 1; \
	  done; \
	done
	$(PY) scripts/notune_sensitivity.py
	@echo "=== no-tune F1 audit: $(NOTUNE_DIR)/ + results/tables/tab21_notune.tex ==="

# R14: the other half of the tuning-objective sensitivity. Tuning is ON at the released
# budget (B=10, 2 inner folds) but the inner criterion is AUUC instead of Qini, so the
# candidate panel is selected by the metric F1 says is the *good* one. Together with
# repro-f1-notune this brackets the objection that F1 is an artifact of Qini-tuned
# selection: one arm removes tuning, this one inverts it. 6 estimators x 10 splits (~1h).
.PHONY: repro-f1-auuctune
repro-f1-auuctune: data-no-criteo
	@for s in 0 1 2 3 4 5 6 7 8 9; do \
	  for m in s_learner t_learner x_learner r_learner dr_learner causal_forest; do \
	    $(PY) -m uplift_bench.experiments.runner dataset=ihdp model=$$m \
	      dataset.loader_kwargs.split_idx=$$s dataset.name=ihdp_s$$s \
	      n_folds=3 n_seeds=3 tuning.n_samples=10 tuning.inner_folds=2 \
	      tuning.objective=auuc results_dir=$(AUUCTUNE_DIR)/ || exit 1; \
	  done; \
	done
	$(PY) scripts/notune_sensitivity.py
	@echo "=== AUUC-tuned F1 arm: $(AUUCTUNE_DIR)/ + results/tables/tab23_auuctune.tex ==="

# ── Analysis: regenerate every paper figure/table/macro, then sync into paper/ ──
# Reads $(RESULTS_DIR) plus the auxiliary result dirs if present (scripts skip missing
# ones gracefully). All outputs are deterministic (seeded).
analyze:
	$(PY) scripts/analyze_wp6.py --results-dir $(RESULTS_DIR) --out-dir $(RESULTS_DIR)
	$(PY) scripts/analyze_metric_reliability.py --results-dir $(RESULTS_DIR) --out-dir $(RESULTS_DIR)
	$(PY) scripts/outcome_transform_experiment.py --out-dir $(RESULTS_DIR)
	$(PY) scripts/calibration_robustness.py
	$(PY) scripts/ihdp_all100_validation.py
	$(PY) scripts/m2_regret.py
	$(PY) scripts/m2_selection_signal.py
	$(PY) scripts/pi1_kurtosis_sweep.py
	$(PY) scripts/correlated_error_sweep.py
	@# r4_prediction_analyses builds results_r4pred/fold_metrics_cache.parquet, which
	@# r6_analyses / m2_risk_uncertainty / f2_threshold all read -- so it must run FIRST.
	@if [ -d results_r4pred/predictions ]; then \
	  $(PY) scripts/r4_prediction_analyses.py \
	    --pred-dirs results_r4pred/predictions results_acic/predictions \
	    --cache results_r4pred/fold_metrics_cache.parquet --out-dir $(RESULTS_DIR); \
	else echo "skipping r4_prediction_analyses (no results_r4pred/predictions)"; fi
	@if [ -f results_r4pred/fold_metrics_cache.parquet ]; then \
	  $(PY) scripts/r6_analyses.py; \
	else echo "skipping r6_analyses (no results_r4pred/fold_metrics_cache.parquet)"; fi
	@# needs the Jobs prediction store (see repro-r4pred); skipped if absent.
	@if [ -d results_r4pred/predictions ]; then \
	  $(PY) scripts/m2_risk_uncertainty.py; \
	else echo "skipping m2_risk_uncertainty (no results_r4pred/predictions)"; fi
	$(PY) scripts/r9_analyses.py
	@if [ -d results_r4pred/predictions ]; then \
	  $(PY) scripts/f2_threshold.py; \
	else echo "skipping f2_threshold (no results_r4pred/predictions)"; fi
	@# Policy-value-vs-budget curves (Fig. 4b); reads the canonical result rows.
	$(PY) scripts/policy_value_curves.py
	@# DR-Learner instability audit macros (Appendix D). The Qini/root-PEHE figures come
	@# from the committed parquets; the B=10-vs-B=50 pair needs `make repro-dr-audit`.
	$(PY) scripts/dr_audit.py
	@# Fixed-hyperparameter F1 sensitivity macros; needs `make repro-f1-notune`, and
	@# emits placeholders naming that target when results_notune/ is absent.
	$(PY) scripts/notune_sensitivity.py
	@# Sync into paper/ only when the manuscript sources are present (they are not part
	@# of the public reproducibility package); the artifacts always land in results/.
	@if [ -f paper/main.tex ]; then $(MAKE) sync-paper; \
	else echo "skipping sync-paper (no paper/ sources; artifacts are in $(RESULTS_DIR)/{tables,figures})"; fi

# ── Smoke reproduction (~5 min): end-to-end pipeline check on tiny data ─────────
repro-smoke: test smoke
	@echo "=== Smoke reproduction complete ==="

# Smoke output goes to its own directory so it can NEVER clobber the canonical
# committed result parquets in results/ (found in the R5 artifact self-review).
smoke:
	$(PY) -m uplift_bench.experiments.runner --config-name=smoke \
		dataset=synthetic model=t_learner results_dir=results_smoke/
	@echo "=== Smoke results: results_smoke/ (canonical results/ untouched) ==="

# ── Testing / quality ─────────────────────────────────────────────────────────

# Fail fast with an actionable message if the package was never installed into the
# interpreter being used: otherwise pytest reports a wall of import errors that do not
# say "you skipped the install step".
.PHONY: check-install
check-install:
	@$(PY) -c "import uplift_bench" 2>/dev/null || { \
	  echo "ERROR: 'uplift_bench' is not importable by $(PY)."; \
	  echo "       Install it first:  make install-frozen   (or: make install)"; \
	  echo "       Then re-run this target. See README.md 'Quick start'."; \
	  exit 1; }

# Consistency of the hand-written public leaderboard against the committed parquets.
.PHONY: check-results
check-results:
	$(PY) scripts/check_results_md.py

test: check-install
	$(PY) -m pytest tests/ -v
	$(PY) scripts/check_results_md.py

test-fast: check-install
	$(PY) -m pytest tests/ -v -m "not network" --timeout=60

# scripts/ is linted too: it produces every reported number (a merge-collision bug in
# scripts/ once shipped precisely because it was outside lint and CI).
lint:
	$(PY) -m ruff check src/ tests/ scripts/
	$(PY) -m black --check src/ tests/

fmt:
	$(PY) -m ruff check --fix src/ tests/ scripts/
	$(PY) -m black src/ tests/

# ── Cleanup ───────────────────────────────────────────────────────────────────

clean:
	find . -name "__pycache__" -exec rm -rf {} + 2>/dev/null; true
	find . -name "*.pyc" -delete 2>/dev/null; true
	rm -f paper/main.aux paper/main.bbl paper/main.blg paper/main.log paper/main.out 2>/dev/null; true

# ── Private manuscript targets (paper/, excluded from the public package) ──────
# Provides sync-paper, paper, repro-paper, sync-paper-kdd, paper-kdd, docx when the
# manuscript sources are present; silently skipped otherwise.
-include paper/Makefile.paper
