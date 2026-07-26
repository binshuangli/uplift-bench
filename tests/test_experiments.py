"""WP4 tests: leakage guards, propensity policy, and the end-to-end smoke run.

All tests use the in-memory synthetic dataset — no network, no large data.
"""

from __future__ import annotations

import numpy as np
import pytest

from uplift_bench.data.synthetic import load_synthetic
from uplift_bench.experiments import propensity as prop_mod
from uplift_bench.experiments.evaluation import outcome_is_binary
from uplift_bench.experiments.propensity import resolve_propensity
from uplift_bench.experiments.runner import METRIC_NAMES, run_experiment, summarize
from uplift_bench.experiments.splitting import (
    assert_no_leakage,
    make_strat_labels,
    repeated_stratified_kfold,
)

# ---------------------------------------------------------------------------
# Leakage guards
# ---------------------------------------------------------------------------


class TestLeakageGuards:
    def test_assert_no_leakage_passes_on_disjoint(self):
        train = np.array([0, 1, 2, 3])
        test = np.array([4, 5])
        assert_no_leakage(train, test, n_total=6)  # no raise

    def test_assert_no_leakage_catches_overlap(self):
        train = np.array([0, 1, 2, 3])
        test = np.array([3, 4, 5])  # 3 appears in both
        with pytest.raises(AssertionError, match="LEAKAGE"):
            assert_no_leakage(train, test, n_total=6)

    def test_assert_no_leakage_catches_incomplete_coverage(self):
        train = np.array([0, 1])
        test = np.array([2])  # missing index 3
        with pytest.raises(AssertionError, match="coverage"):
            assert_no_leakage(train, test, n_total=4)

    def test_repeated_kfold_all_folds_disjoint_and_complete(self):
        ds = load_synthetic(n=400, seed=0)
        n = ds.meta.n
        for seed_idx, fold_idx, tr, te in repeated_stratified_kfold(
            ds.treatment, ds.outcome, n_folds=5, n_seeds=2, base_seed=42
        ):
            assert_no_leakage(tr, te, n)  # would raise on any leak

    def test_repeated_kfold_test_folds_partition_within_repeat(self):
        """Within one repeat, the union of test folds covers every index once."""
        ds = load_synthetic(n=300, seed=1)
        n = ds.meta.n
        test_union: list[int] = []
        for seed_idx, fold_idx, tr, te in repeated_stratified_kfold(
            ds.treatment, ds.outcome, n_folds=5, n_seeds=1, base_seed=7
        ):
            test_union.extend(te.tolist())
        assert sorted(test_union) == list(range(n))

    def test_stratification_preserves_treatment_fraction(self):
        ds = load_synthetic(n=2000, treatment_fraction=0.3, seed=2)
        overall = float(ds.treatment.mean())
        for _, _, tr, te in repeated_stratified_kfold(
            ds.treatment, ds.outcome, n_folds=5, n_seeds=1, base_seed=0
        ):
            test_frac = float(ds.treatment.iloc[te].mean())
            assert (
                abs(test_frac - overall) < 0.05
            ), f"Test fold treatment fraction {test_frac:.3f} deviates from {overall:.3f}"

    def test_strat_labels_handle_continuous_outcome(self):
        ds = load_synthetic(n=300, binary_outcome=False, seed=3)
        labels = make_strat_labels(ds.treatment, ds.outcome)
        assert len(labels) == ds.meta.n
        # Should combine treatment (0/1) with quantile bins -> more than 2 classes
        assert len(np.unique(labels)) > 2


# ---------------------------------------------------------------------------
# Propensity policy
# ---------------------------------------------------------------------------


class TestPropensityPolicy:
    def test_known_propensity_used_directly(self):
        ds = load_synthetic(n=300, treatment_fraction=0.5, known_propensity=True, seed=0)
        train_idx = np.arange(200)
        test_idx = np.arange(200, 300)
        train_p, test_p, policy = resolve_propensity(ds, train_idx, test_idx)
        assert policy == "known"
        assert np.allclose(train_p, 0.5)
        assert np.allclose(test_p, 0.5)

    def test_observational_propensity_estimated(self):
        ds = load_synthetic(n=400, known_propensity=False, seed=0)
        assert ds.propensity is None
        train_idx = np.arange(300)
        test_idx = np.arange(300, 400)
        train_p, test_p, policy = resolve_propensity(ds, train_idx, test_idx)
        assert policy == "estimated_on_train"
        # estimated probabilities must be valid and clipped into (0,1)
        assert (train_p > 0).all() and (train_p < 1).all()
        assert (test_p > 0).all() and (test_p < 1).all()
        assert len(test_p) == len(test_idx)

    def test_propensity_estimator_fits_on_train_only(self, monkeypatch):
        """Leakage guard: the propensity model must never see test-fold rows."""
        ds = load_synthetic(n=400, known_propensity=False, seed=0)
        train_idx = np.arange(300)
        test_idx = np.arange(300, 400)

        seen_fit_sizes = {}

        real_make = prop_mod.make_base_learner

        def spy_make(*args, **kwargs):
            est = real_make(*args, **kwargs)
            orig_fit = est.fit

            def wrapped_fit(X, y, *a, **k):
                seen_fit_sizes["n"] = len(X)
                return orig_fit(X, y, *a, **k)

            est.fit = wrapped_fit
            return est

        monkeypatch.setattr(prop_mod, "make_base_learner", spy_make)
        resolve_propensity(ds, train_idx, test_idx)
        assert seen_fit_sizes["n"] == len(train_idx), (
            f"Propensity model was fit on {seen_fit_sizes['n']} rows, "
            f"expected only the {len(train_idx)} train rows"
        )


# ---------------------------------------------------------------------------
# End-to-end smoke run
# ---------------------------------------------------------------------------


def _smoke_cfg(model="t_learner", binary=True, has_gt=True, known_prop=True):
    return {
        "seed": 42,
        "n_folds": 2,
        "n_seeds": 1,
        "smoke": True,
        "dataset": {
            "name": "synthetic",
            "loader_kwargs": {
                "n": 400,
                "n_features": 5,
                "binary_outcome": binary,
                "has_ground_truth": has_gt,
                "known_propensity": known_prop,
            },
        },
        "model": {"name": model, "base_learner": "lightgbm"},
        "tuning": {"enabled": True, "n_samples": 2, "inner_folds": 2},
        "metrics": {"budget_k": 0.3},
    }


class TestEndToEnd:
    def test_smoke_runs_and_returns_expected_columns(self):
        raw = run_experiment(_smoke_cfg())
        assert len(raw) == 2  # 1 seed x 2 folds
        assert (raw["status"] == "ok").all()
        for col in ("dataset", "model", "git_hash", "config_json", "propensity_policy"):
            assert col in raw.columns
        for m in METRIC_NAMES:
            assert m in raw.columns
        # ranking + ground-truth metrics should be present (synthetic has ITE)
        assert raw["qini"].notna().all()
        assert raw["pehe"].notna().all()

    def test_results_include_cis(self):
        raw = run_experiment(_smoke_cfg())
        summary = summarize(raw)
        assert len(summary) == 1
        for col in ("qini_mean", "qini_ci_low", "qini_ci_high", "qini_n"):
            assert col in summary.columns
        assert summary["qini_ci_low"].iloc[0] <= summary["qini_mean"].iloc[0]
        assert summary["qini_mean"].iloc[0] <= summary["qini_ci_high"].iloc[0]

    def test_run_is_reproducible(self):
        """Same config + seed → identical metric values (config fully reproduces a run)."""
        raw1 = run_experiment(_smoke_cfg())
        raw2 = run_experiment(_smoke_cfg())
        for m in ("qini", "auuc", "pehe", "epsilon_ate"):
            np.testing.assert_allclose(
                raw1[m].to_numpy(),
                raw2[m].to_numpy(),
                rtol=1e-9,
                atol=1e-9,
                err_msg=f"metric {m} not reproducible",
            )

    def test_continuous_outcome_skips_binary_model(self):
        cfg = _smoke_cfg(model="class_transformation", binary=False)
        raw = run_experiment(cfg)
        assert raw["status"].str.startswith("skipped").all()

    def test_observational_dataset_records_estimated_policy(self):
        cfg = _smoke_cfg(known_prop=False)
        raw = run_experiment(cfg)
        assert (raw["propensity_policy"] == "estimated_on_train").all()

    def test_meta_learner_runs_on_continuous_outcome(self):
        cfg = _smoke_cfg(model="t_learner", binary=False)
        raw = run_experiment(cfg)
        assert (raw["status"] == "ok").all()
        assert raw["pehe"].notna().all()


# ---------------------------------------------------------------------------
# outcome_is_binary helper
# ---------------------------------------------------------------------------


def test_outcome_is_binary():
    assert outcome_is_binary(np.array([0, 1, 0, 1]))
    assert outcome_is_binary(np.array([0.0, 1.0, 1.0]))
    assert not outcome_is_binary(np.array([0.1, 0.5, 0.9, 1.2]))


# ---------------------------------------------------------------------------
# Inner-loop tuning objective (R14)
# ---------------------------------------------------------------------------


class TestTuningObjective:
    """`tuning.objective` must be real, not decorative.

    It was a dead config key until R14: `configs/config.yaml` declared
    `objective: qini` while the criterion was hardcoded, so a paper claiming a
    Qini-tuned panel was accidentally right for the wrong reason and no other arm
    was runnable. These tests pin the plumbing.
    """

    def test_registry_exposes_the_documented_objectives(self):
        from uplift_bench.experiments.tuning import _OBJECTIVES

        assert {"qini", "auuc", "uplift_at_k"} <= set(_OBJECTIVES)

    def test_unknown_objective_raises_by_name(self):
        from uplift_bench.experiments.tuning import get_objective

        with pytest.raises(ValueError, match="unknown tuning objective"):
            get_objective("not_a_metric")

    def test_objectives_score_identical_predictions_differently(self):
        """Qini and AUUC are different functions of the same predictions."""
        from uplift_bench.experiments.tuning import get_objective

        rng = np.random.default_rng(0)
        n = 400
        pred = rng.normal(size=n)
        t = rng.integers(0, 2, n)
        y = rng.normal(size=n)
        q = get_objective("qini")(pred, t, y)
        a = get_objective("auuc")(pred, t, y)
        assert np.isfinite(q) and np.isfinite(a)
        assert q != a

    def test_objective_is_threaded_and_changes_selection(self):
        """Tuning under AUUC must be reachable and need not pick the Qini winner."""
        from uplift_bench.experiments.tuning import tune_hyperparameters

        ds = load_synthetic(n=600, seed=0)
        common = dict(
            model_name="t_learner",
            base_learner="lightgbm",
            X=ds.X,
            treatment=ds.treatment,
            outcome=ds.outcome,
            propensity=None,
            n_samples=4,
            inner_folds=2,
            seed=0,
        )
        hp_q, score_q = tune_hyperparameters(objective="qini", **common)
        hp_a, score_a = tune_hyperparameters(objective="auuc", **common)
        assert isinstance(hp_q, dict) and isinstance(hp_a, dict)
        # Both arms must produce a usable score on the same data.
        assert np.isfinite(score_q) and np.isfinite(score_a)
        # The objectives are on different scales, so the scores must not coincide.
        assert score_q != score_a

    def test_invalid_objective_rejected_before_fitting(self):
        """A typo in the config must fail fast, not after hours of fitting."""
        from uplift_bench.experiments.tuning import tune_hyperparameters

        ds = load_synthetic(n=200, seed=0)
        with pytest.raises(ValueError, match="unknown tuning objective"):
            tune_hyperparameters(
                model_name="t_learner",
                base_learner="lightgbm",
                X=ds.X,
                treatment=ds.treatment,
                outcome=ds.outcome,
                propensity=None,
                n_samples=2,
                inner_folds=2,
                seed=0,
                objective="qni",  # typo
            )
