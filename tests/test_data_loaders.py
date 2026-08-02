"""Unit tests for WP1 data loaders.

Tests that can run without network access use synthetic data or mock the
underlying download. Tests that require actual data are marked
``pytest.mark.network`` and skipped by default (run with -m network).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from uplift_bench.data.base import DatasetMeta, UpliftDataset

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_synthetic(
    n: int = 200,
    p: int = 5,
    treatment_frac: float = 0.5,
    outcome_rate: float = 0.3,
    seed: int = 0,
) -> UpliftDataset:
    """Minimal synthetic UpliftDataset for structural tests."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.standard_normal((n, p)), columns=[f"f{i}" for i in range(p)])
    treatment = pd.Series(rng.binomial(1, treatment_frac, n).astype(int), name="treatment")
    outcome = pd.Series(rng.binomial(1, outcome_rate, n).astype(float), name="outcome")
    propensity = pd.Series(np.full(n, treatment_frac), name="propensity")
    meta = DatasetMeta(
        name="synthetic",
        n=n,
        n_features=p,
        treatment_fraction=float(treatment.mean()),
        outcome_base_rate=float(outcome.mean()),
        has_ground_truth_effect=False,
    )
    return UpliftDataset(
        X=X, treatment=treatment, outcome=outcome, propensity=propensity, ite=None, meta=meta
    )


# ---------------------------------------------------------------------------
# Base / UpliftDataset structural tests
# ---------------------------------------------------------------------------


class TestUpliftDatasetStructure:
    def test_validate_passes_on_valid(self):
        ds = _make_synthetic()
        ds.validate()  # should not raise

    def test_validate_catches_length_mismatch(self):
        ds = _make_synthetic()
        ds.treatment = ds.treatment.iloc[:-1]
        with pytest.raises(AssertionError):
            ds.validate()

    def test_validate_catches_non_binary_treatment(self):
        ds = _make_synthetic()
        ds.treatment = ds.treatment.replace({1: 2})
        with pytest.raises(AssertionError):
            ds.validate()

    def test_validate_catches_leaky_column_name(self):
        ds = _make_synthetic()
        ds.X = ds.X.rename(columns={"f0": "treatment"})
        with pytest.raises(ValueError, match="reserved"):
            ds.validate()

    def test_propensity_bounds(self):
        ds = _make_synthetic()
        assert (ds.propensity > 0).all()
        assert (ds.propensity < 1).all()

    def test_subsample_reduces_size(self):
        ds = _make_synthetic(n=200)
        small = ds.subsample(50, seed=7)
        assert len(small.X) == 50
        assert len(small.treatment) == 50
        assert len(small.outcome) == 50
        assert small.meta.n == 50

    def test_subsample_noop_when_larger_than_data(self):
        ds = _make_synthetic(n=100)
        same = ds.subsample(500)
        assert len(same.X) == 100

    def test_meta_fields_present(self):
        ds = _make_synthetic()
        assert ds.meta.n == len(ds.X)
        assert 0.0 < ds.meta.treatment_fraction < 1.0
        assert 0.0 <= ds.meta.outcome_base_rate <= 1.0
        assert isinstance(ds.meta.has_ground_truth_effect, bool)

    def test_no_treatment_outcome_in_X(self):
        ds = _make_synthetic()
        reserved = {"treatment", "outcome", "y", "t", "converted", "visit"}
        assert not set(ds.X.columns) & reserved


# ---------------------------------------------------------------------------
# Treatment-fraction and shape assertions (used for every real loader below)
# ---------------------------------------------------------------------------


def _assert_dataset_invariants(ds: UpliftDataset, name: str) -> None:
    """Reusable invariant checker."""
    assert len(ds.X) == len(ds.treatment) == len(ds.outcome), f"{name}: length mismatch"
    assert set(ds.treatment.unique()).issubset({0, 1}), f"{name}: treatment not binary"
    assert ds.meta.n == len(ds.X), f"{name}: meta.n mismatch"
    assert ds.meta.n_features == ds.X.shape[1], f"{name}: meta.n_features mismatch"
    assert 0.0 < ds.meta.treatment_fraction < 1.0, f"{name}: treatment_fraction out of (0,1)"
    tf_computed = float(ds.treatment.mean())
    assert abs(ds.meta.treatment_fraction - tf_computed) < 1e-4, (
        f"{name}: meta.treatment_fraction {ds.meta.treatment_fraction:.4f} "
        f"≠ actual {tf_computed:.4f}"
    )
    # No treatment/outcome leakage in feature columns
    reserved = {"treatment", "outcome", "y", "t", "converted", "visit"}
    assert not (set(ds.X.columns) & reserved), f"{name}: leaky column(s) in X"
    if ds.propensity is not None:
        assert len(ds.propensity) == len(ds.X), f"{name}: propensity length mismatch"
        assert (ds.propensity > 0).all(), f"{name}: propensity ≤ 0"
        assert (ds.propensity < 1).all(), f"{name}: propensity ≥ 1"


# ---------------------------------------------------------------------------
# Registry smoke test (import only — no download)
# ---------------------------------------------------------------------------


def test_list_datasets():
    from uplift_bench.data.registry import list_datasets

    names = list_datasets()
    expected = {"criteo", "lenta", "x5", "megafon", "hillstrom", "ihdp", "jobs"}
    assert expected.issubset(set(names))


def test_load_dataset_unknown_raises():
    from uplift_bench.data.registry import load_dataset

    with pytest.raises(ValueError, match="Unknown dataset"):
        load_dataset("does_not_exist")


# ---------------------------------------------------------------------------
# Network-dependent tests (skipped in CI unless explicitly requested)
# ---------------------------------------------------------------------------

network = pytest.mark.skipif(
    not pytest.importorskip("urllib.request", reason="network"),
    reason="requires network access",
)


@pytest.mark.network
def test_ihdp_split0():
    from uplift_bench.data.ihdp import load_ihdp

    ds = load_ihdp(split_idx=0)
    _assert_dataset_invariants(ds, "ihdp_split0")
    assert ds.meta.has_ground_truth_effect is True
    assert ds.ite is not None
    assert "ite" in ds.ite.columns
    assert ds.propensity is None  # observational — estimated by runner
    assert ds.meta.n > 0
    assert ds.meta.n_features == 25


@pytest.mark.network
def test_ihdp_all_100_splits_load():
    from uplift_bench.data.ihdp import load_ihdp_all_splits

    splits = load_ihdp_all_splits()
    assert len(splits) == 100
    for i, ds in enumerate(splits[:5]):  # spot-check first 5
        _assert_dataset_invariants(ds, f"ihdp_split{i}")


@pytest.mark.network
def test_jobs_split0():
    from uplift_bench.data.jobs import load_jobs

    ds = load_jobs(split_idx=0)
    _assert_dataset_invariants(ds, "jobs_split0")
    assert ds.meta.has_ground_truth_effect is True
    assert ds.ite is not None
    assert "experimental" in ds.ite.columns
    assert ds.meta.n_features == 17


@pytest.mark.network
def test_hillstrom_visit():
    from uplift_bench.data.hillstrom import load_hillstrom

    ds = load_hillstrom(outcome="visit", treatment_arm="any")
    _assert_dataset_invariants(ds, "hillstrom_visit_any")
    assert ds.meta.has_ground_truth_effect is False
    # Hillstrom has ~64k rows; treatment fraction ≈ 2/3 (two treated arms vs one control)
    assert 0.3 < ds.meta.treatment_fraction < 0.8
    assert ds.meta.n > 50_000


@pytest.mark.network
def test_hillstrom_mens_arm():
    from uplift_bench.data.hillstrom import load_hillstrom

    ds = load_hillstrom(outcome="visit", treatment_arm="mens")
    _assert_dataset_invariants(ds, "hillstrom_visit_mens")
    # Men's arm drops Women's rows; roughly half the full dataset
    assert ds.meta.n < 50_000


@pytest.mark.network
def test_lenta():
    pytest.importorskip("sklift", reason="scikit-uplift not installed")
    from uplift_bench.data.sklift_datasets import load_lenta

    ds = load_lenta()
    _assert_dataset_invariants(ds, "lenta")
    assert ds.meta.n > 100_000


@pytest.mark.network
def test_x5():
    pytest.importorskip("sklift", reason="scikit-uplift not installed")
    from uplift_bench.data.sklift_datasets import load_x5

    ds = load_x5()
    _assert_dataset_invariants(ds, "x5")


@pytest.mark.network
def test_megafon():
    pytest.importorskip("sklift", reason="scikit-uplift not installed")
    from uplift_bench.data.sklift_datasets import load_megafon

    ds = load_megafon()
    _assert_dataset_invariants(ds, "megafon")


@pytest.mark.network
def test_criteo_1m_tier():
    from uplift_bench.data.criteo import load_criteo

    ds = load_criteo(subsample_tier="1M", outcome="visit")
    _assert_dataset_invariants(ds, "criteo_1M")
    assert ds.meta.n == 1_000_000
    assert ds.meta.n_features == 12
    assert ds.propensity is not None  # RCT — known propensity


# ---------------------------------------------------------------------------
# Outcome-type regression guard
# ---------------------------------------------------------------------------


class TestOutcomeType:
    """`meta.outcome_type` must be machine-readable and match the data.

    The paper's central axis is continuous-vs-binary outcomes, and the analysis panels
    are built from it. A hand-maintained split once placed the BINARY `synthetic`
    dataset in the "continuous" panel; these tests make that class of error a failure.
    """

    def test_validate_infers_binary(self):
        ds = _make_synthetic(outcome_rate=0.3)
        ds.meta.outcome_type = None
        ds.validate()
        assert ds.meta.outcome_type == "binary"

    def test_validate_infers_continuous(self):
        ds = _make_synthetic()
        ds.outcome = pd.Series(np.linspace(0.0, 100.0, len(ds.outcome)), name="outcome")
        ds.meta.outcome_type = None
        ds.validate()
        assert ds.meta.outcome_type == "continuous"

    def test_validate_rejects_wrong_declaration(self):
        ds = _make_synthetic(outcome_rate=0.3)  # binary outcome
        ds.meta.outcome_type = "continuous"  # ...declared wrongly
        with pytest.raises(ValueError, match="outcome_type"):
            ds.validate()

    def test_shipped_synthetic_config_is_binary(self):
        """The released `synthetic` benchmark instance is BINARY by configuration.

        Guards the specific regression: `configs/dataset/synthetic.yaml` sets
        binary_outcome: true, so `synthetic` belongs with the binary families in every
        panel, table and figure.
        """
        from uplift_bench.data.synthetic import load_synthetic

        ds = load_synthetic(n=300, binary_outcome=True)
        ds.validate()
        assert ds.meta.outcome_type == "binary"
        assert set(ds.outcome.unique()).issubset({0, 1})

    def test_synthetic_continuous_variant_is_continuous(self):
        from uplift_bench.data.synthetic import load_synthetic

        ds = load_synthetic(n=300, binary_outcome=False)
        ds.validate()
        assert ds.meta.outcome_type == "continuous"


class TestSubsampleContinuous:
    """Regression tests for two subsample() defects (external audit).

    (a) Stratifying on outcome.astype(int) breaks on continuous outcomes (singleton
        strata); continuous outcomes must stratify by treatment alone.
    (b) subsample() rebuilt DatasetMeta without outcome_type, silently dropping the
        machine-readable regime tag the registry backstop depends on.
    """

    def test_continuous_outcome_subsample_succeeds_and_keeps_outcome_type(self):
        from uplift_bench.data.registry import load_dataset

        ds = load_dataset("revenue_synthetic", n=2000, seed=0)
        assert ds.meta.outcome_type == "continuous"
        sub = ds.subsample(500, seed=1)
        assert len(sub.X) == 500
        assert sub.meta.outcome_type == "continuous"

    def test_binary_outcome_subsample_still_stratifies_and_keeps_outcome_type(self):
        from uplift_bench.data.registry import load_dataset

        ds = load_dataset("synthetic", n=2000, seed=0)
        sub = ds.subsample(500, seed=1)
        assert len(sub.X) == 500
        assert sub.meta.outcome_type == ds.meta.outcome_type == "binary"
        # treatment x outcome stratification should roughly preserve the base rate
        assert abs(sub.outcome.mean() - ds.outcome.mean()) < 0.05
