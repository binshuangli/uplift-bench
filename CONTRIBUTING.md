# Contributing to UpliftBench

Thank you for your interest in contributing! This document covers:

1. [Submitting results to the leaderboard](#leaderboard)
2. [Adding a new dataset](#new-dataset)
3. [Adding a new model](#new-model)
4. [Development setup](#dev-setup)
5. [Code style](#code-style)

---

## Submitting results to the leaderboard {#leaderboard}

See [RESULTS.md](RESULTS.md) for full instructions. The short version:

1. Fork the repo and run the benchmark on your model/dataset:
   ```bash
   python scripts/run_bench.py --model your_model --tier medium --workers 4
   python scripts/collect_results.py
   ```
2. Fill in the submission template in `RESULTS.md`.
3. Open a pull request with:
   - Your results parquet(s) in `results/submissions/<your_method>/`
   - Updated `RESULTS.md` with your row filled in
   - A `model_cards/<your_method>.md` model card

All submitted results are independently verified by re-running the benchmark before merging.

---

## Adding a new dataset {#new-dataset}

1. Create a loader in `src/uplift_bench/data/my_dataset.py` returning an `UpliftDataset`
   (see `src/uplift_bench/data/base.py`). Declare `outcome_type` — `validate()` checks it
   against the data and raises if they disagree:
   ```python
   from uplift_bench.data.base import DatasetMeta, UpliftDataset

   def load_my_dataset(data_dir: str = "data", **kwargs) -> UpliftDataset:
       ds = UpliftDataset(
           X=X, treatment=t, outcome=y,
           propensity=e,          # None if unknown (estimated per fold)
           ite=ite_df,            # None if no per-unit ground truth
           meta=DatasetMeta(
               name="my_dataset", n=len(X), n_features=X.shape[1],
               treatment_fraction=float(t.mean()), outcome_base_rate=float(y.mean()),
               has_ground_truth_effect=ite_df is not None,
               outcome_type="binary",      # or "continuous"
           ),
       )
       ds.validate()
       return ds
   ```
2. Register it in `src/uplift_bench/data/registry.py`. Loaders are referenced lazily by
   `"module:function"` string, so importing your module is not required at registry import
   time:
   ```python
   _LOADERS: dict[str, str] = {
       ...,
       "my_dataset": "uplift_bench.data.my_dataset:load_my_dataset",
   }
   ```
3. Add a Hydra config in `configs/dataset/my_dataset.yaml`:
   ```yaml
   name: my_dataset
   loader_kwargs: {}        # forwarded to the loader as **kwargs
   ```
4. Add a unit test in `tests/test_data_loaders.py` (assert `meta.outcome_type` explicitly).
5. Document the dataset in the `README.md` dataset table (regime, n, p, treatment fraction,
   reference objective).

*One-off evaluation of your own data needs none of the above — use
`UpliftDataset.from_frame` (see README, "Bring your own data").*

---

## Adding a new model {#new-model}

1. Create a wrapper in `src/uplift_bench/models/my_model.py` implementing `UpliftEstimator`
   (see `src/uplift_bench/models/base.py`):
   ```python
   from uplift_bench.models.base import UpliftEstimator
   class MyModel(UpliftEstimator):
       def fit(self, X, t, y, propensity=None): ...
       def predict_uplift(self, X) -> np.ndarray: ...   # note: predict_uplift, not predict
   ```
2. Register in `src/uplift_bench/models/registry.py`.
3. Add a Hydra config in `configs/model/my_model.yaml`.
4. Add a unit test in `tests/test_models.py`.
5. Add a model card in `model_cards/my_model.md`.

The benchmark enforces: no test data in fitting; no full-dataset statistics before split.
Non-compliant models will not be accepted to the leaderboard.

---

## Development setup {#dev-setup}

```bash
git clone https://github.com/binshuangli/uplift-bench
cd uplift-bench
pip install -e ".[dev]"
pre-commit install

# Verify everything works
make smoke   # fast end-to-end test (~5 min)
make test    # full pytest suite
```

Requires Python >= 3.11.

---

## Code style {#code-style}

- **Formatter**: `black` (line length 100)
- **Linter**: `ruff` (E, F, W, I rules)
- Run `make fmt` before committing; CI enforces `make lint`.
- No comments explaining *what* the code does — only *why* (non-obvious constraints, workarounds).
- No `n_jobs=-1` in model constructors — use `n_jobs=1` and let the orchestrator control parallelism.

---

## Reporting issues

Open a GitHub issue with:
- The failing command (with full output)
- Your OS, Python version, and package versions (`pip freeze`)
- The git hash you're running (`git rev-parse HEAD`)
