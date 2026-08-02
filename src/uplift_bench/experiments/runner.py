"""Hydra-configured, outer-test-isolated experiment runner (WP4).

No test-fold information enters preprocessing, propensity estimation, tuning, or
fitting for any reported metric. One disclosed inner-nesting imperfection: the
propensity model is fit once per outer-training fold and its predictions sliced for
the inner tuning folds (covariates/treatments only, never outcomes; affects which
hyperparameters are selected, not any reported test-fold number).

For each (dataset, model, seed, fold) it:
  1. splits via repeated stratified K-fold (outer-test-isolated),
  2. tunes hyperparameters by random search vs. validation Qini on TRAIN ONLY,
  3. applies the per-dataset propensity policy (known for RCTs, estimated on
     train for observational),
  4. refits the best config on the training fold and predicts the test fold,
  5. computes all applicable metrics on the held-out test fold.

Tidy per-fold results are written to ``results/<dataset>__<model>.parquet`` and a
CI summary to ``results/<dataset>__<model>__summary.parquet``, both stamped with
the full config and git hash.

Run:
    python -m uplift_bench.experiments.runner                # full (config.yaml)
    python -m uplift_bench.experiments.runner --config-name=smoke
    python -m uplift_bench.experiments.runner -m dataset=ihdp,jobs model=t_learner,x_learner
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from uplift_bench.data.registry import load_dataset
from uplift_bench.experiments.evaluation import (
    BINARY_OUTCOME_MODELS,
    evaluate_fold,
    outcome_is_binary,
)
from uplift_bench.experiments.git_utils import get_git_hash, is_dirty
from uplift_bench.experiments.propensity import resolve_propensity
from uplift_bench.experiments.splitting import repeated_stratified_kfold
from uplift_bench.experiments.tuning import tune_hyperparameters
from uplift_bench.models.registry import get_estimator
from uplift_bench.seed import set_global_seed

log = logging.getLogger(__name__)

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_CONFIG_DIR = os.path.join(_REPO_ROOT, "configs")

METRIC_NAMES = [
    "qini",
    "auuc",
    "uplift_at_k",
    "policy_value_at_k",
    "calibration_ece",
    "pehe",
    "epsilon_ate",
    "jobs_policy_risk",
]


# ---------------------------------------------------------------------------
# Core (Hydra-independent so it is unit-testable)
# ---------------------------------------------------------------------------


def run_experiment(cfg: dict) -> pd.DataFrame:
    """Execute the full repeated-CV loop for one (dataset, model) pair.

    ``cfg`` is a plain dict (OmegaConf is converted before this is called).
    Returns the tidy per-fold results DataFrame.
    """
    set_global_seed(cfg["seed"])

    dataset_name = cfg["dataset"]["name"]
    # loader_name lets us disambiguate display name (e.g. "ihdp_s3") from registry
    # key (e.g. "ihdp") when running multi-split datasets.
    loader_name = cfg["dataset"].get("loader_name") or dataset_name
    loader_kwargs = dict(cfg["dataset"].get("loader_kwargs", {}) or {})
    model_name = cfg["model"]["name"]
    base_learner = cfg["model"].get("base_learner", "lightgbm")
    smoke = cfg.get("smoke", False)
    budget_k = cfg.get("metrics", {}).get("budget_k", 0.3)

    tuning_cfg = cfg.get("tuning", {}) or {}
    tune_enabled = tuning_cfg.get("enabled", True)
    tune_objective = tuning_cfg.get("objective", "qini")
    n_samples = tuning_cfg.get("n_samples", 20)
    inner_folds = tuning_cfg.get("inner_folds", 3)

    # Optional per-unit prediction persistence (R4 reviewer analyses: metric-variant
    # comparison, RATE, budget grids). Written once per (dataset, model) at the end.
    save_predictions = bool(cfg.get("save_predictions", False))
    pred_frames: list[pd.DataFrame] = []

    log.info(
        "=== Running dataset=%s (loader=%s) model=%s base=%s smoke=%s ===",
        dataset_name,
        loader_name,
        model_name,
        base_learner,
        smoke,
    )

    ds = load_dataset(loader_name, **loader_kwargs)

    # Optional subsampling: smoke mode uses 2K; bench_n lets the orchestrator
    # set an explicit cap for large marketing datasets (e.g. 50_000).
    bench_n = cfg.get("bench_n", None)
    if smoke and ds.meta.n > 3000:
        ds = ds.subsample(2000, seed=cfg["seed"])
        log.info("Smoke mode: subsampled %s to n=%d", dataset_name, ds.meta.n)
    elif bench_n and ds.meta.n > bench_n:
        ds = ds.subsample(bench_n, seed=cfg["seed"])
        log.info("bench_n cap: subsampled %s to n=%d", dataset_name, ds.meta.n)

    git_hash = get_git_hash()
    config_json = json.dumps(cfg, default=str, sort_keys=True)
    timestamp = datetime.now(timezone.utc).isoformat()

    # Skip incompatible (model, dataset) combinations up front.
    binary_ok = outcome_is_binary(ds.outcome)
    if model_name in BINARY_OUTCOME_MODELS and not binary_ok:
        log.warning(
            "Skipping %s on %s: model requires a binary outcome.",
            model_name,
            dataset_name,
        )
        return _skipped_frame(
            dataset_name,
            model_name,
            base_learner,
            git_hash,
            config_json,
            timestamp,
            reason="requires_binary_outcome",
        )

    rows: list[dict] = []
    for seed_idx, fold_idx, train_idx, test_idx in repeated_stratified_kfold(
        ds.treatment,
        ds.outcome,
        n_folds=cfg["n_folds"],
        n_seeds=cfg["n_seeds"],
        base_seed=cfg["seed"],
    ):
        fold_seed = cfg["seed"] + seed_idx
        t0 = time.time()

        X_train, X_test = ds.X.iloc[train_idx].copy(), ds.X.iloc[test_idx].copy()
        t_train, t_test = ds.treatment.iloc[train_idx], ds.treatment.iloc[test_idx]
        y_train, y_test = ds.outcome.iloc[train_idx], ds.outcome.iloc[test_idx]
        ite_test = ds.ite.iloc[test_idx] if ds.ite is not None else None

        # Median-impute NaN features (fitted on train only — no leakage).
        if X_train.isnull().any().any():
            from sklearn.impute import SimpleImputer

            imp = SimpleImputer(strategy="median")
            X_train[:] = imp.fit_transform(X_train)
            X_test[:] = imp.transform(X_test)

        # --- propensity policy (train fit only for observational) ---
        train_prop, test_prop, prop_policy = resolve_propensity(
            ds, train_idx, test_idx, base_learner=base_learner, seed=fold_seed
        )

        # --- nested tuning on TRAIN ONLY ---
        best_hp, best_val_qini = tune_hyperparameters(
            model_name,
            base_learner,
            X_train,
            t_train,
            y_train,
            train_prop,
            n_samples=n_samples,
            inner_folds=inner_folds,
            seed=fold_seed,
            enabled=tune_enabled,
            objective=tune_objective,
        )

        status, err_msg = "ok", ""
        metrics: dict[str, float] = {}
        try:
            est = get_estimator(model_name, base_learner=base_learner, seed=fold_seed, **best_hp)
            est.fit(X_train, t_train, y_train, pd.Series(train_prop, index=X_train.index))
            pred = est.predict_uplift(X_test)
            metrics = evaluate_fold(
                pred,
                t_test.to_numpy(),
                y_test.to_numpy(),
                test_prop,
                ite_test,
                budget_k=budget_k,
            )
        except Exception as exc:  # record failure, keep the run going
            status, err_msg = "error", str(exc)
            log.exception(
                "Fold failed: dataset=%s model=%s seed=%d fold=%d",
                dataset_name,
                model_name,
                seed_idx,
                fold_idx,
            )

        # Persist per-unit predictions OUTSIDE the metric try/except: a save
        # problem must never change a fold's status, and failed folds have no
        # predictions to save.
        if save_predictions and status == "ok":
            try:
                _nan = np.full(len(test_idx), np.nan)
                # ite_test is a DataFrame for datasets with auxiliary truth columns
                # (ite/mu0/mu1 on simulated data; the randomized-subset flag
                # "experimental" on Jobs), a Series otherwise.
                if isinstance(ite_test, pd.DataFrame):
                    ite_col = (
                        np.ravel(ite_test["ite"].to_numpy(dtype=float))
                        if "ite" in ite_test.columns
                        else _nan
                    )
                    exp_col = (
                        np.ravel(ite_test["experimental"].to_numpy(dtype=float))
                        if "experimental" in ite_test.columns
                        else _nan
                    )
                elif ite_test is not None:
                    ite_col, exp_col = np.ravel(ite_test.to_numpy(dtype=float)), _nan
                else:
                    ite_col, exp_col = _nan, _nan
                pred_frames.append(
                    pd.DataFrame(
                        {
                            "dataset": dataset_name,
                            "model": model_name,
                            "base_learner": base_learner,
                            "seed_idx": seed_idx,
                            "fold_idx": fold_idx,
                            "unit_id": np.ravel(X_test.index.to_numpy()),
                            "pred": np.ravel(np.asarray(pred, dtype=float)),
                            "y": np.ravel(y_test.to_numpy(dtype=float)),
                            "t": np.ravel(t_test.to_numpy(dtype=float)),
                            "propensity": np.ravel(np.asarray(test_prop, dtype=float)),
                            "ite": ite_col,
                            "experimental": exp_col,
                        }
                    )
                )
            except Exception:
                log.exception(
                    "Prediction save failed (fold result unaffected): "
                    "dataset=%s model=%s seed=%d fold=%d",
                    dataset_name,
                    model_name,
                    seed_idx,
                    fold_idx,
                )

        row = {
            "dataset": dataset_name,
            "model": model_name,
            "base_learner": base_learner,
            "seed_idx": seed_idx,
            "fold_idx": fold_idx,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "treatment_fraction_train": float(t_train.mean()),
            "outcome_base_rate_train": float(y_train.mean()),
            "propensity_policy": prop_policy,
            "best_val_qini": best_val_qini,
            "best_hparams": json.dumps(best_hp, sort_keys=True),
            "status": status,
            "error_msg": err_msg,
            "runtime_sec": round(time.time() - t0, 3),
            "git_hash": git_hash,
            "git_dirty": is_dirty(),
            "config_json": config_json,
            "timestamp": timestamp,
        }
        for m in METRIC_NAMES:
            row[m] = metrics.get(m, float("nan"))
        rows.append(row)
        log.info(
            "seed=%d fold=%d %s qini=%.4f (%.1fs)",
            seed_idx,
            fold_idx,
            status,
            row["qini"],
            row["runtime_sec"],
        )

    if save_predictions and pred_frames:
        pred_dir = cfg.get("results_dir", "results/")
        if not os.path.isabs(pred_dir):
            pred_dir = os.path.join(_REPO_ROOT, pred_dir)
        pred_dir = os.path.join(pred_dir, "predictions")
        os.makedirs(pred_dir, exist_ok=True)
        pred_path = os.path.join(pred_dir, f"{dataset_name}__{model_name}__pred.parquet")
        pd.concat(pred_frames, ignore_index=True).to_parquet(pred_path, index=False)
        log.info("Wrote per-unit predictions -> %s", pred_path)

    return pd.DataFrame(rows)


def _skipped_frame(dataset, model, base, git_hash, config_json, timestamp, reason):
    row = {
        "dataset": dataset,
        "model": model,
        "base_learner": base,
        "seed_idx": -1,
        "fold_idx": -1,
        "n_train": 0,
        "n_test": 0,
        "treatment_fraction_train": float("nan"),
        "outcome_base_rate_train": float("nan"),
        "propensity_policy": "n/a",
        "best_val_qini": float("nan"),
        "best_hparams": "{}",
        "status": f"skipped_{reason}",
        "error_msg": "",
        "runtime_sec": 0.0,
        "git_hash": git_hash,
        "git_dirty": is_dirty(),
        "config_json": config_json,
        "timestamp": timestamp,
    }
    for m in METRIC_NAMES:
        row[m] = float("nan")
    return pd.DataFrame([row])


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-fold metrics into mean + 95% CI across folds/seeds."""
    group_cols = ["dataset", "model", "base_learner"]
    metric_cols = [m for m in METRIC_NAMES if m in raw.columns]
    ok = raw[raw["status"] == "ok"]
    records = []
    for keys, g in (ok if len(ok) else raw).groupby(group_cols):
        keys = keys if isinstance(keys, tuple) else (keys,)
        rec = dict(zip(group_cols, keys))
        for m in metric_cols:
            vals = g[m].dropna().to_numpy()
            if len(vals) == 0:
                rec[f"{m}_mean"] = np.nan
                rec[f"{m}_ci_low"] = np.nan
                rec[f"{m}_ci_high"] = np.nan
            else:
                mean = float(vals.mean())
                sem = float(vals.std(ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
                rec[f"{m}_mean"] = mean
                rec[f"{m}_ci_low"] = mean - 1.96 * sem
                rec[f"{m}_ci_high"] = mean + 1.96 * sem
            rec[f"{m}_n"] = int(len(vals))
        records.append(rec)
    return pd.DataFrame(records)


def write_results(raw: pd.DataFrame, results_dir: str) -> tuple[str, str]:
    """Write raw + summary parquet; return their paths."""
    os.makedirs(results_dir, exist_ok=True)
    dataset = raw["dataset"].iloc[0]
    model = raw["model"].iloc[0]
    raw_path = os.path.join(results_dir, f"{dataset}__{model}.parquet")
    sum_path = os.path.join(results_dir, f"{dataset}__{model}__summary.parquet")
    raw.to_parquet(raw_path, index=False)
    summarize(raw).to_parquet(sum_path, index=False)
    return raw_path, sum_path


# ---------------------------------------------------------------------------
# Hydra entry point
# ---------------------------------------------------------------------------


def _main() -> None:
    import hydra
    from omegaconf import OmegaConf

    @hydra.main(version_base=None, config_path=_CONFIG_DIR, config_name="config")
    def _run(cfg) -> None:
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        results_dir = cfg_dict.get("results_dir", "results/")
        if not os.path.isabs(results_dir):
            results_dir = os.path.join(_REPO_ROOT, results_dir)
        data_dir = cfg_dict.get("data_dir", "data/")
        if not os.path.isabs(data_dir):
            data_dir = os.path.join(_REPO_ROOT, data_dir)
        os.environ.setdefault("UPLIFT_DATA_DIR", data_dir)

        raw = run_experiment(cfg_dict)
        raw_path, sum_path = write_results(raw, results_dir)
        log.info("Wrote %d rows -> %s", len(raw), raw_path)
        log.info("Wrote summary -> %s", sum_path)

    _run()


if __name__ == "__main__":
    _main()
