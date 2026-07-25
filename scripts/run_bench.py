"""WP5 benchmark orchestrator.

Runs all (dataset, model) combinations, logging runtimes and statuses.
Supports parallel execution via ProcessPoolExecutor.

Usage:
    python scripts/run_bench.py                        # full run
    python scripts/run_bench.py --tier small           # synthetic + semi-synthetic only
    python scripts/run_bench.py --tier medium          # + marketing RCT datasets
    python scripts/run_bench.py --tier criteo          # + Criteo 1M
    python scripts/run_bench.py --workers 4            # parallel jobs
    python scripts/run_bench.py --model t_learner      # single model
    python scripts/run_bench.py --dataset ihdp         # single dataset group
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).parent.parent.resolve()
PY = sys.executable

# ---------------------------------------------------------------------------
# Dataset / model grid
# ---------------------------------------------------------------------------

# Each entry: (dataset_group_name, config_name, loader_kwargs_override)
DATASET_GROUPS = {
    "synthetic": [("synthetic", "synthetic", {})],
    "ihdp": [
        # dataset.name is the output label; dataset.loader_name is the registry key
        (f"ihdp_s{i}", "ihdp", {
            "dataset.loader_kwargs.split_idx": str(i),
            "dataset.name": f"ihdp_s{i}",
            "dataset.loader_name": "ihdp",
        })
        for i in range(10)  # 10 representative IHDP splits (of 100)
    ],
    "jobs": [
        (f"jobs_s{i}", "jobs", {
            "dataset.loader_kwargs.split_idx": str(i),
            "dataset.name": f"jobs_s{i}",
            "dataset.loader_name": "jobs",
        })
        for i in range(10)  # all 10 Jobs splits
    ],
    # F1 generality extension: 20 additional IHDP realizations + revenue-uplift synthetic.
    "ihdp_ext": [
        (f"ihdp_s{i}", "ihdp", {
            "dataset.loader_kwargs.split_idx": str(i),
            "dataset.name": f"ihdp_s{i}",
            "dataset.loader_name": "ihdp",
        })
        for i in range(10, 30)
    ],
    "revenue_synthetic": [("revenue_synthetic", "revenue_synthetic", {})],
    # Issue-4 validation: remaining IHDP realizations (30..99) so all 100 can be
    # reported. Run with well-behaved meta-learners only (see --models) as a cheap check.
    "ihdp_val": [
        (f"ihdp_s{i}", "ihdp", {
            "dataset.loader_kwargs.split_idx": str(i),
            "dataset.name": f"ihdp_s{i}",
            "dataset.loader_name": "ihdp",
        })
        for i in range(30, 100)
    ],
    # ACIC 2016 (R4-W1): third independent continuous-outcome family; 9 stratified
    # heterogeneous-effect settings x 2 replicates (see scripts/acic2016_generate.R).
    "acic2016": [
        (f"acic_s{s}r{r}", "acic2016", {
            "dataset.loader_kwargs.setting": str(s),
            "dataset.loader_kwargs.replicate": str(r),
            "dataset.name": f"acic_s{s}r{r}",
            "dataset.loader_name": "acic2016",
        })
        for s in (1, 4, 5, 9, 21, 25, 27, 28, 31)
        for r in (1, 2)
    ],
    "hillstrom": [("hillstrom", "hillstrom", {})],
    "lenta": [("lenta", "lenta", {})],
    "x5": [("x5", "x5", {})],
    "megafon": [("megafon", "megafon", {})],
    "criteo_1m": [("criteo_1m", "criteo", {
        "dataset.loader_kwargs.subsample_tier": "1M",
        "dataset.name": "criteo_1m",
        "dataset.loader_name": "criteo",
    })],
    "criteo_5m": [("criteo_5m", "criteo", {
        "dataset.loader_kwargs.subsample_tier": "5M",
        "dataset.name": "criteo_5m",
        "dataset.loader_name": "criteo",
    })],
}

TIERS = {
    "small": ["synthetic", "ihdp", "jobs"],
    "medium": ["synthetic", "ihdp", "jobs", "hillstrom", "lenta", "x5", "megafon"],
    "criteo": ["criteo_1m"],
    "m1ext": ["ihdp_ext", "revenue_synthetic"],
    "ihdpval": ["ihdp_val"],
    "acic": ["acic2016"],
    "all": list(DATASET_GROUPS.keys()),
}

# For large datasets: subsample to this n to keep runtimes manageable on a laptop.
# Flagged in findings as "subsampled benchmark". Set to None to disable.
BENCH_N_OVERRIDE: dict[str, int | None] = {
    "lenta": 10_000,    # 687K full; 10K keeps fold time ~40s on Apple Silicon
    "x5": 10_000,       # 200K full
    "megafon": 10_000,  # 600K full
    "criteo_1m": 10_000,
    "criteo_5m": 10_000,
}

ALL_MODELS = [
    "s_learner",
    "t_learner",
    "x_learner",
    "r_learner",
    "dr_learner",
    "class_transformation",
    "two_model",
    "solo_model",
    "causal_forest",
    "uplift_rf_kl",
    "uplift_rf_ed",
    "uplift_rf_chi",
]

# Timeout per (dataset_group, model) job in seconds
TIMEOUTS = {
    "synthetic": 900,    # r_learner/dr_learner can take 300-600s (fit 3+ models)
    "ihdp": 600,         # r_learner/dr_learner on n=672 need more headroom
    "ihdp_ext": 600,
    "ihdp_val": 600,
    "revenue_synthetic": 1200,  # n=4000, r/dr fit 3+ nuisance models
    "jobs": 1800,        # ~8 min per model (n=2570)
    "acic2016": 1800,    # n=4802, wide one-hot X; nuisance-heavy learners need headroom
    "hillstrom": 3600,   # ~10 min per model (n=64K, bench_n not applied)
    "lenta": 3600,       # ~20 min per model (bench_n=10K)
    "x5": 3600,
    "megafon": 3600,
    "criteo_1m": 7200,
    "criteo_5m": 14400,
}
DEFAULT_TIMEOUT = 3600


# ---------------------------------------------------------------------------
# Hydra overrides for each (dataset_group, model) pair
# ---------------------------------------------------------------------------


def _build_overrides(
    model_name: str,
    dataset_config: str,
    dataset_group: str,
    extra_overrides: dict,
    n_folds: int,
    n_seeds: int,
    n_samples: int,
    inner_folds: int,
) -> list[str]:
    overrides = [
        f"dataset={dataset_config}",
        f"model={model_name}",
        f"n_folds={n_folds}",
        f"n_seeds={n_seeds}",
        f"tuning.n_samples={n_samples}",
        f"tuning.inner_folds={inner_folds}",
    ]
    bench_n = BENCH_N_OVERRIDE.get(dataset_group)
    if bench_n is not None:
        overrides.append(f"bench_n={bench_n}")
    for k, v in extra_overrides.items():
        overrides.append(f"{k}={v}")
    return overrides


# ---------------------------------------------------------------------------
# Runner for a single (dataset_group_entry, model) job
# ---------------------------------------------------------------------------


def run_one(
    job_id: str,
    dataset_group: str,
    dataset_config: str,
    model_name: str,
    extra_overrides: dict,
    n_folds: int,
    n_seeds: int,
    n_samples: int,
    inner_folds: int,
    results_dir: Path,
    data_dir: Path,
) -> dict:
    overrides = _build_overrides(
        model_name, dataset_config, dataset_group, extra_overrides, n_folds, n_seeds, n_samples, inner_folds
    )
    # Encode split_idx in the output filename by overriding dataset name via loader_kwargs
    # The runner names the parquet by cfg["dataset"]["name"], so dataset_group is key
    # We pass results_dir and data_dir as overrides too
    overrides += [
        f"results_dir={results_dir}",
        f"data_dir={data_dir}",
    ]
    # Skip if results parquet already exists (allows resuming an interrupted run).
    parquet_path = results_dir / f"{job_id}.parquet"
    if parquet_path.exists():
        return {
            "job_id": job_id, "dataset_group": dataset_group,
            "dataset_config": dataset_config, "model": model_name,
            "status": "skipped_already_done", "returncode": 0,
            "elapsed_sec": 0.0, "stdout_tail": "", "stderr_tail": "",
        }

    cmd = [PY, "-m", "uplift_bench.experiments.runner", "--config-name=bench"] + overrides
    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=TIMEOUTS.get(dataset_group, DEFAULT_TIMEOUT),
        )
        elapsed = time.time() - t0
        status = "ok" if proc.returncode == 0 else "error"
        return {
            "job_id": job_id,
            "dataset_group": dataset_group,
            "dataset_config": dataset_config,
            "model": model_name,
            "status": status,
            "returncode": proc.returncode,
            "elapsed_sec": round(elapsed, 1),
            "stdout_tail": proc.stdout[-2000:] if proc.stdout else "",
            "stderr_tail": proc.stderr[-2000:] if proc.stderr else "",
        }
    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        return {
            "job_id": job_id,
            "dataset_group": dataset_group,
            "dataset_config": dataset_config,
            "model": model_name,
            "status": "timeout",
            "returncode": -1,
            "elapsed_sec": round(elapsed, 1),
            "stdout_tail": "",
            "stderr_tail": f"TimeoutExpired after {elapsed:.0f}s",
        }
    except Exception as exc:
        elapsed = time.time() - t0
        return {
            "job_id": job_id,
            "dataset_group": dataset_group,
            "dataset_config": dataset_config,
            "model": model_name,
            "status": "launch_error",
            "returncode": -2,
            "elapsed_sec": round(elapsed, 1),
            "stdout_tail": "",
            "stderr_tail": str(exc),
        }


# ---------------------------------------------------------------------------
# Hardware probe
# ---------------------------------------------------------------------------


def hardware_info() -> dict:
    import os as _os
    uname = platform.uname()
    try:
        import psutil
        ram_gb = psutil.virtual_memory().total / 1e9
    except ImportError:
        ram_gb = None
    return {
        "platform": platform.platform(),
        "machine": uname.machine,
        "processor": uname.processor,
        "python_version": platform.python_version(),
        "cpu_count": _os.cpu_count(),
        "ram_gb": ram_gb,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--tier",
        default="small",
        choices=list(TIERS.keys()),
        help="Which dataset tier to run (default: small)",
    )
    ap.add_argument("--model", default=None, help="Run only this model")
    ap.add_argument("--models", default=None,
                    help="Comma-separated subset of models to run (e.g. "
                         "s_learner,t_learner,x_learner,causal_forest)")
    ap.add_argument("--dataset", default=None, help="Run only this dataset group")
    ap.add_argument("--base-learner", default=None,
                    help="Override base learner for all models (e.g. xgboost). "
                         "Use a separate --results-dir to avoid clobbering the default run.")
    ap.add_argument("--workers", type=int, default=4, help="Parallel jobs")
    ap.add_argument("--n-folds", type=int, default=3,
                    help="Outer CV folds (3 for bench speed, 5 for full spec)")
    ap.add_argument("--n-seeds", type=int, default=3,
                    help="Repeated CV seeds (3 for bench speed, 5 for full spec)")
    ap.add_argument("--n-samples", type=int, default=10,
                    help="HP random search budget (10 for bench speed, 20 for full spec)")
    ap.add_argument("--inner-folds", type=int, default=2,
                    help="Inner tuning folds (2 for bench speed, 3 for full spec)")
    ap.add_argument(
        "--results-dir",
        default=str(REPO_ROOT / "results"),
        help="Where to write parquet files",
    )
    ap.add_argument(
        "--data-dir",
        default=str(REPO_ROOT / "data"),
        help="Where datasets are cached",
    )
    ap.add_argument(
        "--save-predictions", action="store_true",
        help="Persist per-unit test-fold predictions to <results-dir>/predictions/ "
             "(metric-variant / RATE / budget-grid analyses). Use a dedicated "
             "--results-dir so the canonical results/ stay untouched.",
    )
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    data_dir = Path(args.data_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    hw = hardware_info()
    log.info("Hardware: %s", hw)
    hw_path = results_dir / "hardware.json"
    hw_path.write_text(json.dumps(hw, indent=2))

    # Build job list
    dataset_groups_to_run = (
        [args.dataset] if args.dataset else TIERS[args.tier]
    )
    if args.model:
        models_to_run = [args.model]
    elif args.models:
        models_to_run = [m.strip() for m in args.models.split(",") if m.strip()]
    else:
        models_to_run = ALL_MODELS

    jobs = []
    for dg in dataset_groups_to_run:
        if dg not in DATASET_GROUPS:
            log.warning("Unknown dataset group %s, skipping", dg)
            continue
        for (label, ds_cfg, extra) in DATASET_GROUPS[dg]:
            # Optionally override the base learner (e.g. xgboost robustness run).
            if args.base_learner:
                extra = {**extra, "model.base_learner": args.base_learner}
            if args.save_predictions:
                # '+' because save_predictions is not in bench.yaml defaults.
                extra = {**extra, "+save_predictions": "true"}
            for model in models_to_run:
                job_id = f"{label}__{model}"
                jobs.append(
                    (job_id, dg, ds_cfg, model, extra)
                )

    log.info(
        "Launching %d jobs (tier=%s, workers=%d, folds=%d×%d, HP=%d×%d)",
        len(jobs),
        args.tier,
        args.workers,
        args.n_folds,
        args.n_seeds,
        args.n_samples,
        args.inner_folds,
    )

    run_log_path = results_dir / "run_log.jsonl"
    completed = 0
    failed = 0
    t_start = time.time()

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                run_one,
                job_id,
                dg,
                ds_cfg,
                model,
                extra,
                args.n_folds,
                args.n_seeds,
                args.n_samples,
                args.inner_folds,
                results_dir,
                data_dir,
            ): job_id
            for (job_id, dg, ds_cfg, model, extra) in jobs
        }

        with open(run_log_path, "a") as flog:
            for future in as_completed(futures):
                result = future.result()
                flog.write(json.dumps(result) + "\n")
                flog.flush()

                if result["status"] == "ok":
                    completed += 1
                    log.info(
                        "[%d/%d] OK  %-60s %6.0fs",
                        completed + failed,
                        len(jobs),
                        result["job_id"],
                        result["elapsed_sec"],
                    )
                else:
                    failed += 1
                    log.warning(
                        "[%d/%d] %-8s %-60s %6.0fs | %s",
                        completed + failed,
                        len(jobs),
                        result["status"].upper(),
                        result["job_id"],
                        result["elapsed_sec"],
                        result["stderr_tail"][-200:],
                    )

    total_sec = time.time() - t_start
    log.info(
        "Done: %d ok, %d failed/timeout, total wall %.0fs (%.1f min)",
        completed,
        failed,
        total_sec,
        total_sec / 60,
    )
    log.info("Run log: %s", run_log_path)


if __name__ == "__main__":
    main()
