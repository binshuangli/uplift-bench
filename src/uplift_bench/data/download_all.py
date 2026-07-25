"""CLI entry point for ``make data`` — downloads and caches all datasets.

Usage:
    python -m uplift_bench.data.download_all [--skip-criteo] [--skip-large]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Download and cache all benchmark datasets.")
    parser.add_argument(
        "--skip-criteo",
        action="store_true",
        help="Skip Criteo download (~1 GB compressed)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Override default data/ directory",
    )
    args = parser.parse_args(argv)

    results: dict[str, str] = {}

    # --- scikit-uplift datasets (auto-cached by sklift) ---
    for name in ("lenta", "x5", "megafon"):
        try:
            from uplift_bench.data.registry import load_dataset

            ds = load_dataset(name, data_dir=args.data_dir)
            results[name] = f"OK  n={ds.meta.n:,}"
        except Exception as exc:
            results[name] = f"FAIL {exc}"

    # --- Hillstrom ---
    try:
        from uplift_bench.data.registry import load_dataset

        ds = load_dataset("hillstrom", data_dir=args.data_dir)
        results["hillstrom"] = f"OK  n={ds.meta.n:,}"
    except Exception as exc:
        results["hillstrom"] = f"FAIL {exc}"

    # --- IHDP (split 0 only; rest load from the same cached file) ---
    try:
        from uplift_bench.data.registry import load_dataset

        ds = load_dataset("ihdp", data_dir=args.data_dir, split_idx=0)
        results["ihdp"] = f"OK  n={ds.meta.n:,}"
    except Exception as exc:
        results["ihdp"] = f"FAIL {exc}"

    # --- Jobs (split 0 only) ---
    try:
        from uplift_bench.data.registry import load_dataset

        ds = load_dataset("jobs", data_dir=args.data_dir, split_idx=0)
        results["jobs"] = f"OK  n={ds.meta.n:,}"
    except Exception as exc:
        results["jobs"] = f"FAIL {exc}"

    # --- Criteo (optional, large) ---
    if args.skip_criteo:
        results["criteo"] = "SKIPPED (--skip-criteo)"
    else:
        try:
            from uplift_bench.data.registry import load_dataset

            ds = load_dataset("criteo", data_dir=args.data_dir, subsample_tier="1M")
            results["criteo"] = f"OK  n={ds.meta.n:,}"
        except Exception as exc:
            results["criteo"] = f"FAIL {exc}"

    print("\n=== Dataset download summary ===")
    any_fail = False
    for name, status in results.items():
        icon = "✓" if status.startswith("OK") else ("~" if status.startswith("SKIP") else "✗")
        print(f"  {icon}  {name:<12} {status}")
        if status.startswith("FAIL"):
            any_fail = True

    sys.exit(1 if any_fail else 0)


if __name__ == "__main__":
    main()
