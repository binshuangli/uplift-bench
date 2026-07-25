"""Dataset registry — single call to load any dataset by name."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from uplift_bench.data.base import UpliftDataset

# Lazy imports so heavy packages aren't loaded unless needed
_LOADERS: dict[str, str] = {
    "synthetic": "uplift_bench.data.synthetic:load_synthetic",
    "revenue_synthetic": "uplift_bench.data.revenue_synthetic:load_revenue_synthetic",
    "criteo": "uplift_bench.data.criteo:load_criteo",
    "lenta": "uplift_bench.data.sklift_datasets:load_lenta",
    "x5": "uplift_bench.data.sklift_datasets:load_x5",
    "megafon": "uplift_bench.data.sklift_datasets:load_megafon",
    "hillstrom": "uplift_bench.data.hillstrom:load_hillstrom",
    "ihdp": "uplift_bench.data.ihdp:load_ihdp",
    "jobs": "uplift_bench.data.jobs:load_jobs",
    "acic2016": "uplift_bench.data.acic2016:load_acic2016",
}


def list_datasets() -> list[str]:
    return sorted(_LOADERS.keys())


def load_dataset(name: str, data_dir: Optional[Path] = None, **kwargs: Any) -> UpliftDataset:
    """Load a dataset by name, passing extra kwargs to the underlying loader.

    Examples
    --------
    >>> ds = load_dataset("criteo", subsample_tier="1M")
    >>> ds = load_dataset("ihdp", split_idx=3)
    >>> ds = load_dataset("hillstrom", outcome="visit", treatment_arm="any")
    """
    if name not in _LOADERS:
        raise ValueError(f"Unknown dataset '{name}'. Available: {list_datasets()}")

    module_path, fn_name = _LOADERS[name].split(":")
    import importlib

    module = importlib.import_module(module_path)
    fn = getattr(module, fn_name)

    import inspect

    sig = inspect.signature(fn)
    ds = fn(data_dir=data_dir, **kwargs) if "data_dir" in sig.parameters else fn(**kwargs)

    # Guarantee that every dataset reaching a caller carries a machine-readable
    # outcome_type, whether or not its loader declared one or called validate(). The
    # continuous-vs-binary split drives the analysis panels, so it must come from the
    # data rather than from a hand-maintained list in each script.
    if ds.meta.outcome_type is None:
        ds.meta.outcome_type = (
            "binary" if set(ds.outcome.unique()).issubset({0, 1}) else "continuous"
        )
    return ds
