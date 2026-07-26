"""Uplift / CATE model wrappers."""

from uplift_bench.models.base import UpliftEstimator
from uplift_bench.models.registry import get_estimator, list_models

__all__ = ["UpliftEstimator", "get_estimator", "list_models"]
