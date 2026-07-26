"""uplift-bench: outer-test-isolated uplift/CATE benchmarking.

No test-fold information enters any reported number. Inner nesting has one disclosed
imperfection (propensity fit per outer-training fold, sliced for inner folds; see the
paper's Limitations), which is why the protocol is named outer-test-isolated rather
than leakage-free.
"""

from uplift_bench.seed import set_global_seed

__all__ = ["set_global_seed"]
