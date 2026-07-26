import numpy as np

from uplift_bench.seed import set_global_seed


def test_seed_reproducibility():
    set_global_seed(42)
    a = np.random.rand(10)
    set_global_seed(42)
    b = np.random.rand(10)
    np.testing.assert_array_equal(a, b)


def test_seed_different_seeds_differ():
    set_global_seed(0)
    a = np.random.rand(10)
    set_global_seed(1)
    b = np.random.rand(10)
    assert not np.array_equal(a, b)
