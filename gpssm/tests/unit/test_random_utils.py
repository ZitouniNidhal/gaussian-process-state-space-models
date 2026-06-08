import numpy as np
from gpssm.utils.random import set_seed, sample_random_kernel, generate_random_synthetic


def test_random_kernel_sampling():
    set_seed(42)
    k1 = sample_random_kernel()
    set_seed(42)
    k2 = sample_random_kernel()
    # sampling with same seed should give same type and similar repr
    assert type(k1) is type(k2)


def test_generate_random_synthetic_reproducible():
    X1, y1 = generate_random_synthetic(50, seed=123)
    X2, y2 = generate_random_synthetic(50, seed=123)
    assert np.array_equal(X1, X2)
    assert np.allclose(y1, y2)
