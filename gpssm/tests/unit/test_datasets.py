import numpy as np

from gpssm.datasets import generate_synthetic_linear, generate_synthetic_nonlinear, load_real_world_dataset


def test_generate_synthetic_linear_shape():
    X, y = generate_synthetic_linear(50)
    assert X.shape == (50, 1)
    assert y.shape == (50, 1)
    assert np.all(np.isfinite(X))
    assert np.all(np.isfinite(y))


def test_generate_synthetic_nonlinear_shape():
    X, y = generate_synthetic_nonlinear(30)
    assert X.shape == (30, 1)
    assert y.shape == (30, 1)
    assert np.all(np.isfinite(X))
    assert np.all(np.isfinite(y))


def test_load_real_world_dataset_shape():
    X, y = load_real_world_dataset(20)
    assert X.shape == (20, 1)
    assert y.shape == (20, 1)
