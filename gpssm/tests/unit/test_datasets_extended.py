"""Unit tests for extended dataset utilities."""

import numpy as np
import pytest
from gpssm.datasets import generate_lorenz, generate_pendulum, generate_gp_sample
from gpssm.kernels import RBFKernel


def test_lorenz_generator():
    X, y = generate_lorenz(n_points=50, dt=0.01)
    assert X.shape == (50, 1)
    assert y.shape == (50, 3)
    assert not np.any(np.isnan(y))


def test_pendulum_generator():
    X, y = generate_pendulum(n_points=40, dt=0.05)
    assert X.shape == (40, 1)
    assert y.shape == (40, 2)
    assert not np.any(np.isnan(y))


def test_gp_sample_generator():
    kernel = RBFKernel(variance=1.0, lengthscale=1.5)
    X, y = generate_gp_sample(kernel, n_points=30)
    assert X.shape == (30, 1)
    assert y.shape == (30, 1)
    assert not np.any(np.isnan(y))
