"""Synthetic dataset utilities."""

import numpy as np

def generate_synthetic_linear(n_points: int = 100):
    """Generate a simple linear synthetic time series."""
    X = np.linspace(0, 10, n_points).reshape(-1, 1)
    y = 0.5 * X + 0.2 * np.sin(X * 2.0) + 0.05 * np.random.randn(n_points, 1)
    return X, y


def generate_synthetic_nonlinear(n_points: int = 100):
    """Generate a nonlinear synthetic time series."""
    X = np.linspace(0, 10, n_points).reshape(-1, 1)
    y = np.sin(X) + 0.5 * np.cos(2.0 * X) + 0.1 * np.random.randn(n_points, 1)
    return X, y
