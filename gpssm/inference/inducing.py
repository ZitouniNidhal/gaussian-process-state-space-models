"""Inducing point selection helpers."""

import numpy as np

def select_inducing_points(X, m: int = 10):
    """Select inducing points by random subset sampling."""
    X = np.asarray(X)
    if m >= len(X):
        return X.copy()
    indices = np.random.choice(len(X), size=m, replace=False)
    return X[indices]
