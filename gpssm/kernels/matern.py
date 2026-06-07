"""Matern kernel implementation."""

import numpy as np
from scipy.spatial.distance import cdist
from .base import Kernel

class MaternKernel(Kernel):
    """Matern kernel with smoothness parameter nu."""

    def __init__(self, variance: float = 1.0, lengthscale: float = 1.0, nu: float = 1.5):
        super().__init__(variance=variance, lengthscale=lengthscale)
        self.nu = float(nu)

    def __call__(self, X, Y=None):
        X = self._ensure_2d(X)
        if Y is None:
            Y = X
        else:
            Y = self._ensure_2d(Y)
        dists = cdist(X / self.lengthscale, Y / self.lengthscale, metric="euclidean")
        if self.nu == 0.5:
            return self.variance * np.exp(-dists)
        elif self.nu == 1.5:
            return self.variance * (1.0 + np.sqrt(3.0) * dists) * np.exp(-np.sqrt(3.0) * dists)
        elif self.nu == 2.5:
            return self.variance * (1.0 + np.sqrt(5.0) * dists + 5.0 / 3.0 * dists**2) * np.exp(-np.sqrt(5.0) * dists)
        else:
            raise ValueError("Unsupported nu value for this lightweight Matern kernel")
