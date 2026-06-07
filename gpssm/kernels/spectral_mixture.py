"""Spectral mixture kernel."""

import numpy as np
from .base import Kernel

class SpectralMixtureKernel(Kernel):
    """Spectral mixture kernel with a small number of components."""

    def __init__(self, variance: float = 1.0, lengthscale: float = 1.0, n_components: int = 2):
        super().__init__(variance=variance, lengthscale=lengthscale)
        self.n_components = int(n_components)
        self.means = np.linspace(0.1, 1.0, self.n_components)
        self.variances = np.full(self.n_components, 0.5)

    def __call__(self, X, Y=None):
        X = self._ensure_2d(X)
        if Y is None:
            Y = X
        dists = np.linalg.norm(X[:, None, :] - Y[None, :, :], axis=-1)
        kernel = np.zeros((len(X), len(Y)))
        for mean, var in zip(self.means, self.variances):
            kernel += np.exp(-0.5 * (dists ** 2) / (self.lengthscale ** 2)) * np.cos(2 * np.pi * mean * dists)
        return self.variance * kernel
