"""Periodic kernel implementation.

k(x, x') = variance * exp( -2 * sin^2(pi * |x-x'| / period) / lengthscale^2 )
"""

import numpy as np
from .base import Kernel


class PeriodicKernel(Kernel):
    """A stationary periodic kernel.

    Parameters
    - variance: overall scale of the kernel
    - lengthscale: smoothness parameter
    - period: period of the function
    """

    def __init__(self, variance: float = 1.0, lengthscale: float = 1.0, period: float = 1.0):
        super().__init__(variance=variance, lengthscale=lengthscale)
        self.period = float(period)

    def __call__(self, X, Y=None):
        X = np.atleast_2d(X)
        if Y is None:
            Y = X
        Y = np.atleast_2d(Y)
        # Ensure shape (n, d) and (m, d)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        # pairwise Euclidean distances over last axis -> shape (n, m)
        diff = X[:, None, :] - Y[None, :, :]
        dists = np.linalg.norm(diff, axis=2)
        # compute periodic distance
        arg = np.pi * dists / self.period
        sin2 = np.sin(arg) ** 2
        K = self.variance * np.exp(-2.0 * sin2 / (self.lengthscale ** 2))
        return K

    def diag(self, X):
        X = np.atleast_2d(X)
        return np.full(X.shape[0], self.variance)
