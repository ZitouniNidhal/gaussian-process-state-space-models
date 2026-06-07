"""Radial basis function kernel."""

import numpy as np
from .base import Kernel

class RBFKernel(Kernel):
    """Radial basis function kernel."""

    def __call__(self, X, Y=None):
        X = self._ensure_2d(X)
        if Y is None:
            Y = X
        else:
            Y = self._ensure_2d(Y)
        sqdist = np.sum((X[:, None, :] - Y[None, :, :]) ** 2, axis=-1)
        return self.variance * np.exp(-0.5 * sqdist / self.lengthscale ** 2)
