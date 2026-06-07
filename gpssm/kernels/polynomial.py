"""Polynomial kernel implementation."""

import numpy as np
from .base import Kernel

class PolynomialKernel(Kernel):
    """Polynomial kernel."""

    def __init__(self, variance: float = 1.0, lengthscale: float = 1.0, degree: int = 2, offset: float = 1.0):
        super().__init__(variance=variance, lengthscale=lengthscale)
        self.degree = int(degree)
        self.offset = float(offset)

    def __call__(self, X, Y=None):
        X = self._ensure_2d(X)
        if Y is None:
            Y = X
        else:
            Y = self._ensure_2d(Y)
        return self.variance * (X @ Y.T / self.lengthscale + self.offset) ** self.degree
