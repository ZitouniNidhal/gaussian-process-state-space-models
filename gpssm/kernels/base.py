"""Base kernel interface."""

from __future__ import annotations
import numpy as np
from abc import ABC, abstractmethod

class Kernel(ABC):
    """Base class for kernel implementations."""

    def __init__(self, variance: float = 1.0, lengthscale: float = 1.0):
        self.variance = float(variance)
        self.lengthscale = float(lengthscale)

    @abstractmethod
    def __call__(self, X, Y=None):
        raise NotImplementedError

    def _ensure_2d(self, X):
        X = np.asarray(X)
        if X.ndim == 1:
            # Convert 1D arrays into a column vector for kernel math.
            X = X.reshape(-1, 1)
        return X

    def diag(self, X):
        X = self._ensure_2d(X)
        return np.full(X.shape[0], self.variance)
