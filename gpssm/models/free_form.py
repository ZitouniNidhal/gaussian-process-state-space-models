"""Free-form GP state-space model implementation."""

import numpy as np
from .base import BaseModel
from ..kernels import RBFKernel

class FreeFormGPModel(BaseModel):
    """Lightweight free-form GP model for flexible dynamics."""

    def __init__(self, lengthscale: float = 1.0, variance: float = 1.0):
        super().__init__()
        self.X_train = None
        self.y_train = None
        self.kernel = RBFKernel(variance=variance, lengthscale=lengthscale)

    def fit(self, X, y):
        self.X_train = np.asarray(X)
        self.y_train = np.asarray(y)
        self.is_fitted = True
        return self

    def predict(self, X):
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predicting.")
        X = np.asarray(X)
        K = self.kernel(X, self.X_train)
        weights = K / np.maximum(K.sum(axis=1, keepdims=True), 1e-12)
        return weights @ self.y_train
