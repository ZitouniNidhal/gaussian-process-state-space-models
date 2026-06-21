"""Free-form GP state-space model implementation."""

import numpy as np
from .base import BaseModel
from ..kernels import RBFKernel


class FreeFormGPModel(BaseModel):
    """Lightweight free-form GP model for flexible dynamics."""

    def __init__(self, lengthscale: float = 1.0, variance: float = 1.0, noise_variance: float = 1e-2):
        super().__init__()
        self.X_train = None
        self.y_train = None
        self.noise_variance = float(noise_variance)
        self.kernel = RBFKernel(variance=variance, lengthscale=lengthscale)

    def fit(self, X, y):
        self.X_train = np.asarray(X, dtype=float)
        self.y_train = np.asarray(y, dtype=float)
        self.is_fitted = True
        return self

    def predict(self, X):
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predicting.")
        X = np.asarray(X, dtype=float)
        K = self.kernel(X, self.X_train)
        weights = K / np.maximum(K.sum(axis=1, keepdims=True), 1e-12)
        return weights @ self.y_train

    def predict_with_uncertainty(self, X):
        """Estimate predictions along with descriptive heuristic uncertainty band."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predicting.")
        X = np.asarray(X, dtype=float)
        mean = self.predict(X)
        
        # Free-form GP simple uncertainty heuristic based on distance/density
        K = self.kernel(X, self.X_train)
        density = K.sum(axis=1, keepdims=True)
        # Higher density of training points -> lower variance
        variance = self.kernel.variance - (self.kernel.variance - self.noise_variance) * (density / (density + 1.0))
        return mean, np.maximum(variance, 0.0)
