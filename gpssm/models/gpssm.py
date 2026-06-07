"""Gaussian process state-space model implementation."""

import numpy as np
from .base import BaseModel

class GaussianProcessStateSpaceModel(BaseModel):
    """Simple Gaussian process state space model."""

    def __init__(self, kernel, noise_variance: float = 1e-2):
        super().__init__()
        self.kernel = kernel
        self.noise_variance = float(noise_variance)
        self.X_train = None
        self.y_train = None
        self.alpha = None
        self.K_inv = None

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        self.X_train = X
        self.y_train = y
        K = self.kernel(X) + self.noise_variance * np.eye(len(X))
        self.K_inv = np.linalg.inv(K)
        self.alpha = self.K_inv @ y
        self.is_fitted = True
        return self

    def predict(self, X):
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predicting.")
        X = np.asarray(X)
        K_star = self.kernel(X, self.X_train)
        return K_star @ self.alpha

    def predict_with_uncertainty(self, X):
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predicting.")
        X = np.asarray(X)
        K_star = self.kernel(X, self.X_train)
        K_ss = self.kernel.diag(X).reshape(-1, 1)
        variance = K_ss - np.sum(K_star @ self.K_inv * K_star, axis=1, keepdims=True)
        return self.predict(X), np.maximum(variance, 0.0)
