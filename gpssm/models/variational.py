"""Variational GP model implementation."""

import numpy as np
from .base import BaseModel
from ..inference import select_inducing_points

class VariationalGPModel(BaseModel):
    """Sparse variational Gaussian process model."""

    def __init__(self, kernel, n_inducing: int = 10, noise_variance: float = 1e-2):
        super().__init__()
        self.kernel = kernel
        self.n_inducing = int(n_inducing)
        self.noise_variance = float(noise_variance)
        self.Z = None
        self.alpha = None
        self.K_uu_inv = None
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        self.X_train = X
        self.y_train = y
        # Pick a small set of inducing points from the training inputs.
        self.Z = select_inducing_points(X, self.n_inducing)

        # Build the inducing point covariances needed for sparse inference.
        K_uu = self.kernel(self.Z) + self.noise_variance * np.eye(len(self.Z))
        K_uf = self.kernel(self.Z, X)
        self.K_uu_inv = np.linalg.inv(K_uu)
        self.alpha = self.K_uu_inv @ K_uf @ y

        self.is_fitted = True
        return self

    def predict(self, X):
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predicting.")
        X = np.asarray(X)
        K_star = self.kernel(X, self.Z)
        return K_star @ self.alpha

    def predict_with_uncertainty(self, X):
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predicting.")
        X = np.asarray(X)
        # Return both the mean prediction and the predictive variance.
        K_star = self.kernel(X, self.Z)
        mean = K_star @ self.alpha
        variance = self.kernel.diag(X).reshape(-1, 1) - np.sum(K_star @ self.K_uu_inv * K_star, axis=1, keepdims=True)
        return mean, np.maximum(variance, 0.0)
