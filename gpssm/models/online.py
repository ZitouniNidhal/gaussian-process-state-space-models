"""Online GP model implementation."""

import numpy as np
from .base import BaseModel


class OnlineGPModel(BaseModel):
    """Online model that fits incrementally with recursive ridge regression/GP updates."""

    def __init__(self, lambda_reg: float = 1e-3, variance: float = 1.0):
        super().__init__()
        self.X_train = None
        self.y_train = None
        self.weights = None
        self.lambda_reg = float(lambda_reg)
        self.variance = float(variance)
        self._A_inv = None  # Store inverse of covariance matrix for Woodbury update
        self._b = None

    def fit(self, X, y):
        self.X_train = np.atleast_2d(np.asarray(X, dtype=float))
        self.y_train = np.atleast_2d(np.asarray(y, dtype=float))
        if self.y_train.ndim == 1:
            self.y_train = self.y_train.reshape(-1, 1)
            
        d = self.X_train.shape[1]
        A = self.X_train.T @ self.X_train + self.lambda_reg * np.eye(d)
        self._A_inv = np.linalg.inv(A)
        self._b = self.X_train.T @ self.y_train
        self.weights = self._A_inv @ self._b
        self.is_fitted = True
        return self

    def update(self, X_new, y_new):
        X_new = np.atleast_2d(np.asarray(X_new, dtype=float))
        y_new = np.atleast_2d(np.asarray(y_new, dtype=float))
        if y_new.ndim == 1:
            y_new = y_new.reshape(-1, 1)
            
        if self.X_train is None:
            return self.fit(X_new, y_new)
            
        self.X_train = np.vstack([self.X_train, X_new])
        self.y_train = np.vstack([self.y_train, y_new])
        
        # Woodbury rank-1 or batch update of inverse matrix
        # A_new = A + X_new^T @ X_new
        # A_new^-1 = A^-1 - A^-1 X_new^T (I + X_new A^-1 X_new^T)^-1 X_new A^-1
        temp = np.eye(len(X_new)) + X_new @ self._A_inv @ X_new.T
        self._A_inv -= self._A_inv @ X_new.T @ np.linalg.inv(temp) @ X_new @ self._A_inv
        
        self._b += X_new.T @ y_new
        self.weights = self._A_inv @ self._b
        return self

    def predict(self, X):
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predicting.")
        X = np.atleast_2d(np.asarray(X, dtype=float))
        return X @ self.weights

    def predict_with_uncertainty(self, X):
        """Predict mean and variance recursively/online."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predicting.")
        X = np.atleast_2d(np.asarray(X, dtype=float))
        mean = self.predict(X)
        
        # Variance calculation based on weight covariance matrix: Sigma = variance * A^-1
        variance = self.variance * np.sum(X @ self._A_inv * X, axis=1, keepdims=True)
        return mean, np.maximum(variance, 0.0)
