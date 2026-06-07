"""Online GP model implementation."""

import numpy as np
from .base import BaseModel

class OnlineGPModel(BaseModel):
    """Online model that fits incrementally with ridge regression."""

    def __init__(self, lambda_reg: float = 1e-6):
        super().__init__()
        self.X_train = None
        self.y_train = None
        self.weights = None
        self.lambda_reg = float(lambda_reg)
        self._A = None
        self._b = None

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        self.X_train = X
        self.y_train = y
        self._A = X.T @ X + self.lambda_reg * np.eye(X.shape[1])
        self._b = X.T @ y
        self.weights = np.linalg.solve(self._A, self._b)
        self.is_fitted = True
        return self

    def update(self, X_new, y_new):
        X_new = np.asarray(X_new)
        y_new = np.asarray(y_new)
        if self.X_train is None:
            return self.fit(X_new, y_new)
        self.X_train = np.vstack([self.X_train, X_new])
        self.y_train = np.vstack([self.y_train, y_new])
        self._A += X_new.T @ X_new
        self._b += X_new.T @ y_new
        self.weights = np.linalg.solve(self._A, self._b)
        return self

    def predict(self, X):
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predicting.")
        X = np.asarray(X)
        return X @ self.weights
