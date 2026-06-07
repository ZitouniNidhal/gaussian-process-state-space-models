"""Online GP model implementation."""

import numpy as np
from .base import BaseModel

class OnlineGPModel(BaseModel):
    """Online model that fits incrementally."""

    def __init__(self):
        super().__init__()
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        self.X_train = X
        self.y_train = y
        self.is_fitted = True
        return self

    def update(self, X_new, y_new):
        if self.X_train is None:
            return self.fit(X_new, y_new)
        self.X_train = np.vstack([self.X_train, X_new])
        self.y_train = np.vstack([self.y_train, y_new])
        return self

    def predict(self, X):
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predicting.")
        return np.zeros((len(X), 1))
