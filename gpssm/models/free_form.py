"""Free-form GP state-space model implementation."""

import numpy as np
from .base import BaseModel

class FreeFormGPModel(BaseModel):
    """Lightweight free-form GP model for flexible dynamics."""

    def __init__(self):
        super().__init__()
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = np.asarray(X)
        self.y_train = np.asarray(y)
        self.is_fitted = True
        return self

    def predict(self, X):
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predicting.")
        return np.zeros((len(X), 1))
