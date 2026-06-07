"""Base model abstractions."""

import pickle
from abc import ABC, abstractmethod

class BaseModel(ABC):
    """Abstract base class for GPSM models."""

    def __init__(self):
        self.is_fitted = False

    @abstractmethod
    def fit(self, X, y):
        raise NotImplementedError

    @abstractmethod
    def predict(self, X):
        raise NotImplementedError

    def score(self, X, y, metric):
        predictions = self.predict(X)
        return metric(y, predictions)

    def save(self, filepath):
        """Save the model to disk using pickle."""
        with open(filepath, "wb") as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, filepath):
        """Load a saved model from disk."""
        with open(filepath, "rb") as file:
            return pickle.load(file)
