"""Base model abstractions."""

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
