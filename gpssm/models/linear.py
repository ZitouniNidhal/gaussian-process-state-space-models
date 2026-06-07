"""Linear state-space model implementation."""

import numpy as np
from .base import BaseModel

class LinearStateSpaceModel(BaseModel):
    """A linear Gaussian state-space model."""

    def __init__(self, transition, observation, process_noise, observation_noise):
        super().__init__()
        self.transition = np.asarray(transition)
        self.observation = np.asarray(observation)
        self.process_noise = np.asarray(process_noise)
        self.observation_noise = np.asarray(observation_noise)
        self.state = None

    def fit(self, X, y):
        self.state = np.zeros((self.transition.shape[0], 1))
        self.is_fitted = True
        return self

    def predict(self, X):
        X = np.asarray(X)
        return X @ self.observation.T
