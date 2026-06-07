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
        self.initialized = False

    def fit(self, X, y, initial_state=None):
        self.state = np.asarray(initial_state) if initial_state is not None else np.zeros((self.transition.shape[0], 1))
        self.is_fitted = True
        self.initialized = True
        return self

    def predict(self, X):
        X = np.asarray(X)
        return X @ self.observation.T

    def simulate(self, n_steps: int):
        if not self.initialized:
            raise RuntimeError("Model must be initialized with fit() before simulation.")
        state = self.state.copy()
        trajectory = []
        for _ in range(n_steps):
            state = self.transition @ state
            observation = self.observation @ state
            trajectory.append(observation.flatten())
        return np.vstack(trajectory)
