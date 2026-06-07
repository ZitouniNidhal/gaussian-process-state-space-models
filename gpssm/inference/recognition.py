"""Recognition network helpers."""

import numpy as np

class RecognitionModel:
    """Simple linear recognition network."""

    def __init__(self, input_dim: int, latent_dim: int = 1):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.weights = np.ones((input_dim, latent_dim)) * 0.1
        self.bias = np.zeros(latent_dim)

    def encode(self, inputs):
        inputs = np.asarray(inputs)
        return inputs @ self.weights + self.bias

    def decode(self, latent):
        latent = np.asarray(latent)
        return latent @ self.weights.T + self.bias

    def fit(self, inputs, targets):
        inputs = np.asarray(inputs)
        targets = np.asarray(targets)
        self.weights, residuals, rank, s = np.linalg.lstsq(inputs, targets, rcond=None)
        self.bias = np.zeros(self.latent_dim)
        return self
