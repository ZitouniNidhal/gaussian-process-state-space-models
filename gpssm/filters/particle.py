"""Simple particle filter implementation with ESS and adaptive resampling."""

import numpy as np


class ParticleFilter:
    """A bootstrap particle filter with ESS diagnostic and adaptive resampling."""

    def __init__(self, n_particles: int, transition_fn, observation_fn, process_noise, observation_noise):
        self.n_particles = n_particles
        self.transition_fn = transition_fn
        self.observation_fn = observation_fn
        self.process_noise = process_noise
        self.observation_noise = observation_noise
        self.particles = None
        self.weights = None

    def initialize(self, initial_particles):
        self.particles = np.asarray(initial_particles, dtype=float)
        self.weights = np.ones(self.n_particles) / self.n_particles

    def predict(self):
        # Move each particle through the transition function and add process noise.
        noise = np.random.randn(*self.particles.shape) * self.process_noise
        self.particles = self.transition_fn(self.particles) + noise
        return self.particles

    def update(self, observation):
        observation = np.asarray(observation, dtype=float)
        # Weight particles by how likely the observation is under each particle.
        diff = self.observation_fn(self.particles) - observation
        likelihoods = np.exp(-0.5 * np.sum(diff ** 2, axis=-1) / self.observation_noise)
        
        self.weights *= likelihoods
        self.weights += 1e-15  # Avoid complete division by zero
        self.weights /= self.weights.sum()
        
        # Adaptive resampling: only resample when ESS falls below N_particles / 2
        if self.effective_sample_size() < self.n_particles / 2.0:
            self.resample()
            
        return self.particles, self.weights

    def resample(self):
        cumulative = np.cumsum(self.weights)
        cumulative[-1] = 1.0
        indexes = np.searchsorted(cumulative, np.random.rand(self.n_particles))
        self.particles = self.particles[indexes]
        self.weights.fill(1.0 / self.n_particles)

    def effective_sample_size(self) -> float:
        """Compute the Effective Sample Size (ESS) diagnostic."""
        return 1.0 / np.sum(self.weights ** 2)

    def estimate(self):
        """Compute the estimated state mean and covariance from weighted particles.

        Returns
        -------
        mean : np.ndarray
        covariance : np.ndarray
        """
        # Weighted mean
        mean = np.sum(self.particles * self.weights[:, None], axis=0)
        
        # Weighted covariance
        diff = self.particles - mean
        covariance = (diff.T * self.weights) @ diff
        return mean, covariance
