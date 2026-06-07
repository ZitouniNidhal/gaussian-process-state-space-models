"""Simple particle filter implementation."""

import numpy as np

class ParticleFilter:
    """A lightweight bootstrap particle filter."""

    def __init__(self, n_particles: int, transition_fn, observation_fn, process_noise, observation_noise):
        self.n_particles = n_particles
        self.transition_fn = transition_fn
        self.observation_fn = observation_fn
        self.process_noise = process_noise
        self.observation_noise = observation_noise
        self.particles = None
        self.weights = None

    def initialize(self, initial_particles):
        self.particles = np.asarray(initial_particles)
        self.weights = np.ones(self.n_particles) / self.n_particles

    def predict(self):
        # Move each particle through the transition function and add process noise.
        noise = np.random.randn(*self.particles.shape) * self.process_noise
        self.particles = self.transition_fn(self.particles) + noise
        return self.particles

    def update(self, observation):
        observation = np.asarray(observation)
        # Weight particles by how likely the observation is under each particle.
        likelihoods = np.exp(-0.5 * np.sum((self.observation_fn(self.particles) - observation) ** 2, axis=-1) / self.observation_noise)
        self.weights *= likelihoods
        self.weights += 1e-12
        self.weights /= self.weights.sum()
        self.resample()
        return self.particles, self.weights

    def resample(self):
        cumulative = np.cumsum(self.weights)
        cumulative[-1] = 1.0
        indexes = np.searchsorted(cumulative, np.random.rand(self.n_particles))
        self.particles = self.particles[indexes]
        self.weights.fill(1.0 / self.n_particles)
