"""Simple Kalman filter implementation."""

import numpy as np

class KalmanFilter:
    """A lightweight linear Kalman filter."""

    def __init__(self, transition, observation, process_noise, observation_noise):
        self.transition = np.asarray(transition)
        self.observation = np.asarray(observation)
        self.process_noise = np.asarray(process_noise)
        self.observation_noise = np.asarray(observation_noise)
        self.state = None
        self.covariance = None

    def initialize(self, initial_state, initial_covariance):
        self.state = np.asarray(initial_state)
        self.covariance = np.asarray(initial_covariance)

    def predict(self):
        self.state = self.transition @ self.state
        self.covariance = self.transition @ self.covariance @ self.transition.T + self.process_noise
        return self.state, self.covariance

    def update(self, measurement):
        measurement = np.asarray(measurement)
        S = self.observation @ self.covariance @ self.observation.T + self.observation_noise
        K = self.covariance @ self.observation.T @ np.linalg.inv(S)
        innovation = measurement - self.observation @ self.state
        self.state = self.state + K @ innovation
        self.covariance = (np.eye(self.covariance.shape[0]) - K @ self.observation) @ self.covariance
        return self.state, self.covariance
