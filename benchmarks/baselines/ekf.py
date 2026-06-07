"""Extended Kalman filter baseline implementation."""

import numpy as np

class ExtendedKalmanFilter:
    """A lightweight extended Kalman filter for nonlinear systems."""

    def __init__(self, transition_fn, observation_fn, jacobian_fn, process_noise, observation_noise):
        self.transition_fn = transition_fn
        self.observation_fn = observation_fn
        self.jacobian_fn = jacobian_fn
        self.process_noise = process_noise
        self.observation_noise = observation_noise
        self.state = None
        self.covariance = None

    def initialize(self, initial_state, initial_covariance):
        self.state = np.asarray(initial_state)
        self.covariance = np.asarray(initial_covariance)

    def predict(self):
        self.state = self.transition_fn(self.state)
        F = self.jacobian_fn(self.state)
        self.covariance = F @ self.covariance @ F.T + self.process_noise
        return self.state, self.covariance

    def update(self, measurement):
        measurement = np.asarray(measurement)
        H = self.jacobian_fn(self.state)
        predicted = self.observation_fn(self.state)
        S = H @ self.covariance @ H.T + self.observation_noise
        K = self.covariance @ H.T @ np.linalg.inv(S)
        innovation = measurement - predicted
        self.state = self.state + K @ innovation
        self.covariance = (np.eye(self.covariance.shape[0]) - K @ H) @ self.covariance
        return self.state, self.covariance
