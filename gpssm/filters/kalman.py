"""Simple Kalman filter implementation with RTS smoother."""

import numpy as np


class KalmanFilter:
    """A lightweight linear Kalman filter supporting Joseph form and RTS smoothing."""

    def __init__(self, transition, observation, process_noise, observation_noise):
        self.transition = np.asarray(transition, dtype=float)
        self.observation = np.asarray(observation, dtype=float)
        self.process_noise = np.asarray(process_noise, dtype=float)
        self.observation_noise = np.asarray(observation_noise, dtype=float)
        self.state = None
        self.covariance = None

    def initialize(self, initial_state, initial_covariance):
        self.state = np.asarray(initial_state, dtype=float)
        self.covariance = np.asarray(initial_covariance, dtype=float)

    def predict(self):
        # Predict the next state and uncertainty before seeing the next measurement.
        self.state = self.transition @ self.state
        self.covariance = self.transition @ self.covariance @ self.transition.T + self.process_noise
        return self.state, self.covariance

    def update(self, measurement):
        measurement = np.asarray(measurement, dtype=float)
        # Compute the Kalman gain
        S = self.observation @ self.covariance @ self.observation.T + self.observation_noise
        K = self.covariance @ self.observation.T @ np.linalg.inv(S)
        
        # Correct the state estimate
        innovation = measurement - self.observation @ self.state
        self.state = self.state + K @ innovation
        
        # Joseph form covariance update for numerical stability:
        # P = (I - K H) P (I - K H)^T + K R K^T
        I = np.eye(self.covariance.shape[0])
        KH_diff = I - K @ self.observation
        self.covariance = KH_diff @ self.covariance @ KH_diff.T + K @ self.observation_noise @ K.T
        return self.state, self.covariance

    def smooth(self, filtered_states, filtered_covariances):
        """Perform Rauch-Tung-Striebel (RTS) smoothing on a sequence of filtered states/covs.

        Parameters
        ----------
        filtered_states : list of np.ndarray
            The filtered state estimates from the forward pass.
        filtered_covariances : list of np.ndarray
            The filtered covariance estimates from the forward pass.

        Returns
        -------
        smoothed_states : list of np.ndarray
        smoothed_covariances : list of np.ndarray
        """
        n_steps = len(filtered_states)
        smoothed_states = [None] * n_steps
        smoothed_covariances = [None] * n_steps

        # Terminal state is unchanged
        smoothed_states[-1] = filtered_states[-1].copy()
        smoothed_covariances[-1] = filtered_covariances[-1].copy()

        # Backward recursion
        for t in range(n_steps - 2, -1, -1):
            P_f = filtered_covariances[t]
            # Predicted covariance for t+1 step based on step t state
            P_pred = self.transition @ P_f @ self.transition.T + self.process_noise
            
            # Smoother gain C
            C = P_f @ self.transition.T @ np.linalg.inv(P_pred)
            
            # Smooth state and covariance
            smoothed_states[t] = filtered_states[t] + C @ (smoothed_states[t + 1] - self.transition @ filtered_states[t])
            smoothed_covariances[t] = P_f + C @ (smoothed_covariances[t + 1] - P_pred) @ C.T

        return smoothed_states, smoothed_covariances
