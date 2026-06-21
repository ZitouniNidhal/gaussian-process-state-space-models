"""Extended Kalman Filter (EKF) implementation."""

import numpy as np


class ExtendedKalmanFilter:
    """An Extended Kalman Filter (EKF) for nonlinear state estimation."""

    def __init__(self, process_noise, observation_noise):
        self.process_noise = np.asarray(process_noise, dtype=float)
        self.observation_noise = np.asarray(observation_noise, dtype=float)
        self.state = None
        self.covariance = None

    def initialize(self, initial_state, initial_covariance):
        self.state = np.asarray(initial_state, dtype=float)
        self.covariance = np.asarray(initial_covariance, dtype=float)

    def predict(self, transition_fn, jacobian_fn=None):
        """Predict the next state and uncertainty.

        Parameters
        ----------
        transition_fn : callable
            Nonlinear transition function mapping state to next state: x_{t+1} = f(x_t).
        jacobian_fn : callable, optional
            Function returning the Jacobian of transition_fn evaluated at the current state.
            If None, the Jacobian is estimated numerically.
        """
        # Linearize or compute Jacobian
        if jacobian_fn is not None:
            F = jacobian_fn(self.state)
        else:
            # Simple numerical Jacobian approximation
            eps = 1e-6
            F = []
            flat_state = self.state.flatten()
            for i in range(len(flat_state)):
                state_plus = self.state.copy()
                state_minus = self.state.copy()
                
                # Check dimensions
                if self.state.ndim == 2:
                    state_plus[i, 0] += eps
                    state_minus[i, 0] -= eps
                else:
                    state_plus[i] += eps
                    state_minus[i] -= eps
                    
                deriv = (transition_fn(state_plus) - transition_fn(state_minus)) / (2 * eps)
                F.append(deriv.flatten())
            F = np.column_stack(F)

        # Predict
        self.state = transition_fn(self.state)
        self.covariance = F @ self.covariance @ F.T + self.process_noise
        return self.state, self.covariance

    def update(self, measurement, observation_fn, jacobian_fn=None):
        """Update the estimate with a new observation.

        Parameters
        ----------
        measurement : array-like
            The current observation y_t.
        observation_fn : callable
            Nonlinear observation function mapping state to observation space: y_t = h(x_t).
        jacobian_fn : callable, optional
            Function returning the Jacobian of observation_fn evaluated at the current state.
            If None, the Jacobian is estimated numerically.
        """
        measurement = np.asarray(measurement, dtype=float)
        
        if jacobian_fn is not None:
            H = jacobian_fn(self.state)
        else:
            # Numerical Jacobian for observation function
            eps = 1e-6
            H = []
            flat_state = self.state.flatten()
            for i in range(len(flat_state)):
                state_plus = self.state.copy()
                state_minus = self.state.copy()
                
                if self.state.ndim == 2:
                    state_plus[i, 0] += eps
                    state_minus[i, 0] -= eps
                else:
                    state_plus[i] += eps
                    state_minus[i] -= eps
                    
                deriv = (observation_fn(state_plus) - observation_fn(state_minus)) / (2 * eps)
                H.append(deriv.flatten())
            H = np.column_stack(H)

        innovation = measurement - observation_fn(self.state)
        S = H @ self.covariance @ H.T + self.observation_noise
        K = self.covariance @ H.T @ np.linalg.inv(S)

        # Correct state and covariance (Joseph form for stability)
        self.state = self.state + K @ innovation
        I = np.eye(self.covariance.shape[0])
        self.covariance = (I - K @ H) @ self.covariance @ (I - K @ H).T + K @ self.observation_noise @ K.T
        return self.state, self.covariance
