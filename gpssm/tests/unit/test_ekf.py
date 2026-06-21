"""Unit tests for Extended Kalman Filter."""

import numpy as np
import pytest
from gpssm.filters import ExtendedKalmanFilter, KalmanFilter


def test_ekf_against_linear_kf():
    # A simple 1D linear system: state_t = 1.2 * state_{t-1} + noise
    # observation_t = 0.8 * state_t + noise
    transition = np.array([[1.2]])
    observation = np.array([[0.8]])
    process_noise = np.array([[0.01]])
    obs_noise = np.array([[0.05]])

    kf = KalmanFilter(transition, observation, process_noise, obs_noise)
    ekf = ExtendedKalmanFilter(process_noise, obs_noise)

    initial_state = np.array([[1.0]])
    initial_cov = np.array([[0.1]])

    kf.initialize(initial_state, initial_cov)
    ekf.initialize(initial_state, initial_cov)

    # Nonlinear transition and observation functions representing the same system
    def f(x):
        return transition @ x

    def h(x):
        return observation @ x

    # 1. Predict step
    kf_state, kf_cov = kf.predict()
    ekf_state, ekf_cov = ekf.predict(f)

    assert np.allclose(kf_state, ekf_state)
    assert np.allclose(kf_cov, ekf_cov)

    # 2. Update step
    measurement = np.array([[1.1]])
    kf_state, kf_cov = kf.update(measurement)
    ekf_state, ekf_cov = ekf.update(measurement, h)

    assert np.allclose(kf_state, ekf_state)
    assert np.allclose(kf_cov, ekf_cov)
