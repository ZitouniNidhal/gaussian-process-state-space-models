"""Synthetic dataset utilities."""

import numpy as np


def generate_synthetic_linear(n_points: int = 100):
    """Generate a simple linear synthetic time series."""
    X = np.linspace(0, 10, n_points).reshape(-1, 1)
    y = 0.5 * X + 0.2 * np.sin(X * 2.0) + 0.05 * np.random.randn(n_points, 1)
    return X, y


def generate_synthetic_nonlinear(n_points: int = 100):
    """Generate a nonlinear synthetic time series."""
    X = np.linspace(0, 10, n_points).reshape(-1, 1)
    y = np.sin(X) + 0.5 * np.cos(2.0 * X) + 0.1 * np.random.randn(n_points, 1)
    return X, y


def generate_lorenz(n_points: int = 1000, dt: float = 0.01, s=10.0, r=28.0, b=8.0/3.0):
    """Generate 3D chaotic trajectory from the Lorenz-63 system."""
    xs = np.empty(n_points)
    ys = np.empty(n_points)
    zs = np.empty(n_points)
    
    # Initial state
    xs[0], ys[0], zs[0] = (1.0, 1.0, 1.0)
    
    # Run ODE integration
    for i in range(1, n_points):
        dx = s * (ys[i - 1] - xs[i - 1])
        dy = xs[i - 1] * (r - zs[i - 1]) - ys[i - 1]
        dz = xs[i - 1] * ys[i - 1] - b * zs[i - 1]
        xs[i] = xs[i - 1] + dx * dt
        ys[i] = ys[i - 1] + dy * dt
        zs[i] = zs[i - 1] + dz * dt
        
    X = np.linspace(0, dt * n_points, n_points).reshape(-1, 1)
    # Stack output dimensions (n_points, 3)
    y = np.column_stack([xs, ys, zs])
    return X, y


def generate_pendulum(n_points: int = 200, dt: float = 0.05, g=9.81, L=1.0, noise_std=0.02):
    """Simulate a simple nonlinear pendulum with noise.

    State: [theta, theta_dot]
    """
    states = []
    # Initial state: 1 radian displacement, 0 velocity
    state = np.array([1.0, 0.0])
    
    for _ in range(n_points):
        states.append(state.copy())
        # Semi-implicit Euler integration
        theta, theta_dot = state
        theta_dot_next = theta_dot - (g / L) * np.sin(theta) * dt
        theta_next = theta + theta_dot_next * dt
        state = np.array([theta_next, theta_dot_next]) + np.random.randn(2) * noise_std
        
    X = np.linspace(0, dt * n_points, n_points).reshape(-1, 1)
    y = np.vstack(states)
    return X, y


def generate_gp_sample(kernel, n_points: int = 100, x_min: float = 0.0, x_max: float = 10.0, noise_std: float = 1e-3):
    """Generate a sample path directly from a Gaussian Process prior."""
    X = np.linspace(x_min, x_max, n_points).reshape(-1, 1)
    K = kernel(X) + (noise_std ** 2 + 1e-8) * np.eye(n_points)
    L = np.linalg.cholesky(K)
    # y = L * z where z ~ N(0, I)
    z = np.random.randn(n_points, 1)
    y = L @ z
    return X, y
