"""Visualization helpers for gpssm."""

import matplotlib.pyplot as plt
import numpy as np


def plot_timeseries(X, y, label="Series", ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(X.flatten(), y.flatten(), label=label)
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.legend()
    return ax


def plot_gp_predictions(X_train, y_train, X_test, mean, variance, ax=None, title="GP Predictions"):
    """Plot the GP training data, mean predictions, and confidence intervals."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
        
    X_train = np.asarray(X_train).flatten()
    y_train = np.asarray(y_train).flatten()
    X_test = np.asarray(X_test).flatten()
    mean = np.asarray(mean).flatten()
    std = np.sqrt(np.asarray(variance).flatten())
    
    # Sort test inputs for proper line plotting
    idx = np.argsort(X_test)
    X_test = X_test[idx]
    mean = mean[idx]
    std = std[idx]
    
    ax.plot(X_train, y_train, "kx", markersize=6, label="Training Data")
    ax.plot(X_test, mean, "b-", linewidth=2, label="Mean Prediction")
    ax.fill_between(X_test, mean - 1.96 * std, mean + 1.96 * std, color="blue", alpha=0.2, label="95% Confidence Interval")
    
    ax.set_xlabel("Input X")
    ax.set_ylabel("Target y")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    return ax


def plot_filter_trajectory(times, true_states, filtered_states, smoothed_states=None, ax=None):
    """Plot true states vs filtered states (and optionally smoothed states)."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
        
    times = np.asarray(times).flatten()
    true_states = np.asarray(true_states).flatten()
    filtered_states = np.asarray(filtered_states).flatten()
    
    ax.plot(times, true_states, "k-", linewidth=2, label="True State")
    ax.plot(times, filtered_states, "r--", linewidth=1.5, label="Filtered Estimate")
    
    if smoothed_states is not None:
        smoothed_states = np.asarray(smoothed_states).flatten()
        ax.plot(times, smoothed_states, "g:", linewidth=2, label="Smoothed (RTS) Estimate")
        
    ax.set_xlabel("Time")
    ax.set_ylabel("State value")
    ax.set_title("Filtering and Smoothing Trajectory")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    return ax


def plot_kernel_heatmap(kernel, x_min: float = -3.0, x_max: float = 3.0, n_points: int = 100, ax=None):
    """Visualize the covariance matrix structure on a grid."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
        
    X = np.linspace(x_min, x_max, n_points).reshape(-1, 1)
    K = kernel(X)
    
    im = ax.imshow(K, extent=[x_min, x_max, x_min, x_max], origin="lower", cmap="viridis")
    plt.colorbar(im, ax=ax, label="Covariance")
    ax.set_title(f"Kernel Heatmap: {type(kernel).__name__}")
    ax.set_xlabel("X")
    ax.set_ylabel("X'")
    return ax
