"""Metric helpers for gpssm."""

import numpy as np
from scipy.stats import norm


def rmse(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def negative_log_likelihood(y_true, y_pred, variance=1.0):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    residual = y_true - y_pred
    return 0.5 * np.mean(np.log(2 * np.pi * variance) + (residual ** 2) / variance)


def mae(y_true, y_pred):
    """Compute Mean Absolute Error."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.mean(np.abs(y_true - y_pred))


def r2_score(y_true, y_pred):
    """Compute R^2 (coefficient of determination) score."""
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0.0:
        return 1.0 if ss_res == 0.0 else 0.0
    return 1.0 - (ss_res / ss_tot)


def crps_gaussian(y_true, mean, std):
    """Compute Continuous Ranked Probability Score (CRPS) for Gaussian distributions."""
    y_true = np.asarray(y_true).flatten()
    mean = np.asarray(mean).flatten()
    std = np.asarray(std).flatten()
    
    # Avoid division by zero
    std = np.maximum(std, 1e-12)
    
    z = (y_true - mean) / std
    crps = std * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))
    return np.mean(crps)


def coverage_interval(y_true, lower_bound, upper_bound):
    """Compute the empirical coverage percentage of a prediction interval."""
    y_true = np.asarray(y_true).flatten()
    lower_bound = np.asarray(lower_bound).flatten()
    upper_bound = np.asarray(upper_bound).flatten()
    inside = (y_true >= lower_bound) & (y_true <= upper_bound)
    return np.mean(inside)
