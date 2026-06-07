"""Metric helpers for gpssm."""

import numpy as np

def rmse(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def negative_log_likelihood(y_true, y_pred, variance=1.0):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    residual = y_true - y_pred
    return 0.5 * np.mean(np.log(2 * np.pi * variance) + (residual ** 2) / variance)
