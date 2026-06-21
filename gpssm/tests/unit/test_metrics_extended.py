"""Unit tests for extended metrics."""

import numpy as np
import pytest
from gpssm.utils.metrics import mae, r2_score, crps_gaussian, coverage_interval


def test_mae():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.5, 2.0, 2.5])
    assert np.allclose(mae(y_true, y_pred), 0.3333333333333333)


def test_r2_score():
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert np.allclose(r2_score(y_true, y_pred), 1.0)
    
    y_pred_bad = np.array([3.0, 3.0, 3.0, 3.0, 3.0])
    assert np.allclose(r2_score(y_true, y_pred_bad), 0.0)


def test_crps_gaussian():
    y_true = np.array([0.0])
    mean = np.array([0.0])
    std = np.array([1.0])
    # For z=0, CRPS is 1 * (0 + 2*pdf(0) - 1/sqrt(pi))
    # pdf(0) = 1/sqrt(2*pi)
    # CRPS = 2/sqrt(2*pi) - 1/sqrt(pi) = sqrt(2/pi) - 1/sqrt(pi)
    expected = np.sqrt(2 / np.pi) - 1 / np.sqrt(np.pi)
    assert np.allclose(crps_gaussian(y_true, mean, std), expected)


def test_coverage_interval():
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    lower = np.array([0.5, 1.5, 3.5, 3.5, 4.5])
    upper = np.array([1.5, 2.5, 4.5, 4.5, 5.5])
    # 3.0 is outside [3.5, 4.5], all others are inside -> coverage 4/5 = 0.8
    assert np.allclose(coverage_interval(y_true, lower, upper), 0.8)
