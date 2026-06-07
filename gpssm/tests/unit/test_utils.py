import numpy as np
import matplotlib
matplotlib.use("Agg")

from gpssm.utils import rmse, negative_log_likelihood, plot_timeseries


def test_rmse_zero():
    y_true = np.array([[1.0], [2.0]])
    y_pred = np.array([[1.0], [2.0]])
    assert rmse(y_true, y_pred) == 0.0


def test_negative_log_likelihood():
    y_true = np.array([[0.0], [1.0]])
    y_pred = np.array([[0.0], [1.0]])
    nll = negative_log_likelihood(y_true, y_pred, variance=1.0)
    assert np.isfinite(nll)


def test_plot_timeseries_returns_axes():
    X = np.array([[0.0], [1.0]])
    y = np.array([[0.0], [1.0]])
    ax = plot_timeseries(X, y, label="Test")
    assert ax is not None
    assert len(ax.lines) == 1
