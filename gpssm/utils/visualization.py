"""Visualization helpers for gpssm."""

import matplotlib.pyplot as plt

def plot_timeseries(X, y, label="Series", ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    # Plot a simple time series line with labels.
    ax.plot(X.flatten(), y.flatten(), label=label)
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.legend()
    return ax
