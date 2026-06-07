"""Visualization helpers for gpssm."""

import matplotlib.pyplot as plt

def plot_timeseries(X, y, label="Series", ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(X.flatten(), y.flatten(), label=label)
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.legend()
    return ax
