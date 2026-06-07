"""Visualization script for gpssm."""

import matplotlib.pyplot as plt
from gpssm.datasets import generate_synthetic_linear
from gpssm.kernels import RBFKernel
from gpssm.models import GaussianProcessStateSpaceModel
from gpssm.utils.visualization import plot_timeseries


def main():
    X, y = generate_synthetic_linear(100)
    model = GaussianProcessStateSpaceModel(kernel=RBFKernel())
    model.fit(X, y)
    predictions = model.predict(X)
    fig, ax = plt.subplots(figsize=(8, 4))
    plot_timeseries(X, y, label="Ground truth", ax=ax)
    plot_timeseries(X, predictions, label="Predictions", ax=ax)
    plt.title("GPSM predictions")
    plt.show()

if __name__ == "__main__":
    main()
