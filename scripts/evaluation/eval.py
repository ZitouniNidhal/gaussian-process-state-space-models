"""Evaluation script for gpssm."""

import argparse
from gpssm.datasets import generate_synthetic_linear
from gpssm.kernels import RBFKernel
from gpssm.models import GaussianProcessStateSpaceModel
from gpssm.utils.metrics import rmse


def main():
    parser = argparse.ArgumentParser(description="Evaluate GPSM model performance.")
    parser.add_argument("--n-points", type=int, default=100)
    args = parser.parse_args()

    X, y = generate_synthetic_linear(args.n_points)
    model = GaussianProcessStateSpaceModel(kernel=RBFKernel())
    model.fit(X, y)
    preds = model.predict(X)
    print(f"RMSE: {rmse(y, preds):.4f}")

if __name__ == "__main__":
    main()
