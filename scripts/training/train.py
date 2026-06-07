"""Training script for gpssm."""

import argparse
from gpssm.datasets import generate_synthetic_linear
from gpssm.kernels import RBFKernel
from gpssm.models import GaussianProcessStateSpaceModel


def main():
    parser = argparse.ArgumentParser(description="Train a GPSM model on synthetic data.")
    parser.add_argument("--n-points", type=int, default=100)
    args = parser.parse_args()

    X, y = generate_synthetic_linear(args.n_points)
    model = GaussianProcessStateSpaceModel(kernel=RBFKernel())
    model.fit(X, y)
    print(f"Trained model on {args.n_points} points.")

if __name__ == "__main__":
    main()
