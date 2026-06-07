"""Training script for gpssm."""

import argparse
from gpssm.datasets import generate_synthetic_linear
from gpssm.kernels import RBFKernel, MaternKernel, PolynomialKernel, SpectralMixtureKernel
from gpssm.models import GaussianProcessStateSpaceModel, VariationalGPModel

KERNEL_CLASSES = {
    "rbf": RBFKernel,
    "matern": MaternKernel,
    "polynomial": PolynomialKernel,
    "spectral": SpectralMixtureKernel,
}

MODEL_CLASSES = {
    "gpssm": GaussianProcessStateSpaceModel,
    "variational": VariationalGPModel,
}


def build_kernel(name: str):
    kernel_cls = KERNEL_CLASSES.get(name.lower(), RBFKernel)
    return kernel_cls()


def build_model(name: str, kernel):
    model_cls = MODEL_CLASSES.get(name.lower(), GaussianProcessStateSpaceModel)
    if model_cls is VariationalGPModel:
        return model_cls(kernel=kernel, n_inducing=20)
    return model_cls(kernel=kernel)


def main():
    parser = argparse.ArgumentParser(description="Train a GPSM model on synthetic data.")
    parser.add_argument("--n-points", type=int, default=100)
    parser.add_argument("--model", type=str, default="gpssm", choices=MODEL_CLASSES)
    parser.add_argument("--kernel", type=str, default="rbf", choices=KERNEL_CLASSES)
    args = parser.parse_args()

    X, y = generate_synthetic_linear(args.n_points)
    kernel = build_kernel(args.kernel)
    model = build_model(args.model, kernel)
    model.fit(X, y)
    print(f"Trained {args.model} model with {args.kernel} kernel on {args.n_points} points.")

if __name__ == "__main__":
    main()
