"""Random utilities: sampling kernel parameters and randomized synthetic datasets.

These helpers let you quickly generate randomized kernel instances and
synthetic datasets with varying amplitude and noise levels for experiments
and robustness testing.
"""

import numpy as np
from typing import Tuple

from gpssm.datasets.synthetic import generate_synthetic_linear, generate_synthetic_nonlinear
from gpssm.kernels import RBFKernel, MaternKernel, PolynomialKernel, SpectralMixtureKernel, PeriodicKernel


def set_seed(seed: int):
    """Set the global numpy RNG seed for reproducible randomness."""
    np.random.seed(int(seed))


def sample_random_kernel(name: str = None):
    """Return a kernel instance with randomized hyperparameters.

    If `name` is None, a kernel type is chosen at random from available types.
    """
    choices = {
        "rbf": RBFKernel,
        "matern": MaternKernel,
        "polynomial": PolynomialKernel,
        "spectral": SpectralMixtureKernel,
        "periodic": PeriodicKernel,
    }
    if name is None:
        name = np.random.choice(list(choices.keys()))
    name = name.lower()
    cls = choices.get(name, RBFKernel)

    if cls is RBFKernel:
        variance = float(10 ** np.random.uniform(-2, 1))
        lengthscale = float(np.random.uniform(0.05, 5.0))
        return RBFKernel(variance=variance, lengthscale=lengthscale)
    if cls is MaternKernel:
        variance = float(10 ** np.random.uniform(-2, 1))
        lengthscale = float(np.random.uniform(0.05, 5.0))
        nu = float(np.random.choice([0.5, 1.5, 2.5]))
        return MaternKernel(variance=variance, lengthscale=lengthscale, nu=nu)
    if cls is PolynomialKernel:
        variance = float(10 ** np.random.uniform(-2, 1))
        lengthscale = float(np.random.uniform(0.1, 5.0))
        degree = int(np.random.randint(1, 5))
        offset = float(np.random.uniform(0.0, 2.0))
        return PolynomialKernel(variance=variance, lengthscale=lengthscale, degree=degree, offset=offset)
    if cls is SpectralMixtureKernel:
        variance = float(10 ** np.random.uniform(-2, 1))
        lengthscale = float(np.random.uniform(0.05, 2.0))
        n_components = int(np.random.randint(1, 5))
        return SpectralMixtureKernel(variance=variance, lengthscale=lengthscale, n_components=n_components)
    if cls is PeriodicKernel:
        variance = float(10 ** np.random.uniform(-2, 1))
        lengthscale = float(np.random.uniform(0.01, 2.0))
        period = float(np.random.uniform(0.5, 5.0))
        return PeriodicKernel(variance=variance, lengthscale=lengthscale, period=period)


def generate_random_synthetic(n_points: int = 100, kind: str = "linear", amplitude_range=(0.5, 2.0), noise_scale_range=(0.01, 0.2), seed: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a synthetic dataset with randomized amplitude and noise scale.

    - `kind` chooses between 'linear' and 'nonlinear'.
    - `amplitude_range` and `noise_scale_range` are (min, max) tuples.
    """
    if seed is not None:
        set_seed(seed)
    amp = float(np.random.uniform(amplitude_range[0], amplitude_range[1]))
    noise_scale = float(np.random.uniform(noise_scale_range[0], noise_scale_range[1]))

    if kind == "nonlinear":
        X, y = generate_synthetic_nonlinear(n_points)
    else:
        X, y = generate_synthetic_linear(n_points)

    y = amp * y + noise_scale * np.random.randn(*y.shape)
    return X, y
