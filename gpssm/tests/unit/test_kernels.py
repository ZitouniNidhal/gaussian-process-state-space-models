import numpy as np

from gpssm.kernels import RBFKernel, MaternKernel, PolynomialKernel, SpectralMixtureKernel


def test_rbf_kernel_matrix_shape():
    kernel = RBFKernel()
    X = np.linspace(0, 1, 5).reshape(-1, 1)
    K = kernel(X)
    assert K.shape == (5, 5)
    assert np.allclose(K, K.T)


def test_matern_kernel_values():
    kernel = MaternKernel(nu=1.5)
    X = np.array([[0.0], [1.0]])
    K = kernel(X)
    assert K.shape == (2, 2)
    assert np.isfinite(K).all()


def test_polynomial_kernel_degree():
    kernel = PolynomialKernel(degree=3)
    X = np.array([[0.0], [1.0]])
    K = kernel(X)
    assert K.shape == (2, 2)
    assert np.all(K >= 0)


def test_spectral_mixture_shape():
    kernel = SpectralMixtureKernel(n_components=2)
    X = np.linspace(0, 1, 4).reshape(-1, 1)
    K = kernel(X)
    assert K.shape == (4, 4)
