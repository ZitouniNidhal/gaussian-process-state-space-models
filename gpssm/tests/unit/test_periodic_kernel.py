import numpy as np
from gpssm.kernels import PeriodicKernel


def test_periodic_kernel_matrix_properties():
    X = np.linspace(0, 2.0, 5).reshape(-1, 1)
    kernel = PeriodicKernel(variance=2.0, lengthscale=0.5, period=1.0)
    K = kernel(X)
    # symmetric
    assert np.allclose(K, K.T, atol=1e-8)
    # diagonal equals variance
    diag = np.diag(K)
    assert np.allclose(diag, 2.0, atol=1e-8)
    # positive semi-definite (eigenvalues non-negative within tolerance)
    eigs = np.linalg.eigvalsh(K)
    assert np.all(eigs >= -1e-8)
