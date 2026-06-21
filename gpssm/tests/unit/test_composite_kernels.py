"""Unit tests for composite kernels."""

import numpy as np
import pytest
from gpssm.kernels import RBFKernel, MaternKernel, SumKernel, ProductKernel


def test_kernel_sum():
    k1 = RBFKernel(variance=1.5, lengthscale=2.0)
    k2 = MaternKernel(variance=0.5, lengthscale=1.0)
    
    # Check operator overloading
    k_sum = k1 + k2
    assert isinstance(k_sum, SumKernel)
    
    X = np.array([[1.0], [2.0], [3.0]])
    K = k_sum(X)
    
    assert K.shape == (3, 3)
    # k(x, x') = k1(x, x') + k2(x, x')
    expected = k1(X) + k2(X)
    assert np.allclose(K, expected)
    
    # Check diagonal
    assert np.allclose(k_sum.diag(X), k1.diag(X) + k2.diag(X))


def test_kernel_product():
    k1 = RBFKernel(variance=1.2, lengthscale=1.5)
    k2 = MaternKernel(variance=0.8, lengthscale=0.5)
    
    # Check operator overloading
    k_prod = k1 * k2
    assert isinstance(k_prod, ProductKernel)
    
    X = np.array([[1.0], [2.0], [3.0]])
    K = k_prod(X)
    
    assert K.shape == (3, 3)
    # k(x, x') = k1(x, x') * k2(x, x')
    expected = k1(X) * k2(X)
    assert np.allclose(K, expected)
    
    # Check diagonal
    assert np.allclose(k_prod.diag(X), k1.diag(X) * k2.diag(X))
