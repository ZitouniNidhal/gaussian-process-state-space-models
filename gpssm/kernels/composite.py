"""Composite kernels: Sum and Product of Kernels."""

import numpy as np
from .base import Kernel


class SumKernel(Kernel):
    """Sum of two kernels k(x, y) = k1(x, y) + k2(x, y)."""

    def __init__(self, k1: Kernel, k2: Kernel):
        super().__init__(variance=k1.variance + k2.variance, lengthscale=(k1.lengthscale + k2.lengthscale) / 2.0)
        self.k1 = k1
        self.k2 = k2

    def __call__(self, X, Y=None) -> np.ndarray:
        return self.k1(X, Y) + self.k2(X, Y)

    def diag(self, X) -> np.ndarray:
        return self.k1.diag(X) + self.k2.diag(X)

    def gradient_x(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        return self.k1.gradient_x(X, Y) + self.k2.gradient_x(X, Y)

    def __repr__(self) -> str:
        return f"({self.k1!r} + {self.k2!r})"


class ProductKernel(Kernel):
    """Product of two kernels k(x, y) = k1(x, y) * k2(x, y)."""

    def __init__(self, k1: Kernel, k2: Kernel):
        super().__init__(variance=k1.variance * k2.variance, lengthscale=(k1.lengthscale + k2.lengthscale) / 2.0)
        self.k1 = k1
        self.k2 = k2

    def __call__(self, X, Y=None) -> np.ndarray:
        return self.k1(X, Y) * self.k2(X, Y)

    def diag(self, X) -> np.ndarray:
        return self.k1.diag(X) * self.k2.diag(X)

    def gradient_x(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        # Product rule: d(fg) = f dg + g df
        K1 = self.k1(X, Y)
        K2 = self.k2(X, Y)
        grad1 = self.k1.gradient_x(X, Y)
        grad2 = self.k2.gradient_x(X, Y)
        return K1[:, :, None] * grad2 + K2[:, :, None] * grad1

    def __repr__(self) -> str:
        return f"({self.k1!r} * {self.k2!r})"
