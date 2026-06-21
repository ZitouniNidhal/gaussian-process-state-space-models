"""Base kernel interface with algebra operators for kernel composition."""

from __future__ import annotations
import numpy as np
from abc import ABC, abstractmethod


class Kernel(ABC):
    """Abstract base class for all kernel (covariance) functions.

    Subclasses must implement ``__call__``.  The ``+`` and ``*`` operators
    are overloaded so that kernels can be composed naturally::

        k = RBFKernel() + PeriodicKernel()   # sum kernel
        k = RBFKernel() * MaternKernel()     # product kernel
    """

    def __init__(self, variance: float = 1.0, lengthscale: float = 1.0):
        self.variance = float(variance)
        self.lengthscale = float(lengthscale)

    @abstractmethod
    def __call__(self, X, Y=None) -> np.ndarray:
        """Evaluate the kernel matrix K(X, Y).

        Parameters
        ----------
        X : array-like, shape (n, d)
        Y : array-like, shape (m, d), optional.  If None, Y = X.

        Returns
        -------
        K : np.ndarray, shape (n, m)
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Composition operators
    # ------------------------------------------------------------------

    def __add__(self, other: "Kernel") -> "SumKernel":
        """Return a sum kernel ``k(x,x') = k1(x,x') + k2(x,x')``."""
        from gpssm.kernels.composite import SumKernel
        return SumKernel(self, other)

    def __mul__(self, other: "Kernel") -> "ProductKernel":
        """Return a product kernel ``k(x,x') = k1(x,x') * k2(x,x')``."""
        from gpssm.kernels.composite import ProductKernel
        return ProductKernel(self, other)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _ensure_2d(self, X: np.ndarray) -> np.ndarray:
        """Ensure X is at least 2-D (shape (n, d))."""
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return X

    def diag(self, X: np.ndarray) -> np.ndarray:
        """Return the diagonal of K(X, X) — shape (n,).

        For stationary kernels k(0) = variance, so the default
        implementation is correct.  Override for non-stationary kernels.
        """
        X = self._ensure_2d(X)
        return np.full(X.shape[0], self.variance)

    def gradient_x(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Gradient of k(X, Y) w.r.t. X — shape (n, m, d).

        Subclasses should override this for use in gradient-based
        hyperparameter optimisation.  Default raises NotImplementedError.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement gradient_x."
        )

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"variance={self.variance:.4g}, "
            f"lengthscale={self.lengthscale:.4g})"
        )
