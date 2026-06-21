"""Radial basis function (RBF / squared-exponential) kernel."""

import numpy as np
from .base import Kernel


class RBFKernel(Kernel):
    """Radial basis function (squared-exponential) kernel.

    Supports **Automatic Relevance Determination (ARD)**: pass a 1-D array
    as ``lengthscale`` to assign one length-scale per input dimension.

    .. math::

        k(x, x') = \\sigma^2 \\exp\\!\\left(-\\frac{1}{2}
                    \\sum_d \\frac{(x_d - x'_d)^2}{\\ell_d^2}\\right)

    Parameters
    ----------
    variance : float
        Output variance :math:`\\sigma^2`.
    lengthscale : float or array-like
        Length-scale(s).  A scalar uses the same length-scale for every
        dimension; a vector enables ARD.
    """

    def __init__(self, variance: float = 1.0, lengthscale=1.0):
        # Store lengthscale as an array to support ARD.
        ls = np.atleast_1d(np.asarray(lengthscale, dtype=float))
        # Call super with a scalar for the __repr__; we override self.lengthscale below.
        super().__init__(variance=variance, lengthscale=float(ls.mean()))
        self._lengthscales = ls  # shape (d,) or (1,)

    # ------------------------------------------------------------------
    # Core computation
    # ------------------------------------------------------------------

    def _scaled_sqdist(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Compute the scaled squared distances between rows of X and Y."""
        # Broadcast lengthscales across all dimensions.
        ls = self._lengthscales  # (d,) or (1,)
        diff = (X[:, None, :] - Y[None, :, :]) / ls  # (n, m, d)
        return np.sum(diff ** 2, axis=-1)              # (n, m)

    def __call__(self, X, Y=None) -> np.ndarray:
        X = self._ensure_2d(X)
        Y = X if Y is None else self._ensure_2d(Y)
        sqdist = self._scaled_sqdist(X, Y)
        return self.variance * np.exp(-0.5 * sqdist)

    def diag(self, X) -> np.ndarray:
        X = self._ensure_2d(X)
        return np.full(X.shape[0], self.variance)

    def gradient_x(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Gradient of k(X, Y) w.r.t. X — shape (n, m, d).

        Useful for computing predictive gradients and Hamiltonian Monte Carlo.
        """
        X = self._ensure_2d(X)
        Y = self._ensure_2d(Y)
        ls = self._lengthscales                        # (d,) or (1,)
        diff = X[:, None, :] - Y[None, :, :]          # (n, m, d)
        K = self(X, Y)                                 # (n, m)
        # dk/dX_i = K(x,y) * (-(x-y) / l^2)
        return -K[:, :, None] * diff / (ls ** 2)       # (n, m, d)

    def __repr__(self) -> str:
        if self._lengthscales.size == 1:
            ls_str = f"{self._lengthscales[0]:.4g}"
        else:
            ls_str = f"ARD({self._lengthscales})"
        return f"RBFKernel(variance={self.variance:.4g}, lengthscale={ls_str})"
