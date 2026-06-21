"""Diagnostics helpers for gpssm."""

import numpy as np


def summarize_fit(model):
    """Summarize model training status and hyperparameters in a dictionary."""
    summary = {
        "is_fitted": getattr(model, "is_fitted", False),
        "parameters": {},
    }
    
    # Extract kernel or other details if available
    if hasattr(model, "kernel"):
        kernel = model.kernel
        summary["parameters"]["kernel_type"] = type(kernel).__name__
        summary["parameters"]["kernel_variance"] = getattr(kernel, "variance", None)
        summary["parameters"]["kernel_lengthscale"] = getattr(kernel, "lengthscale", None)
        if hasattr(kernel, "_lengthscales"):
            summary["parameters"]["kernel_lengthscales_all"] = kernel._lengthscales.tolist()
            
    if hasattr(model, "noise_variance"):
        summary["parameters"]["noise_variance"] = model.noise_variance
        
    if hasattr(model, "elbo_"):
        summary["parameters"]["elbo"] = model.elbo_

    return summary


def check_positive_definite(matrix, tol: float = 1e-12) -> bool:
    """Check if a symmetric matrix is positive-definite by examining eigenvalues."""
    matrix = np.asarray(matrix)
    if not np.allclose(matrix, matrix.T):
        return False
    eigenvalues = np.linalg.eigvalsh(matrix)
    return np.all(eigenvalues > tol)


def condition_number(matrix) -> float:
    """Compute the L2 condition number of a matrix."""
    matrix = np.asarray(matrix)
    return np.linalg.cond(matrix)
