"""ELBO utilities for variational inference."""

import numpy as np

def compute_elbo(likelihood, prior, variational_density):
    """Compute a simple evidence lower bound."""
    return np.mean(likelihood) + np.mean(prior) - np.mean(variational_density)
