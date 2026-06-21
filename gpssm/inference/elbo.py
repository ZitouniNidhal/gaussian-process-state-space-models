"""ELBO utilities for variational inference."""

import numpy as np


def gaussian_kl_divergence(mu_q, Sigma_q, mu_p, Sigma_p) -> float:
    """Compute the KL divergence between two multivariate Gaussians.

    KL(q || p) where q = N(mu_q, Sigma_q) and p = N(mu_p, Sigma_p).
    """
    d = mu_q.shape[0]
    Sigma_p_inv = np.linalg.inv(Sigma_p)
    term_trace = np.trace(Sigma_p_inv @ Sigma_q)
    term_diff = (mu_p - mu_q).T @ Sigma_p_inv @ (mu_p - mu_q)
    # Log determinants
    sign_p, logdet_p = np.linalg.slogdet(Sigma_p)
    sign_q, logdet_q = np.linalg.slogdet(Sigma_q)
    term_logdet = logdet_p - logdet_q
    return 0.5 * (term_trace + term_diff - d + term_logdet)


def compute_elbo(K_ff_diag, K_uf, K_uu_inv, y, noise_var, mu_u, Sigma_u) -> float:
    """Compute the Evidence Lower Bound (ELBO) for a sparse variational GP.

    This implements the Titsias (2009) variational sparse GP objective.
    """
    y = np.asarray(y).flatten()
    n = len(y)
    
    # 1. Log likelihood term (reconstruction)
    # Expectation of log p(y | f) under q(f)
    Sigma_u_inv = np.linalg.inv(Sigma_u)
    # Mean of q(f) is K_fu K_uu^-1 mu_u
    # Let's denote A = K_uf^T K_uu^-1
    A = K_uf.T @ K_uu_inv
    mean_f = A @ mu_u
    
    reconstruction_err = -0.5 * np.sum((y - mean_f) ** 2) / noise_var
    log_norm = -0.5 * n * np.log(2 * np.pi * noise_var)
    
    # Trace term: -0.5/noise_var * sum_i [ K_ii - tr(K_iu K_uu^-1 K_ui) + tr(K_iu K_uu^-1 Sigma_u K_uu^-1 K_ui) ]
    # which is tr(K_ff) - tr(A K_uf) + tr(A Sigma_u A^T)
    tr_prior_cov = np.sum(K_ff_diag) - np.trace(A @ K_uf)
    tr_post_cov = np.trace(A @ Sigma_u @ A.T)
    trace_err = -0.5 / noise_var * (tr_prior_cov + tr_post_cov)
    
    likelihood_term = reconstruction_err + log_norm + trace_err
    
    # 2. KL Divergence term KL(q(u) || p(u))
    # p(u) = N(0, K_uu)
    mu_p = np.zeros_like(mu_u)
    K_uu = np.linalg.inv(K_uu_inv)
    kl_term = gaussian_kl_divergence(mu_u, Sigma_u, mu_p, K_uu)
    
    return likelihood_term - kl_term
