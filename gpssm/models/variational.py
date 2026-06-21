"""Variational GP model implementation."""

import numpy as np
from .base import BaseModel
from ..inference import select_inducing_points, compute_elbo


class VariationalGPModel(BaseModel):
    """Sparse variational Gaussian process model."""

    def __init__(self, kernel, n_inducing: int = 10, noise_variance: float = 1e-2):
        super().__init__()
        self.kernel = kernel
        self.n_inducing = int(n_inducing)
        self.noise_variance = float(noise_variance)
        self.Z = None
        self.alpha = None
        self.K_uu_inv = None
        self.L_uu = None  # Cholesky factor of K_uu
        self.X_train = None
        self.y_train = None
        self.elbo_ = None
        self.mu_u = None
        self.Sigma_u = None

    def fit(self, X, y):
        self.X_train = np.asarray(X, dtype=float)
        self.y_train = np.asarray(y, dtype=float).reshape(-1, 1)
        
        # Select inducing points using k-means
        self.Z = select_inducing_points(self.X_train, self.n_inducing, method="kmeans")

        # Build inducing point covariances
        K_uu = self.kernel(self.Z) + 1e-6 * np.eye(len(self.Z))  # Small jitter
        K_uf = self.kernel(self.Z, self.X_train)
        K_ff_diag = self.kernel.diag(self.X_train)
        
        self.L_uu = np.linalg.cholesky(K_uu)
        self.K_uu_inv = np.linalg.inv(K_uu)

        # Variational parameters (Titsias 2009 optimal parameters)
        # Sigma_u = (K_uu^-1 + noise_variance^-1 K_uu^-1 K_uf K_fu K_uu^-1)^-1
        # Let Q = K_uu^-1 K_uf
        Q = self.K_uu_inv @ K_uf
        Lambda = self.K_uu_inv + (1.0 / self.noise_variance) * (Q @ Q.T)
        self.Sigma_u = np.linalg.inv(Lambda)
        
        # mu_u = noise_variance^-1 Sigma_u K_uu^-1 K_uf y
        self.mu_u = (1.0 / self.noise_variance) * self.Sigma_u @ Q @ self.y_train

        # Compute alpha for quick predictions: mean(f*) = K_*u K_uu^-1 mu_u = K_*u alpha
        self.alpha = self.K_uu_inv @ self.mu_u
        
        # Compute and cache ELBO
        self.elbo_ = compute_elbo(
            K_ff_diag, K_uf, self.K_uu_inv, self.y_train, 
            self.noise_variance, self.mu_u, self.Sigma_u
        )

        self.is_fitted = True
        return self

    def predict(self, X):
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predicting.")
        X = np.asarray(X, dtype=float)
        K_star = self.kernel(X, self.Z)
        return K_star @ self.alpha

    def predict_with_uncertainty(self, X):
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predicting.")
        X = np.asarray(X, dtype=float)
        K_star = self.kernel(X, self.Z)
        
        # Predictive mean: K_*u alpha
        mean = K_star @ self.alpha
        
        # Predictive variance: k_** - K_*u K_uu^-1 K_u* + K_*u K_uu^-1 Sigma_u K_uu^-1 K_u*
        # Let W_star = K_uu^-1 K_u*
        W_star = self.K_uu_inv @ K_star.T
        prior_var = self.kernel.diag(X).reshape(-1, 1) - np.sum(K_star * W_star.T, axis=1, keepdims=True)
        post_var = np.sum(K_star @ self.K_uu_inv @ self.Sigma_u * W_star.T, axis=1, keepdims=True)
        
        variance = prior_var + post_var
        return mean, np.maximum(variance, 0.0)
