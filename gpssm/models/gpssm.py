"""Gaussian process state-space model implementation."""

import numpy as np
from scipy.optimize import minimize
from .base import BaseModel


class GaussianProcessStateSpaceModel(BaseModel):
    """Gaussian process state space model with Cholesky and hyperparameter tuning."""

    def __init__(self, kernel, noise_variance: float = 1e-2):
        super().__init__()
        self.kernel = kernel
        self.noise_variance = float(noise_variance)
        self.X_train = None
        self.y_train = None
        self.L = None  # Cholesky factor of K + σ²I
        self.alpha = None  # L^-T @ L^-1 @ y

    def fit(self, X, y):
        self.X_train = np.asarray(X, dtype=float)
        self.y_train = np.asarray(y, dtype=float)
        
        # Build the training kernel matrix and add observation noise.
        K = self.kernel(self.X_train) + self.noise_variance * np.eye(len(self.X_train))
        
        # Stable decomposition: Cholesky decomposition K = L L^T
        # Add small jitter if not positive definite
        try:
            self.L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            K += 1e-6 * np.eye(len(self.X_train))
            self.L = np.linalg.cholesky(K)
            
        # Solve L L^T alpha = y
        self.alpha = np.linalg.solve(self.L.T, np.linalg.solve(self.L, self.y_train))
        self.is_fitted = True
        return self

    def predict(self, X):
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predicting.")
        X = np.asarray(X, dtype=float)
        K_star = self.kernel(X, self.X_train)
        return K_star @ self.alpha

    def predict_with_uncertainty(self, X):
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predicting.")
        X = np.asarray(X, dtype=float)
        K_star = self.kernel(X, self.X_train)
        
        mean = K_star @ self.alpha
        
        # Solve L v = K_star^T to get v = L^-1 K_star^T
        v = np.linalg.solve(self.L, K_star.T)
        variance = self.kernel.diag(X).reshape(-1, 1) - np.sum(v ** 2, axis=0, keepdims=True).T
        return mean, np.maximum(variance, 0.0)

    def log_marginal_likelihood(self) -> float:
        """Compute the log marginal likelihood of the trained GP."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted to compute log marginal likelihood.")
        
        n = len(self.y_train)
        # log p(y|X) = -0.5 * y^T alpha - sum(log L_ii) - 0.5 * n * log(2*pi)
        data_fit = -0.5 * (self.y_train.T @ self.alpha)
        complexity_penalty = -np.sum(np.log(np.diagonal(self.L)))
        log_norm = -0.5 * n * np.log(2 * np.pi)
        
        # Extract scalar value from matrix multiply if necessary
        return float(np.squeeze(data_fit + complexity_penalty + log_norm))

    def optimize_hyperparameters(self, n_restarts: int = 3):
        """Optimize kernel parameters and noise variance using SciPy.

        Uses the negative log marginal likelihood as the objective function.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted on data before optimization.")

        # Save initial parameters
        init_variance = self.kernel.variance
        init_lengthscale = self.kernel.lengthscale
        init_noise = self.noise_variance

        def objective(params):
            # params = [log(variance), log(lengthscale), log(noise_variance)]
            self.kernel.variance = float(np.exp(params[0]))
            self.kernel.lengthscale = float(np.exp(params[1]))
            self.noise_variance = float(np.exp(params[2]))
            try:
                self.fit(self.X_train, self.y_train)
                return -self.log_marginal_likelihood()
            except np.linalg.LinAlgError:
                return 1e10

        best_val = np.inf
        best_params = None

        # Grid/Random restarts
        for i in range(n_restarts):
            if i == 0:
                x0 = np.log([init_variance, init_lengthscale, init_noise])
            else:
                x0 = np.log([
                    init_variance * np.random.uniform(0.1, 10.0),
                    init_lengthscale * np.random.uniform(0.1, 10.0),
                    init_noise * np.random.uniform(0.1, 10.0)
                ])
            
            res = minimize(objective, x0, method="L-BFGS-B", bounds=[
                (-10, 10),  # log variance
                (-10, 10),  # log lengthscale
                (-15, 2)    # log noise
            ])
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x

        # Set back best parameters and refit
        if best_params is not None:
            self.kernel.variance = float(np.exp(best_params[0]))
            self.kernel.lengthscale = float(np.exp(best_params[1]))
            self.noise_variance = float(np.exp(best_params[2]))
            self.fit(self.X_train, self.y_train)
        return self
