"""Inducing point selection helpers."""

import numpy as np


def select_inducing_points(X, m: int = 10, method: str = "kmeans"):
    """Select inducing points from input data.

    Parameters
    ----------
    X : array-like, shape (n, d)
        Input features.
    m : int
        Number of inducing points to select.
    method : str
        'random' for random subset selection, 'kmeans' for placing
        inducing points at the centroids of k-means clustering.

    Returns
    -------
    Z : np.ndarray, shape (m, d)
        Inducing points.
    """
    X = np.asarray(X, dtype=float)
    if m >= len(X):
        return X.copy()

    if method == "random":
        indices = np.random.choice(len(X), size=m, replace=False)
        return X[indices]

    elif method == "kmeans":
        # Simple k-means implementation to avoid external dependencies like scikit-learn
        # Initialize centroids randomly from data points
        indices = np.random.choice(len(X), size=m, replace=False)
        centroids = X[indices].copy()
        
        for _ in range(10):  # Run for 10 iterations which is usually enough for simple datasets
            # Compute distances to all centroids
            dists = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=-1)
            labels = np.argmin(dists, axis=1)
            
            # Update centroids
            new_centroids = np.zeros_like(centroids)
            for i in range(m):
                mask = labels == i
                if np.any(mask):
                    new_centroids[i] = np.mean(X[mask], axis=0)
                else:
                    # Re-initialize empty clusters to a random point
                    new_centroids[i] = X[np.random.choice(len(X))]
            centroids = new_centroids
        return centroids

    else:
        raise ValueError(f"Unknown inducing point selection method: {method}")
