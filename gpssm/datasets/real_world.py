"""Real world dataset utilities."""

import numpy as np

def load_real_world_dataset(n_points: int = 100):
    """Load a small synthetic real-world-style dataset."""
    t = np.linspace(0, 10, n_points)
    signal = np.sin(t) + 0.25 * np.cos(2.0 * t) + 0.1 * np.random.randn(n_points)
    return t.reshape(-1, 1), signal.reshape(-1, 1)
