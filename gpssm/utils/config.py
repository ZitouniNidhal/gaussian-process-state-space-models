"""Default configuration for gpssm."""

def default_config():
    return {
        "kernel": "RBF",
        "noise_variance": 1e-2,
        "n_inducing": 10,
    }
