# gaussian-process-state-space-models

This project is a compact Python package for building Gaussian process state space models.
It also includes support for synthetic datasets, filters, model classes, and simple visualization helpers.

## Features

- **Kernel functions**: RBF (with ARD support), Matern, Polynomial, Spectral Mixture, Periodic, and Composite (Sum & Product) kernels
- **Synthetic datasets**: Linear, Nonlinear, Lorenz-63 chaotic system, Pendulum, and GP prior sample generators
- **Filters**: Kalman, Extended Kalman (EKF), and Particle Filter implementations (with RTS smoother and ESS adaptive resampling)
- **Variational inference**: Titsias optimal variational GP parameter learning, KL divergence computation, and k-means inducing point selection
- **Model classes**: GPSSMs (with Cholesky stabilization and hyperparameter tuning), Online learning (with recursive Woodbury matrix updates), linear state-space systems, and free-form GP models
- **Utilities**: Diagnostic criteria (positive definite check, conditioning) and evaluation metrics (CRPS, R², MAE, RMSE, NLL)
- **Example scripts and tutorial documentation**

## Installation

Install the package so you can use it from Python and update it as you work:

```bash
python -m pip install -e .
```

## Quick start

Use the package to generate data, train a model, and get predictions.

### GPSSM with Hyperparameter Optimization
```python
from gpssm import models, datasets
from gpssm.kernels import RBFKernel

X, y = datasets.generate_synthetic_linear(100)
# Initialize GPSSM
model = models.GaussianProcessStateSpaceModel(kernel=RBFKernel(variance=1.0, lengthscale=1.0))
model.fit(X, y)

# Optimize lengthscale, variance, and noise variance
model.optimize_hyperparameters(n_restarts=3)
predictions, variance = model.predict_with_uncertainty(X)
```

### Composite Kernels
```python
from gpssm.kernels import RBFKernel, PeriodicKernel

# Construct composite kernel
composite_kernel = RBFKernel(variance=1.0, lengthscale=2.0) + PeriodicKernel(variance=0.5, period=1.0)
```

### Extended Kalman Filter (EKF)
```python
import numpy as np
from gpssm.filters import ExtendedKalmanFilter

# EKF on nonlinear system
process_noise = np.array([[0.01]])
obs_noise = np.array([[0.05]])
ekf = ExtendedKalmanFilter(process_noise, obs_noise)
ekf.initialize(np.array([[1.0]]), np.array([[0.1]]))

# Transition function f(x)
ekf.predict(transition_fn=lambda x: 1.1 * x + np.sin(x))
```

## Tests

Run the unit tests with:

```bash
python -m pytest -q
```

## Scripts

Train a model using the sample training script:

```bash
python scripts/training/train.py --model variational --kernel matern --n-points 120
```

## Documentation

Read `docs/source/index.rst` for project documentation and tutorial pages.
