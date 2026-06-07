# gaussian-process-state-space-models

This project is a compact Python package for building Gaussian process state space models.
It also includes support for synthetic datasets, filters, model classes, and simple visualization helpers.

## Features

- Kernel functions: RBF, Matern, Polynomial, Spectral Mixture
- Synthetic and real-world dataset utilities
- Kalman and particle filter implementations
- Variational inference support and recognition helpers
- Model classes for GPSSMs, online learning, linear state-space systems, and free-form GP models
- Example scripts and tutorial documentation

## Installation

Install the package so you can use it from Python and update it as you work:

```bash
python -m pip install -e .
```

## Quick start

Use the package to generate data, train a model, and get predictions.

```python
from gpssm import models, datasets
from gpssm.kernels import RBFKernel

X, y = datasets.generate_synthetic_linear(100)
model = models.GaussianProcessStateSpaceModel(kernel=RBFKernel())
model.fit(X, y)
predictions = model.predict(X)
```

```python
from gpssm import models, datasets
from gpssm.kernels import RBFKernel

X, y = datasets.generate_synthetic_linear(100)
model = models.VariationalGPModel(kernel=RBFKernel(), n_inducing=15)
model.fit(X, y)
mean, variance = model.predict_with_uncertainty(X)
```

```python
from gpssm import models, datasets
from gpssm.kernels import RBFKernel

X, y = datasets.generate_synthetic_linear(100)
model = models.FreeFormGPModel()
model.fit(X, y)
predictions = model.predict(X)
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
