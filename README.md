# gaussian-process-state-space-models

A lightweight Python package for Gaussian process state space models (GPSSMs), inference tools, filters, and visualization helpers.

## Features

- Kernel implementations: RBF, Matern, Polynomial, Spectral Mixture
- Dataset generators for synthetic and real-world examples
- State-space filters: Kalman and particle filters
- Inference utilities: ELBO, inducing point selection, recognition networks
- Model abstractions for GPSSMs, linear state-space and variational workflows
- Basic benchmarking and tutorial documentation

## Installation

```bash
python -m pip install .
```

## Quick start

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
predictions = model.predict(X)
```

## Documentation

See `docs/source/index.rst` for documentation sources and tutorials.

## Tests

Run the unit tests with:

```bash
python -m pytest
```

## Scripts

Train a model using the training script:

```bash
python scripts/training/train.py --model variational --kernel matern --n-points 120
```
