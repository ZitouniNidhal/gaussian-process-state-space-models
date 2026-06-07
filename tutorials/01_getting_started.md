# Getting Started

This tutorial introduces the basic usage of the `gpssm` package.

1. Install the package.
2. Load a synthetic dataset.
3. Create a model and fit it.
4. Use the model to predict new values.

```python
from gpssm.datasets import generate_synthetic_linear
from gpssm.kernels import RBFKernel
from gpssm.models import GaussianProcessStateSpaceModel

X, y = generate_synthetic_linear(100)
model = GaussianProcessStateSpaceModel(kernel=RBFKernel())
model.fit(X, y)
predictions = model.predict(X)
```
