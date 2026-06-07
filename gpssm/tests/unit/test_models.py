import numpy as np

from gpssm.kernels import RBFKernel
from gpssm.models import GaussianProcessStateSpaceModel, VariationalGPModel


def test_gaussian_process_state_space_model_fit_predict():
    kernel = RBFKernel()
    X = np.linspace(0, 1, 10).reshape(-1, 1)
    y = np.sin(X)
    model = GaussianProcessStateSpaceModel(kernel=kernel)
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (10, 1)
    assert np.all(np.isfinite(preds))


def test_variational_gp_model_predict_uncertainty():
    kernel = RBFKernel()
    X = np.linspace(0, 1, 15).reshape(-1, 1)
    y = np.sin(X)
    model = VariationalGPModel(kernel=kernel, n_inducing=5)
    model.fit(X, y)
    mean, var = model.predict_with_uncertainty(X)
    assert mean.shape == (15, 1)
    assert var.shape == (15, 1)
    assert np.all(var >= 0)
