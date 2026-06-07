import numpy as np

from gpssm.kernels import RBFKernel
from gpssm.models import GaussianProcessStateSpaceModel, VariationalGPModel, FreeFormGPModel, OnlineGPModel, LinearStateSpaceModel


def test_gaussian_process_state_space_model_fit_predict():
    kernel = RBFKernel()
    X = np.linspace(0, 1, 10).reshape(-1, 1)
    y = np.sin(X)
    model = GaussianProcessStateSpaceModel(kernel=kernel)
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (10, 1)
    assert np.all(np.isfinite(preds))


def test_gaussian_process_state_space_model_uncertainty():
    kernel = RBFKernel()
    X = np.linspace(0, 1, 10).reshape(-1, 1)
    y = np.sin(X)
    model = GaussianProcessStateSpaceModel(kernel=kernel)
    model.fit(X, y)
    mean, var = model.predict_with_uncertainty(X)
    assert mean.shape == (10, 1)
    assert var.shape == (10, 1)
    assert np.all(var >= 0)


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


def test_free_form_model_predict():
    model = FreeFormGPModel()
    X = np.linspace(0, 5, 20).reshape(-1, 1)
    y = np.sin(X)
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == y.shape
    assert np.all(np.isfinite(preds))


def test_online_model_update():
    model = OnlineGPModel(lambda_reg=1e-3)
    X = np.linspace(0, 1, 10).reshape(-1, 1)
    y = np.sin(X)
    model.fit(X, y)
    new_X = np.linspace(1, 2, 5).reshape(-1, 1)
    new_y = np.sin(new_X)
    model.update(new_X, new_y)
    preds = model.predict(new_X)
    assert preds.shape == new_y.shape
    assert np.all(np.isfinite(preds))


def test_linear_state_space_simulate():
    transition = np.array([[1.0]])
    observation = np.array([[1.0]])
    process_noise = np.array([[0.01]])
    observation_noise = np.array([[0.1]])
    model = LinearStateSpaceModel(transition, observation, process_noise, observation_noise)
    model.fit(None, None, initial_state=np.array([[0.5]]))
    trajectory = model.simulate(5)
    assert trajectory.shape == (5, 1)
    assert np.all(np.isfinite(trajectory))
