import numpy as np
from gpssm.inference import RecognitionModel


def test_recognition_model_fit_and_project():
    inputs = np.random.randn(20, 3)
    targets = np.random.randn(20, 1)
    model = RecognitionModel(input_dim=3, latent_dim=1)
    model.fit(inputs, targets)
    latent = model.project(inputs)
    assert latent.shape == (20, 1)
    decoded = model.decode(latent)
    assert decoded.shape == (20, 3)
