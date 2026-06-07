import numpy as np

from gpssm.filters import KalmanFilter, ParticleFilter


def test_kalman_filter_prediction_and_update():
    transition = np.array([[1.0]])
    observation = np.array([[1.0]])
    process_noise = np.array([[0.01]])
    observation_noise = np.array([[0.1]])

    kf = KalmanFilter(transition, observation, process_noise, observation_noise)
    kf.initialize(np.array([[0.0]]), np.array([[1.0]]))
    state_pred, cov_pred = kf.predict()
    assert state_pred.shape == (1, 1)
    assert cov_pred.shape == (1, 1)

    state_upd, cov_upd = kf.update(np.array([[1.0]]))
    assert state_upd.shape == (1, 1)
    assert cov_upd.shape == (1, 1)
    assert np.all(np.isfinite(state_upd))
    assert np.all(np.isfinite(cov_upd))


def test_particle_filter_predict_and_update():
    def transition_fn(particles):
        return particles

    def observation_fn(particles):
        return particles

    particles = np.zeros((10, 1))
    pf = ParticleFilter(n_particles=10, transition_fn=transition_fn, observation_fn=observation_fn, process_noise=0.1, observation_noise=0.5)
    pf.initialize(particles)
    predicted = pf.predict()
    assert predicted.shape == (10, 1)

    new_particles, weights = pf.update(np.array([[0.0]]))
    assert new_particles.shape == (10, 1)
    assert weights.shape == (10,)
    assert np.isclose(weights.sum(), 1.0)
