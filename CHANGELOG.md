# Changelog

## 0.2.0

- Added **Extended Kalman Filter (EKF)** for nonlinear state estimation
- Implemented **composite kernels** (Sum & Product of kernels) via operator overloading (`+` and `*`)
- Added **Automatic Relevance Determination (ARD)** support and analytic gradients for `RBFKernel`
- Enhanced Kalman Filter with **Rauch-Tung-Striebel (RTS) smoothing** and stable **Joseph form** covariance updates
- Added **Effective Sample Size (ESS)** diagnostics and adaptive resampling to the Particle Filter
- Upgraded `GaussianProcessStateSpaceModel` to support **log marginal likelihood** calculation and **hyperparameter optimization**
- Added optimal **Titsias (2009) sparse GP ELBO** calculation and **k-means clustering** for inducing point selection
- Implemented recursive **Woodbury matrix updates** and predictive uncertainty for `OnlineGPModel`
- Added **Lorenz-63 chaotic system**, **nonlinear pendulum**, and **GP prior sample path** generators
- Added extra validation metrics (CRPS, R², MAE, coverage intervals) and rich plotting functions

## 0.1.0

- Initial package scaffolding
- Core kernel and model abstractions
- Dataset utilities and basic filters
- Documentation and tutorials
