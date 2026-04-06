//! Hyperparameter Optimization Module (MLOPS-011)
//!
//! Bayesian optimization with TPE and Hyperband schedulers.
//!
//! # Toyota Way: Kaizen
//!
//! Continuous improvement through intelligent search. Each trial informs the next,
//! building knowledge iteratively rather than wasteful exhaustive search.
//!
//! # Example
//!
//! ```ignore
//! use entrenar::optim::hpo::{HyperparameterSpace, ParameterDomain, TPEOptimizer};
//!
//! let mut space = HyperparameterSpace::new();
//! space.add("learning_rate", ParameterDomain::Continuous {
//!     low: 1e-5, high: 1e-1, log_scale: true
//! });
//! space.add("batch_size", ParameterDomain::Discrete { low: 8, high: 128 });
//!
//! let optimizer = TPEOptimizer::new(space);
//! let config = optimizer.suggest(&trials);
//! ```
//!
//! # References
//!
//! \[1\] Bergstra et al. (2011) - Algorithms for Hyper-Parameter Optimization (TPE)
//! \[2\] Li et al. (2018) - Hyperband: A Novel Bandit-Based Approach

mod error;
mod grid;
mod hyperband;
mod tpe;
mod types;

pub use error::{HPOError, Result};
pub use grid::GridSearch;
pub use hyperband::HyperbandScheduler;
pub use tpe::TPEOptimizer;
pub use types::{
    AcquisitionFunction, HyperparameterSpace, ParameterDomain, ParameterValue, SearchStrategy,
    SurrogateModel, Trial, TrialStatus,
};
