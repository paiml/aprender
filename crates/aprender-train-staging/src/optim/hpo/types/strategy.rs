//! Search strategy types for HPO

use serde::{Deserialize, Serialize};

/// Search strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SearchStrategy {
    /// Exhaustive grid search
    Grid,
    /// Random search
    Random { n_samples: usize },
    /// Bayesian optimization
    Bayesian { n_initial: usize, acquisition: AcquisitionFunction, surrogate: SurrogateModel },
    /// Hyperband (successive halving)
    Hyperband {
        max_iter: usize,
        eta: f64, // Reduction factor (typically 3)
    },
}

/// Acquisition function for Bayesian optimization
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum AcquisitionFunction {
    /// Expected Improvement
    ExpectedImprovement,
    /// Upper Confidence Bound
    UpperConfidenceBound { kappa: f64 },
    /// Probability of Improvement
    ProbabilityOfImprovement,
}

/// Surrogate model for Bayesian optimization
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum SurrogateModel {
    /// Tree-structured Parzen Estimator (recommended)
    TPE,
    /// Gaussian Process
    GaussianProcess,
    /// Random Forest
    RandomForest { n_trees: usize },
}
