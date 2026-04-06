//! Evaluation configuration

use super::super::classification::Average;
use super::metric::Metric;

/// Configuration for model evaluation
#[derive(Clone, Debug)]
pub struct EvalConfig {
    /// Metrics to compute
    pub metrics: Vec<Metric>,
    /// Number of cross-validation folds (0 = no CV)
    pub cv_folds: usize,
    /// Random seed for reproducibility
    pub seed: u64,
    /// Parallel evaluation (requires rayon feature)
    pub parallel: bool,
    /// Enable tracing (for renacer integration)
    pub trace_enabled: bool,
}

impl Default for EvalConfig {
    fn default() -> Self {
        Self {
            metrics: vec![Metric::Accuracy, Metric::F1(Average::Weighted)],
            cv_folds: 0,
            seed: 42,
            parallel: false,
            trace_enabled: false,
        }
    }
}
