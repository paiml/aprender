//! Core Metric trait definition

use crate::Tensor;

/// Trait for evaluation metrics
pub trait Metric {
    /// Compute the metric given predictions and targets
    fn compute(&self, predictions: &Tensor, targets: &Tensor) -> f32;

    /// Name of the metric
    fn name(&self) -> &str;

    /// Whether higher values are better (true) or lower (false)
    fn higher_is_better(&self) -> bool {
        true
    }
}
