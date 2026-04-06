//! Loss function trait

use crate::Tensor;

/// Trait for loss functions
pub trait LossFn {
    /// Compute loss given predictions and targets
    ///
    /// Returns a scalar loss value and sets up gradients for backpropagation
    fn forward(&self, predictions: &Tensor, targets: &Tensor) -> Tensor;

    /// Name of the loss function
    fn name(&self) -> &str;
}
