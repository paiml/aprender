//! Module trait for neural network layers.
//!
//! The Module trait defines the interface for all neural network components,
//! following `PyTorch`'s design (Paszke et al., 2019).

use crate::autograd::Tensor;

/// Base trait for all neural network modules.
///
/// Every layer, activation function, and container implements this trait,
/// providing a uniform interface for:
/// - Forward computation
/// - Parameter access (for optimizers)
/// - Training/evaluation mode switching
///
/// # Example
///
/// ```ignore
/// use aprender::nn::{Module, Linear};
/// use aprender::autograd::Tensor;
///
/// let layer = Linear::new(10, 5);
/// let x = Tensor::randn(&[32, 10]);
/// let output = layer.forward(&x);  // [32, 5]
///
/// // Access parameters for gradient descent
/// for param in layer.parameters() {
///     println!("Shape: {:?}", param.shape());
/// }
/// ```
pub trait Module: Send + Sync {
    /// Perform forward computation.
    ///
    /// This is the main computation method. Given an input tensor,
    /// it returns the output tensor. The computation graph is
    /// automatically recorded for backpropagation.
    fn forward(&self, input: &Tensor) -> Tensor;

    /// Get references to all learnable parameters.
    ///
    /// Used by optimizers to iterate over parameters for gradient updates.
    /// Parameters are returned in a deterministic order.
    fn parameters(&self) -> Vec<&Tensor> {
        vec![]
    }

    /// Get mutable references to all learnable parameters.
    ///
    /// Used by optimizers to update parameters in-place.
    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        vec![]
    }

    /// Refresh any cached computations after parameters have been modified.
    ///
    /// Called after loading weights via `parameters_mut()` to ensure
    /// derived values (like transposed weight matrices) are up-to-date.
    fn refresh_caches(&mut self) {
        // Default: no-op for modules without caches
    }

    /// Set the module to training mode.
    ///
    /// This affects layers like Dropout (active during training)
    /// and `BatchNorm` (uses batch statistics during training).
    fn train(&mut self) {
        // Default: no-op for stateless modules
    }

    /// Set the module to evaluation mode.
    ///
    /// This affects layers like Dropout (disabled during eval)
    /// and `BatchNorm` (uses running statistics during eval).
    fn eval(&mut self) {
        // Default: no-op for stateless modules
    }

    /// Check if the module is in training mode.
    fn training(&self) -> bool {
        true // Default: always training for stateless modules
    }

    /// Zero out gradients for all parameters.
    ///
    /// Should be called before each training iteration.
    fn zero_grad(&mut self) {
        for param in self.parameters_mut() {
            param.zero_grad_();
        }
    }

    /// Get the number of learnable parameters.
    fn num_parameters(&self) -> usize {
        self.parameters().iter().map(|p| p.numel()).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct DummyModule {
        weight: Tensor,
    }

    impl DummyModule {
        fn new() -> Self {
            Self {
                weight: Tensor::ones(&[3, 3]),
            }
        }
    }

    impl Module for DummyModule {
        fn forward(&self, input: &Tensor) -> Tensor {
            input.clone()
        }

        fn parameters(&self) -> Vec<&Tensor> {
            vec![&self.weight]
        }

        fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
            vec![&mut self.weight]
        }
    }

    #[test]
    fn test_module_num_parameters() {
        let module = DummyModule::new();
        assert_eq!(module.num_parameters(), 9); // 3x3 = 9
    }

    #[test]
    fn test_module_parameters() {
        let module = DummyModule::new();
        let params = module.parameters();
        assert_eq!(params.len(), 1);
        assert_eq!(params[0].shape(), &[3, 3]);
    }

    #[test]
    fn test_module_forward() {
        let module = DummyModule::new();
        let input = Tensor::from_slice(&[1.0, 2.0, 3.0]);
        let output = module.forward(&input);
        assert_eq!(output.data(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_module_training() {
        let module = DummyModule::new();
        assert!(module.training());
    }

    #[test]
    fn test_module_zero_grad() {
        let mut module = DummyModule::new();
        module.zero_grad();
        // zero_grad should complete without panic
    }

    #[test]
    fn test_module_parameters_mut() {
        let mut module = DummyModule::new();
        let params = module.parameters_mut();
        assert_eq!(params.len(), 1);
    }

    // Test module that uses all default trait implementations
    struct MinimalModule;

    impl Module for MinimalModule {
        fn forward(&self, input: &Tensor) -> Tensor {
            input.clone()
        }
    }

    #[test]
    fn test_module_default_parameters() {
        let module = MinimalModule;
        let params = module.parameters();
        assert!(params.is_empty());
    }

    #[test]
    fn test_module_default_parameters_mut() {
        let mut module = MinimalModule;
        let params = module.parameters_mut();
        assert!(params.is_empty());
    }

    #[test]
    fn test_module_default_refresh_caches() {
        let mut module = MinimalModule;
        module.refresh_caches(); // Should not panic
    }

    #[test]
    fn test_module_default_train() {
        let mut module = MinimalModule;
        module.train(); // Should not panic
    }

    #[test]
    fn test_module_default_eval() {
        let mut module = MinimalModule;
        module.eval(); // Should not panic
    }

    #[test]
    fn test_module_default_training() {
        let module = MinimalModule;
        assert!(module.training()); // Default is true
    }

    #[test]
    fn test_module_default_zero_grad() {
        let mut module = MinimalModule;
        module.zero_grad(); // Should not panic with empty params
    }

    #[test]
    fn test_module_default_num_parameters() {
        let module = MinimalModule;
        assert_eq!(module.num_parameters(), 0); // No parameters
    }
}
