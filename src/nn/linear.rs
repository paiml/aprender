//! Fully connected (linear) layer.
//!
//! Implements the transformation y = xW^T + b.
//!
//! # References
//!
//! - Glorot, X., & Bengio, Y. (2010). Understanding the difficulty of training
//!   deep feedforward neural networks. AISTATS.

use super::init::{xavier_uniform, zeros};
use super::module::Module;
use crate::autograd::Tensor;

/// Fully connected layer: y = xW^T + b
///
/// Applies a linear transformation to the incoming data.
/// Weight initialization follows Xavier/Glorot (Glorot & Bengio, 2010).
///
/// # Shape
///
/// - Input: `(*, in_features)` where `*` means any number of batch dimensions
/// - Output: `(*, out_features)`
///
/// # Example
///
/// ```ignore
/// use aprender::nn::{Module, Linear};
/// use aprender::autograd::Tensor;
///
/// let layer = Linear::new(20, 30);  // 20 inputs, 30 outputs
/// let x = Tensor::randn(&[128, 20]);  // batch of 128
/// let output = layer.forward(&x);     // [128, 30]
///
/// assert_eq!(output.shape(), &[128, 30]);
/// ```
pub struct Linear {
    /// Weight matrix, shape: [out_features, in_features]
    weight: Tensor,

    /// Cached transposed weight [in_features, out_features] for fast forward
    /// Computed once when weight is set, avoids transpose overhead every forward.
    weight_t: Option<Tensor>,

    /// Bias vector, shape: [out_features], or None if bias=false
    bias: Option<Tensor>,

    /// Number of input features
    in_features: usize,

    /// Number of output features
    out_features: usize,
}

impl Linear {
    /// Create a new Linear layer with Xavier initialization.
    ///
    /// # Arguments
    ///
    /// * `in_features` - Number of input features
    /// * `out_features` - Number of output features
    ///
    /// # Example
    ///
    /// ```ignore
    /// let layer = Linear::new(784, 256);
    /// ```
    pub fn new(in_features: usize, out_features: usize) -> Self {
        Self::with_seed(in_features, out_features, None)
    }

    /// Create a Linear layer with a specific random seed.
    pub fn with_seed(in_features: usize, out_features: usize, seed: Option<u64>) -> Self {
        let weight = xavier_uniform(
            &[out_features, in_features],
            in_features,
            out_features,
            seed,
        )
        .requires_grad();
        let weight_t = Some(weight.transpose());
        let bias = zeros(&[out_features]).requires_grad();

        Self {
            weight,
            weight_t,
            bias: Some(bias),
            in_features,
            out_features,
        }
    }

    /// Create a Linear layer without bias.
    ///
    /// Useful when followed by BatchNorm which has its own bias.
    pub fn without_bias(in_features: usize, out_features: usize) -> Self {
        Self::without_bias_with_seed(in_features, out_features, None)
    }

    /// Create a Linear layer without bias with a specific random seed.
    pub fn without_bias_with_seed(
        in_features: usize,
        out_features: usize,
        seed: Option<u64>,
    ) -> Self {
        let weight = xavier_uniform(
            &[out_features, in_features],
            in_features,
            out_features,
            seed,
        )
        .requires_grad();
        let weight_t = Some(weight.transpose());

        Self {
            weight,
            weight_t,
            bias: None,
            in_features,
            out_features,
        }
    }

    /// Get the input feature dimension.
    pub fn in_features(&self) -> usize {
        self.in_features
    }

    /// Get the output feature dimension.
    pub fn out_features(&self) -> usize {
        self.out_features
    }

    /// Check if this layer has a bias term.
    pub fn has_bias(&self) -> bool {
        self.bias.is_some()
    }

    /// Set weight tensor from external data.
    ///
    /// Used for loading pre-trained weights from SafeTensors or other formats.
    /// Automatically computes and caches the transposed weight for fast forward.
    pub fn set_weight(&mut self, weight: Tensor) {
        // Pre-compute transpose once during loading (not every forward pass)
        self.weight_t = Some(weight.transpose());
        self.weight = weight;
    }

    /// Set bias tensor from external data.
    ///
    /// Used for loading pre-trained weights.
    pub fn set_bias(&mut self, bias: Tensor) {
        self.bias = Some(bias);
    }

    /// Create a placeholder Linear layer with minimal memory allocation.
    ///
    /// Used for lazy initialization when loading pre-trained weights.
    /// The placeholder uses 1-element tensors instead of full matrices,
    /// reducing memory from O(in*out) to O(1).
    ///
    /// **IMPORTANT**: This layer will NOT work for inference until
    /// `set_weight()` is called with real weights.
    pub fn placeholder(in_features: usize, out_features: usize) -> Self {
        // Use 1-element placeholder tensors to save memory
        Self {
            weight: Tensor::new(&[0.0], &[1]),
            weight_t: None, // Will be set when set_weight() is called
            bias: None,
            in_features,
            out_features,
        }
    }

    /// Get reference to weight tensor.
    pub fn weight(&self) -> &Tensor {
        &self.weight
    }

    /// Get reference to bias tensor if present.
    pub fn bias(&self) -> Option<&Tensor> {
        self.bias.as_ref()
    }

    /// Check if this layer is ready for inference (weight_t is cached).
    ///
    /// Returns false for placeholder layers that haven't had set_weight() called.
    /// This is useful for verifying all layers are properly initialized before forward.
    #[must_use]
    pub fn is_ready(&self) -> bool {
        self.weight_t.is_some()
    }
}

impl Module for Linear {
    fn forward(&self, input: &Tensor) -> Tensor {
        // y = x @ W^T + b
        // Input: [*, in_features] where * is any number of batch dimensions
        // Weight: [out_features, in_features]
        // Output: [*, out_features]

        let input_shape = input.shape();
        let ndim = input_shape.len();

        // Handle N-dimensional input by flattening batch dimensions
        let (reshaped, batch_shape) = if ndim > 2 {
            // Flatten all but last dimension
            let batch_size: usize = input_shape[..ndim - 1].iter().product();
            let in_features = input_shape[ndim - 1];
            let batch_shape: Vec<usize> = input_shape[..ndim - 1].to_vec();

            (input.view(&[batch_size, in_features]), Some(batch_shape))
        } else {
            (input.clone(), None)
        };

        // Use cached transposed weight (computed once during set_weight, not every forward)
        // This eliminates ~450M element copies per forward pass for Qwen2-0.5B
        let weight_t = self.weight_t.as_ref().unwrap_or_else(|| {
            panic!("Linear layer has no cached weight_t. Call set_weight() first or use new().");
        });
        let output = reshaped.matmul(weight_t);

        // Add bias with autograd
        let output = match &self.bias {
            Some(b) => output.broadcast_add(b),
            None => output,
        };

        // Reshape back to original batch dimensions
        match batch_shape {
            Some(mut shape) => {
                shape.push(self.out_features);
                output.view(&shape)
            }
            None => output,
        }
    }

    fn parameters(&self) -> Vec<&Tensor> {
        match &self.bias {
            Some(b) => vec![&self.weight, b],
            None => vec![&self.weight],
        }
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        match &mut self.bias {
            Some(b) => vec![&mut self.weight, b],
            None => vec![&mut self.weight],
        }
    }

    fn refresh_caches(&mut self) {
        // Recompute cached transposed weight after parameters were modified
        self.weight_t = Some(self.weight.transpose());
    }
}

impl std::fmt::Debug for Linear {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Linear")
            .field("in_features", &self.in_features)
            .field("out_features", &self.out_features)
            .field("bias", &self.bias.is_some())
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_forward_shape() {
        let layer = Linear::new(10, 5);
        let x = Tensor::ones(&[32, 10]);
        let output = layer.forward(&x);

        assert_eq!(output.shape(), &[32, 5]);
    }

    #[test]
    fn test_linear_parameters() {
        let layer = Linear::new(10, 5);
        let params = layer.parameters();

        assert_eq!(params.len(), 2); // weight + bias
        assert_eq!(params[0].shape(), &[5, 10]); // weight
        assert_eq!(params[1].shape(), &[5]); // bias
    }

    #[test]
    fn test_linear_without_bias() {
        let layer = Linear::without_bias(10, 5);
        let params = layer.parameters();

        assert_eq!(params.len(), 1); // weight only
        assert!(!layer.has_bias());
    }

    #[test]
    fn test_linear_num_parameters() {
        let layer = Linear::new(10, 5);
        // weight: 10*5 = 50, bias: 5, total: 55
        assert_eq!(layer.num_parameters(), 55);
    }

    #[test]
    fn test_linear_reproducible() {
        let layer1 = Linear::with_seed(10, 5, Some(42));
        let layer2 = Linear::with_seed(10, 5, Some(42));

        assert_eq!(layer1.weight.data(), layer2.weight.data());
    }

    #[test]
    fn test_linear_identity_like() {
        // Create a layer with known weights to verify computation
        let mut layer = Linear::with_seed(3, 3, Some(42));

        // Set weight to identity, bias to zero (use set_weight to update cached transpose)
        let identity = Tensor::new(&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], &[3, 3]);
        let zero_bias = Tensor::zeros(&[3]);

        layer.set_weight(identity.requires_grad());
        layer.set_bias(zero_bias.requires_grad());

        let x = Tensor::new(&[1.0, 2.0, 3.0], &[1, 3]);
        let output = layer.forward(&x);

        // With identity weight and zero bias, output should equal input
        assert_eq!(output.shape(), &[1, 3]);

        let out_data = output.data();
        assert!((out_data[0] - 1.0).abs() < 1e-5);
        assert!((out_data[1] - 2.0).abs() < 1e-5);
        assert!((out_data[2] - 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_linear_with_bias() {
        let mut layer = Linear::with_seed(2, 2, Some(42));

        // Set known weights (use set_weight to update cached transpose)
        layer.set_weight(Tensor::new(&[1.0, 0.0, 0.0, 1.0], &[2, 2]).requires_grad());
        layer.set_bias(Tensor::new(&[10.0, 20.0], &[2]).requires_grad());

        let x = Tensor::new(&[1.0, 2.0], &[1, 2]);
        let output = layer.forward(&x);

        // y = x @ W^T + b = [1, 2] @ [[1,0],[0,1]] + [10, 20] = [1, 2] + [10, 20] = [11, 22]
        let out_data = output.data();
        assert!((out_data[0] - 11.0).abs() < 1e-5);
        assert!((out_data[1] - 22.0).abs() < 1e-5);
    }

    // =========================================================================
    // Property tests: weight_t cache invariant
    // =========================================================================

    #[test]
    fn test_placeholder_is_not_ready() {
        // PROPERTY: Linear::placeholder() always creates a layer that is_ready() == false
        let layer = Linear::placeholder(64, 128);
        assert!(!layer.is_ready(), "Placeholder must not be ready");
    }

    #[test]
    fn test_new_is_ready() {
        // PROPERTY: Linear::new() always creates a layer that is_ready() == true
        let layer = Linear::new(64, 128);
        assert!(layer.is_ready(), "Linear::new() must be ready");
    }

    #[test]
    fn test_set_weight_makes_ready() {
        // PROPERTY: For any placeholder, set_weight() makes is_ready() == true
        let mut layer = Linear::placeholder(32, 64);
        assert!(!layer.is_ready(), "Precondition");

        let weight = Tensor::ones(&[64, 32]);
        layer.set_weight(weight);

        assert!(layer.is_ready(), "set_weight must make layer ready");
    }

    #[test]
    fn test_is_ready_implies_forward_succeeds() {
        // PROPERTY: If is_ready() == true, forward() does not panic
        let layer = Linear::new(8, 4);
        assert!(layer.is_ready());

        let x = Tensor::ones(&[2, 8]);
        let output = layer.forward(&x); // Should not panic
        assert_eq!(output.shape(), &[2, 4]);
    }

    #[test]
    #[should_panic(expected = "weight_t")]
    fn test_not_ready_forward_panics() {
        // PROPERTY: If is_ready() == false, forward() panics
        let layer = Linear::placeholder(8, 4);
        assert!(!layer.is_ready());

        let x = Tensor::ones(&[2, 8]);
        let _ = layer.forward(&x); // Should panic
    }
}
