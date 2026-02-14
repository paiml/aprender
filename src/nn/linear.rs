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
    /// Weight matrix, shape: [`out_features`, `in_features`]
    weight: Tensor,

    /// Cached transposed weight [`in_features`, `out_features`] for fast forward
    /// Computed once when weight is set, avoids transpose overhead every forward.
    weight_t: Option<Tensor>,

    /// Bias vector, shape: [`out_features`], or None if bias=false
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
    #[must_use]
    pub fn new(in_features: usize, out_features: usize) -> Self {
        Self::with_seed(in_features, out_features, None)
    }

    /// Create a Linear layer with a specific random seed.
    #[must_use]
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
    /// Useful when followed by `BatchNorm` which has its own bias.
    #[must_use]
    pub fn without_bias(in_features: usize, out_features: usize) -> Self {
        Self::without_bias_with_seed(in_features, out_features, None)
    }

    /// Create a Linear layer without bias with a specific random seed.
    #[must_use]
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
    #[must_use]
    pub fn in_features(&self) -> usize {
        self.in_features
    }

    /// Get the output feature dimension.
    #[must_use]
    pub fn out_features(&self) -> usize {
        self.out_features
    }

    /// Check if this layer has a bias term.
    #[must_use]
    pub fn has_bias(&self) -> bool {
        self.bias.is_some()
    }

    /// Set weight tensor from external data.
    ///
    /// Used for loading pre-trained weights from `SafeTensors` or other formats.
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
    #[must_use]
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
    #[must_use]
    pub fn weight(&self) -> &Tensor {
        &self.weight
    }

    /// Get reference to bias tensor if present.
    #[must_use]
    pub fn bias(&self) -> Option<&Tensor> {
        self.bias.as_ref()
    }

    /// Check if this layer is ready for inference (`weight_t` is cached).
    ///
    /// Returns false for placeholder layers that haven't had `set_weight()` called.
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
#[path = "linear_tests.rs"]
mod tests;
