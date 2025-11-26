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
        let bias = zeros(&[out_features]).requires_grad();

        Self {
            weight,
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

        Self {
            weight,
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

        // Perform matmul with autograd: [batch, in_features] @ [in_features, out_features]
        let weight_t = self.weight.transpose();
        let output = reshaped.matmul(&weight_t);

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

        // Set weight to identity, bias to zero
        let identity = Tensor::new(&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], &[3, 3]);
        let zero_bias = Tensor::zeros(&[3]);

        layer.weight = identity.requires_grad();
        layer.bias = Some(zero_bias.requires_grad());

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

        // Set known weights
        layer.weight = Tensor::new(&[1.0, 0.0, 0.0, 1.0], &[2, 2]).requires_grad();
        layer.bias = Some(Tensor::new(&[10.0, 20.0], &[2]).requires_grad());

        let x = Tensor::new(&[1.0, 2.0], &[1, 2]);
        let output = layer.forward(&x);

        // y = x @ W^T + b = [1, 2] @ [[1,0],[0,1]] + [10, 20] = [1, 2] + [10, 20] = [11, 22]
        let out_data = output.data();
        assert!((out_data[0] - 11.0).abs() < 1e-5);
        assert!((out_data[1] - 22.0).abs() < 1e-5);
    }
}
