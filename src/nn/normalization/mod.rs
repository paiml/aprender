//! Normalization layers for neural networks.
//!
//! These layers normalize activations to stabilize training and improve
//! convergence.
//!
//! # References
//!
//! - Ioffe, S., & Szegedy, C. (2015). Batch normalization: Accelerating
//!   deep network training. ICML.
//! - Ba, J. L., Kiros, J. R., & Hinton, G. E. (2016). Layer normalization.
//!   arXiv:1607.06450.
//! - Wu, Y., & He, K. (2018). Group normalization. ECCV.

use super::init::{constant, zeros};
use super::module::Module;
use crate::autograd::Tensor;

/// Layer Normalization (Ba et al., 2016).
///
/// Normalizes across the last dimension(s) for each sample independently.
/// Used extensively in transformers.
///
/// ```text
/// y = (x - E[x]) / sqrt(Var[x] + eps) * gamma + beta
/// ```
///
/// # Example
///
/// ```ignore
/// use aprender::nn::{LayerNorm, Module};
/// use aprender::autograd::Tensor;
///
/// let norm = LayerNorm::new(&[256]);  // Normalize over 256 features
/// let x = Tensor::randn(&[32, 10, 256]);  // [batch, seq, features]
/// let y = norm.forward(&x);  // Normalized
/// ```
#[derive(Debug)]
pub struct LayerNorm {
    /// Shape of the normalized dimensions
    normalized_shape: Vec<usize>,
    /// Small constant for numerical stability
    eps: f32,
    /// Learnable scale parameter (gamma)
    weight: Tensor,
    /// Learnable shift parameter (beta)
    bias: Tensor,
    /// Whether to learn affine parameters
    elementwise_affine: bool,
}

impl LayerNorm {
    /// Create a new `LayerNorm` layer.
    ///
    /// # Arguments
    ///
    /// * `normalized_shape` - Shape of the dimensions to normalize over
    #[must_use]
    pub fn new(normalized_shape: &[usize]) -> Self {
        let numel: usize = normalized_shape.iter().product();
        Self {
            normalized_shape: normalized_shape.to_vec(),
            eps: 1e-5,
            weight: constant(&[numel], 1.0).requires_grad(),
            bias: zeros(&[numel]).requires_grad(),
            elementwise_affine: true,
        }
    }

    /// Create `LayerNorm` with custom epsilon.
    #[must_use]
    pub fn with_eps(normalized_shape: &[usize], eps: f32) -> Self {
        let mut layer = Self::new(normalized_shape);
        layer.eps = eps;
        layer
    }

    /// Create `LayerNorm` without learnable parameters.
    #[must_use]
    pub fn without_affine(normalized_shape: &[usize]) -> Self {
        let numel: usize = normalized_shape.iter().product();
        Self {
            normalized_shape: normalized_shape.to_vec(),
            eps: 1e-5,
            weight: constant(&[numel], 1.0),
            bias: zeros(&[numel]),
            elementwise_affine: false,
        }
    }

    /// Get the normalized shape.
    #[must_use]
    pub fn normalized_shape(&self) -> &[usize] {
        &self.normalized_shape
    }
}

impl Module for LayerNorm {
    /// ONE PATH: Delegates computation to `nn::functional::layer_norm` (UCBD §4).
    /// Shape validation and non-affine path handled here (Module layer).
    #[provable_contracts_macros::contract("layernorm-kernel-v1", equation = "layernorm")]
    fn forward(&self, input: &Tensor) -> Tensor {
        let shape = input.shape();
        let norm_size: usize = self.normalized_shape.iter().product();

        assert!(
            shape.len() >= self.normalized_shape.len(),
            "Input must have at least as many dimensions as normalized_shape"
        );

        let start_dim = shape.len() - self.normalized_shape.len();
        for (i, &ns) in self.normalized_shape.iter().enumerate() {
            assert_eq!(
                shape[start_dim + i],
                ns,
                "Input shape doesn't match normalized_shape at dim {i}"
            );
        }

        if self.elementwise_affine {
            // ONE PATH: delegate to canonical functional layer_norm
            crate::nn::functional::layer_norm(input, &self.weight, &self.bias, self.eps)
        } else {
            // Non-affine: normalize without weight/bias
            let batch_dims: usize = shape[..start_dim].iter().product();
            let input_data = input.data();
            let mut output_data = vec![0.0; input_data.len()];

            for b in 0..batch_dims {
                let offset = b * norm_size;
                let slice = &input_data[offset..offset + norm_size];
                let mean: f32 = slice.iter().sum::<f32>() / norm_size as f32;
                let var: f32 =
                    slice.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / norm_size as f32;
                let std_inv = 1.0 / (var + self.eps).sqrt();

                for i in 0..norm_size {
                    output_data[offset + i] = (slice[i] - mean) * std_inv;
                }
            }

            Tensor::new(&output_data, shape)
        }
    }

    fn parameters(&self) -> Vec<&Tensor> {
        if self.elementwise_affine {
            vec![&self.weight, &self.bias]
        } else {
            vec![]
        }
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        if self.elementwise_affine {
            vec![&mut self.weight, &mut self.bias]
        } else {
            vec![]
        }
    }
}

/// Batch Normalization for 1D inputs (Ioffe & Szegedy, 2015).
///
/// Normalizes across the batch dimension during training.
/// Uses running statistics during evaluation.
#[derive(Debug)]
pub struct BatchNorm1d {
    num_features: usize,
    eps: f32,
    momentum: f32,
    /// Learnable scale
    weight: Tensor,
    /// Learnable shift
    bias: Tensor,
    /// Running mean (not learnable)
    running_mean: Tensor,
    /// Running variance (not learnable)
    running_var: Tensor,
    /// Training mode
    training: bool,
}

impl BatchNorm1d {
    /// Create a new `BatchNorm1d` layer.
    ///
    /// # Arguments
    ///
    /// * `num_features` - Number of features (channels)
    #[must_use]
    pub fn new(num_features: usize) -> Self {
        Self {
            num_features,
            eps: 1e-5,
            momentum: 0.1,
            weight: constant(&[num_features], 1.0).requires_grad(),
            bias: zeros(&[num_features]).requires_grad(),
            running_mean: zeros(&[num_features]),
            running_var: constant(&[num_features], 1.0),
            training: true,
        }
    }

    /// Set momentum for running statistics update.
    #[must_use]
    pub fn with_momentum(mut self, momentum: f32) -> Self {
        self.momentum = momentum;
        self
    }

    /// Set epsilon for numerical stability.
    #[must_use]
    pub fn with_eps(mut self, eps: f32) -> Self {
        self.eps = eps;
        self
    }
}

impl BatchNorm1d {
    /// Collect indices for a single feature across batch and spatial dims.
    fn feature_indices(shape: &[usize], feature: usize) -> Vec<usize> {
        let (batch_size, features) = (shape[0], shape[1]);
        if shape.len() == 2 {
            (0..batch_size).map(|b| b * features + feature).collect()
        } else {
            let length = shape[2];
            let mut indices = Vec::with_capacity(batch_size * length);
            for b in 0..batch_size {
                for l in 0..length {
                    indices.push(b * features * length + feature * length + l);
                }
            }
            indices
        }
    }

    /// Normalize feature values in-place given mean, std_inv, and affine params.
    fn normalize_feature(
        input_data: &[f32],
        output_data: &mut [f32],
        indices: &[usize],
        mean: f32,
        std_inv: f32,
        gamma: f32,
        beta: f32,
    ) {
        for &idx in indices {
            let normalized = (input_data[idx] - mean) * std_inv;
            output_data[idx] = normalized * gamma + beta;
        }
    }
}

impl Module for BatchNorm1d {
    #[provable_contracts_macros::contract("batchnorm-kernel-v1", equation = "batchnorm_train")]
    fn forward(&self, input: &Tensor) -> Tensor {
        assert!(
            input.ndim() == 2 || input.ndim() == 3,
            "BatchNorm1d expects 2D or 3D input, got {}D",
            input.ndim()
        );

        let shape = input.shape();
        let features = shape[1];

        assert_eq!(
            features, self.num_features,
            "Expected {} features, got {}",
            self.num_features, features
        );

        let input_data = input.data();
        let mut output_data = vec![0.0; input_data.len()];

        for f in 0..features {
            let indices = Self::feature_indices(shape, f);

            let (mean, var) = if self.training {
                let sum: f32 = indices.iter().map(|&i| input_data[i]).sum();
                let mean = sum / indices.len() as f32;
                let var_sum: f32 = indices
                    .iter()
                    .map(|&i| (input_data[i] - mean).powi(2))
                    .sum();
                (mean, var_sum / indices.len() as f32)
            } else {
                (self.running_mean.data()[f], self.running_var.data()[f])
            };

            let std_inv = 1.0 / (var + self.eps).sqrt();
            Self::normalize_feature(
                input_data,
                &mut output_data,
                &indices,
                mean,
                std_inv,
                self.weight.data()[f],
                self.bias.data()[f],
            );
        }

        Tensor::new(&output_data, shape)
    }

    fn parameters(&self) -> Vec<&Tensor> {
        vec![&self.weight, &self.bias]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        vec![&mut self.weight, &mut self.bias]
    }

    fn train(&mut self) {
        self.training = true;
    }

    fn eval(&mut self) {
        self.training = false;
    }

    fn training(&self) -> bool {
        self.training
    }
}

/// Group Normalization (Wu & He, 2018).
///
/// Divides channels into groups and normalizes within each group.
/// Unlike `BatchNorm`, `GroupNorm` is independent of batch size, making it
/// suitable for small batch training (e.g., object detection, segmentation).
///
/// # Shape
///
/// - Input: `(N, C, *)` where C must be divisible by `num_groups`
/// - Output: Same as input
///
/// # Example
///
/// ```ignore
/// use aprender::nn::{GroupNorm, Module};
/// use aprender::autograd::Tensor;
///
/// let norm = GroupNorm::new(32, 256);  // 32 groups, 256 channels
/// let x = Tensor::randn(&[4, 256, 14, 14]);
/// let y = norm.forward(&x);
/// ```
#[derive(Debug)]
pub struct GroupNorm {
    /// Number of groups to divide channels into
    num_groups: usize,
    /// Number of channels (must be divisible by `num_groups`)
    num_channels: usize,
    /// Small constant for numerical stability
    eps: f32,
    /// Learnable scale parameter (gamma)
    weight: Tensor,
    /// Learnable shift parameter (beta)
    bias: Tensor,
    /// Whether to learn affine parameters
    affine: bool,
}

mod mod_part_02;
pub use mod_part_02::*;
