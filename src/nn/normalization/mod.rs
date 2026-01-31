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
    fn forward(&self, input: &Tensor) -> Tensor {
        // Compute mean and variance over normalized dimensions
        let shape = input.shape();
        let norm_size: usize = self.normalized_shape.iter().product();

        // For simplicity, assume we normalize over the last dimension(s)
        // that match normalized_shape
        assert!(
            shape.len() >= self.normalized_shape.len(),
            "Input must have at least as many dimensions as normalized_shape"
        );

        // Check that the last dimensions match
        let start_dim = shape.len() - self.normalized_shape.len();
        for (i, &ns) in self.normalized_shape.iter().enumerate() {
            assert_eq!(
                shape[start_dim + i],
                ns,
                "Input shape doesn't match normalized_shape at dim {i}"
            );
        }

        // Compute statistics over the last normalized_shape dimensions
        let batch_dims: usize = shape[..start_dim].iter().product();
        let input_data = input.data();

        let mut output_data = vec![0.0; input_data.len()];

        for b in 0..batch_dims {
            let offset = b * norm_size;
            let slice = &input_data[offset..offset + norm_size];

            // Mean
            let mean: f32 = slice.iter().sum::<f32>() / norm_size as f32;

            // Variance
            let var: f32 =
                slice.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / norm_size as f32;

            // Normalize and apply affine transformation
            let std_inv = 1.0 / (var + self.eps).sqrt();

            for i in 0..norm_size {
                let normalized = (slice[i] - mean) * std_inv;
                output_data[offset + i] = if self.elementwise_affine {
                    normalized * self.weight.data()[i] + self.bias.data()[i]
                } else {
                    normalized
                };
            }
        }

        Tensor::new(&output_data, shape)
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

impl Module for BatchNorm1d {
    fn forward(&self, input: &Tensor) -> Tensor {
        assert!(
            input.ndim() == 2 || input.ndim() == 3,
            "BatchNorm1d expects 2D or 3D input, got {}D",
            input.ndim()
        );

        let shape = input.shape();
        // For both 2D [batch, features] and 3D [batch, features, length],
        // features is always at shape[1]
        let (batch_size, features) = (shape[0], shape[1]);

        assert_eq!(
            features, self.num_features,
            "Expected {} features, got {}",
            self.num_features, features
        );

        let input_data = input.data();
        let mut output_data = vec![0.0; input_data.len()];

        if self.training {
            // Compute batch statistics
            for f in 0..features {
                let mut sum = 0.0;
                let mut count = 0;

                // Gather all values for this feature
                if input.ndim() == 2 {
                    for b in 0..batch_size {
                        sum += input_data[b * features + f];
                        count += 1;
                    }
                } else {
                    let length = shape[2];
                    for b in 0..batch_size {
                        for l in 0..length {
                            sum += input_data[b * features * length + f * length + l];
                            count += 1;
                        }
                    }
                }

                let mean = sum / count as f32;

                // Compute variance
                let mut var_sum = 0.0;
                if input.ndim() == 2 {
                    for b in 0..batch_size {
                        let val = input_data[b * features + f];
                        var_sum += (val - mean).powi(2);
                    }
                } else {
                    let length = shape[2];
                    for b in 0..batch_size {
                        for l in 0..length {
                            let val = input_data[b * features * length + f * length + l];
                            var_sum += (val - mean).powi(2);
                        }
                    }
                }
                let var = var_sum / count as f32;

                // Normalize
                let std_inv = 1.0 / (var + self.eps).sqrt();

                if input.ndim() == 2 {
                    for b in 0..batch_size {
                        let idx = b * features + f;
                        let normalized = (input_data[idx] - mean) * std_inv;
                        output_data[idx] = normalized * self.weight.data()[f] + self.bias.data()[f];
                    }
                } else {
                    let length = shape[2];
                    for b in 0..batch_size {
                        for l in 0..length {
                            let idx = b * features * length + f * length + l;
                            let normalized = (input_data[idx] - mean) * std_inv;
                            output_data[idx] =
                                normalized * self.weight.data()[f] + self.bias.data()[f];
                        }
                    }
                }
            }
        } else {
            // Use running statistics
            for f in 0..features {
                let mean = self.running_mean.data()[f];
                let var = self.running_var.data()[f];
                let std_inv = 1.0 / (var + self.eps).sqrt();

                if input.ndim() == 2 {
                    for b in 0..batch_size {
                        let idx = b * features + f;
                        let normalized = (input_data[idx] - mean) * std_inv;
                        output_data[idx] = normalized * self.weight.data()[f] + self.bias.data()[f];
                    }
                } else {
                    let length = shape[2];
                    for b in 0..batch_size {
                        for l in 0..length {
                            let idx = b * features * length + f * length + l;
                            let normalized = (input_data[idx] - mean) * std_inv;
                            output_data[idx] =
                                normalized * self.weight.data()[f] + self.bias.data()[f];
                        }
                    }
                }
            }
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

impl GroupNorm {
    /// Create a new `GroupNorm` layer.
    ///
    /// # Arguments
    ///
    /// * `num_groups` - Number of groups to divide channels into
    /// * `num_channels` - Number of channels (must be divisible by `num_groups`)
    ///
    /// # Panics
    ///
    /// Panics if `num_channels` is not divisible by `num_groups`.
    #[must_use]
    pub fn new(num_groups: usize, num_channels: usize) -> Self {
        assert!(
            num_channels % num_groups == 0,
            "num_channels ({num_channels}) must be divisible by num_groups ({num_groups})"
        );

        Self {
            num_groups,
            num_channels,
            eps: 1e-5,
            weight: constant(&[num_channels], 1.0).requires_grad(),
            bias: zeros(&[num_channels]).requires_grad(),
            affine: true,
        }
    }

    /// Create `GroupNorm` with custom epsilon.
    #[must_use]
    pub fn with_eps(num_groups: usize, num_channels: usize, eps: f32) -> Self {
        let mut layer = Self::new(num_groups, num_channels);
        layer.eps = eps;
        layer
    }

    /// Create `GroupNorm` without learnable parameters.
    #[must_use]
    pub fn without_affine(num_groups: usize, num_channels: usize) -> Self {
        assert!(
            num_channels % num_groups == 0,
            "num_channels ({num_channels}) must be divisible by num_groups ({num_groups})"
        );

        Self {
            num_groups,
            num_channels,
            eps: 1e-5,
            weight: constant(&[num_channels], 1.0),
            bias: zeros(&[num_channels]),
            affine: false,
        }
    }

    /// Get number of groups.
    #[must_use]
    pub fn num_groups(&self) -> usize {
        self.num_groups
    }

    /// Get number of channels.
    #[must_use]
    pub fn num_channels(&self) -> usize {
        self.num_channels
    }
}

impl Module for GroupNorm {
    fn forward(&self, input: &Tensor) -> Tensor {
        let shape = input.shape();
        assert!(
            shape.len() >= 2,
            "GroupNorm expects at least 2D input, got {}D",
            shape.len()
        );

        let (batch_size, channels) = (shape[0], shape[1]);
        assert_eq!(
            channels, self.num_channels,
            "Expected {} channels, got {}",
            self.num_channels, channels
        );

        let channels_per_group = channels / self.num_groups;
        let spatial_size: usize = shape[2..].iter().product();
        let group_size = channels_per_group * spatial_size;

        let input_data = input.data();
        let mut output_data = vec![0.0; input_data.len()];

        for n in 0..batch_size {
            for g in 0..self.num_groups {
                // Compute mean and variance for this group
                let mut sum = 0.0;

                for c in 0..channels_per_group {
                    let channel_idx = g * channels_per_group + c;
                    for s in 0..spatial_size {
                        let idx = n * channels * spatial_size + channel_idx * spatial_size + s;
                        sum += input_data[idx];
                    }
                }

                let mean = sum / group_size as f32;

                let mut var_sum = 0.0;
                for c in 0..channels_per_group {
                    let channel_idx = g * channels_per_group + c;
                    for s in 0..spatial_size {
                        let idx = n * channels * spatial_size + channel_idx * spatial_size + s;
                        var_sum += (input_data[idx] - mean).powi(2);
                    }
                }

                let var = var_sum / group_size as f32;
                let std_inv = 1.0 / (var + self.eps).sqrt();

                // Normalize and apply affine transformation
                for c in 0..channels_per_group {
                    let channel_idx = g * channels_per_group + c;
                    for s in 0..spatial_size {
                        let idx = n * channels * spatial_size + channel_idx * spatial_size + s;
                        let normalized = (input_data[idx] - mean) * std_inv;

                        output_data[idx] = if self.affine {
                            normalized * self.weight.data()[channel_idx]
                                + self.bias.data()[channel_idx]
                        } else {
                            normalized
                        };
                    }
                }
            }
        }

        Tensor::new(&output_data, shape)
    }

    fn parameters(&self) -> Vec<&Tensor> {
        if self.affine {
            vec![&self.weight, &self.bias]
        } else {
            vec![]
        }
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        if self.affine {
            vec![&mut self.weight, &mut self.bias]
        } else {
            vec![]
        }
    }
}

/// Root Mean Square Layer Normalization (Zhang & Sennrich, 2019).
///
/// A simplified version of `LayerNorm` that only uses the root mean square
/// for normalization, without centering (no mean subtraction).
/// This is faster than `LayerNorm` while achieving similar results.
///
/// ```text
/// y = x / RMS(x) * gamma
/// RMS(x) = sqrt(mean(x^2) + eps)
/// ```
///
/// Used in `LLaMA`, Gemma, and other modern transformers.
///
/// # Example
///
/// ```ignore
/// use aprender::nn::{RMSNorm, Module};
/// use aprender::autograd::Tensor;
///
/// let norm = RMSNorm::new(&[256]);  // Normalize over 256 features
/// let x = Tensor::randn(&[32, 10, 256]);  // [batch, seq, features]
/// let y = norm.forward(&x);  // Normalized
/// ```
///
/// # References
///
/// - Zhang, B., & Sennrich, R. (2019). Root Mean Square Layer Normalization.
///   `NeurIPS`.
#[derive(Debug)]
pub struct RMSNorm {
    /// Shape of the normalized dimensions
    normalized_shape: Vec<usize>,
    /// Small constant for numerical stability
    eps: f32,
    /// Learnable scale parameter (gamma)
    weight: Tensor,
    /// Whether to use learnable scale parameter
    elementwise_affine: bool,
}

impl RMSNorm {
    /// Create a new `RMSNorm` layer.
    ///
    /// # Arguments
    ///
    /// * `normalized_shape` - Shape of the dimensions to normalize over
    #[must_use]
    pub fn new(normalized_shape: &[usize]) -> Self {
        let numel: usize = normalized_shape.iter().product();
        Self {
            normalized_shape: normalized_shape.to_vec(),
            eps: 1e-6, // Smaller default eps than LayerNorm (common in LLMs)
            weight: constant(&[numel], 1.0).requires_grad(),
            elementwise_affine: true,
        }
    }

    /// Create `RMSNorm` with custom epsilon.
    #[must_use]
    pub fn with_eps(normalized_shape: &[usize], eps: f32) -> Self {
        let mut layer = Self::new(normalized_shape);
        layer.eps = eps;
        layer
    }

    /// Create `RMSNorm` without learnable parameters.
    #[must_use]
    pub fn without_affine(normalized_shape: &[usize]) -> Self {
        let numel: usize = normalized_shape.iter().product();
        Self {
            normalized_shape: normalized_shape.to_vec(),
            eps: 1e-6,
            weight: constant(&[numel], 1.0),
            elementwise_affine: false,
        }
    }

    /// Get the normalized shape.
    #[must_use]
    pub fn normalized_shape(&self) -> &[usize] {
        &self.normalized_shape
    }

    /// Get the epsilon value.
    #[must_use]
    pub fn eps(&self) -> f32 {
        self.eps
    }

    /// Set weight tensor from external data.
    ///
    /// Used for loading pre-trained weights.
    pub fn set_weight(&mut self, weight: Tensor) {
        self.weight = weight;
    }

    /// Get reference to weight tensor.
    #[must_use]
    pub fn weight(&self) -> &Tensor {
        &self.weight
    }

    /// Create a placeholder `RMSNorm` layer with minimal memory allocation.
    ///
    /// Used for lazy initialization when loading pre-trained weights.
    /// The placeholder uses 1-element tensors instead of full vectors,
    /// reducing memory from O(n) to O(1).
    ///
    /// **IMPORTANT**: This layer will NOT work for inference until
    /// `set_weight()` is called with real weights.
    #[must_use]
    pub fn placeholder(normalized_shape: &[usize]) -> Self {
        Self {
            normalized_shape: normalized_shape.to_vec(),
            eps: 1e-6,
            weight: Tensor::new(&[1.0], &[1]),
            elementwise_affine: true,
        }
    }
}

impl Module for RMSNorm {
    fn forward(&self, input: &Tensor) -> Tensor {
        let shape = input.shape();
        let norm_size: usize = self.normalized_shape.iter().product();

        // Check dimensions
        assert!(
            shape.len() >= self.normalized_shape.len(),
            "Input must have at least as many dimensions as normalized_shape"
        );

        // Check that the last dimensions match
        let start_dim = shape.len() - self.normalized_shape.len();
        for (i, &ns) in self.normalized_shape.iter().enumerate() {
            assert_eq!(
                shape[start_dim + i],
                ns,
                "Input shape doesn't match normalized_shape at dim {i}"
            );
        }

        let batch_dims: usize = shape[..start_dim].iter().product();
        let input_data = input.data();

        let mut output_data = vec![0.0; input_data.len()];

        for b in 0..batch_dims {
            let offset = b * norm_size;
            let slice = &input_data[offset..offset + norm_size];

            // Compute root mean square (no mean subtraction!)
            let mean_sq: f32 = slice.iter().map(|&x| x * x).sum::<f32>() / norm_size as f32;
            let rms = (mean_sq + self.eps).sqrt();
            let rms_inv = 1.0 / rms;

            // Normalize and apply scale
            for i in 0..norm_size {
                let normalized = slice[i] * rms_inv;
                output_data[offset + i] = if self.elementwise_affine {
                    normalized * self.weight.data()[i]
                } else {
                    normalized
                };
            }
        }

        Tensor::new(&output_data, shape)
    }

    fn parameters(&self) -> Vec<&Tensor> {
        if self.elementwise_affine {
            vec![&self.weight]
        } else {
            vec![]
        }
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        if self.elementwise_affine {
            vec![&mut self.weight]
        } else {
            vec![]
        }
    }
}

/// Instance Normalization.
///
/// Normalizes each channel independently for each sample.
/// Commonly used in style transfer networks.
///
/// This is equivalent to `GroupNorm` with `num_groups` = `num_channels`.
#[derive(Debug)]
pub struct InstanceNorm {
    inner: GroupNorm,
}

impl InstanceNorm {
    /// Create a new `InstanceNorm` layer.
    #[must_use]
    pub fn new(num_channels: usize) -> Self {
        Self {
            inner: GroupNorm::new(num_channels, num_channels),
        }
    }

    /// Create `InstanceNorm` without learnable parameters.
    #[must_use]
    pub fn without_affine(num_channels: usize) -> Self {
        Self {
            inner: GroupNorm::without_affine(num_channels, num_channels),
        }
    }
}

impl Module for InstanceNorm {
    fn forward(&self, input: &Tensor) -> Tensor {
        self.inner.forward(input)
    }

    fn parameters(&self) -> Vec<&Tensor> {
        self.inner.parameters()
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        self.inner.parameters_mut()
    }
}


#[cfg(test)]
mod tests;
