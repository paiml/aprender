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
mod tests {
    use super::*;

    #[test]
    fn test_layer_norm_shape() {
        let norm = LayerNorm::new(&[256]);
        let x = Tensor::ones(&[32, 10, 256]);
        let y = norm.forward(&x);

        assert_eq!(y.shape(), x.shape());
    }

    #[test]
    fn test_layer_norm_normalization() {
        let norm = LayerNorm::without_affine(&[4]);

        // Input: single sample with known values
        let x = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[1, 4]);
        let y = norm.forward(&x);

        // After normalization, mean should be ~0, std ~1
        let y_data = y.data();
        let mean: f32 = y_data.iter().sum::<f32>() / 4.0;
        let var: f32 = y_data.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / 4.0;

        assert!((mean).abs() < 1e-5, "Mean should be ~0, got {mean}");
        assert!((var - 1.0).abs() < 0.1, "Var should be ~1, got {var}");
    }

    #[test]
    fn test_layer_norm_parameters() {
        let norm = LayerNorm::new(&[64]);
        let params = norm.parameters();

        assert_eq!(params.len(), 2); // weight and bias
        assert_eq!(params[0].numel(), 64); // weight
        assert_eq!(params[1].numel(), 64); // bias
    }

    #[test]
    fn test_layer_norm_without_affine() {
        let norm = LayerNorm::without_affine(&[64]);
        let params = norm.parameters();

        assert!(params.is_empty());
    }

    #[test]
    fn test_batch_norm_1d_shape() {
        let norm = BatchNorm1d::new(64);
        let x = Tensor::ones(&[32, 64]);
        let y = norm.forward(&x);

        assert_eq!(y.shape(), x.shape());
    }

    #[test]
    fn test_batch_norm_1d_train_eval() {
        let mut norm = BatchNorm1d::new(64);

        assert!(norm.training());

        norm.eval();
        assert!(!norm.training());

        norm.train();
        assert!(norm.training());
    }

    #[test]
    fn test_group_norm_shape() {
        let norm = GroupNorm::new(32, 256);
        let x = Tensor::ones(&[4, 256, 14, 14]);
        let y = norm.forward(&x);

        assert_eq!(y.shape(), x.shape());
    }

    #[test]
    fn test_group_norm_2d_input() {
        // GroupNorm should also work with 2D input (no spatial dims)
        let norm = GroupNorm::new(8, 64);
        let x = Tensor::ones(&[4, 64]);
        let y = norm.forward(&x);

        assert_eq!(y.shape(), &[4, 64]);
    }

    #[test]
    fn test_group_norm_parameters() {
        let norm = GroupNorm::new(32, 256);
        let params = norm.parameters();

        assert_eq!(params.len(), 2);
        assert_eq!(params[0].numel(), 256); // weight
        assert_eq!(params[1].numel(), 256); // bias
    }

    #[test]
    fn test_group_norm_without_affine() {
        let norm = GroupNorm::without_affine(32, 256);
        let params = norm.parameters();

        assert!(params.is_empty());
    }

    #[test]
    fn test_group_norm_normalization() {
        let norm = GroupNorm::without_affine(2, 4);

        // 2 groups of 2 channels each
        let x = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[1, 4]);
        let y = norm.forward(&x);

        // Each group should be normalized independently
        // Group 0: [1, 2] -> mean=1.5, std=0.5 -> [-1, 1] (approx)
        // Group 1: [3, 4] -> mean=3.5, std=0.5 -> [-1, 1] (approx)
        let y_data = y.data();

        // Check first group normalized
        let g0_mean = (y_data[0] + y_data[1]) / 2.0;
        assert!(
            g0_mean.abs() < 1e-5,
            "Group 0 mean should be ~0, got {g0_mean}"
        );

        // Check second group normalized
        let g1_mean = (y_data[2] + y_data[3]) / 2.0;
        assert!(
            g1_mean.abs() < 1e-5,
            "Group 1 mean should be ~0, got {g1_mean}"
        );
    }

    #[test]
    fn test_instance_norm_shape() {
        let norm = InstanceNorm::new(64);
        let x = Tensor::ones(&[4, 64, 8, 8]);
        let y = norm.forward(&x);

        assert_eq!(y.shape(), x.shape());
    }

    #[test]
    fn test_instance_norm_is_group_norm_with_num_groups_equal_channels() {
        // InstanceNorm is equivalent to GroupNorm with num_groups = num_channels
        let instance_norm = InstanceNorm::without_affine(4);
        let group_norm = GroupNorm::without_affine(4, 4);

        let x = Tensor::new(
            &[
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0,
            ],
            &[1, 4, 2, 2],
        );

        let y_instance = instance_norm.forward(&x);
        let y_group = group_norm.forward(&x);

        // Should produce identical results
        for (a, b) in y_instance.data().iter().zip(y_group.data().iter()) {
            assert!(
                (a - b).abs() < 1e-5,
                "InstanceNorm and GroupNorm should match"
            );
        }
    }

    // ==========================================================================
    // RMSNorm Tests
    // ==========================================================================

    #[test]
    fn test_rms_norm_shape() {
        let norm = RMSNorm::new(&[256]);
        let x = Tensor::ones(&[32, 10, 256]);
        let y = norm.forward(&x);

        assert_eq!(y.shape(), x.shape());
    }

    #[test]
    fn test_rms_norm_basic_normalization() {
        let norm = RMSNorm::without_affine(&[4]);

        // Input: single sample with known values
        let x = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[1, 4]);
        let y = norm.forward(&x);

        // RMS = sqrt(mean(x^2)) = sqrt((1+4+9+16)/4) = sqrt(7.5) ≈ 2.739
        // Normalized values: x / RMS
        let expected_rms = (7.5_f32 + 1e-6).sqrt();
        let y_data = y.data();

        for i in 0..4 {
            let expected = (i + 1) as f32 / expected_rms;
            assert!(
                (y_data[i] - expected).abs() < 1e-5,
                "Element {i}: expected {expected}, got {}",
                y_data[i]
            );
        }
    }

    #[test]
    fn test_rms_norm_unit_vector_preserved() {
        // A unit vector should be nearly preserved (scaled by ~1)
        let norm = RMSNorm::without_affine(&[3]);

        // Unit vector: [1/sqrt(3), 1/sqrt(3), 1/sqrt(3)]
        let val = 1.0 / 3.0_f32.sqrt();
        let x = Tensor::new(&[val, val, val], &[1, 3]);
        let y = norm.forward(&x);

        // RMS of unit vector is 1/sqrt(3) ≈ 0.577
        // Dividing by RMS gives [1, 1, 1]
        let y_data = y.data();
        for &v in y_data {
            assert!(
                (v - 1.0).abs() < 1e-5,
                "Unit vector should normalize to 1s, got {v}"
            );
        }
    }

    #[test]
    fn test_rms_norm_vs_layer_norm_no_centering() {
        // RMSNorm doesn't center, so mean of output is NOT zero in general
        let rms_norm = RMSNorm::without_affine(&[4]);
        let layer_norm = LayerNorm::without_affine(&[4]);

        let x = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[1, 4]);

        let y_rms = rms_norm.forward(&x);
        let y_layer = layer_norm.forward(&x);

        // LayerNorm output mean should be ~0
        let layer_mean: f32 = y_layer.data().iter().sum::<f32>() / 4.0;
        assert!(layer_mean.abs() < 1e-5, "LayerNorm should have zero mean");

        // RMSNorm output mean is NOT zero (no centering)
        let rms_mean: f32 = y_rms.data().iter().sum::<f32>() / 4.0;
        assert!(
            rms_mean > 0.1,
            "RMSNorm should NOT center, mean should be > 0, got {rms_mean}"
        );

        // Both should produce different outputs
        let diff: f32 = y_rms
            .data()
            .iter()
            .zip(y_layer.data().iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(
            diff > 0.1,
            "RMSNorm and LayerNorm should produce different outputs"
        );
    }

    #[test]
    fn test_rms_norm_parameters() {
        let norm = RMSNorm::new(&[64]);
        let params = norm.parameters();

        // RMSNorm has only weight (no bias like LayerNorm)
        assert_eq!(params.len(), 1);
        assert_eq!(params[0].numel(), 64);
    }

    #[test]
    fn test_rms_norm_without_affine() {
        let norm = RMSNorm::without_affine(&[64]);
        let params = norm.parameters();

        assert!(params.is_empty());
    }

    #[test]
    fn test_rms_norm_with_custom_eps() {
        let norm = RMSNorm::with_eps(&[4], 1e-3);
        assert!((norm.eps() - 1e-3).abs() < 1e-8);
    }

    #[test]
    fn test_rms_norm_batch_processing() {
        let norm = RMSNorm::without_affine(&[4]);

        // Two samples
        let x = Tensor::new(&[1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0], &[2, 4]);
        let y = norm.forward(&x);
        let y_data = y.data();

        // First sample: all 1s -> RMS = 1 -> output = 1s
        for i in 0..4 {
            assert!((y_data[i] - 1.0).abs() < 1e-5);
        }

        // Second sample: all 2s -> RMS = 2 -> output = 1s
        for i in 4..8 {
            assert!((y_data[i] - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_rms_norm_3d_input() {
        let norm = RMSNorm::new(&[256]);
        let x = Tensor::ones(&[4, 10, 256]); // [batch, seq, features]
        let y = norm.forward(&x);

        assert_eq!(y.shape(), &[4, 10, 256]);
    }

    #[test]
    fn test_rms_norm_scaling_factor() {
        // RMSNorm scales input by 1/RMS, verify this is consistent
        let norm = RMSNorm::without_affine(&[4]);

        // If we scale input by 2, RMS doubles, output stays same
        let x1 = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[1, 4]);
        let x2 = Tensor::new(&[2.0, 4.0, 6.0, 8.0], &[1, 4]);

        let y1 = norm.forward(&x1);
        let y2 = norm.forward(&x2);

        // Outputs should be identical (RMSNorm is scale-invariant)
        for (a, b) in y1.data().iter().zip(y2.data().iter()) {
            assert!((a - b).abs() < 1e-5, "RMSNorm should be scale-invariant");
        }
    }

    #[test]
    fn test_rms_norm_with_learnable_weight() {
        let norm = RMSNorm::new(&[4]);

        let x = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[1, 4]);
        let y = norm.forward(&x);

        // Default weight is 1.0, so output should be same as without_affine
        let norm_no_affine = RMSNorm::without_affine(&[4]);
        let y_no_affine = norm_no_affine.forward(&x);

        for (a, b) in y.data().iter().zip(y_no_affine.data().iter()) {
            assert!(
                (a - b).abs() < 1e-5,
                "Default weights should produce same result as no affine"
            );
        }
    }

    #[test]
    fn test_rms_norm_numerical_stability() {
        // Test with very small values
        let norm = RMSNorm::without_affine(&[4]);

        let x = Tensor::new(&[1e-6, 1e-6, 1e-6, 1e-6], &[1, 4]);
        let y = norm.forward(&x);

        // Should not produce NaN or Inf
        for &v in y.data() {
            assert!(v.is_finite(), "Output should be finite");
        }
    }

    // ==========================================================================
    // Additional LayerNorm Tests
    // ==========================================================================

    #[test]
    fn test_layer_norm_with_eps() {
        let norm = LayerNorm::with_eps(&[64], 1e-3);
        let x = Tensor::ones(&[4, 64]);
        let y = norm.forward(&x);
        assert_eq!(y.shape(), &[4, 64]);
    }

    #[test]
    fn test_layer_norm_normalized_shape_getter() {
        let norm = LayerNorm::new(&[32, 64]);
        assert_eq!(norm.normalized_shape(), &[32, 64]);
    }

    #[test]
    fn test_layer_norm_parameters_mut() {
        let mut norm = LayerNorm::new(&[64]);
        let params = norm.parameters_mut();
        assert_eq!(params.len(), 2);
        // Can mutate parameters
        assert_eq!(params[0].numel(), 64);
    }

    #[test]
    fn test_layer_norm_without_affine_parameters_mut() {
        let mut norm = LayerNorm::without_affine(&[64]);
        let params = norm.parameters_mut();
        assert!(params.is_empty());
    }

    #[test]
    fn test_layer_norm_multi_dim_shape() {
        // Normalize over last 2 dimensions
        let norm = LayerNorm::new(&[8, 16]);
        let x = Tensor::ones(&[4, 8, 16]);
        let y = norm.forward(&x);
        assert_eq!(y.shape(), &[4, 8, 16]);
    }

    // ==========================================================================
    // Additional BatchNorm1d Tests
    // ==========================================================================

    #[test]
    fn test_batch_norm_1d_with_momentum() {
        let norm = BatchNorm1d::new(64).with_momentum(0.2);
        let x = Tensor::ones(&[32, 64]);
        let y = norm.forward(&x);
        assert_eq!(y.shape(), x.shape());
    }

    #[test]
    fn test_batch_norm_1d_with_eps() {
        let norm = BatchNorm1d::new(64).with_eps(1e-3);
        let x = Tensor::ones(&[32, 64]);
        let y = norm.forward(&x);
        assert_eq!(y.shape(), x.shape());
    }

    #[test]
    fn test_batch_norm_1d_parameters() {
        let norm = BatchNorm1d::new(32);
        let params = norm.parameters();
        assert_eq!(params.len(), 2);
        assert_eq!(params[0].numel(), 32); // weight
        assert_eq!(params[1].numel(), 32); // bias
    }

    #[test]
    fn test_batch_norm_1d_parameters_mut() {
        let mut norm = BatchNorm1d::new(32);
        let params = norm.parameters_mut();
        assert_eq!(params.len(), 2);
        assert_eq!(params[0].numel(), 32);
    }

    #[test]
    fn test_batch_norm_1d_3d_input_training() {
        // 3D input: [batch, features, length]
        let norm = BatchNorm1d::new(4);
        let x = Tensor::ones(&[2, 4, 8]);
        let y = norm.forward(&x);
        assert_eq!(y.shape(), &[2, 4, 8]);
    }

    #[test]
    fn test_batch_norm_1d_3d_input_eval() {
        // 3D input in eval mode (uses running statistics)
        let mut norm = BatchNorm1d::new(4);
        norm.eval();
        let x = Tensor::ones(&[2, 4, 8]);
        let y = norm.forward(&x);
        assert_eq!(y.shape(), &[2, 4, 8]);
    }

    #[test]
    fn test_batch_norm_1d_eval_mode() {
        // Test eval mode uses running statistics
        let mut norm = BatchNorm1d::new(4);
        norm.eval();
        assert!(!norm.training());

        let x = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 4]);
        let y = norm.forward(&x);
        assert_eq!(y.shape(), &[2, 4]);

        // Running mean defaults to 0, running var to 1
        // So output should be (x - 0) / 1 * gamma + beta = x * 1 + 0 = x
        for (i, &v) in y.data().iter().enumerate() {
            let expected = (i + 1) as f32;
            assert!((v - expected).abs() < 1e-4, "Expected {expected}, got {v}");
        }
    }

    #[test]
    fn test_batch_norm_1d_debug() {
        let norm = BatchNorm1d::new(32);
        let debug_str = format!("{:?}", norm);
        assert!(debug_str.contains("BatchNorm1d"));
    }

    // ==========================================================================
    // Additional GroupNorm Tests
    // ==========================================================================

    #[test]
    fn test_group_norm_with_eps() {
        let norm = GroupNorm::with_eps(8, 64, 1e-3);
        let x = Tensor::ones(&[4, 64]);
        let y = norm.forward(&x);
        assert_eq!(y.shape(), &[4, 64]);
    }

    #[test]
    fn test_group_norm_num_groups_getter() {
        let norm = GroupNorm::new(8, 64);
        assert_eq!(norm.num_groups(), 8);
    }

    #[test]
    fn test_group_norm_num_channels_getter() {
        let norm = GroupNorm::new(8, 64);
        assert_eq!(norm.num_channels(), 64);
    }

    #[test]
    fn test_group_norm_parameters_mut() {
        let mut norm = GroupNorm::new(8, 64);
        let params = norm.parameters_mut();
        assert_eq!(params.len(), 2);
        assert_eq!(params[0].numel(), 64);
    }

    #[test]
    fn test_group_norm_without_affine_parameters_mut() {
        let mut norm = GroupNorm::without_affine(8, 64);
        let params = norm.parameters_mut();
        assert!(params.is_empty());
    }

    #[test]
    fn test_group_norm_debug() {
        let norm = GroupNorm::new(8, 64);
        let debug_str = format!("{:?}", norm);
        assert!(debug_str.contains("GroupNorm"));
    }

    // ==========================================================================
    // Additional RMSNorm Tests
    // ==========================================================================

    #[test]
    fn test_rms_norm_normalized_shape_getter() {
        let norm = RMSNorm::new(&[128]);
        assert_eq!(norm.normalized_shape(), &[128]);
    }

    #[test]
    fn test_rms_norm_set_weight() {
        let mut norm = RMSNorm::new(&[4]);
        let new_weight = Tensor::new(&[2.0, 2.0, 2.0, 2.0], &[4]);
        norm.set_weight(new_weight);

        let x = Tensor::new(&[1.0, 1.0, 1.0, 1.0], &[1, 4]);
        let y = norm.forward(&x);

        // RMS of all 1s = 1, normalized = 1, scaled by 2 = 2
        for &v in y.data() {
            assert!((v - 2.0).abs() < 1e-5, "Expected 2.0, got {v}");
        }
    }

    #[test]
    fn test_rms_norm_weight_getter() {
        let norm = RMSNorm::new(&[64]);
        let weight = norm.weight();
        assert_eq!(weight.numel(), 64);
    }

    #[test]
    fn test_rms_norm_placeholder() {
        let norm = RMSNorm::placeholder(&[256]);
        assert_eq!(norm.normalized_shape(), &[256]);
        // Placeholder has minimal weight (1 element)
        assert_eq!(norm.weight().numel(), 1);
    }

    #[test]
    fn test_rms_norm_parameters_mut() {
        let mut norm = RMSNorm::new(&[64]);
        let params = norm.parameters_mut();
        assert_eq!(params.len(), 1);
        assert_eq!(params[0].numel(), 64);
    }

    #[test]
    fn test_rms_norm_without_affine_parameters_mut() {
        let mut norm = RMSNorm::without_affine(&[64]);
        let params = norm.parameters_mut();
        assert!(params.is_empty());
    }

    #[test]
    fn test_rms_norm_debug() {
        let norm = RMSNorm::new(&[64]);
        let debug_str = format!("{:?}", norm);
        assert!(debug_str.contains("RMSNorm"));
    }

    // ==========================================================================
    // Additional InstanceNorm Tests
    // ==========================================================================

    #[test]
    fn test_instance_norm_parameters() {
        let norm = InstanceNorm::new(32);
        let params = norm.parameters();
        assert_eq!(params.len(), 2);
        assert_eq!(params[0].numel(), 32);
    }

    #[test]
    fn test_instance_norm_parameters_mut() {
        let mut norm = InstanceNorm::new(32);
        let params = norm.parameters_mut();
        assert_eq!(params.len(), 2);
        assert_eq!(params[0].numel(), 32);
    }

    #[test]
    fn test_instance_norm_without_affine_parameters() {
        let norm = InstanceNorm::without_affine(32);
        let params = norm.parameters();
        assert!(params.is_empty());
    }

    #[test]
    fn test_instance_norm_debug() {
        let norm = InstanceNorm::new(32);
        let debug_str = format!("{:?}", norm);
        assert!(debug_str.contains("InstanceNorm"));
    }

    #[test]
    fn test_instance_norm_2d_input() {
        let norm = InstanceNorm::new(4);
        let x = Tensor::ones(&[2, 4]);
        let y = norm.forward(&x);
        assert_eq!(y.shape(), &[2, 4]);
    }

    // ==========================================================================
    // Edge Case Tests
    // ==========================================================================

    #[test]
    fn test_layer_norm_single_element() {
        let norm = LayerNorm::new(&[1]);
        let x = Tensor::new(&[5.0], &[1, 1]);
        let y = norm.forward(&x);
        // Single element: normalized = 0 (after mean subtraction), then scaled
        assert_eq!(y.shape(), &[1, 1]);
    }

    #[test]
    fn test_group_norm_single_group() {
        // num_groups = 1 means normalize all channels together
        let norm = GroupNorm::new(1, 4);
        let x = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[1, 4]);
        let y = norm.forward(&x);
        assert_eq!(y.shape(), &[1, 4]);

        // All channels in one group = LayerNorm behavior
        let y_data = y.data();
        let mean: f32 = y_data.iter().sum::<f32>() / 4.0;
        assert!(mean.abs() < 1e-4, "Single group should center to 0");
    }

    #[test]
    fn test_rms_norm_negative_values() {
        let norm = RMSNorm::without_affine(&[4]);
        let x = Tensor::new(&[-1.0, -2.0, 1.0, 2.0], &[1, 4]);
        let y = norm.forward(&x);

        // RMS ignores sign (squares values)
        let y_data = y.data();
        assert!(y_data.iter().all(|&v| v.is_finite()));
    }

    #[test]
    fn test_batch_norm_varied_values() {
        let norm = BatchNorm1d::new(4);
        // Varied input to ensure proper batch statistics computation
        let x = Tensor::new(
            &[
                1.0, 2.0, 3.0, 4.0, // batch 0
                5.0, 6.0, 7.0, 8.0, // batch 1
            ],
            &[2, 4],
        );
        let y = norm.forward(&x);
        assert_eq!(y.shape(), &[2, 4]);

        // Each feature normalized across batch
        let y_data = y.data();
        // Feature 0: [1, 5] -> mean=3, std=2 -> [-1, 1]
        assert!((y_data[0] - y_data[4]).abs() > 0.5, "Should be different");
    }
}
