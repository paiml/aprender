#[allow(clippy::wildcard_imports)]
use super::*;

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
    /// ONE PATH: Delegates computation to `nn::functional::rms_norm` (UCBD §4).
    /// Shape validation and non-affine path handled here (Module layer).
    #[provable_contracts_macros::contract("rmsnorm-kernel-v1", equation = "rmsnorm")]
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

        if self.elementwise_affine {
            // ONE PATH: delegate to canonical functional rms_norm
            crate::nn::functional::rms_norm(input, &self.weight, self.eps)
        } else {
            // Non-affine: normalize without weight
            let batch_dims: usize = shape[..start_dim].iter().product();
            let input_data = input.data();
            let mut output_data = vec![0.0; input_data.len()];

            for b in 0..batch_dims {
                let offset = b * norm_size;
                let slice = &input_data[offset..offset + norm_size];

                let mean_sq: f32 =
                    slice.iter().map(|&x| x * x).sum::<f32>() / norm_size as f32;
                let rms_inv = 1.0 / (mean_sq + self.eps).sqrt();

                for i in 0..norm_size {
                    output_data[offset + i] = slice[i] * rms_inv;
                }
            }

            Tensor::new(&output_data, shape)
        }
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
#[path = "tests.rs"]
mod tests;
