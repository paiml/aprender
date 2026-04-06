//! Neural network layers for transformer models
//!
//! Implements core building blocks for transformer architectures:
//! - Layer normalization
//! - Multi-head attention (MHA, MQA, GQA)
//! - Feed-forward networks
//! - Position embeddings: `RoPE` and `ALiBi`
//!
//! ## Example
//!
//! ```rust,ignore
//! use realizar::layers::LayerNorm;
//!
//! let layer_norm = LayerNorm::new(512, 1e-5)?;
//! let normalized = layer_norm.forward(&input)?;
//! ```
//!
//! ## Loading Models from Files
//!
//! Realizar supports loading models from GGUF and Safetensors formats.
//!
//! ### GGUF Example
//!
//! ```rust,ignore
//! use realizar::{gguf::GGUFModel, layers::{Model, ModelConfig}};
//!
//! // Load GGUF file
//! let file_data = std::fs::read("model.gguf")?;
//! let gguf = GGUFModel::from_bytes(&file_data)?;
//!
//! // Extract model config from metadata
//! let config = ModelConfig {
//!     vocab_size: 32000,
//!     hidden_dim: 4096,
//!     num_heads: 32,
//!     num_layers: 32,
//!     intermediate_dim: 11008,
//!     eps: 1e-5,
//! };
//!
//! // Create model and load weights
//! let mut model = Model::new(config)?;
//!
//! // Extract tensors by name (requires knowledge of naming convention)
//! let embedding_weights = gguf.get_tensor_f32("token_embd.weight", &file_data)?;
//! // ... load weights into model layers ...
//! ```
//!
//! ### Safetensors Example
//!
//! ```rust,ignore
//! use realizar::{safetensors::SafetensorsModel, layers::{Model, ModelConfig}};
//!
//! // Load Safetensors file
//! let file_data = std::fs::read("model.safetensors")?;
//! let safetensors = SafetensorsModel::from_bytes(&file_data)?;
//!
//! // Extract tensors
//! let embedding_weights = safetensors.get_tensor_f32("model.embed_tokens.weight")?;
//! // ... load weights into model layers ...
//! ```
//!
//! ### Tensor Naming Conventions
//!
//! Different model families use different tensor naming conventions:
//!
//! - **`LLaMA` models (GGUF):**
//!   - `token_embd.weight` - Token embeddings
//!   - `blk.{layer}.attn_q.weight` - Query projection for layer N
//!   - `blk.{layer}.attn_k.weight` - Key projection
//!   - `blk.{layer}.attn_v.weight` - Value projection
//!   - `blk.{layer}.ffn_up.weight` - FFN up projection
//!
//! - **`HuggingFace` models (Safetensors):**
//!   - `model.embed_tokens.weight` - Token embeddings
//!   - `model.layers.{layer}.self_attn.q_proj.weight` - Query projection
//!   - `model.layers.{layer}.self_attn.k_proj.weight` - Key projection
//!   - `model.layers.{layer}.mlp.up_proj.weight` - FFN up projection
//!
//! Consult model documentation for specific naming conventions.

use crate::{
    error::{RealizarError, Result},
    tensor::Tensor,
};

// PMAT-802: Extracted modules
mod position;
pub use position::{ALiBi, RoPE, RopeScalingType, ScaledRoPE};
mod model;
pub use model::{Embedding, KVCache, Model, ModelConfig, TransformerBlock};
mod attention;
pub use attention::{Attention, FusedQKVAttention, MultiHeadAttention, SlidingWindowAttention};

/// Apply softmax activation function
///
/// Softmax: `y[i] = exp(x[i]) / sum(exp(x[j]))` for all j
///
/// Applies softmax normalization along the last dimension. Uses numerically stable
/// implementation with max subtraction to prevent overflow.
///
/// Used in attention mechanisms for probability distributions.
///
/// # Arguments
///
/// * `input` - Input tensor
///
/// # Returns
///
/// Tensor with softmax applied along last dimension (values sum to 1.0)
///
/// # Errors
///
/// Returns error if input is empty
///
/// # Examples
///
/// ```rust,ignore
/// let input = Tensor::from_vec(vec![3], vec![1.0, 2.0, 3.0])?;
/// let output = softmax(&input)?;
/// // output sums to 1.0
/// ```
pub fn softmax(input: &Tensor<f32>) -> Result<Tensor<f32>> {
    contract_pre_softmax!(input.data());
    let data = input.data();
    let shape = input.shape();

    if data.is_empty() {
        return Err(RealizarError::InvalidShape {
            reason: "Cannot apply softmax to empty tensor".to_string(),
        });
    }

    if shape.is_empty() {
        return Err(RealizarError::InvalidShape {
            reason: "Cannot apply softmax to tensor with empty shape".to_string(),
        });
    }

    let last_dim = shape[shape.len() - 1];
    let num_groups = data.len() / last_dim;
    let mut output = Vec::with_capacity(data.len());

    // Apply softmax to each group (row) independently
    for group_idx in 0..num_groups {
        let start = group_idx * last_dim;
        let end = start + last_dim;
        let group = &data[start..end];

        // Find max for numerical stability
        let max_val = group.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        // Compute exp(x - max) for each element
        let exp_vals: Vec<f32> = group.iter().map(|&x| (x - max_val).exp()).collect();

        // Sum of exponentials
        let sum_exp: f32 = exp_vals.iter().sum();

        // Normalize to get probabilities — check postcondition per row (contract is 1D)
        let row_start = output.len();
        for &exp_val in &exp_vals {
            output.push(exp_val / sum_exp);
        }
        contract_post_softmax!(&output[row_start..]);
    }
    Tensor::from_vec(shape.to_vec(), output)
}

/// Apply GELU activation function
///
/// GELU (Gaussian Error Linear Unit): `y = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))`
///
/// Used in transformer models like BERT, GPT-2, GPT-3.
///
/// # Arguments
///
/// * `input` - Input tensor
///
/// # Returns
///
/// Tensor with GELU applied element-wise
///
/// # Errors
///
/// Returns error if input is empty
///
/// # Examples
///
/// ```rust,ignore
/// let input = Tensor::from_vec(vec![3], vec![-1.0, 0.0, 1.0])?;
/// let output = gelu(&input)?;
/// ```
pub fn gelu(input: &Tensor<f32>) -> Result<Tensor<f32>> {
    contract_pre_gelu!(input.data());
    let data = input.data();
    if data.is_empty() {
        return Err(RealizarError::InvalidShape {
            reason: "Cannot apply GELU to empty tensor".to_string(),
        });
    }

    // ONE PATH: Per-element delegates to trueno::gelu_scalar (UCBD §4).
    let output: Vec<f32> = data.iter().map(|&x| trueno::gelu_scalar(x)).collect();

    contract_post_gelu!(&output);
    Tensor::from_vec(input.shape().to_vec(), output)
}

/// Layer normalization
///
/// Normalizes activations across the feature dimension using:
/// ```text
/// y = (x - mean(x)) / sqrt(variance(x) + eps) * gamma + beta
/// ```
///
/// # References
///
/// Layer Normalization: <https://arxiv.org/abs/1607.06450>
#[derive(Debug, Clone)]
pub struct LayerNorm {
    /// Normalized shape (feature dimension)
    normalized_shape: usize,
    /// Epsilon for numerical stability
    eps: f32,
    /// Scale parameter (gamma)
    weight: Vec<f32>,
    /// Shift parameter (beta)
    bias: Vec<f32>,
}

impl LayerNorm {
    /// Create a new layer normalization layer
    ///
    /// # Arguments
    ///
    /// * `normalized_shape` - Size of the feature dimension to normalize
    /// * `eps` - Small constant for numerical stability (default: `1e-5`)
    ///
    /// # Errors
    ///
    /// Returns error if `normalized_shape` is zero
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let layer_norm = LayerNorm::new(512, 1e-5)?;
    /// ```
    pub fn new(normalized_shape: usize, eps: f32) -> Result<Self> {
        if normalized_shape == 0 {
            return Err(RealizarError::InvalidShape {
                reason: "normalized_shape must be > 0".to_string(),
            });
        }

        // Initialize weight (gamma) to 1.0
        let weight = vec![1.0; normalized_shape];
        // Initialize bias (beta) to 0.0
        let bias = vec![0.0; normalized_shape];

        Ok(Self {
            normalized_shape,
            eps,
            weight,
            bias,
        })
    }

    /// Forward pass through layer normalization
    ///
    /// # Arguments
    ///
    /// * `input` - Input tensor with shape `[..., normalized_shape]`
    ///
    /// # Returns
    ///
    /// Normalized tensor with same shape as input
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Input is empty
    /// - Last dimension doesn't match `normalized_shape`
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let input = Tensor::from_vec(vec![2, 512], data)?;
    /// let output = layer_norm.forward(&input)?;
    /// assert_eq!(output.shape(), &[2, 512]);
    /// ```
    pub fn forward(&self, input: &Tensor<f32>) -> Result<Tensor<f32>> {
        let shape = input.shape();
        if shape.is_empty() {
            return Err(RealizarError::InvalidShape {
                reason: "Input tensor cannot be empty".to_string(),
            });
        }

        let last_dim = shape[shape.len() - 1];
        if last_dim != self.normalized_shape {
            return Err(RealizarError::InvalidShape {
                reason: format!(
                    "Last dimension {} doesn't match normalized_shape {}",
                    last_dim, self.normalized_shape
                ),
            });
        }

        let data = input.data();
        let total_size = data.len();
        let num_groups = total_size / self.normalized_shape;

        let mut output = Vec::with_capacity(total_size);

        for group_idx in 0..num_groups {
            let start = group_idx * self.normalized_shape;
            let end = start + self.normalized_shape;
            let group = &data[start..end];

            // Compute mean
            #[allow(clippy::cast_precision_loss)]
            let mean: f32 = group.iter().sum::<f32>() / self.normalized_shape as f32;

            // Compute variance
            #[allow(clippy::cast_precision_loss)]
            let variance: f32 = group
                .iter()
                .map(|&x| {
                    let diff = x - mean;
                    diff * diff
                })
                .sum::<f32>()
                / self.normalized_shape as f32;

            // Normalize and apply affine transformation
            for (i, &x) in group.iter().enumerate() {
                let normalized = (x - mean) / (variance + self.eps).sqrt();
                let transformed = normalized * self.weight[i] + self.bias[i];
                output.push(transformed);
            }
        }

        // Debug assertion for numerical stability
        debug_assert!(
            output.iter().all(|&x| x.is_finite()),
            "LayerNorm produced NaN or Inf values - check input distribution"
        );

        Tensor::from_vec(shape.to_vec(), output)
    }

    /// Get the normalized shape
    #[must_use]
    pub fn normalized_shape(&self) -> usize {
        self.normalized_shape
    }

    /// Get epsilon value
    #[must_use]
    pub fn eps(&self) -> f32 {
        self.eps
    }
}

/// Linear transformation layer
///
/// Applies linear transformation: `y = x * W + b`
/// where W is weight matrix and b is bias vector.
///
/// # References
///
/// Standard fully-connected layer used in neural networks.
#[derive(Debug, Clone)]
pub struct Linear {
    /// Input features
    in_features: usize,
    /// Output features
    out_features: usize,
    /// Weight matrix `[in_features, out_features]`
    weight: Vec<f32>,
    /// Bias vector `[out_features]`
    bias: Vec<f32>,
}

include!("linear.rs");
include!("fused_layer_norm_linear.rs");
