//! Position embeddings for transformer models
//!
//! Extracted from layers/mod.rs (PMAT-802) to reduce module size.
//! Contains:
//! - RoPE: Rotary Position Embeddings (RoFormer, LLaMA, PaLM)
//! - RopeScalingType: Context length extension methods (NTK, YaRN, Linear)
//! - ScaledRoPE: RoPE with scaling for extended context
//! - ALiBi: Attention with Linear Biases

use crate::{
    error::{RealizarError, Result},
    tensor::Tensor,
};

/// Rotary Position Embeddings (`RoPE`)
///
/// Applies position-dependent rotations to query and key vectors.
/// Used in `LLaMA`, `PaLM`, and other modern transformers for relative
/// position encoding.
///
/// The rotation is applied pairwise to dimensions, encoding position
/// information directly into the embeddings.
///
/// # Formula
///
/// For each pair of dimensions (2i, 2i+1):
/// ```text
/// x'_{2i} = x_{2i} * cos(θ_i * pos) - x_{2i+1} * sin(θ_i * pos)
/// x'_{2i+1} = x_{2i} * sin(θ_i * pos) + x_{2i+1} * cos(θ_i * pos)
/// ```
///
/// Where `θ_i` = base^(-2i/dim)
///
/// # References
///
/// `RoFormer`: Enhanced Transformer with Rotary Position Embedding - Su et al., 2021
#[derive(Debug, Clone)]
pub struct RoPE {
    /// Embedding dimension (must be even)
    dim: usize,
    /// Base for computing frequencies (default: 10000)
    base: f32,
    /// Precomputed inverse frequencies for each dimension pair
    inv_freq: Vec<f32>,
}

impl RoPE {
    /// Create a new `RoPE` layer
    ///
    /// # Arguments
    ///
    /// * `dim` - Embedding dimension (must be even)
    /// * `base` - Base for computing frequencies (typically 10000)
    ///
    /// # Errors
    ///
    /// Returns error if `dim` is zero or odd
    pub fn new(dim: usize, base: f32) -> Result<Self> {
        if dim == 0 {
            return Err(RealizarError::InvalidShape {
                reason: "dim must be > 0".to_string(),
            });
        }
        if !dim.is_multiple_of(2) {
            return Err(RealizarError::InvalidShape {
                reason: "dim must be even for RoPE".to_string(),
            });
        }

        contract_pre_ntk_scaled_base!();
        contract_pre_base_frequency!();
        // Compute inverse frequencies: base^(-2i/dim) for i in 0..dim/2
        let half_dim = dim / 2;
        let mut inv_freq = Vec::with_capacity(half_dim);

        #[allow(clippy::cast_precision_loss)]
        for i in 0..half_dim {
            let exponent = -2.0 * (i as f32) / (dim as f32);
            inv_freq.push(base.powf(exponent));
        }

        contract_post_base_frequency!(&inv_freq);
        contract_post_ntk_scaled_base!(&());
        Ok(Self {
            dim,
            base,
            inv_freq,
        })
    }

    /// Create `RoPE` with default base (10000)
    ///
    /// # Errors
    ///
    /// Returns error if `dim` is zero or odd
    pub fn with_default_base(dim: usize) -> Result<Self> {
        Self::new(dim, 10000.0)
    }

    /// Apply rotary embeddings to input at given position
    ///
    /// # Arguments
    ///
    /// * `input` - Input tensor with last dimension equal to `dim`
    /// * `position` - Position index for computing rotation angles
    ///
    /// # Returns
    ///
    /// Tensor with same shape as input, with rotary embeddings applied
    ///
    /// # Errors
    ///
    /// Returns error if input's last dimension doesn't match `dim`
    pub fn forward(&self, input: &Tensor<f32>, position: usize) -> Result<Tensor<f32>> {
        let shape = input.shape();

        if shape.is_empty() {
            return Err(RealizarError::InvalidShape {
                reason: "Input tensor must have at least 1 dimension".to_string(),
            });
        }

        let last_dim = shape[shape.len() - 1];
        if last_dim != self.dim {
            return Err(RealizarError::InvalidShape {
                reason: format!("Expected last dimension {}, got {}", self.dim, last_dim),
            });
        }

        let data = input.data();
        let num_vectors = data.len() / self.dim;
        let mut output = Vec::with_capacity(data.len());

        // Compute sin/cos for this position
        let half_dim = self.dim / 2;
        let mut cos_vals = Vec::with_capacity(half_dim);
        let mut sin_vals = Vec::with_capacity(half_dim);

        #[allow(clippy::cast_precision_loss)]
        for inv_f in &self.inv_freq {
            let angle = inv_f * (position as f32);
            cos_vals.push(angle.cos());
            sin_vals.push(angle.sin());
        }

        // Apply rotation to each vector
        for vec_idx in 0..num_vectors {
            let offset = vec_idx * self.dim;

            for i in 0..half_dim {
                let x0 = data[offset + 2 * i];
                let x1 = data[offset + 2 * i + 1];
                let cos_val = cos_vals[i];
                let sin_val = sin_vals[i];

                // Apply 2D rotation
                let y0 = x0 * cos_val - x1 * sin_val;
                let y1 = x0 * sin_val + x1 * cos_val;

                output.push(y0);
                output.push(y1);
            }
        }

        Tensor::from_vec(shape.to_vec(), output)
    }

    /// Get embedding dimension
    #[must_use]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Get base frequency
    #[must_use]
    pub fn base(&self) -> f32 {
        self.base
    }

    /// Get inverse frequencies
    #[must_use]
    pub fn inv_freq(&self) -> &[f32] {
        &self.inv_freq
    }
}

// ============================================================================
// RoPE Scaling Methods (NTK, YaRN, Linear, Dynamic NTK)
// ============================================================================
//
// These methods extend RoPE to handle longer context lengths than trained.
// References:
// - NTK-aware: https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/
// - YaRN: https://arxiv.org/abs/2309.00071
// - Code Llama linear scaling: https://arxiv.org/abs/2308.12950
// ============================================================================

/// RoPE scaling type for context length extension
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum RopeScalingType {
    /// No scaling (original RoPE)
    #[default]
    None,
    /// Linear interpolation (Code Llama style)
    /// scale = trained_length / target_length
    Linear {
        /// Scale factor (typically trained_length / target_length)
        scale: f32,
    },
    /// NTK-aware scaling
    /// Modifies base frequency: base' = base * scale^(dim / (dim - 2))
    Ntk {
        /// Scale factor for context extension
        scale: f32,
    },
    /// Dynamic NTK-aware scaling
    /// Adjusts scale dynamically based on current sequence length
    DynamicNtk {
        /// Original training context length
        original_max_len: usize,
        /// Target extended context length
        target_max_len: usize,
    },
    /// YaRN (Yet another RoPE extensioN)
    /// Combines NTK interpolation with attention scaling
    Yarn {
        /// Original training context length
        original_max_len: usize,
        /// Target extended context length
        target_max_len: usize,
        /// Attention scaling factor (typically sqrt(scale))
        attn_factor: f32,
        /// Beta for interpolation ramp (default: 32)
        beta_fast: f32,
        /// Beta for extrapolation (default: 1)
        beta_slow: f32,
    },
}

/// Scaled Rotary Position Embeddings
///
/// Extends `RoPE` with various scaling methods for context length extension.
/// Supports NTK-aware, Linear, Dynamic NTK, and YaRN scaling.
///
/// # Scaling Methods
///
/// ## Linear Scaling (Code Llama)
/// Simply scales down the position: pos' = pos / scale
///
/// ## NTK-aware Scaling
/// Modifies the base frequency to reduce high-frequency component decay:
/// base' = base * scale^(dim / (dim - 2))
///
/// ## Dynamic NTK
/// Dynamically adjusts NTK scale based on current sequence length
///
/// ## YaRN (Yet another RoPE extensioN)
/// Combines NTK with attention factor and interpolation ramp
///
/// # References
///
/// - "Code Llama: Open Foundation Models for Code" - Rozière et al., 2023
/// - "YaRN: Efficient Context Window Extension of Large Language Models" - Peng et al., 2023
#[derive(Debug, Clone)]
pub struct ScaledRoPE {
    /// Base RoPE parameters
    dim: usize,
    /// Original base frequency
    original_base: f32,
    /// Scaled base frequency (after NTK adjustment)
    scaled_base: f32,
    /// Scaling configuration
    scaling: RopeScalingType,
    /// Precomputed inverse frequencies (with scaling applied)
    inv_freq: Vec<f32>,
    /// Attention scaling factor (for YaRN)
    mscale: f32,
}

include!("scaled_rope.rs");
include!("alibi.rs");
include!("position_alibi_get.rs");
