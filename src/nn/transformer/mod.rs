//! Transformer architecture components (Vaswani et al., 2017).
//!
//! Implements the attention mechanism and transformer layers for
//! sequence-to-sequence modeling.
//!
//! # Example
//!
//! ```ignore
//! use aprender::nn::{MultiHeadAttention, TransformerEncoderLayer, Module};
//! use aprender::autograd::Tensor;
//!
//! // Create a transformer encoder layer
//! let encoder = TransformerEncoderLayer::new(512, 8, 2048);
//!
//! // Process a sequence
//! let x = Tensor::randn(&[32, 10, 512]);  // [batch, seq_len, d_model]
//! let y = encoder.forward(&x);            // [batch, seq_len, d_model]
//! ```
//!
//! # References
//!
//! - Vaswani, A., et al. (2017). Attention is all you need. `NeurIPS`.

use super::dropout::Dropout;
use super::linear::Linear;
use super::module::Module;
use super::normalization::LayerNorm;
use crate::autograd::Tensor;
use trueno::Matrix;

/// Scaled Dot-Product Attention.
///
/// ```text
/// Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V
/// ```
#[provable_contracts_macros::contract("attention-kernel-v1", equation = "attention")]
fn scaled_dot_product_attention(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    attn_mask: Option<&Tensor>,
    dropout_p: f32,
    training: bool,
) -> (Tensor, Tensor) {
    let d_k = query.shape()[query.ndim() - 1] as f32;
    let scale = 1.0 / d_k.sqrt();

    // Compute attention scores: Q @ K^T / sqrt(d_k)
    let key_t = transpose_last_two(key);
    let scores = matmul_batched(query, &key_t);
    let scores = scale_tensor(&scores, scale);

    // Apply mask (for causal attention or padding)
    let scores = match attn_mask {
        Some(mask) => add_mask(&scores, mask),
        None => scores,
    };

    // Softmax over last dimension
    let attn_weights = softmax_last_dim(&scores);

    // Apply dropout if training
    let attn_weights = if training && dropout_p > 0.0 {
        apply_dropout(&attn_weights, dropout_p)
    } else {
        attn_weights
    };

    // Weighted sum: attn_weights @ V
    let output = matmul_batched(&attn_weights, value);

    (output, attn_weights)
}

/// Multi-Head Attention (Vaswani et al., 2017).
///
/// Allows the model to jointly attend to information from different
/// representation subspaces at different positions.
///
/// # Example
///
/// ```ignore
/// let mha = MultiHeadAttention::new(512, 8);  // d_model=512, num_heads=8
/// let q = Tensor::randn(&[32, 10, 512]);
/// let k = Tensor::randn(&[32, 20, 512]);
/// let v = Tensor::randn(&[32, 20, 512]);
/// let (output, attn_weights) = mha.forward_qkv(&q, &k, &v, None);
/// ```
pub struct MultiHeadAttention {
    embed_dim: usize,
    num_heads: usize,
    head_dim: usize,
    dropout_p: f32,

    /// Query projection
    q_proj: Linear,
    /// Key projection
    k_proj: Linear,
    /// Value projection
    v_proj: Linear,
    /// Output projection
    out_proj: Linear,

    training: bool,
}

impl MultiHeadAttention {
    /// Create a new Multi-Head Attention layer.
    ///
    /// # Arguments
    ///
    /// * `embed_dim` - Total dimension of the model (must be divisible by `num_heads`)
    /// * `num_heads` - Number of attention heads
    ///
    /// # Panics
    ///
    /// Panics if `embed_dim` is not divisible by `num_heads`.
    #[must_use]
    pub fn new(embed_dim: usize, num_heads: usize) -> Self {
        assert!(
            embed_dim % num_heads == 0,
            "embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        );

        let head_dim = embed_dim / num_heads;

        Self {
            embed_dim,
            num_heads,
            head_dim,
            dropout_p: 0.0,
            q_proj: Linear::new(embed_dim, embed_dim),
            k_proj: Linear::new(embed_dim, embed_dim),
            v_proj: Linear::new(embed_dim, embed_dim),
            out_proj: Linear::new(embed_dim, embed_dim),
            training: true,
        }
    }

    /// Set dropout probability.
    #[must_use]
    pub fn with_dropout(mut self, dropout_p: f32) -> Self {
        self.dropout_p = dropout_p;
        self
    }

    /// Forward pass with separate query, key, value inputs.
    ///
    /// # Arguments
    ///
    /// * `query` - Query tensor [batch, `target_len`, `embed_dim`]
    /// * `key` - Key tensor [batch, `source_len`, `embed_dim`]
    /// * `value` - Value tensor [batch, `source_len`, `embed_dim`]
    /// * `attn_mask` - Optional attention mask [batch, `target_len`, `source_len`]
    ///
    /// # Returns
    ///
    /// Tuple of (output, `attention_weights`)
    #[must_use]
    pub fn forward_qkv(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        attn_mask: Option<&Tensor>,
    ) -> (Tensor, Tensor) {
        let batch_size = query.shape()[0];
        let tgt_len = query.shape()[1];
        let src_len = key.shape()[1];

        // Project Q, K, V
        let q = self.q_proj.forward(query);
        let k = self.k_proj.forward(key);
        let v = self.v_proj.forward(value);

        // Reshape for multi-head: [batch, seq, embed] -> [batch, heads, seq, head_dim]
        let q = reshape_for_attention(&q, batch_size, tgt_len, self.num_heads, self.head_dim);
        let k = reshape_for_attention(&k, batch_size, src_len, self.num_heads, self.head_dim);
        let v = reshape_for_attention(&v, batch_size, src_len, self.num_heads, self.head_dim);

        // Scaled dot-product attention
        let (attn_output, attn_weights) =
            scaled_dot_product_attention(&q, &k, &v, attn_mask, self.dropout_p, self.training);

        // Reshape back: [batch, heads, seq, head_dim] -> [batch, seq, embed]
        let attn_output = reshape_from_attention(&attn_output, batch_size, tgt_len, self.embed_dim);

        // Output projection
        let output = self.out_proj.forward(&attn_output);

        (output, attn_weights)
    }

    /// Self-attention: query, key, value are the same.
    #[must_use]
    pub fn forward_self(&self, x: &Tensor, attn_mask: Option<&Tensor>) -> (Tensor, Tensor) {
        self.forward_qkv(x, x, x, attn_mask)
    }

    /// Get `embed_dim`.
    #[must_use]
    pub fn embed_dim(&self) -> usize {
        self.embed_dim
    }

    /// Get `num_heads`.
    #[must_use]
    pub fn num_heads(&self) -> usize {
        self.num_heads
    }
}

impl Module for MultiHeadAttention {
    #[provable_contracts_macros::contract("gqa-kernel-v1", equation = "gqa")]
    fn forward(&self, input: &Tensor) -> Tensor {
        let (output, _) = self.forward_self(input, None);
        output
    }

    fn parameters(&self) -> Vec<&Tensor> {
        let mut params = self.q_proj.parameters();
        params.extend(self.k_proj.parameters());
        params.extend(self.v_proj.parameters());
        params.extend(self.out_proj.parameters());
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        let mut params = self.q_proj.parameters_mut();
        params.extend(self.k_proj.parameters_mut());
        params.extend(self.v_proj.parameters_mut());
        params.extend(self.out_proj.parameters_mut());
        params
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

impl std::fmt::Debug for MultiHeadAttention {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MultiHeadAttention")
            .field("embed_dim", &self.embed_dim)
            .field("num_heads", &self.num_heads)
            .field("head_dim", &self.head_dim)
            .field("dropout_p", &self.dropout_p)
            .finish_non_exhaustive()
    }
}

/// Transformer Encoder Layer.
///
/// Consists of self-attention followed by a feed-forward network,
/// with residual connections and layer normalization.
///
/// ```text
/// x = x + Dropout(SelfAttention(LayerNorm(x)))
/// x = x + Dropout(FFN(LayerNorm(x)))
/// ```
pub struct TransformerEncoderLayer {
    self_attn: MultiHeadAttention,
    linear1: Linear,
    linear2: Linear,
    norm1: LayerNorm,
    norm2: LayerNorm,
    dropout: Dropout,
    dropout1: Dropout,
    dropout2: Dropout,
    d_model: usize,
    training: bool,
}

impl TransformerEncoderLayer {
    /// Create a new Transformer Encoder Layer.
    ///
    /// # Arguments
    ///
    /// * `d_model` - Dimension of the model
    /// * `nhead` - Number of attention heads
    /// * `dim_feedforward` - Dimension of the feedforward network (typically 4 * `d_model`)
    #[must_use]
    pub fn new(d_model: usize, nhead: usize, dim_feedforward: usize) -> Self {
        Self {
            self_attn: MultiHeadAttention::new(d_model, nhead),
            linear1: Linear::new(d_model, dim_feedforward),
            linear2: Linear::new(dim_feedforward, d_model),
            norm1: LayerNorm::new(&[d_model]),
            norm2: LayerNorm::new(&[d_model]),
            dropout: Dropout::new(0.1),
            dropout1: Dropout::new(0.1),
            dropout2: Dropout::new(0.1),
            d_model,
            training: true,
        }
    }

    /// Set dropout probability.
    pub fn with_dropout(mut self, dropout: f32) -> Self {
        self.dropout = Dropout::new(dropout);
        self.dropout1 = Dropout::new(dropout);
        self.dropout2 = Dropout::new(dropout);
        self.self_attn = self.self_attn.with_dropout(dropout);
        self
    }

    /// Forward with optional attention mask.
    pub fn forward_with_mask(&self, src: &Tensor, src_mask: Option<&Tensor>) -> Tensor {
        // Pre-norm architecture (more stable)
        // Self-attention block
        let src_norm = self.norm1.forward(src);
        let (attn_out, _) = self.self_attn.forward_self(&src_norm, src_mask);
        let attn_out = self.dropout1.forward(&attn_out);
        let src = src.add(&attn_out);

        // Feed-forward block
        let src_norm = self.norm2.forward(&src);
        let ff_out = self.linear1.forward(&src_norm);
        let ff_out = gelu(&ff_out);
        let ff_out = self.dropout.forward(&ff_out);
        let ff_out = self.linear2.forward(&ff_out);
        let ff_out = self.dropout2.forward(&ff_out);

        src.add(&ff_out)
    }
}

impl Module for TransformerEncoderLayer {
    fn forward(&self, input: &Tensor) -> Tensor {
        self.forward_with_mask(input, None)
    }

    fn parameters(&self) -> Vec<&Tensor> {
        let mut params = self.self_attn.parameters();
        params.extend(self.linear1.parameters());
        params.extend(self.linear2.parameters());
        params.extend(self.norm1.parameters());
        params.extend(self.norm2.parameters());
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        let mut params = self.self_attn.parameters_mut();
        params.extend(self.linear1.parameters_mut());
        params.extend(self.linear2.parameters_mut());
        params.extend(self.norm1.parameters_mut());
        params.extend(self.norm2.parameters_mut());
        params
    }

    fn train(&mut self) {
        self.training = true;
        self.self_attn.train();
        self.dropout.train();
        self.dropout1.train();
        self.dropout2.train();
    }

    fn eval(&mut self) {
        self.training = false;
        self.self_attn.eval();
        self.dropout.eval();
        self.dropout1.eval();
        self.dropout2.eval();
    }

    fn training(&self) -> bool {
        self.training
    }
}

impl std::fmt::Debug for TransformerEncoderLayer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TransformerEncoderLayer")
            .field("d_model", &self.d_model)
            .field("self_attn", &self.self_attn)
            .finish_non_exhaustive()
    }
}

/// Transformer Decoder Layer.
///
/// Like encoder but with an additional cross-attention layer.
pub struct TransformerDecoderLayer {
    pub(crate) self_attn: MultiHeadAttention,
    pub(crate) cross_attn: MultiHeadAttention,
    pub(crate) linear1: Linear,
    pub(crate) linear2: Linear,
    pub(crate) norm1: LayerNorm,
    pub(crate) norm2: LayerNorm,
    pub(crate) norm3: LayerNorm,
    pub(crate) dropout: Dropout,
    pub(crate) dropout1: Dropout,
    pub(crate) dropout2: Dropout,
    pub(crate) dropout3: Dropout,
    pub(crate) d_model: usize,
    pub(crate) training: bool,
}

#[path = "positional_encoding.rs"]
mod positional_encoding;
pub use positional_encoding::*;
// ONE PATH: Re-export canonical attention utilities for crate-internal use (UCBD §4).
pub(crate) use positional_encoding::{matmul_batched, reshape_from_attention, transpose_last_two};

#[path = "mod_part_03.rs"]
mod mod_part_03;
pub use mod_part_03::*;

#[path = "mod_part_04.rs"]
mod mod_part_04;
pub use mod_part_04::*;

#[path = "mod_part_05.rs"]
mod mod_part_05;
