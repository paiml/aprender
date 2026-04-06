//! Attention mechanisms for transformer models
//!
//! Extracted from layers/mod.rs (PMAT-802) to reduce module size.
//! Contains:
//! - Attention: Basic scaled dot-product attention
//! - SlidingWindowAttention: Efficient attention with fixed window size
//! - FusedQKVAttention: FlashAttention-style tiled attention
//! - MultiHeadAttention: Full multi-head attention with Q/K/V projections

use crate::{
    error::{RealizarError, Result},
    tensor::Tensor,
};

use super::{softmax, Linear};

/// Scaled dot-product attention
///
/// Computes attention as:
/// ```text
/// Attention(Q, K, V) = softmax(Q @ K.T / sqrt(d_k)) @ V
/// ```
///
/// This is a building block for multi-head attention.
///
/// # References
///
/// "Attention is All You Need" - Vaswani et al., 2017
#[derive(Debug, Clone)]
pub struct Attention {
    /// Head dimension (`d_k` = `d_model` / `num_heads`)
    head_dim: usize,
    /// Scale factor: 1 / `sqrt(head_dim)`
    scale: f32,
}

include!("product.rs");
include!("sliding_window.rs");
include!("fused_qkv_attention.rs");
include!("attention_constructor.rs");
include!("attention_attention.rs");
