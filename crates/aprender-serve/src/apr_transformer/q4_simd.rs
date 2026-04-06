//! SIMD-Accelerated Quantized APR Transformer (Q4_0)
//!
//! High-performance Q4_0 inference using SIMD-accelerated matmul primitives.
//! Extracted from apr_transformer/mod.rs (PMAT-802).

#![allow(clippy::too_many_arguments)]
#![allow(clippy::similar_names)]
use super::{AprKVCache, AprTransformerConfig};
use crate::error::{RealizarError, Result};

// SIMD-Accelerated Quantized APR Transformer (Q4_0)
// ============================================================================

/// Q4_0 quantized tensor for SIMD-accelerated inference
///
/// Stores raw Q4_0 bytes (18 bytes per 32 values) with dimensions for matmul.
#[derive(Debug, Clone)]
pub struct QuantizedAprTensorQ4 {
    /// Raw Q4_0 quantized data
    pub data: Vec<u8>,
    /// Input dimension (columns in weight matrix)
    pub in_dim: usize,
    /// Output dimension (rows in weight matrix)
    pub out_dim: usize,
}

impl QuantizedAprTensorQ4 {
    /// Create a new Q4_0 tensor from raw data
    #[must_use]
    pub fn new(data: Vec<u8>, in_dim: usize, out_dim: usize) -> Self {
        Self {
            data,
            in_dim,
            out_dim,
        }
    }

    /// Create empty tensor with proper Q4_0 allocation
    #[must_use]
    pub fn zeros(in_dim: usize, out_dim: usize) -> Self {
        const Q4_0_BLOCK_BYTES: usize = 18;
        const Q4_0_BLOCK_SIZE: usize = 32;
        let num_elements = in_dim * out_dim;
        let num_blocks = num_elements.div_ceil(Q4_0_BLOCK_SIZE);
        let data = vec![0u8; num_blocks * Q4_0_BLOCK_BYTES];
        Self {
            data,
            in_dim,
            out_dim,
        }
    }

    /// Get expected data size in bytes
    #[must_use]
    pub fn expected_bytes(num_elements: usize) -> usize {
        const Q4_0_BLOCK_BYTES: usize = 18;
        const Q4_0_BLOCK_SIZE: usize = 32;
        let num_blocks = num_elements.div_ceil(Q4_0_BLOCK_SIZE);
        num_blocks * Q4_0_BLOCK_BYTES
    }
}

/// Q4_0 quantized layer for SIMD-accelerated inference
///
/// Stores individual Q4_0 tensors for each weight matrix, enabling
/// direct use of `fused_q4_0_q8_0_parallel_matvec`.
#[derive(Debug, Clone)]
pub struct QuantizedAprLayerQ4 {
    /// Attention norm weight (F32, small)
    pub attn_norm_weight: Vec<f32>,
    /// QKV projection weights (Q4_0)
    pub qkv_weight: QuantizedAprTensorQ4,
    /// Attention output projection (Q4_0)
    pub attn_output_weight: QuantizedAprTensorQ4,
    /// FFN up projection (Q4_0)
    pub ffn_up_weight: QuantizedAprTensorQ4,
    /// FFN down projection (Q4_0)
    pub ffn_down_weight: QuantizedAprTensorQ4,
    /// FFN gate projection for SwiGLU (Q4_0, optional)
    pub ffn_gate_weight: Option<QuantizedAprTensorQ4>,
    /// FFN norm weight (F32, optional)
    pub ffn_norm_weight: Option<Vec<f32>>,
}

/// SIMD-accelerated Quantized APR Transformer
///
/// Stores weights in Q4_0 format and uses integer SIMD matmul
/// (AVX2 maddubs) for near-GGUF performance.
///
/// # Performance
///
/// Expected throughput: ~17 tok/s on TinyLlama-1.1B (1.36x vs GGUF)
/// With KV cache: ~25-34 tok/s expected (1.5-2x additional speedup)
#[derive(Debug, Clone)]
pub struct QuantizedAprTransformerQ4 {
    /// Model configuration
    pub config: AprTransformerConfig,
    /// Token embedding (F32 for fast lookup)
    pub token_embedding: Vec<f32>,
    /// Quantized layers
    pub layers: Vec<QuantizedAprLayerQ4>,
    /// Output norm weight (F32)
    pub output_norm_weight: Vec<f32>,
    /// LM head weight (Q4_0)
    pub lm_head_weight: QuantizedAprTensorQ4,
}

/// Scratch buffer for zero-allocation inference
///
/// Pre-allocates all intermediate buffers needed for a forward pass.
/// Reuse across multiple forward calls to eliminate per-token allocations.
#[derive(Debug)]
pub struct AprInferenceScratch {
    /// Hidden state [hidden_dim]
    pub hidden: Vec<f32>,
    /// Normalized hidden state [hidden_dim]
    pub normed: Vec<f32>,
    /// QKV projection output [qkv_dim]
    pub qkv_out: Vec<f32>,
    /// Query vectors [q_dim]
    pub q: Vec<f32>,
    /// Key vectors [k_dim]
    pub k: Vec<f32>,
    /// Value vectors [v_dim]
    pub v: Vec<f32>,
    /// Attention output [hidden_dim]
    pub attn_out: Vec<f32>,
    /// FFN input [hidden_dim]
    pub ffn_input: Vec<f32>,
    /// FFN up projection [intermediate_dim]
    pub ffn_up: Vec<f32>,
    /// FFN gate projection [intermediate_dim]
    pub ffn_gate: Vec<f32>,
    /// FFN output [hidden_dim]
    pub ffn_out: Vec<f32>,
}

impl AprInferenceScratch {
    /// Create scratch buffer sized for a model config
    #[must_use]
    pub fn from_config(config: &AprTransformerConfig) -> Self {
        let hidden_dim = config.hidden_dim;
        let qkv_dim = hidden_dim * 3; // Conservative estimate
        let intermediate_dim = config.intermediate_dim;

        Self {
            hidden: vec![0.0; hidden_dim],
            normed: vec![0.0; hidden_dim],
            qkv_out: vec![0.0; qkv_dim],
            q: vec![0.0; hidden_dim],
            k: vec![0.0; hidden_dim],
            v: vec![0.0; hidden_dim],
            attn_out: vec![0.0; hidden_dim],
            ffn_input: vec![0.0; hidden_dim],
            ffn_up: vec![0.0; intermediate_dim],
            ffn_gate: vec![0.0; intermediate_dim],
            ffn_out: vec![0.0; hidden_dim],
        }
    }

    /// Clear all buffers (set to zero)
    pub fn clear(&mut self) {
        self.hidden.fill(0.0);
        self.normed.fill(0.0);
        self.qkv_out.fill(0.0);
        self.q.fill(0.0);
        self.k.fill(0.0);
        self.v.fill(0.0);
        self.attn_out.fill(0.0);
        self.ffn_input.fill(0.0);
        self.ffn_up.fill(0.0);
        self.ffn_gate.fill(0.0);
        self.ffn_out.fill(0.0);
    }
}

include!("attention_kernels.rs");
include!("q4_simd_tests.rs");
