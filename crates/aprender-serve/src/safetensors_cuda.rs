//! SafeTensors CUDA Inference (PMAT-116)
//!
//! Direct GPU loading for HuggingFace SafeTensors models without intermediate
//! format conversion. Achieves GGUF GPU parity (200+ tok/s).
//!
//! ## Architecture
//!
//! ```text
//! SafeTensors file
//!     ↓ (mmap)
//! TensorView<'data>
//!     ↓ (F16/BF16 → F32 conversion)
//! &[f32] slice
//!     ↓ (executor.load_weights)
//! GPU memory (CudaSlice<f32>)
//!     ↓ (forward_single_cuda)
//! Logits → Token
//! ```
//!
//! ## Usage
//!
//! ```rust,ignore
//! use realizar::safetensors_cuda::SafeTensorsCudaModel;
//!
//! let mut model = SafeTensorsCudaModel::load("model.safetensors", 0)?;
//! let tokens = model.generate(&[1, 2, 3], 32, 151645)?;
//! ```

use crate::cuda::CudaExecutor;
use crate::error::{RealizarError, Result};
use crate::safetensors::{MappedSafeTensorsModel, SafetensorsConfig};
use std::path::Path;

/// PMAT-120 FIX: Weight transposition for GEMM.
///
/// GEMM kernel computes C[m,n] = A[m,k] × B[k,n] with ROW-MAJOR storage:
/// - A[i,j] at offset `i * k + j`
/// - B[i,j] at offset `i * n + j`
/// - C[i,j] at offset `i * n + j`
///
/// HuggingFace stores Linear weights as [out_features, in_features] = [n, k].
/// GEMM needs B as [k, n]. Therefore: TRANSPOSE IS REQUIRED.
impl SafeTensorsCudaModel {
    /// Transpose weight from HuggingFace [n, k] to GEMM-required [k, n].
    ///
    /// HuggingFace: W[i, j] at offset `i * k + j` where i=0..n, j=0..k
    /// GEMM needs:  B[j, i] at offset `j * n + i` where j=0..k, i=0..n
    fn transpose_for_gemm(weight: &[f32], n: usize, k: usize) -> Vec<f32> {
        let expected_len = n * k;
        // Guard against index out of bounds (PMAT-805 fix)
        if weight.len() < expected_len {
            // Return zero-padded transposed array if weight is undersized
            // This handles edge cases with tied embeddings or partial weights
            let mut transposed = vec![0.0f32; expected_len];
            for i in 0..n {
                for j in 0..k {
                    let src_idx = i * k + j;
                    if src_idx < weight.len() {
                        let dst_idx = j * n + i;
                        transposed[dst_idx] = weight[src_idx];
                    }
                }
            }
            return transposed;
        }

        let mut transposed = vec![0.0f32; expected_len];
        for i in 0..n {
            for j in 0..k {
                // HuggingFace element at row i, col j
                let src_idx = i * k + j;
                // GEMM needs element at row j, col i
                let dst_idx = j * n + i;
                transposed[dst_idx] = weight[src_idx];
            }
        }
        transposed
    }

    /// Concatenate Q, K, V weights and transpose for GEMM.
    ///
    /// HuggingFace stores separately:
    /// - Q: [hidden_dim, hidden_dim] (n=hidden, k=hidden)
    /// - K: [kv_dim, hidden_dim] (n=kv_dim, k=hidden)
    /// - V: [kv_dim, hidden_dim] (n=kv_dim, k=hidden)
    ///
    /// GEMM needs combined QKV as [hidden_dim, hidden_dim + kv_dim + kv_dim].
    fn concat_qkv_transposed(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        hidden_dim: usize,
        kv_dim: usize,
    ) -> Vec<f32> {
        // Transpose each weight matrix
        let q_t = Self::transpose_for_gemm(q, hidden_dim, hidden_dim);
        let k_t = Self::transpose_for_gemm(k, kv_dim, hidden_dim);
        let v_t = Self::transpose_for_gemm(v, kv_dim, hidden_dim);

        // After transpose:
        // q_t: [hidden_dim, hidden_dim] row-major
        // k_t: [hidden_dim, kv_dim] row-major
        // v_t: [hidden_dim, kv_dim] row-major

        // Concatenate along columns (output dimension):
        // Result: [hidden_dim, hidden_dim + kv_dim + kv_dim]
        let total_out = hidden_dim + kv_dim + kv_dim;
        let mut qkv = vec![0.0f32; hidden_dim * total_out];

        for row in 0..hidden_dim {
            let dst_start = row * total_out;

            // Copy Q row (hidden_dim elements)
            let q_src = row * hidden_dim;
            qkv[dst_start..dst_start + hidden_dim].copy_from_slice(&q_t[q_src..q_src + hidden_dim]);

            // Copy K row (kv_dim elements)
            let k_src = row * kv_dim;
            qkv[dst_start + hidden_dim..dst_start + hidden_dim + kv_dim]
                .copy_from_slice(&k_t[k_src..k_src + kv_dim]);

            // Copy V row (kv_dim elements)
            let v_src = row * kv_dim;
            qkv[dst_start + hidden_dim + kv_dim..dst_start + hidden_dim + 2 * kv_dim]
                .copy_from_slice(&v_t[v_src..v_src + kv_dim]);
        }

        qkv
    }
}

/// CUDA-accelerated SafeTensors model (PMAT-116)
///
/// Loads HuggingFace SafeTensors directly to GPU memory for high-performance
/// inference. Mirrors `AprV2ModelCuda` API for consistency.
///
/// ## GH-201: Streaming Mode
///
/// Supports two modes based on available VRAM:
/// - **Full Cache**: Pre-cache all weights (default when VRAM sufficient)
/// - **Layer Streaming**: Stream layer weights on-demand (when VRAM limited)
#[cfg(feature = "cuda")]
pub struct SafeTensorsCudaModel {
    /// CUDA executor with cached weights
    executor: CudaExecutor,
    /// Model configuration
    config: SafeTensorsCudaConfig,
    /// GPU device name
    device_name: String,
    /// GPU memory (free, total) in bytes
    memory_info: (usize, usize),
    /// Current KV cache position
    kv_position: u32,
    /// Cached embedding table (F32) - kept on CPU for token lookup
    embedding_cache: Vec<f32>,
    /// RMS norm epsilon
    epsilon: f32,
    /// RMS norm gamma weights (CPU copy for hybrid GPU/CPU path)
    /// Key format: "attn.{layer_idx}" or "ffn.{layer_idx}" or "output"
    gamma_cache: std::collections::HashMap<String, Vec<f32>>,
    /// PMAT-120 FIX: QKV bias cache (Qwen2 has attention bias terms)
    /// Key format: "qkv_bias.{layer_idx}" - concatenated Q+K+V biases
    qkv_bias_cache: std::collections::HashMap<String, Vec<f32>>,
    /// PMAT-120 FIX: Output projection bias cache
    /// Key format: "o_bias.{layer_idx}"
    o_bias_cache: std::collections::HashMap<String, Vec<f32>>,
    /// GH-279: QK norm weight cache (Qwen3 per-head RMSNorm)
    /// Key format: "q_norm.{layer_idx}" or "k_norm.{layer_idx}"
    qk_norm_cache: std::collections::HashMap<String, Vec<f32>>,
    /// GH-201: Streaming mode (true = layer-by-layer, false = full cache)
    streaming_mode: bool,
    /// GH-201: Path to SafeTensors file (kept for streaming mode weight loading)
    model_path: Option<std::path::PathBuf>,
}

/// Configuration extracted from config.json
#[cfg(feature = "cuda")]
#[derive(Debug, Clone)]
pub struct SafeTensorsCudaConfig {
    /// Model architecture (e.g., "Qwen2")
    pub architecture: String,
    /// Hidden dimension
    pub hidden_dim: usize,
    /// Number of transformer layers
    pub num_layers: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Number of KV heads (for GQA)
    pub num_kv_heads: usize,
    /// Vocabulary size
    pub vocab_size: usize,
    /// FFN intermediate dimension
    pub intermediate_dim: usize,
    /// Maximum context length
    pub context_length: usize,
    /// RoPE theta
    pub rope_theta: f32,
    /// RMS norm epsilon
    pub eps: f32,
    /// F-GT-002: Whether to use tied embeddings (lm_head = embed_tokens)
    pub tie_word_embeddings: bool,
    /// GH-279: Whether Q/K projections have per-head RMSNorm (Qwen3)
    pub has_qk_norm: bool,
    /// GH-279: Whether attention projections have bias terms (Qwen2, phi)
    pub has_bias: bool,
    /// GH-330: EOS token ID from config.json (Design by Contract)
    pub eos_token_id: Option<u32>,
}

include!("apply.rs");
include!("safetensors_cuda_config_extraction.rs");
