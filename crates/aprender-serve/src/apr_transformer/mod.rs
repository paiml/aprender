//! APR Transformer Format for WASM-compatible LLM inference
//!
//! This module provides a WASM-compatible transformer implementation that stores
//! all weights as F32, enabling fair comparison between APR and GGUF formats.
//!
//! ## Design Goals
//!
//! 1. **WASM Compatibility**: Pure F32 weights, no SIMD requirements
//! 2. **Fair Comparison**: Same inference algorithm as GGUFTransformer
//! 3. **Serialization**: APR format with model type `TransformerLM` (0x0050)
//!
//! ## Example
//!
//! ```rust,ignore
//! use realizar::apr_transformer::AprTransformer;
//! use realizar::gguf::{GGUFModel, GGUFTransformer};
//!
//! // Load GGUF model
//! let gguf_data = std::fs::read("model.gguf")?;
//! let gguf_model = GGUFModel::from_bytes(&gguf_data)?;
//! let gguf_transformer = GGUFTransformer::from_gguf(&gguf_model, &gguf_data)?;
//!
//! // Convert to APR format
//! let apr_transformer = AprTransformer::from_gguf_transformer(&gguf_transformer);
//!
//! // Run inference (should match GGUF output)
//! let logits = apr_transformer.forward(&[1, 2, 3, 4])?;
//! ```

use std::fs::File;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::error::{RealizarError, Result};

// PMAT-802: Extracted modules
mod config;
mod convert;
mod dequant;
mod generation;
mod helpers;
mod loader;
mod q4_simd;
pub use config::{
    AprKVCache, AprTransformerConfig, AprTransformerLayer, GenerateConfig, Q4KLayerWeights,
};
use dequant::{
    dequantize_apr_q4_native, dequantize_apr_q8_native, dequantize_q4_k_apr, dequantize_q6_k_apr,
    dequantize_q8_0_apr, f16_to_f32,
};
use helpers::{matmul_q4k_rowmajor, matmul_q6k_rowmajor, simd_add_weighted, simd_dot_f32};
pub use loader::{
    AprQuantizationType, MmapAprTransformer, QuantizedAprTransformer, APR_TRANSFORMER_HEADER_SIZE,
};
pub use q4_simd::{
    AprInferenceScratch, QuantizedAprLayerQ4, QuantizedAprTensorQ4, QuantizedAprTransformerQ4,
};

// APR Benchmark Infrastructure (Y6) - extracted from mod.rs (PMAT-802)
mod benchmark;
pub use benchmark::{
    AprBenchmarkResult, AprBenchmarkRunner, AprLoadResult, AprParityComparison, AprPrefillResult,
    APR_CPU_DECODE_THRESHOLD_TOK_S, APR_PARITY_THRESHOLD_PCT, APR_PREFILL_THRESHOLD_TOK_S,
};

// GH-202 FIX: Per-row dequantization for padded quantized matrices.
// quantize_q{4,6}_k_matrix pads each row to 256-element boundary.
// Flat dequant reads blocks sequentially and corrupts data when cols % 256 != 0.
// This function dequantizes per-row, skipping padding at the end of each row.
fn dequant_perrow(
    data: &[u8],
    dims: &[usize],
    block_elems: usize,
    block_bytes: usize,
    dequant_block: impl Fn(&[u8], &mut [f32]),
) -> Vec<f32> {
    let rows = dims[0];
    let cols = dims[1];
    let blocks_per_row = cols.div_ceil(block_elems);
    let bytes_per_row = blocks_per_row * block_bytes;
    let mut result = Vec::with_capacity(rows * cols);

    for row in 0..rows {
        let row_start = row * bytes_per_row;
        if row_start + bytes_per_row > data.len() {
            // Not enough data, fill remaining with zeros
            result.resize(rows * cols, 0.0);
            return result;
        }
        // Dequantize all blocks in this row
        let mut row_values = vec![0.0f32; blocks_per_row * block_elems];
        for b in 0..blocks_per_row {
            let block_start = row_start + b * block_bytes;
            let out_start = b * block_elems;
            dequant_block(
                &data[block_start..block_start + block_bytes],
                &mut row_values[out_start..out_start + block_elems],
            );
        }
        // Keep only the actual cols (discard padding)
        result.extend_from_slice(&row_values[..cols]);
    }
    result
}

// Dequantize a single Q6K super-block (210 bytes → 256 f32 values)
fn dequant_q6k_block(block: &[u8], out: &mut [f32]) {
    let ql = block.get(0..128).expect("Q6K block requires 128 ql bytes");
    let qh = block
        .get(128..192)
        .expect("Q6K block requires 64 qh bytes at offset 128");
    let mut scales = [0i8; 16];
    #[allow(clippy::cast_possible_wrap)]
    for (i, s) in scales.iter_mut().enumerate() {
        *s = block[192 + i] as i8;
    }
    let d = dequant::f16_to_f32(u16::from_le_bytes([block[208], block[209]]));

    for n in (0..256).step_by(128) {
        let idx = n / 128;
        let sc = &scales[8 * idx..];
        let ql_s = &ql[64 * idx..];
        let qh_s = &qh[32 * idx..];
        for l in 0..32 {
            let is = l / 16;
            let q1 = ((ql_s[l] & 0xF) | ((qh_s[l] & 3) << 4)) as i32 - 32;
            let q2 = ((ql_s[l + 32] & 0xF) | (((qh_s[l] >> 2) & 3) << 4)) as i32 - 32;
            let q3 = ((ql_s[l] >> 4) | (((qh_s[l] >> 4) & 3) << 4)) as i32 - 32;
            let q4 = ((ql_s[l + 32] >> 4) | (((qh_s[l] >> 6) & 3) << 4)) as i32 - 32;
            out[n + l] = d * (sc[is] as f32) * (q1 as f32);
            out[n + l + 32] = d * (sc[is + 2] as f32) * (q2 as f32);
            out[n + l + 64] = d * (sc[is + 4] as f32) * (q3 as f32);
            out[n + l + 96] = d * (sc[is + 6] as f32) * (q4 as f32);
        }
    }
}

// Dequantize a single Q4K super-block (144 bytes → 256 f32 values)
// Inlined from dequantize_q4_k_apr single-block logic.
fn dequant_q4k_block(block: &[u8], out: &mut [f32]) {
    // d (f16) at bytes 0-1, dmin (f16) at bytes 2-3
    let d = dequant::f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    let dmin = dequant::f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
    // scales: 12 bytes at offset 4
    let scales = block
        .get(4..16)
        .expect("Q4K block requires 12 scale bytes at offset 4");
    // qs: 128 bytes at offset 16
    let qs = block
        .get(16..144)
        .expect("Q4K block requires 128 qs bytes at offset 16");

    let mut ys_index = 0;
    for j in (0..256).step_by(64) {
        let q = &qs[j / 2..j / 2 + 32];
        let is = j / 32;
        let (sc1, m1) = dequant::extract_scale_min_apr(scales, is);
        let d1 = d * sc1;
        let dm1 = dmin * m1;
        let (sc2, m2) = dequant::extract_scale_min_apr(scales, is + 1);
        let d2 = d * sc2;
        let dm2 = dmin * m2;
        // 32 low nibbles
        for &byte in q {
            out[ys_index] = d1 * (byte & 0xF) as f32 - dm1;
            ys_index += 1;
        }
        // 32 high nibbles
        for &byte in q {
            out[ys_index] = d2 * (byte >> 4) as f32 - dm2;
            ys_index += 1;
        }
    }
}

/// Statistics for a vector of activations
#[derive(Debug, Clone, Default)]
pub struct ActivationStats {
    /// Minimum value
    pub min: f32,
    /// Maximum value
    pub max: f32,
    /// Mean value
    pub mean: f32,
    /// Standard deviation
    pub std_dev: f32,
    /// Number of NaN values
    pub nan_count: usize,
    /// Number of Inf values
    pub inf_count: usize,
    /// Number of zeros
    pub zero_count: usize,
    /// Total number of elements
    pub count: usize,
}

impl ActivationStats {
    /// Compute statistics from a slice of floats
    #[must_use]
    pub fn from_slice(data: &[f32]) -> Self {
        if data.is_empty() {
            return Self::default();
        }

        let count = data.len();
        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;
        let mut sum = 0.0f64;
        let mut nan_count = 0;
        let mut inf_count = 0;
        let mut zero_count = 0;

        for &v in data {
            if v.is_nan() {
                nan_count += 1;
                continue;
            }
            if v.is_infinite() {
                inf_count += 1;
                continue;
            }
            if v == 0.0 {
                zero_count += 1;
            }
            min = min.min(v);
            max = max.max(v);
            sum += v as f64;
        }

        let valid_count = count - nan_count - inf_count;
        let mean = if valid_count > 0 {
            (sum / valid_count as f64) as f32
        } else {
            0.0
        };

        // Compute std dev
        let mut var_sum = 0.0f64;
        for &v in data {
            if !v.is_nan() && !v.is_infinite() {
                let diff = v as f64 - mean as f64;
                var_sum += diff * diff;
            }
        }
        let std_dev = if valid_count > 1 {
            ((var_sum / (valid_count - 1) as f64).sqrt()) as f32
        } else {
            0.0
        };

        Self {
            min,
            max,
            mean,
            std_dev,
            nan_count,
            inf_count,
            zero_count,
            count,
        }
    }
}

/// Last-token-only activation slice for sample-size parity with GGUF traces.
///
/// Per `contracts/apr-vs-gguf-forward-parity-v1.yaml` v1.1.0 +
/// SPEC-SHIP-TWO-001 §37: APR's `forward_traced` historically emits stats
/// over ALL prompt tokens (seq_len * dim elements), while GGUF's
/// `forward_traced` emits stats over only the LAST token (1 * dim
/// elements). FALSIFY-APR-GGUF-PARITY-007 enforces count parity per layer
/// per stat slot — a precondition for ratio-gate credibility (-001/-002/-003).
///
/// Each field mirrors the corresponding `LayerActivation::*_stats` field but
/// is computed only over the last token's slice, e.g.
/// `&hidden[(seq_len - 1) * hidden_dim..]`. When `seq_len == 1`, these stats
/// are identical to the all-tokens stats (single-element optimization).
#[derive(Debug, Clone, Default)]
pub struct LastTokenStats {
    /// attn_norm output, last-token only
    pub attn_norm_stats: ActivationStats,
    /// QKV projection output, last-token only
    pub qkv_stats: ActivationStats,
    /// Attention output, last-token only
    pub attn_out_stats: ActivationStats,
    /// FFN norm output, last-token only
    pub ffn_norm_stats: ActivationStats,
    /// FFN gate matmul output, last-token only
    pub ffn_gate_stats: ActivationStats,
    /// FFN up matmul output, last-token only
    pub ffn_up_stats: ActivationStats,
    /// silu(gate), last-token only
    pub ffn_silu_gate_stats: ActivationStats,
    /// silu(gate) * up, last-token only
    pub ffn_swiglu_inner_stats: ActivationStats,
    /// FFN output (post-down-proj), last-token only
    pub ffn_out_stats: ActivationStats,
    /// Layer output (post-residual), last-token only
    pub output_stats: ActivationStats,
}

/// Per-layer activation trace
///
/// Sub-FFN telemetry fields (`ffn_gate_stats`, `ffn_up_stats`,
/// `ffn_silu_gate_stats`, `ffn_swiglu_inner_stats`) added per
/// `contracts/trace-ffn-sub-block-v1.yaml` v1.0.0 to support sub-block
/// bisection within a transformer block's FFN. Load-bearing for the
/// SHIP-007 fix per ship-two-models-spec.md §15.5 + §17.4.
///
/// `last_token` field added per `contracts/apr-vs-gguf-forward-parity-v1.yaml`
/// v1.1.0 + SPEC-SHIP-TWO-001 §37: enables apples-to-apples comparison with
/// GGUF's last-token-only stats, satisfying FALSIFY-APR-GGUF-PARITY-007
/// count-parity precondition.
#[derive(Debug, Clone)]
pub struct LayerActivation {
    /// Layer index (0-indexed)
    pub layer_idx: usize,
    /// Statistics after attention layer norm
    pub attn_norm_stats: ActivationStats,
    /// Statistics after QKV projection
    pub qkv_stats: ActivationStats,
    /// Statistics after attention output
    pub attn_out_stats: ActivationStats,
    /// Statistics after FFN layer norm
    pub ffn_norm_stats: ActivationStats,
    /// Statistics after gate projection matmul (pre-SiLU)
    /// Per `contracts/trace-ffn-sub-block-v1.yaml` OBL-SUB-FFN-001
    pub ffn_gate_stats: ActivationStats,
    /// Statistics after up projection matmul
    /// Per `contracts/trace-ffn-sub-block-v1.yaml` OBL-SUB-FFN-002
    pub ffn_up_stats: ActivationStats,
    /// Statistics after SiLU activation on gate projection (silu(gate))
    /// Per `contracts/trace-ffn-sub-block-v1.yaml` OBL-SUB-FFN-003
    pub ffn_silu_gate_stats: ActivationStats,
    /// Statistics after SwiGLU inner multiply (silu(gate) * up)
    /// Per `contracts/trace-ffn-sub-block-v1.yaml` OBL-SUB-FFN-004
    pub ffn_swiglu_inner_stats: ActivationStats,
    /// Statistics after FFN output (post-down-proj-matmul, residual contribution)
    pub ffn_out_stats: ActivationStats,
    /// Statistics after residual connection (layer output)
    pub output_stats: ActivationStats,
    /// Last-token-only stats for sample-size parity with GGUF
    /// (FALSIFY-APR-GGUF-PARITY-007). `None` when reporter does not populate
    /// (e.g., backwards-compat callers). When populated, count == hidden_dim
    /// or intermediate_dim per stat slot. Mirrors GGUF reporter semantics.
    pub last_token: Option<LastTokenStats>,
}

/// Forward pass trace with layer-by-layer activations
#[derive(Debug, Clone)]
pub struct ForwardTrace {
    /// Input token IDs
    pub input_tokens: Vec<u32>,
    /// Embedding statistics
    pub embed_stats: ActivationStats,
    /// Per-layer activations
    pub layer_activations: Vec<LayerActivation>,
    /// Final layer norm statistics
    pub final_norm_stats: ActivationStats,
    /// Output logits statistics
    pub logits_stats: ActivationStats,
    /// Final logits vector (for top-k analysis)
    pub logits: Vec<f32>,
}

/// PMAT-216: Trait for inference backends that support layer-by-layer tracing
///
/// This trait ensures all inference backends (CPU, GPU, etc.) provide consistent
/// tracing capability for diagnostics and parity testing.
///
/// # Five Whys Root Cause
///
/// The GPU path lacked tracing while CPU had it, allowing bugs to ship undetected.
/// This trait enforces that any new backend MUST implement tracing from day one.
///
/// # Example
///
/// ```rust,ignore
/// // All backends must implement this:
/// impl TracedForward for MyNewBackend {
///     fn forward_traced(&mut self, tokens: &[u32]) -> Result<ForwardTrace> {
///         // Must capture layer-by-layer statistics
///     }
/// }
/// ```
pub trait TracedForward {
    /// Run forward pass with layer-by-layer activation statistics
    ///
    /// Returns a `ForwardTrace` containing:
    /// - Embedding statistics
    /// - Per-layer activation statistics (attn_norm, qkv, attn_out, ffn_norm, ffn_out, output)
    /// - Final norm statistics
    /// - Logits statistics and values
    ///
    /// # Errors
    ///
    /// Returns error if inference fails
    fn forward_traced(&mut self, tokens: &[u32]) -> Result<ForwardTrace>;
}

/// APR Transformer model with all weights
///
/// For Q4K models, raw Q4K bytes can be stored in `q4k_layers` to enable
/// fused kernel inference (F-GPU-130) without full dequantization overhead.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AprTransformer {
    /// Model configuration
    pub config: AprTransformerConfig,
    /// Token embedding weights [vocab_size * hidden_dim]
    pub token_embedding: Vec<f32>,
    /// Transformer layers
    pub layers: Vec<AprTransformerLayer>,
    /// Output norm weight [hidden_dim]
    pub output_norm_weight: Vec<f32>,
    /// Output norm bias (optional) [hidden_dim]
    pub output_norm_bias: Option<Vec<f32>>,
    /// LM head weight [hidden_dim * vocab_size]
    ///
    /// PMAT-788: For tied-embedding models (`tie_word_embeddings = true`), this
    /// field is left EMPTY and `lm_head_tied` is set; the f32 logits matmul
    /// sources the (byte-identical) `token_embedding` buffer via
    /// [`AprTransformer::lm_head_f32`]. This eliminates a full-size (≈519 MiB on
    /// a 0.5B model) duplicate of the embedding matrix in serving RSS. When NOT
    /// tied this holds the model's own separate `lm_head` weights unchanged.
    pub lm_head_weight: Vec<f32>,
    /// LM head bias (optional) [vocab_size]
    pub lm_head_bias: Option<Vec<f32>>,
    /// PMAT-788: when `true`, the f32 `lm_head` weight is tied to (byte-identical
    /// to) `token_embedding` and `lm_head_weight` is left empty to avoid storing
    /// a redundant 519 MiB copy. Logits computation sources `token_embedding`
    /// instead via [`AprTransformer::lm_head_f32`]. Defaults to `false` for
    /// backward-compatible deserialization of models that always materialized a
    /// separate `lm_head_weight`.
    #[serde(default)]
    pub lm_head_tied: bool,
    /// Q4K raw layer weights for fused kernel inference (F-GPU-130)
    /// When present, enables direct Q4K matmul without dequantization
    #[serde(default)]
    pub q4k_layers: Option<Vec<Q4KLayerWeights>>,
    /// LM head weight in Q6K format for fused kernel inference
    /// When present, enables direct Q6K matmul without dequantization
    #[serde(default)]
    pub lm_head_weight_q6k: Option<Vec<u8>>,
    /// LM head weight in Q4K format for fused kernel inference
    #[serde(default)]
    pub lm_head_weight_q4k: Option<Vec<u8>>,
}

include!("generation_delegates.rs");
include!("traced_forward.rs");

mod traced_save_tensor;
