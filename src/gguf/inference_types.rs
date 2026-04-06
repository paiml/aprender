//! Inference support types for quantized model execution
//!
//! This module contains pre-allocated buffers and caches for zero-allocation inference.
//!
//! Types included:
//! - `InferenceScratchBuffer`: Scratch buffers for forward passes
//! - `OwnedInferenceScratchBuffer`: Owned variant with Q8K support
//! - `ContiguousKVCache`: Cache-aligned KV cache for efficient attention
//! - `DispatchMetrics`: Thread-safe CPU/GPU dispatch metrics

use super::config::GGUFConfig;

/// Cache line size in bytes (typical x86-64)
const CACHE_LINE_BYTES: usize = 64;

/// Number of f32 elements per cache line (64 bytes / 4 bytes per f32)
const FLOATS_PER_CACHE_LINE: usize = CACHE_LINE_BYTES / std::mem::size_of::<f32>();

// ============================================================================
// InferenceScratchBuffer
// ============================================================================

/// Pre-allocated scratch buffers for zero-allocation forward passes
///
/// ## Buffer Reuse Pattern
///
/// - First use: hidden → normed → qkv → q/k/v → attn_out
/// - FFN pass: normed → ffn_up/ffn_gate → ffn_down → hidden
///
/// PAR-126: Added Q8K scratch buffers for VNNI-accelerated Q4K×Q8K matmul path.
#[derive(Debug)]
pub struct InferenceScratchBuffer {
    /// Hidden state buffer [hidden_dim]
    pub hidden: Vec<f32>,
    /// Normalized hidden state [hidden_dim]
    pub normed: Vec<f32>,
    /// Combined QKV projection [q_dim + k_dim + v_dim]
    pub qkv: Vec<f32>,
    /// Query projection [q_dim]
    pub q: Vec<f32>,
    /// Key projection [k_dim]
    pub k: Vec<f32>,
    /// Value projection [v_dim]
    pub v: Vec<f32>,
    /// Attention output [hidden_dim]
    pub attn_out: Vec<f32>,
    /// Attention projection output [hidden_dim]
    pub attn_proj: Vec<f32>,
    /// FFN up projection [intermediate_dim]
    pub ffn_up: Vec<f32>,
    /// FFN gate projection [intermediate_dim] (for SwiGLU)
    pub ffn_gate: Vec<f32>,
    /// FFN down projection [hidden_dim]
    pub ffn_down: Vec<f32>,
    /// Output logits [vocab_size]
    pub logits: Vec<f32>,
    // PAR-126: Q8K scratch buffers for VNNI-accelerated matmul
    /// Q8K scales for hidden-dim activations [hidden_dim/256]
    pub q8k_hidden_scales: Vec<f32>,
    /// Q8K quants for hidden-dim activations [hidden_dim]
    pub q8k_hidden_quants: Vec<i8>,
    /// Q8K scales for intermediate-dim activations [intermediate_dim/256]
    pub q8k_inter_scales: Vec<f32>,
    /// Q8K quants for intermediate-dim activations [intermediate_dim]
    pub q8k_inter_quants: Vec<i8>,
}

impl InferenceScratchBuffer {
    /// Create scratch buffer from model config
    ///
    /// Pre-allocates all buffers to their maximum required size.
    /// Total memory: ~2.5MB for TinyLlama-1.1B, ~10MB for 7B models.
    #[must_use]
    pub fn from_config(config: &GGUFConfig) -> Self {
        let hidden_dim = config.hidden_dim;
        let intermediate_dim = config.intermediate_dim;
        let vocab_size = config.vocab_size;
        // GH-305: q_dim may differ from hidden_dim (Qwen3-0.6B: q_dim=2048, hidden=1024)
        let q_dim = config.q_dim();
        let kv_dim = config.kv_dim();
        let qkv_dim = q_dim + 2 * kv_dim;

        // PAR-126: Q8K uses 256-element super-blocks for VNNI path
        const QK_K: usize = 256;
        // Q8K buffers need to cover the larger of hidden_dim and q_dim for attention output
        let q8k_attn_dim = q_dim.max(hidden_dim);
        let q8k_hidden_padded = q8k_attn_dim.div_ceil(QK_K) * QK_K;
        let q8k_inter_padded = intermediate_dim.div_ceil(QK_K) * QK_K;

        Self {
            hidden: vec![0.0; hidden_dim],
            normed: vec![0.0; hidden_dim],
            qkv: vec![0.0; qkv_dim],
            q: vec![0.0; q_dim],
            k: vec![0.0; kv_dim],
            v: vec![0.0; kv_dim],
            attn_out: vec![0.0; q_dim],
            attn_proj: vec![0.0; hidden_dim],
            ffn_up: vec![0.0; intermediate_dim],
            ffn_gate: vec![0.0; intermediate_dim],
            ffn_down: vec![0.0; hidden_dim],
            logits: vec![0.0; vocab_size],
            q8k_hidden_scales: vec![0.0f32; q8k_hidden_padded / QK_K],
            q8k_hidden_quants: vec![0i8; q8k_hidden_padded],
            q8k_inter_scales: vec![0.0f32; q8k_inter_padded / QK_K],
            q8k_inter_quants: vec![0i8; q8k_inter_padded],
        }
    }

    /// Reset all buffers to zero for a new forward pass
    #[inline]
    pub fn reset(&mut self) {
        self.hidden.iter_mut().for_each(|x| *x = 0.0);
        self.normed.iter_mut().for_each(|x| *x = 0.0);
    }
}

// ============================================================================
// OwnedInferenceScratchBuffer
// ============================================================================

/// Pre-allocated scratch buffers for OwnedQuantizedModel forward passes
///
/// Eliminates per-token allocations by reusing buffers across forward passes.
/// For Qwen2.5-0.5B with intermediate_dim=4864, this saves ~40KB per token.
///
/// PAR-126: Added Q8K scratch buffers for fused Q4K×Q8K matmul path.
#[derive(Debug)]
pub struct OwnedInferenceScratchBuffer {
    /// QKV output buffer [hidden_dim + 2*kv_dim]
    pub qkv: Vec<f32>,
    /// Attention output buffer [hidden_dim]
    pub attn_out: Vec<f32>,
    /// FFN up projection buffer [intermediate_dim]
    pub ffn_up: Vec<f32>,
    /// FFN gate projection buffer [intermediate_dim]
    pub ffn_gate: Vec<f32>,
    /// FFN down output buffer [hidden_dim]
    pub ffn_down: Vec<f32>,
    /// Expanded V buffer for first token GQA [hidden_dim]
    pub expanded_v: Vec<f32>,
    /// Logits buffer [vocab_size]
    pub logits: Vec<f32>,
    /// Q8 quantization scales scratch [num_blocks]
    pub q8_scales: Vec<f32>,
    /// Q8 quantization values scratch [num_blocks * 32]
    pub q8_quants: Vec<i8>,
    // PAR-126: Q8K scratch buffers for VNNI-accelerated matmul
    /// Q8K scales for hidden-dim activations [hidden_dim/256]
    pub q8k_hidden_scales: Vec<f32>,
    /// Q8K quants for hidden-dim activations [hidden_dim]
    pub q8k_hidden_quants: Vec<i8>,
    /// Q8K scales for intermediate-dim activations [intermediate_dim/256]
    pub q8k_inter_scales: Vec<f32>,
    /// Q8K quants for intermediate-dim activations [intermediate_dim]
    pub q8k_inter_quants: Vec<i8>,
}

impl OwnedInferenceScratchBuffer {
    /// Create scratch buffer from model config
    #[must_use]
    pub fn from_config(config: &GGUFConfig) -> Self {
        let hidden_dim = config.hidden_dim;
        // GH-305: Use config.head_dim (from GGUF metadata) instead of hidden_dim / num_heads
        let q_dim = config.q_dim();
        let kv_dim = config.kv_dim();
        let qkv_dim = q_dim + 2 * kv_dim;
        let intermediate_dim = hidden_dim * 6; // Conservative estimate
        let num_blocks = q_dim.max(hidden_dim).div_ceil(32);

        const QK_K: usize = 256;
        let q8k_attn_dim = q_dim.max(hidden_dim);
        let q8k_hidden_padded = q8k_attn_dim.div_ceil(QK_K) * QK_K;
        let q8k_inter_padded = intermediate_dim.div_ceil(QK_K) * QK_K;

        Self {
            qkv: vec![0.0f32; qkv_dim],
            attn_out: vec![0.0f32; q_dim],
            ffn_up: vec![0.0f32; intermediate_dim],
            ffn_gate: vec![0.0f32; intermediate_dim],
            ffn_down: vec![0.0f32; hidden_dim],
            expanded_v: vec![0.0f32; q_dim],
            logits: vec![0.0f32; config.vocab_size],
            q8_scales: vec![0.0f32; num_blocks],
            q8_quants: vec![0i8; num_blocks * 32],
            q8k_hidden_scales: vec![0.0f32; q8k_hidden_padded / QK_K],
            q8k_hidden_quants: vec![0i8; q8k_hidden_padded],
            q8k_inter_scales: vec![0.0f32; q8k_inter_padded / QK_K],
            q8k_inter_quants: vec![0i8; q8k_inter_padded],
        }
    }

    /// Reset all buffers (clear without deallocating)
    pub fn reset(&mut self) {
        self.qkv.clear();
        self.attn_out.clear();
        self.ffn_up.clear();
        self.ffn_gate.clear();
        self.ffn_down.clear();
        self.expanded_v.clear();
        self.logits.clear();
        self.q8_scales.clear();
        self.q8_quants.clear();
        self.q8k_hidden_scales.clear();
        self.q8k_hidden_quants.clear();
        self.q8k_inter_scales.clear();
        self.q8k_inter_quants.clear();
    }
}

// ============================================================================
// ContiguousKVCache (PARITY-005)
// ============================================================================

/// Contiguous KV cache with 64-byte cache line alignment (PARITY-005)
///
/// This cache uses a single contiguous allocation for all K and V data,
/// aligned to 64-byte cache lines for optimal L2 cache performance.
///
/// ## Memory Layout
///
/// ```text
/// K cache: [layer_0][layer_1]...[layer_n] (all contiguous)
/// V cache: [layer_0][layer_1]...[layer_n] (all contiguous)
///
/// Each layer: [pos_0][pos_1]...[pos_max_seq] where each pos is [hidden_dim]
/// ```
#[derive(Debug)]
pub struct ContiguousKVCache {
    num_layers: usize,
    hidden_dim: usize,
    max_seq_len: usize,
    seq_len: usize,
    layer_stride: usize,
    k_data: Vec<f32>,
    v_data: Vec<f32>,
}

impl ContiguousKVCache {
    /// Create a new contiguous KV cache
    #[must_use]
    pub fn new(num_layers: usize, hidden_dim: usize, max_seq_len: usize) -> Self {
        let raw_layer_size = max_seq_len * hidden_dim;
        let layer_stride = Self::align_to_cache_line(raw_layer_size);
        let total_size = num_layers * layer_stride;

        Self {
            num_layers,
            hidden_dim,
            max_seq_len,
            seq_len: 0,
            layer_stride,
            k_data: vec![0.0f32; total_size],
            v_data: vec![0.0f32; total_size],
        }
    }

    #[inline]
    fn align_to_cache_line(size: usize) -> usize {
        let remainder = size % FLOATS_PER_CACHE_LINE;
        if remainder == 0 {
            size
        } else {
            size + FLOATS_PER_CACHE_LINE - remainder
        }
    }

    /// Create cache from model configuration
    #[must_use]
    pub fn from_config(config: &GGUFConfig, max_seq_len: usize) -> Self {
        Self::new(config.num_layers, config.hidden_dim, max_seq_len)
    }

    /// Check if this cache has contiguous layout
    #[must_use]
    pub const fn is_contiguous(&self) -> bool {
        true
    }

    /// Check if data is cache-line aligned
    #[must_use]
    pub fn is_cache_aligned(&self) -> bool {
        self.layer_stride.is_multiple_of(FLOATS_PER_CACHE_LINE)
    }

    /// Get the layer stride
    #[must_use]
    pub fn layer_stride(&self) -> usize {
        self.layer_stride
    }

    #[inline]
    fn layer_offset(&self, layer: usize) -> usize {
        layer * self.layer_stride
    }

    /// Append K and V vectors for a single position to a layer's cache
    pub fn append(&mut self, layer: usize, k: &[f32], v: &[f32]) {
        if layer >= self.num_layers || self.seq_len >= self.max_seq_len {
            return;
        }
        let start = self.layer_offset(layer) + self.seq_len * self.hidden_dim;
        let end = start + self.hidden_dim;
        if end <= self.k_data.len() {
            self.k_data[start..end].copy_from_slice(k);
            self.v_data[start..end].copy_from_slice(v);
        }
    }

    /// Advance the sequence position
    pub fn advance(&mut self) {
        if self.seq_len < self.max_seq_len {
            self.seq_len += 1;
        }
    }

    /// Get cached keys for a layer
    #[must_use]
    pub fn get_k(&self, layer: usize) -> &[f32] {
        if layer >= self.num_layers {
            return &[];
        }
        let start = self.layer_offset(layer);
        &self.k_data[start..start + self.seq_len * self.hidden_dim]
    }

    /// Get cached values for a layer
    #[must_use]
    pub fn get_v(&self, layer: usize) -> &[f32] {
        if layer >= self.num_layers {
            return &[];
        }
        let start = self.layer_offset(layer);
        &self.v_data[start..start + self.seq_len * self.hidden_dim]
    }

    /// Get mutable cached keys for a layer
    pub fn get_k_mut(&mut self, layer: usize) -> &mut [f32] {
        if layer >= self.num_layers {
            return &mut [];
        }
        let start = self.layer_offset(layer);
        let len = self.seq_len * self.hidden_dim;
        &mut self.k_data[start..start + len]
    }

    /// Get mutable cached values for a layer
    pub fn get_v_mut(&mut self, layer: usize) -> &mut [f32] {
        if layer >= self.num_layers {
            return &mut [];
        }
        let start = self.layer_offset(layer);
        let len = self.seq_len * self.hidden_dim;
        &mut self.v_data[start..start + len]
    }

    /// Current sequence length
    #[must_use]
    pub fn len(&self) -> usize {
        self.seq_len
    }

    /// Check if cache is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.seq_len == 0
    }

    /// Reset cache for new generation
    pub fn reset(&mut self) {
        self.seq_len = 0;
    }

    /// Reset cache and zero all data
    pub fn reset_and_zero(&mut self) {
        self.seq_len = 0;
        self.k_data.fill(0.0);
        self.v_data.fill(0.0);
    }

    /// Get maximum sequence length
    #[must_use]
    pub fn max_len(&self) -> usize {
        self.max_seq_len
    }

    /// Get total memory usage in bytes
    #[must_use]
    pub fn memory_bytes(&self) -> usize {
        (self.k_data.len() + self.v_data.len()) * std::mem::size_of::<f32>()
    }

    /// Prefetch K cache for a layer
    #[inline]
    pub fn prefetch_k(&self, layer: usize) {
        if layer < self.num_layers {
            let _ = self.k_data.get(self.layer_offset(layer));
        }
    }

    /// Prefetch V cache for a layer
    #[inline]
    pub fn prefetch_v(&self, layer: usize) {
        if layer < self.num_layers {
            let _ = self.v_data.get(self.layer_offset(layer));
        }
    }
}

// ============================================================================
// DispatchMetrics (IMP-123)
// ============================================================================

/// Thread-safe metrics for tracking CPU vs GPU dispatch decisions
///
/// Tracks dispatch counts and latency histograms for performance analysis.
#[derive(Debug)]
pub struct DispatchMetrics {
    cpu_dispatches: std::sync::atomic::AtomicUsize,
    gpu_dispatches: std::sync::atomic::AtomicUsize,
    cpu_latency_count: std::sync::atomic::AtomicUsize,
    cpu_latency_sum_us: std::sync::atomic::AtomicU64,
    gpu_latency_count: std::sync::atomic::AtomicUsize,
    gpu_latency_sum_us: std::sync::atomic::AtomicU64,
    cpu_latency_buckets: [std::sync::atomic::AtomicUsize; 5],
    gpu_latency_buckets: [std::sync::atomic::AtomicUsize; 5],
    cpu_latency_min_us: std::sync::atomic::AtomicU64,
    cpu_latency_max_us: std::sync::atomic::AtomicU64,
    gpu_latency_min_us: std::sync::atomic::AtomicU64,
    gpu_latency_max_us: std::sync::atomic::AtomicU64,
    cpu_latency_sum_sq_us: std::sync::atomic::AtomicU64,
    gpu_latency_sum_sq_us: std::sync::atomic::AtomicU64,
    start_time_ms: std::sync::atomic::AtomicU64,
}

include!("latency.rs");
include!("inference_types_config_default.rs");
