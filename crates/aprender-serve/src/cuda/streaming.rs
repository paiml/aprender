//! GH-201: Layer Streaming Mode for Limited VRAM
//!
//! Provides shared infrastructure for streaming layer weights to GPU when
//! pre-caching all weights would exceed available VRAM.
//!
//! ## Problem
//!
//! SafeTensors and APR GPU paths pre-cache ALL weights upfront:
//! - `safetensors_cuda.rs::upload_weights()` - ~6GB for 1.5B model
//! - `apr/cuda.rs::pre_cache_weights()` - ~6GB for 1.5B model
//!
//! This causes OOM on GPUs with < 6GB VRAM.
//!
//! ## Solution
//!
//! Two modes:
//! 1. **Full Cache Mode**: Pre-cache all weights (when VRAM sufficient)
//! 2. **Layer Streaming Mode**: Stream layer weights on-demand (when VRAM limited)
//!
//! ## Memory Layout
//!
//! ```text
//! Full Cache (~6GB):              Streaming (~1.5GB):
//! ┌─────────────────────┐         ┌─────────────────────┐
//! │ LM Head (~900MB)    │         │ LM Head (~900MB)    │
//! │ Layer 0 (~187MB)    │         │ Layer Buffer (~200MB)│ ← Reused
//! │ Layer 1 (~187MB)    │         │ KV Cache (~57MB)    │
//! │ ...                 │         └─────────────────────┘
//! │ Layer 27 (~187MB)   │
//! │ KV Cache (~57MB)    │
//! └─────────────────────┘
//! ```

/// Model configuration for VRAM estimation
#[derive(Debug, Clone)]
pub struct StreamingConfig {
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
    /// Maximum sequence length for KV cache
    pub max_seq_len: usize,
}

/// F32 = 4 bytes per element
const F32_SIZE: usize = 4;

/// Safety margin: use 90% of free VRAM to leave room for CUDA runtime overhead
const VRAM_SAFETY_MARGIN: f64 = 0.90;

impl StreamingConfig {
    /// Estimate VRAM required for full cache mode (all layers pre-cached).
    ///
    /// Components:
    /// - LM head: hidden_dim × vocab_size × 4 bytes
    /// - Per layer (×num_layers):
    ///   - QKV: hidden_dim × (hidden_dim + 2×kv_dim) × 4
    ///   - O projection: hidden_dim × hidden_dim × 4
    ///   - FFN gate: intermediate_dim × hidden_dim × 4
    ///   - FFN up: intermediate_dim × hidden_dim × 4
    ///   - FFN down: hidden_dim × intermediate_dim × 4
    ///   - Norms: 2 × hidden_dim × 4
    /// - KV cache: 2 × num_layers × max_seq_len × kv_dim × 4
    #[must_use]
    pub fn estimate_full_cache_vram(&self) -> usize {
        let head_dim = self.hidden_dim / self.num_heads;
        let kv_dim = self.num_kv_heads * head_dim;

        // LM head (transposed: hidden_dim × vocab_size)
        let lm_head_bytes = self.hidden_dim * self.vocab_size * F32_SIZE;

        // Output norm gamma
        let output_norm_bytes = self.hidden_dim * F32_SIZE;

        // Per-layer weights
        let per_layer_bytes = self.estimate_layer_vram();
        let total_layer_bytes = self.num_layers * per_layer_bytes;

        // KV cache: 2 (K + V) × num_layers × max_seq_len × kv_dim
        let kv_cache_bytes = 2 * self.num_layers * self.max_seq_len * kv_dim * F32_SIZE;

        lm_head_bytes + output_norm_bytes + total_layer_bytes + kv_cache_bytes
    }

    /// Estimate VRAM required for streaming mode (only one layer at a time).
    ///
    /// Components:
    /// - LM head: hidden_dim × vocab_size × 4 bytes (always needed)
    /// - Layer buffer: single layer's weights (reused)
    /// - KV cache: 2 × num_layers × max_seq_len × kv_dim × 4 (always needed)
    #[must_use]
    pub fn estimate_streaming_vram(&self) -> usize {
        let head_dim = self.hidden_dim / self.num_heads;
        let kv_dim = self.num_kv_heads * head_dim;

        // LM head (always on GPU)
        let lm_head_bytes = self.hidden_dim * self.vocab_size * F32_SIZE;

        // Output norm gamma
        let output_norm_bytes = self.hidden_dim * F32_SIZE;

        // Single layer buffer (reused for all layers)
        let layer_buffer_bytes = self.estimate_layer_vram();

        // KV cache (always on GPU, grows with sequence)
        let kv_cache_bytes = 2 * self.num_layers * self.max_seq_len * kv_dim * F32_SIZE;

        lm_head_bytes + output_norm_bytes + layer_buffer_bytes + kv_cache_bytes
    }

    /// Estimate VRAM required for a single layer's weights.
    #[must_use]
    pub fn estimate_layer_vram(&self) -> usize {
        let head_dim = self.hidden_dim / self.num_heads;
        let kv_dim = self.num_kv_heads * head_dim;
        let qkv_out_dim = self.hidden_dim + 2 * kv_dim;

        // QKV (transposed: hidden_dim × qkv_out_dim)
        let qkv = self.hidden_dim * qkv_out_dim * F32_SIZE;
        // O projection (transposed: hidden_dim × hidden_dim)
        let o_proj = self.hidden_dim * self.hidden_dim * F32_SIZE;
        // FFN gate (transposed: hidden_dim × intermediate_dim)
        let ffn_gate = self.hidden_dim * self.intermediate_dim * F32_SIZE;
        // FFN up (transposed: hidden_dim × intermediate_dim)
        let ffn_up = self.hidden_dim * self.intermediate_dim * F32_SIZE;
        // FFN down (transposed: intermediate_dim × hidden_dim)
        let ffn_down = self.intermediate_dim * self.hidden_dim * F32_SIZE;
        // Attn + FFN norms
        let norms = 2 * self.hidden_dim * F32_SIZE;

        qkv + o_proj + ffn_gate + ffn_up + ffn_down + norms
    }

    /// GH-178: Estimate VRAM for FP8/FP16 prefill weight cache.
    ///
    /// The prefill cache dequantizes Q4K weights to FP8 (1 byte/elem)
    /// or FP16 (2 bytes/elem) for cuBLAS HGEMM prefill acceleration.
    /// This is a LARGE allocation (~1.5GB FP8, ~3GB FP16 for 1.5B)
    /// that must be budgeted before model load.
    #[must_use]
    pub fn estimate_prefill_cache_vram(&self, fp8: bool) -> usize {
        let bytes_per_elem: usize = if fp8 { 1 } else { 2 }; // FP8=1, FP16=2
        let head_dim = self.hidden_dim / self.num_heads.max(1);
        let kv_dim = self.num_kv_heads * head_dim;
        let qkv_out = self.hidden_dim + 2 * kv_dim;

        // Each matmul weight matrix: N × K elements
        // Per layer: QKV(hidden×qkv) + O(hidden×hidden) +
        //   gate(hidden×inter) + up(hidden×inter) + down(inter×hidden)
        let per_layer = self.hidden_dim * qkv_out
            + self.hidden_dim * self.hidden_dim
            + self.hidden_dim * self.intermediate_dim * 3;
        let lm_head = self.hidden_dim * self.vocab_size;

        (per_layer * self.num_layers + lm_head) * bytes_per_elem
    }
}

/// Determine whether to use streaming mode based on available VRAM.
///
/// Returns `true` if streaming mode should be used (VRAM insufficient for full cache).
#[must_use]
pub fn should_use_streaming(free_vram: usize, config: &StreamingConfig) -> bool {
    let full_cache_required = config.estimate_full_cache_vram();
    let usable_vram = (free_vram as f64 * VRAM_SAFETY_MARGIN) as usize;

    if full_cache_required > usable_vram {
        let streaming_required = config.estimate_streaming_vram();
        // Only use streaming if it actually fits
        streaming_required <= usable_vram
    } else {
        false // Full cache fits, no need for streaming
    }
}

/// Check if model can fit in VRAM at all (even with streaming).
///
/// Returns `Err` with descriptive message if neither mode fits.
pub fn check_vram_sufficient(
    free_vram: usize,
    total_vram: usize,
    config: &StreamingConfig,
) -> Result<StreamingMode, String> {
    let full_cache_required = config.estimate_full_cache_vram();
    let streaming_required = config.estimate_streaming_vram();
    let usable_vram = (free_vram as f64 * VRAM_SAFETY_MARGIN) as usize;

    let full_mb = full_cache_required / (1024 * 1024);
    let streaming_mb = streaming_required / (1024 * 1024);
    let free_mb = free_vram / (1024 * 1024);
    let total_mb = total_vram / (1024 * 1024);

    if full_cache_required <= usable_vram {
        Ok(StreamingMode::FullCache)
    } else if streaming_required <= usable_vram {
        Ok(StreamingMode::LayerStreaming)
    } else {
        Err(format!(
            "Insufficient VRAM for GPU inference (GH-201). \
             Full cache: {full_mb} MB, Streaming: {streaming_mb} MB, \
             Available: {free_mb} MB (of {total_mb} MB total). \
             Solutions: (1) Use GGUF format: `apr run model.gguf`, \
             (2) Use CPU inference: `--device cpu`, \
             (3) Free GPU memory by closing other applications."
        ))
    }
}

/// Streaming mode selection result
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamingMode {
    /// Pre-cache all layer weights on GPU (maximum throughput)
    FullCache,
    /// Stream layer weights on-demand (reduced VRAM usage)
    LayerStreaming,
}

impl StreamingMode {
    /// Human-readable description of the mode
    #[must_use]
    pub fn description(&self) -> &'static str {
        match self {
            Self::FullCache => "Full Cache (all layers pre-cached on GPU)",
            Self::LayerStreaming => "Layer Streaming (weights loaded per-layer)",
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn qwen2_1_5b_config() -> StreamingConfig {
        StreamingConfig {
            hidden_dim: 1536,
            num_layers: 28,
            num_heads: 12,
            num_kv_heads: 2,
            vocab_size: 151936,
            intermediate_dim: 8960,
            max_seq_len: 2048,
        }
    }

    fn small_config() -> StreamingConfig {
        StreamingConfig {
            hidden_dim: 256,
            num_layers: 4,
            num_heads: 4,
            num_kv_heads: 4,
            vocab_size: 1000,
            intermediate_dim: 512,
            max_seq_len: 512,
        }
    }

    #[test]
    fn test_full_cache_vram_qwen2_1_5b() {
        let config = qwen2_1_5b_config();
        let vram = config.estimate_full_cache_vram();
        let vram_mb = vram / (1024 * 1024);

        // Qwen2.5-Coder-1.5B should require ~5.5-7 GB in F32
        assert!(
            vram_mb > 5500 && vram_mb < 7000,
            "Expected 5.5-7 GB, got {} MB",
            vram_mb
        );
    }

    #[test]
    fn test_streaming_vram_much_smaller() {
        let config = qwen2_1_5b_config();
        let full = config.estimate_full_cache_vram();
        let streaming = config.estimate_streaming_vram();

        // Streaming should use significantly less VRAM
        assert!(
            streaming < full / 2,
            "Streaming ({} MB) should be < half of full cache ({} MB)",
            streaming / (1024 * 1024),
            full / (1024 * 1024)
        );
    }

    #[test]
    fn test_streaming_vram_includes_lm_head_and_kv() {
        let config = qwen2_1_5b_config();
        let streaming = config.estimate_streaming_vram();

        // LM head alone is ~900 MB for Qwen2
        let lm_head = config.hidden_dim * config.vocab_size * F32_SIZE;
        assert!(
            streaming > lm_head,
            "Streaming VRAM should include more than just LM head"
        );
    }

    #[test]
    fn test_layer_vram_estimate() {
        let config = qwen2_1_5b_config();
        let layer = config.estimate_layer_vram();
        let layer_mb = layer / (1024 * 1024);

        // Each layer should be ~180-200 MB for 1.5B model
        assert!(
            layer_mb > 150 && layer_mb < 250,
            "Expected 150-250 MB per layer, got {} MB",
            layer_mb
        );
    }

    #[test]
    fn test_should_use_streaming_small_vram() {
        let config = qwen2_1_5b_config();

        // 2GB VRAM - should use streaming
        let free_vram = 2 * 1024 * 1024 * 1024;
        assert!(
            should_use_streaming(free_vram, &config),
            "2GB VRAM should trigger streaming mode"
        );
    }

    #[test]
    fn test_should_use_streaming_large_vram() {
        let config = qwen2_1_5b_config();

        // 12GB VRAM - should NOT use streaming
        let free_vram = 12 * 1024 * 1024 * 1024;
        assert!(
            !should_use_streaming(free_vram, &config),
            "12GB VRAM should use full cache mode"
        );
    }

    #[test]
    fn test_check_vram_sufficient_full_cache() {
        let config = small_config();
        let free_vram = 1024 * 1024 * 1024; // 1GB
        let total_vram = 2 * 1024 * 1024 * 1024; // 2GB

        let result = check_vram_sufficient(free_vram, total_vram, &config);
        assert!(result.is_ok());
        assert_eq!(result.expect("result"), StreamingMode::FullCache);
    }

    #[test]
    fn test_check_vram_sufficient_streaming() {
        let config = qwen2_1_5b_config();
        let free_vram = 2 * 1024 * 1024 * 1024; // 2GB
        let total_vram = 4 * 1024 * 1024 * 1024; // 4GB

        let result = check_vram_sufficient(free_vram, total_vram, &config);
        assert!(result.is_ok());
        assert_eq!(result.expect("result"), StreamingMode::LayerStreaming);
    }

    #[test]
    fn test_check_vram_insufficient() {
        let config = qwen2_1_5b_config();
        let free_vram = 512 * 1024 * 1024; // 512MB - too small even for streaming
        let total_vram = 1024 * 1024 * 1024; // 1GB

        let result = check_vram_sufficient(free_vram, total_vram, &config);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Insufficient VRAM"));
    }

    #[test]
    fn test_streaming_mode_description() {
        assert!(StreamingMode::FullCache
            .description()
            .contains("pre-cached"));
        assert!(StreamingMode::LayerStreaming
            .description()
            .contains("per-layer"));
    }
}
