//! SIMD-accelerated inference engine
//!
//! Provides high-performance transformer inference using trueno's SIMD primitives.
//! Designed to compete with llama.cpp on CPU performance.
//!
//! ## Architecture
//!
//! ```text
//! GGUF Model → GGUFTransformer → TruenoInferenceEngine → Tokens
//! ```
//!
//! ## Modules
//!
//! - [`thread`] - Thread configuration for dynamic thread allocation
//! - [`simd`] - SIMD-accelerated operations (matmul, dot, activations)
//! - [`kv_cache`] - Key-value cache for autoregressive generation
//! - [`norm`] - Layer and RMS normalization
//! - [`rope`] - Rotary position embeddings
//! - [`engine`] - Main inference engine implementations
//! - [`quantized`] - Quantized weight storage and inference
//!
//! ## Performance Targets
//!
//! - Use trueno's SIMD Vector::dot for all dot products
//! - Use trueno's Matrix::matmul for weight projections
//! - Target >100 tokens/sec on CPU for 1B models

// Submodules
mod kv_cache;
mod norm;
mod simd;
mod thread;

#[cfg(test)]
mod coverage_tests;

// Re-exports for public API
pub use kv_cache::{attention_with_cache, attention_with_transposed_v, KVCache, OptimizedKVCache};
pub use norm::{apply_rope, simd_layer_norm, simd_rms_norm};
pub use simd::{
    simd_add, simd_bf16_dot, simd_bf16_matmul, simd_bf16_to_f32, simd_dot, simd_f16_to_f32,
    simd_gelu, simd_matmul, simd_mul, simd_silu, simd_softmax,
};
pub use thread::{
    configure_optimal_thread_pool, configure_thread_pool, InferenceMode, ThreadConfig,
};

use crate::error::{RealizarError, Result};
use crate::quantize::{fused_q4k_tiled_matvec, QK_K};

// ============================================================================
// QUANTIZED WEIGHT STORAGE (Phase 3: Memory Bandwidth Optimization)
// ============================================================================
//
// Keeps weights in quantized format for 8x memory bandwidth reduction.
// Uses fused dequant+dot operations during inference.
// ============================================================================

/// Quantized weight matrix stored in Q4_K format
///
/// Uses fused dequantize+dot operations for 8x memory bandwidth reduction.
/// Each row is stored as raw Q4_K bytes, dequantized on-the-fly during matmul.
#[derive(Clone)]
pub struct Q4KWeight {
    /// Raw Q4_K quantized data
    pub data: Vec<u8>,
    /// Input dimension (number of columns when dequantized)
    pub in_dim: usize,
    /// Output dimension (number of rows)
    pub out_dim: usize,
}

impl Q4KWeight {
    /// Create a new quantized weight from raw Q4_K data
    ///
    /// # Arguments
    ///
    /// * `data` - Raw Q4_K quantized bytes
    /// * `in_dim` - Number of input features (must be multiple of 256)
    /// * `out_dim` - Number of output features
    ///
    /// # Errors
    ///
    /// Returns error if dimensions don't match the data size
    pub fn new(data: Vec<u8>, in_dim: usize, out_dim: usize) -> Result<Self> {
        // Q4_K uses 256-element super-blocks, each taking 144 bytes
        let blocks_per_row = in_dim.div_ceil(QK_K);
        let bytes_per_row = blocks_per_row * 144; // Q4_K block size
        let expected_bytes = out_dim * bytes_per_row;

        if data.len() != expected_bytes {
            return Err(RealizarError::InvalidShape {
                reason: format!(
                    "Q4KWeight data size {} doesn't match expected {} for {}x{} matrix",
                    data.len(),
                    expected_bytes,
                    out_dim,
                    in_dim
                ),
            });
        }

        Ok(Self {
            data,
            in_dim,
            out_dim,
        })
    }

    /// Perform matrix-vector multiplication using fused Q4_K operations
    ///
    /// Uses tiled operations for cache efficiency. Each output element is
    /// computed by fused dequantize+dot against the input vector.
    ///
    /// # Arguments
    ///
    /// * `input` - Input vector of length `in_dim`
    ///
    /// # Returns
    ///
    /// Output vector of length `out_dim`
    ///
    /// # Errors
    ///
    /// Returns error if input dimension doesn't match
    pub fn matvec(&self, input: &[f32]) -> Result<Vec<f32>> {
        if input.len() != self.in_dim {
            return Err(RealizarError::InvalidShape {
                reason: format!(
                    "Input length {} doesn't match weight in_dim {}",
                    input.len(),
                    self.in_dim
                ),
            });
        }

        // Use tiled Q4_K matmul for cache efficiency
        fused_q4k_tiled_matvec(&self.data, input, self.in_dim, self.out_dim, None)
    }

    /// Get memory usage in bytes
    #[must_use]
    pub fn memory_bytes(&self) -> usize {
        self.data.len()
    }

    /// Get equivalent f32 memory usage for comparison
    #[must_use]
    pub fn f32_equivalent_bytes(&self) -> usize {
        self.in_dim * self.out_dim * 4
    }

    /// Get compression ratio vs f32
    #[must_use]
    pub fn compression_ratio(&self) -> f32 {
        self.f32_equivalent_bytes() as f32 / self.memory_bytes() as f32
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    // ------------------------------------------------------------------------
    // SIMD Operation Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_simd_dot_basic() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![1.0, 1.0, 1.0, 1.0];
        let result = simd_dot(&a, &b);
        assert!((result - 10.0).abs() < 1e-6);
    }

    #[test]
    fn test_simd_dot_zeros() {
        let a = vec![0.0; 100];
        let b = vec![1.0; 100];
        assert!((simd_dot(&a, &b)).abs() < 1e-6);
    }

    #[test]
    fn test_simd_add() {
        let mut a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        simd_add(&mut a, &b);
        assert_eq!(a, vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_simd_mul() {
        let mut a = vec![1.0, 2.0, 3.0];
        let b = vec![2.0, 3.0, 4.0];
        simd_mul(&mut a, &b);
        assert_eq!(a, vec![2.0, 6.0, 12.0]);
    }

    #[test]
    fn test_simd_softmax_basic() {
        let mut data = vec![1.0, 2.0, 3.0];
        simd_softmax(&mut data);

        // Check sum to 1
        let sum: f32 = data.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);

        // Check ordering preserved
        assert!(data[2] > data[1]);
        assert!(data[1] > data[0]);
    }

    #[test]
    fn test_simd_softmax_empty() {
        let mut data: Vec<f32> = vec![];
        simd_softmax(&mut data);
        assert!(data.is_empty());
    }

    #[test]
    fn test_simd_softmax_single() {
        let mut data = vec![5.0];
        simd_softmax(&mut data);
        assert!((data[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_simd_silu() {
        let mut data = vec![0.0, 1.0, -1.0];
        simd_silu(&mut data);
        assert!((data[0]).abs() < 1e-6); // silu(0) = 0
        assert!(data[1] > 0.5); // silu(1) > 0.5
        assert!(data[2] < 0.0); // silu(-1) < 0
    }

    #[test]
    fn test_simd_gelu() {
        let mut data = vec![0.0, 1.0, -1.0];
        simd_gelu(&mut data);
        assert!((data[0]).abs() < 1e-6); // gelu(0) ≈ 0
        assert!(data[1] > 0.8); // gelu(1) ≈ 0.84
        assert!(data[2] < 0.0); // gelu(-1) < 0
    }

    // ------------------------------------------------------------------------
    // KVCache Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_kv_cache_new() {
        let cache = KVCache::new(4, 128, 512);
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
    }

    #[test]
    fn test_kv_cache_store_and_retrieve() {
        let mut cache = KVCache::new(2, 4, 10);
        let k = vec![1.0, 2.0, 3.0, 4.0];
        let v = vec![5.0, 6.0, 7.0, 8.0];

        cache.store(0, &k, &v);
        cache.advance();

        assert_eq!(cache.len(), 1);
        assert_eq!(cache.get_k(0), &k[..]);
        assert_eq!(cache.get_v(0), &v[..]);
    }

    #[test]
    fn test_kv_cache_reset() {
        let mut cache = KVCache::new(2, 4, 10);
        let k = vec![1.0; 4];
        let v = vec![2.0; 4];

        cache.store(0, &k, &v);
        cache.advance();
        assert_eq!(cache.len(), 1);

        cache.reset();
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
    }

    // ------------------------------------------------------------------------
    // Normalization Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_simd_layer_norm() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let weight = vec![1.0; 4];
        let result = simd_layer_norm(&input, &weight, None, 1e-5);

        // Mean should be 2.5, normalized values should sum to ~0
        let sum: f32 = result.iter().sum();
        assert!(sum.abs() < 1e-5);
    }

    #[test]
    fn test_simd_rms_norm() {
        let input = vec![3.0, 4.0]; // RMS = sqrt((9+16)/2) = sqrt(12.5)
        let weight = vec![1.0, 1.0];
        let result = simd_rms_norm(&input, &weight, 1e-5);

        // Result should be normalized by RMS
        let rms = (12.5f32).sqrt();
        assert!((result[0] - 3.0 / rms).abs() < 1e-5);
        assert!((result[1] - 4.0 / rms).abs() < 1e-5);
    }

    // ------------------------------------------------------------------------
    // RoPE Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_rope_position_zero() {
        let mut x = vec![1.0, 0.0, 1.0, 0.0];
        apply_rope(&mut x, 4, 1, 0, 10000.0);
        // At position 0, cos(0)=1, sin(0)=0, so no change
        assert!((x[0] - 1.0).abs() < 1e-5);
        assert!((x[2] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_rope_changes_values() {
        let original = vec![1.0, 2.0, 3.0, 4.0];
        let mut x = original.clone();
        apply_rope(&mut x, 4, 1, 10, 10000.0);
        // Values should change at non-zero positions
        assert!(x != original);
    }

    // ------------------------------------------------------------------------
    // Q4KWeight Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_q4k_weight_memory_stats() {
        // Create minimal valid Q4_K data
        let in_dim = 256; // Minimum for Q4_K (one super-block)
        let out_dim = 1;
        let bytes_per_row = 144; // Q4_K block size for 256 elements
        let data = vec![0u8; out_dim * bytes_per_row];

        let weight = Q4KWeight::new(data, in_dim, out_dim).expect("operation failed");
        assert_eq!(weight.memory_bytes(), bytes_per_row);
        assert_eq!(weight.f32_equivalent_bytes(), in_dim * out_dim * 4);
        assert!(weight.compression_ratio() > 1.0);
    }

    #[test]
    fn test_q4k_weight_invalid_size() {
        let data = vec![0u8; 100]; // Wrong size
        let result = Q4KWeight::new(data, 256, 1);
        assert!(result.is_err());
    }
}
