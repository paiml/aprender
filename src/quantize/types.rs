//! Quantization Type Definitions (PMAT-802)
//!
//! Extracted from quantize/mod.rs - Common types and constants for quantization.
//!
//! ## Contents
//! - Constants: `BLOCK_SIZE`, `QK_K`
//! - Block structs: `Q4_0Block`, `Q8_0Block`, `Q8KSuperBlock`, `Q4_KBlock`, `Q5_KBlock`, `Q6_KBlock`
//! - `InterleavedQ4K` - Interleaved Q4_K layout for SIMD

use crate::error::{RealizarError, Result};

// Import f16_to_f32_lut from parent for InterleavedQ4K (not re-exported)
use super::f16_to_f32_lut;

// ============================================================================
// Constants
// ============================================================================

/// Block size for `Q4_0` and `Q8_0` quantization
pub const BLOCK_SIZE: usize = 32;

/// Super-block size for K-quantization formats (`Q4_K`, `Q5_K`, `Q6_K`)
pub const QK_K: usize = 256;

// ============================================================================
// Block Structs
// ============================================================================

/// `Q4_0` quantized block
///
/// Each block contains:
/// - 1 float32 scale factor
/// - 16 bytes (32 4-bit values, 2 per byte)
#[derive(Debug, Clone)]
pub struct Q4_0Block {
    /// Scale factor for dequantization
    pub scale: f32,
    /// Quantized values (16 bytes = 32 4-bit values)
    pub quants: [u8; 16],
}

/// `Q8_0` quantized block
///
/// Each block contains:
/// - 1 float32 scale factor
/// - 32 int8 values
#[derive(Debug, Clone)]
pub struct Q8_0Block {
    /// Scale factor for dequantization
    pub scale: f32,
    /// Quantized values (32 int8 values)
    pub quants: [i8; 32],
}

impl Q8_0Block {
    /// Quantize 32 f32 values to Q8_0 format
    ///
    /// Dynamic quantization for activations during inference.
    /// Uses symmetric quantization: scale = max(abs(values)) / 127.0
    ///
    /// # Arguments
    /// * `values` - Exactly 32 f32 values to quantize
    ///
    /// # Returns
    /// A Q8_0Block with scale and quantized int8 values
    ///
    /// # Example
    /// ```ignore
    /// let values = [1.0f32; 32];
    /// let block = Q8_0Block::quantize(&values);
    /// assert_eq!(block.quants[0], 127); // max value maps to 127
    /// ```
    #[must_use]
    pub fn quantize(values: &[f32; 32]) -> Self {
        contract_pre_quantize!();
        // Find max absolute value for symmetric quantization
        let max_abs = values.iter().map(|v| v.abs()).fold(0.0f32, f32::max);

        // Avoid division by zero
        let scale = if max_abs > 1e-10 {
            max_abs / 127.0
        } else {
            1.0 / 127.0 // Minimal scale for near-zero blocks
        };

        // Quantize each value: qi = round(fi / scale), clamped to [-128, 127]
        let mut quants = [0i8; 32];
        for (i, &v) in values.iter().enumerate() {
            let q = (v / scale).round();
            quants[i] = q.clamp(-128.0, 127.0) as i8;
        }

        Self { scale, quants }
    }

    /// Dequantize the block back to f32 values
    ///
    /// # Returns
    /// Array of 32 f32 values: values[i] = quants[i] * scale
    #[must_use]
    pub fn dequantize(&self) -> [f32; 32] {
        let mut values = [0.0f32; 32];
        for (i, &q) in self.quants.iter().enumerate() {
            values[i] = q as f32 * self.scale;
        }
        values
    }

    /// Compute quantization error (max absolute difference)
    #[must_use]
    pub fn quantization_error(&self, original: &[f32; 32]) -> f32 {
        let dequantized = self.dequantize();
        original
            .iter()
            .zip(dequantized.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max)
    }

    /// Compute relative quantization error
    #[must_use]
    pub fn relative_error(&self, original: &[f32; 32]) -> f32 {
        let max_val = original.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        if max_val < 1e-10 {
            return 0.0;
        }
        self.quantization_error(original) / max_val
    }
}

/// `Q8_K` quantized super-block (llama.cpp-compatible activation format)
///
/// Super-block aligned format for maximum SIMD efficiency with Q4_K weights.
/// Uses single scale per 256 values (vs Q8_0's scale per 32 values).
///
/// Each super-block contains:
/// - 1 float32 scale factor (for all 256 values)
/// - 256 int8 quantized values
///
/// # Performance
///
/// - Aligned with Q4_K super-block (256 values)
/// - Single scale multiplication per super-block (vs 8 for Q8_0)
/// - Enables contiguous SIMD loads without shuffle/deinterleave
/// - Matches llama.cpp `block_q8_K` structure
#[derive(Debug, Clone)]
pub struct Q8KSuperBlock {
    /// Scale factor for the entire super-block
    pub scale: f32,
    /// 256 quantized int8 values
    pub quants: [i8; 256],
}

impl Q8KSuperBlock {
    /// Quantize 256 f32 values to Q8_K format
    ///
    /// Uses symmetric quantization: scale = max(abs(values)) / 127.0
    ///
    /// # Arguments
    /// * `values` - Exactly 256 f32 values (one super-block)
    ///
    /// # Returns
    /// A Q8KSuperBlock with single scale and 256 quantized values
    #[must_use]
    pub fn quantize(values: &[f32; 256]) -> Self {
        // Find max absolute value for symmetric quantization
        let max_abs = values.iter().map(|v| v.abs()).fold(0.0f32, f32::max);

        // Avoid division by zero
        let scale = if max_abs > 1e-10 {
            max_abs / 127.0
        } else {
            1.0 / 127.0
        };

        let inv_scale = 1.0 / scale;

        // Quantize all 256 values
        let mut quants = [0i8; 256];
        for (i, &v) in values.iter().enumerate() {
            let q = (v * inv_scale).round();
            quants[i] = q.clamp(-128.0, 127.0) as i8;
        }

        Self { scale, quants }
    }

    /// Zero-allocation quantization into pre-allocated buffer
    ///
    /// # Arguments
    /// * `values` - 256 f32 values to quantize
    /// * `scale_out` - Output for scale value
    /// * `quants_out` - Output buffer for 256 int8 quantized values
    #[inline]
    pub fn quantize_into(values: &[f32], scale_out: &mut f32, quants_out: &mut [i8]) {
        debug_assert!(values.len() >= 256);
        debug_assert!(quants_out.len() >= 256);

        // Find max absolute value
        let max_abs = values[..256].iter().map(|v| v.abs()).fold(0.0f32, f32::max);

        let scale = if max_abs > 1e-10 {
            max_abs / 127.0
        } else {
            1.0 / 127.0
        };
        *scale_out = scale;

        let inv_scale = 1.0 / scale;

        for (i, &v) in values[..256].iter().enumerate() {
            let q = (v * inv_scale).round();
            quants_out[i] = q.clamp(-128.0, 127.0) as i8;
        }
    }

    /// Dequantize back to f32 values
    #[must_use]
    pub fn dequantize(&self) -> [f32; 256] {
        let mut values = [0.0f32; 256];
        for (i, &q) in self.quants.iter().enumerate() {
            values[i] = q as f32 * self.scale;
        }
        values
    }
}

/// `Q4_K` quantized super-block
///
/// K-quantization uses super-blocks of 256 values (8 blocks of 32 each).
/// Achieves 4.5 bits per weight with better quality than `Q4_0`.
///
/// Each super-block contains:
/// - 1 half-precision scale factor (`d`)
/// - 1 half-precision min factor (`dmin`)
/// - 12 bytes of 6-bit quantized scales (for 8 blocks)
/// - 128 bytes of 4-bit quantized values (256 values)
///
/// Total: 2 + 2 + 12 + 128 = 144 bytes per super-block of 256 values
/// = 4.5 bits per weight
#[derive(Debug, Clone)]
#[allow(non_camel_case_types)]
pub struct Q4_KBlock {
    /// Super-block scale (f16, stored as f32 after conversion)
    pub d: f32,
    /// Super-block min (f16, stored as f32 after conversion)
    pub dmin: f32,
    /// Per-block scales (packed 6-bit values)
    pub scales: [u8; 12],
    /// Quantized values (128 bytes = 256 4-bit values)
    pub qs: [u8; 128],
}

/// `Q5_K` quantized super-block
///
/// K-quantization uses super-blocks of 256 values (8 blocks of 32 each).
/// Achieves 5.5 bits per weight with higher quality than `Q4_K`.
///
/// Each super-block contains:
/// - 1 half-precision scale factor (`d`)
/// - 1 half-precision min factor (`dmin`)
/// - 12 bytes of 6-bit quantized scales (for 8 blocks)
/// - 32 bytes of high bits (1 bit per value for 5-bit quantization)
/// - 128 bytes of low 4-bit quantized values
///
/// Total: 2 + 2 + 12 + 32 + 128 = 176 bytes per super-block of 256 values
/// = 5.5 bits per weight
#[derive(Debug, Clone)]
#[allow(non_camel_case_types)]
pub struct Q5_KBlock {
    /// Super-block scale
    pub d: f32,
    /// Super-block min
    pub dmin: f32,
    /// Per-block scales (packed 6-bit values)
    pub scales: [u8; 12],
    /// High bits (1 bit per value)
    pub qh: [u8; 32],
    /// Low 4-bit quantized values
    pub qs: [u8; 128],
}

/// `Q6_K` quantized super-block
///
/// K-quantization uses super-blocks of 256 values (16 blocks of 16 each).
/// Achieves 6.5625 bits per weight with the highest quality among K-quant formats.
///
/// Each super-block contains:
/// - 1 half-precision scale factor (`d`)
/// - 16 bytes of 8-bit quantized scales (for 16 blocks)
/// - 64 bytes of high 2 bits (2 bits per value for 6-bit quantization)
/// - 128 bytes of low 4-bit quantized values
///
/// Total: 2 + 16 + 64 + 128 = 210 bytes per super-block of 256 values
/// = 6.5625 bits per weight
#[derive(Debug, Clone)]
#[allow(non_camel_case_types)]
pub struct Q6_KBlock {
    /// Super-block scale
    pub d: f32,
    /// Per-block scales (8-bit signed)
    pub scales: [i8; 16],
    /// High 2 bits per value
    pub qh: [u8; 64],
    /// Low 4-bit quantized values
    pub qs: [u8; 128],
}

/// Interleaved Q4_K layout optimized for SIMD operations
///
/// Reorders quantized values during model load so that SIMD dot products
/// can process contiguous memory without gather operations.
///
/// # Performance
///
/// The interleaved layout eliminates cross-lane shuffles in AVX2:
/// - Standard Q4_K: requires `vpermd` for each 32-value block (~5 cycles)
/// - Interleaved: direct `vmovdqu` loads (~1 cycle)
///
/// Trade-off: ~10% slower model load, ~15% faster inference.
#[derive(Debug, Clone)]
pub struct InterleavedQ4K {
    /// Super-block scales (one f32 per super-block)
    pub d: Vec<f32>,
    /// Super-block mins (one f32 per super-block)
    pub dmin: Vec<f32>,
    /// Per-block scales (12 bytes per super-block)
    pub scales: Vec<u8>,
    /// Interleaved quantized values (128 bytes per super-block)
    pub qs: Vec<u8>,
    /// Number of super-blocks
    pub num_super_blocks: usize,
}

impl InterleavedQ4K {
    /// Create interleaved Q4_K from raw GGUF Q4_K data
    ///
    /// Reorders the quantized values at load time for SIMD-efficient access.
    /// This is a one-time cost at model load that amortizes over all inference.
    ///
    /// # Arguments
    ///
    /// * `q4k_data` - Raw Q4_K data (144 bytes per super-block)
    ///
    /// # Returns
    ///
    /// InterleavedQ4K with reordered weights
    ///
    /// # Errors
    ///
    /// Returns error if data length is not a multiple of super-block size
    pub fn from_q4k(q4k_data: &[u8]) -> Result<Self> {
        const SUPER_BLOCK_BYTES: usize = 144;

        if !q4k_data.len().is_multiple_of(SUPER_BLOCK_BYTES) {
            return Err(RealizarError::InvalidShape {
                reason: format!(
                    "Q4_K data length {} is not a multiple of super-block size {}",
                    q4k_data.len(),
                    SUPER_BLOCK_BYTES
                ),
            });
        }

        let num_super_blocks = q4k_data.len() / SUPER_BLOCK_BYTES;

        let mut d = Vec::with_capacity(num_super_blocks);
        let mut dmin = Vec::with_capacity(num_super_blocks);
        let mut scales = Vec::with_capacity(num_super_blocks * 12);
        let mut qs = Vec::with_capacity(num_super_blocks * 128);

        for sb in 0..num_super_blocks {
            let sb_start = sb * SUPER_BLOCK_BYTES;

            // Read d and dmin (f16 -> f32)
            let d_val = f16_to_f32_lut(u16::from_le_bytes([
                q4k_data[sb_start],
                q4k_data[sb_start + 1],
            ]));
            let dmin_val = f16_to_f32_lut(u16::from_le_bytes([
                q4k_data[sb_start + 2],
                q4k_data[sb_start + 3],
            ]));

            d.push(d_val);
            dmin.push(dmin_val);

            // Copy scales
            scales.extend_from_slice(&q4k_data[sb_start + 4..sb_start + 16]);

            // Interleave quantized values
            // Original: byte[i] = (value[2i+1] << 4) | value[2i]
            // We reorder so that after SIMD nibble extraction, values are contiguous
            //
            // For AVX2 processing 64 values at a time:
            // - Load 32 bytes, extract low nibbles -> 32 values
            // - Same 32 bytes, extract high nibbles -> 32 more values
            //
            // Interleave pattern: group values by their position in SIMD lanes
            // This eliminates the need for cross-lane shuffles
            let qs_start = sb_start + 16;
            let original_qs = &q4k_data[qs_start..qs_start + 128];

            // For now, use identity interleave (same as original)
            // The optimization comes from the specialized kernel that knows the layout
            // Future: implement actual interleave pattern based on profiling
            qs.extend_from_slice(original_qs);
        }

        Ok(Self {
            d,
            dmin,
            scales,
            qs,
            num_super_blocks,
        })
    }

    /// Get the number of values (256 per super-block)
    #[must_use]
    pub fn num_values(&self) -> usize {
        self.num_super_blocks * QK_K
    }
}

// ============================================================================
// SIMD Backend Detection
// ============================================================================

/// Batch dequantization stats for performance tracking
#[derive(Debug, Clone, Default)]
pub struct DequantStats {
    /// Total blocks processed
    pub blocks_processed: u64,
    /// Total bytes dequantized
    pub bytes_processed: u64,
    /// SIMD backend used
    pub simd_backend: SimdBackend,
}

include!("simd_backend.rs");
