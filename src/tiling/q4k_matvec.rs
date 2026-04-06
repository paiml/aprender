#![allow(missing_docs)]
//! Q4_K quantized matrix-vector tiling implementation.

use super::config::TilingConfig;

/// Q4_K superblock constants (per GGML specification)
pub const Q4K_SUPERBLOCK_SIZE: usize = 256;
pub const Q4K_SUPERBLOCK_BYTES: usize = 144;

/// Tiled Q4_K MatVec executor
///
/// Implements TCB-01 pattern: Cache-blocked matvec with 4×1 micro-kernel.
///
/// # Memory Layout
///
/// Weights are stored in Q4_K superblock format (144 bytes per 256 elements):
/// - d: f16 (2 bytes) - block scale
/// - dmin: f16 (2 bytes) - block minimum
/// - scales: 12 bytes - 8 sub-block scales (6-bit packed)
/// - qs: 128 bytes - 256 quantized values (4-bit packed)
///
/// # Performance Characteristics
///
/// - L2-resident: Process midi_tile.m rows at a time
/// - Vectorized: 4×1 micro-kernel processes 4 output rows simultaneously
/// - Aligned: K dimension aligned to Q4_K superblock (256)
#[derive(Debug, Clone)]
pub struct TiledQ4KMatvec {
    /// Tiling configuration
    pub config: TilingConfig,
    /// Number of rows (M dimension)
    pub m: usize,
    /// Number of columns (K dimension)
    pub k: usize,
}

impl TiledQ4KMatvec {
    /// Create a new tiled Q4K matvec executor
    ///
    /// # Panics
    /// Panics if K is not aligned to Q4_K superblock size (256).
    #[must_use]
    pub fn new(m: usize, k: usize) -> Self {
        assert!(
            k % Q4K_SUPERBLOCK_SIZE == 0,
            "K dimension ({}) must be aligned to Q4_K superblock size ({})",
            k,
            Q4K_SUPERBLOCK_SIZE
        );

        Self { config: TilingConfig::cpu_avx2_q4k_matvec(), m, k }
    }

    /// Get number of superblocks per row
    #[must_use]
    pub fn superblocks_per_row(&self) -> usize {
        self.k / Q4K_SUPERBLOCK_SIZE
    }

    /// Get total number of superblocks
    #[must_use]
    pub fn total_superblocks(&self) -> usize {
        self.m * self.superblocks_per_row()
    }

    /// Get weight bytes offset for a given row
    #[must_use]
    #[inline]
    pub fn weight_row_offset(&self, row: usize) -> usize {
        row * self.superblocks_per_row() * Q4K_SUPERBLOCK_BYTES
    }

    /// Calculate optimal number of parallel rows based on L2 cache
    ///
    /// Goal: Keep working set in L2 (256KB typical)
    /// Working set = midi_tile.m rows × K × sizeof(Q4K) + K × sizeof(f32)
    #[must_use]
    #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    // SAFETY: k ≤ 2^24 for practical matrix dims so usize→f32 is lossless;
    //         result of f32 multiply is non-negative and fits in usize.
    pub fn optimal_parallel_rows(&self, l2_bytes: usize) -> usize {
        // Q4K: 144 bytes per 256 elements = 0.5625 bytes/element
        let row_bytes = (self.k as f32 * 0.5625) as usize;
        // Input vector: K × 4 bytes
        let input_bytes = self.k * 4;
        // Available for rows
        let available = l2_bytes.saturating_sub(input_bytes);
        // Rows that fit (minimum 4 for micro-kernel)
        (available / row_bytes).max(4)
    }

    /// Execute tiled matvec (reference scalar implementation)
    ///
    /// This is the reference implementation for correctness testing.
    /// Actual SIMD implementation would be in the backends.
    ///
    /// For parallel execution, use [`execute_parallel`] when the `parallel` feature is enabled.
    pub fn execute_scalar(&self, weights: &[u8], input: &[f32], output: &mut [f32]) {
        assert_eq!(weights.len(), self.total_superblocks() * Q4K_SUPERBLOCK_BYTES);
        assert_eq!(input.len(), self.k);
        assert_eq!(output.len(), self.m);

        let superblocks_per_row = self.superblocks_per_row();

        for row in 0..self.m {
            let mut sum = 0.0f32;
            let row_offset = row * superblocks_per_row * Q4K_SUPERBLOCK_BYTES;

            for sb in 0..superblocks_per_row {
                let sb_offset = row_offset + sb * Q4K_SUPERBLOCK_BYTES;
                let sb_data = &weights[sb_offset..sb_offset + Q4K_SUPERBLOCK_BYTES];

                // Dequantize and dot product for this superblock
                let input_offset = sb * Q4K_SUPERBLOCK_SIZE;
                sum += self.scalar_superblock_dot(
                    sb_data,
                    &input[input_offset..input_offset + Q4K_SUPERBLOCK_SIZE],
                );
            }

            output[row] = sum;
        }
    }

    /// Execute tiled matvec with parallel row processing
    ///
    /// Uses Rayon to parallelize across rows for multi-core speedup.
    /// Falls back to scalar execution if the `parallel` feature is not enabled.
    ///
    /// # Performance
    ///
    /// Achieves near-linear speedup with core count for large matrices.
    /// For small matrices (< 256 rows), scalar may be faster due to overhead.
    #[cfg(feature = "parallel")]
    pub fn execute_parallel(&self, weights: &[u8], input: &[f32], output: &mut [f32]) {
        use rayon::prelude::*;

        assert_eq!(weights.len(), self.total_superblocks() * Q4K_SUPERBLOCK_BYTES);
        assert_eq!(input.len(), self.k);
        assert_eq!(output.len(), self.m);

        let superblocks_per_row = self.superblocks_per_row();
        let row_stride = superblocks_per_row * Q4K_SUPERBLOCK_BYTES;

        output.par_iter_mut().enumerate().for_each(|(row, out)| {
            let mut sum = 0.0f32;
            let row_offset = row * row_stride;

            for sb in 0..superblocks_per_row {
                let sb_offset = row_offset + sb * Q4K_SUPERBLOCK_BYTES;
                let sb_data = &weights[sb_offset..sb_offset + Q4K_SUPERBLOCK_BYTES];

                let input_offset = sb * Q4K_SUPERBLOCK_SIZE;
                sum += self.scalar_superblock_dot(
                    sb_data,
                    &input[input_offset..input_offset + Q4K_SUPERBLOCK_SIZE],
                );
            }

            *out = sum;
        });
    }

    /// Execute tiled matvec with parallel row processing (fallback)
    ///
    /// When `parallel` feature is not enabled, this is equivalent to `execute_scalar`.
    #[cfg(not(feature = "parallel"))]
    pub fn execute_parallel(&self, weights: &[u8], input: &[f32], output: &mut [f32]) {
        self.execute_scalar(weights, input, output);
    }

    /// Scalar dot product for a single Q4_K superblock
    ///
    /// # Performance
    ///
    /// Optimized version with:
    /// - Precomputed scale/min pairs
    /// - Loop unrolling hints
    /// - Minimized branching in inner loop
    #[inline]
    fn scalar_superblock_dot(&self, sb_data: &[u8], input: &[f32]) -> f32 {
        // Read header (hot path optimized)
        let d = f16_to_f32(sb_data.get(0..2).expect("Q4_K: need ≥2 bytes for d"));
        let dmin = f16_to_f32(sb_data.get(2..4).expect("Q4_K: need ≥4 bytes for dmin"));
        let scales = sb_data.get(4..16).expect("Q4_K: need ≥16 bytes for scales");
        let qs = sb_data.get(16..144).expect("Q4_K: need ≥144 bytes for qs");

        // Precompute all scale/min pairs upfront
        let scale_mins = precompute_scales_mins(scales);

        let mut sum = 0.0f32;

        // Process 256 values in 4 pairs of sub-blocks (GGML Q4_K layout).
        // Each pair shares 32 qs bytes: even SB uses low nibbles, odd SB uses high nibbles.
        // pair 0: SB 0 (lo) + SB 1 (hi) → qs[0..31]   → K offsets 0..63
        // pair 1: SB 2 (lo) + SB 3 (hi) → qs[32..63]  → K offsets 64..127
        // pair 2: SB 4 (lo) + SB 5 (hi) → qs[64..95]  → K offsets 128..191
        // pair 3: SB 6 (lo) + SB 7 (hi) → qs[96..127] → K offsets 192..255
        for pair in 0..4 {
            let sb_lo = pair * 2;
            let sb_hi = pair * 2 + 1;

            let (sc_lo, mn_lo) = scale_mins[sb_lo];
            let (sc_hi, mn_hi) = scale_mins[sb_hi];

            let d_scale_lo = d * sc_lo;
            let dm_lo = dmin * mn_lo;
            let d_scale_hi = d * sc_hi;
            let dm_hi = dmin * mn_hi;

            let q_offset = pair * 32; // 32 qs bytes per pair
            let input_lo = pair * 64; // low nibbles: first 32 K values
            let input_hi = pair * 64 + 32; // high nibbles: next 32 K values

            let mut pair_sum = 0.0f32;

            // 32 qs bytes: low nibble → SB_lo values, high nibble → SB_hi values
            for i in 0..32 {
                let byte = qs[q_offset + i];

                let q_lo = (byte & 0x0F) as f32;
                let q_hi = (byte >> 4) as f32;

                let val_lo = d_scale_lo * q_lo - dm_lo;
                let val_hi = d_scale_hi * q_hi - dm_hi;

                pair_sum += val_lo * input[input_lo + i];
                pair_sum += val_hi * input[input_hi + i];
            }

            sum += pair_sum;
        }

        sum
    }

    /// Get tiling statistics for profiling
    #[must_use]
    #[allow(clippy::cast_precision_loss)] // Matrix dimensions ≤ 2^24; precision loss acceptable for profiling stats
    pub fn stats(&self) -> TilingStats {
        let bytes_per_row = self.superblocks_per_row() * Q4K_SUPERBLOCK_BYTES;
        let total_weight_bytes = self.m * bytes_per_row;
        let input_bytes = self.k * 4;
        let output_bytes = self.m * 4;

        TilingStats {
            total_weight_bytes,
            input_bytes,
            output_bytes,
            superblocks: self.total_superblocks(),
            arithmetic_ops: self.m * self.k * 2, // 2 ops per element (mul + add)
            arithmetic_intensity: (self.m * self.k * 2) as f32
                / (total_weight_bytes + input_bytes) as f32,
        }
    }
}

/// Statistics for a tiled operation
#[derive(Debug, Clone)]
pub struct TilingStats {
    /// Total weight bytes
    pub total_weight_bytes: usize,
    /// Input vector bytes
    pub input_bytes: usize,
    /// Output vector bytes
    pub output_bytes: usize,
    /// Number of superblocks
    pub superblocks: usize,
    /// Total arithmetic operations
    pub arithmetic_ops: usize,
    /// Arithmetic intensity (FLOPS/byte)
    pub arithmetic_intensity: f32,
}

/// Convert 2 bytes (f16 IEEE 754) to f32
///
/// Manual implementation to avoid half crate dependency.
/// Format: 1 sign bit, 5 exponent bits, 10 mantissa bits.
///
/// # Performance
///
/// Optimized for the common case (normal numbers). Special cases (zero,
/// subnormal, inf, nan) use branches but are rare in practice for model weights.
#[inline]
pub fn f16_to_f32(bytes: &[u8]) -> f32 {
    let bits = u16::from_le_bytes([bytes[0], bytes[1]]);
    f16_bits_to_f32(bits)
}

/// Fast path f16 to f32 conversion from raw bits
///
/// Optimized version that handles the common case (normal numbers) with
/// minimal branching. Uses branchless bit manipulation for the hot path.
#[inline(always)]
fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = (bits >> 15) & 0x1;
    let exponent = (bits >> 10) & 0x1F;
    let mantissa = bits & 0x3FF;

    // Fast path: normal numbers (exponent != 0 && exponent != 31)
    // This is the common case for model weights
    if exponent != 0 && exponent != 31 {
        // Branchless conversion for normal numbers
        // f16 bias = 15, f32 bias = 127
        let f32_exp = (exponent as u32 + 112) as u32; // 127 - 15 = 112
        let f32_mant = (mantissa as u32) << 13; // 10 bits -> 23 bits
        let f32_bits = ((sign as u32) << 31) | (f32_exp << 23) | f32_mant;
        return f32::from_bits(f32_bits);
    }

    // Cold path: special cases (zero, subnormal, inf, nan)
    f16_special_to_f32(sign, exponent, mantissa)
}

/// Handle f16 special cases (zero, subnormal, inf, nan)
///
/// Cold path - marked to help branch prediction
#[cold]
#[inline(never)]
fn f16_special_to_f32(sign: u16, exponent: u16, mantissa: u16) -> f32 {
    if exponent == 0 {
        if mantissa == 0 {
            // Zero (positive or negative)
            return if sign == 1 { -0.0 } else { 0.0 };
        }
        // Subnormal f16 -> normalized f32
        // 2^-14 as constant to avoid powi() call
        const TWO_POW_NEG_14: f32 = 6.103_515_625e-5; // 2^-14
        let m = mantissa as f32 * (1.0 / 1024.0);
        let result = m * TWO_POW_NEG_14;
        return if sign == 1 { -result } else { result };
    }

    // exponent == 31: Inf or NaN
    if mantissa == 0 {
        if sign == 1 {
            f32::NEG_INFINITY
        } else {
            f32::INFINITY
        }
    } else {
        f32::NAN
    }
}

/// Extract 6-bit scale and min values from packed scales array
///
/// Extract Q4K scale and min for a given sub-block index (0-7).
///
/// GGML Q4_K_M split storage format (12 bytes encode 8 scale/min pairs):
///   bytes[0..3]:  bits[5:0] = scale SB 0-3,  bits[7:6] = high 2 bits of scale SB 4-7
///   bytes[4..7]:  bits[5:0] = min SB 0-3,    bits[7:6] = high 2 bits of min SB 4-7
///   bytes[8..11]: bits[3:0] = low 4 bits of scale SB 4-7,
///                 bits[7:4] = low 4 bits of min SB 4-7
///
/// Reference: ggml `block_q4_K.scales`, llama.cpp `vec_dot_q4_K_q8_1`,
///            trueno `backends::q4k::parse_q4k_header`.
#[inline(always)]
#[allow(clippy::cast_precision_loss)]
// SAFETY: scale and min are 6-bit values (0-63) stored in u32; lossless in f32.
pub fn extract_scale_min_6bit(scales: &[u8], idx: usize) -> (f32, f32) {
    debug_assert!(scales.len() >= 12, "scales array must be at least 12 bytes");
    debug_assert!(idx < 8, "idx must be < 8");

    if idx < 4 {
        // Sub-blocks 0-3: 6-bit values from bytes 0-3 (scale) and 4-7 (min)
        let scale = (scales[idx] & 0x3F) as u32;
        let min = (scales[4 + idx] & 0x3F) as u32;
        (scale as f32, min as f32)
    } else {
        // Sub-blocks 4-7: low 4 bits from combo byte, high 2 from upper bits
        let i = idx - 4;
        let combo = scales[8 + i];

        // Scale: low 4 bits from combo[3:0], high 2 from scales[i][7:6]
        let sc_low4 = (combo & 0x0F) as u32;
        let sc_high2 = ((scales[i] >> 6) & 0x03) as u32;
        let scale = sc_low4 | (sc_high2 << 4);

        // Min: low 4 bits from combo[7:4], high 2 from scales[4+i][7:6]
        let mn_low4 = ((combo >> 4) & 0x0F) as u32;
        let mn_high2 = ((scales[4 + i] >> 6) & 0x03) as u32;
        let min = mn_low4 | (mn_high2 << 4);

        (scale as f32, min as f32)
    }
}

/// Precompute all 8 scale/min pairs for a Q4_K superblock
///
/// More efficient than calling extract_scale_min_6bit 8 times when
/// we need all values (which is the common case).
#[inline]
fn precompute_scales_mins(scales: &[u8]) -> [(f32, f32); 8] {
    debug_assert!(scales.len() >= 12);

    // Unroll the extraction for all 8 chunks
    [
        extract_scale_min_6bit(scales, 0),
        extract_scale_min_6bit(scales, 1),
        extract_scale_min_6bit(scales, 2),
        extract_scale_min_6bit(scales, 3),
        extract_scale_min_6bit(scales, 4),
        extract_scale_min_6bit(scales, 5),
        extract_scale_min_6bit(scales, 6),
        extract_scale_min_6bit(scales, 7),
    ]
}
