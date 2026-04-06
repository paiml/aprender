//! Quantized Block Format Trait (Contract: quantized-dot-product-v1.yaml)
//!
//! Encodes the mathematical structure of blocked quantization formats as compile-time
//! constants. Enables generic kernels that are monomorphized per format — zero runtime
//! overhead, maximum code reuse.
//!
//! ## Paper Citations
//!
//! - GPTQ (Frantar 2022): Blocked quantization with per-block scales
//! - LLM.int8() (Dettmers 2022): Affine quantization x = d*s*q - dmin*m
//! - GGML K-quant (ggerganov): 256-element super-blocks with 6-bit packed scales
//!
//! ## Key Algebra
//!
//! General dequantization: `x_i = d * s_j * q_i - dmin * m_j`
//!
//! Dot product decomposition:
//! ```text
//! dot(W, x) = Σ_sb [ d*Σ_j(s_j*Σ_i(q_W*q_x)) - dmin*Σ_j(m_j*Σ_i(q_x)) ]
//! ```
//!
//! The offset term depends ONLY on activation sums (bsums), not weights.

use crate::error::{RealizarError, Result};

// Import helpers from sibling modules
use super::simd::{extract_scale_min, read_f16};

/// Quantization format family
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantFamily {
    /// K-quantization: 256-element super-blocks with sub-block scales
    KQuant,
    /// Simple quantization: 32-element blocks with single scale
    Simple,
}

/// Trait encoding the mathematical structure of a blocked quantization format.
///
/// All associated constants are known at compile time. Implementations are
/// monomorphized, so generic kernels parameterized by this trait have zero
/// runtime overhead vs hand-written format-specific kernels.
///
/// # Contract
///
/// Every implementation must match the corresponding entry in
/// `contracts/quantized-dot-product-v1.yaml`.
pub trait QuantBlockFormat: Send + Sync + 'static {
    /// Format identifier (must match YAML key in format registry)
    const FORMAT_ID: &'static str;

    /// Quantization family: KQuant (256-element super-blocks) or Simple (32-element)
    const FAMILY: QuantFamily;

    /// Number of quantized values per super-block
    const ELEMENTS_PER_SUPERBLOCK: usize;

    /// Number of sub-blocks within each super-block
    const SUBBLOCKS_PER_SUPERBLOCK: usize;

    /// Number of elements per sub-block
    const ELEMENTS_PER_SUBBLOCK: usize;

    /// Byte size of one super-block in the packed format
    const SUPERBLOCK_BYTES: usize;

    /// Number of quantization bits per value
    const QUANT_BITS: u8;

    /// Whether format has a dmin (minimum) correction term
    /// When true: dequant = d*s*q - dmin*m
    /// When false: offset term vanishes from dot product
    const HAS_DMIN: bool;

    /// Whether quantized values are signed
    const QUANT_SIGNED: bool;

    /// Zero offset subtracted during dequantization (8 for Q4_0, 32 for Q6_K, 0 otherwise)
    const ZERO_OFFSET: i32;

    /// Maximum acceptable scalar-SIMD divergence in ULPs
    const ULP_TOLERANCE: u32;

    /// Bits per weight (including metadata overhead)
    const BITS_PER_WEIGHT: f32;

    /// Byte offset of the super-block scale (d) within a super-block
    const D_OFFSET: usize;

    /// Byte offset of the super-block min (dmin) — 0 if no dmin
    const DMIN_OFFSET: usize;

    /// Byte offset of the sub-block scales array
    const SCALES_OFFSET: usize;

    /// Byte count of the scales array
    const SCALES_BYTES: usize;

    /// Byte offset of the quantized values (qs)
    const QS_OFFSET: usize;

    /// Byte count of the quantized values
    const QS_BYTES: usize;

    /// Read the super-block scale `d` (f16 → f32)
    fn read_d(superblock: &[u8]) -> f32;

    /// Read the super-block min `dmin` (f16 → f32), returns 0.0 if format has no dmin
    fn read_dmin(superblock: &[u8]) -> f32;

    /// Extract the sub-block scale for sub-block `idx`
    fn extract_subblock_scale(superblock: &[u8], idx: usize) -> f32;

    /// Extract the sub-block min for sub-block `idx`, returns 0.0 if format has no dmin
    fn extract_subblock_min(superblock: &[u8], idx: usize) -> f32;

    /// Dequantize a single value at position `i` within the super-block
    fn dequant_value(superblock: &[u8], i: usize) -> f32;

    /// Validate that `data` is a valid sequence of super-blocks.
    /// Returns the number of super-blocks on success.
    fn validate_data_length(data: &[u8]) -> Result<usize> {
        if data.is_empty() {
            return Err(RealizarError::InvalidShape {
                reason: format!("{} data is empty", Self::FORMAT_ID),
            });
        }
        if !data.len().is_multiple_of(Self::SUPERBLOCK_BYTES) {
            return Err(RealizarError::InvalidShape {
                reason: format!(
                    "{} data length {} is not a multiple of super-block size {}",
                    Self::FORMAT_ID,
                    data.len(),
                    Self::SUPERBLOCK_BYTES
                ),
            });
        }
        Ok(data.len() / Self::SUPERBLOCK_BYTES)
    }
}

// =============================================================================
// Q4_K Implementation
// =============================================================================

/// Q4_K format: 4-bit K-quantization with 256-element super-blocks
///
/// Layout: d(2) + dmin(2) + scales(12) + qs(128) = 144 bytes
/// Dequant: d * s_j * q_i - dmin * m_j
pub struct Q4K;

impl QuantBlockFormat for Q4K {
    const FORMAT_ID: &'static str = "Q4_K";
    const FAMILY: QuantFamily = QuantFamily::KQuant;
    const ELEMENTS_PER_SUPERBLOCK: usize = 256;
    const SUBBLOCKS_PER_SUPERBLOCK: usize = 8;
    const ELEMENTS_PER_SUBBLOCK: usize = 32;
    const SUPERBLOCK_BYTES: usize = 144;
    const QUANT_BITS: u8 = 4;
    const HAS_DMIN: bool = true;
    const QUANT_SIGNED: bool = false;
    const ZERO_OFFSET: i32 = 0;
    const ULP_TOLERANCE: u32 = 8;
    const BITS_PER_WEIGHT: f32 = 4.5;
    const D_OFFSET: usize = 0;
    const DMIN_OFFSET: usize = 2;
    const SCALES_OFFSET: usize = 4;
    const SCALES_BYTES: usize = 12;
    const QS_OFFSET: usize = 16;
    const QS_BYTES: usize = 128;

    #[inline]
    fn read_d(superblock: &[u8]) -> f32 {
        read_f16(&superblock[0..2])
    }

    #[inline]
    fn read_dmin(superblock: &[u8]) -> f32 {
        read_f16(&superblock[2..4])
    }

    #[inline]
    fn extract_subblock_scale(superblock: &[u8], idx: usize) -> f32 {
        let mut scales = [0u8; 12];
        scales.copy_from_slice(&superblock[4..16]);
        let (scale, _min) = extract_scale_min(&scales, idx);
        scale
    }

    #[inline]
    fn extract_subblock_min(superblock: &[u8], idx: usize) -> f32 {
        let mut scales = [0u8; 12];
        scales.copy_from_slice(&superblock[4..16]);
        let (_scale, min) = extract_scale_min(&scales, idx);
        min
    }

    #[inline]
    fn dequant_value(superblock: &[u8], i: usize) -> f32 {
        contract_pre_dequantization!();
        let d = Self::read_d(superblock);
        let dmin = Self::read_dmin(superblock);
        let block_idx = i / 32;
        let scale = Self::extract_subblock_scale(superblock, block_idx);
        let min = Self::extract_subblock_min(superblock, block_idx);

        // Extract 4-bit value from packed byte
        let byte_idx = i / 2;
        let byte = superblock[16 + byte_idx];
        let q = if i.is_multiple_of(2) {
            (byte & 0x0F) as i32
        } else {
            ((byte >> 4) & 0x0F) as i32
        };

        let result = d * scale * (q as f32) - dmin * min;
        contract_post_dequantization!(&result);
        result
    }
}

// =============================================================================
// Q5_K Implementation
// =============================================================================

/// Q5_K format: 5-bit K-quantization with 256-element super-blocks
///
/// Layout: d(2) + dmin(2) + scales(12) + qh(32) + qs(128) = 176 bytes
/// Dequant: d * s_j * q_i - dmin * m_j where q_i = low4 | (high1 << 4)
pub struct Q5K;

impl QuantBlockFormat for Q5K {
    const FORMAT_ID: &'static str = "Q5_K";
    const FAMILY: QuantFamily = QuantFamily::KQuant;
    const ELEMENTS_PER_SUPERBLOCK: usize = 256;
    const SUBBLOCKS_PER_SUPERBLOCK: usize = 8;
    const ELEMENTS_PER_SUBBLOCK: usize = 32;
    const SUPERBLOCK_BYTES: usize = 176;
    const QUANT_BITS: u8 = 5;
    const HAS_DMIN: bool = true;
    const QUANT_SIGNED: bool = false;
    const ZERO_OFFSET: i32 = 0;
    const ULP_TOLERANCE: u32 = 8;
    const BITS_PER_WEIGHT: f32 = 5.5;
    const D_OFFSET: usize = 0;
    const DMIN_OFFSET: usize = 2;
    const SCALES_OFFSET: usize = 4;
    const SCALES_BYTES: usize = 12;
    const QS_OFFSET: usize = 48; // After qh (32 bytes at offset 16)
    const QS_BYTES: usize = 128;

    #[inline]
    fn read_d(superblock: &[u8]) -> f32 {
        read_f16(&superblock[0..2])
    }

    #[inline]
    fn read_dmin(superblock: &[u8]) -> f32 {
        read_f16(&superblock[2..4])
    }

    #[inline]
    fn extract_subblock_scale(superblock: &[u8], idx: usize) -> f32 {
        let mut scales = [0u8; 12];
        scales.copy_from_slice(&superblock[4..16]);
        let (scale, _min) = extract_scale_min(&scales, idx);
        scale
    }

    #[inline]
    fn extract_subblock_min(superblock: &[u8], idx: usize) -> f32 {
        let mut scales = [0u8; 12];
        scales.copy_from_slice(&superblock[4..16]);
        let (_scale, min) = extract_scale_min(&scales, idx);
        min
    }

    #[inline]
    fn dequant_value(superblock: &[u8], i: usize) -> f32 {
        let d = Self::read_d(superblock);
        let dmin = Self::read_dmin(superblock);
        let block_idx = i / 32;
        let scale = Self::extract_subblock_scale(superblock, block_idx);
        let min = Self::extract_subblock_min(superblock, block_idx);

        // Extract 5-bit value: 4 low bits from qs + 1 high bit from qh
        let byte_idx = i / 2;
        let qs_byte = superblock[48 + byte_idx];
        let q_low = if i.is_multiple_of(2) {
            qs_byte & 0x0F
        } else {
            (qs_byte >> 4) & 0x0F
        };

        // High bit from qh array
        let qh_byte_idx = i / 8;
        let qh_bit_offset = i % 8;
        let qh_byte = superblock[16 + qh_byte_idx];
        let q_high = (qh_byte >> qh_bit_offset) & 0x01;

        let q = ((q_high << 4) | q_low) as i32;
        d * scale * (q as f32) - dmin * min
    }
}

// =============================================================================
// Q6_K Implementation
// =============================================================================

/// Q6_K format: 6-bit K-quantization with 256-element super-blocks
///
/// Layout: ql(128) + qh(64) + scales(16) + d(2) = 210 bytes
/// Dequant: d * sc[j] * (q_i - 32)
///
/// Note: Q6_K has NO dmin. Values are centered by subtracting 32.
pub struct Q6K;

impl QuantBlockFormat for Q6K {
    const FORMAT_ID: &'static str = "Q6_K";
    const FAMILY: QuantFamily = QuantFamily::KQuant;
    const ELEMENTS_PER_SUPERBLOCK: usize = 256;
    const SUBBLOCKS_PER_SUPERBLOCK: usize = 16;
    const ELEMENTS_PER_SUBBLOCK: usize = 16;
    const SUPERBLOCK_BYTES: usize = 210;
    const QUANT_BITS: u8 = 6;
    const HAS_DMIN: bool = false;
    const QUANT_SIGNED: bool = true;
    const ZERO_OFFSET: i32 = 32;
    const ULP_TOLERANCE: u32 = 8;
    const BITS_PER_WEIGHT: f32 = 6.5625;
    const D_OFFSET: usize = 208; // d is at the END for Q6_K
    const DMIN_OFFSET: usize = 0; // No dmin
    const SCALES_OFFSET: usize = 192;
    const SCALES_BYTES: usize = 16;
    const QS_OFFSET: usize = 0; // ql starts at offset 0
    const QS_BYTES: usize = 128;

    #[inline]
    fn read_d(superblock: &[u8]) -> f32 {
        read_f16(&superblock[208..210])
    }

    #[inline]
    fn read_dmin(_superblock: &[u8]) -> f32 {
        0.0 // Q6_K has no dmin
    }

    #[inline]
    fn extract_subblock_scale(superblock: &[u8], idx: usize) -> f32 {
        // Q6_K uses direct i8 scales (16 bytes at offset 192)
        #[allow(clippy::cast_possible_wrap)]
        let scale = superblock[192 + idx] as i8;
        f32::from(scale)
    }

    #[inline]
    fn extract_subblock_min(_superblock: &[u8], _idx: usize) -> f32 {
        0.0 // Q6_K has no min
    }

    #[inline]
    #[allow(clippy::many_single_char_names)]
    fn dequant_value(superblock: &[u8], i: usize) -> f32 {
        let d = Self::read_d(superblock);

        // Q6_K layout: ql[128] + qh[64] + scales[16] + d[2]
        // Values are processed in 128-value halves (half=0, half=128)
        let half_offset = (i / 128) * 128;
        let pos = i % 32; // Position within the 32-value group
        let group = (i % 128) / 32; // Which group of 32 within the 128

        let half_idx = half_offset / 128;
        let ql_slice_start = 64 * half_idx;
        let qh_slice_start = 32 * half_idx;

        let scale_sel = pos / 16; // Scale index selector within group

        // Extract 6-bit value from ql + qh
        let quant = match group {
            0 => {
                let ql_val = superblock[ql_slice_start + pos] & 0xF;
                let qh_val = (superblock[128 + qh_slice_start + pos] & 3) << 4;
                (ql_val | qh_val) as i32 - 32
            },
            1 => {
                let ql_val = superblock[ql_slice_start + pos + 32] & 0xF;
                let qh_val = ((superblock[128 + qh_slice_start + pos] >> 2) & 3) << 4;
                (ql_val | qh_val) as i32 - 32
            },
            2 => {
                let ql_val = superblock[ql_slice_start + pos] >> 4;
                let qh_val = ((superblock[128 + qh_slice_start + pos] >> 4) & 3) << 4;
                (ql_val | qh_val) as i32 - 32
            },
            3 => {
                let ql_val = superblock[ql_slice_start + pos + 32] >> 4;
                let qh_val = ((superblock[128 + qh_slice_start + pos] >> 6) & 3) << 4;
                (ql_val | qh_val) as i32 - 32
            },
            _ => unreachable!(),
        };

        // Scale index: 8*half_idx + scale_sel + 2*group for the scale table
        let scale_idx = 8 * half_idx + scale_sel + 2 * group;
        #[allow(clippy::cast_possible_wrap)]
        let sc = superblock[192 + scale_idx] as i8;

        d * f32::from(sc) * (quant as f32)
    }
}

// =============================================================================
// Q4_0 Implementation
// =============================================================================

/// Q4_0 format: Simple 4-bit quantization with 32-element blocks
///
/// Layout: d(2) + qs(16) = 18 bytes
/// Dequant: scale * (q_i - 8)
pub struct Q4_0Fmt;

impl QuantBlockFormat for Q4_0Fmt {
    const FORMAT_ID: &'static str = "Q4_0";
    const FAMILY: QuantFamily = QuantFamily::Simple;
    const ELEMENTS_PER_SUPERBLOCK: usize = 32;
    const SUBBLOCKS_PER_SUPERBLOCK: usize = 1;
    const ELEMENTS_PER_SUBBLOCK: usize = 32;
    const SUPERBLOCK_BYTES: usize = 18;
    const QUANT_BITS: u8 = 4;
    const HAS_DMIN: bool = false;
    const QUANT_SIGNED: bool = false;
    const ZERO_OFFSET: i32 = 8;
    const ULP_TOLERANCE: u32 = 4;
    const BITS_PER_WEIGHT: f32 = 4.0;
    const D_OFFSET: usize = 0;
    const DMIN_OFFSET: usize = 0;
    const SCALES_OFFSET: usize = 0; // No separate scales
    const SCALES_BYTES: usize = 0;
    const QS_OFFSET: usize = 2;
    const QS_BYTES: usize = 16;

    #[inline]
    fn read_d(superblock: &[u8]) -> f32 {
        read_f16(&superblock[0..2])
    }

    #[inline]
    fn read_dmin(_superblock: &[u8]) -> f32 {
        0.0
    }

    #[inline]
    fn extract_subblock_scale(_superblock: &[u8], _idx: usize) -> f32 {
        1.0 // Single scale is the d value itself; sub-block scale is 1.0
    }

    #[inline]
    fn extract_subblock_min(_superblock: &[u8], _idx: usize) -> f32 {
        0.0
    }

    #[inline]
    fn dequant_value(superblock: &[u8], i: usize) -> f32 {
        let scale = Self::read_d(superblock);
        let byte_idx = i / 2;
        let byte = superblock[2 + byte_idx];
        let q = if i.is_multiple_of(2) {
            (byte & 0x0F) as i32
        } else {
            ((byte >> 4) & 0x0F) as i32
        };
        scale * ((q - 8) as f32)
    }
}

// =============================================================================
// Q8_0 Implementation
// =============================================================================

/// Q8_0 format: Simple 8-bit quantization with 32-element blocks
///
/// Layout: d(2) + qs(32) = 34 bytes
/// Dequant: scale * q_i (signed i8)
pub struct Q8_0Fmt;

impl QuantBlockFormat for Q8_0Fmt {
    const FORMAT_ID: &'static str = "Q8_0";
    const FAMILY: QuantFamily = QuantFamily::Simple;
    const ELEMENTS_PER_SUPERBLOCK: usize = 32;
    const SUBBLOCKS_PER_SUPERBLOCK: usize = 1;
    const ELEMENTS_PER_SUBBLOCK: usize = 32;
    const SUPERBLOCK_BYTES: usize = 34;
    const QUANT_BITS: u8 = 8;
    const HAS_DMIN: bool = false;
    const QUANT_SIGNED: bool = true;
    const ZERO_OFFSET: i32 = 0;
    const ULP_TOLERANCE: u32 = 2;
    const BITS_PER_WEIGHT: f32 = 8.0;
    const D_OFFSET: usize = 0;
    const DMIN_OFFSET: usize = 0;
    const SCALES_OFFSET: usize = 0;
    const SCALES_BYTES: usize = 0;
    const QS_OFFSET: usize = 2;
    const QS_BYTES: usize = 32;

    #[inline]
    fn read_d(superblock: &[u8]) -> f32 {
        read_f16(&superblock[0..2])
    }

    #[inline]
    fn read_dmin(_superblock: &[u8]) -> f32 {
        0.0
    }

    #[inline]
    fn extract_subblock_scale(_superblock: &[u8], _idx: usize) -> f32 {
        1.0
    }

    #[inline]
    fn extract_subblock_min(_superblock: &[u8], _idx: usize) -> f32 {
        0.0
    }

    #[inline]
    fn dequant_value(superblock: &[u8], i: usize) -> f32 {
        let scale = Self::read_d(superblock);
        #[allow(clippy::cast_possible_wrap)]
        let q = superblock[2 + i] as i8;
        scale * f32::from(q)
    }
}

// =============================================================================
// Format Registry (for exhaustiveness checks)
// =============================================================================

/// All format IDs known to the contract.
/// Used by FALSIFY-QDOT-004 to verify completeness.
pub const ALL_FORMAT_IDS: &[&str] = &["Q4_K", "Q5_K", "Q6_K", "Q4_0", "Q8_0"];

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // Compile-time constant verification (YAML parity)
    // =========================================================================

    #[test]
    fn test_q4k_constants_match_yaml() {
        assert_eq!(Q4K::FORMAT_ID, "Q4_K");
        assert_eq!(Q4K::FAMILY, QuantFamily::KQuant);
        assert_eq!(Q4K::ELEMENTS_PER_SUPERBLOCK, 256);
        assert_eq!(Q4K::SUBBLOCKS_PER_SUPERBLOCK, 8);
        assert_eq!(Q4K::ELEMENTS_PER_SUBBLOCK, 32);
        assert_eq!(Q4K::SUPERBLOCK_BYTES, 144);
        assert_eq!(Q4K::QUANT_BITS, 4);
        assert!(Q4K::HAS_DMIN);
        assert!(!Q4K::QUANT_SIGNED);
        assert_eq!(Q4K::ZERO_OFFSET, 0);
    }

    #[test]
    fn test_q5k_constants_match_yaml() {
        assert_eq!(Q5K::FORMAT_ID, "Q5_K");
        assert_eq!(Q5K::SUPERBLOCK_BYTES, 176);
        assert_eq!(Q5K::QUANT_BITS, 5);
        assert!(Q5K::HAS_DMIN);
        assert_eq!(Q5K::SUBBLOCKS_PER_SUPERBLOCK, 8);
    }

    #[test]
    fn test_q6k_constants_match_yaml() {
        assert_eq!(Q6K::FORMAT_ID, "Q6_K");
        assert_eq!(Q6K::SUPERBLOCK_BYTES, 210);
        assert_eq!(Q6K::QUANT_BITS, 6);
        assert!(!Q6K::HAS_DMIN);
        assert!(Q6K::QUANT_SIGNED);
        assert_eq!(Q6K::ZERO_OFFSET, 32);
        assert_eq!(Q6K::SUBBLOCKS_PER_SUPERBLOCK, 16);
        assert_eq!(Q6K::ELEMENTS_PER_SUBBLOCK, 16);
    }

    #[test]
    fn test_q4_0_constants_match_yaml() {
        assert_eq!(Q4_0Fmt::FORMAT_ID, "Q4_0");
        assert_eq!(Q4_0Fmt::FAMILY, QuantFamily::Simple);
        assert_eq!(Q4_0Fmt::SUPERBLOCK_BYTES, 18);
        assert_eq!(Q4_0Fmt::ELEMENTS_PER_SUPERBLOCK, 32);
        assert!(!Q4_0Fmt::HAS_DMIN);
        assert_eq!(Q4_0Fmt::ZERO_OFFSET, 8);
    }

    #[test]
    fn test_q8_0_constants_match_yaml() {
        assert_eq!(Q8_0Fmt::FORMAT_ID, "Q8_0");
        assert_eq!(Q8_0Fmt::FAMILY, QuantFamily::Simple);
        assert_eq!(Q8_0Fmt::SUPERBLOCK_BYTES, 34);
        assert_eq!(Q8_0Fmt::ELEMENTS_PER_SUPERBLOCK, 32);
        assert!(Q8_0Fmt::QUANT_SIGNED);
        assert_eq!(Q8_0Fmt::ZERO_OFFSET, 0);
    }

    // =========================================================================
    // Validation tests
    // =========================================================================

    #[test]
    fn test_validate_data_length_q4k() {
        // Valid: exactly 1 super-block
        assert_eq!(Q4K::validate_data_length(&[0u8; 144]).ok(), Some(1));
        // Valid: 2 super-blocks
        assert_eq!(Q4K::validate_data_length(&[0u8; 288]).ok(), Some(2));
        // Invalid: not a multiple
        assert!(Q4K::validate_data_length(&[0u8; 100]).is_err());
        // Invalid: empty
        assert!(Q4K::validate_data_length(&[]).is_err());
    }

    #[test]
    fn test_validate_data_length_q6k() {
        assert_eq!(Q6K::validate_data_length(&[0u8; 210]).ok(), Some(1));
        assert!(Q6K::validate_data_length(&[0u8; 100]).is_err());
    }

    #[test]
    fn test_validate_data_length_q8_0() {
        assert_eq!(Q8_0Fmt::validate_data_length(&[0u8; 34]).ok(), Some(1));
        assert_eq!(Q8_0Fmt::validate_data_length(&[0u8; 68]).ok(), Some(2));
    }

    // =========================================================================
    // Dequantization correctness
    // =========================================================================

    #[test]
    fn test_q4_0_dequant_value_zero_scale() {
        // f16 zero = 0x0000
        let mut block = [0u8; 18];
        block[0] = 0; // scale = 0.0 (f16 zero)
        block[1] = 0;
        // All quant values = 0 (which means value = 0 * (0 - 8) = 0)
        let val = Q4_0Fmt::dequant_value(&block, 0);
        assert_eq!(val, 0.0);
    }

    #[test]
    fn test_q8_0_dequant_value() {
        // Construct a Q8_0 block with known scale and quants
        let mut block = [0u8; 34];
        // f16 for 1.0 = 0x3C00
        block[0] = 0x00;
        block[1] = 0x3C;
        // Set first quant to 42
        #[allow(clippy::cast_sign_loss)]
        {
            block[2] = 42u8; // i8 = 42
        }
        let val = Q8_0Fmt::dequant_value(&block, 0);
        assert!((val - 42.0).abs() < 0.01, "Expected ~42.0, got {val}");
    }

    // =========================================================================
    // Format registry completeness
    // =========================================================================

    #[test]
    fn test_all_format_ids_are_unique() {
        let mut seen = std::collections::HashSet::new();
        for &id in ALL_FORMAT_IDS {
            assert!(seen.insert(id), "Duplicate format ID: {id}");
        }
    }

    #[test]
    fn test_format_count() {
        assert_eq!(ALL_FORMAT_IDS.len(), 5, "Expected 5 formats in registry");
    }

    // =========================================================================
    // Structural invariants
    // =========================================================================

    #[test]
    fn test_elements_equals_subblocks_times_elements_per_subblock() {
        assert_eq!(
            Q4K::ELEMENTS_PER_SUPERBLOCK,
            Q4K::SUBBLOCKS_PER_SUPERBLOCK * Q4K::ELEMENTS_PER_SUBBLOCK
        );
        assert_eq!(
            Q5K::ELEMENTS_PER_SUPERBLOCK,
            Q5K::SUBBLOCKS_PER_SUPERBLOCK * Q5K::ELEMENTS_PER_SUBBLOCK
        );
        assert_eq!(
            Q6K::ELEMENTS_PER_SUPERBLOCK,
            Q6K::SUBBLOCKS_PER_SUPERBLOCK * Q6K::ELEMENTS_PER_SUBBLOCK
        );
        assert_eq!(
            Q4_0Fmt::ELEMENTS_PER_SUPERBLOCK,
            Q4_0Fmt::SUBBLOCKS_PER_SUPERBLOCK * Q4_0Fmt::ELEMENTS_PER_SUBBLOCK
        );
        assert_eq!(
            Q8_0Fmt::ELEMENTS_PER_SUPERBLOCK,
            Q8_0Fmt::SUBBLOCKS_PER_SUPERBLOCK * Q8_0Fmt::ELEMENTS_PER_SUBBLOCK
        );
    }

    #[test]
    fn test_kquant_formats_have_256_elements() {
        assert_eq!(Q4K::ELEMENTS_PER_SUPERBLOCK, 256);
        assert_eq!(Q5K::ELEMENTS_PER_SUPERBLOCK, 256);
        assert_eq!(Q6K::ELEMENTS_PER_SUPERBLOCK, 256);
    }

    #[test]
    fn test_simple_formats_have_32_elements() {
        assert_eq!(Q4_0Fmt::ELEMENTS_PER_SUPERBLOCK, 32);
        assert_eq!(Q8_0Fmt::ELEMENTS_PER_SUPERBLOCK, 32);
    }
}
