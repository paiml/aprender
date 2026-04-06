//! Quantization SIMD Helpers (PMAT-802)
//!
//! Extracted from quantize/mod.rs - Shared SIMD utility functions.
//!
//! ## Contents
//! - f16 reading: `read_f16`
//! - Scale extraction: `extract_scale_min`, `extract_scale_min_from_slice`
//!
//! Note: `f16_to_f32` is exported from `dequant.rs`.
//! SIMD activations (`softmax_simd`, `fused_swiglu_simd`, `apply_rope_rotation_simd`)
//! are exported from `activation.rs` and `parallel_dequant.rs` respectively.
//!
//! Horizontal sum helpers (hsum_epi32_*, horizontal_sum_*) are internal to
//! the files that use them (fused_k.rs, etc.) to avoid dead code.

// ============================================================================
// f16 Conversion
// ============================================================================

/// Helper: Read f16 from bytes and convert to f32
#[inline]
pub fn read_f16(bytes: &[u8]) -> f32 {
    let bits = u16::from_le_bytes([bytes[0], bytes[1]]);
    half::f16::from_bits(bits).to_f32()
}

// ============================================================================
// Scale Extraction for K-Quantization
// ============================================================================

/// Extract 6-bit scale and min values from packed scales array
///
/// PAR-001 FIX: Matches llama.cpp's get_scale_min_k4 packing scheme:
/// - Blocks 0-3: scale = q[j] & 63, min = q[j+4] & 63
/// - Blocks 4-7: scale = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4)
///   min = (q[j+4] >> 4) | ((q[j] >> 6) << 4)
#[inline]
pub fn extract_scale_min(scales: &[u8; 12], block_idx: usize) -> (f32, f32) {
    let j = block_idx;
    let (scale_bits, min_bits) = if j < 4 {
        // First 4 blocks: simple layout
        let d = scales[j] & 63;
        let m = scales[j + 4] & 63;
        (d, m)
    } else {
        // Last 4 blocks: packed layout using high bits from first 4 bytes
        let d = (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4);
        let m = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
        (d, m)
    };

    // Return raw 6-bit values as floats
    // The GGUF header's d/dmin values already include the /63 normalization
    let scale = f32::from(scale_bits);
    let min = f32::from(min_bits);

    (scale, min)
}

/// Extract scale and min from packed 6-bit scales (helper for InterleavedQ4K)
pub fn extract_scale_min_from_slice(scales: &[u8], idx: usize) -> (f32, f32) {
    // Same logic as extract_scale_min but works with slice
    let scale_idx = idx / 2;
    let min_idx = idx / 2 + 4;

    let (scale_raw, min_raw) = if idx.is_multiple_of(2) {
        (scales[scale_idx] & 0x3F, scales[min_idx] & 0x3F)
    } else {
        (
            (scales[scale_idx] >> 6) | ((scales[scale_idx + 2] & 0x0F) << 2),
            (scales[min_idx] >> 6) | ((scales[min_idx + 2] & 0x0F) << 2),
        )
    };

    (scale_raw as f32, min_raw as f32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quantize::f16_to_f32;

    // ============= f16_to_f32 tests =============

    #[test]
    fn test_f16_to_f32_zero() {
        assert_eq!(f16_to_f32(0x0000), 0.0);
    }

    #[test]
    fn test_f16_to_f32_negative_zero() {
        let val = f16_to_f32(0x8000);
        assert!(val == 0.0 && val.is_sign_negative());
    }

    #[test]
    fn test_f16_to_f32_one() {
        let val = f16_to_f32(0x3C00);
        assert!((val - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_f16_to_f32_negative_one() {
        let val = f16_to_f32(0xBC00);
        assert!((val - (-1.0)).abs() < 0.001);
    }

    #[test]
    fn test_f16_to_f32_two() {
        let val = f16_to_f32(0x4000);
        assert!((val - 2.0).abs() < 0.001);
    }

    #[test]
    fn test_f16_to_f32_half() {
        let val = f16_to_f32(0x3800);
        assert!((val - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_f16_to_f32_infinity() {
        assert_eq!(f16_to_f32(0x7C00), f32::INFINITY);
    }

    #[test]
    fn test_f16_to_f32_neg_infinity() {
        assert_eq!(f16_to_f32(0xFC00), f32::NEG_INFINITY);
    }

    #[test]
    fn test_f16_to_f32_nan() {
        assert!(f16_to_f32(0x7C01).is_nan());
    }

    #[test]
    fn test_f16_to_f32_subnormal() {
        // Smallest positive subnormal: 2^-24
        let val = f16_to_f32(0x0001);
        assert!(val > 0.0);
        assert!(val < 0.0001);
    }

    // ============= read_f16 tests =============

    #[test]
    fn test_read_f16_one() {
        let bytes = 0x3C00u16.to_le_bytes();
        let val = read_f16(&bytes);
        assert!((val - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_read_f16_two() {
        let bytes = 0x4000u16.to_le_bytes();
        let val = read_f16(&bytes);
        assert!((val - 2.0).abs() < 0.001);
    }

    // ============= extract_scale_min tests =============

    #[test]
    fn test_extract_scale_min_first_block() {
        let scales: [u8; 12] = [10, 20, 30, 40, 5, 15, 25, 35, 0, 0, 0, 0];
        let (scale, min) = extract_scale_min(&scales, 0);
        assert_eq!(scale, 10.0); // scales[0] & 63
        assert_eq!(min, 5.0); // scales[4] & 63
    }

    #[test]
    fn test_extract_scale_min_second_block() {
        let scales: [u8; 12] = [10, 20, 30, 40, 5, 15, 25, 35, 0, 0, 0, 0];
        let (scale, min) = extract_scale_min(&scales, 1);
        assert_eq!(scale, 20.0);
        assert_eq!(min, 15.0);
    }

    #[test]
    fn test_extract_scale_min_packed_block() {
        // Block 4+ uses packed layout
        let mut scales: [u8; 12] = [0; 12];
        scales[0] = 0b11_000010; // high bits for block 4
        scales[8] = 0b0011_0001; // low bits: scale=0x31, min high nibble=0x3

        let (scale, min) = extract_scale_min(&scales, 4);
        // scale = (scales[8] & 0x0F) | ((scales[0] >> 6) << 4) = 1 | (3 << 4) = 1 | 48 = 49
        // min = (scales[8] >> 4) | ((scales[4] >> 6) << 4) = 3 | 0 = 3
        assert_eq!(scale, 49.0);
        assert_eq!(min, 3.0);
    }

    // ============= extract_scale_min_from_slice tests =============

    #[test]
    fn test_extract_scale_min_from_slice_even() {
        let scales: [u8; 8] = [10, 20, 30, 40, 5, 15, 25, 35];
        let (scale, min) = extract_scale_min_from_slice(&scales, 0);
        assert_eq!(scale, 10.0);
        assert_eq!(min, 5.0);
    }
}
