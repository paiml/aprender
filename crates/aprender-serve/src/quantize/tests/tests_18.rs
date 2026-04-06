//! Part 18: SIMD Coverage Enhancement Tests
//!
//! This module targets uncovered code paths in `src/quantize/simd.rs`:
//! - Horizontal sum helper functions (x86_64 specific)
//! - AVX2 SIMD remainder loops in softmax, swiglu, rope
//! - Scalar fallback edge cases
//! - f16 conversion special cases
//! - extract_scale_min_from_slice odd index paths
//!
//! Focus: Edge cases, alignment requirements, and fallback paths.

use crate::quantize::simd::{extract_scale_min, extract_scale_min_from_slice, read_f16};
use crate::quantize::{f16_to_f32, fused_swiglu_simd, softmax_simd};

// =============================================================================
// f16 Conversion: Complete Branch Coverage
// =============================================================================

/// Test f16_to_f32 with sign=1, exp=0, mantissa=0 (negative zero)
#[test]
fn test_f16_to_f32_negative_zero_exact() {
    // 0x8000 = negative zero (sign=1, exp=0, mantissa=0)
    let result = f16_to_f32(0x8000);
    // Should be -0.0 or 0.0 (both are valid)
    assert!(
        result == 0.0 || result == -0.0,
        "Negative zero: got {}",
        result
    );
    // Check sign bit if -0.0
    if result.is_sign_negative() {
        assert_eq!(result, -0.0);
    }
}

/// Test f16_to_f32 with sign=1, exp=0, mantissa!=0 (negative subnormal)
#[test]
fn test_f16_to_f32_negative_subnormal_complete() {
    // Test various negative subnormal patterns
    // 0x8001 = negative smallest subnormal
    let result = f16_to_f32(0x8001);
    assert!(
        result < 0.0,
        "Negative subnormal should be negative: {}",
        result
    );
    let expected = -(1.0 / 1024.0) * (2.0_f32).powi(-14);
    assert!(
        (result - expected).abs() < 1e-12,
        "Negative subnormal 0x8001: got {}, expected {}",
        result,
        expected
    );

    // 0x8100 = negative subnormal with mantissa=256
    let result = f16_to_f32(0x8100);
    assert!(result < 0.0, "Negative subnormal 0x8100 should be negative");
    let expected = -(256.0 / 1024.0) * (2.0_f32).powi(-14);
    assert!(
        (result - expected).abs() < 1e-12,
        "Negative subnormal 0x8100: got {}, expected {}",
        result,
        expected
    );

    // 0x83FF = negative largest subnormal
    let result = f16_to_f32(0x83FF);
    assert!(result < 0.0, "Negative max subnormal should be negative");
    let expected = -(1023.0 / 1024.0) * (2.0_f32).powi(-14);
    assert!(
        (result - expected).abs() < 1e-12,
        "Negative max subnormal: got {}, expected {}",
        result,
        expected
    );
}

/// Test f16_to_f32 with sign=1, exp in [1,30], mantissa (negative normal)
#[test]
fn test_f16_to_f32_negative_normal_complete() {
    // -1.5: sign=1, exp=15, mantissa=512 -> (1 + 512/1024) * 2^0 = 1.5
    // Bits: 1_01111_1000000000 = 0xBE00
    let result = f16_to_f32(0xBE00);
    assert!(
        (result - (-1.5)).abs() < 1e-3,
        "-1.5 conversion: got {}",
        result
    );

    // -3.5: sign=1, exp=16, mantissa=768 -> (1 + 768/1024) * 2^1 = 3.5
    // Bits: 1_10000_1100000000 = 0xC300
    let result = f16_to_f32(0xC300);
    assert!(
        (result - (-3.5)).abs() < 1e-3,
        "-3.5 conversion: got {}",
        result
    );

    // -0.125: sign=1, exp=12, mantissa=0 -> 1.0 * 2^-3 = 0.125
    // Bits: 1_01100_0000000000 = 0xB000
    let result = f16_to_f32(0xB000);
    assert!(
        (result - (-0.125)).abs() < 1e-4,
        "-0.125 conversion: got {}",
        result
    );
}

/// Test f16_to_f32 with sign=0, exp=31, mantissa=0 (positive infinity)
#[test]
fn test_f16_to_f32_positive_infinity_direct() {
    let result = f16_to_f32(0x7C00);
    assert!(result.is_infinite(), "Should be infinite");
    assert!(result > 0.0, "Should be positive infinity");
    assert_eq!(result, f32::INFINITY);
}

/// Test f16_to_f32 with sign=1, exp=31, mantissa=0 (negative infinity)
#[test]
fn test_f16_to_f32_negative_infinity_direct() {
    let result = f16_to_f32(0xFC00);
    assert!(result.is_infinite(), "Should be infinite");
    assert!(result < 0.0, "Should be negative infinity");
    assert_eq!(result, f32::NEG_INFINITY);
}

/// Test f16_to_f32 with exp=31, mantissa!=0 (NaN variants)
#[test]
fn test_f16_to_f32_nan_complete_coverage() {
    // Various NaN patterns
    let nan_patterns: &[u16] = &[
        0x7C01, // Positive quiet NaN (min mantissa)
        0x7DFF, // Positive quiet NaN (near max mantissa)
        0x7FFF, // Positive quiet NaN (max mantissa)
        0xFC01, // Negative quiet NaN (min mantissa)
        0xFDFF, // Negative quiet NaN
        0xFFFF, // Negative quiet NaN (max mantissa)
    ];

    for &pattern in nan_patterns {
        let result = f16_to_f32(pattern);
        assert!(
            result.is_nan(),
            "Pattern 0x{:04X} should be NaN, got {}",
            pattern,
            result
        );
    }
}

/// Test read_f16 with various special values
#[test]
fn test_read_f16_special_values() {
    // Positive infinity
    let bytes = 0x7C00u16.to_le_bytes();
    let result = read_f16(&bytes);
    assert!(
        result.is_infinite() && result > 0.0,
        "read_f16(+inf): got {}",
        result
    );

    // Negative infinity
    let bytes = 0xFC00u16.to_le_bytes();
    let result = read_f16(&bytes);
    assert!(
        result.is_infinite() && result < 0.0,
        "read_f16(-inf): got {}",
        result
    );

    // NaN
    let bytes = 0x7C01u16.to_le_bytes();
    let result = read_f16(&bytes);
    assert!(result.is_nan(), "read_f16(NaN): got {}", result);

    // Negative zero
    let bytes = 0x8000u16.to_le_bytes();
    let result = read_f16(&bytes);
    assert!(result == 0.0, "read_f16(-0): got {}", result);
}

// =============================================================================
// extract_scale_min_from_slice: Complete Branch Coverage
// =============================================================================

/// Test extract_scale_min_from_slice with all even indices 0-6
#[test]
fn test_extract_scale_min_from_slice_all_even() {
    let scales: [u8; 12] = [
        1, 2, 3, 4, // bytes 0-3 for scale indices
        11, 12, 13, 14, // bytes 4-7 for min indices
        0, 0, 0, 0,
    ];

    // idx=0: scale_idx=0, min_idx=4
    let (s0, m0) = extract_scale_min_from_slice(&scales, 0);
    assert_eq!(s0, 1.0, "idx=0 scale");
    assert_eq!(m0, 11.0, "idx=0 min");

    // idx=2: scale_idx=1, min_idx=5
    let (s2, m2) = extract_scale_min_from_slice(&scales, 2);
    assert_eq!(s2, 2.0, "idx=2 scale");
    assert_eq!(m2, 12.0, "idx=2 min");

    // idx=4: scale_idx=2, min_idx=6
    let (s4, m4) = extract_scale_min_from_slice(&scales, 4);
    assert_eq!(s4, 3.0, "idx=4 scale");
    assert_eq!(m4, 13.0, "idx=4 min");

    // idx=6: scale_idx=3, min_idx=7
    let (s6, m6) = extract_scale_min_from_slice(&scales, 6);
    assert_eq!(s6, 4.0, "idx=6 scale");
    assert_eq!(m6, 14.0, "idx=6 min");
}

/// Test extract_scale_min_from_slice with all odd indices 1,3,5,7
#[test]
fn test_extract_scale_min_from_slice_all_odd() {
    // For odd indices, the formula uses different bit extraction:
    // scale = (scales[scale_idx] >> 6) | ((scales[scale_idx + 2] & 0x0F) << 2)
    // min = (scales[min_idx] >> 6) | ((scales[min_idx + 2] & 0x0F) << 2)
    let scales: [u8; 12] = [
        0b11_000000, // byte 0: high bits = 3 for scale idx=1
        0b10_000000, // byte 1: high bits = 2 for scale idx=3
        0b00000101,  // byte 2: low nibble = 5 for scale idx=1
        0b00000110,  // byte 3: low nibble = 6 for scale idx=3
        0b01_000000, // byte 4: high bits = 1 for min idx=1
        0b00_000000, // byte 5: high bits = 0 for min idx=3
        0b00000111,  // byte 6: low nibble = 7 for min idx=1
        0b00001000,  // byte 7: low nibble = 8 for min idx=3
        0,
        0,
        0,
        0,
    ];

    // idx=1: scale = (byte0 >> 6) | ((byte2 & 0x0F) << 2) = 3 | (5 << 2) = 3 | 20 = 23
    //        min = (byte4 >> 6) | ((byte6 & 0x0F) << 2) = 1 | (7 << 2) = 1 | 28 = 29
    let (s1, m1) = extract_scale_min_from_slice(&scales, 1);
    assert_eq!(s1, 23.0, "idx=1 scale");
    assert_eq!(m1, 29.0, "idx=1 min");

    // idx=3: scale = (byte1 >> 6) | ((byte3 & 0x0F) << 2) = 2 | (6 << 2) = 2 | 24 = 26
    //        min = (byte5 >> 6) | ((byte7 & 0x0F) << 2) = 0 | (8 << 2) = 0 | 32 = 32
    let (s3, m3) = extract_scale_min_from_slice(&scales, 3);
    assert_eq!(s3, 26.0, "idx=3 scale");
    assert_eq!(m3, 32.0, "idx=3 min");
}

/// Test extract_scale_min_from_slice boundary with max 6-bit values
#[test]
fn test_extract_scale_min_from_slice_max_values() {
    // Set up for idx=0 to return max value 63 (all 1s in low 6 bits)
    let scales: [u8; 12] = [
        0b00_111111, // byte 0: 63 for scale idx=0
        0,
        0,
        0,
        0b00_111111, // byte 4: 63 for min idx=0
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ];

    let (s0, m0) = extract_scale_min_from_slice(&scales, 0);
    assert_eq!(s0, 63.0, "idx=0 max scale");
    assert_eq!(m0, 63.0, "idx=0 max min");
}

// =============================================================================
// extract_scale_min: Extended Coverage for Blocks 4-7
// =============================================================================

/// Test extract_scale_min block 4 with different high bit patterns
#[test]
fn test_extract_scale_min_block4_variations() {
    // Block 4: d = (scales[8] & 0x0F) | ((scales[0] >> 6) << 4)
    //          m = (scales[8] >> 4) | ((scales[4] >> 6) << 4)

    // Test with byte[0] high bits = 0, byte[4] high bits = 0
    let scales1: [u8; 12] = [
        0b00_000000, // byte 0
        0,
        0,
        0,
        0b00_000000, // byte 4
        0,
        0,
        0,
        0b0001_0010, // byte 8: low=2 (scale), high=1 (min)
        0,
        0,
        0,
    ];
    let (s4, m4) = extract_scale_min(&scales1, 4);
    assert_eq!(s4, 2.0, "Block 4 scale with zero high bits");
    assert_eq!(m4, 1.0, "Block 4 min with zero high bits");

    // Test with byte[0] high bits = 3, byte[4] high bits = 3
    let scales2: [u8; 12] = [
        0b11_000000, // byte 0: high bits = 3
        0,
        0,
        0,
        0b11_000000, // byte 4: high bits = 3
        0,
        0,
        0,
        0b0001_0010, // byte 8: low=2 (scale), high=1 (min)
        0,
        0,
        0,
    ];
    let (s4, m4) = extract_scale_min(&scales2, 4);
    // d = 2 | (3 << 4) = 2 | 48 = 50
    // m = 1 | (3 << 4) = 1 | 48 = 49
    assert_eq!(s4, 50.0, "Block 4 scale with max high bits");
    assert_eq!(m4, 49.0, "Block 4 min with max high bits");
}

/// Test extract_scale_min blocks 5, 6, 7 detailed
#[test]
fn test_extract_scale_min_blocks_5_6_7_detailed() {
    // Block 5: d = (scales[9] & 0x0F) | ((scales[1] >> 6) << 4)
    //          m = (scales[9] >> 4) | ((scales[5] >> 6) << 4)
    // Block 6: d = (scales[10] & 0x0F) | ((scales[2] >> 6) << 4)
    //          m = (scales[10] >> 4) | ((scales[6] >> 6) << 4)
    // Block 7: d = (scales[11] & 0x0F) | ((scales[3] >> 6) << 4)
    //          m = (scales[11] >> 4) | ((scales[7] >> 6) << 4)

    let scales: [u8; 12] = [
        0b00_111111, // byte 0
        0b01_111111, // byte 1: high bits = 1
        0b10_111111, // byte 2: high bits = 2
        0b11_111111, // byte 3: high bits = 3
        0b00_111111, // byte 4
        0b01_111111, // byte 5: high bits = 1
        0b10_111111, // byte 6: high bits = 2
        0b11_111111, // byte 7: high bits = 3
        0b0000_0000, // byte 8
        0b0010_0001, // byte 9: scale5=1, min5=2
        0b0100_0011, // byte 10: scale6=3, min6=4
        0b0110_0101, // byte 11: scale7=5, min7=6
    ];

    // Block 5: d = 1 | (1 << 4) = 17, m = 2 | (1 << 4) = 18
    let (s5, m5) = extract_scale_min(&scales, 5);
    assert_eq!(s5, 17.0, "Block 5 scale");
    assert_eq!(m5, 18.0, "Block 5 min");

    // Block 6: d = 3 | (2 << 4) = 35, m = 4 | (2 << 4) = 36
    let (s6, m6) = extract_scale_min(&scales, 6);
    assert_eq!(s6, 35.0, "Block 6 scale");
    assert_eq!(m6, 36.0, "Block 6 min");

    // Block 7: d = 5 | (3 << 4) = 53, m = 6 | (3 << 4) = 54
    let (s7, m7) = extract_scale_min(&scales, 7);
    assert_eq!(s7, 53.0, "Block 7 scale");
    assert_eq!(m7, 54.0, "Block 7 min");
}

// =============================================================================
// Softmax SIMD: Remainder Loop and Scalar Fallback Coverage
// =============================================================================

/// Test softmax with sizes that exercise remainder loops in AVX2
#[test]
fn test_softmax_simd_remainder_loops() {
    // Test sizes 8+1 through 8+7 to cover all remainder paths
    for remainder in 1..=7 {
        let size = 8 + remainder;
        let mut x: Vec<f32> = (0..size)
            .map(|i| (i as f32 - size as f32 / 2.0) * 0.2)
            .collect();

        softmax_simd(&mut x);

        let sum: f32 = x.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "Size {}: sum should be 1.0, got {}",
            size,
            sum
        );

        // All values should be non-negative
        for (i, v) in x.iter().enumerate() {
            assert!(*v >= 0.0, "Size {}: negative value at {}: {}", size, i, v);
        }
    }
}

/// Test softmax with sizes 16+1 through 16+7
#[test]
fn test_softmax_simd_two_chunks_remainder() {
    for remainder in 1..=7 {
        let size = 16 + remainder;
        let mut x: Vec<f32> = (0..size).map(|i| (i as f32) * 0.1).collect();

        softmax_simd(&mut x);

        let sum: f32 = x.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "Size {}: sum should be 1.0, got {}",
            size,
            sum
        );
    }
}

/// Test softmax with sizes 1-7 (scalar fallback only)
#[test]
fn test_softmax_simd_scalar_fallback_all() {
    for size in 1..=7 {
        let mut x: Vec<f32> = (0..size).map(|i| i as f32).collect();

        softmax_simd(&mut x);

        let sum: f32 = x.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "Size {}: sum should be 1.0, got {}",
            size,
            sum
        );
    }
}

include!("softmax_simd_03.rs");
