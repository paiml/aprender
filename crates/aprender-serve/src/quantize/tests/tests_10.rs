//! Phase 37: Additional SIMD Coverage Tests
//!
//! This module provides comprehensive tests for the SIMD helper functions
//! in `src/quantize/simd.rs` to achieve higher code coverage.
//!
//! Focus areas:
//! - SIMD dequantization functions with various input sizes
//! - Multi-block processing
//! - Aligned vs unaligned input sizes
//! - Boundary conditions for AVX2 code paths
//! - Scalar fallback paths
//! - Horizontal sum helpers (x86_64 specific)

use crate::quantize::simd::{extract_scale_min, extract_scale_min_from_slice, read_f16};
use crate::quantize::{f16_to_f32, fused_swiglu_simd, softmax_simd};

// =============================================================================
// F16 Conversion Edge Cases
// =============================================================================

#[test]
fn test_f16_to_f32_positive_subnormal_various() {
    // Test various subnormal patterns
    // Subnormal: exp=0, mantissa!=0
    // Value = (mantissa / 1024) * 2^-14

    // Smallest positive subnormal: 0x0001 = 1/1024 * 2^-14
    let result = f16_to_f32(0x0001);
    let expected = (1.0 / 1024.0) * (2.0_f32).powi(-14);
    assert!(
        (result - expected).abs() < 1e-10,
        "Smallest subnormal: got {}, expected {}",
        result,
        expected
    );

    // Larger subnormal: 0x03FF = 1023/1024 * 2^-14 (max subnormal)
    let result = f16_to_f32(0x03FF);
    let expected = (1023.0 / 1024.0) * (2.0_f32).powi(-14);
    assert!(
        (result - expected).abs() < 1e-10,
        "Max subnormal: got {}, expected {}",
        result,
        expected
    );

    // Mid subnormal: 0x0200 = 512/1024 * 2^-14
    let result = f16_to_f32(0x0200);
    let expected = (512.0 / 1024.0) * (2.0_f32).powi(-14);
    assert!(
        (result - expected).abs() < 1e-10,
        "Mid subnormal: got {}, expected {}",
        result,
        expected
    );
}

#[test]
fn test_f16_to_f32_negative_subnormal() {
    // Negative subnormal: sign=1, exp=0, mantissa!=0
    // 0x8001 = negative smallest subnormal
    let result = f16_to_f32(0x8001);
    let expected = -(1.0 / 1024.0) * (2.0_f32).powi(-14);
    assert!(
        (result - expected).abs() < 1e-10,
        "Negative smallest subnormal: got {}, expected {}",
        result,
        expected
    );

    // 0x83FF = negative max subnormal
    let result = f16_to_f32(0x83FF);
    let expected = -(1023.0 / 1024.0) * (2.0_f32).powi(-14);
    assert!(
        (result - expected).abs() < 1e-10,
        "Negative max subnormal: got {}, expected {}",
        result,
        expected
    );
}

#[test]
fn test_f16_to_f32_various_normal_values() {
    // Test various normal values to exercise the normal path

    // 3.0: sign=0, exp=16, mantissa=512 (0.5 in fraction) -> 0x4200
    let result = f16_to_f32(0x4200);
    assert!(
        (result - 3.0).abs() < 1e-3,
        "3.0 conversion: got {}",
        result
    );

    // 4.0: sign=0, exp=17, mantissa=0 -> 0x4400
    let result = f16_to_f32(0x4400);
    assert!(
        (result - 4.0).abs() < 1e-3,
        "4.0 conversion: got {}",
        result
    );

    // 0.25: sign=0, exp=13, mantissa=0 -> 0x3400
    let result = f16_to_f32(0x3400);
    assert!(
        (result - 0.25).abs() < 1e-3,
        "0.25 conversion: got {}",
        result
    );

    // -2.0: sign=1, exp=16, mantissa=0 -> 0xC000
    let result = f16_to_f32(0xC000);
    assert!(
        (result - (-2.0)).abs() < 1e-3,
        "-2.0 conversion: got {}",
        result
    );

    // -0.5: sign=1, exp=14, mantissa=0 -> 0xB800
    let result = f16_to_f32(0xB800);
    assert!(
        (result - (-0.5)).abs() < 1e-3,
        "-0.5 conversion: got {}",
        result
    );
}

#[test]
fn test_f16_to_f32_nan_variants() {
    // Various NaN patterns (exp=31, mantissa!=0)
    // Quiet NaN
    let result = f16_to_f32(0x7E00);
    assert!(result.is_nan(), "0x7E00 should be NaN");

    // Signaling NaN
    let result = f16_to_f32(0x7C10);
    assert!(result.is_nan(), "0x7C10 should be NaN");

    // Negative NaN
    let result = f16_to_f32(0xFC01);
    assert!(result.is_nan(), "0xFC01 should be NaN");

    // Max mantissa NaN
    let result = f16_to_f32(0x7FFF);
    assert!(result.is_nan(), "0x7FFF should be NaN");
}

#[test]
fn test_read_f16_various_values() {
    // Test read_f16 with various byte patterns
    // Note: read_f16 uses half crate internally, so we test the interface

    // 2.0 (0x4000)
    let bytes = 0x4000u16.to_le_bytes();
    let result = read_f16(&bytes);
    assert!((result - 2.0).abs() < 1e-3, "read_f16(2.0): got {}", result);

    // -1.0 (0xBC00)
    let bytes = 0xBC00u16.to_le_bytes();
    let result = read_f16(&bytes);
    assert!(
        (result - (-1.0)).abs() < 1e-3,
        "read_f16(-1.0): got {}",
        result
    );

    // 0.0 (0x0000)
    let bytes = 0x0000u16.to_le_bytes();
    let result = read_f16(&bytes);
    assert!(result == 0.0, "read_f16(0.0): got {}", result);

    // 0.125 (0x3000)
    let bytes = 0x3000u16.to_le_bytes();
    let result = read_f16(&bytes);
    assert!(
        (result - 0.125).abs() < 1e-3,
        "read_f16(0.125): got {}",
        result
    );
}

// =============================================================================
// Scale Extraction - Additional Coverage
// =============================================================================

#[test]
fn test_extract_scale_min_blocks_5_6_7() {
    // Test blocks 5, 6, 7 with specific patterns
    let scales: [u8; 12] = [
        0b10_000000, // byte 0: high bits = 2 (for scale 4)
        0b11_000000, // byte 1: high bits = 3 (for scale 5)
        0b00_000000, // byte 2: high bits = 0 (for scale 6)
        0b01_000000, // byte 3: high bits = 1 (for scale 7)
        0b00_000000, // byte 4: high bits = 0 (for min 4)
        0b01_000000, // byte 5: high bits = 1 (for min 5)
        0b10_000000, // byte 6: high bits = 2 (for min 6)
        0b11_000000, // byte 7: high bits = 3 (for min 7)
        0b0001_0001, // byte 8: scale4=1, min4=1
        0b0010_0010, // byte 9: scale5=2, min5=2
        0b0011_0011, // byte 10: scale6=3, min6=3
        0b0100_0100, // byte 11: scale7=4, min7=4
    ];

    // Block 5: d = (scales[9] & 0x0F) | ((scales[1] >> 6) << 4) = 2 | (3 << 4) = 50
    //          m = (scales[9] >> 4) | ((scales[5] >> 6) << 4) = 2 | (1 << 4) = 18
    let (s5, m5) = extract_scale_min(&scales, 5);
    assert_eq!(s5, 50.0, "Block 5 scale");
    assert_eq!(m5, 18.0, "Block 5 min");

    // Block 6: d = (scales[10] & 0x0F) | ((scales[2] >> 6) << 4) = 3 | (0 << 4) = 3
    //          m = (scales[10] >> 4) | ((scales[6] >> 6) << 4) = 3 | (2 << 4) = 35
    let (s6, m6) = extract_scale_min(&scales, 6);
    assert_eq!(s6, 3.0, "Block 6 scale");
    assert_eq!(m6, 35.0, "Block 6 min");

    // Block 7: d = (scales[11] & 0x0F) | ((scales[3] >> 6) << 4) = 4 | (1 << 4) = 20
    //          m = (scales[11] >> 4) | ((scales[7] >> 6) << 4) = 4 | (3 << 4) = 52
    let (s7, m7) = extract_scale_min(&scales, 7);
    assert_eq!(s7, 20.0, "Block 7 scale");
    assert_eq!(m7, 52.0, "Block 7 min");
}

#[test]
fn test_extract_scale_min_max_values() {
    // Test maximum values (63 for 6-bit)
    let scales: [u8; 12] = [
        0xFF, 0xFF, 0xFF, 0xFF, // bytes 0-3: all 1s
        0xFF, 0xFF, 0xFF, 0xFF, // bytes 4-7: all 1s
        0xFF, 0xFF, 0xFF, 0xFF, // bytes 8-11: all 1s
    ];

    // First 4 blocks: scale = 0xFF & 63 = 63, min = 0xFF & 63 = 63
    for i in 0..4 {
        let (s, m) = extract_scale_min(&scales, i);
        assert_eq!(s, 63.0, "Block {} scale should be 63", i);
        assert_eq!(m, 63.0, "Block {} min should be 63", i);
    }
}

#[test]
fn test_extract_scale_min_from_slice_odd_indices() {
    // Test odd indices which use different bit extraction
    let scales: [u8; 12] = [
        0b11_001010, // byte 0
        0b10_001100, // byte 1
        0b00001111,  // byte 2: contributes to odd index extractions
        0b00110011,  // byte 3
        0b01_010101, // byte 4
        0b11_011011, // byte 5
        0b00001111,  // byte 6: contributes to odd index extractions
        0b00110011,  // byte 7
        0,
        0,
        0,
        0,
    ];

    // idx=1 is odd: scale_idx=0, uses different formula
    let (s1, m1) = extract_scale_min_from_slice(&scales, 1);
    // scale = (scales[0] >> 6) | ((scales[2] & 0x0F) << 2) = 3 | (0xF << 2) = 3 | 60 = 63
    // min = (scales[4] >> 6) | ((scales[6] & 0x0F) << 2) = 1 | (0xF << 2) = 1 | 60 = 61
    assert_eq!(s1, 63.0, "Index 1 scale");
    assert_eq!(m1, 61.0, "Index 1 min");
}

#[test]
fn test_extract_scale_min_from_slice_even_indices() {
    // Test even indices
    let scales: [u8; 12] = [
        10, 20, 30, 40, // scales for indices 0, 2, 4, 6
        5, 15, 25, 35, // mins for indices 0, 2, 4, 6
        0, 0, 0, 0,
    ];

    // idx=0 (even): scale = scales[0] & 0x3F = 10, min = scales[4] & 0x3F = 5
    let (s0, m0) = extract_scale_min_from_slice(&scales, 0);
    assert_eq!(s0, 10.0, "Index 0 scale");
    assert_eq!(m0, 5.0, "Index 0 min");

    // idx=2 (even): scale = scales[1] & 0x3F = 20, min = scales[5] & 0x3F = 15
    let (s2, m2) = extract_scale_min_from_slice(&scales, 2);
    assert_eq!(s2, 20.0, "Index 2 scale");
    assert_eq!(m2, 15.0, "Index 2 min");
}

// =============================================================================
// Softmax SIMD - Additional Coverage for SIMD Paths
// =============================================================================

#[test]
fn test_softmax_simd_exactly_8_elements() {
    // Exactly 8 elements - minimum for SIMD path on AVX2
    let mut x = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
    let reference = softmax_reference(&x);
    softmax_simd(&mut x);

    for (i, (actual, expected)) in x.iter().zip(reference.iter()).enumerate() {
        assert!(
            (actual - expected).abs() < 1e-5,
            "Exactly 8 elements: mismatch at {}: got {}, expected {}",
            i,
            actual,
            expected
        );
    }
}

#[test]
fn test_softmax_simd_16_elements() {
    // 16 elements - 2 SIMD iterations
    let mut x: Vec<f32> = (0..16).map(|i| i as f32 * 0.1 - 0.8).collect();
    let reference = softmax_reference(&x);
    softmax_simd(&mut x);

    for (i, (actual, expected)) in x.iter().zip(reference.iter()).enumerate() {
        assert!(
            (actual - expected).abs() < 1e-5,
            "16 elements: mismatch at {}: got {}, expected {}",
            i,
            actual,
            expected
        );
    }
}

#[test]
fn test_softmax_simd_17_elements_unaligned() {
    // 17 elements - tests remainder handling (17 = 2*8 + 1)
    let mut x: Vec<f32> = (0..17).map(|i| (i as f32 - 8.0) * 0.5).collect();
    let reference = softmax_reference(&x);
    softmax_simd(&mut x);

    for (i, (actual, expected)) in x.iter().zip(reference.iter()).enumerate() {
        assert!(
            (actual - expected).abs() < 1e-5,
            "17 elements: mismatch at {}: got {}, expected {}",
            i,
            actual,
            expected
        );
    }
}

#[test]
fn test_softmax_simd_7_elements_scalar_fallback() {
    // 7 elements - should use scalar fallback (< 8)
    let mut x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
    let reference = softmax_reference(&x);
    softmax_simd(&mut x);

    for (i, (actual, expected)) in x.iter().zip(reference.iter()).enumerate() {
        assert!(
            (actual - expected).abs() < 1e-5,
            "7 elements: mismatch at {}: got {}, expected {}",
            i,
            actual,
            expected
        );
    }
}

#[test]
fn test_softmax_simd_64_elements() {
    // 64 elements - 8 SIMD iterations, tests larger scale
    let mut x: Vec<f32> = (0..64).map(|i| ((i as f32 * 0.1) - 3.0).sin()).collect();
    let reference = softmax_reference(&x);
    softmax_simd(&mut x);

    for (i, (actual, expected)) in x.iter().zip(reference.iter()).enumerate() {
        assert!(
            (actual - expected).abs() < 1e-5,
            "64 elements: mismatch at {}: got {}, expected {}",
            i,
            actual,
            expected
        );
    }
}

#[test]
fn test_softmax_simd_mixed_signs() {
    // Mix of large positive and negative values
    let mut x = vec![100.0, -100.0, 50.0, -50.0, 0.0, 25.0, -25.0, 10.0, -10.0];
    softmax_simd(&mut x);

    // Check sum is 1.0
    let sum: f32 = x.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-5,
        "Mixed signs: sum should be 1.0, got {}",
        sum
    );

    // First element (100.0) should dominate
    assert!(
        x[0] > 0.99,
        "Element with value 100.0 should dominate: {}",
        x[0]
    );
}

// Reference softmax implementation
fn softmax_reference(x: &[f32]) -> Vec<f32> {
    if x.is_empty() {
        return vec![];
    }
    let max_val = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp_vals: Vec<f32> = x.iter().map(|v| (*v - max_val).exp()).collect();
    let sum: f32 = exp_vals.iter().sum();
    exp_vals.iter().map(|v| v / sum).collect()
}

// =============================================================================
// Fused SwiGLU - Additional Coverage
// =============================================================================

#[test]
fn test_fused_swiglu_simd_exactly_8_elements() {
    // Exactly 8 elements - minimum for SIMD path
    let mut gate = vec![1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 1.5, -1.5];
    let up = vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0];
    let expected = swiglu_reference(&gate, &up);

    fused_swiglu_simd(&mut gate, &up);

    for (i, (g, e)) in gate.iter().zip(expected.iter()).enumerate() {
        assert!(
            (g - e).abs() < 0.15, // Lenient for AVX2 polynomial approx
            "8 elements SwiGLU: mismatch at {}: got {}, expected {}",
            i,
            g,
            e
        );
    }
}

include!("fused_swiglu.rs");
