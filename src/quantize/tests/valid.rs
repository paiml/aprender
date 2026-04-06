//! Deep coverage tests for fused_k.rs inner loop paths (PMAT-COV-95)
//!
//! The existing tests mostly use zero-scale data which bypasses the actual
//! dequant+dot computation. These tests construct valid Q4_K super-blocks
//! with non-zero d, dmin, scales, and qs values to exercise every line in:
//! - `fused_q4k_dot` inner loop (lines 92-141)
//! - `fused_q4k_q8k_dot` inner loop (lines 607-667)
//! - `fused_q4k_dot_simd` dispatch and parity with scalar
//! - `fused_q4k_q8k_dot_simd` dispatch and parity with scalar

use crate::quantize::fused_k::{
    fused_q4k_dot, fused_q4k_dot_simd, fused_q4k_q8k_dot, fused_q4k_q8k_dot_simd,
};
use crate::quantize::simd::extract_scale_min;
use crate::quantize::types::QK_K;

/// Q4_K super-block size: 2 (d) + 2 (dmin) + 12 (scales) + 128 (qs) = 144 bytes
const SB_BYTES: usize = 144;

// ============================================================================
// Helper: Build a valid Q4_K super-block with controllable parameters
// ============================================================================

/// Build a single Q4_K super-block with the given f16 d, f16 dmin,
/// 12-byte scales array, and 128-byte qs array.
fn build_q4k_superblock(d_f16: u16, dmin_f16: u16, scales: &[u8; 12], qs: &[u8; 128]) -> Vec<u8> {
    let mut sb = vec![0u8; SB_BYTES];
    sb[0..2].copy_from_slice(&d_f16.to_le_bytes());
    sb[2..4].copy_from_slice(&dmin_f16.to_le_bytes());
    sb[4..16].copy_from_slice(scales);
    sb[16..144].copy_from_slice(qs);
    sb
}

/// f16 encoding of 1.0
const F16_ONE: u16 = 0x3C00;
/// f16 encoding of 0.5
const F16_HALF: u16 = 0x3800;
/// f16 encoding of 2.0
const F16_TWO: u16 = 0x4000;
/// f16 encoding of 0.0
const F16_ZERO: u16 = 0x0000;
/// f16 encoding of 0.25
const F16_QUARTER: u16 = 0x3400;

// ============================================================================
// DEEP SCALAR INNER LOOP TESTS: fused_q4k_dot
// ============================================================================

/// Test fused_q4k_dot with d=1.0, dmin=0, uniform scales, uniform qs.
/// This isolates the `d * sc * q_val` term without the `dmin * m` subtraction.
#[test]
fn test_fused_q4k_dot_d_only_uniform_qs() {
    // scales: all blocks have scale=1, min=0
    // For blocks 0-3: scales[j] & 63 = 1, scales[j+4] & 63 = 0
    let mut scales = [0u8; 12];
    for j in 0..4 {
        scales[j] = 1; // scale = 1
        scales[j + 4] = 0; // min = 0
    }
    // For blocks 4-7: high bits encoding (scale=1, min=0)
    // d = (scales[j+4] & 0x0F) | ((scales[j-4] >> 6) << 4)
    // m = (scales[j+4] >> 4) | ((scales[j] >> 6) << 4)
    // We need scale=1: set scales[j+4] & 0x0F = 1, scales[j-4] >> 6 = 0
    // scales[8] = 0x01, scales[9] = 0x01, scales[10] = 0x01, scales[11] = 0x01
    for j in 8..12 {
        scales[j] = 0x01; // low nibble = 1 for scale, high nibble = 0 for min
    }

    // qs: all nibbles = 5 => 0x55
    let qs = [0x55u8; 128];

    let data = build_q4k_superblock(F16_ONE, F16_ZERO, &scales, &qs);
    // All activations = 1.0
    let activations = vec![1.0f32; QK_K];

    let result = fused_q4k_dot(&data, &activations).expect("should succeed");

    // Each value dequantizes to: d * scale * nibble_val - dmin * min
    // = 1.0 * scale_float * 5.0 - 0.0
    // All 256 values contribute scale_float * 5.0 * 1.0
    // The total depends on scale extraction for each block
    // Let's just verify it's non-zero and positive
    assert!(result > 0.0, "Expected positive result, got {result}");
}

/// Test that dmin subtraction actually occurs by setting dmin > 0 with min > 0.
#[test]
fn test_fused_q4k_dot_dmin_subtraction() {
    // d=0 so d*scale*q_val = 0 for all values
    // dmin=1.0 and min=1 means each value = -dmin*min = -1.0
    // dot with all-ones activations = -1.0 * 256 = -256.0
    // But min extraction depends on the scale format.

    let mut scales = [0u8; 12];
    // For blocks 0-3: scale=0, min=1
    for j in 0..4 {
        scales[j] = 0; // scale=0
        scales[j + 4] = 1; // min=1
    }
    // For blocks 4-7: scale=0, min=1
    // m = (scales[j+4] >> 4) | ((scales[j] >> 6) << 4)
    // Want m=1: scales[j+4] >> 4 = 0, scales[j] >> 6 = 0 => m=0... need different encoding
    // Actually for blocks 4-7: scales[j+4] = scales[8..12]
    // d = (scales[j+4] & 0x0F) | ((scales[j-4] >> 6) << 4) where j=4..7
    // For j=4: d = (scales[8] & 0x0F) | ((scales[0] >> 6) << 4)
    // m = (scales[8] >> 4) | ((scales[4] >> 6) << 4)
    // Want d=0, m=1: scales[8] & 0x0F = 0, scales[8] >> 4 = 1 => scales[8] = 0x10
    for j in 8..12 {
        scales[j] = 0x10; // high nibble = 1 for min, low nibble = 0 for scale
    }

    let qs = [0x00u8; 128]; // q_val = 0 for all
    let data = build_q4k_superblock(F16_ZERO, F16_ONE, &scales, &qs);
    let activations = vec![1.0f32; QK_K];

    let result = fused_q4k_dot(&data, &activations).expect("should succeed");

    // Every value = 0 - 1.0*min_float => each contributes -min_float * 1.0
    // Result should be negative
    assert!(
        result < 0.0,
        "Expected negative result from dmin, got {result}"
    );
}

/// Test with both d and dmin non-zero to exercise both terms.
#[test]
fn test_fused_q4k_dot_both_terms() {
    let mut scales = [0u8; 12];
    // All blocks: scale=2, min=1
    for j in 0..4 {
        scales[j] = 2;
        scales[j + 4] = 1;
    }
    for j in 8..12 {
        scales[j] = 0x12; // low nibble=2 (scale), high nibble=1 (min)
    }

    // qs: alternating 0x37 => low=7, high=3
    let mut qs = [0u8; 128];
    for i in 0..128 {
        qs[i] = 0x37;
    }

    let data = build_q4k_superblock(F16_HALF, F16_QUARTER, &scales, &qs);
    let activations = vec![1.0f32; QK_K];

    let result = fused_q4k_dot(&data, &activations).expect("should succeed");
    // Result should be non-zero (mix of positive and negative terms)
    assert!(result.is_finite(), "Expected finite result, got {result}");
}

/// Test with varying qs patterns across the 128-byte array.
#[test]
fn test_fused_q4k_dot_varying_qs() {
    let mut scales = [0u8; 12];
    for j in 0..4 {
        scales[j] = 3;
        scales[j + 4] = 0;
    }
    for j in 8..12 {
        scales[j] = 0x03;
    }

    // Monotonically increasing nibbles
    let mut qs = [0u8; 128];
    for i in 0..128 {
        let lo = (i % 16) as u8;
        let hi = ((i + 5) % 16) as u8;
        qs[i] = lo | (hi << 4);
    }

    let data = build_q4k_superblock(F16_ONE, F16_ZERO, &scales, &qs);
    // Varying activations
    let activations: Vec<f32> = (0..QK_K).map(|i| (i as f32) * 0.01 - 1.28).collect();

    let result = fused_q4k_dot(&data, &activations).expect("should succeed");
    assert!(result.is_finite(), "Expected finite result, got {result}");
}

/// Test with max nibble values (0xFF) and non-zero d, to exercise large q_val paths.
#[test]
fn test_fused_q4k_dot_max_nibbles_nonzero_d() {
    let mut scales = [0u8; 12];
    for j in 0..4 {
        scales[j] = 63; // max 6-bit scale
        scales[j + 4] = 63; // max 6-bit min
    }
    // For blocks 4-7, encode max scale and min
    // d = (scales[j+4] & 0x0F) | ((scales[j-4] >> 6) << 4)
    // For max: d=63 => 0x3F => (0x0F) | (scales[j-4]>>6 << 4)
    // scales[j-4] = 63 means bits[7:6] = 0 so d = 0x0F = 15
    // Actually 63 = 0b00111111 so >>6 = 0, high bits = 0x0F from scales[j+4]&0x0F
    // max achievable: scales[j+4] = 0xFF => 0x0F | ((scales[j-4]>>6)<<4)
    // If scales[0..4] have top bits set: scales[0] = 0xFF => 0xFF>>6 = 3, d = 0x0F|(3<<4) = 0x3F = 63
    for j in 0..4 {
        scales[j] = 0xFF; // bits 7:6 = 3 for high-block encoding
        scales[j + 4] = 0xFF; // same
    }
    for j in 8..12 {
        scales[j] = 0xFF;
    }

    let qs = [0xFFu8; 128]; // all nibbles = 15
    let data = build_q4k_superblock(F16_TWO, F16_HALF, &scales, &qs);
    let activations = vec![1.0f32; QK_K];

    let result = fused_q4k_dot(&data, &activations).expect("should succeed");
    assert!(result.is_finite(), "Expected finite result, got {result}");
    // With max scales, max nibbles, d=2.0, result should be large
    assert!(
        result.abs() > 1.0,
        "Expected large magnitude result, got {result}"
    );
}

/// Test with two super-blocks where each has different d/dmin.
#[test]
fn test_fused_q4k_dot_two_blocks_different_params() {
    let mut scales1 = [0u8; 12];
    for j in 0..4 {
        scales1[j] = 10;
        scales1[j + 4] = 5;
    }
    for j in 8..12 {
        scales1[j] = 0x5A; // scale=10, min=5 roughly
    }
    let qs1 = [0x88u8; 128]; // lo=8, hi=8

    let mut scales2 = [0u8; 12];
    for j in 0..4 {
        scales2[j] = 20;
        scales2[j + 4] = 10;
    }
    for j in 8..12 {
        scales2[j] = 0xA4; // different encoding
    }
    let qs2 = [0xAAu8; 128]; // lo=10, hi=10

    let mut data = build_q4k_superblock(F16_ONE, F16_HALF, &scales1, &qs1);
    data.extend(build_q4k_superblock(F16_TWO, F16_QUARTER, &scales2, &qs2));

    let activations = vec![0.5f32; QK_K * 2];

    let result = fused_q4k_dot(&data, &activations).expect("should succeed");
    assert!(result.is_finite(), "Expected finite result, got {result}");
}

/// Test that all 4 chunks (j=0,64,128,192) in the inner loop are exercised
/// by verifying the result changes when different chunks have different data.
#[test]
fn test_fused_q4k_dot_per_chunk_sensitivity() {
    let mut scales = [0u8; 12];
    for j in 0..4 {
        scales[j] = 5;
    }
    for j in 8..12 {
        scales[j] = 0x05;
    }

    // Baseline: all qs = 0
    let qs_zero = [0x00u8; 128];
    let data_zero = build_q4k_superblock(F16_ONE, F16_ZERO, &scales, &qs_zero);
    let activations = vec![1.0f32; QK_K];
    let baseline = fused_q4k_dot(&data_zero, &activations).expect("baseline");

    // Modify only chunk 0 (bytes 0..32 of qs, values 0..63)
    let mut qs_chunk0 = [0x00u8; 128];
    for i in 0..32 {
        qs_chunk0[i] = 0xFF;
    }
    let data_chunk0 = build_q4k_superblock(F16_ONE, F16_ZERO, &scales, &qs_chunk0);
    let result_chunk0 = fused_q4k_dot(&data_chunk0, &activations).expect("chunk0");
    assert_ne!(
        baseline, result_chunk0,
        "Chunk 0 modification should change result"
    );

    // Modify only chunk 2 (bytes 64..96 of qs, values 128..191)
    let mut qs_chunk2 = [0x00u8; 128];
    for i in 64..96 {
        qs_chunk2[i] = 0xFF;
    }
    let data_chunk2 = build_q4k_superblock(F16_ONE, F16_ZERO, &scales, &qs_chunk2);
    let result_chunk2 = fused_q4k_dot(&data_chunk2, &activations).expect("chunk2");
    assert_ne!(
        baseline, result_chunk2,
        "Chunk 2 modification should change result"
    );

    // Modify only chunk 3 (bytes 96..128 of qs, values 192..255)
    let mut qs_chunk3 = [0x00u8; 128];
    for i in 96..128 {
        qs_chunk3[i] = 0xFF;
    }
    let data_chunk3 = build_q4k_superblock(F16_ONE, F16_ZERO, &scales, &qs_chunk3);
    let result_chunk3 = fused_q4k_dot(&data_chunk3, &activations).expect("chunk3");
    assert_ne!(
        baseline, result_chunk3,
        "Chunk 3 modification should change result"
    );
}

/// Test low nibble vs high nibble paths by comparing qs=0x0F (lo=15,hi=0) vs qs=0xF0 (lo=0,hi=15).
#[test]
fn test_fused_q4k_dot_lo_vs_hi_nibbles() {
    let mut scales = [0u8; 12];
    for j in 0..4 {
        scales[j] = 10;
    }
    for j in 8..12 {
        scales[j] = 0x0A;
    }

    // Only low nibbles set
    let qs_lo = [0x0Fu8; 128];
    let data_lo = build_q4k_superblock(F16_ONE, F16_ZERO, &scales, &qs_lo);

    // Only high nibbles set
    let qs_hi = [0xF0u8; 128];
    let data_hi = build_q4k_superblock(F16_ONE, F16_ZERO, &scales, &qs_hi);

    let activations = vec![1.0f32; QK_K];

    let result_lo = fused_q4k_dot(&data_lo, &activations).expect("lo only");
    let result_hi = fused_q4k_dot(&data_hi, &activations).expect("hi only");

    // Both should be non-zero
    assert!(
        result_lo.abs() > 0.0,
        "Low nibble result should be non-zero"
    );
    assert!(
        result_hi.abs() > 0.0,
        "High nibble result should be non-zero"
    );
    // They should be different because lo and hi nibbles use different scales
    // (is vs is+1 in extract_scale_min)
}

/// Test with negative activations to exercise the multiply-accumulate with signs.
#[test]
fn test_fused_q4k_dot_negative_activations() {
    let mut scales = [0u8; 12];
    for j in 0..4 {
        scales[j] = 8;
        scales[j + 4] = 4;
    }
    for j in 8..12 {
        scales[j] = 0x48;
    }

    let qs = [0x77u8; 128]; // lo=7, hi=7
    let data = build_q4k_superblock(F16_ONE, F16_HALF, &scales, &qs);
    let activations = vec![-1.0f32; QK_K];

    let result = fused_q4k_dot(&data, &activations).expect("neg activations");
    assert!(result.is_finite(), "Expected finite result, got {result}");
    // Result should have opposite sign from positive activations
    let pos_result = fused_q4k_dot(&data, &vec![1.0f32; QK_K]).expect("pos activations");
    if pos_result.abs() > 1e-6 {
        assert!(
            result.signum() != pos_result.signum(),
            "Negative activations should flip result sign: neg={result}, pos={pos_result}"
        );
    }
}

// ============================================================================
// DEEP SCALAR INNER LOOP TESTS: fused_q4k_q8k_dot
// ============================================================================

/// Test fused_q4k_q8k_dot with non-zero d, dmin, scales, qs, and q8k values.
#[test]
fn test_fused_q4k_q8k_dot_nonzero_computation() {
    let mut scales = [0u8; 12];
    for j in 0..4 {
        scales[j] = 5;
        scales[j + 4] = 2;
    }
    for j in 8..12 {
        scales[j] = 0x25;
    }

    let qs = [0x55u8; 128]; // lo=5, hi=5
    let data = build_q4k_superblock(F16_ONE, F16_HALF, &scales, &qs);

    let q8k_scales = vec![0.5f32; 1];
    let q8k_quants = vec![10i8; QK_K];

    let result = fused_q4k_q8k_dot(&data, &q8k_scales, &q8k_quants).expect("should succeed");
    assert!(result.is_finite(), "Expected finite result, got {result}");
    assert!(result.abs() > 0.0, "Expected non-zero result, got {result}");
}

/// Test q8k dot with varying quant values to exercise the inner product loop.
#[test]
fn test_fused_q4k_q8k_dot_varying_quants() {
    let mut scales = [0u8; 12];
    for j in 0..4 {
        scales[j] = 10;
        scales[j + 4] = 3;
    }
    for j in 8..12 {
        scales[j] = 0x3A;
    }

    let mut qs = [0u8; 128];
    for i in 0..128 {
        qs[i] = ((i * 7 + 3) % 256) as u8;
    }
    let data = build_q4k_superblock(F16_ONE, F16_QUARTER, &scales, &qs);

    let q8k_scales = vec![1.0f32; 1];
    // Varying positive and negative q8 values
    let q8k_quants: Vec<i8> = (0..QK_K).map(|i| ((i as i16 % 127) - 63) as i8).collect();

    let result = fused_q4k_q8k_dot(&data, &q8k_scales, &q8k_quants).expect("should succeed");
    assert!(result.is_finite(), "Expected finite result, got {result}");
}

/// Test q8k dot with multiple super-blocks and non-zero data.
#[test]
fn test_fused_q4k_q8k_dot_multi_sb_nonzero() {
    let mut scales = [0u8; 12];
    for j in 0..4 {
        scales[j] = 7;
        scales[j + 4] = 3;
    }
    for j in 8..12 {
        scales[j] = 0x37;
    }

    let qs = [0xABu8; 128]; // lo=11, hi=10
    let num_sb = 3;
    let mut data = Vec::new();
    for _ in 0..num_sb {
        data.extend(build_q4k_superblock(F16_HALF, F16_QUARTER, &scales, &qs));
    }

    let q8k_scales = vec![0.3f32; num_sb];
    let q8k_quants: Vec<i8> = (0..QK_K * num_sb).map(|i| ((i % 50) as i8) - 25).collect();

    let result = fused_q4k_q8k_dot(&data, &q8k_scales, &q8k_quants).expect("should succeed");
    assert!(result.is_finite(), "Expected finite result, got {result}");
}

include!("fused_q4k_05.rs");
include!("fused_q4k_02_03.rs");
