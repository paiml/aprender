//! T-COV-95 Coverage tests for quantize/mod.rs
//!
//! Covers:
//! - quantize_activations_q8k_into: success, error paths (not multiple of 256, buffer too small)
//! - quantize_to_q8_blocks: success, error path (not multiple of 32)
//! - dequantize_q8_blocks: round-trip
//! - InterleavedQ4K::from_q4k: success, error path
//! - InterleavedQ4K::num_values
//! - InterleavedQ4K::dot (scalar)
//! - fused_q4_0_q8_0_dot_scalar: known values
//! - fused_q4_0_q8_0_parallel_matvec: success and error paths
//! - fused_q4_0_q8_0_parallel_matvec_into: success and error paths
//! - f16_to_f32_lut: known values
//! - SimdBackend Display + Default
//! - DequantStats Default
//! - detect_simd_backend
//! - Q8_0Block::quantize, dequantize, quantization_error, relative_error
//! - Q8KSuperBlock::quantize, quantize_into, dequantize

use super::*;

// ============================================================================
// quantize_activations_q8k_into
// ============================================================================

#[test]
fn test_quantize_activations_q8k_into_success() {
    let activations = vec![1.0f32; 256];
    let mut scales = vec![0.0f32; 1];
    let mut quants = vec![0i8; 256];

    let result = quantize_activations_q8k_into(&activations, &mut scales, &mut quants);
    assert!(result.is_ok());
    assert!(scales[0] > 0.0, "Scale should be positive");
    // All same value -> all quants should be the same
    assert!(quants.iter().all(|&q| q == quants[0]));
}

#[test]
fn test_quantize_activations_q8k_into_two_superblocks() {
    let activations = vec![0.5f32; 512];
    let mut scales = vec![0.0f32; 2];
    let mut quants = vec![0i8; 512];

    let result = quantize_activations_q8k_into(&activations, &mut scales, &mut quants);
    assert!(result.is_ok());
    assert!(scales[0] > 0.0);
    assert!(scales[1] > 0.0);
}

#[test]
fn test_quantize_activations_q8k_into_not_multiple_of_256() {
    let activations = vec![1.0f32; 100]; // Not multiple of 256
    let mut scales = vec![0.0f32; 1];
    let mut quants = vec![0i8; 100];

    let result = quantize_activations_q8k_into(&activations, &mut scales, &mut quants);
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(err.contains("256"), "Error should mention 256: {}", err);
}

#[test]
fn test_quantize_activations_q8k_into_scales_too_small() {
    let activations = vec![1.0f32; 512]; // 2 superblocks
    let mut scales = vec![0.0f32; 1]; // Only room for 1
    let mut quants = vec![0i8; 512];

    let result = quantize_activations_q8k_into(&activations, &mut scales, &mut quants);
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("Scales") || err.contains("scales") || err.contains("too small"),
        "Error should mention scales buffer: {}",
        err
    );
}

#[test]
fn test_quantize_activations_q8k_into_quants_too_small() {
    let activations = vec![1.0f32; 256];
    let mut scales = vec![0.0f32; 1];
    let mut quants = vec![0i8; 100]; // Too small

    let result = quantize_activations_q8k_into(&activations, &mut scales, &mut quants);
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("Quants") || err.contains("quants") || err.contains("too small"),
        "Error should mention quants buffer: {}",
        err
    );
}

// ============================================================================
// quantize_to_q8_blocks / dequantize_q8_blocks
// ============================================================================

#[test]
fn test_quantize_to_q8_blocks_success() {
    let values = vec![1.0f32; 64]; // 2 blocks of 32
    let blocks = quantize_to_q8_blocks(&values).expect("should quantize");
    assert_eq!(blocks.len(), 2);
}

#[test]
fn test_quantize_to_q8_blocks_not_multiple_of_32() {
    let values = vec![1.0f32; 50]; // Not multiple of 32
    let result = quantize_to_q8_blocks(&values);
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(err.contains("32"), "Error should mention 32: {}", err);
}

#[test]
fn test_quantize_dequantize_round_trip() {
    let original = vec![0.5f32; 32];
    let blocks = quantize_to_q8_blocks(&original).expect("should quantize");
    let dequantized = dequantize_q8_blocks(&blocks);
    assert_eq!(dequantized.len(), 32);

    // Check approximate round-trip (quantization introduces error)
    for (o, d) in original.iter().zip(dequantized.iter()) {
        assert!(
            (o - d).abs() < 0.01,
            "Round-trip error too large: {} vs {}",
            o,
            d
        );
    }
}

#[test]
fn test_quantize_dequantize_varied_values() {
    let original: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) * 0.1).collect();
    let blocks = quantize_to_q8_blocks(&original).expect("should quantize");
    let dequantized = dequantize_q8_blocks(&blocks);

    for (o, d) in original.iter().zip(dequantized.iter()) {
        assert!(
            (o - d).abs() < 0.02,
            "Round-trip error too large: {} vs {}",
            o,
            d
        );
    }
}

#[test]
fn test_dequantize_q8_blocks_empty() {
    let blocks: Vec<Q8_0Block> = vec![];
    let output = dequantize_q8_blocks(&blocks);
    assert!(output.is_empty());
}

// ============================================================================
// InterleavedQ4K
// ============================================================================

#[test]
fn test_interleaved_q4k_from_q4k_success() {
    // 144 bytes per super-block
    let data = vec![0u8; 144];
    let result = InterleavedQ4K::from_q4k(&data);
    assert!(result.is_ok());
    let iq = result.unwrap();
    assert_eq!(iq.num_super_blocks, 1);
    assert_eq!(iq.num_values(), 256);
}

#[test]
fn test_interleaved_q4k_from_q4k_two_superblocks() {
    let data = vec![0u8; 288]; // 2 super-blocks
    let result = InterleavedQ4K::from_q4k(&data);
    assert!(result.is_ok());
    let iq = result.unwrap();
    assert_eq!(iq.num_super_blocks, 2);
    assert_eq!(iq.num_values(), 512);
    assert_eq!(iq.d.len(), 2);
    assert_eq!(iq.dmin.len(), 2);
    assert_eq!(iq.scales.len(), 24);
    assert_eq!(iq.qs.len(), 256);
}

#[test]
fn test_interleaved_q4k_from_q4k_invalid_length() {
    let data = vec![0u8; 100]; // Not multiple of 144
    let result = InterleavedQ4K::from_q4k(&data);
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(err.contains("144"), "Error should mention 144: {}", err);
}

#[test]
fn test_interleaved_q4k_from_q4k_empty() {
    let data: Vec<u8> = vec![];
    let result = InterleavedQ4K::from_q4k(&data);
    assert!(result.is_ok());
    let iq = result.unwrap();
    assert_eq!(iq.num_super_blocks, 0);
    assert_eq!(iq.num_values(), 0);
}

#[test]
fn test_interleaved_q4k_dot_scalar() {
    // Create a super-block with known values
    let mut data = vec![0u8; 144];
    // d = 1.0 as f16 (bits = 0x3C00)
    data[0] = 0x00;
    data[1] = 0x3C;
    // dmin = 0.0 as f16
    data[2] = 0x00;
    data[3] = 0x00;
    // scales: all zeros (results in zero output)
    // qs: all zeros

    let iq = InterleavedQ4K::from_q4k(&data).unwrap();
    let activations = vec![1.0f32; 256];
    let result = iq.dot(&activations);
    assert!(result.is_ok());
    // With zero scales and zero qs, result should be 0 or very small
    let val = result.unwrap();
    assert!(
        val.abs() < 0.1,
        "Dot product with zero data should be near zero, got {}",
        val
    );
}

// ============================================================================
// fused_q4_0_q8_0_dot_scalar
// ============================================================================

#[test]
fn test_fused_q4_0_q8_0_dot_scalar_zero() {
    // Q4_0 block: 18 bytes (2 scale + 16 quants)
    // All zeros
    let q4_data = vec![0u8; 18];
    let q8_scales = vec![1.0f32];
    let q8_quants = vec![0i8; 32];
    let result = fused_q4_0_q8_0_dot_scalar(&q4_data, &q8_scales, &q8_quants, 32);
    assert_eq!(result, 0.0);
}

#[test]
fn test_fused_q4_0_q8_0_dot_scalar_empty() {
    let result = fused_q4_0_q8_0_dot_scalar(&[], &[], &[], 0);
    assert_eq!(result, 0.0);
}

// ============================================================================
// fused_q4_0_q8_0_parallel_matvec
// ============================================================================

#[test]
fn test_fused_q4_0_q8_0_parallel_matvec_weight_too_small() {
    let weight_data = vec![0u8; 10]; // Too small
    let activations = vec![1.0f32; 32];
    let result = fused_q4_0_q8_0_parallel_matvec(&weight_data, &activations, 32, 2);
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("too small"),
        "Error should mention too small: {}",
        err
    );
}

#[test]
fn test_fused_q4_0_q8_0_parallel_matvec_activation_mismatch() {
    // Need enough weight data for out_dim=1, in_dim=32 -> 1 block -> 18 bytes
    let weight_data = vec![0u8; 18];
    let activations = vec![1.0f32; 64]; // Wrong size
    let result = fused_q4_0_q8_0_parallel_matvec(&weight_data, &activations, 32, 1);
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("Activation") || err.contains("activation"),
        "Error should mention activation mismatch: {}",
        err
    );
}

#[test]
fn test_fused_q4_0_q8_0_parallel_matvec_success() {
    // 1 row, 32 elements = 1 Q4_0 block (18 bytes)
    let weight_data = vec![0u8; 18];
    let activations = vec![0.0f32; 32];
    let result = fused_q4_0_q8_0_parallel_matvec(&weight_data, &activations, 32, 1);
    assert!(result.is_ok());
    let output = result.unwrap();
    assert_eq!(output.len(), 1);
}

// ============================================================================
// fused_q4_0_q8_0_parallel_matvec_into
// ============================================================================

#[test]
fn test_fused_q4_0_q8_0_parallel_matvec_into_weight_too_small() {
    let weight_data = vec![0u8; 10];
    let activations = vec![1.0f32; 32];
    let mut output = vec![0.0f32; 2];
    let result = fused_q4_0_q8_0_parallel_matvec_into(&weight_data, &activations, 32, &mut output);
    assert!(result.is_err());
}

#[test]
fn test_fused_q4_0_q8_0_parallel_matvec_into_activation_mismatch() {
    let weight_data = vec![0u8; 18];
    let activations = vec![1.0f32; 64]; // Wrong
    let mut output = vec![0.0f32; 1];
    let result = fused_q4_0_q8_0_parallel_matvec_into(&weight_data, &activations, 32, &mut output);
    assert!(result.is_err());
}

#[test]
fn test_fused_q4_0_q8_0_parallel_matvec_into_success() {
    let weight_data = vec![0u8; 18];
    let activations = vec![0.0f32; 32];
    let mut output = vec![0.0f32; 1];
    let result = fused_q4_0_q8_0_parallel_matvec_into(&weight_data, &activations, 32, &mut output);
    assert!(result.is_ok());
    assert_eq!(output.len(), 1);
}

// ============================================================================
// f16_to_f32_lut
// ============================================================================

#[test]
fn test_f16_to_f32_lut_zero() {
    let val = f16_to_f32_lut(0);
    assert_eq!(val, 0.0);
}

#[test]
fn test_f16_to_f32_lut_one() {
    // f16 1.0 = 0x3C00
    let val = f16_to_f32_lut(0x3C00);
    assert!((val - 1.0).abs() < 0.001, "Expected 1.0, got {}", val);
}

#[test]
fn test_f16_to_f32_lut_negative_one() {
    // f16 -1.0 = 0xBC00
    let val = f16_to_f32_lut(0xBC00);
    assert!((val - (-1.0)).abs() < 0.001, "Expected -1.0, got {}", val);
}

#[test]
fn test_f16_to_f32_lut_max() {
    // f16 max positive = 0x7BFF (~65504)
    let val = f16_to_f32_lut(0x7BFF);
    assert!(val > 65000.0, "Expected large value, got {}", val);
}

// ============================================================================
// SimdBackend
// ============================================================================

#[test]
fn test_simd_backend_display() {
    assert_eq!(format!("{}", SimdBackend::Avx2), "AVX2");
    assert_eq!(format!("{}", SimdBackend::Sse2), "SSE2");
    assert_eq!(format!("{}", SimdBackend::Neon), "NEON");
    assert_eq!(format!("{}", SimdBackend::Scalar), "Scalar");
}

#[test]
fn test_simd_backend_default() {
    let backend: SimdBackend = Default::default();
    assert_eq!(backend, SimdBackend::Scalar);
}

#[test]
fn test_simd_backend_eq() {
    assert_eq!(SimdBackend::Avx2, SimdBackend::Avx2);
    assert_ne!(SimdBackend::Avx2, SimdBackend::Sse2);
}

#[test]
fn test_simd_backend_clone() {
    let backend = SimdBackend::Avx2;
    let cloned = backend;
    assert_eq!(backend, cloned);
}

#[test]
fn test_simd_backend_debug() {
    let debug = format!("{:?}", SimdBackend::Avx2);
    assert!(debug.contains("Avx2"));
}

// ============================================================================
// DequantStats
// ============================================================================

#[test]
fn test_dequant_stats_default() {
    let stats: DequantStats = Default::default();
    assert_eq!(stats.blocks_processed, 0);
    assert_eq!(stats.bytes_processed, 0);
    assert_eq!(stats.simd_backend, SimdBackend::Scalar);
}

#[test]
fn test_dequant_stats_construction() {
    let stats = DequantStats {
        blocks_processed: 100,
        bytes_processed: 1800,
        simd_backend: SimdBackend::Avx2,
    };
    assert_eq!(stats.blocks_processed, 100);
    assert_eq!(stats.bytes_processed, 1800);
    assert_eq!(stats.simd_backend, SimdBackend::Avx2);
}

#[test]
fn test_dequant_stats_clone() {
    let stats = DequantStats {
        blocks_processed: 50,
        bytes_processed: 900,
        simd_backend: SimdBackend::Sse2,
    };
    let cloned = stats.clone();
    assert_eq!(cloned.blocks_processed, 50);
    assert_eq!(cloned.simd_backend, SimdBackend::Sse2);
}

#[test]
fn test_dequant_stats_debug() {
    let stats = DequantStats::default();
    let debug = format!("{:?}", stats);
    assert!(debug.contains("DequantStats"));
}

include!("tests_coverage_detect_simd.rs");
include!("q8_block_tests.rs");
include!("tests_coverage_04.rs");
