//! Deep coverage tests for realizar/src/quantize.rs
//!
//! This module provides additional coverage for quantization functions,
//! SIMD backends, RoPE rotation, and helper types not covered by existing tests.

#![allow(clippy::needless_range_loop)]

use realizar::quantize::{
    dequantize_f16, dequantize_q4_0, dequantize_q4_1, dequantize_q4_k, dequantize_q4_k_parallel,
    dequantize_q4_k_simd, dequantize_q5_0, dequantize_q5_1, dequantize_q5_k, dequantize_q6_k,
    dequantize_q8_0, dequantize_q8_0_parallel, dequantize_q8_0_simd, detect_simd_backend,
    f16_to_f32, fused_q4k_dot, fused_q4k_dot_simd, fused_q4k_parallel_matvec,
    fused_q4k_parallel_matvec_into, fused_q4k_tiled_matvec, fused_q5k_dot, fused_q5k_dot_simd,
    fused_q5k_parallel_matvec, fused_q5k_parallel_matvec_into, fused_q6k_dot, fused_q6k_dot_simd,
    fused_q6k_parallel_matvec, fused_q6k_parallel_matvec_into, fused_swiglu_simd,
    quantize_activations_q8_0, quantize_rmsnorm_q8_0, quantize_rmsnorm_q8_0_into,
    quantize_to_q8_blocks, softmax_simd, DequantStats, Q4_0Block, Q4_KBlock, Q5_KBlock, Q6_KBlock,
    Q8KSuperBlock, Q8_0Block, SimdBackend, BLOCK_SIZE, QK_K,
};

// ============================================================================
// Test 1-10: SimdBackend enum and Display
// ============================================================================

#[test]
fn test_simd_backend_display_avx2() {
    let backend = SimdBackend::Avx2;
    assert_eq!(format!("{backend}"), "AVX2");
}

#[test]
fn test_simd_backend_display_sse2() {
    let backend = SimdBackend::Sse2;
    assert_eq!(format!("{backend}"), "SSE2");
}

#[test]
fn test_simd_backend_display_neon() {
    let backend = SimdBackend::Neon;
    assert_eq!(format!("{backend}"), "NEON");
}

#[test]
fn test_simd_backend_display_scalar() {
    let backend = SimdBackend::Scalar;
    assert_eq!(format!("{backend}"), "Scalar");
}

#[test]
fn test_simd_backend_default() {
    let backend = SimdBackend::default();
    assert_eq!(backend, SimdBackend::Scalar);
}

#[test]
fn test_simd_backend_clone() {
    let backend = SimdBackend::Avx2;
    let cloned = backend;
    assert_eq!(backend, cloned);
}

#[test]
fn test_simd_backend_debug() {
    let backend = SimdBackend::Avx2;
    let debug = format!("{backend:?}");
    assert!(debug.contains("Avx2"));
}

#[test]
fn test_detect_simd_backend_returns_valid() {
    let backend = detect_simd_backend();
    match backend {
        SimdBackend::Avx2 | SimdBackend::Sse2 | SimdBackend::Neon | SimdBackend::Scalar => (),
    }
}

// ============================================================================
// Test 11-20: DequantStats struct
// ============================================================================

#[test]
fn test_dequant_stats_default() {
    let stats = DequantStats::default();
    assert_eq!(stats.blocks_processed, 0);
    assert_eq!(stats.bytes_processed, 0);
    assert_eq!(stats.simd_backend, SimdBackend::Scalar);
}

#[test]
fn test_dequant_stats_clone() {
    let stats = DequantStats {
        blocks_processed: 100,
        bytes_processed: 1600,
        simd_backend: SimdBackend::Avx2,
    };
    let cloned = stats;
    assert_eq!(cloned.blocks_processed, 100);
    assert_eq!(cloned.bytes_processed, 1600);
    assert_eq!(cloned.simd_backend, SimdBackend::Avx2);
}

#[test]
fn test_dequant_stats_debug() {
    let stats = DequantStats {
        blocks_processed: 50,
        bytes_processed: 800,
        simd_backend: SimdBackend::Sse2,
    };
    let debug = format!("{stats:?}");
    assert!(debug.contains("DequantStats"));
    assert!(debug.contains("50"));
}

// ============================================================================
// Test 21-30: Constants and Block Sizes
// ============================================================================

#[test]
fn test_block_size_constant() {
    assert_eq!(BLOCK_SIZE, 32);
}

#[test]
fn test_qk_k_constant() {
    assert_eq!(QK_K, 256);
}

#[test]
fn test_block_sizes_relationship() {
    assert_eq!(QK_K, 8 * BLOCK_SIZE);
}

// ============================================================================
// Test 31-40: Q8_0Block methods
// ============================================================================

#[test]
fn test_q8_0_block_quantize_all_zeros() {
    let values = [0.0f32; 32];
    let block = Q8_0Block::quantize(&values);
    assert!(block.scale > 0.0);
    for q in block.quants {
        assert_eq!(q, 0);
    }
}

#[test]
fn test_q8_0_block_quantize_positive() {
    let mut values = [0.0f32; 32];
    values[0] = 127.0;
    let block = Q8_0Block::quantize(&values);
    assert_eq!(block.quants[0], 127);
}

#[test]
fn test_q8_0_block_quantize_negative() {
    let mut values = [0.0f32; 32];
    values[0] = -127.0;
    let block = Q8_0Block::quantize(&values);
    assert_eq!(block.quants[0], -127);
}

#[test]
fn test_q8_0_block_dequantize_roundtrip() {
    let mut values = [0.0f32; 32];
    for i in 0..32 {
        values[i] = (i as f32 - 16.0) * 2.0;
    }
    let block = Q8_0Block::quantize(&values);
    let dequant = block.dequantize();
    for i in 0..32 {
        let diff = (values[i] - dequant[i]).abs();
        assert!(diff < 1.0, "diff too large at {i}: {diff}");
    }
}

#[test]
fn test_q8_0_block_quantization_error() {
    let mut values = [0.0f32; 32];
    for i in 0..32 {
        values[i] = i as f32;
    }
    let block = Q8_0Block::quantize(&values);
    let error = block.quantization_error(&values);
    assert!(error < 1.0);
}

#[test]
fn test_q8_0_block_relative_error_near_zero() {
    let values = [1e-15f32; 32];
    let block = Q8_0Block::quantize(&values);
    let error = block.relative_error(&values);
    assert_eq!(error, 0.0);
}

// ============================================================================
// Test 41-50: Q8KSuperBlock methods
// ============================================================================

#[test]
fn test_q8k_super_block_quantize_all_zeros() {
    let values = [0.0f32; 256];
    let block = Q8KSuperBlock::quantize(&values);
    assert!(block.scale > 0.0);
    for q in block.quants {
        assert_eq!(q, 0);
    }
}

#[test]
fn test_q8k_super_block_quantize_max_value() {
    let mut values = [0.0f32; 256];
    values[0] = 127.0;
    let block = Q8KSuperBlock::quantize(&values);
    assert_eq!(block.quants[0], 127);
}

#[test]
fn test_q8k_super_block_dequantize() {
    let mut values = [0.0f32; 256];
    for i in 0..256 {
        values[i] = (i as f32 - 128.0) * 0.5;
    }
    let block = Q8KSuperBlock::quantize(&values);
    let dequant = block.dequantize();
    for i in 0..256 {
        let diff = (values[i] - dequant[i]).abs();
        assert!(diff < 1.0);
    }
}

#[test]
fn test_q8k_super_block_quantize_into() {
    let values = [1.0f32; 256];
    let mut scale = 0.0f32;
    let mut quants = [0i8; 256];
    Q8KSuperBlock::quantize_into(&values, &mut scale, &mut quants);
    assert!(scale > 0.0);
    for q in quants {
        assert_eq!(q, 127);
    }
}

// ============================================================================
// Test 51-60: quantize_to_q8_blocks
// ============================================================================

#[test]
fn test_quantize_to_q8_blocks_single_block() {
    let values: Vec<f32> = (0..32).map(|i| i as f32).collect();
    let blocks = quantize_to_q8_blocks(&values).expect("quantize");
    assert_eq!(blocks.len(), 1);
}

#[test]
fn test_quantize_to_q8_blocks_multiple_blocks() {
    let values: Vec<f32> = (0..64).map(|i| i as f32).collect();
    let blocks = quantize_to_q8_blocks(&values).expect("quantize");
    assert_eq!(blocks.len(), 2);
}

#[test]
fn test_quantize_to_q8_blocks_empty() {
    let values: Vec<f32> = vec![];
    let blocks = quantize_to_q8_blocks(&values).expect("quantize");
    assert!(blocks.is_empty());
}

#[test]
fn test_quantize_to_q8_blocks_not_multiple() {
    let values: Vec<f32> = (0..50).map(|i| i as f32).collect();
    let result = quantize_to_q8_blocks(&values);
    assert!(result.is_err());
}

// ============================================================================
// Test 61-70: f16_to_f32 conversion
// ============================================================================

#[test]
fn test_f16_to_f32_zero() {
    let result = f16_to_f32(0x0000);
    assert_eq!(result, 0.0);
}

#[test]
fn test_f16_to_f32_one() {
    let result = f16_to_f32(0x3C00);
    assert!((result - 1.0).abs() < 1e-6);
}

#[test]
fn test_f16_to_f32_negative() {
    let result = f16_to_f32(0xBC00);
    assert!((result + 1.0).abs() < 1e-6);
}

#[test]
fn test_f16_to_f32_inf() {
    let result = f16_to_f32(0x7C00);
    assert!(result.is_infinite() && result > 0.0);
}

#[test]
fn test_f16_to_f32_neg_inf() {
    let result = f16_to_f32(0xFC00);
    assert!(result.is_infinite() && result < 0.0);
}

#[test]
fn test_f16_to_f32_nan() {
    let result = f16_to_f32(0x7E00);
    assert!(result.is_nan());
}

// ============================================================================
// Test 71-80: dequantize_f16
// ============================================================================

#[test]
fn test_dequantize_f16_empty() {
    let result = dequantize_f16(&[]).expect("dequantize");
    assert!(result.is_empty());
}

#[test]
fn test_dequantize_f16_single_value() {
    let data = [0x00, 0x3C];
    let result = dequantize_f16(&data).expect("dequantize");
    assert_eq!(result.len(), 1);
    assert!((result[0] - 1.0).abs() < 1e-6);
}

#[test]
fn test_dequantize_f16_odd_bytes_error() {
    let data = [0x00, 0x3C, 0x00];
    let result = dequantize_f16(&data);
    assert!(result.is_err());
}

// ============================================================================
// Test 81-90: dequantize_q4_0 edge cases
// ============================================================================

#[test]
fn test_dequantize_q4_0_empty() {
    let result = dequantize_q4_0(&[]).expect("dequantize");
    assert!(result.is_empty());
}

#[test]
fn test_dequantize_q4_0_single_block() {
    // Q4_0: 2 bytes f16 scale + 16 bytes quants = 18 bytes
    let mut data = vec![0u8; 18];
    // Set f16 scale = 1.0 (0x3C00)
    data[0..2].copy_from_slice(&0x3C00u16.to_le_bytes());
    let result = dequantize_q4_0(&data).expect("dequantize");
    assert_eq!(result.len(), 32);
}

#[test]
fn test_dequantize_q4_0_invalid_size() {
    let data = vec![0u8; 10];
    let result = dequantize_q4_0(&data);
    assert!(result.is_err());
}

// ============================================================================
// Test 91-100: dequantize_q8_0 edge cases
// ============================================================================

#[test]
fn test_dequantize_q8_0_empty() {
    let result = dequantize_q8_0(&[]).expect("dequantize");
    assert!(result.is_empty());
}

#[test]
fn test_dequantize_q8_0_single_block() {
    // Q8_0: 2 bytes f16 scale + 32 bytes quants = 34 bytes
    let mut data = vec![0u8; 34];
    // Set f16 scale = 1.0 (0x3C00)
    data[0..2].copy_from_slice(&0x3C00u16.to_le_bytes());
    data[2] = 127;
    let result = dequantize_q8_0(&data).expect("dequantize");
    assert_eq!(result.len(), 32);
    assert!((result[0] - 127.0).abs() < 1e-6);
}

#[test]
fn test_dequantize_q8_0_invalid_size() {
    let data = vec![0u8; 10];
    let result = dequantize_q8_0(&data);
    assert!(result.is_err());
}

// ============================================================================
// Test 101-110: dequantize_q4_k edge cases
// ============================================================================

#[test]
fn test_dequantize_q4_k_empty() {
    let result = dequantize_q4_k(&[]).expect("dequantize");
    assert!(result.is_empty());
}

#[test]
fn test_dequantize_q4_k_single_super_block() {
    let data = vec![0u8; 144];
    let result = dequantize_q4_k(&data).expect("dequantize");
    assert_eq!(result.len(), 256);
}

#[test]
fn test_dequantize_q4_k_invalid_size() {
    let data = vec![0u8; 100];
    let result = dequantize_q4_k(&data);
    assert!(result.is_err());
}

#[test]
fn test_dequantize_q4_k_parallel_empty() {
    let result = dequantize_q4_k_parallel(&[]).expect("dequantize");
    assert!(result.is_empty());
}

#[test]
fn test_dequantize_q4_k_simd_empty() {
    let result = dequantize_q4_k_simd(&[]).expect("dequantize");
    assert!(result.is_empty());
}

// ============================================================================
// Test 111-120: dequantize_q5_k and q6_k edge cases
// ============================================================================

#[test]
fn test_dequantize_q5_k_empty() {
    let result = dequantize_q5_k(&[]).expect("dequantize");
    assert!(result.is_empty());
}

#[test]
fn test_dequantize_q5_k_single_super_block() {
    let data = vec![0u8; 176];
    let result = dequantize_q5_k(&data).expect("dequantize");
    assert_eq!(result.len(), 256);
}

#[test]
fn test_dequantize_q6_k_empty() {
    let result = dequantize_q6_k(&[]).expect("dequantize");
    assert!(result.is_empty());
}

#[test]
fn test_dequantize_q6_k_single_super_block() {
    let data = vec![0u8; 210];
    let result = dequantize_q6_k(&data).expect("dequantize");
    assert_eq!(result.len(), 256);
}

// ============================================================================
// Test 121-130: dequantize_q4_1, q5_0, q5_1
// ============================================================================

#[test]
fn test_dequantize_q4_1_empty() {
    let result = dequantize_q4_1(&[]).expect("dequantize");
    assert!(result.is_empty());
}

#[test]
fn test_dequantize_q4_1_single_block() {
    // Q4_1: 2 bytes scale + 2 bytes min + 16 bytes quants = 20 bytes
    let data = vec![0u8; 20];
    let result = dequantize_q4_1(&data).expect("dequantize");
    assert_eq!(result.len(), 32);
}

#[test]
fn test_dequantize_q5_0_empty() {
    let result = dequantize_q5_0(&[]).expect("dequantize");
    assert!(result.is_empty());
}

#[test]
fn test_dequantize_q5_0_single_block() {
    // Q5_0: 2 bytes scale + 4 bytes high bits + 16 bytes low bits = 22 bytes
    let data = vec![0u8; 22];
    let result = dequantize_q5_0(&data).expect("dequantize");
    assert_eq!(result.len(), 32);
}

#[test]
fn test_dequantize_q5_1_empty() {
    let result = dequantize_q5_1(&[]).expect("dequantize");
    assert!(result.is_empty());
}

#[test]
fn test_dequantize_q5_1_single_block() {
    // Q5_1: 2 bytes scale + 2 bytes min + 4 bytes high bits + 16 bytes low bits = 24 bytes
    let data = vec![0u8; 24];
    let result = dequantize_q5_1(&data).expect("dequantize");
    assert_eq!(result.len(), 32);
}

// ============================================================================
// Test 131-140: dequantize_q8_0 parallel and SIMD
// ============================================================================

#[test]
fn test_dequantize_q8_0_parallel_empty() {
    let result = dequantize_q8_0_parallel(&[]).expect("dequantize");
    assert!(result.is_empty());
}

#[test]
fn test_dequantize_q8_0_simd_empty() {
    let result = dequantize_q8_0_simd(&[]).expect("dequantize");
    assert!(result.is_empty());
}

// ============================================================================
// Test 141-150: fused_q4k_dot variations
// ============================================================================

#[test]
fn test_fused_q4k_dot_size_mismatch() {
    let data = vec![0u8; 144];
    let activations = vec![1.0f32; 128];
    let result = fused_q4k_dot(&data, &activations);
    assert!(result.is_err());
}

#[test]
fn test_fused_q4k_dot_simd_size_mismatch() {
    let data = vec![0u8; 144];
    let activations = vec![1.0f32; 128];
    let result = fused_q4k_dot_simd(&data, &activations);
    assert!(result.is_err());
}

#[test]
fn test_fused_q4k_dot_all_zeros() {
    let data = vec![0u8; 144];
    let activations = vec![0.0f32; 256];
    let result = fused_q4k_dot(&data, &activations).expect("dot");
    assert_eq!(result, 0.0);
}

// ============================================================================
// Test 151-160: fused_q5k_dot and fused_q6k_dot
// ============================================================================

#[test]
fn test_fused_q5k_dot_size_mismatch() {
    let data = vec![0u8; 176];
    let activations = vec![1.0f32; 128];
    let result = fused_q5k_dot(&data, &activations);
    assert!(result.is_err());
}

#[test]
fn test_fused_q5k_dot_simd_all_zeros() {
    let data = vec![0u8; 176];
    let activations = vec![0.0f32; 256];
    let result = fused_q5k_dot_simd(&data, &activations).expect("dot");
    assert_eq!(result, 0.0);
}

#[test]
fn test_fused_q6k_dot_size_mismatch() {
    let data = vec![0u8; 210];
    let activations = vec![1.0f32; 128];
    let result = fused_q6k_dot(&data, &activations);
    assert!(result.is_err());
}

#[test]
fn test_fused_q6k_dot_simd_all_zeros() {
    let data = vec![0u8; 210];
    let activations = vec![0.0f32; 256];
    let result = fused_q6k_dot_simd(&data, &activations).expect("dot");
    assert_eq!(result, 0.0);
}

// ============================================================================
// Test 161-170: parallel matvec functions
// ============================================================================

#[test]
fn test_fused_q4k_parallel_matvec_empty() {
    let weights: Vec<u8> = vec![];
    let activations: Vec<f32> = vec![];
    let result = fused_q4k_parallel_matvec(&weights, &activations, 0, 0);
    // Empty 0x0 dimensions are valid and return empty result
    assert!(result.is_ok());
    assert!(result.unwrap().is_empty());
}

#[test]
fn test_fused_q4k_parallel_matvec_into_empty() {
    let weights: Vec<u8> = vec![];
    let activations: Vec<f32> = vec![];
    let mut output: Vec<f32> = vec![];
    let result = fused_q4k_parallel_matvec_into(&weights, &activations, 0, 0, &mut output);
    // Empty 0x0 dimensions are valid
    assert!(result.is_ok());
}

#[test]
fn test_fused_q5k_parallel_matvec_empty() {
    let weights: Vec<u8> = vec![];
    let activations: Vec<f32> = vec![];
    let result = fused_q5k_parallel_matvec(&weights, &activations, 0, 0);
    // Empty 0x0 dimensions are valid
    assert!(result.is_ok());
    assert!(result.unwrap().is_empty());
}

#[test]
fn test_fused_q6k_parallel_matvec_empty() {
    let weights: Vec<u8> = vec![];
    let activations: Vec<f32> = vec![];
    let result = fused_q6k_parallel_matvec(&weights, &activations, 0, 0);
    // Empty 0x0 dimensions are valid
    assert!(result.is_ok());
    assert!(result.unwrap().is_empty());
}

#[test]
fn test_fused_q5k_parallel_matvec_into_empty() {
    let weights: Vec<u8> = vec![];
    let activations: Vec<f32> = vec![];
    let mut output: Vec<f32> = vec![];
    let result = fused_q5k_parallel_matvec_into(&weights, &activations, 0, 0, &mut output);
    // Empty 0x0 dimensions are valid
    assert!(result.is_ok());
}

#[test]
fn test_fused_q6k_parallel_matvec_into_empty() {
    let weights: Vec<u8> = vec![];
    let activations: Vec<f32> = vec![];
    let mut output: Vec<f32> = vec![];
    let result = fused_q6k_parallel_matvec_into(&weights, &activations, 0, 0, &mut output);
    // Empty 0x0 dimensions are valid
    assert!(result.is_ok());
}

// ============================================================================
// Test 171-180: Q4_KBlock, Q5_KBlock, Q6_KBlock structs
// ============================================================================

#[test]
fn test_q4k_block_default_values() {
    let block = Q4_KBlock {
        d: 0.0,
        dmin: 0.0,
        scales: [0u8; 12],
        qs: [0u8; 128],
    };
    assert_eq!(block.d, 0.0);
    assert_eq!(block.dmin, 0.0);
}

#[test]
fn test_q4k_block_debug() {
    let block = Q4_KBlock {
        d: 1.0,
        dmin: 0.5,
        scales: [0u8; 12],
        qs: [0u8; 128],
    };
    let debug = format!("{block:?}");
    assert!(debug.contains("Q4_KBlock"));
}

#[test]
fn test_q5k_block_default_values() {
    let block = Q5_KBlock {
        d: 0.0,
        dmin: 0.0,
        scales: [0u8; 12],
        qh: [0u8; 32],
        qs: [0u8; 128],
    };
    assert_eq!(block.d, 0.0);
}

#[test]
fn test_q6k_block_default_values() {
    let block = Q6_KBlock {
        qs: [0u8; 128],
        qh: [0u8; 64],
        scales: [0i8; 16],
        d: 0.0,
    };
    assert_eq!(block.d, 0.0);
}

// ============================================================================
// Test 181-190: quantize_activations functions
// ============================================================================

#[test]
fn test_quantize_activations_q8_0_empty() {
    let (scales, quants) = quantize_activations_q8_0(&[]);
    assert!(scales.is_empty());
    assert!(quants.is_empty());
}

#[test]
fn test_quantize_activations_q8_0_partial_block() {
    let activations: Vec<f32> = (0..16).map(|i| i as f32).collect();
    let (scales, quants) = quantize_activations_q8_0(&activations);
    assert_eq!(scales.len(), 1);
    assert_eq!(quants.len(), 32);
}

#[test]
fn test_quantize_activations_q8_0_full_block() {
    let activations: Vec<f32> = (0..32).map(|i| i as f32).collect();
    let (scales, quants) = quantize_activations_q8_0(&activations);
    assert_eq!(scales.len(), 1);
    assert_eq!(quants.len(), 32);
}

// ============================================================================
// Test 191-200: softmax_simd edge cases
// ============================================================================

#[test]
fn test_softmax_simd_empty() {
    let mut x: Vec<f32> = vec![];
    softmax_simd(&mut x);
    assert!(x.is_empty());
}

#[test]
fn test_softmax_simd_single() {
    let mut x = vec![1.0f32];
    softmax_simd(&mut x);
    assert!((x[0] - 1.0).abs() < 1e-6);
}

#[test]
fn test_softmax_simd_two_equal() {
    let mut x = vec![0.0f32, 0.0f32];
    softmax_simd(&mut x);
    assert!((x[0] - 0.5).abs() < 1e-6);
    assert!((x[1] - 0.5).abs() < 1e-6);
}

#[test]
fn test_softmax_simd_numerical_stability() {
    let mut x = vec![1000.0f32, 1000.0f32, 1000.0f32];
    softmax_simd(&mut x);
    let sum: f32 = x.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);
}

// ============================================================================
// Test 201-210: fused_swiglu_simd edge cases
// ============================================================================

#[test]
fn test_fused_swiglu_simd_empty() {
    let mut gate: Vec<f32> = vec![];
    let up: Vec<f32> = vec![];
    fused_swiglu_simd(&mut gate, &up);
    assert!(gate.is_empty());
}

#[test]
fn test_fused_swiglu_simd_single() {
    let mut gate = vec![0.0f32];
    let up = vec![1.0f32];
    fused_swiglu_simd(&mut gate, &up);
    assert!(gate[0].abs() < 1e-6);
}

#[test]
fn test_fused_swiglu_simd_positive() {
    let mut gate = vec![1.0f32; 8];
    let up = vec![1.0f32; 8];
    fused_swiglu_simd(&mut gate, &up);
    for v in &gate {
        assert!(*v > 0.0);
    }
}

// ============================================================================
// Test 211-220: quantize_rmsnorm functions
// ============================================================================

#[test]
fn test_quantize_rmsnorm_q8_0_basic() {
    let input: Vec<f32> = (0..32).map(|i| i as f32 / 32.0).collect();
    let weights = vec![1.0f32; 32];
    let (scales, quants) = quantize_rmsnorm_q8_0(&input, &weights, 1e-5);
    // 32 elements = 1 block, so 1 scale
    assert_eq!(scales.len(), 1);
    // quants padded to block size of 32
    assert_eq!(quants.len(), 32);
}

#[test]
fn test_quantize_rmsnorm_q8_0_into_basic() {
    let input: Vec<f32> = (0..32).map(|i| i as f32 / 32.0).collect();
    let weights = vec![1.0f32; 32];
    let mut norm = vec![0.0f32; 32];
    let mut quants = vec![0i8; 32];
    quantize_rmsnorm_q8_0_into(&input, &weights, 1e-5, &mut norm, &mut quants);
    assert!(norm.iter().any(|&v| v != 0.0));
}

// ============================================================================
// Test 221-230: Additional struct tests
// ============================================================================

#[test]
fn test_q4_0_block_default_values() {
    let block = Q4_0Block {
        scale: 0.0,
        quants: [0u8; 16],
    };
    assert_eq!(block.scale, 0.0);
    assert_eq!(block.quants.len(), 16);
}

#[test]
fn test_q4_0_block_debug() {
    let block = Q4_0Block {
        scale: 1.5,
        quants: [0u8; 16],
    };
    let debug = format!("{block:?}");
    assert!(debug.contains("Q4_0Block"));
}

#[test]
fn test_q4_0_block_clone() {
    let block = Q4_0Block {
        scale: 2.0,
        quants: [0xFF; 16],
    };
    let cloned = block;
    assert_eq!(cloned.scale, 2.0);
    assert_eq!(cloned.quants[0], 0xFF);
}

#[test]
fn test_q8_0_block_clone() {
    let block = Q8_0Block {
        scale: 1.0,
        quants: [127i8; 32],
    };
    let cloned = block;
    assert_eq!(cloned.scale, 1.0);
    assert_eq!(cloned.quants[0], 127);
}

#[test]
fn test_q8k_super_block_clone() {
    let block = Q8KSuperBlock {
        scale: 0.5,
        quants: [64i8; 256],
    };
    let cloned = block;
    assert_eq!(cloned.scale, 0.5);
    assert_eq!(cloned.quants[0], 64);
}

#[test]
fn test_q8k_super_block_debug() {
    let block = Q8KSuperBlock {
        scale: 0.5,
        quants: [0i8; 256],
    };
    let debug = format!("{block:?}");
    assert!(debug.contains("Q8KSuperBlock"));
}

// ============================================================================
// Test 231-240: Edge case error tests
// ============================================================================

#[test]
fn test_fused_q4k_tiled_matvec_empty() {
    let weights: Vec<u8> = vec![];
    let activations: Vec<f32> = vec![];
    let result = fused_q4k_tiled_matvec(&weights, &activations, 0, 0, Some(64));
    // Empty 0x0 dimensions are valid
    assert!(result.is_ok());
    assert!(result.unwrap().is_empty());
}
