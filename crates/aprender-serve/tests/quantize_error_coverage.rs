//! Error path coverage tests for quantize.rs
//!
//! Targets: quantize.rs (76.07% -> 85%+)
//!
//! EXTREME TDD: Tests all error conditions and edge cases

use realizar::quantize::{
    dequantize_f16, dequantize_q4_0, dequantize_q4_1, dequantize_q4_k, dequantize_q5_0,
    dequantize_q5_1, dequantize_q5_k, dequantize_q6_k, dequantize_q8_0, f16_to_f32, fused_q4k_dot,
    fused_q4k_dot_simd, fused_q4k_parallel_matvec, fused_q4k_parallel_matvec_into,
    fused_q4k_q8_dot, fused_q4k_q8k_dot, fused_q4k_q8k_dot_simd, fused_q5k_dot, fused_q5k_dot_simd,
    fused_q5k_parallel_matvec, fused_q5k_parallel_matvec_into, fused_q6k_dot, fused_q6k_dot_simd,
    fused_q6k_parallel_matvec, fused_q6k_parallel_matvec_into, quantize_activations_q8_0,
    quantize_activations_q8k_into, quantize_rmsnorm_q8_0, quantize_rmsnorm_q8_0_into,
    quantize_to_q8_blocks, softmax_simd, Q8_0Block,
};

// =============================================================================
// UNIT TESTS: dequantize_q4_0 error paths
// =============================================================================

#[test]
fn test_dequantize_q4_0_empty() {
    let result = dequantize_q4_0(&[]);
    // May return Ok with empty vec or Err - just verify it doesn't panic
    let _ = result;
}

#[test]
fn test_dequantize_q4_0_not_multiple_of_block_size() {
    // Q4_0 block is 18 bytes (2 + 16)
    let data = vec![0u8; 17]; // Not a multiple of 18
    let result = dequantize_q4_0(&data);
    assert!(result.is_err());
}

#[test]
fn test_dequantize_q4_0_partial_block() {
    let data = vec![0u8; 10]; // Less than one block
    let result = dequantize_q4_0(&data);
    assert!(result.is_err());
}

// =============================================================================
// UNIT TESTS: dequantize_q8_0 error paths
// =============================================================================

#[test]
fn test_dequantize_q8_0_empty() {
    let result = dequantize_q8_0(&[]);
    let _ = result;
}

#[test]
fn test_dequantize_q8_0_not_multiple_of_block_size() {
    // Q8_0 block is 34 bytes (2 + 32)
    let data = vec![0u8; 33]; // Not a multiple of 34
    let result = dequantize_q8_0(&data);
    assert!(result.is_err());
}

#[test]
fn test_dequantize_q8_0_partial_block() {
    let data = vec![0u8; 20]; // Less than one block
    let result = dequantize_q8_0(&data);
    assert!(result.is_err());
}

// =============================================================================
// UNIT TESTS: dequantize_f16 error paths
// =============================================================================

#[test]
fn test_dequantize_f16_empty() {
    let result = dequantize_f16(&[]);
    let _ = result;
}

#[test]
fn test_dequantize_f16_odd_bytes() {
    // F16 requires 2 bytes per value
    let data = vec![0u8; 5]; // Odd number
    let result = dequantize_f16(&data);
    assert!(result.is_err());
}

// =============================================================================
// UNIT TESTS: dequantize_q4_1 error paths
// =============================================================================

#[test]
fn test_dequantize_q4_1_empty() {
    let result = dequantize_q4_1(&[]);
    let _ = result;
}

#[test]
fn test_dequantize_q4_1_not_multiple_of_block_size() {
    // Q4_1 block is 20 bytes (2 + 2 + 16)
    let data = vec![0u8; 19];
    let result = dequantize_q4_1(&data);
    assert!(result.is_err());
}

// =============================================================================
// UNIT TESTS: dequantize_q5_0 error paths
// =============================================================================

#[test]
fn test_dequantize_q5_0_empty() {
    let result = dequantize_q5_0(&[]);
    let _ = result;
}

#[test]
fn test_dequantize_q5_0_not_multiple_of_block_size() {
    // Q5_0 block is 22 bytes
    let data = vec![0u8; 21];
    let result = dequantize_q5_0(&data);
    assert!(result.is_err());
}

// =============================================================================
// UNIT TESTS: dequantize_q5_1 error paths
// =============================================================================

#[test]
fn test_dequantize_q5_1_empty() {
    let result = dequantize_q5_1(&[]);
    let _ = result;
}

#[test]
fn test_dequantize_q5_1_not_multiple_of_block_size() {
    // Q5_1 block is 24 bytes
    let data = vec![0u8; 23];
    let result = dequantize_q5_1(&data);
    assert!(result.is_err());
}

// =============================================================================
// UNIT TESTS: dequantize_q4_k error paths
// =============================================================================

#[test]
fn test_dequantize_q4_k_empty() {
    let result = dequantize_q4_k(&[]);
    let _ = result;
}

#[test]
fn test_dequantize_q4_k_not_multiple_of_block_size() {
    // Q4_K block is 144 bytes
    let data = vec![0u8; 143];
    let result = dequantize_q4_k(&data);
    assert!(result.is_err());
}

#[test]
fn test_dequantize_q4_k_partial_block() {
    let data = vec![0u8; 100];
    let result = dequantize_q4_k(&data);
    assert!(result.is_err());
}

// =============================================================================
// UNIT TESTS: dequantize_q5_k error paths
// =============================================================================

#[test]
fn test_dequantize_q5_k_empty() {
    let result = dequantize_q5_k(&[]);
    let _ = result;
}

#[test]
fn test_dequantize_q5_k_not_multiple_of_block_size() {
    // Q5_K block is 176 bytes
    let data = vec![0u8; 175];
    let result = dequantize_q5_k(&data);
    assert!(result.is_err());
}

// =============================================================================
// UNIT TESTS: dequantize_q6_k error paths
// =============================================================================

#[test]
fn test_dequantize_q6_k_empty() {
    let result = dequantize_q6_k(&[]);
    let _ = result;
}

#[test]
fn test_dequantize_q6_k_not_multiple_of_block_size() {
    // Q6_K block is 210 bytes
    let data = vec![0u8; 209];
    let result = dequantize_q6_k(&data);
    assert!(result.is_err());
}

// =============================================================================
// UNIT TESTS: fused_q4k_dot error paths
// =============================================================================

#[test]
fn test_fused_q4k_dot_empty_data() {
    let result = fused_q4k_dot(&[], &[1.0, 2.0, 3.0]);
    assert!(result.is_err());
}

#[test]
fn test_fused_q4k_dot_mismatched_size() {
    // Q4_K block is 144 bytes for 256 values
    let data = vec![0u8; 144];
    let activations = vec![1.0f32; 128]; // Wrong size - should be 256
    let result = fused_q4k_dot(&data, &activations);
    assert!(result.is_err());
}

#[test]
fn test_fused_q4k_dot_activations_too_small() {
    let data = vec![0u8; 144];
    let activations = vec![1.0f32; 100];
    let result = fused_q4k_dot(&data, &activations);
    assert!(result.is_err());
}

// =============================================================================
// UNIT TESTS: fused_q4k_dot_simd error paths
// =============================================================================

#[test]
fn test_fused_q4k_dot_simd_empty_data() {
    let result = fused_q4k_dot_simd(&[], &[1.0, 2.0, 3.0]);
    assert!(result.is_err());
}

#[test]
fn test_fused_q4k_dot_simd_mismatched_size() {
    let data = vec![0u8; 144];
    let activations = vec![1.0f32; 128];
    let result = fused_q4k_dot_simd(&data, &activations);
    assert!(result.is_err());
}

// =============================================================================
// UNIT TESTS: fused_q4k_q8_dot error paths
// =============================================================================

#[test]
fn test_fused_q4k_q8_dot_empty_data() {
    let blocks: Vec<Q8_0Block> = vec![];
    let result = fused_q4k_q8_dot(&[], &blocks);
    let _ = result;
}

#[test]
fn test_fused_q4k_q8_dot_mismatched_blocks() {
    let data = vec![0u8; 144]; // 1 Q4_K block for 256 values
                               // Create Q8_0 blocks for fewer values
    let blocks = vec![Q8_0Block::quantize(&[0.0; 32])]; // Only 32 values
    let result = fused_q4k_q8_dot(&data, &blocks);
    assert!(result.is_err());
}

// =============================================================================
// UNIT TESTS: fused_q4k_q8k_dot error paths
// =============================================================================

#[test]
fn test_fused_q4k_q8k_dot_empty_data() {
    let result = fused_q4k_q8k_dot(&[], &[], &[]);
    let _ = result;
}

#[test]
fn test_fused_q4k_q8k_dot_mismatched_sizes() {
    let data = vec![0u8; 144]; // 1 Q4_K block for 256 values
    let scales = vec![1.0f32]; // 1 scale for 256 quants
    let quants = vec![0i8; 128]; // Wrong size
    let result = fused_q4k_q8k_dot(&data, &scales, &quants);
    assert!(result.is_err());
}

#[test]
fn test_fused_q4k_q8k_dot_scales_mismatch() {
    let data = vec![0u8; 144]; // 1 Q4_K block for 256 values
    let scales = vec![1.0f32; 2]; // Wrong number of scales
    let quants = vec![0i8; 256];
    let result = fused_q4k_q8k_dot(&data, &scales, &quants);
    let _ = result;
}

// =============================================================================
// UNIT TESTS: fused_q4k_q8k_dot_simd error paths
// =============================================================================

#[test]
fn test_fused_q4k_q8k_dot_simd_empty() {
    let result = fused_q4k_q8k_dot_simd(&[], &[], &[]);
    let _ = result;
}

// =============================================================================
// UNIT TESTS: fused_q5k_dot error paths
// =============================================================================

#[test]
fn test_fused_q5k_dot_empty_data() {
    let result = fused_q5k_dot(&[], &[1.0, 2.0, 3.0]);
    assert!(result.is_err());
}

#[test]
fn test_fused_q5k_dot_mismatched_size() {
    let data = vec![0u8; 176]; // 1 Q5_K block
    let activations = vec![1.0f32; 128]; // Wrong size
    let result = fused_q5k_dot(&data, &activations);
    assert!(result.is_err());
}

// =============================================================================
// UNIT TESTS: fused_q5k_dot_simd error paths
// =============================================================================

#[test]
fn test_fused_q5k_dot_simd_empty_data() {
    let result = fused_q5k_dot_simd(&[], &[1.0, 2.0, 3.0]);
    assert!(result.is_err());
}

// =============================================================================
// UNIT TESTS: fused_q6k_dot error paths
// =============================================================================

#[test]
fn test_fused_q6k_dot_empty_data() {
    let result = fused_q6k_dot(&[], &[1.0, 2.0, 3.0]);
    assert!(result.is_err());
}

#[test]
fn test_fused_q6k_dot_mismatched_size() {
    let data = vec![0u8; 210]; // 1 Q6_K block
    let activations = vec![1.0f32; 128]; // Wrong size
    let result = fused_q6k_dot(&data, &activations);
    assert!(result.is_err());
}

// =============================================================================
// UNIT TESTS: fused_q6k_dot_simd error paths
// =============================================================================

#[test]
fn test_fused_q6k_dot_simd_empty_data() {
    let result = fused_q6k_dot_simd(&[], &[1.0, 2.0, 3.0]);
    assert!(result.is_err());
}

// =============================================================================
// UNIT TESTS: quantize_to_q8_blocks error paths
// =============================================================================

#[test]
fn test_quantize_to_q8_blocks_empty() {
    let result = quantize_to_q8_blocks(&[]);
    let _ = result;
}

#[test]
fn test_quantize_to_q8_blocks_not_multiple_of_32() {
    // Q8_0 blocks require multiples of 32
    let values = vec![1.0f32; 33];
    let result = quantize_to_q8_blocks(&values);
    assert!(result.is_err());
}

#[test]
fn test_quantize_to_q8_blocks_15_values() {
    let values = vec![1.0f32; 15];
    let result = quantize_to_q8_blocks(&values);
    assert!(result.is_err());
}

// =============================================================================
// UNIT TESTS: quantize_activations_q8k_into error paths
// =============================================================================

#[test]
fn test_quantize_activations_q8k_into_empty() {
    let mut scale_out = vec![0.0f32; 1];
    let mut quants_out = vec![0i8; 0];
    let result = quantize_activations_q8k_into(&[], &mut scale_out, &mut quants_out);
    let _ = result;
}

#[test]
fn test_quantize_activations_q8k_into_not_multiple_of_256() {
    let values = vec![1.0f32; 100];
    let mut scale_out = vec![0.0f32; 1];
    let mut quants_out = vec![0i8; 100];
    let result = quantize_activations_q8k_into(&values, &mut scale_out, &mut quants_out);
    assert!(result.is_err());
}

#[test]
fn test_quantize_activations_q8k_into_output_too_small() {
    let values = vec![1.0f32; 256];
    let mut scale_out = vec![0.0f32; 1];
    let mut quants_out = vec![0i8; 100]; // Too small
    let result = quantize_activations_q8k_into(&values, &mut scale_out, &mut quants_out);
    assert!(result.is_err());
}

// =============================================================================
// UNIT TESTS: fused_q4k_parallel_matvec error paths
// =============================================================================

#[test]
fn test_fused_q4k_parallel_matvec_empty() {
    let result = fused_q4k_parallel_matvec(&[], &[], 0, 0);
    // Just verify it doesn't panic and returns empty or error
    let _ = result;
}

// =============================================================================
// UNIT TESTS: fused_q4k_parallel_matvec_into error paths
// =============================================================================

#[test]
fn test_fused_q4k_parallel_matvec_into_empty() {
    let mut output = vec![0.0f32; 10];
    let _ = fused_q4k_parallel_matvec_into(&[], &[], 0, 0, &mut output);
    // Just verify it doesn't panic
}

// =============================================================================
// UNIT TESTS: fused_q5k_parallel_matvec error paths
// =============================================================================

#[test]
fn test_fused_q5k_parallel_matvec_empty() {
    let result = fused_q5k_parallel_matvec(&[], &[], 0, 0);
    let _ = result;
}

// =============================================================================
// UNIT TESTS: fused_q5k_parallel_matvec_into error paths
// =============================================================================

#[test]
fn test_fused_q5k_parallel_matvec_into_empty() {
    let mut output = vec![0.0f32; 10];
    let _ = fused_q5k_parallel_matvec_into(&[], &[], 0, 0, &mut output);
}

// =============================================================================
// UNIT TESTS: fused_q6k_parallel_matvec error paths
// =============================================================================

#[test]
fn test_fused_q6k_parallel_matvec_empty() {
    let result = fused_q6k_parallel_matvec(&[], &[], 0, 0);
    let _ = result;
}

// =============================================================================
// UNIT TESTS: fused_q6k_parallel_matvec_into error paths
// =============================================================================

#[test]
fn test_fused_q6k_parallel_matvec_into_empty() {
    let mut output = vec![0.0f32; 10];
    let _ = fused_q6k_parallel_matvec_into(&[], &[], 0, 0, &mut output);
}

// =============================================================================
// UNIT TESTS: Q8_0Block
// =============================================================================

#[test]
fn test_q8_0_block_quantize_zeros() {
    let values = [0.0f32; 32];
    let block = Q8_0Block::quantize(&values);
    let dequant = block.dequantize();
    // All zeros should stay zeros
    for val in &dequant {
        assert!(val.abs() < 1e-6);
    }
}

#[test]
fn test_q8_0_block_quantize_ones() {
    let values = [1.0f32; 32];
    let block = Q8_0Block::quantize(&values);
    let dequant = block.dequantize();
    // Should be close to 1.0
    for val in &dequant {
        assert!((val - 1.0).abs() < 0.1);
    }
}

#[test]
fn test_q8_0_block_quantize_negatives() {
    let values = [-1.0f32; 32];
    let block = Q8_0Block::quantize(&values);
    let dequant = block.dequantize();
    // Should be close to -1.0
    for val in &dequant {
        assert!((val + 1.0).abs() < 0.1);
    }
}

#[test]
fn test_q8_0_block_quantization_error() {
    let values = [1.0f32; 32];
    let block = Q8_0Block::quantize(&values);
    let error = block.quantization_error(&values);
    // Error should be small
    assert!(error < 0.1);
}

#[test]
fn test_q8_0_block_relative_error() {
    let values = [1.0f32; 32];
    let block = Q8_0Block::quantize(&values);
    let error = block.relative_error(&values);
    // Relative error should be small
    assert!(error < 0.1);
}

// =============================================================================
// UNIT TESTS: softmax_simd
// =============================================================================

#[test]
fn test_softmax_simd_single_element() {
    let mut x = vec![1.0f32];
    softmax_simd(&mut x);
    assert!((x[0] - 1.0).abs() < 1e-6);
}

#[test]
fn test_softmax_simd_two_elements() {
    let mut x = vec![0.0f32, 0.0];
    softmax_simd(&mut x);
    // Equal inputs -> equal outputs
    assert!((x[0] - 0.5).abs() < 1e-6);
    assert!((x[1] - 0.5).abs() < 1e-6);
}

#[test]
fn test_softmax_simd_large_difference() {
    let mut x = vec![100.0f32, 0.0];
    softmax_simd(&mut x);
    // Large difference -> winner takes all
    assert!(x[0] > 0.99);
    assert!(x[1] < 0.01);
}

#[test]
fn test_softmax_simd_negative_values() {
    let mut x = vec![-1.0f32, -2.0, -3.0];
    softmax_simd(&mut x);
    // Should sum to 1
    let sum: f32 = x.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);
}

#[test]
fn test_softmax_simd_16_elements() {
    let mut x = vec![1.0f32; 16];
    softmax_simd(&mut x);
    // Equal inputs -> equal outputs
    let expected = 1.0 / 16.0;
    for val in &x {
        assert!((val - expected).abs() < 1e-5);
    }
}

#[test]
fn test_softmax_simd_mixed_values() {
    let mut x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    softmax_simd(&mut x);
    // Sum should be 1
    let sum: f32 = x.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);
    // Should be monotonically increasing
    for i in 1..x.len() {
        assert!(x[i] > x[i - 1]);
    }
}

// =============================================================================
// UNIT TESTS: quantize_rmsnorm_q8_0
// =============================================================================

#[test]
fn test_quantize_rmsnorm_q8_0_empty() {
    let result = quantize_rmsnorm_q8_0(&[], &[], 1e-5);
    // Should handle empty gracefully
    let _ = result;
}

#[test]
fn test_quantize_rmsnorm_q8_0_single_value() {
    let input = vec![1.0f32];
    let weight = vec![1.0f32];
    let (normed, quants) = quantize_rmsnorm_q8_0(&input, &weight, 1e-5);
    // Single value normalization
    let _ = (normed, quants);
}

// =============================================================================
// UNIT TESTS: quantize_rmsnorm_q8_0_into
// =============================================================================

#[test]
fn test_quantize_rmsnorm_q8_0_into_basic() {
    let input = vec![1.0f32; 32];
    let weight = vec![1.0f32; 32];
    let mut normed_out = vec![0.0f32; 32];
    let mut quant_out = vec![0i8; 32];
    quantize_rmsnorm_q8_0_into(&input, &weight, 1e-5, &mut normed_out, &mut quant_out);
    // Just verify it doesn't panic
}

// =============================================================================
// UNIT TESTS: quantize_activations_q8_0
// =============================================================================

#[test]
fn test_quantize_activations_q8_0_zeros() {
    let activations = vec![0.0f32; 256];
    let (scales, quants) = quantize_activations_q8_0(&activations);
    // Zeros should produce zeros
    for q in &quants {
        assert_eq!(*q, 0);
    }
    let _ = scales;
}

#[test]
fn test_quantize_activations_q8_0_ones() {
    let activations = vec![1.0f32; 256];
    let (scales, quants) = quantize_activations_q8_0(&activations);
    // Should produce positive quants
    for q in &quants {
        assert!(*q >= 0);
    }
    let _ = scales;
}

// =============================================================================
// UNIT TESTS: f16_to_f32
// =============================================================================

#[test]
fn test_f16_to_f32_zero() {
    assert_eq!(f16_to_f32(0x0000), 0.0);
}

#[test]
fn test_f16_to_f32_negative_zero() {
    assert_eq!(f16_to_f32(0x8000), -0.0);
}

#[test]
fn test_f16_to_f32_one() {
    // FP16 1.0 = 0x3C00
    let result = f16_to_f32(0x3C00);
    assert!((result - 1.0).abs() < 1e-6);
}

#[test]
fn test_f16_to_f32_negative_one() {
    // FP16 -1.0 = 0xBC00
    let result = f16_to_f32(0xBC00);
    assert!((result + 1.0).abs() < 1e-6);
}

#[test]
fn test_f16_to_f32_two() {
    // FP16 2.0 = 0x4000
    let result = f16_to_f32(0x4000);
    assert!((result - 2.0).abs() < 1e-6);
}

#[test]
fn test_f16_to_f32_infinity() {
    // FP16 infinity = 0x7C00
    let result = f16_to_f32(0x7C00);
    assert!(result.is_infinite());
    assert!(result > 0.0);
}

#[test]
fn test_f16_to_f32_negative_infinity() {
    // FP16 -infinity = 0xFC00
    let result = f16_to_f32(0xFC00);
    assert!(result.is_infinite());
    assert!(result < 0.0);
}

#[test]
fn test_f16_to_f32_nan() {
    // FP16 NaN = 0x7E00 (one of many NaN representations)
    let result = f16_to_f32(0x7E00);
    assert!(result.is_nan());
}

#[test]
fn test_f16_to_f32_denormal() {
    // Smallest positive denormal
    let result = f16_to_f32(0x0001);
    assert!(result > 0.0);
    assert!(result < 1e-6);
}

#[test]
fn test_f16_to_f32_max_normal() {
    // Largest finite FP16 value = 0x7BFF ≈ 65504
    let result = f16_to_f32(0x7BFF);
    assert!((result - 65504.0).abs() < 100.0);
}

// =============================================================================
// VALID INPUT TESTS (ensure functions work correctly)
// =============================================================================

#[test]
fn test_dequantize_q4_0_valid() {
    // 1 block of Q4_0 (18 bytes)
    let mut data = vec![0u8; 18];
    // Scale bytes (f16 for 1.0 = 0x3C00)
    data[0..2].copy_from_slice(&0x3C00u16.to_le_bytes());
    let result = dequantize_q4_0(&data);
    assert!(result.is_ok());
    assert_eq!(result.unwrap().len(), 32);
}

#[test]
fn test_dequantize_q8_0_valid() {
    // 1 block of Q8_0 (34 bytes)
    let mut data = vec![0u8; 34];
    data[0..2].copy_from_slice(&0x3C00u16.to_le_bytes()); // scale = 1.0
    let result = dequantize_q8_0(&data);
    assert!(result.is_ok());
    assert_eq!(result.unwrap().len(), 32);
}

#[test]
fn test_dequantize_f16_valid() {
    // 4 f16 values
    let data = vec![0u8; 8];
    // 4 zeros
    let result = dequantize_f16(&data);
    assert!(result.is_ok());
    assert_eq!(result.unwrap().len(), 4);
}

#[test]
fn test_dequantize_q4_1_valid() {
    // 1 block of Q4_1 (20 bytes)
    let data = vec![0u8; 20];
    let result = dequantize_q4_1(&data);
    assert!(result.is_ok());
    assert_eq!(result.unwrap().len(), 32);
}

#[test]
fn test_dequantize_q5_0_valid() {
    // 1 block of Q5_0 (22 bytes)
    let data = vec![0u8; 22];
    let result = dequantize_q5_0(&data);
    assert!(result.is_ok());
    assert_eq!(result.unwrap().len(), 32);
}

#[test]
fn test_dequantize_q5_1_valid() {
    // 1 block of Q5_1 (24 bytes)
    let data = vec![0u8; 24];
    let result = dequantize_q5_1(&data);
    assert!(result.is_ok());
    assert_eq!(result.unwrap().len(), 32);
}

#[test]
fn test_dequantize_q4_k_valid() {
    // 1 block of Q4_K (144 bytes)
    let data = vec![0u8; 144];
    let result = dequantize_q4_k(&data);
    assert!(result.is_ok());
    assert_eq!(result.unwrap().len(), 256);
}

#[test]
fn test_dequantize_q5_k_valid() {
    // 1 block of Q5_K (176 bytes)
    let data = vec![0u8; 176];
    let result = dequantize_q5_k(&data);
    assert!(result.is_ok());
    assert_eq!(result.unwrap().len(), 256);
}

#[test]
fn test_dequantize_q6_k_valid() {
    // 1 block of Q6_K (210 bytes)
    let data = vec![0u8; 210];
    let result = dequantize_q6_k(&data);
    assert!(result.is_ok());
    assert_eq!(result.unwrap().len(), 256);
}

#[test]
fn test_quantize_to_q8_blocks_valid() {
    let values = vec![1.0f32; 32];
    let result = quantize_to_q8_blocks(&values);
    assert!(result.is_ok());
    assert_eq!(result.unwrap().len(), 1);
}

#[test]
fn test_quantize_to_q8_blocks_multiple() {
    let values = vec![1.0f32; 64];
    let result = quantize_to_q8_blocks(&values);
    assert!(result.is_ok());
    assert_eq!(result.unwrap().len(), 2);
}

#[test]
fn test_quantize_activations_q8k_into_valid() {
    let values = vec![1.0f32; 256];
    let mut scale_out = vec![0.0f32; 1];
    let mut quants_out = vec![0i8; 256];
    let result = quantize_activations_q8k_into(&values, &mut scale_out, &mut quants_out);
    assert!(result.is_ok());
}

// =============================================================================
// MULTIPLE BLOCK TESTS
// =============================================================================

#[test]
fn test_dequantize_q4_0_two_blocks() {
    let data = vec![0u8; 36]; // 2 blocks
    let result = dequantize_q4_0(&data);
    assert!(result.is_ok());
    assert_eq!(result.unwrap().len(), 64);
}

#[test]
fn test_dequantize_q8_0_two_blocks() {
    let data = vec![0u8; 68]; // 2 blocks
    let result = dequantize_q8_0(&data);
    assert!(result.is_ok());
    assert_eq!(result.unwrap().len(), 64);
}

#[test]
fn test_dequantize_q4_k_two_blocks() {
    let data = vec![0u8; 288]; // 2 blocks
    let result = dequantize_q4_k(&data);
    assert!(result.is_ok());
    assert_eq!(result.unwrap().len(), 512);
}
