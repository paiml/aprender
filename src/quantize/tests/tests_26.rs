//! Part 26: Error Path and Edge Case Coverage Tests
//!
//! This module targets uncovered error handling paths in quantize/mod.rs:
//! - InterleavedQ4K::from_q4k invalid data errors
//! - InterleavedQ4K::dot dimension mismatch errors
//! - fused_q4_0_q8_0_parallel_matvec validation errors
//! - fused_q8_0_q8_0_parallel_matvec validation errors
//! - Boundary conditions and edge cases
//!
//! Refs: T-COV-001

use crate::quantize::{
    dequantize_q8_blocks, fused_q4_0_q8_0_parallel_matvec, fused_q4_0_q8_0_parallel_matvec_into,
    fused_q8_0_q8_0_parallel_matvec, fused_q8_0_q8_0_parallel_matvec_into,
    quantize_activations_q8k_into, quantize_to_q8_blocks, InterleavedQ4K,
};

// =============================================================================
// Error Path Tests: InterleavedQ4K
// =============================================================================

/// Test InterleavedQ4K::from_q4k with data not aligned to super-block size
#[test]
fn test_interleaved_q4k_from_q4k_invalid_length() {
    // Q4_K super-block is 144 bytes; test with misaligned data
    let invalid_data = vec![0u8; 143]; // One byte short
    let result = InterleavedQ4K::from_q4k(&invalid_data);
    assert!(
        result.is_err(),
        "Should fail with 143 bytes (not multiple of 144)"
    );

    let err_msg = format!("{:?}", result.unwrap_err());
    assert!(
        err_msg.contains("multiple") || err_msg.contains("144"),
        "Error should mention super-block size: {}",
        err_msg
    );
}

/// Test InterleavedQ4K::from_q4k with partial super-block
#[test]
fn test_interleaved_q4k_from_q4k_partial_superblock() {
    // 145 bytes = 1 super-block + 1 extra byte
    let invalid_data = vec![0u8; 145];
    let result = InterleavedQ4K::from_q4k(&invalid_data);
    assert!(result.is_err());
}

/// Test InterleavedQ4K::from_q4k with empty data
#[test]
fn test_interleaved_q4k_from_q4k_empty() {
    let empty_data: Vec<u8> = vec![];
    // Empty is technically a multiple of 144 (0 super-blocks)
    let result = InterleavedQ4K::from_q4k(&empty_data);
    assert!(result.is_ok());
    let interleaved = result.expect("test value should be present");
    assert_eq!(interleaved.num_values(), 0);
}

/// Test InterleavedQ4K::dot with mismatched activation length
#[test]
fn test_interleaved_q4k_dot_dimension_mismatch() {
    // Create valid 1 super-block = 256 values
    let data = vec![0u8; 144];
    let interleaved = InterleavedQ4K::from_q4k(&data).expect("test value should be present");
    assert_eq!(interleaved.num_values(), 256);

    // Try with wrong activation length
    let activations = vec![1.0f32; 128]; // Half the expected length
    let result = interleaved.dot(&activations);
    assert!(result.is_err());

    let err_msg = format!("{:?}", result.unwrap_err());
    assert!(
        err_msg.contains("128") || err_msg.contains("256"),
        "Error should mention dimension mismatch: {}",
        err_msg
    );
}

/// Test InterleavedQ4K::dot with extra activations
#[test]
fn test_interleaved_q4k_dot_extra_activations() {
    let data = vec![0u8; 144];
    let interleaved = InterleavedQ4K::from_q4k(&data).expect("test value should be present");

    // Try with too many activations
    let activations = vec![1.0f32; 512]; // Double the expected length
    let result = interleaved.dot(&activations);
    assert!(result.is_err());
}

// =============================================================================
// Error Path Tests: fused_q4_0_q8_0_parallel_matvec
// =============================================================================

/// Test parallel matvec with insufficient weight data
#[test]
fn test_fused_q4_0_q8_0_parallel_matvec_weight_too_small() {
    let in_dim = 64;
    let out_dim = 4;
    let bytes_per_row = (in_dim / 32) * 18; // 2 blocks * 18 bytes = 36

    // Provide less data than needed
    let weight_data = vec![0u8; out_dim * bytes_per_row - 1];
    let activations = vec![1.0f32; in_dim];

    let result = fused_q4_0_q8_0_parallel_matvec(&weight_data, &activations, in_dim, out_dim);
    assert!(result.is_err());
}

/// Test parallel matvec with activation dimension mismatch
#[test]
fn test_fused_q4_0_q8_0_parallel_matvec_activation_mismatch() {
    let in_dim = 64;
    let out_dim = 4;
    let bytes_per_row = (in_dim / 32) * 18;

    let weight_data = vec![0u8; out_dim * bytes_per_row];
    let activations = vec![1.0f32; in_dim + 1]; // One extra element

    let result = fused_q4_0_q8_0_parallel_matvec(&weight_data, &activations, in_dim, out_dim);
    assert!(result.is_err());
}

/// Test parallel matvec exceeding parallel threshold
#[test]
fn test_fused_q4_0_q8_0_parallel_matvec_large_matrix() {
    // Use out_dim >= 1024 to trigger parallel path
    let in_dim = 64;
    let out_dim = 1024;
    let bytes_per_row = (in_dim / 32) * 18;

    let weight_data = vec![0u8; out_dim * bytes_per_row];
    let activations = vec![1.0f32; in_dim];

    let result = fused_q4_0_q8_0_parallel_matvec(&weight_data, &activations, in_dim, out_dim);
    assert!(result.is_ok());
    assert_eq!(result.expect("test value should be present").len(), out_dim);
}

// =============================================================================
// Error Path Tests: fused_q4_0_q8_0_parallel_matvec_into
// =============================================================================

/// Test into variant with activation mismatch
#[test]
fn test_fused_q4_0_q8_0_parallel_matvec_into_activation_mismatch() {
    let in_dim = 64;
    let out_dim = 8;
    let bytes_per_row = (in_dim / 32) * 18;

    let weight_data = vec![0u8; out_dim * bytes_per_row];
    let activations = vec![1.0f32; in_dim + 1]; // One extra element
    let mut output = vec![0.0f32; out_dim];

    let result =
        fused_q4_0_q8_0_parallel_matvec_into(&weight_data, &activations, in_dim, &mut output);
    assert!(result.is_err());
}

/// Test into variant with correct dimensions
#[test]
fn test_fused_q4_0_q8_0_parallel_matvec_into_success() {
    let in_dim = 64;
    let out_dim = 8;
    let bytes_per_row = (in_dim / 32) * 18;

    let weight_data = vec![0u8; out_dim * bytes_per_row];
    let activations = vec![1.0f32; in_dim];
    let mut output = vec![0.0f32; out_dim];

    let result =
        fused_q4_0_q8_0_parallel_matvec_into(&weight_data, &activations, in_dim, &mut output);
    assert!(result.is_ok());
}

// =============================================================================
// Error Path Tests: fused_q8_0_q8_0_parallel_matvec
// =============================================================================

/// Test Q8_0 parallel matvec with weight data too small
#[test]
fn test_fused_q8_0_q8_0_parallel_matvec_weight_too_small() {
    let in_dim = 64;
    let out_dim = 4;
    let bytes_per_row = (in_dim / 32) * 34; // Q8_0: 34 bytes per block

    let weight_data = vec![0u8; out_dim * bytes_per_row - 1];
    let activations = vec![1.0f32; in_dim];

    let result = fused_q8_0_q8_0_parallel_matvec(&weight_data, &activations, in_dim, out_dim);
    assert!(result.is_err());
}

/// Test Q8_0 parallel matvec with activation mismatch
#[test]
fn test_fused_q8_0_q8_0_parallel_matvec_activation_mismatch() {
    let in_dim = 64;
    let out_dim = 4;
    let bytes_per_row = (in_dim / 32) * 34;

    let weight_data = vec![0u8; out_dim * bytes_per_row];
    let activations = vec![1.0f32; in_dim - 1]; // One element short

    let result = fused_q8_0_q8_0_parallel_matvec(&weight_data, &activations, in_dim, out_dim);
    assert!(result.is_err());
}

/// Test Q8_0 parallel matvec large matrix (parallel path)
#[test]
fn test_fused_q8_0_q8_0_parallel_matvec_large() {
    let in_dim = 64;
    let out_dim = 1024;
    let bytes_per_row = (in_dim / 32) * 34;

    let weight_data = vec![0u8; out_dim * bytes_per_row];
    let activations = vec![1.0f32; in_dim];

    let result = fused_q8_0_q8_0_parallel_matvec(&weight_data, &activations, in_dim, out_dim);
    assert!(result.is_ok());
    assert_eq!(result.expect("test value should be present").len(), out_dim);
}

// =============================================================================
// Error Path Tests: fused_q8_0_q8_0_parallel_matvec_into
// =============================================================================

/// Test Q8_0 into variant with output buffer too small
#[test]
fn test_fused_q8_0_q8_0_parallel_matvec_into_output_small() {
    let in_dim = 64;
    let out_dim = 8;
    let bytes_per_row = (in_dim / 32) * 34;

    let weight_data = vec![0u8; out_dim * bytes_per_row];
    let activations = vec![1.0f32; in_dim];
    let mut output = vec![0.0f32; out_dim - 1];

    let result = fused_q8_0_q8_0_parallel_matvec_into(
        &weight_data,
        &activations,
        in_dim,
        out_dim,
        &mut output,
    );
    assert!(result.is_err());
}

/// Test Q8_0 into variant success
#[test]
fn test_fused_q8_0_q8_0_parallel_matvec_into_success() {
    let in_dim = 64;
    let out_dim = 8;
    let bytes_per_row = (in_dim / 32) * 34;

    let weight_data = vec![0u8; out_dim * bytes_per_row];
    let activations = vec![1.0f32; in_dim];
    let mut output = vec![0.0f32; out_dim];

    let result = fused_q8_0_q8_0_parallel_matvec_into(
        &weight_data,
        &activations,
        in_dim,
        out_dim,
        &mut output,
    );
    assert!(result.is_ok());
}

// =============================================================================
// Error Path Tests: quantize_activations_q8k_into
// =============================================================================

/// Test Q8K into with buffers too small
#[test]
fn test_quantize_activations_q8k_into_buffers_small() {
    let activations = vec![1.0f32; 512]; // 2 super-blocks
    let mut scales = vec![0.0f32; 1]; // Need 2
    let mut quants = vec![0i8; 512];

    let result = quantize_activations_q8k_into(&activations, &mut scales, &mut quants);
    assert!(result.is_err());
}

/// Test Q8K into with quants buffer too small
#[test]
fn test_quantize_activations_q8k_into_quants_small() {
    let activations = vec![1.0f32; 512];
    let mut scales = vec![0.0f32; 2];
    let mut quants = vec![0i8; 256]; // Need 512

    let result = quantize_activations_q8k_into(&activations, &mut scales, &mut quants);
    assert!(result.is_err());
}

/// Test Q8K into with non-aligned activation length
#[test]
fn test_quantize_activations_q8k_into_non_aligned() {
    let activations = vec![1.0f32; 300]; // Not multiple of 256
    let mut scales = vec![0.0f32; 2];
    let mut quants = vec![0i8; 512];

    let result = quantize_activations_q8k_into(&activations, &mut scales, &mut quants);
    assert!(result.is_err());
}

/// Test Q8K into success case
#[test]
fn test_quantize_activations_q8k_into_success() {
    let activations = vec![1.0f32; 512];
    let mut scales = vec![0.0f32; 2];
    let mut quants = vec![0i8; 512];

    let result = quantize_activations_q8k_into(&activations, &mut scales, &mut quants);
    assert!(result.is_ok());
}

// =============================================================================
// Edge Case Tests: quantize_to_q8_blocks
// =============================================================================

/// Test Q8 block quantization with empty input
#[test]
fn test_quantize_to_q8_blocks_empty() {
    let values: Vec<f32> = vec![];
    let result = quantize_to_q8_blocks(&values);
    assert!(result.is_ok());
    assert!(result.expect("test value should be present").is_empty());
}

/// Test Q8 block quantization with non-aligned input
#[test]
fn test_quantize_to_q8_blocks_non_aligned() {
    let values = vec![1.0f32; 50]; // Not multiple of 32
    let result = quantize_to_q8_blocks(&values);
    assert!(result.is_err());
}

/// Test Q8 block quantization with special values
#[test]
fn test_quantize_to_q8_blocks_special_values() {
    let mut values = vec![0.0f32; 32];
    values[0] = f32::INFINITY;
    values[1] = f32::NEG_INFINITY;
    values[2] = f32::NAN;

    // Should still produce blocks (implementation handles special floats)
    let result = quantize_to_q8_blocks(&values);
    // Result depends on implementation - just verify it doesn't panic
    let _ = result;
}

/// Test Q8 block round-trip (quantize then dequantize)
#[test]
fn test_q8_block_round_trip() {
    let original: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) / 10.0).collect();

    let blocks = quantize_to_q8_blocks(&original).expect("test value should be present");
    assert_eq!(blocks.len(), 2); // 64 values = 2 blocks of 32

    let reconstructed = dequantize_q8_blocks(&blocks);
    assert_eq!(reconstructed.len(), original.len());

    // Check reconstruction is reasonable (quantization introduces error)
    for (orig, recon) in original.iter().zip(reconstructed.iter()) {
        let diff = (orig - recon).abs();
        assert!(
            diff < 0.5,
            "Round-trip error too large: {} -> {}",
            orig,
            recon
        );
    }
}

// =============================================================================
// Edge Case Tests: InterleavedQ4K dot with actual values
// =============================================================================

/// Test InterleavedQ4K dot with all-zero weights
#[test]
fn test_interleaved_q4k_dot_zero_weights() {
    let data = vec![0u8; 144];
    let interleaved = InterleavedQ4K::from_q4k(&data).expect("test value should be present");
    let activations = vec![1.0f32; 256];

    let result = interleaved.dot(&activations).expect("test value should be present");
    // All-zero weights should give zero result
    assert!(result.abs() < 1e-6);
}

/// Test InterleavedQ4K dot with large values
#[test]
fn test_interleaved_q4k_dot_large_values() {
    // Create super-block with d=1.0 (f16 0x3C00)
    let mut data = vec![0u8; 144];
    data[0] = 0x00;
    data[1] = 0x3C; // d = 1.0

    let interleaved = InterleavedQ4K::from_q4k(&data).expect("test value should be present");
    let activations = vec![1000.0f32; 256];

    let result = interleaved.dot(&activations).expect("test value should be present");
    assert!(result.is_finite());
}

/// Test InterleavedQ4K dot with multiple super-blocks
#[test]
fn test_interleaved_q4k_dot_multiple_superblocks() {
    let num_superblocks = 8;
    let mut data = vec![0u8; num_superblocks * 144];

    // Set d=1.0 for each super-block
    for sb in 0..num_superblocks {
        let offset = sb * 144;
        data[offset] = 0x00;
        data[offset + 1] = 0x3C;
    }

    let interleaved = InterleavedQ4K::from_q4k(&data).expect("test value should be present");
    assert_eq!(interleaved.num_values(), num_superblocks * 256);

    let activations = vec![1.0f32; interleaved.num_values()];
    let result = interleaved.dot(&activations).expect("test value should be present");
    assert!(result.is_finite());
}

// =============================================================================
// Boundary Condition Tests
// =============================================================================

/// Test minimum valid Q4_0 matvec (1 block, 1 output)
#[test]
fn test_minimum_q4_0_matvec() {
    let in_dim = 32;
    let out_dim = 1;
    let weight_data = vec![0u8; 18];
    let activations = vec![1.0f32; in_dim];

    let result = fused_q4_0_q8_0_parallel_matvec(&weight_data, &activations, in_dim, out_dim);
    assert!(result.is_ok());
}

include!("minimum_exactly.rs");
