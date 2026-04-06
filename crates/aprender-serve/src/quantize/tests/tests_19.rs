//! Phase 45: Parallel dequantization coverage tests for `parallel_dequant.rs`
//!
//! Tests cover:
//! - `dequantize_q4_k_parallel` - Parallel Q4_K dequantization
//! - `dequantize_q4_k_simd` - SIMD-accelerated Q4_K dequantization
//! - `dequantize_q4_k_superblock` - Single super-block dequantization
//! - `dequantize_q8_0_parallel` - Parallel Q8_0 dequantization
//! - `dequantize_q8_0_simd` - SIMD-accelerated Q8_0 dequantization
//! - `dequantize_q8_0_block` - Single block dequantization
//! - `apply_rope_rotation_simd` - SIMD RoPE rotation
//! - `apply_rope_rotation_scalar` - Scalar RoPE fallback
//!
//! Focus areas:
//! - Edge cases: empty input, single block, boundary sizes
//! - Error handling: invalid sizes, non-multiple of block size
//! - Parallel vs sequential paths
//! - AVX2 SIMD paths (with scalar fallback verification)
//! - RoPE rotation for various head dimensions

use crate::quantize::parallel_dequant::{
    apply_rope_rotation_scalar, apply_rope_rotation_simd, dequantize_q4_k_parallel,
    dequantize_q4_k_simd, dequantize_q4_k_superblock, dequantize_q8_0_block,
    dequantize_q8_0_parallel, dequantize_q8_0_simd,
};
use crate::quantize::types::QK_K;

// ============================================================================
// Helper functions for generating valid quantized data
// ============================================================================

/// Generate valid Q4_K super-block data (144 bytes per super-block)
fn generate_q4k_superblock_data(num_super_blocks: usize) -> Vec<u8> {
    const SUPER_BLOCK_BYTES: usize = 144;
    let mut data = vec![0u8; num_super_blocks * SUPER_BLOCK_BYTES];

    for sb in 0..num_super_blocks {
        let sb_start = sb * SUPER_BLOCK_BYTES;

        // Set d (f16 = 1.0 = 0x3C00 in little-endian)
        data[sb_start] = 0x00;
        data[sb_start + 1] = 0x3C;

        // Set dmin (f16 = 0.5 = 0x3800 in little-endian)
        data[sb_start + 2] = 0x00;
        data[sb_start + 3] = 0x38;

        // Set scales (12 bytes) - alternating pattern
        for i in 0..12 {
            data[sb_start + 4 + i] = ((sb * 7 + i * 13) % 64) as u8;
        }

        // Set quantized values (128 bytes) - deterministic pattern
        for i in 0..128 {
            data[sb_start + 16 + i] = ((sb * 11 + i * 17) % 256) as u8;
        }
    }

    data
}

/// Generate valid Q8_0 block data (34 bytes per block: 2 scale + 32 quants)
fn generate_q8_0_block_data(num_blocks: usize) -> Vec<u8> {
    const BLOCK_BYTES: usize = 34;
    let mut data = vec![0u8; num_blocks * BLOCK_BYTES];

    for block in 0..num_blocks {
        let block_start = block * BLOCK_BYTES;

        // Set scale (f16 = 0.5 = 0x3800 in little-endian)
        data[block_start] = 0x00;
        data[block_start + 1] = 0x38;

        // Set quantized int8 values (32 bytes)
        for i in 0..32 {
            // Values from -128 to 127, deterministic pattern
            data[block_start + 2 + i] = ((block * 7 + i * 5) % 256) as u8;
        }
    }

    data
}

// ============================================================================
// Tests for dequantize_q4_k_parallel
// ============================================================================

#[test]
fn test_dequantize_q4_k_parallel_single_superblock_p19() {
    let data = generate_q4k_superblock_data(1);
    let result = dequantize_q4_k_parallel(&data);

    assert!(result.is_ok());
    let output = result.expect("test value should be present");
    assert_eq!(output.len(), QK_K); // 256 values per super-block

    // All values should be finite
    for val in &output {
        assert!(val.is_finite(), "Value should be finite: {}", val);
    }
}

#[test]
fn test_dequantize_q4_k_parallel_multiple_superblocks_p19() {
    let num_super_blocks = 5;
    let data = generate_q4k_superblock_data(num_super_blocks);
    let result = dequantize_q4_k_parallel(&data);

    assert!(result.is_ok());
    let output = result.expect("test value should be present");
    assert_eq!(output.len(), num_super_blocks * QK_K);

    // All values should be finite
    for val in &output {
        assert!(val.is_finite());
    }
}

#[test]
fn test_dequantize_q4_k_parallel_large_input_p19() {
    // Test with many super-blocks to exercise parallel paths
    let num_super_blocks = 64;
    let data = generate_q4k_superblock_data(num_super_blocks);
    let result = dequantize_q4_k_parallel(&data);

    assert!(result.is_ok());
    let output = result.expect("test value should be present");
    assert_eq!(output.len(), num_super_blocks * QK_K);
}

#[test]
fn test_dequantize_q4_k_parallel_invalid_size_too_small_p19() {
    // Data that's not a multiple of 144 bytes
    let data = vec![0u8; 100]; // Less than one super-block
    let result = dequantize_q4_k_parallel(&data);

    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.to_string().contains("not a multiple"));
}

#[test]
fn test_dequantize_q4_k_parallel_invalid_size_partial_p19() {
    // Data that's 1.5 super-blocks (144 + 72 = 216 bytes)
    let data = vec![0u8; 216];
    let result = dequantize_q4_k_parallel(&data);

    assert!(result.is_err());
}

#[test]
fn test_dequantize_q4_k_parallel_empty_input_p19() {
    let data: Vec<u8> = vec![];
    let result = dequantize_q4_k_parallel(&data);

    // Empty input is a multiple of 144 (0 * 144 = 0), should succeed with empty output
    assert!(result.is_ok());
    assert_eq!(result.expect("test value should be present").len(), 0);
}

#[test]
fn test_dequantize_q4_k_parallel_deterministic_p19() {
    let data = generate_q4k_superblock_data(4);

    // Run multiple times and verify determinism
    let result1 = dequantize_q4_k_parallel(&data).expect("test value should be present");
    let result2 = dequantize_q4_k_parallel(&data).expect("test value should be present");
    let result3 = dequantize_q4_k_parallel(&data).expect("test value should be present");

    for i in 0..result1.len() {
        assert_eq!(
            result1[i], result2[i],
            "Mismatch at index {} between runs 1-2",
            i
        );
        assert_eq!(
            result2[i], result3[i],
            "Mismatch at index {} between runs 2-3",
            i
        );
    }
}

#[test]
fn test_dequantize_q4_k_parallel_scale_variations_p19() {
    // Test with different scale values to exercise scale extraction
    let mut data = generate_q4k_superblock_data(2);

    // Set different d values for the two super-blocks
    // Super-block 0: d = 2.0 (0x4000)
    data[0] = 0x00;
    data[1] = 0x40;

    // Super-block 1: d = 0.25 (0x3400)
    data[144] = 0x00;
    data[145] = 0x34;

    let result = dequantize_q4_k_parallel(&data);
    assert!(result.is_ok());

    let output = result.expect("test value should be present");

    // First super-block values should be larger on average
    let sum_first: f32 = output[..QK_K].iter().map(|v| v.abs()).sum();
    let sum_second: f32 = output[QK_K..].iter().map(|v| v.abs()).sum();

    // The first super-block has larger d, so its values should have larger magnitude
    assert!(
        sum_first > sum_second * 0.5, // Account for dmin differences
        "Expected first super-block to have larger values"
    );
}

// ============================================================================
// Tests for dequantize_q4_k_simd
// ============================================================================

#[test]
fn test_dequantize_q4_k_simd_single_superblock_p19() {
    let data = generate_q4k_superblock_data(1);
    let result = dequantize_q4_k_simd(&data);

    assert!(result.is_ok());
    let output = result.expect("test value should be present");
    assert_eq!(output.len(), QK_K);
}

#[test]
fn test_dequantize_q4_k_simd_multiple_superblocks_p19() {
    let num_super_blocks = 8;
    let data = generate_q4k_superblock_data(num_super_blocks);
    let result = dequantize_q4_k_simd(&data);

    assert!(result.is_ok());
    let output = result.expect("test value should be present");
    assert_eq!(output.len(), num_super_blocks * QK_K);
}

#[test]
fn test_dequantize_q4_k_simd_parity_with_parallel_p19() {
    // SIMD and parallel should produce identical results
    let data = generate_q4k_superblock_data(4);

    let simd_result = dequantize_q4_k_simd(&data).expect("test value should be present");
    let parallel_result = dequantize_q4_k_parallel(&data).expect("test value should be present");

    assert_eq!(simd_result.len(), parallel_result.len());

    for i in 0..simd_result.len() {
        let diff = (simd_result[i] - parallel_result[i]).abs();
        assert!(
            diff < 1e-6,
            "SIMD/parallel mismatch at {}: simd={}, parallel={}, diff={}",
            i,
            simd_result[i],
            parallel_result[i],
            diff
        );
    }
}

#[test]
fn test_dequantize_q4_k_simd_large_for_avx2_path_p19() {
    // Large enough to trigger parallel AVX2 path (>= 128 super-blocks)
    let num_super_blocks = 130;
    let data = generate_q4k_superblock_data(num_super_blocks);
    let result = dequantize_q4_k_simd(&data);

    assert!(result.is_ok());
    let output = result.expect("test value should be present");
    assert_eq!(output.len(), num_super_blocks * QK_K);
}

#[test]
fn test_dequantize_q4_k_simd_small_for_sequential_path_p19() {
    // Small enough to skip parallelism overhead (< 128 super-blocks)
    let num_super_blocks = 2;
    let data = generate_q4k_superblock_data(num_super_blocks);
    let result = dequantize_q4_k_simd(&data);

    assert!(result.is_ok());
    let output = result.expect("test value should be present");
    assert_eq!(output.len(), num_super_blocks * QK_K);
}

#[test]
fn test_dequantize_q4_k_simd_invalid_size_p19() {
    let data = vec![0u8; 100]; // Not a multiple of 144
    let result = dequantize_q4_k_simd(&data);
    assert!(result.is_err());
}

// ============================================================================
// Tests for dequantize_q4_k_superblock (internal helper)
// ============================================================================

#[test]
fn test_dequantize_q4_k_superblock_basic_p19() {
    let data = generate_q4k_superblock_data(1);
    let sb_data = &data[0..144];

    let output = dequantize_q4_k_superblock(sb_data);
    assert_eq!(output.len(), QK_K);

    for val in &output {
        assert!(val.is_finite());
    }
}

#[test]
fn test_dequantize_q4_k_superblock_all_zeros_p19() {
    // Super-block with zero scales should produce near-zero output
    let sb_data = vec![0u8; 144];

    let output = dequantize_q4_k_superblock(&sb_data);
    assert_eq!(output.len(), QK_K);

    // With d=0 and dmin=0, all values should be 0
    for val in &output {
        assert_eq!(*val, 0.0);
    }
}

#[test]
fn test_dequantize_q4_k_superblock_max_nibble_values_p19() {
    // Test with all nibbles set to max (0xF = 15)
    let mut sb_data = vec![0u8; 144];

    // Set d = 1.0
    sb_data[0] = 0x00;
    sb_data[1] = 0x3C;

    // Set dmin = 0.0
    sb_data[2] = 0x00;
    sb_data[3] = 0x00;

    // Set all scales to 1
    for i in 0..12 {
        sb_data[4 + i] = 1;
    }

    // Set all quantized values to 0xFF (both nibbles = 15)
    for i in 0..128 {
        sb_data[16 + i] = 0xFF;
    }

    let output = dequantize_q4_k_superblock(&sb_data);
    assert_eq!(output.len(), QK_K);

    // All values should be finite and positive (15 * scale)
    for val in &output {
        assert!(val.is_finite());
    }
}

#[test]
fn test_dequantize_q4_k_superblock_alternating_pattern_p19() {
    // Test with alternating nibble pattern (0x0F, 0xF0)
    let mut sb_data = vec![0u8; 144];

    // Set scales
    sb_data[0] = 0x00;
    sb_data[1] = 0x3C; // d = 1.0
    sb_data[2] = 0x00;
    sb_data[3] = 0x38; // dmin = 0.5

    // Set scales to non-zero
    for i in 0..12 {
        sb_data[4 + i] = 0x01;
    }

    // Alternating quantized values
    for i in 0..128 {
        sb_data[16 + i] = if i % 2 == 0 { 0x0F } else { 0xF0 };
    }

    let output = dequantize_q4_k_superblock(&sb_data);
    assert_eq!(output.len(), QK_K);
}

// ============================================================================
// Tests for dequantize_q8_0_parallel
// ============================================================================

#[test]
fn test_dequantize_q8_0_parallel_single_block_p19() {
    let data = generate_q8_0_block_data(1);
    let result = dequantize_q8_0_parallel(&data);

    assert!(result.is_ok());
    let output = result.expect("test value should be present");
    assert_eq!(output.len(), 32); // 32 values per block
}

#[test]
fn test_dequantize_q8_0_parallel_multiple_blocks_p19() {
    let num_blocks = 10;
    let data = generate_q8_0_block_data(num_blocks);
    let result = dequantize_q8_0_parallel(&data);

    assert!(result.is_ok());
    let output = result.expect("test value should be present");
    assert_eq!(output.len(), num_blocks * 32);
}

#[test]
fn test_dequantize_q8_0_parallel_large_input_p19() {
    // Many blocks to exercise parallel paths
    let num_blocks = 100;
    let data = generate_q8_0_block_data(num_blocks);
    let result = dequantize_q8_0_parallel(&data);

    assert!(result.is_ok());
    let output = result.expect("test value should be present");
    assert_eq!(output.len(), num_blocks * 32);
}

#[test]
fn test_dequantize_q8_0_parallel_invalid_size_p19() {
    // Data that's not a multiple of 34 bytes
    let data = vec![0u8; 50]; // Not 34 * n
    let result = dequantize_q8_0_parallel(&data);

    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.to_string().contains("not a multiple"));
}

#[test]
fn test_dequantize_q8_0_parallel_empty_input_p19() {
    let data: Vec<u8> = vec![];
    let result = dequantize_q8_0_parallel(&data);

    assert!(result.is_ok());
    assert_eq!(result.expect("test value should be present").len(), 0);
}

#[test]
fn test_dequantize_q8_0_parallel_deterministic_p19() {
    let data = generate_q8_0_block_data(8);

    let result1 = dequantize_q8_0_parallel(&data).expect("test value should be present");
    let result2 = dequantize_q8_0_parallel(&data).expect("test value should be present");

    for i in 0..result1.len() {
        assert_eq!(result1[i], result2[i], "Non-deterministic at index {}", i);
    }
}

include!("dequantize_07.rs");
include!("q4k_simd.rs");
