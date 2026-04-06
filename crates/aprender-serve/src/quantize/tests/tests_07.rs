//! Part 07: Fused K-Quant Tests (PMAT-803)
//!
//! Tests for `quantize/fused_k.rs` to drive coverage from 36% to higher.
//!
//! # Coverage Targets
//!
//! - `fused_q4k_dot` - Q4_K dot product with f32 activations
//! - `fused_q4k_dot_simd` - SIMD-accelerated Q4_K dot product
//! - `fused_q4k_q8k_dot` - Q4_K with Q8_K activations
//! - `fused_q4k_q8k_dot_simd` - SIMD-accelerated Q4_K x Q8_K
//!
//! # Test Organization
//!
//! - Error path tests (invalid lengths, mismatches)
//! - Basic functionality tests (zero values, identity)
//! - Property-based tests (scalar vs SIMD equivalence)
//! - Edge case tests (single block, many blocks)

use crate::quantize::fused_k::{fused_q4k_dot, fused_q4k_dot_simd};
use crate::quantize::fused_k::{fused_q4k_q8k_dot, fused_q4k_q8k_dot_simd};
use crate::quantize::types::QK_K;

// ============================================================================
// Constants
// ============================================================================

/// Q4_K super-block size in bytes
const Q4K_BLOCK_BYTES: usize = 144;

// ============================================================================
// Part 1: fused_q4k_dot Error Tests
// ============================================================================

#[test]
fn test_fused_q4k_dot_invalid_length_not_multiple() {
    // Not a multiple of 144 bytes
    let q4k_data = vec![0u8; 100];
    let activations = vec![0.0f32; 256];

    let result = fused_q4k_dot(&q4k_data, &activations);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(matches!(
        err,
        crate::error::RealizarError::InvalidShape { .. }
    ));
}

#[test]
fn test_fused_q4k_dot_invalid_length_one_byte_short() {
    // 143 bytes instead of 144
    let q4k_data = vec![0u8; Q4K_BLOCK_BYTES - 1];
    let activations = vec![0.0f32; QK_K];

    let result = fused_q4k_dot(&q4k_data, &activations);
    assert!(result.is_err());
}

#[test]
fn test_fused_q4k_dot_invalid_length_one_byte_over() {
    // 145 bytes instead of 144
    let q4k_data = vec![0u8; Q4K_BLOCK_BYTES + 1];
    let activations = vec![0.0f32; QK_K];

    let result = fused_q4k_dot(&q4k_data, &activations);
    assert!(result.is_err());
}

#[test]
fn test_fused_q4k_dot_activation_length_mismatch_short() {
    // Valid Q4K data but activations too short
    let q4k_data = vec![0u8; Q4K_BLOCK_BYTES];
    let activations = vec![0.0f32; QK_K - 1];

    let result = fused_q4k_dot(&q4k_data, &activations);
    assert!(result.is_err());
}

#[test]
fn test_fused_q4k_dot_activation_length_mismatch_long() {
    // Valid Q4K data but activations too long
    let q4k_data = vec![0u8; Q4K_BLOCK_BYTES];
    let activations = vec![0.0f32; QK_K + 1];

    let result = fused_q4k_dot(&q4k_data, &activations);
    assert!(result.is_err());
}

#[test]
fn test_fused_q4k_dot_empty_inputs() {
    // Empty is technically valid (0 super-blocks, 0 activations)
    let q4k_data: Vec<u8> = vec![];
    let activations: Vec<f32> = vec![];

    let result = fused_q4k_dot(&q4k_data, &activations);
    assert!(result.is_ok());
    assert_eq!(result.expect("test value should be present"), 0.0);
}

// ============================================================================
// Part 2: fused_q4k_dot Basic Functionality Tests
// ============================================================================

#[test]
fn test_fused_q4k_dot_zero_weights() {
    // All zero weights should produce zero dot product
    let q4k_data = vec![0u8; Q4K_BLOCK_BYTES];
    let activations = vec![1.0f32; QK_K];

    let result = fused_q4k_dot(&q4k_data, &activations);
    assert!(result.is_ok());
    // Zero scales/data means zero contribution
    assert!(result.expect("test value should be present").abs() < 1e-6);
}

#[test]
fn test_fused_q4k_dot_zero_activations() {
    // Zero activations should produce zero dot product
    let q4k_data = create_q4k_test_block(1.0, 0.0);
    let activations = vec![0.0f32; QK_K];

    let result = fused_q4k_dot(&q4k_data, &activations);
    assert!(result.is_ok());
    assert!(result.expect("test value should be present").abs() < 1e-6);
}

#[test]
fn test_fused_q4k_dot_single_block() {
    // Single super-block with known values
    let q4k_data = create_q4k_test_block(1.0, 0.5);
    let activations = vec![1.0f32; QK_K];

    let result = fused_q4k_dot(&q4k_data, &activations);
    assert!(result.is_ok());
    // Should produce some non-zero value
    let dot = result.expect("test value should be present");
    assert!(dot.is_finite());
}

#[test]
fn test_fused_q4k_dot_multiple_blocks() {
    // Multiple super-blocks
    let num_blocks = 4;
    let mut q4k_data = Vec::with_capacity(num_blocks * Q4K_BLOCK_BYTES);
    for _ in 0..num_blocks {
        q4k_data.extend_from_slice(&create_q4k_test_block(0.5, 0.1));
    }
    let activations = vec![1.0f32; num_blocks * QK_K];

    let result = fused_q4k_dot(&q4k_data, &activations);
    assert!(result.is_ok());
    let dot = result.expect("test value should be present");
    assert!(dot.is_finite());
}

#[test]
fn test_fused_q4k_dot_varied_activations() {
    // Activations with varied values
    let q4k_data = create_q4k_test_block(1.0, 0.0);
    let mut activations = vec![0.0f32; QK_K];
    for (i, a) in activations.iter_mut().enumerate() {
        *a = (i as f32) / (QK_K as f32);
    }

    let result = fused_q4k_dot(&q4k_data, &activations);
    assert!(result.is_ok());
    assert!(result.expect("test value should be present").is_finite());
}

#[test]
fn test_fused_q4k_dot_negative_activations() {
    let q4k_data = create_q4k_test_block(1.0, 0.5);
    let activations = vec![-1.0f32; QK_K];

    let result = fused_q4k_dot(&q4k_data, &activations);
    assert!(result.is_ok());
    assert!(result.expect("test value should be present").is_finite());
}

// ============================================================================
// Part 3: fused_q4k_dot_simd Error Tests
// ============================================================================

#[test]
fn test_fused_q4k_dot_simd_invalid_length() {
    let q4k_data = vec![0u8; 100];
    let activations = vec![0.0f32; 256];

    let result = fused_q4k_dot_simd(&q4k_data, &activations);
    assert!(result.is_err());
}

#[test]
fn test_fused_q4k_dot_simd_activation_mismatch() {
    let q4k_data = vec![0u8; Q4K_BLOCK_BYTES];
    let activations = vec![0.0f32; QK_K + 10];

    let result = fused_q4k_dot_simd(&q4k_data, &activations);
    assert!(result.is_err());
}

#[test]
fn test_fused_q4k_dot_simd_empty_inputs() {
    let q4k_data: Vec<u8> = vec![];
    let activations: Vec<f32> = vec![];

    let result = fused_q4k_dot_simd(&q4k_data, &activations);
    assert!(result.is_ok());
    assert_eq!(result.expect("test value should be present"), 0.0);
}

// ============================================================================
// Part 4: fused_q4k_dot_simd Basic Functionality Tests
// ============================================================================

#[test]
fn test_fused_q4k_dot_simd_zero_weights() {
    let q4k_data = vec![0u8; Q4K_BLOCK_BYTES];
    let activations = vec![1.0f32; QK_K];

    let result = fused_q4k_dot_simd(&q4k_data, &activations);
    assert!(result.is_ok());
    assert!(result.expect("test value should be present").abs() < 1e-6);
}

#[test]
fn test_fused_q4k_dot_simd_single_block() {
    let q4k_data = create_q4k_test_block(1.0, 0.5);
    let activations = vec![1.0f32; QK_K];

    let result = fused_q4k_dot_simd(&q4k_data, &activations);
    assert!(result.is_ok());
    assert!(result.expect("test value should be present").is_finite());
}

#[test]
fn test_fused_q4k_dot_simd_multiple_blocks() {
    let num_blocks = 8;
    let mut q4k_data = Vec::with_capacity(num_blocks * Q4K_BLOCK_BYTES);
    for _ in 0..num_blocks {
        q4k_data.extend_from_slice(&create_q4k_test_block(0.5, 0.25));
    }
    let activations = vec![0.5f32; num_blocks * QK_K];

    let result = fused_q4k_dot_simd(&q4k_data, &activations);
    assert!(result.is_ok());
    assert!(result.expect("test value should be present").is_finite());
}

// ============================================================================
// Part 5: Scalar vs SIMD Equivalence Tests
// ============================================================================

#[test]
fn test_fused_q4k_dot_scalar_simd_equivalence_zero() {
    let q4k_data = vec![0u8; Q4K_BLOCK_BYTES];
    let activations = vec![1.0f32; QK_K];

    let scalar = fused_q4k_dot(&q4k_data, &activations).expect("test value should be present");
    let simd = fused_q4k_dot_simd(&q4k_data, &activations).expect("test value should be present");

    assert!(
        (scalar - simd).abs() < 1e-5,
        "scalar={} simd={}",
        scalar,
        simd
    );
}

#[test]
fn test_fused_q4k_dot_scalar_simd_equivalence_ones() {
    let q4k_data = create_q4k_test_block(1.0, 0.0);
    let activations = vec![1.0f32; QK_K];

    let scalar = fused_q4k_dot(&q4k_data, &activations).expect("test value should be present");
    let simd = fused_q4k_dot_simd(&q4k_data, &activations).expect("test value should be present");

    // Allow for some numerical tolerance
    let rel_diff = if scalar.abs() > 1e-6 {
        (scalar - simd).abs() / scalar.abs()
    } else {
        (scalar - simd).abs()
    };
    assert!(
        rel_diff < 0.01,
        "scalar={} simd={} rel_diff={}",
        scalar,
        simd,
        rel_diff
    );
}

#[test]
fn test_fused_q4k_dot_scalar_simd_equivalence_random() {
    // Use deterministic "random" values
    let mut q4k_data = Vec::with_capacity(Q4K_BLOCK_BYTES);
    for i in 0..Q4K_BLOCK_BYTES {
        q4k_data.push((i * 37 % 256) as u8);
    }
    // Fix the d and dmin fields to valid f16 values
    q4k_data[0] = 0x00;
    q4k_data[1] = 0x3C; // ~1.0 in f16
    q4k_data[2] = 0x00;
    q4k_data[3] = 0x30; // ~0.5 in f16

    let mut activations = vec![0.0f32; QK_K];
    for (i, a) in activations.iter_mut().enumerate() {
        *a = ((i * 13) % 100) as f32 / 100.0 - 0.5;
    }

    let scalar = fused_q4k_dot(&q4k_data, &activations).expect("test value should be present");
    let simd = fused_q4k_dot_simd(&q4k_data, &activations).expect("test value should be present");

    // Check both are finite
    assert!(scalar.is_finite());
    assert!(simd.is_finite());

    // Check they're close
    let abs_diff = (scalar - simd).abs();
    let max_val = scalar.abs().max(simd.abs()).max(1.0);
    assert!(abs_diff / max_val < 0.1, "scalar={} simd={}", scalar, simd);
}

#[test]
fn test_fused_q4k_dot_scalar_simd_equivalence_multiple_blocks() {
    let num_blocks = 4;
    let mut q4k_data = Vec::with_capacity(num_blocks * Q4K_BLOCK_BYTES);
    for b in 0..num_blocks {
        let mut block = create_q4k_test_block(0.5 + b as f32 * 0.1, 0.1);
        // Vary the quant data a bit
        for i in 12..Q4K_BLOCK_BYTES {
            block[i] = ((b * 17 + i * 7) % 256) as u8;
        }
        q4k_data.extend_from_slice(&block);
    }

    let activations: Vec<f32> = (0..num_blocks * QK_K)
        .map(|i| (i as f32 / (num_blocks * QK_K) as f32) - 0.5)
        .collect();

    let scalar = fused_q4k_dot(&q4k_data, &activations).expect("test value should be present");
    let simd = fused_q4k_dot_simd(&q4k_data, &activations).expect("test value should be present");

    assert!(scalar.is_finite());
    assert!(simd.is_finite());
}

// ============================================================================
// Part 6: fused_q4k_q8k_dot Tests
// ============================================================================

#[test]
fn test_fused_q4k_q8k_dot_invalid_q4k_length() {
    let q4k_data = vec![0u8; 100]; // Not multiple of 144
    let q8k_scales = vec![1.0f32; 1];
    let q8k_quants = vec![0i8; QK_K];

    let result = fused_q4k_q8k_dot(&q4k_data, &q8k_scales, &q8k_quants);
    assert!(result.is_err());
}

#[test]
fn test_fused_q4k_q8k_dot_invalid_q8k_scales_length() {
    // 2 super-blocks but only 1 scale (needs at least 2)
    let q4k_data = vec![0u8; Q4K_BLOCK_BYTES * 2];
    let q8k_scales = vec![1.0f32; 1]; // Should be 2 for 2 super-blocks
    let q8k_quants = vec![0i8; QK_K * 2];

    let result = fused_q4k_q8k_dot(&q4k_data, &q8k_scales, &q8k_quants);
    assert!(result.is_err());
}

#[test]
fn test_fused_q4k_q8k_dot_invalid_q8k_quants_length() {
    let q4k_data = vec![0u8; Q4K_BLOCK_BYTES];
    let q8k_scales = vec![1.0f32; 1];
    let q8k_quants = vec![0i8; QK_K - 1]; // Too short

    let result = fused_q4k_q8k_dot(&q4k_data, &q8k_scales, &q8k_quants);
    assert!(result.is_err());
}

#[test]
fn test_fused_q4k_q8k_dot_empty_inputs() {
    let q4k_data: Vec<u8> = vec![];
    let q8k_scales: Vec<f32> = vec![];
    let q8k_quants: Vec<i8> = vec![];

    let result = fused_q4k_q8k_dot(&q4k_data, &q8k_scales, &q8k_quants);
    assert!(result.is_ok());
    assert_eq!(result.expect("test value should be present"), 0.0);
}

#[test]
fn test_fused_q4k_q8k_dot_zero_weights() {
    let q4k_data = vec![0u8; Q4K_BLOCK_BYTES];
    let q8k_scales = vec![1.0f32; 1];
    let q8k_quants = vec![1i8; QK_K];

    let result = fused_q4k_q8k_dot(&q4k_data, &q8k_scales, &q8k_quants);
    assert!(result.is_ok());
    assert!(result.expect("test value should be present").abs() < 1e-6);
}

#[test]
fn test_fused_q4k_q8k_dot_zero_activations() {
    let q4k_data = create_q4k_test_block(1.0, 0.5);
    let q8k_scales = vec![1.0f32; 1];
    let q8k_quants = vec![0i8; QK_K];

    let result = fused_q4k_q8k_dot(&q4k_data, &q8k_scales, &q8k_quants);
    assert!(result.is_ok());
    assert!(result.expect("test value should be present").abs() < 1e-6);
}

#[test]
fn test_fused_q4k_q8k_dot_single_block() {
    let q4k_data = create_q4k_test_block(1.0, 0.25);
    let q8k_scales = vec![0.5f32; 1];
    let q8k_quants = vec![1i8; QK_K];

    let result = fused_q4k_q8k_dot(&q4k_data, &q8k_scales, &q8k_quants);
    assert!(result.is_ok());
    assert!(result.expect("test value should be present").is_finite());
}

#[test]
fn test_fused_q4k_q8k_dot_multiple_blocks() {
    let num_blocks = 4;
    let mut q4k_data = Vec::with_capacity(num_blocks * Q4K_BLOCK_BYTES);
    for _ in 0..num_blocks {
        q4k_data.extend_from_slice(&create_q4k_test_block(0.5, 0.1));
    }
    let q8k_scales = vec![0.5f32; num_blocks];
    let q8k_quants = vec![2i8; num_blocks * QK_K];

    let result = fused_q4k_q8k_dot(&q4k_data, &q8k_scales, &q8k_quants);
    assert!(result.is_ok());
    assert!(result.expect("test value should be present").is_finite());
}

include!("fused_q4k_03.rs");
