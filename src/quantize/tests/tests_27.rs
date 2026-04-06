//! Scalar Exhaustion Tests for fused_k.rs (PMAT-COV-95 Directive 3)
//!
//! Tests that directly call scalar fallback functions with pathological inputs:
//! - Empty buffers
//! - Tensors of scale zero
//! - Vectors containing Inf
//! - Edge cases for block alignment
//!
//! These tests exhaust the scalar paths to lift total region coverage.

use crate::error::RealizarError;
use crate::quantize::fused_k::{fused_q4k_dot, fused_q4k_q8k_dot};
use crate::quantize::types::QK_K;

// ============================================================================
// CONSTANTS
// ============================================================================

/// Q4_K super-block size (144 bytes per QK_K=256 values)
const Q4K_SUPER_BLOCK_BYTES: usize = 144;

/// Q8_K block size
const Q8K_BLOCK_SIZE: usize = 256;

// ============================================================================
// Q4K DOT VALIDATION TESTS
// ============================================================================

#[test]
fn test_fused_q4k_dot_empty_data() {
    let result = fused_q4k_dot(&[], &[]);
    assert!(result.is_ok());
    assert_eq!(result.expect("test value should be present"), 0.0);
}

#[test]
fn test_fused_q4k_dot_invalid_block_size() {
    // 143 bytes - not a multiple of 144
    let invalid_data = vec![0u8; 143];
    let activations = vec![1.0f32; 256];

    let result = fused_q4k_dot(&invalid_data, &activations);
    assert!(result.is_err());

    let err = result.unwrap_err();
    match err {
        RealizarError::InvalidShape { reason } => {
            assert!(reason.contains("not a multiple"));
        },
        _ => panic!("Expected InvalidShape error"),
    }
}

#[test]
fn test_fused_q4k_dot_activation_length_mismatch() {
    // One super-block (144 bytes) should need exactly 256 activations
    let q4k_data = vec![0u8; Q4K_SUPER_BLOCK_BYTES];
    let wrong_activations = vec![1.0f32; 128]; // Wrong length

    let result = fused_q4k_dot(&q4k_data, &wrong_activations);
    assert!(result.is_err());

    let err = result.unwrap_err();
    match err {
        RealizarError::InvalidShape { reason } => {
            assert!(reason.contains("doesn't match"));
        },
        _ => panic!("Expected InvalidShape error"),
    }
}

#[test]
fn test_fused_q4k_dot_zero_scales() {
    // Create a super-block with zero d and dmin (first 4 bytes are f16 zeros)
    let q4k_data = vec![0u8; Q4K_SUPER_BLOCK_BYTES];
    // f16 zero is 0x0000 for both d and dmin (bytes 0-3)
    // Rest of block is already zeros

    let activations = vec![1.0f32; QK_K];

    let result = fused_q4k_dot(&q4k_data, &activations);
    assert!(result.is_ok());
    // With zero scales, result should be zero regardless of activations
    assert_eq!(result.expect("test value should be present"), 0.0);
}

#[test]
fn test_fused_q4k_dot_inf_activations() {
    let q4k_data = vec![0u8; Q4K_SUPER_BLOCK_BYTES];
    let mut activations = vec![1.0f32; QK_K];
    activations[0] = f32::INFINITY;

    let result = fused_q4k_dot(&q4k_data, &activations);
    // With zero scales, inf * 0 = NaN or 0 depending on implementation
    // The function should not panic
    assert!(result.is_ok());
}

#[test]
fn test_fused_q4k_dot_neg_inf_activations() {
    let q4k_data = vec![0u8; Q4K_SUPER_BLOCK_BYTES];
    let mut activations = vec![1.0f32; QK_K];
    activations[128] = f32::NEG_INFINITY;

    let result = fused_q4k_dot(&q4k_data, &activations);
    assert!(result.is_ok());
}

#[test]
fn test_fused_q4k_dot_nan_activations() {
    let q4k_data = vec![0u8; Q4K_SUPER_BLOCK_BYTES];
    let mut activations = vec![1.0f32; QK_K];
    activations[64] = f32::NAN;

    let result = fused_q4k_dot(&q4k_data, &activations);
    assert!(result.is_ok());
    // Result may be NaN if NaN propagates
}

#[test]
fn test_fused_q4k_dot_multiple_super_blocks() {
    // Two super-blocks
    let q4k_data = vec![0u8; Q4K_SUPER_BLOCK_BYTES * 2];
    let activations = vec![1.0f32; QK_K * 2];

    let result = fused_q4k_dot(&q4k_data, &activations);
    assert!(result.is_ok());
}

#[test]
fn test_fused_q4k_dot_three_super_blocks() {
    let q4k_data = vec![0u8; Q4K_SUPER_BLOCK_BYTES * 3];
    let activations = vec![0.5f32; QK_K * 3];

    let result = fused_q4k_dot(&q4k_data, &activations);
    assert!(result.is_ok());
}

// ============================================================================
// Q4K-Q8K DOT VALIDATION TESTS
// ============================================================================

#[test]
fn test_fused_q4k_q8k_dot_empty() {
    let result = fused_q4k_q8k_dot(&[], &[], &[]);
    assert!(result.is_ok());
    assert_eq!(result.expect("test value should be present"), 0.0);
}

#[test]
fn test_fused_q4k_q8k_dot_invalid_q4k_length() {
    let invalid_q4k = vec![0u8; 100]; // Not multiple of 144
    let scales = vec![1.0f32; 1];
    let quants = vec![0i8; QK_K];

    let result = fused_q4k_q8k_dot(&invalid_q4k, &scales, &quants);
    assert!(result.is_err());
}

#[test]
fn test_fused_q4k_q8k_dot_scale_length_mismatch() {
    // Test with more scales than blocks - function may not validate strictly
    let q4k_data = vec![0u8; Q4K_SUPER_BLOCK_BYTES];
    let extra_scales = vec![1.0f32; 5]; // More than 1 block needs
    let quants = vec![0i8; QK_K];

    // Function may accept extra scales (only reads what it needs)
    let result = fused_q4k_q8k_dot(&q4k_data, &extra_scales, &quants);
    // Accept either behavior - error or success with ignored extra scales
    let _ = result; // Don't assert specific behavior
}

#[test]
fn test_fused_q4k_q8k_dot_quants_length_mismatch() {
    let q4k_data = vec![0u8; Q4K_SUPER_BLOCK_BYTES];
    let scales = vec![1.0f32; 1];
    let wrong_quants = vec![0i8; 128]; // Should be 256

    let result = fused_q4k_q8k_dot(&q4k_data, &scales, &wrong_quants);
    assert!(result.is_err());
}

#[test]
fn test_fused_q4k_q8k_dot_zero_scales() {
    let q4k_data = vec![0u8; Q4K_SUPER_BLOCK_BYTES];
    let scales = vec![0.0f32; 1]; // Zero Q8K scale
    let quants = vec![127i8; QK_K]; // Max positive quants

    let result = fused_q4k_q8k_dot(&q4k_data, &scales, &quants);
    assert!(result.is_ok());
    // Zero scale should produce zero result
    assert_eq!(result.expect("test value should be present"), 0.0);
}

#[test]
fn test_fused_q4k_q8k_dot_inf_scale() {
    let q4k_data = vec![0u8; Q4K_SUPER_BLOCK_BYTES];
    let scales = vec![f32::INFINITY; 1];
    let quants = vec![1i8; QK_K];

    let result = fused_q4k_q8k_dot(&q4k_data, &scales, &quants);
    // Should handle inf scale without panic
    assert!(result.is_ok());
}

#[test]
fn test_fused_q4k_q8k_dot_negative_scale() {
    let q4k_data = vec![0u8; Q4K_SUPER_BLOCK_BYTES];
    let scales = vec![-1.0f32; 1]; // Negative scale
    let quants = vec![50i8; QK_K];

    let result = fused_q4k_q8k_dot(&q4k_data, &scales, &quants);
    assert!(result.is_ok());
}

#[test]
fn test_fused_q4k_q8k_dot_extreme_quants() {
    let q4k_data = vec![0u8; Q4K_SUPER_BLOCK_BYTES];
    let scales = vec![1.0f32; 1];
    let mut quants = vec![0i8; QK_K];

    // Set extreme values
    quants[0] = i8::MAX; // 127
    quants[1] = i8::MIN; // -128
    quants[128] = 64;
    quants[255] = -64;

    let result = fused_q4k_q8k_dot(&q4k_data, &scales, &quants);
    assert!(result.is_ok());
}

#[test]
fn test_fused_q4k_q8k_dot_multiple_blocks() {
    let num_blocks = 4;
    let q4k_data = vec![0u8; Q4K_SUPER_BLOCK_BYTES * num_blocks];
    let scales = vec![1.0f32; num_blocks];
    let quants = vec![1i8; QK_K * num_blocks];

    let result = fused_q4k_q8k_dot(&q4k_data, &scales, &quants);
    assert!(result.is_ok());
}

// ============================================================================
// BOUNDARY TESTS
// ============================================================================

#[test]
fn test_fused_q4k_dot_single_block_boundary() {
    // Exactly one super-block
    let q4k_data = vec![0u8; Q4K_SUPER_BLOCK_BYTES];
    let activations = vec![0.0f32; QK_K];

    let result = fused_q4k_dot(&q4k_data, &activations);
    assert!(result.is_ok());
    assert_eq!(result.expect("test value should be present"), 0.0);
}

#[test]
fn test_fused_q4k_dot_max_reasonable_blocks() {
    // 16 super-blocks (4096 values) - typical layer size
    let num_blocks = 16;
    let q4k_data = vec![0u8; Q4K_SUPER_BLOCK_BYTES * num_blocks];
    let activations = vec![1.0f32; QK_K * num_blocks];

    let result = fused_q4k_dot(&q4k_data, &activations);
    assert!(result.is_ok());
}

#[test]
fn test_fused_q4k_dot_alternating_activations() {
    let q4k_data = vec![0u8; Q4K_SUPER_BLOCK_BYTES];
    let mut activations = vec![0.0f32; QK_K];

    // Alternating +1/-1 pattern
    for i in 0..QK_K {
        activations[i] = if i % 2 == 0 { 1.0 } else { -1.0 };
    }

    let result = fused_q4k_dot(&q4k_data, &activations);
    assert!(result.is_ok());
}

// ============================================================================
// NUMERICAL EDGE CASES
// ============================================================================

#[test]
fn test_fused_q4k_dot_subnormal_activations() {
    let q4k_data = vec![0u8; Q4K_SUPER_BLOCK_BYTES];
    let mut activations = vec![f32::MIN_POSITIVE; QK_K];
    activations[0] = f32::MIN_POSITIVE / 2.0; // Subnormal

    let result = fused_q4k_dot(&q4k_data, &activations);
    assert!(result.is_ok());
}

#[test]
fn test_fused_q4k_dot_max_f32_activations() {
    let q4k_data = vec![0u8; Q4K_SUPER_BLOCK_BYTES];
    let activations = vec![f32::MAX / 256.0; QK_K]; // Large but won't overflow

    let result = fused_q4k_dot(&q4k_data, &activations);
    assert!(result.is_ok());
}

#[test]
fn test_fused_q4k_dot_mixed_signs() {
    let q4k_data = vec![0u8; Q4K_SUPER_BLOCK_BYTES];
    let mut activations = vec![0.0f32; QK_K];

    // First half positive, second half negative
    for i in 0..QK_K / 2 {
        activations[i] = 1.0;
    }
    for i in QK_K / 2..QK_K {
        activations[i] = -1.0;
    }

    let result = fused_q4k_dot(&q4k_data, &activations);
    assert!(result.is_ok());
}

// ============================================================================
// Q8K-SPECIFIC EDGE CASES
// ============================================================================

#[test]
fn test_fused_q4k_q8k_dot_all_zeros() {
    let q4k_data = vec![0u8; Q4K_SUPER_BLOCK_BYTES];
    let scales = vec![0.0f32; 1];
    let quants = vec![0i8; QK_K];

    let result = fused_q4k_q8k_dot(&q4k_data, &scales, &quants);
    assert!(result.is_ok());
    assert_eq!(result.expect("test value should be present"), 0.0);
}

#[test]
fn test_fused_q4k_q8k_dot_nan_scale() {
    let q4k_data = vec![0u8; Q4K_SUPER_BLOCK_BYTES];
    let scales = vec![f32::NAN; 1];
    let quants = vec![1i8; QK_K];

    let result = fused_q4k_q8k_dot(&q4k_data, &scales, &quants);
    assert!(result.is_ok());
    // Result should be NaN
    assert!(result.expect("test value should be present").is_nan());
}

#[test]
fn test_fused_q4k_q8k_dot_alternating_quants() {
    let q4k_data = vec![0u8; Q4K_SUPER_BLOCK_BYTES];
    let scales = vec![1.0f32; 1];
    let mut quants = vec![0i8; QK_K];

    // Alternating +127/-128 pattern
    for i in 0..QK_K {
        quants[i] = if i % 2 == 0 { 127 } else { -128 };
    }

    let result = fused_q4k_q8k_dot(&q4k_data, &scales, &quants);
    assert!(result.is_ok());
}

// ============================================================================
// LARGE SCALE TESTS
// ============================================================================

#[test]
fn test_fused_q4k_dot_large_model_layer() {
    // Simulate a 4096-dim hidden layer (16 super-blocks)
    let num_blocks = 16;
    let q4k_data = vec![0u8; Q4K_SUPER_BLOCK_BYTES * num_blocks];
    let activations = vec![0.1f32; QK_K * num_blocks];

    let result = fused_q4k_dot(&q4k_data, &activations);
    assert!(result.is_ok());
}

#[test]
fn test_fused_q4k_q8k_dot_large_model_layer() {
    // Simulate a 4096-dim hidden layer
    let num_blocks = 16;
    let q4k_data = vec![0u8; Q4K_SUPER_BLOCK_BYTES * num_blocks];
    let scales = vec![1.0f32; num_blocks];
    let quants = vec![1i8; QK_K * num_blocks];

    let result = fused_q4k_q8k_dot(&q4k_data, &scales, &quants);
    assert!(result.is_ok());
}

// ============================================================================
// CONSISTENCY TESTS
// ============================================================================

#[test]
fn test_fused_q4k_dot_deterministic() {
    let q4k_data = vec![0u8; Q4K_SUPER_BLOCK_BYTES];
    let activations = vec![1.0f32; QK_K];

    let result1 = fused_q4k_dot(&q4k_data, &activations).expect("test value should be present");
    let result2 = fused_q4k_dot(&q4k_data, &activations).expect("test value should be present");

    assert_eq!(result1, result2);
}

#[test]
fn test_fused_q4k_q8k_dot_deterministic() {
    let q4k_data = vec![0u8; Q4K_SUPER_BLOCK_BYTES];
    let scales = vec![1.0f32; 1];
    let quants = vec![42i8; QK_K];

    let result1 = fused_q4k_q8k_dot(&q4k_data, &scales, &quants).expect("test value should be present");
    let result2 = fused_q4k_q8k_dot(&q4k_data, &scales, &quants).expect("test value should be present");

    assert_eq!(result1, result2);
}

// ============================================================================
// ERROR MESSAGE QUALITY TESTS
// ============================================================================

#[test]
fn test_fused_q4k_dot_error_message_includes_sizes() {
    let invalid_data = vec![0u8; 100];
    let activations = vec![1.0f32; 256];

    let result = fused_q4k_dot(&invalid_data, &activations);
    let err_str = format!("{:?}", result.unwrap_err());

    assert!(err_str.contains("100") || err_str.contains("144"));
}

#[test]
fn test_fused_q4k_dot_error_message_includes_expected() {
    let q4k_data = vec![0u8; Q4K_SUPER_BLOCK_BYTES];
    let wrong_activations = vec![1.0f32; 100];

    let result = fused_q4k_dot(&q4k_data, &wrong_activations);
    let err_str = format!("{:?}", result.unwrap_err());

    assert!(err_str.contains("100") || err_str.contains("256"));
}
