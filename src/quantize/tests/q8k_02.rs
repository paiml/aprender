//! T-COV-95 Coverage Bridge: quantize/mod.rs extended
//!
//! Targets: quantize_activations_q8k_into, f16_to_f32_lut, dequantize functions,
//! error paths and edge cases.

use crate::error::RealizarError;
use crate::quantize::{
    dequantize_q4_0, dequantize_q4_1, dequantize_q5_0, dequantize_q5_1, dequantize_q8_0,
    quantize_activations_q8k_into, BLOCK_SIZE, QK_K,
};

// ============================================================================
// quantize_activations_q8k_into tests
// ============================================================================

#[test]
fn test_q8k_into_valid_single_superblock() {
    let activations = vec![1.0f32; 256];
    let mut scales = vec![0.0f32; 1];
    let mut quants = vec![0i8; 256];

    let result = quantize_activations_q8k_into(&activations, &mut scales, &mut quants);
    assert!(result.is_ok());
    assert!(scales[0] > 0.0); // Scale should be non-zero
}

#[test]
fn test_q8k_into_valid_multiple_superblocks() {
    let activations = vec![0.5f32; 512];
    let mut scales = vec![0.0f32; 2];
    let mut quants = vec![0i8; 512];

    let result = quantize_activations_q8k_into(&activations, &mut scales, &mut quants);
    assert!(result.is_ok());
}

#[test]
fn test_q8k_into_error_not_multiple_of_256() {
    let activations = vec![1.0f32; 100]; // Not multiple of 256
    let mut scales = vec![0.0f32; 1];
    let mut quants = vec![0i8; 100];

    let result = quantize_activations_q8k_into(&activations, &mut scales, &mut quants);
    assert!(result.is_err());
    match result {
        Err(RealizarError::FormatError { reason }) => {
            assert!(reason.contains("multiple of 256"));
        },
        _ => panic!("Expected FormatError"),
    }
}

#[test]
fn test_q8k_into_error_scales_too_small() {
    let activations = vec![1.0f32; 512]; // 2 superblocks
    let mut scales = vec![0.0f32; 1]; // Only 1 scale
    let mut quants = vec![0i8; 512];

    let result = quantize_activations_q8k_into(&activations, &mut scales, &mut quants);
    assert!(result.is_err());
    match result {
        Err(RealizarError::InvalidShape { reason }) => {
            assert!(reason.contains("Scales buffer too small"));
        },
        _ => panic!("Expected InvalidShape"),
    }
}

#[test]
fn test_q8k_into_error_quants_too_small() {
    let activations = vec![1.0f32; 256];
    let mut scales = vec![0.0f32; 1];
    let mut quants = vec![0i8; 100]; // Too small

    let result = quantize_activations_q8k_into(&activations, &mut scales, &mut quants);
    assert!(result.is_err());
    match result {
        Err(RealizarError::InvalidShape { reason }) => {
            assert!(reason.contains("Quants buffer too small"));
        },
        _ => panic!("Expected InvalidShape"),
    }
}

#[test]
fn test_q8k_into_empty_input() {
    let activations: Vec<f32> = vec![];
    let mut scales: Vec<f32> = vec![];
    let mut quants: Vec<i8> = vec![];

    // Empty is a multiple of 256 (0 superblocks)
    let result = quantize_activations_q8k_into(&activations, &mut scales, &mut quants);
    assert!(result.is_ok());
}

#[test]
fn test_q8k_into_varied_values() {
    let mut activations = vec![0.0f32; 256];
    for i in 0..256 {
        activations[i] = (i as f32 - 128.0) / 128.0; // Range -1 to 1
    }
    let mut scales = vec![0.0f32; 1];
    let mut quants = vec![0i8; 256];

    let result = quantize_activations_q8k_into(&activations, &mut scales, &mut quants);
    assert!(result.is_ok());
    // Should have varied quants
    let unique: std::collections::HashSet<i8> = quants.iter().copied().collect();
    assert!(unique.len() > 1);
}

// ============================================================================
// dequantize_q4_0 tests
// ============================================================================

#[test]
fn test_dequant_q4_0_single_block() {
    // Q4_0: 18 bytes per 32 values (2 byte f16 scale + 16 bytes quants)
    let mut data = vec![0u8; 18];
    // Scale = 1.0 in f16 (0x3C00)
    data[0..2].copy_from_slice(&0x3C00u16.to_le_bytes());
    // All quants = 0 (neutral)
    let result = dequantize_q4_0(&data);
    assert!(result.is_ok());
    assert_eq!(result.expect("test value should be present").len(), 32);
}

#[test]
fn test_dequant_q4_0_multiple_blocks() {
    let num_blocks = 4;
    let data = vec![0u8; num_blocks * 18];
    let result = dequantize_q4_0(&data);
    assert!(result.is_ok());
    assert_eq!(result.expect("test value should be present").len(), num_blocks * 32);
}

#[test]
fn test_dequant_q4_0_empty() {
    let data: Vec<u8> = vec![];
    let result = dequantize_q4_0(&data);
    assert!(result.is_ok());
    assert!(result.expect("test value should be present").is_empty());
}

// ============================================================================
// dequantize_q8_0 tests
// ============================================================================

#[test]
fn test_dequant_q8_0_single_block() {
    // Q8_0: 34 bytes per 32 values (2 byte f16 scale + 32 bytes quants)
    let mut data = vec![0u8; 34];
    // Scale = 1.0 in f16
    data[0..2].copy_from_slice(&0x3C00u16.to_le_bytes());
    let result = dequantize_q8_0(&data);
    assert!(result.is_ok());
    assert_eq!(result.expect("test value should be present").len(), 32);
}

#[test]
fn test_dequant_q8_0_multiple_blocks() {
    let num_blocks = 4;
    let data = vec![0u8; num_blocks * 34];
    let result = dequantize_q8_0(&data);
    assert!(result.is_ok());
    assert_eq!(result.expect("test value should be present").len(), num_blocks * 32);
}

#[test]
fn test_dequant_q8_0_nonzero_quants() {
    let mut data = vec![0u8; 34];
    // Scale = 1.0 in f16
    data[0..2].copy_from_slice(&0x3C00u16.to_le_bytes());
    // Set some quants to non-zero
    for i in 2..34 {
        data[i] = ((i - 2) % 256) as u8;
    }
    let result = dequantize_q8_0(&data);
    assert!(result.is_ok());
    let values = result.expect("test value should be present");
    assert!(values.iter().any(|&v| v != 0.0));
}

// ============================================================================
// dequantize_q4_1 tests
// ============================================================================

#[test]
fn test_dequant_q4_1_single_block() {
    // Q4_1: 20 bytes per 32 values (2 byte delta + 2 byte min + 16 bytes quants)
    let mut data = vec![0u8; 20];
    // Delta = 1.0 in f16
    data[0..2].copy_from_slice(&0x3C00u16.to_le_bytes());
    // Min = 0.0 in f16
    data[2..4].copy_from_slice(&0x0000u16.to_le_bytes());
    let result = dequantize_q4_1(&data);
    assert!(result.is_ok());
    assert_eq!(result.expect("test value should be present").len(), 32);
}

#[test]
fn test_dequant_q4_1_multiple_blocks() {
    let num_blocks = 3;
    let data = vec![0u8; num_blocks * 20];
    let result = dequantize_q4_1(&data);
    assert!(result.is_ok());
    assert_eq!(result.expect("test value should be present").len(), num_blocks * 32);
}

// ============================================================================
// dequantize_q5_0 tests
// ============================================================================

#[test]
fn test_dequant_q5_0_single_block() {
    // Q5_0: 22 bytes per 32 values (2 byte scale + 4 byte high_bits + 16 bytes low_quants)
    let mut data = vec![0u8; 22];
    // Scale = 1.0 in f16
    data[0..2].copy_from_slice(&0x3C00u16.to_le_bytes());
    let result = dequantize_q5_0(&data);
    assert!(result.is_ok());
    assert_eq!(result.expect("test value should be present").len(), 32);
}

#[test]
fn test_dequant_q5_0_multiple_blocks() {
    let num_blocks = 2;
    let data = vec![0u8; num_blocks * 22];
    let result = dequantize_q5_0(&data);
    assert!(result.is_ok());
    assert_eq!(result.expect("test value should be present").len(), num_blocks * 32);
}

// ============================================================================
// dequantize_q5_1 tests
// ============================================================================

#[test]
fn test_dequant_q5_1_single_block() {
    // Q5_1: 24 bytes per 32 values (2 byte delta + 2 byte min + 4 byte high + 16 bytes low)
    let mut data = vec![0u8; 24];
    // Delta = 1.0 in f16
    data[0..2].copy_from_slice(&0x3C00u16.to_le_bytes());
    // Min = 0.0 in f16
    data[2..4].copy_from_slice(&0x0000u16.to_le_bytes());
    let result = dequantize_q5_1(&data);
    assert!(result.is_ok());
    assert_eq!(result.expect("test value should be present").len(), 32);
}

#[test]
fn test_dequant_q5_1_multiple_blocks() {
    let num_blocks = 2;
    let data = vec![0u8; num_blocks * 24];
    let result = dequantize_q5_1(&data);
    assert!(result.is_ok());
    assert_eq!(result.expect("test value should be present").len(), num_blocks * 32);
}

// ============================================================================
// Constants tests
// ============================================================================

#[test]
fn test_block_size_constant() {
    assert_eq!(BLOCK_SIZE, 32);
}

#[test]
fn test_qk_k_constant() {
    assert_eq!(QK_K, 256);
}

// ============================================================================
// Edge cases
// ============================================================================

#[test]
fn test_dequant_q4_0_large() {
    // 100 blocks = 3200 values
    let data = vec![0u8; 100 * 18];
    let result = dequantize_q4_0(&data);
    assert!(result.is_ok());
    assert_eq!(result.expect("test value should be present").len(), 3200);
}

#[test]
fn test_dequant_q8_0_all_positive_quants() {
    let mut data = vec![0u8; 34];
    // Scale = 0.01 in f16 (approximately 0x211E)
    data[0..2].copy_from_slice(&0x211Eu16.to_le_bytes());
    // All quants = 127 (max positive i8)
    for i in 2..34 {
        data[i] = 127;
    }
    let result = dequantize_q8_0(&data);
    assert!(result.is_ok());
    let values = result.expect("test value should be present");
    assert!(values.iter().all(|&v| v > 0.0));
}

#[test]
fn test_dequant_q8_0_all_negative_quants() {
    let mut data = vec![0u8; 34];
    // Scale = 0.01 in f16
    data[0..2].copy_from_slice(&0x211Eu16.to_le_bytes());
    // All quants = -128 (min i8) as u8 = 128
    for i in 2..34 {
        data[i] = 128; // -128 as u8
    }
    let result = dequantize_q8_0(&data);
    assert!(result.is_ok());
    let values = result.expect("test value should be present");
    assert!(values.iter().all(|&v| v < 0.0));
}

#[test]
fn test_q8k_into_large_values() {
    // Test with large values that need scaling
    let mut activations = vec![0.0f32; 256];
    for i in 0..256 {
        activations[i] = (i as f32) * 100.0; // Large values
    }
    let mut scales = vec![0.0f32; 1];
    let mut quants = vec![0i8; 256];

    let result = quantize_activations_q8k_into(&activations, &mut scales, &mut quants);
    assert!(result.is_ok());
    // Scale should be large to accommodate the range
    assert!(scales[0] > 10.0);
}

#[test]
fn test_q8k_into_negative_values() {
    let activations = vec![-1.0f32; 256];
    let mut scales = vec![0.0f32; 1];
    let mut quants = vec![0i8; 256];

    let result = quantize_activations_q8k_into(&activations, &mut scales, &mut quants);
    assert!(result.is_ok());
    // All quants should be negative
    assert!(quants.iter().all(|&q| q < 0));
}
