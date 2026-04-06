//! Part 24: Comprehensive coverage for quantize/mod.rs functions
//!
//! Target: 90% coverage for quantize/mod.rs
//! Focus on:
//! - quantize_activations_q8k_into error branches
//! - quantize_to_q8_blocks error branches
//! - InterleavedQ4K all paths
//! - fused_q4_0_q8_0_parallel_matvec all paths
//! - fused_q8_0_q8_0_parallel_matvec all paths
//! - f16_to_f32_lut

use crate::quantize::{
    dequantize_q8_blocks, f16_to_f32_lut, fused_q4_0_q8_0_parallel_matvec,
    fused_q4_0_q8_0_parallel_matvec_into, fused_q8_0_q8_0_parallel_matvec,
    fused_q8_0_q8_0_parallel_matvec_into, quantize_activations_q8k_into, quantize_to_q8_blocks,
    InterleavedQ4K, Q8_0Block,
};

// Import internal functions for direct testing
use crate::quantize::{
    extract_scale_min, extract_scale_min_from_slice, fused_q4_0_q8_0_dot_scalar,
    fused_q8_0_q8_0_dot_scalar,
};

// =============================================================================
// f16_to_f32_lut tests
// =============================================================================

#[test]
fn test_f16_lut_zero() {
    let result = f16_to_f32_lut(0);
    assert_eq!(result, 0.0);
}

#[test]
fn test_f16_lut_one() {
    // f16 representation of 1.0 is 0x3C00
    let result = f16_to_f32_lut(0x3C00);
    assert!((result - 1.0).abs() < 1e-6);
}

#[test]
fn test_f16_lut_negative_one() {
    // f16 representation of -1.0 is 0xBC00
    let result = f16_to_f32_lut(0xBC00);
    assert!((result - (-1.0)).abs() < 1e-6);
}

#[test]
fn test_f16_lut_half() {
    // f16 representation of 0.5 is 0x3800
    let result = f16_to_f32_lut(0x3800);
    assert!((result - 0.5).abs() < 1e-6);
}

#[test]
fn test_f16_lut_various_values() {
    // Test a range of values to ensure LUT is correctly populated
    let test_cases = [
        (0x0000u16, 0.0f32), // Zero
        (0x8000, -0.0),      // Negative zero
        (0x3C00, 1.0),       // One
        (0x4000, 2.0),       // Two
        (0x4200, 3.0),       // Three
        (0x3E00, 1.5),       // 1.5
        (0x4400, 4.0),       // Four
        (0x4500, 5.0),       // Five
    ];

    for (bits, expected) in test_cases {
        let result = f16_to_f32_lut(bits);
        assert!(
            (result - expected).abs() < 1e-3,
            "f16 bits {:#06x}: expected {}, got {}",
            bits,
            expected,
            result
        );
    }
}

#[test]
fn test_f16_lut_max_value() {
    // Max normal f16 value
    let result = f16_to_f32_lut(0x7BFF);
    assert!(result > 60000.0 && result < 70000.0);
}

#[test]
fn test_f16_lut_infinity() {
    // f16 positive infinity
    let result = f16_to_f32_lut(0x7C00);
    assert!(result.is_infinite() && result.is_sign_positive());
}

#[test]
fn test_f16_lut_nan() {
    // f16 NaN (any value with exp=0x1F and mantissa!=0)
    let result = f16_to_f32_lut(0x7C01);
    assert!(result.is_nan());
}

// =============================================================================
// quantize_activations_q8k_into tests
// =============================================================================

#[test]
fn test_q8k_into_not_multiple_of_256() {
    let activations = vec![0.0f32; 255]; // Not multiple of 256
    let mut scales = vec![0.0f32; 1];
    let mut quants = vec![0i8; 255];

    let result = quantize_activations_q8k_into(&activations, &mut scales, &mut quants);
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(err.contains("multiple of 256"));
}

#[test]
fn test_q8k_into_scales_too_small() {
    let activations = vec![0.0f32; 512]; // 2 superblocks
    let mut scales = vec![0.0f32; 1]; // Only 1 scale, need 2
    let mut quants = vec![0i8; 512];

    let result = quantize_activations_q8k_into(&activations, &mut scales, &mut quants);
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(err.contains("Scales buffer too small"));
}

#[test]
fn test_q8k_into_quants_too_small() {
    let activations = vec![0.0f32; 512]; // 2 superblocks
    let mut scales = vec![0.0f32; 2];
    let mut quants = vec![0i8; 256]; // Only 256, need 512

    let result = quantize_activations_q8k_into(&activations, &mut scales, &mut quants);
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(err.contains("Quants buffer too small"));
}

#[test]
fn test_q8k_into_success() {
    let activations = vec![1.0f32; 256];
    let mut scales = vec![0.0f32; 1];
    let mut quants = vec![0i8; 256];

    let result = quantize_activations_q8k_into(&activations, &mut scales, &mut quants);
    assert!(result.is_ok());
    // Scale should be non-zero for non-zero input
    assert!(scales[0].abs() > 0.0);
}

#[test]
fn test_q8k_into_two_superblocks() {
    let activations = vec![0.5f32; 512];
    let mut scales = vec![0.0f32; 2];
    let mut quants = vec![0i8; 512];

    let result = quantize_activations_q8k_into(&activations, &mut scales, &mut quants);
    assert!(result.is_ok());
    assert!(scales[0].abs() > 0.0);
    assert!(scales[1].abs() > 0.0);
}

// =============================================================================
// quantize_to_q8_blocks tests
// =============================================================================

#[test]
fn test_quantize_to_q8_blocks_not_multiple_of_32() {
    let values = vec![0.0f32; 33]; // Not multiple of 32
    let result = quantize_to_q8_blocks(&values);
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(err.contains("multiple of 32"));
}

#[test]
fn test_quantize_to_q8_blocks_success() {
    let values = vec![1.0f32; 64]; // 2 blocks
    let result = quantize_to_q8_blocks(&values);
    assert!(result.is_ok());
    let blocks = result.expect("test value should be present");
    assert_eq!(blocks.len(), 2);
}

#[test]
fn test_quantize_to_q8_blocks_empty() {
    let values: Vec<f32> = vec![];
    let result = quantize_to_q8_blocks(&values);
    assert!(result.is_ok());
    let blocks = result.expect("test value should be present");
    assert!(blocks.is_empty());
}

#[test]
fn test_quantize_to_q8_blocks_single_block() {
    let values = vec![0.5f32; 32];
    let result = quantize_to_q8_blocks(&values);
    assert!(result.is_ok());
    let blocks = result.expect("test value should be present");
    assert_eq!(blocks.len(), 1);
}

// =============================================================================
// dequantize_q8_blocks tests
// =============================================================================

#[test]
fn test_dequantize_q8_blocks_roundtrip() {
    let original = vec![0.5f32; 64];
    let blocks = quantize_to_q8_blocks(&original).expect("test value should be present");
    let dequantized = dequantize_q8_blocks(&blocks);

    assert_eq!(dequantized.len(), original.len());

    // Check approximate equality (quantization introduces error)
    for (o, d) in original.iter().zip(dequantized.iter()) {
        assert!((o - d).abs() < 0.1, "original={}, dequantized={}", o, d);
    }
}

#[test]
fn test_dequantize_q8_blocks_empty() {
    let blocks: Vec<Q8_0Block> = vec![];
    let result = dequantize_q8_blocks(&blocks);
    assert!(result.is_empty());
}

// =============================================================================
// InterleavedQ4K tests
// =============================================================================

#[test]
fn test_interleaved_q4k_invalid_size() {
    let data = vec![0u8; 100]; // Not multiple of 144
    let result = InterleavedQ4K::from_q4k(&data);
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(err.contains("not a multiple of super-block size"));
}

#[test]
fn test_interleaved_q4k_empty() {
    let data: Vec<u8> = vec![];
    let result = InterleavedQ4K::from_q4k(&data);
    assert!(result.is_ok());
    let interleaved = result.expect("test value should be present");
    assert_eq!(interleaved.num_super_blocks, 0);
    assert_eq!(interleaved.num_values(), 0);
}

#[test]
fn test_interleaved_q4k_single_superblock() {
    // Create valid Q4_K super-block data (144 bytes)
    let mut data = vec![0u8; 144];
    // Set d (f16 at offset 0-1) to some value
    data[0] = 0x00;
    data[1] = 0x3C; // f16 1.0

    // Set dmin (f16 at offset 2-3)
    data[2] = 0x00;
    data[3] = 0x00; // f16 0.0

    // Scales at offset 4-15 (12 bytes)
    // Quantized values at offset 16-143 (128 bytes)

    let result = InterleavedQ4K::from_q4k(&data);
    assert!(result.is_ok());
    let interleaved = result.expect("test value should be present");
    assert_eq!(interleaved.num_super_blocks, 1);
    assert_eq!(interleaved.num_values(), 256);
    assert_eq!(interleaved.d.len(), 1);
    assert_eq!(interleaved.dmin.len(), 1);
    assert_eq!(interleaved.scales.len(), 12);
    assert_eq!(interleaved.qs.len(), 128);
}

#[test]
fn test_interleaved_q4k_two_superblocks() {
    let data = vec![0u8; 288]; // 2 super-blocks
    let result = InterleavedQ4K::from_q4k(&data);
    assert!(result.is_ok());
    let interleaved = result.expect("test value should be present");
    assert_eq!(interleaved.num_super_blocks, 2);
    assert_eq!(interleaved.num_values(), 512);
}

#[test]
fn test_interleaved_q4k_dot_wrong_length() {
    let data = vec![0u8; 144];
    let interleaved = InterleavedQ4K::from_q4k(&data).expect("test value should be present");

    // Wrong activation length
    let activations = vec![1.0f32; 100]; // Should be 256
    let result = interleaved.dot(&activations);
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(err.contains("doesn't match"));
}

#[test]
fn test_interleaved_q4k_dot_success() {
    // Create Q4_K data with known values
    let mut data = vec![0u8; 144];
    // Set d to 1.0 (f16)
    data[0] = 0x00;
    data[1] = 0x3C;
    // Set some quantized values
    for i in 16..144 {
        data[i] = 0x55; // Arbitrary pattern
    }

    let interleaved = InterleavedQ4K::from_q4k(&data).expect("test value should be present");
    let activations = vec![1.0f32; 256];

    let result = interleaved.dot(&activations);
    assert!(result.is_ok());
    // Result should be finite
    assert!(result.expect("test value should be present").is_finite());
}

#[test]
fn test_interleaved_q4k_dot_zero_activations() {
    let data = vec![0u8; 144];
    let interleaved = InterleavedQ4K::from_q4k(&data).expect("test value should be present");

    let activations = vec![0.0f32; 256];
    let result = interleaved.dot(&activations).expect("test value should be present");
    assert_eq!(result, 0.0);
}

// =============================================================================
// fused_q4_0_q8_0_parallel_matvec tests
// =============================================================================

#[test]
fn test_fused_q4_0_q8_0_weight_too_small() {
    let weight_data = vec![0u8; 10]; // Too small
    let activations = vec![1.0f32; 32];

    let result = fused_q4_0_q8_0_parallel_matvec(&weight_data, &activations, 32, 1);
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(err.contains("weight data too small"));
}

#[test]
fn test_fused_q4_0_q8_0_activation_mismatch() {
    // Q4_0: 18 bytes per block of 32
    let weight_data = vec![0u8; 18]; // 1 row, 1 block
    let activations = vec![1.0f32; 64]; // Wrong size

    let result = fused_q4_0_q8_0_parallel_matvec(&weight_data, &activations, 32, 1);
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(err.contains("doesn't match in_dim"));
}

#[test]
fn test_fused_q4_0_q8_0_sequential_path() {
    // Small matrix - sequential path (out_dim < 1024)
    let in_dim = 32;
    let out_dim = 4;
    let _blocks_per_row = 1;
    let bytes_per_row = 18; // 1 block

    let weight_data = vec![0u8; out_dim * bytes_per_row];
    let activations = vec![1.0f32; in_dim];

    let result = fused_q4_0_q8_0_parallel_matvec(&weight_data, &activations, in_dim, out_dim);
    assert!(result.is_ok());
    let output = result.expect("test value should be present");
    assert_eq!(output.len(), out_dim);
}

#[test]
fn test_fused_q4_0_q8_0_parallel_path() {
    // Large matrix - parallel path (out_dim >= 1024)
    let in_dim = 32;
    let out_dim = 2048;
    let bytes_per_row = 18; // 1 block

    let weight_data = vec![0u8; out_dim * bytes_per_row];
    let activations = vec![0.5f32; in_dim];

    let result = fused_q4_0_q8_0_parallel_matvec(&weight_data, &activations, in_dim, out_dim);
    assert!(result.is_ok());
    let output = result.expect("test value should be present");
    assert_eq!(output.len(), out_dim);
}

#[test]
fn test_fused_q4_0_q8_0_boundary_1024() {
    // Exactly at threshold
    let in_dim = 32;
    let out_dim = 1024;
    let bytes_per_row = 18;

    let weight_data = vec![0u8; out_dim * bytes_per_row];
    let activations = vec![1.0f32; in_dim];

    let result = fused_q4_0_q8_0_parallel_matvec(&weight_data, &activations, in_dim, out_dim);
    assert!(result.is_ok());
    let output = result.expect("test value should be present");
    assert_eq!(output.len(), out_dim);
}

// =============================================================================
// fused_q4_0_q8_0_parallel_matvec_into tests
// =============================================================================

#[test]
fn test_fused_q4_0_q8_0_into_weight_too_small() {
    let weight_data = vec![0u8; 10];
    let activations = vec![1.0f32; 32];
    let mut output = vec![0.0f32; 1];

    let result = fused_q4_0_q8_0_parallel_matvec_into(&weight_data, &activations, 32, &mut output);
    assert!(result.is_err());
}

#[test]
fn test_fused_q4_0_q8_0_into_activation_mismatch() {
    let weight_data = vec![0u8; 18];
    let activations = vec![1.0f32; 64];
    let mut output = vec![0.0f32; 1];

    let result = fused_q4_0_q8_0_parallel_matvec_into(&weight_data, &activations, 32, &mut output);
    assert!(result.is_err());
}

#[test]
fn test_fused_q4_0_q8_0_into_success() {
    let in_dim = 32;
    let out_dim = 4;
    let weight_data = vec![0u8; out_dim * 18];
    let activations = vec![1.0f32; in_dim];
    let mut output = vec![0.0f32; out_dim];

    let result =
        fused_q4_0_q8_0_parallel_matvec_into(&weight_data, &activations, in_dim, &mut output);
    assert!(result.is_ok());
}

include!("fused_03.rs");
include!("fused_02_03.rs");
