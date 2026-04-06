//! T-COV-95 Deep Coverage Bridge: quantize/mod.rs
//!
//! Targets: quantize_activations_q8k_into, quantize_to_q8_blocks,
//! dequantize_q8_blocks, InterleavedQ4K, f16_to_f32_lut.

use crate::quantize::*;

// ============================================================================
// quantize_activations_q8k_into
// ============================================================================

#[test]
fn test_q8k_into_valid_256_elements() {
    let activations: Vec<f32> = (0..256).map(|i| (i as f32) * 0.01).collect();
    let mut scales = vec![0.0f32; 1];
    let mut quants = vec![0i8; 256];
    let result = quantize_activations_q8k_into(&activations, &mut scales, &mut quants);
    assert!(result.is_ok());
    // Scale should be non-zero for non-zero input
    assert!(scales[0] != 0.0, "scale should be set");
}

#[test]
fn test_q8k_into_valid_512_elements() {
    let activations: Vec<f32> = (0..512).map(|i| (i as f32) * 0.1 - 25.6).collect();
    let mut scales = vec![0.0f32; 2];
    let mut quants = vec![0i8; 512];
    let result = quantize_activations_q8k_into(&activations, &mut scales, &mut quants);
    assert!(result.is_ok());
    assert!(scales[0] != 0.0);
    assert!(scales[1] != 0.0);
}

#[test]
fn test_q8k_into_not_multiple_of_256() {
    let activations = vec![1.0f32; 100]; // Not multiple of 256
    let mut scales = vec![0.0f32; 1];
    let mut quants = vec![0i8; 100];
    let result = quantize_activations_q8k_into(&activations, &mut scales, &mut quants);
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("256") || err.contains("multiple"),
        "got: {err}"
    );
}

#[test]
fn test_q8k_into_scales_buffer_too_small() {
    let activations = vec![1.0f32; 512]; // 2 super-blocks
    let mut scales = vec![0.0f32; 1]; // Need 2, only have 1
    let mut quants = vec![0i8; 512];
    let result = quantize_activations_q8k_into(&activations, &mut scales, &mut quants);
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("Scales") || err.contains("small"),
        "got: {err}"
    );
}

#[test]
fn test_q8k_into_quants_buffer_too_small() {
    let activations = vec![1.0f32; 256];
    let mut scales = vec![0.0f32; 1];
    let mut quants = vec![0i8; 128]; // Need 256, only have 128
    let result = quantize_activations_q8k_into(&activations, &mut scales, &mut quants);
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("Quants") || err.contains("small"),
        "got: {err}"
    );
}

#[test]
fn test_q8k_into_all_zeros() {
    let activations = vec![0.0f32; 256];
    let mut scales = vec![0.0f32; 1];
    let mut quants = vec![0i8; 256];
    let result = quantize_activations_q8k_into(&activations, &mut scales, &mut quants);
    assert!(result.is_ok());
}

#[test]
fn test_q8k_into_empty_input() {
    // 0 elements is a multiple of 256 (0 * 256 = 0)
    let activations: Vec<f32> = vec![];
    let mut scales: Vec<f32> = vec![];
    let mut quants: Vec<i8> = vec![];
    let result = quantize_activations_q8k_into(&activations, &mut scales, &mut quants);
    assert!(result.is_ok());
}

// ============================================================================
// quantize_to_q8_blocks
// ============================================================================

#[test]
fn test_q8_blocks_valid_32_elements() {
    let values: Vec<f32> = (0..32).map(|i| (i as f32) * 0.1).collect();
    let blocks = quantize_to_q8_blocks(&values).expect("test value should be present");
    assert_eq!(blocks.len(), 1);
}

#[test]
fn test_q8_blocks_valid_64_elements() {
    let values: Vec<f32> = (0..64).map(|i| (i as f32) * 0.5 - 16.0).collect();
    let blocks = quantize_to_q8_blocks(&values).expect("test value should be present");
    assert_eq!(blocks.len(), 2);
}

#[test]
fn test_q8_blocks_not_multiple_of_32() {
    let values = vec![1.0f32; 33];
    let result = quantize_to_q8_blocks(&values);
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(err.contains("32") || err.contains("multiple"), "got: {err}");
}

#[test]
fn test_q8_blocks_empty() {
    let values: Vec<f32> = vec![];
    let blocks = quantize_to_q8_blocks(&values).expect("test value should be present");
    assert!(blocks.is_empty());
}

// ============================================================================
// dequantize_q8_blocks (roundtrip)
// ============================================================================

#[test]
fn test_q8_blocks_roundtrip() {
    let original: Vec<f32> = (0..32).map(|i| (i as f32) * 0.1 - 1.6).collect();
    let blocks = quantize_to_q8_blocks(&original).expect("test value should be present");
    let dequantized = dequantize_q8_blocks(&blocks);
    assert_eq!(dequantized.len(), 32);
    // Quantization error should be small
    for (i, (orig, deq)) in original.iter().zip(dequantized.iter()).enumerate() {
        let error = (orig - deq).abs();
        assert!(
            error < 0.1,
            "element {i}: orig={orig}, deq={deq}, error={error}"
        );
    }
}

#[test]
fn test_q8_blocks_roundtrip_multi_block() {
    let original: Vec<f32> = (0..96).map(|i| (i as f32) * 0.05 - 2.4).collect();
    let blocks = quantize_to_q8_blocks(&original).expect("test value should be present");
    assert_eq!(blocks.len(), 3);
    let dequantized = dequantize_q8_blocks(&blocks);
    assert_eq!(dequantized.len(), 96);
    for (orig, deq) in original.iter().zip(dequantized.iter()) {
        assert!((orig - deq).abs() < 0.1, "orig={orig}, deq={deq}");
    }
}

#[test]
fn test_dequantize_q8_empty_blocks() {
    let blocks: Vec<Q8_0Block> = vec![];
    let result = dequantize_q8_blocks(&blocks);
    assert!(result.is_empty());
}

// ============================================================================
// InterleavedQ4K
// ============================================================================

#[test]
fn test_interleaved_q4k_invalid_length() {
    // Not a multiple of 144 (super-block size)
    let data = vec![0u8; 100];
    let result = InterleavedQ4K::from_q4k(&data);
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("144") || err.contains("super-block") || err.contains("multiple"),
        "got: {err}"
    );
}

#[test]
fn test_interleaved_q4k_empty_data() {
    let data: Vec<u8> = vec![];
    let interleaved = InterleavedQ4K::from_q4k(&data).expect("test value should be present");
    assert_eq!(interleaved.num_super_blocks, 0);
    assert_eq!(interleaved.num_values(), 0);
}

#[test]
fn test_interleaved_q4k_single_superblock() {
    // Create 144 bytes of synthetic Q4_K data
    let mut data = vec![0u8; 144];
    // Set d (f16) - use 0x3C00 = 1.0 in f16
    data[0] = 0x00;
    data[1] = 0x3C;
    // Set dmin (f16)
    data[2] = 0x00;
    data[3] = 0x38; // 0.5 in f16
                    // Scales (12 bytes at offset 4)
    for i in 4..16 {
        data[i] = 1;
    }
    // Quantized values (128 bytes at offset 16)
    for i in 16..144 {
        data[i] = 0x12; // some nibble values
    }

    let interleaved = InterleavedQ4K::from_q4k(&data).expect("test value should be present");
    assert_eq!(interleaved.num_super_blocks, 1);
    assert_eq!(interleaved.num_values(), 256); // QK_K = 256
    assert_eq!(interleaved.d.len(), 1);
    assert_eq!(interleaved.dmin.len(), 1);
    assert_eq!(interleaved.scales.len(), 12);
    assert_eq!(interleaved.qs.len(), 128);
}

#[test]
fn test_interleaved_q4k_two_superblocks() {
    let data = vec![0u8; 288]; // 2 * 144
    let interleaved = InterleavedQ4K::from_q4k(&data).expect("test value should be present");
    assert_eq!(interleaved.num_super_blocks, 2);
    assert_eq!(interleaved.num_values(), 512);
}

#[test]
fn test_interleaved_q4k_dot_length_mismatch() {
    let data = vec![0u8; 144];
    let interleaved = InterleavedQ4K::from_q4k(&data).expect("test value should be present");
    // Activation length doesn't match (need 256, provide 100)
    let activations = vec![1.0f32; 100];
    let result = interleaved.dot(&activations);
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("length") || err.contains("match"),
        "got: {err}"
    );
}

#[test]
fn test_interleaved_q4k_dot_valid() {
    // Create a valid single super-block
    let mut data = vec![0u8; 144];
    // d = 1.0 in f16 (0x3C00)
    data[0] = 0x00;
    data[1] = 0x3C;
    // dmin = 0.0
    data[2] = 0x00;
    data[3] = 0x00;
    // All other bytes zero

    let interleaved = InterleavedQ4K::from_q4k(&data).expect("test value should be present");
    let activations = vec![1.0f32; 256];
    let result = interleaved.dot(&activations);
    assert!(result.is_ok());
    // With all zeros in qs and dmin=0, result should be 0
    let dot_val = result.expect("test value should be present");
    assert!(
        dot_val.abs() < 1e-6,
        "expected ~0 for zero weights, got {dot_val}"
    );
}

// ============================================================================
// f16_to_f32_lut known values
// ============================================================================

#[test]
fn test_f16_lut_zero() {
    let val = f16_to_f32_lut(0x0000);
    assert!((val - 0.0).abs() < f32::EPSILON, "got: {val}");
}

#[test]
fn test_f16_lut_one() {
    // f16 1.0 = 0x3C00
    let val = f16_to_f32_lut(0x3C00);
    assert!((val - 1.0).abs() < 1e-6, "got: {val}");
}

#[test]
fn test_f16_lut_negative_one() {
    // f16 -1.0 = 0xBC00
    let val = f16_to_f32_lut(0xBC00);
    assert!((val - (-1.0)).abs() < 1e-6, "got: {val}");
}

#[test]
fn test_f16_lut_half() {
    // f16 0.5 = 0x3800
    let val = f16_to_f32_lut(0x3800);
    assert!((val - 0.5).abs() < 1e-6, "got: {val}");
}

#[test]
fn test_f16_lut_two() {
    // f16 2.0 = 0x4000
    let val = f16_to_f32_lut(0x4000);
    assert!((val - 2.0).abs() < 1e-6, "got: {val}");
}

// ============================================================================
// InterleavedQ4K Debug + Clone
// ============================================================================

#[test]
fn test_interleaved_q4k_debug() {
    let data = vec![0u8; 144];
    let interleaved = InterleavedQ4K::from_q4k(&data).expect("test value should be present");
    let debug = format!("{:?}", interleaved);
    assert!(debug.contains("InterleavedQ4K"));
    assert!(debug.contains("num_super_blocks: 1"));
}

#[test]
fn test_interleaved_q4k_clone() {
    let data = vec![0u8; 144];
    let interleaved = InterleavedQ4K::from_q4k(&data).expect("test value should be present");
    let cloned = interleaved.clone();
    assert_eq!(cloned.num_super_blocks, interleaved.num_super_blocks);
    assert_eq!(cloned.d.len(), interleaved.d.len());
    assert_eq!(cloned.qs.len(), interleaved.qs.len());
}
