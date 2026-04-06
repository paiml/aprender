
#[test]
fn test_interleaved_q4k_from_q4k_invalid_size_cov() {
    // Not a multiple of 144
    let data = vec![0u8; 100];
    let result = InterleavedQ4K::from_q4k(&data);
    assert!(result.is_err());
}

#[test]
fn test_interleaved_q4k_from_q4k_empty_deep2() {
    let data: Vec<u8> = vec![];
    let result = InterleavedQ4K::from_q4k(&data);
    assert!(result.is_ok());
    let iq4k = result.expect("quantization failed");
    assert_eq!(iq4k.num_super_blocks, 0);
}

// =========================================================================
// Coverage Tests: quantize_activations_q8k_into error paths (extended)
// =========================================================================

#[test]
fn test_quantize_activations_q8k_into_invalid_length_ext2_cov() {
    let activations = vec![1.0f32; 100]; // Not multiple of 256
    let mut scales = vec![0.0f32; 1];
    let mut quants = vec![0i8; 100];
    let result = quantize_activations_q8k_into(&activations, &mut scales, &mut quants);
    assert!(result.is_err());
}

#[test]
fn test_quantize_activations_q8k_into_scales_too_small_ext2_cov() {
    let activations = vec![1.0f32; 512]; // 2 super-blocks
    let mut scales = vec![0.0f32; 1]; // Only space for 1
    let mut quants = vec![0i8; 512];
    let result = quantize_activations_q8k_into(&activations, &mut scales, &mut quants);
    assert!(result.is_err());
}

#[test]
fn test_quantize_activations_q8k_into_quants_too_small_ext2_cov() {
    let activations = vec![1.0f32; 256];
    let mut scales = vec![0.0f32; 1];
    let mut quants = vec![0i8; 100]; // Too small
    let result = quantize_activations_q8k_into(&activations, &mut scales, &mut quants);
    assert!(result.is_err());
}

#[test]
fn test_quantize_activations_q8k_into_success_ext2_cov() {
    let activations = vec![1.5f32; 256];
    let mut scales = vec![0.0f32; 1];
    let mut quants = vec![0i8; 256];
    let result = quantize_activations_q8k_into(&activations, &mut scales, &mut quants);
    assert!(result.is_ok());
    assert!(scales[0] > 0.0);
}

// =========================================================================
// Coverage Tests: quantize_to_q8_blocks (extended)
// =========================================================================

#[test]
fn test_quantize_to_q8_blocks_invalid_length_ext2_cov() {
    let values = vec![1.0f32; 50]; // Not multiple of 32
    let result = quantize_to_q8_blocks(&values);
    assert!(result.is_err());
}

#[test]
fn test_quantize_to_q8_blocks_success_ext2_cov() {
    let values = vec![1.0f32; 64]; // 2 blocks
    let result = quantize_to_q8_blocks(&values);
    assert!(result.is_ok());
    let blocks = result.expect("quantization failed");
    assert_eq!(blocks.len(), 2);
}

#[test]
fn test_quantize_to_q8_blocks_empty_ext2_cov() {
    let values: Vec<f32> = vec![];
    let result = quantize_to_q8_blocks(&values);
    assert!(result.is_ok());
    assert!(result.expect("quantization failed").is_empty());
}

// =========================================================================
// Coverage Tests: dequantize_q8_blocks (extended)
// =========================================================================

#[test]
fn test_dequantize_q8_blocks_multiple_ext2_cov() {
    let blocks = vec![
        Q8_0Block {
            scale: 1.0,
            quants: [10i8; 32],
        },
        Q8_0Block {
            scale: 2.0,
            quants: [5i8; 32],
        },
    ];
    let result = dequantize_q8_blocks(&blocks);
    assert_eq!(result.len(), 64);
    // First block values
    assert!((result[0] - 10.0).abs() < 0.01);
    // Second block values
    assert!((result[32] - 10.0).abs() < 0.01);
}

// =========================================================================
// Coverage Tests: dequantize_q4_1 error path (extended)
// =========================================================================

#[test]
fn test_dequantize_q4_1_invalid_length_ext_cov() {
    let data = vec![0u8; 10]; // Not multiple of 20
    let result = dequantize_q4_1(&data);
    assert!(result.is_err());
}

#[test]
fn test_dequantize_q4_1_empty_ext_cov() {
    let data: Vec<u8> = vec![];
    let result = dequantize_q4_1(&data);
    assert!(result.is_ok());
    assert!(result.expect("quantization failed").is_empty());
}

// =========================================================================
// Coverage Tests: dequantize_q5_0 error path (extended)
// =========================================================================

#[test]
fn test_dequantize_q5_0_invalid_length_ext_cov() {
    let data = vec![0u8; 15]; // Not multiple of 22
    let result = dequantize_q5_0(&data);
    assert!(result.is_err());
}

#[test]
fn test_dequantize_q5_0_empty_ext_cov() {
    let data: Vec<u8> = vec![];
    let result = dequantize_q5_0(&data);
    assert!(result.is_ok());
    assert!(result.expect("quantization failed").is_empty());
}

// =========================================================================
// Coverage Tests: dequantize_q5_1 error path (extended)
// =========================================================================

#[test]
fn test_dequantize_q5_1_invalid_length_ext_cov() {
    let data = vec![0u8; 15]; // Not multiple of 24
    let result = dequantize_q5_1(&data);
    assert!(result.is_err());
}

#[test]
fn test_dequantize_q5_1_empty_ext_cov() {
    let data: Vec<u8> = vec![];
    let result = dequantize_q5_1(&data);
    assert!(result.is_ok());
    assert!(result.expect("quantization failed").is_empty());
}

// =========================================================================
// Coverage Tests: dequantize_q4_k error path (extended)
// =========================================================================

#[test]
fn test_dequantize_q4_k_invalid_length_ext_cov() {
    let data = vec![0u8; 100]; // Not multiple of 144
    let result = dequantize_q4_k(&data);
    assert!(result.is_err());
}

// =========================================================================
// Coverage Tests: dequantize_q5_k error path (extended)
// =========================================================================

#[test]
fn test_dequantize_q5_k_invalid_length_ext_cov() {
    let data = vec![0u8; 100]; // Not multiple of 176
    let result = dequantize_q5_k(&data);
    assert!(result.is_err());
}

// =========================================================================
// Coverage Tests: dequantize_q6_k error path (extended)
// =========================================================================

#[test]
fn test_dequantize_q6_k_invalid_length_ext_cov() {
    let data = vec![0u8; 100]; // Not multiple of 210
    let result = dequantize_q6_k(&data);
    assert!(result.is_err());
}

// =========================================================================
// Coverage Tests: Q8KSuperBlock::quantize_into
// =========================================================================

#[test]
fn test_q8k_superblock_quantize_into_cov() {
    let values = vec![1.0f32; 256];
    let mut scale = 0.0f32;
    let mut quants = vec![0i8; 256];
    Q8KSuperBlock::quantize_into(&values, &mut scale, &mut quants);
    assert!(scale > 0.0);
    // All same positive values should produce same quants
    assert_eq!(quants[0], quants[255]);
}

#[test]
fn test_q8k_superblock_quantize_into_varied_cov() {
    let mut values = vec![0.0f32; 256];
    for (i, v) in values.iter_mut().enumerate() {
        *v = (i as f32) * 0.5 - 64.0;
    }
    let mut scale = 0.0f32;
    let mut quants = vec![0i8; 256];
    Q8KSuperBlock::quantize_into(&values, &mut scale, &mut quants);
    assert!(scale > 0.0);
    // Should have varied quant values
    assert_ne!(quants[0], quants[200]);
}

// =========================================================================
// Coverage Tests: f16_to_f32 and f16_to_f32_lut
// =========================================================================

#[test]
fn test_f16_to_f32_lut_one_cov() {
    // f16 representation of 1.0 = 0x3C00
    let result = f16_to_f32_lut(0x3C00);
    assert!((result - 1.0).abs() < 0.001);
}

#[test]
fn test_f16_to_f32_lut_negative_one_cov() {
    // f16 representation of -1.0 = 0xBC00
    let result = f16_to_f32_lut(0xBC00);
    assert!((result + 1.0).abs() < 0.001);
}

#[test]
fn test_f16_to_f32_lut_half_cov() {
    // f16 representation of 0.5 = 0x3800
    let result = f16_to_f32_lut(0x3800);
    assert!((result - 0.5).abs() < 0.001);
}

// =========================================================================
// Coverage Tests: Block size and QK_K constants
// =========================================================================

#[test]
fn test_block_size_constant_cov() {
    assert_eq!(BLOCK_SIZE, 32);
}

#[test]
fn test_qk_k_constant_cov() {
    assert_eq!(QK_K, 256);
}

// =========================================================================
// Extended Coverage Tests for Q8_0Block methods
// =========================================================================

#[test]
fn test_q8_0_block_quantization_error_ext_cov() {
    let values = [1.0f32; 32];
    let block = Q8_0Block::quantize(&values);
    let error = block.quantization_error(&values);
    // Error should be small for uniform values
    assert!(error < 0.1);
}

#[test]
fn test_q8_0_block_quantization_error_zeros_ext_cov() {
    let values = [0.0f32; 32];
    let block = Q8_0Block::quantize(&values);
    let error = block.quantization_error(&values);
    assert!(error < 1e-5);
}

#[test]
fn test_q8_0_block_relative_error_ext_cov() {
    let values = [10.0f32; 32];
    let block = Q8_0Block::quantize(&values);
    let rel_error = block.relative_error(&values);
    // Relative error should be small
    assert!(rel_error < 0.01);
}

#[test]
fn test_q8_0_block_relative_error_zeros_ext_cov() {
    let values = [0.0f32; 32];
    let block = Q8_0Block::quantize(&values);
    let rel_error = block.relative_error(&values);
    // For zeros, relative error should be 0
    assert!(rel_error.abs() < 1e-5);
}

#[test]
fn test_q8_0_block_relative_error_varied_ext_cov() {
    let mut values = [0.0f32; 32];
    for (i, v) in values.iter_mut().enumerate() {
        *v = (i as f32 - 16.0) * 2.0;
    }
    let block = Q8_0Block::quantize(&values);
    let rel_error = block.relative_error(&values);
    assert!(rel_error < 0.05);
}

#[test]
fn test_q8_0_block_dequantize_roundtrip_ext_cov() {
    let mut values = [0.0f32; 32];
    for (i, v) in values.iter_mut().enumerate() {
        *v = (i as f32) * 0.5 - 8.0;
    }
    let block = Q8_0Block::quantize(&values);
    let deq = block.dequantize();
    for (orig, d) in values.iter().zip(deq.iter()) {
        let diff = (orig - d).abs();
        assert!(diff < 0.5); // Quantization tolerance
    }
}

// =========================================================================
// Extended Coverage Tests for f16_to_f32_lut
// =========================================================================

#[test]
fn test_f16_to_f32_lut_zero_cov() {
    let result = f16_to_f32_lut(0);
    assert!(result.abs() < 1e-10);
}

#[test]
fn test_f16_to_f32_lut_negative_zero_cov() {
    // Negative zero in f16 is 0x8000
    let result = f16_to_f32_lut(0x8000);
    assert!(result.abs() < 1e-10);
}

#[test]
fn test_f16_to_f32_lut_infinity_cov() {
    // Positive infinity in f16 is 0x7C00
    let result = f16_to_f32_lut(0x7C00);
    assert!(result.is_infinite() && result > 0.0);
}

#[test]
fn test_f16_to_f32_lut_negative_infinity_cov() {
    // Negative infinity in f16 is 0xFC00
    let result = f16_to_f32_lut(0xFC00);
    assert!(result.is_infinite() && result < 0.0);
}

#[test]
fn test_f16_to_f32_lut_small_positive_cov() {
    // f16 smallest positive normal: 0x0400
    let result = f16_to_f32_lut(0x0400);
    assert!(result > 0.0 && result < 1.0);
}

// =========================================================================
// Extended Coverage Tests for Q4_0Block
// =========================================================================

#[test]
fn test_q4_0_block_debug_ext_cov() {
    let block = Q4_0Block {
        scale: 1.0,
        quants: [0u8; 16],
    };
    let debug_str = format!("{:?}", block);
    assert!(debug_str.contains("Q4_0Block"));
}

#[test]
fn test_q4_0_block_clone_ext_cov() {
    let block = Q4_0Block {
        scale: 2.5,
        quants: [0xFF; 16],
    };
    let cloned = block.clone();
    assert!((cloned.scale - block.scale).abs() < 1e-6);
    assert_eq!(cloned.quants, block.quants);
}

// =========================================================================
// Extended Coverage Tests for dequantize_q4_0 success path
// =========================================================================

#[test]
fn test_dequantize_q4_0_one_block_ext_cov() {
    // Q4_0 block = 2 byte f16 scale + 16 byte quants = 18 bytes
    let mut data = vec![0u8; 18];
    // Scale = 1.0 in f16 little-endian: 0x3C00
    data[0..2].copy_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
    // All quants zero = all zeros out
    let result = dequantize_q4_0(&data);
    assert!(result.is_ok());
    let vals = result.expect("quantization failed");
    assert_eq!(vals.len(), 32);
}
