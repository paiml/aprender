
// =========================================================================
// Coverage Tests: quantize_to_q8_blocks
// =========================================================================

#[test]
fn test_quantize_to_q8_blocks_valid() {
    let values = vec![1.0f32; 64]; // 2 blocks
    let blocks = quantize_to_q8_blocks(&values).expect("test");
    assert_eq!(blocks.len(), 2);
}

#[test]
fn test_quantize_to_q8_blocks_error_cov() {
    let values = vec![1.0f32; 50]; // Not multiple of 32
    let result = quantize_to_q8_blocks(&values);
    assert!(result.is_err());
}

#[test]
fn test_dequantize_q8_blocks() {
    let values: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) / 10.0).collect();
    let blocks = quantize_to_q8_blocks(&values).expect("test");
    let dequant = dequantize_q8_blocks(&blocks);

    assert_eq!(dequant.len(), 32);
    for (o, d) in values.iter().zip(dequant.iter()) {
        assert!((o - d).abs() < 0.02);
    }
}

// =========================================================================
// Coverage Tests: Q8KSuperBlock::quantize_into
// =========================================================================

#[test]
fn test_q8ksuperblock_quantize_into() {
    let values: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) / 100.0).collect();
    let mut scale: f32 = 0.0;
    let mut quants = vec![0i8; 256];

    Q8KSuperBlock::quantize_into(&values, &mut scale, &mut quants);

    assert!(scale > 0.0);
    // Verify some quants are non-zero
    let non_zero: usize = quants.iter().filter(|&&q| q != 0).count();
    assert!(non_zero > 200);
}

// =========================================================================
// Coverage Tests: f16_to_f32 additional edge cases
// =========================================================================

#[test]
fn test_f16_to_f32_half_cov() {
    // f16 representation of 0.5 is 0x3800
    let half = f16_to_f32(0x3800);
    assert!((half - 0.5).abs() < 0.001);
}

#[test]
fn test_f16_to_f32_two_cov() {
    // f16 representation of 2.0 is 0x4000
    let two = f16_to_f32(0x4000);
    assert!((two - 2.0).abs() < 0.001);
}

#[test]
fn test_f16_to_f32_small_positive_cov() {
    // f16 representation of small positive number
    let small = f16_to_f32(0x0001);
    assert!(small > 0.0 && small < 0.0001);
}

// =========================================================================
// Coverage Tests: dequantize_q4_0 error paths
// =========================================================================

#[test]
fn test_dequantize_q4_0_empty_cov() {
    let data: Vec<u8> = vec![];
    let result = dequantize_q4_0(&data);
    // Empty data should work but return empty
    assert!(result.is_ok());
    assert!(result.expect("quantization failed").is_empty());
}

#[test]
fn test_dequantize_q4_0_invalid_size_cov() {
    // Q4_0 block is 18 bytes (2 for scale + 16 for 32 nibbles)
    let data = vec![0u8; 17]; // Invalid size
    let result = dequantize_q4_0(&data);
    assert!(result.is_err());
}

// =========================================================================
// Coverage Tests: dequantize_q8_0 error paths
// =========================================================================

#[test]
fn test_dequantize_q8_0_empty_cov() {
    let data: Vec<u8> = vec![];
    let result = dequantize_q8_0(&data);
    assert!(result.is_ok());
    assert!(result.expect("quantization failed").is_empty());
}

#[test]
fn test_dequantize_q8_0_invalid_size_cov() {
    // Q8_0 block is 34 bytes (2 for scale + 32 for int8)
    let data = vec![0u8; 33]; // Invalid size
    let result = dequantize_q8_0(&data);
    assert!(result.is_err());
}

// =========================================================================
// Coverage Tests: dequantize_f16 error paths
// =========================================================================

#[test]
fn test_dequantize_f16_empty_cov() {
    let data: Vec<u8> = vec![];
    let result = dequantize_f16(&data);
    assert!(result.is_ok());
    assert!(result.expect("quantization failed").is_empty());
}

#[test]
fn test_dequantize_f16_odd_size_cov() {
    // f16 requires 2 bytes per value
    let data = vec![0u8; 3]; // Odd, invalid
    let result = dequantize_f16(&data);
    assert!(result.is_err());
}

// =========================================================================
// Coverage Tests: Q8_0Block methods
// =========================================================================

#[test]
fn test_q8_0block_quantization_error_cov() {
    let values: [f32; 32] = std::array::from_fn(|i| (i as f32 - 16.0) * 0.1);
    let block = Q8_0Block::quantize(&values);
    let error = block.quantization_error(&values);
    // Error should be small
    assert!(error < 0.1);
}

#[test]
fn test_q8_0block_relative_error_cov() {
    let values: [f32; 32] = std::array::from_fn(|i| (i as f32 + 1.0) * 0.1);
    let block = Q8_0Block::quantize(&values);
    let rel_error = block.relative_error(&values);
    // Relative error should be small
    assert!(rel_error < 0.1);
}

// =========================================================================
// Coverage Tests: Q8KSuperBlock methods
// =========================================================================

#[test]
fn test_q8ksuperblock_quantize_cov() {
    let values: [f32; 256] = std::array::from_fn(|i| (i as f32 - 128.0) * 0.01);
    let block = Q8KSuperBlock::quantize(&values);
    assert!(block.scale > 0.0 || values.iter().all(|&v| v.abs() < 1e-6));
}

#[test]
fn test_q8ksuperblock_dequantize_cov() {
    let values: [f32; 256] = std::array::from_fn(|i| (i as f32 - 128.0) * 0.01);
    let block = Q8KSuperBlock::quantize(&values);
    let dequant = block.dequantize();
    assert_eq!(dequant.len(), 256);
    // Should roughly match original
    for (o, d) in values.iter().zip(dequant.iter()) {
        assert!((o - d).abs() < 0.05);
    }
}

// =========================================================================
// Coverage Tests: InterleavedQ4K methods
// =========================================================================

#[test]
fn test_interleavedq4k_num_values_cov() {
    // Create a minimal Q4K data (256 values per super-block)
    // Q4_K super-block is 144 bytes for 256 values
    let q4k_data = vec![0u8; 144];
    let interleaved = InterleavedQ4K::from_q4k(&q4k_data);
    assert!(interleaved.is_ok());
    let interleaved = interleaved.expect("quantization failed");
    assert_eq!(interleaved.num_values(), 256);
}

// =========================================================================
// Coverage Tests: fused_q4k_dot error paths
// =========================================================================

#[test]
fn test_fused_q4k_dot_length_mismatch_cov() {
    let q4k_data = vec![0u8; 144]; // 1 super-block = 256 values
    let activations = vec![1.0f32; 100]; // Wrong size
    let result = fused_q4k_dot(&q4k_data, &activations);
    assert!(result.is_err());
}

// =========================================================================
// Coverage Tests: fused_q6k_dot error paths
// =========================================================================

#[test]
fn test_fused_q6k_dot_length_mismatch_cov() {
    // Q6_K super-block is 210 bytes for 256 values
    let q6k_data = vec![0u8; 210];
    let activations = vec![1.0f32; 100]; // Wrong size
    let result = fused_q6k_dot(&q6k_data, &activations);
    assert!(result.is_err());
}

// =========================================================================
// Coverage Tests: fused_q5k_dot error paths
// =========================================================================

#[test]
fn test_fused_q5k_dot_length_mismatch_cov() {
    // Q5_K super-block is 176 bytes for 256 values
    let q5k_data = vec![0u8; 176];
    let activations = vec![1.0f32; 100]; // Wrong size
    let result = fused_q5k_dot(&q5k_data, &activations);
    assert!(result.is_err());
}

// =========================================================================
// Coverage Tests: quantize_activations_q8k_into success path
// =========================================================================

#[test]
fn test_quantize_activations_q8k_into_success_cov() {
    let activations: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) * 0.01).collect();
    let mut scales = vec![0.0f32; 1];
    let mut quants = vec![0i8; 256];

    let result = quantize_activations_q8k_into(&activations, &mut scales, &mut quants);
    assert!(result.is_ok());
    assert!(scales[0] > 0.0);
}

// =========================================================================
// Coverage Tests: dequantize_q4_1 error paths
// =========================================================================

#[test]
fn test_dequantize_q4_1_invalid_size_cov() {
    // Q4_1 block is 20 bytes (4 bytes min + 4 bytes max + 16 nibbles) for 32 values
    let data = vec![0u8; 19]; // Invalid - not a multiple of 20
    let result = dequantize_q4_1(&data);
    assert!(result.is_err());
}

#[test]
fn test_dequantize_q4_1_empty_cov() {
    let data: Vec<u8> = vec![];
    let result = dequantize_q4_1(&data);
    assert!(result.is_ok());
    assert!(result.expect("quantization failed").is_empty());
}

// =========================================================================
// Coverage Tests: dequantize_q5_0 error paths
// =========================================================================

#[test]
fn test_dequantize_q5_0_invalid_size_cov() {
    // Q5_0 block is 22 bytes (2 scale + 4 high bits + 16 nibbles) for 32 values
    let data = vec![0u8; 21]; // Invalid
    let result = dequantize_q5_0(&data);
    assert!(result.is_err());
}

#[test]
fn test_dequantize_q5_0_empty_cov() {
    let data: Vec<u8> = vec![];
    let result = dequantize_q5_0(&data);
    assert!(result.is_ok());
    assert!(result.expect("quantization failed").is_empty());
}

// =========================================================================
// Coverage Tests: dequantize_q5_1 error paths
// =========================================================================

#[test]
fn test_dequantize_q5_1_invalid_size_cov() {
    // Q5_1 block is 24 bytes
    let data = vec![0u8; 23]; // Invalid
    let result = dequantize_q5_1(&data);
    assert!(result.is_err());
}

#[test]
fn test_dequantize_q5_1_empty_cov() {
    let data: Vec<u8> = vec![];
    let result = dequantize_q5_1(&data);
    assert!(result.is_ok());
    assert!(result.expect("quantization failed").is_empty());
}

// =========================================================================
// Coverage Tests: fused_q4k_q8_dot error paths
// =========================================================================

#[test]
fn test_fused_q4k_q8_dot_invalid_q4k_size_cov() {
    let q4k_data = vec![0u8; 100]; // Invalid - not a multiple of 144
    let values: [f32; 32] = [0.0; 32];
    let q8_blocks = vec![Q8_0Block::quantize(&values); 8];
    let result = fused_q4k_q8_dot(&q4k_data, &q8_blocks);
    assert!(result.is_err());
}

#[test]
fn test_fused_q4k_q8_dot_block_mismatch_cov() {
    let q4k_data = vec![0u8; 144]; // 1 super-block = 256 values
    let values: [f32; 32] = [0.0; 32];
    let q8_blocks = vec![Q8_0Block::quantize(&values); 4]; // 4 * 32 = 128 values - mismatch
    let result = fused_q4k_q8_dot(&q4k_data, &q8_blocks);
    assert!(result.is_err());
}

// =========================================================================
// Coverage Tests: fused_q4k_q8k_dot error paths
// =========================================================================

#[test]
fn test_fused_q4k_q8k_dot_invalid_q4k_size_cov() {
    let q4k_data = vec![0u8; 100]; // Invalid
    let scales = vec![1.0f32; 1];
    let quants = vec![0i8; 256];
    let result = fused_q4k_q8k_dot(&q4k_data, &scales, &quants);
    assert!(result.is_err());
}

#[test]
fn test_fused_q4k_q8k_dot_scale_too_few_cov() {
    let q4k_data = vec![0u8; 288]; // 2 super-blocks = 512 values
    let scales = vec![1.0f32; 1]; // Only 1 scale but need 2
    let quants = vec![0i8; 512];
    let result = fused_q4k_q8k_dot(&q4k_data, &scales, &quants);
    assert!(result.is_err());
}

// =========================================================================
// Coverage Tests: quantize_to_q8_blocks error paths
// =========================================================================

#[test]
fn test_quantize_to_q8_blocks_not_multiple_cov() {
    let values = vec![1.0f32; 30]; // Not a multiple of 32
    let result = quantize_to_q8_blocks(&values);
    assert!(result.is_err());
}

#[test]
fn test_quantize_to_q8_blocks_empty_cov() {
    let values: Vec<f32> = vec![];
    let result = quantize_to_q8_blocks(&values);
    assert!(result.is_ok());
    assert!(result.expect("quantization failed").is_empty());
}

#[test]
fn test_quantize_to_q8_blocks_one_block_cov() {
    let values: Vec<f32> = (0..32).map(|i| i as f32 * 0.1).collect();
    let result = quantize_to_q8_blocks(&values);
    assert!(result.is_ok());
    assert_eq!(result.expect("quantization failed").len(), 1);
}

// =========================================================================
// Coverage Tests: dequantize_q8_blocks
// =========================================================================
