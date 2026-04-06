use crate::quantize::*;

#[test]
fn test_interleaved_q4k_from_q4k_invalid_length_cov() {
    // Not a multiple of 144
    let data = vec![0u8; 143];
    let result = InterleavedQ4K::from_q4k(&data);
    assert!(result.is_err());
}

#[test]
fn test_interleaved_q4k_from_q4k_empty_deep3() {
    let result = InterleavedQ4K::from_q4k(&[]);
    assert!(result.is_ok());
    let iq = result.expect("quantization failed");
    assert_eq!(iq.num_super_blocks, 0);
    assert_eq!(iq.num_values(), 0);
}

#[test]
fn test_interleaved_q4k_num_values_cov() {
    // One super-block = 256 values
    let data = vec![0u8; 144];
    let iq = InterleavedQ4K::from_q4k(&data).expect("quantization failed");
    assert_eq!(iq.num_values(), 256);
}

#[test]
fn test_interleaved_q4k_dot_dim_mismatch_cov() {
    let data = vec![0u8; 144];
    let iq = InterleavedQ4K::from_q4k(&data).expect("quantization failed");
    let activations = vec![1.0f32; 100]; // Wrong size
    let result = iq.dot(&activations);
    assert!(result.is_err());
}

#[test]
fn test_interleaved_q4k_dot_valid_cov() {
    // Create valid Q4_K data
    let mut data = vec![0u8; 144];
    // Set d=1.0 as f16
    let d_bytes = half::f16::from_f32(1.0).to_le_bytes();
    data[0] = d_bytes[0];
    data[1] = d_bytes[1];
    // dmin = 0.0
    data[2] = 0;
    data[3] = 0;

    let iq = InterleavedQ4K::from_q4k(&data).expect("quantization failed");
    let activations = vec![1.0f32; 256];
    let result = iq.dot(&activations);
    assert!(result.is_ok());
}

// =========================================================================
// Deep Coverage Tests: Q8_0Block additional methods
// =========================================================================

#[test]
fn test_q8_0_block_quantize_all_zeros_cov() {
    let values = [0.0f32; 32];
    let block = Q8_0Block::quantize(&values);
    // Scale should be minimal (1/127)
    assert!(block.scale > 0.0);
    assert!(block.scale < 0.01);
}

#[test]
fn test_q8_0_block_relative_error_near_zero_cov() {
    let values = [1e-12f32; 32];
    let block = Q8_0Block::quantize(&values);
    let rel_err = block.relative_error(&values);
    // Should return 0.0 for near-zero inputs
    assert_eq!(rel_err, 0.0);
}

#[test]
fn test_q8_0_block_quantization_error_deep2() {
    let values: [f32; 32] = std::array::from_fn(|i| i as f32 - 16.0);
    let block = Q8_0Block::quantize(&values);
    let error = block.quantization_error(&values);
    // Error should be small for linear values
    assert!(error < 0.5);
}

#[test]
fn test_q8_0_block_dequantize_roundtrip_cov() {
    let values: [f32; 32] = std::array::from_fn(|i| (i as f32 - 15.5) * 2.0);
    let block = Q8_0Block::quantize(&values);
    let dequantized = block.dequantize();
    // Check roundtrip error is reasonable
    for (orig, deq) in values.iter().zip(dequantized.iter()) {
        let err = (orig - deq).abs();
        assert!(err < 1.0, "Error too large: {} vs {}", orig, deq);
    }
}

// =========================================================================
// Deep Coverage Tests: Q8KSuperBlock
// =========================================================================

#[test]
fn test_q8k_superblock_quantize_alternating_cov() {
    let values: [f32; 256] = std::array::from_fn(|i| if i % 2 == 0 { 10.0 } else { -10.0 });
    let sb = Q8KSuperBlock::quantize(&values);
    assert!(sb.scale > 0.0);
    // Check that quants alternate in sign
    assert!(sb.quants[0] > 0);
    assert!(sb.quants[1] < 0);
}

#[test]
fn test_q8k_superblock_quantize_increasing_cov() {
    let values: [f32; 256] = std::array::from_fn(|i| (i as f32 - 128.0) / 10.0);
    let sb = Q8KSuperBlock::quantize(&values);
    assert!(sb.scale > 0.0);
    // First quant should be negative, last should be positive
    assert!(sb.quants[0] < 0);
    assert!(sb.quants[255] > 0);
}

// =========================================================================
// Deep Coverage Tests: quantize_to_q8_blocks
// =========================================================================

#[test]
fn test_quantize_to_q8_blocks_exact_blocks_cov() {
    let values: Vec<f32> = (0..64).map(|i| i as f32).collect();
    let blocks = quantize_to_q8_blocks(&values).expect("quantization failed");
    assert_eq!(blocks.len(), 2); // 64 values = 2 blocks
}

#[test]
fn test_quantize_to_q8_blocks_partial_block_cov() {
    // Function requires multiple of 32, so 50 should error
    let values: Vec<f32> = (0..50).map(|i| i as f32).collect();
    let result = quantize_to_q8_blocks(&values);
    // Should error because 50 is not a multiple of 32
    assert!(result.is_err());
}

#[test]
fn test_quantize_to_q8_blocks_empty_deep2() {
    let values: Vec<f32> = vec![];
    let blocks = quantize_to_q8_blocks(&values).expect("quantization failed");
    assert!(blocks.is_empty());
}

// =========================================================================
// Deep Coverage Tests: dequantize_q8_blocks
// =========================================================================

#[test]
fn test_dequantize_q8_blocks_roundtrip_deep2() {
    let values: Vec<f32> = (0..32).map(|i| i as f32 - 16.0).collect();
    let blocks = quantize_to_q8_blocks(&values).expect("quantization failed");
    let dequantized = dequantize_q8_blocks(&blocks);
    assert_eq!(dequantized.len(), 32);
    // Check roundtrip error
    for (orig, deq) in values.iter().zip(dequantized.iter()) {
        let err = (orig - deq).abs();
        assert!(err < 1.0);
    }
}

// =========================================================================
// Deep Coverage Tests: f16_to_f32
// =========================================================================

#[test]
fn test_f16_to_f32_special_values_cov() {
    // Zero
    assert_eq!(f16_to_f32(0x0000), 0.0);
    // One (f16 representation of 1.0)
    let one = half::f16::from_f32(1.0).to_bits();
    assert!((f16_to_f32(one) - 1.0).abs() < 1e-3);
    // Negative one
    let neg_one = half::f16::from_f32(-1.0).to_bits();
    assert!((f16_to_f32(neg_one) - (-1.0)).abs() < 1e-3);
}

#[test]
fn test_f16_to_f32_small_values_cov() {
    let small = half::f16::from_f32(0.001).to_bits();
    let result = f16_to_f32(small);
    assert!((result - 0.001).abs() < 1e-4);
}

// =========================================================================
// Deep Coverage Tests: dequantize_f16
// =========================================================================

#[test]
fn test_dequantize_f16_valid_deep2() {
    // 4 bytes = 2 f16 values
    let one = half::f16::from_f32(1.0).to_le_bytes();
    let two = half::f16::from_f32(2.0).to_le_bytes();
    let data = [one[0], one[1], two[0], two[1]];
    let result = dequantize_f16(&data).expect("quantization failed");
    assert_eq!(result.len(), 2);
    assert!((result[0] - 1.0).abs() < 1e-3);
    assert!((result[1] - 2.0).abs() < 1e-3);
}

#[test]
fn test_dequantize_f16_odd_length_cov() {
    let data = [0u8; 3]; // Not a multiple of 2
    let result = dequantize_f16(&data);
    assert!(result.is_err());
}

// =========================================================================
// Deep Coverage Tests: dequantize_q4_1
// =========================================================================

#[test]
fn test_dequantize_q4_1_valid_cov() {
    // Q4_1 block: 2 bytes scale + 2 bytes min + 16 bytes quants = 20 bytes
    let mut data = vec![0u8; 20];
    // scale = 1.0 as f16
    let scale = half::f16::from_f32(1.0).to_le_bytes();
    data[0] = scale[0];
    data[1] = scale[1];
    // min = 0.0 as f16
    data[2] = 0;
    data[3] = 0;
    // quants: all zeros

    let result = dequantize_q4_1(&data).expect("quantization failed");
    assert_eq!(result.len(), 32);
}

#[test]
fn test_dequantize_q4_1_invalid_length_deep2() {
    let data = vec![0u8; 19]; // Not a multiple of 20
    let result = dequantize_q4_1(&data);
    assert!(result.is_err());
}

// =========================================================================
// Deep Coverage Tests: dequantize_q5_0
// =========================================================================

#[test]
fn test_dequantize_q5_0_valid_cov() {
    // Q5_0 block: 2 bytes scale + 4 bytes high bits + 16 bytes low quants = 22 bytes
    let mut data = vec![0u8; 22];
    // scale = 1.0 as f16
    let scale = half::f16::from_f32(1.0).to_le_bytes();
    data[0] = scale[0];
    data[1] = scale[1];

    let result = dequantize_q5_0(&data).expect("quantization failed");
    assert_eq!(result.len(), 32);
}

#[test]
fn test_dequantize_q5_0_invalid_length_deep2() {
    let data = vec![0u8; 21]; // Not a multiple of 22
    let result = dequantize_q5_0(&data);
    assert!(result.is_err());
}

// =========================================================================
// Deep Coverage Tests: dequantize_q5_1
// =========================================================================

#[test]
fn test_dequantize_q5_1_valid_deep2() {
    // Q5_1 block: 2 bytes scale + 2 bytes min + 4 bytes high bits + 16 bytes low = 24 bytes
    let mut data = vec![0u8; 24];
    // scale = 1.0 as f16
    let scale = half::f16::from_f32(1.0).to_le_bytes();
    data[0] = scale[0];
    data[1] = scale[1];

    let result = dequantize_q5_1(&data).expect("quantization failed");
    assert_eq!(result.len(), 32);
}

#[test]
fn test_dequantize_q5_1_invalid_length_deep2() {
    let data = vec![0u8; 23]; // Not a multiple of 24
    let result = dequantize_q5_1(&data);
    assert!(result.is_err());
}

// =========================================================================
// Deep Coverage Tests: dequantize_q5_k
// =========================================================================

#[test]
fn test_dequantize_q5_k_valid_cov() {
    // Q5_K super-block: 176 bytes
    let data = vec![0u8; 176];
    let result = dequantize_q5_k(&data).expect("quantization failed");
    assert_eq!(result.len(), 256);
}

#[test]
fn test_dequantize_q5_k_invalid_length_cov() {
    let data = vec![0u8; 175]; // Not a multiple of 176
    let result = dequantize_q5_k(&data);
    assert!(result.is_err());
}

// =========================================================================
// Deep Coverage Tests: dequantize_q6_k
// =========================================================================

#[test]
fn test_dequantize_q6_k_valid_cov() {
    // Q6_K super-block: 210 bytes
    let data = vec![0u8; 210];
    let result = dequantize_q6_k(&data).expect("quantization failed");
    assert_eq!(result.len(), 256);
}

#[test]
fn test_dequantize_q6_k_invalid_length_cov() {
    let data = vec![0u8; 209]; // Not a multiple of 210
    let result = dequantize_q6_k(&data);
    assert!(result.is_err());
}

// =========================================================================
// Deep Coverage Tests: fused dot products
// =========================================================================

#[test]
fn test_fused_q4k_dot_valid_cov() {
    let data = vec![0u8; 144]; // One super-block
    let activations = vec![1.0f32; 256];
    let result = fused_q4k_dot(&data, &activations);
    assert!(result.is_ok());
}

#[test]
fn test_fused_q4k_dot_invalid_data_cov() {
    let data = vec![0u8; 143]; // Invalid length
    let activations = vec![1.0f32; 256];
    let result = fused_q4k_dot(&data, &activations);
    assert!(result.is_err());
}

#[test]
fn test_fused_q4k_dot_dim_mismatch_cov() {
    let data = vec![0u8; 144];
    let activations = vec![1.0f32; 100]; // Wrong size
    let result = fused_q4k_dot(&data, &activations);
    assert!(result.is_err());
}

#[test]
fn test_fused_q4k_dot_simd_valid_cov() {
    let data = vec![0u8; 144];
    let activations = vec![1.0f32; 256];
    let result = fused_q4k_dot_simd(&data, &activations);
    assert!(result.is_ok());
}

#[test]
fn test_fused_q6k_dot_valid_cov() {
    let data = vec![0u8; 210];
    let activations = vec![1.0f32; 256];
    let result = fused_q6k_dot(&data, &activations);
    assert!(result.is_ok());
}

#[test]
fn test_fused_q5k_dot_valid_cov() {
    let data = vec![0u8; 176];
    let activations = vec![1.0f32; 256];
    let result = fused_q5k_dot(&data, &activations);
    assert!(result.is_ok());
}

// =========================================================================
// Deep Coverage Tests: fused_q4k_q8_dot
// =========================================================================

#[test]
fn test_fused_q4k_q8_dot_valid_cov() {
    let q4k_data = vec![0u8; 144];
    let q8_blocks: Vec<Q8_0Block> = (0..8)
        .map(|_| Q8_0Block {
            scale: 1.0,
            quants: [0i8; 32],
        })
        .collect();
    let result = fused_q4k_q8_dot(&q4k_data, &q8_blocks);
    assert!(result.is_ok());
}

#[test]
fn test_fused_q4k_q8_dot_invalid_q4k_cov() {
    let q4k_data = vec![0u8; 143]; // Invalid length
    let q8_blocks: Vec<Q8_0Block> = (0..8)
        .map(|_| Q8_0Block {
            scale: 1.0,
            quants: [0i8; 32],
        })
        .collect();
    let result = fused_q4k_q8_dot(&q4k_data, &q8_blocks);
    assert!(result.is_err());
}

// =========================================================================
// Deep Coverage Tests: fused_q4k_q8k_dot
// =========================================================================

#[test]
fn test_fused_q4k_q8k_dot_valid_cov() {
    let q4k_data = vec![0u8; 144];
    let q8k_scales = vec![1.0f32; 1]; // One scale per super-block
    let q8k_quants = vec![0i8; 256];
    let result = fused_q4k_q8k_dot(&q4k_data, &q8k_scales, &q8k_quants);
    assert!(result.is_ok());
}

#[test]
fn test_fused_q4k_q8k_dot_invalid_q4k_cov() {
    let q4k_data = vec![0u8; 143];
    let q8k_scales = vec![1.0f32; 1];
    let q8k_quants = vec![0i8; 256];
    let result = fused_q4k_q8k_dot(&q4k_data, &q8k_scales, &q8k_quants);
    assert!(result.is_err());
}

// =========================================================================
// Deep Coverage Tests: quantize_activations_q8k_into
// =========================================================================

#[test]
fn test_quantize_activations_q8k_into_valid_cov() {
    let activations: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) / 10.0).collect();
    let mut scales = vec![0.0f32; 1]; // One scale per 256 values
    let mut quants = vec![0i8; 256];
    let result = quantize_activations_q8k_into(&activations, &mut scales, &mut quants);
    assert!(result.is_ok());
    assert!(scales[0] > 0.0);
}

include!("quantize_activations_03.rs");
include!("q4_1_matmul.rs");
