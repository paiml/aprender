
/// Test dequantize_q8_0_simd
#[test]
fn test_dequantize_q8_0_simd_basic() {
    let mut data = vec![0u8; 34];
    data[0..2].copy_from_slice(&0x3C00_u16.to_le_bytes()); // Scale = 1.0
    let result = dequantize_q8_0_simd(&data).expect("test");
    assert_eq!(result.len(), 32);
}

/// Test fused_q4_0_q8_0_parallel_matvec
#[test]
fn test_fused_q4_0_q8_0_parallel_matvec_coverage() {
    let num_rows = 2;
    let k = 32;
    // Q4_0 weight data
    let weight_data = vec![0u8; 18 * num_rows];
    // Float activations
    let activations = vec![1.0f32; k];
    let result =
        fused_q4_0_q8_0_parallel_matvec(&weight_data, &activations, k, num_rows).expect("test");
    assert_eq!(result.len(), num_rows);
}

/// Test fused_q8_0_q8_0_parallel_matvec
#[test]
fn test_fused_q8_0_q8_0_parallel_matvec_coverage() {
    let num_rows = 2;
    let k = 32;
    // Q8_0 weight data: 34 bytes per block of 32
    let mut weight_data = vec![0u8; 34 * num_rows];
    // Set scale to 1.0 for first block
    weight_data[0..2].copy_from_slice(&0x3C00_u16.to_le_bytes());
    weight_data[34..36].copy_from_slice(&0x3C00_u16.to_le_bytes());
    // Float activations (the function quantizes them internally)
    let activations = vec![1.0f32; k];
    let result =
        fused_q8_0_q8_0_parallel_matvec(&weight_data, &activations, k, num_rows).expect("test");
    assert_eq!(result.len(), num_rows);
}

/// Test quantize_activations_q8k_into
#[test]
fn test_quantize_activations_q8k_into_basic() {
    let activations = vec![1.0f32; 256];
    let mut scales = vec![0.0f32; 1];
    let mut quants = vec![0i8; 256];
    let _ = quantize_activations_q8k_into(&activations, &mut scales, &mut quants);
    assert!(scales[0] > 0.0);
    assert!(quants.iter().any(|&q| q != 0));
}

/// Test dequantize_q8_blocks
#[test]
fn test_dequantize_q8_blocks_basic() {
    let blocks = vec![Q8_0Block {
        scale: 1.0,
        quants: [64i8; 32],
    }];
    let result = dequantize_q8_blocks(&blocks);
    assert_eq!(result.len(), 32);
    for v in &result {
        assert!((v - 64.0).abs() < 1e-5);
    }
}

/// Test fused_rmsnorm_ffn_up_gate
#[test]
fn test_fused_rmsnorm_ffn_up_gate_basic() {
    let hidden_dim = 32;
    let intermediate_dim = 64;
    let input = vec![1.0f32; hidden_dim];
    let norm_weight = vec![1.0f32; hidden_dim];
    // Q4_0: 18 bytes per 32 values
    let up_weight = vec![0u8; 18 * (intermediate_dim * hidden_dim / 32)];
    let gate_weight = vec![0u8; 18 * (intermediate_dim * hidden_dim / 32)];

    let (up_result, gate_result) = fused_rmsnorm_ffn_up_gate(
        &input,
        &norm_weight,
        1e-5,
        &up_weight,
        &gate_weight,
        hidden_dim,
        intermediate_dim,
    )
    .expect("test");
    assert_eq!(up_result.len(), intermediate_dim);
    assert_eq!(gate_result.len(), intermediate_dim);
}

/// Test quantize_rmsnorm_q8_0_into
#[test]
fn test_quantize_rmsnorm_q8_0_into_coverage() {
    let input = vec![1.0f32; 32];
    let norm_weight = vec![1.0f32; 32];
    let mut scales = vec![0.0f32; 1];
    let mut quants = vec![0i8; 32];
    quantize_rmsnorm_q8_0_into(&input, &norm_weight, 1e-5, &mut scales, &mut quants);
    assert!(scales[0] > 0.0);
}

// =========================================================================
// Coverage Tests: fused_q4k_q8_dot functions
// =========================================================================

/// Test fused_q4k_q8_dot with valid inputs
#[test]
fn test_fused_q4k_q8_dot_basic() {
    // Create a Q4_K super-block (144 bytes for 256 values)
    let mut q4k_data = vec![0u8; 144];
    // Set d = 1.0 (f16)
    q4k_data[0..2].copy_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
    // Set dmin = 0.0 (f16)
    q4k_data[2..4].copy_from_slice(&half::f16::from_f32(0.0).to_le_bytes());
    // scales (12 bytes) and qs (128 bytes) remain zero

    // Create Q8_0 blocks (8 blocks for 256 values)
    let q8_blocks: Vec<Q8_0Block> = (0..8)
        .map(|_| Q8_0Block {
            scale: 1.0,
            quants: [1i8; 32],
        })
        .collect();

    let result = fused_q4k_q8_dot(&q4k_data, &q8_blocks);
    assert!(result.is_ok());
}

/// Test fused_q4k_q8_dot with invalid Q4_K data length
#[test]
fn test_fused_q4k_q8_dot_invalid_q4k_length() {
    let q4k_data = vec![0u8; 143]; // Not a multiple of 144
    let q8_blocks = vec![Q8_0Block {
        scale: 1.0,
        quants: [0i8; 32],
    }];
    let result = fused_q4k_q8_dot(&q4k_data, &q8_blocks);
    assert!(result.is_err());
}

/// Test fused_q4k_q8_dot with mismatched Q8 block count
#[test]
fn test_fused_q4k_q8_dot_mismatched_blocks() {
    let q4k_data = vec![0u8; 144]; // 1 super-block = 256 values
    // Only provide 4 Q8 blocks instead of 8
    let q8_blocks: Vec<Q8_0Block> = (0..4)
        .map(|_| Q8_0Block {
            scale: 1.0,
            quants: [0i8; 32],
        })
        .collect();
    let result = fused_q4k_q8_dot(&q4k_data, &q8_blocks);
    assert!(result.is_err());
}

// =========================================================================
// Coverage Tests: fused_q4k_q8k_dot functions
// =========================================================================

/// Test fused_q4k_q8k_dot with valid inputs
#[test]
fn test_fused_q4k_q8k_dot_basic() {
    // Q4_K super-block: 144 bytes for 256 values
    let mut q4k_data = vec![0u8; 144];
    q4k_data[0..2].copy_from_slice(&half::f16::from_f32(1.0).to_le_bytes());

    // Q8K format: 1 scale per 32 values = 8 scales for 256 values
    let q8k_scales = vec![1.0f32; 8];
    // 256 int8 quantized values
    let q8k_quants = vec![1i8; 256];

    let result = fused_q4k_q8k_dot(&q4k_data, &q8k_scales, &q8k_quants);
    assert!(result.is_ok());
}

/// Test fused_q4k_q8k_dot with invalid Q4_K data length
#[test]
fn test_fused_q4k_q8k_dot_invalid_length() {
    let q4k_data = vec![0u8; 145]; // Not a multiple of 144
    let q8k_scales = vec![1.0f32; 8];
    let q8k_quants = vec![1i8; 256];
    let result = fused_q4k_q8k_dot(&q4k_data, &q8k_scales, &q8k_quants);
    assert!(result.is_err());
}

/// Test fused_q4k_q8k_dot with multiple super-blocks works
#[test]
fn test_fused_q4k_q8k_dot_double_superblock() {
    let mut q4k_data = vec![0u8; 288]; // 2 super-blocks
    q4k_data[0..2].copy_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
    q4k_data[144..146].copy_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
    let q8k_scales = vec![1.0f32; 16]; // 16 scales for 512 values
    let q8k_quants = vec![1i8; 512];
    let result = fused_q4k_q8k_dot(&q4k_data, &q8k_scales, &q8k_quants);
    assert!(result.is_ok());
}

/// Test fused_q4k_q8k_dot with mismatched quants
#[test]
fn test_fused_q4k_q8k_dot_mismatched_quants() {
    let q4k_data = vec![0u8; 144];
    let q8k_scales = vec![1.0f32; 8];
    let q8k_quants = vec![1i8; 128]; // Should be 256
    let result = fused_q4k_q8k_dot(&q4k_data, &q8k_scales, &q8k_quants);
    assert!(result.is_err());
}

/// Test fused_q4k_q8k_dot_simd dispatches correctly
#[test]
fn test_fused_q4k_q8k_dot_simd_basic() {
    let mut q4k_data = vec![0u8; 144];
    q4k_data[0..2].copy_from_slice(&half::f16::from_f32(1.0).to_le_bytes());

    let q8k_scales = vec![1.0f32; 8];
    let q8k_quants = vec![1i8; 256];

    let result = fused_q4k_q8k_dot_simd(&q4k_data, &q8k_scales, &q8k_quants);
    assert!(result.is_ok());
}

/// Test fused_q4k_q8k_dot_simd error handling
#[test]
fn test_fused_q4k_q8k_dot_simd_error() {
    let q4k_data = vec![0u8; 100]; // Invalid length
    let q8k_scales = vec![1.0f32; 8];
    let q8k_quants = vec![1i8; 256];
    let result = fused_q4k_q8k_dot_simd(&q4k_data, &q8k_scales, &q8k_quants);
    assert!(result.is_err());
}

// =========================================================================
// Coverage Tests: fused_q4k_q8k_parallel_matvec_into
// =========================================================================

/// Test fused_q4k_q8k_parallel_matvec_into basic operation
#[test]
fn test_fused_q4k_q8k_parallel_matvec_into_basic() {
    let out_dim = 2;
    let in_dim = 256; // One Q4_K super-block per row
                      // 144 bytes per row
    let weight_data = vec![0u8; 144 * out_dim];
    let q8k_scales = vec![1.0f32; 8]; // 8 scales for 256 values
    let q8k_quants = vec![1i8; in_dim];
    let mut output = vec![0.0f32; out_dim];

    fused_q4k_q8k_parallel_matvec_into(
        &weight_data,
        &q8k_scales,
        &q8k_quants,
        in_dim,
        out_dim,
        &mut output,
    )
    .expect("should succeed");
    assert_eq!(output.len(), out_dim);
}

// =========================================================================
// Coverage Tests: fused_q4k_q8k_ffn_up_gate_into
// =========================================================================

/// Test fused_q4k_q8k_ffn_up_gate_into basic operation
#[test]
fn test_fused_q4k_q8k_ffn_up_gate_into_basic() {
    let hidden_dim: usize = 256;
    let intermediate_dim: usize = 256; // 1 super-block
    // Weight size: intermediate_dim rows * ceil(hidden_dim/256) super-blocks * 144 bytes
    let super_blocks_per_row = hidden_dim.div_ceil(256);
    let weight_size = intermediate_dim * super_blocks_per_row * 144;
    let up_weight = vec![0u8; weight_size];
    let gate_weight = vec![0u8; weight_size];
    let q8k_scales = vec![1.0f32; 8];
    let q8k_quants = vec![1i8; hidden_dim];
    let mut up_out = vec![0.0f32; intermediate_dim];
    let mut gate_out = vec![0.0f32; intermediate_dim];

    fused_q4k_q8k_ffn_up_gate_into(
        &up_weight,
        &gate_weight,
        &q8k_scales,
        &q8k_quants,
        hidden_dim,
        intermediate_dim,
        &mut up_out,
        &mut gate_out,
    )
    .expect("should succeed");
    assert_eq!(up_out.len(), intermediate_dim);
    assert_eq!(gate_out.len(), intermediate_dim);
}

// =========================================================================
// Coverage Tests: quantize_rmsnorm_q8_0 scalar path
// =========================================================================

/// Test quantize_rmsnorm_q8_0 with small input
#[test]
fn test_quantize_rmsnorm_q8_0_path() {
    let input = vec![1.0f32; 32];
    let norm_weight = vec![1.0f32; 32];
    let (scales, quants) = quantize_rmsnorm_q8_0(&input, &norm_weight, 1e-5);
    assert!(!scales.is_empty());
    assert!(!quants.is_empty());
}

/// Test quantize_rmsnorm_q8_0 with various sizes
#[test]
fn test_quantize_rmsnorm_q8_0_various_sizes() {
    for size in [32, 64, 128, 256] {
        let input = vec![0.5f32; size];
        let norm_weight = vec![2.0f32; size];
        let (scales, quants) = quantize_rmsnorm_q8_0(&input, &norm_weight, 1e-6);
        assert_eq!(scales.len(), size / 32);
        assert_eq!(quants.len(), size);
    }
}

// =========================================================================
// Coverage Tests: quantize_to_q8_blocks - extended
// =========================================================================

/// Test quantize_to_q8_blocks with multiple blocks
#[test]
fn test_quantize_to_q8_blocks_multiple() {
    let values = vec![0.5f32; 128]; // 4 blocks
    let blocks = quantize_to_q8_blocks(&values).expect("should succeed");
    assert_eq!(blocks.len(), 4);
}

// =========================================================================
// Coverage Tests: f16_to_f32
// =========================================================================

/// Test f16_to_f32 edge cases
#[test]
fn test_f16_to_f32_edge_cases() {
    // Zero
    assert_eq!(f16_to_f32(0x0000), 0.0);

    // One
    let one = half::f16::from_f32(1.0).to_bits();
    assert!((f16_to_f32(one) - 1.0).abs() < 1e-3);

    // Negative
    let neg = half::f16::from_f32(-2.0).to_bits();
    assert!((f16_to_f32(neg) - (-2.0)).abs() < 1e-2);

    // Small value
    let small = half::f16::from_f32(0.001).to_bits();
    assert!((f16_to_f32(small) - 0.001).abs() < 1e-3);
}

// =========================================================================
// Coverage Tests: dequantize_f16 - extended
// =========================================================================

/// Test dequantize_f16 with negative values
#[test]
fn test_dequantize_f16_negative_values() {
    let f16_neg = half::f16::from_f32(-1.5).to_le_bytes();
    let f16_pos = half::f16::from_f32(2.5).to_le_bytes();
    let data = [f16_neg[0], f16_neg[1], f16_pos[0], f16_pos[1]];

    let result = dequantize_f16(&data).expect("should succeed");
    assert_eq!(result.len(), 2);
    assert!((result[0] - (-1.5)).abs() < 1e-1);
    assert!((result[1] - 2.5).abs() < 1e-1);
}

// =========================================================================
// Coverage Tests: dequantize_q5_0 - extended
// =========================================================================

/// Test dequantize_q5_0 with nonzero values
#[test]
fn test_dequantize_q5_0_with_nonzero() {
    // Q5_0 block: 22 bytes per block
    let mut data = vec![0u8; 44]; // 2 blocks
    data[0..2].copy_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
    data[22..24].copy_from_slice(&half::f16::from_f32(2.0).to_le_bytes());
    // Set some nonzero qs values
    for i in 6..22 {
        data[i] = 0x55;
    }

    let result = dequantize_q5_0(&data).expect("should succeed");
    assert_eq!(result.len(), 64);
}

// =========================================================================
// Coverage Tests: dequantize_q5_1 - extended
// =========================================================================

/// Test dequantize_q5_1 with nonzero values
#[test]
fn test_dequantize_q5_1_with_nonzero() {
    // Q5_1 block: 24 bytes per block
    let mut data = vec![0u8; 48]; // 2 blocks
    data[0..2].copy_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
    data[24..26].copy_from_slice(&half::f16::from_f32(2.0).to_le_bytes());
    // Set some nonzero qs values
    for i in 8..24 {
        data[i] = 0xAA;
    }

    let result = dequantize_q5_1(&data).expect("should succeed");
    assert_eq!(result.len(), 64);
}

// =========================================================================
// Coverage Tests: dequantize_q6_k - extended
// =========================================================================

/// Test dequantize_q6_k multiple super-blocks
#[test]
fn test_dequantize_q6_k_multiple_superblocks() {
    // Q6_K super-block: 210 bytes for 256 values
    let mut data = vec![0u8; 420]; // 2 super-blocks
    data[208..210].copy_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
    data[418..420].copy_from_slice(&half::f16::from_f32(2.0).to_le_bytes());

    let result = dequantize_q6_k(&data).expect("should succeed");
    assert_eq!(result.len(), 512);
}

// =========================================================================
// Coverage Tests: quantize_activations_q8_0 - extended
// =========================================================================

/// Test quantize_activations_q8_0 with various values
#[test]
fn test_quantize_activations_q8_0_various_values() {
    let activations: Vec<f32> = (0..32).map(|i| i as f32 * 0.1).collect();
    let (scales, quants) = quantize_activations_q8_0(&activations);
    assert_eq!(scales.len(), 1);
    assert_eq!(quants.len(), 32);
    // Verify non-zero output
    assert!(quants.iter().any(|&q| q != 0));
}
