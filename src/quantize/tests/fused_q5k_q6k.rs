
#[test]
fn test_fused_q5k_parallel_matvec_into_dim_mismatch_ext_cov() {
    let data = vec![0u8; 176]; // Valid for 1 row
    let activations = vec![0.0f32; 128]; // Wrong dimension
    let mut output = vec![0.0f32; 1];
    let result = fused_q5k_parallel_matvec_into(&data, &activations, 1, 256, &mut output);
    assert!(result.is_err());
}

#[test]
fn test_fused_q6k_parallel_matvec_into_invalid_input_ext_cov() {
    let data = vec![0u8; 100]; // Invalid
    let activations = vec![0.0f32; 256];
    let mut output = vec![0.0f32; 1];
    let result = fused_q6k_parallel_matvec_into(&data, &activations, 1, 256, &mut output);
    assert!(result.is_err());
}

#[test]
fn test_fused_q6k_parallel_matvec_into_dim_mismatch_ext_cov() {
    let data = vec![0u8; 210]; // Valid for 1 row
    let activations = vec![0.0f32; 128]; // Wrong dimension
    let mut output = vec![0.0f32; 1];
    let result = fused_q6k_parallel_matvec_into(&data, &activations, 1, 256, &mut output);
    assert!(result.is_err());
}

// =========================================================================
// Extended Coverage Tests: fused dot error paths
// =========================================================================

#[test]
fn test_fused_q6k_dot_invalid_data_ext_cov() {
    let data = vec![0u8; 100]; // Invalid size
    let activations = vec![0.0f32; 256];
    let result = fused_q6k_dot(&data, &activations);
    assert!(result.is_err());
}

#[test]
fn test_fused_q5k_dot_invalid_data_ext_cov() {
    let data = vec![0u8; 100]; // Invalid size
    let activations = vec![0.0f32; 256];
    let result = fused_q5k_dot(&data, &activations);
    assert!(result.is_err());
}

#[test]
fn test_fused_q6k_dot_simd_basic_ext_cov() {
    let data = vec![0u8; 210]; // Valid Q6_K super-block
    let activations = vec![0.0f32; 256];
    let result = fused_q6k_dot_simd(&data, &activations);
    assert!(result.is_ok());
}

#[test]
fn test_fused_q5k_dot_simd_basic_ext_cov() {
    let data = vec![0u8; 176]; // Valid Q5_K super-block
    let activations = vec![0.0f32; 256];
    let result = fused_q5k_dot_simd(&data, &activations);
    assert!(result.is_ok());
}

// =========================================================================
// Extended Coverage Tests: Q8K/Q4K fused operations
// =========================================================================

#[test]
fn test_fused_q4k_q8k_dot_simd_dim_mismatch_ext_cov() {
    let q4k_data = vec![0u8; 144]; // 1 super-block
    let scales = vec![1.0f32; 1];
    let quants = vec![0i8; 128]; // Wrong size
    let result = fused_q4k_q8k_dot_simd(&q4k_data, &scales, &quants);
    assert!(result.is_err());
}

#[test]
fn test_fused_q4k_q8k_dot_invalid_data_ext_cov() {
    let q4k_data = vec![0u8; 100]; // Invalid
    let scales = vec![1.0f32; 1];
    let quants = vec![0i8; 256];
    let result = fused_q4k_q8k_dot(&q4k_data, &scales, &quants);
    assert!(result.is_err());
}

// =========================================================================
// Extended Coverage Tests: parallel dequantize
// =========================================================================

#[test]
fn test_dequantize_q4_k_parallel_empty_ext_cov() {
    let result = dequantize_q4_k_parallel(&[]);
    assert!(result.is_ok());
    assert!(result.expect("quantization failed").is_empty());
}

#[test]
fn test_dequantize_q4_k_simd_empty_ext_cov() {
    let result = dequantize_q4_k_simd(&[]);
    assert!(result.is_ok());
    assert!(result.expect("quantization failed").is_empty());
}

#[test]
fn test_dequantize_q8_0_parallel_empty_ext_cov() {
    let result = dequantize_q8_0_parallel(&[]);
    assert!(result.is_ok());
    assert!(result.expect("quantization failed").is_empty());
}

#[test]
fn test_dequantize_q8_0_simd_empty_ext_cov() {
    let result = dequantize_q8_0_simd(&[]);
    assert!(result.is_ok());
    assert!(result.expect("quantization failed").is_empty());
}

// =========================================================================
// Extended Coverage Tests: Q4_0/Q8_0 parallel matvec
// =========================================================================

#[test]
fn test_fused_q4_0_q8_0_parallel_matvec_invalid_weights_ext_cov() {
    let weights = vec![0u8; 10]; // Invalid
    let activations = vec![1.0f32; 32];
    let result = fused_q4_0_q8_0_parallel_matvec(&weights, &activations, 32, 1);
    assert!(result.is_err());
}

#[test]
fn test_fused_q4_0_q8_0_parallel_matvec_dim_mismatch_ext_cov() {
    let weights = vec![0u8; 18]; // 1 block for 32 values
    let activations = vec![1.0f32; 16]; // Wrong size
    let result = fused_q4_0_q8_0_parallel_matvec(&weights, &activations, 32, 1);
    assert!(result.is_err());
}

#[test]
fn test_fused_q4_0_q8_0_parallel_matvec_into_invalid_weights_ext_cov() {
    let weights = vec![0u8; 10]; // Invalid
    let activations = vec![1.0f32; 32];
    let mut output = vec![0.0f32; 1];
    let result = fused_q4_0_q8_0_parallel_matvec_into(&weights, &activations, 32, &mut output);
    assert!(result.is_err());
}

#[test]
fn test_fused_q8_0_q8_0_parallel_matvec_invalid_weights_ext_cov() {
    let weights = vec![0u8; 10]; // Invalid
    let activations = vec![1.0f32; 32];
    let result = fused_q8_0_q8_0_parallel_matvec(&weights, &activations, 32, 1);
    assert!(result.is_err());
}

#[test]
fn test_fused_q8_0_q8_0_parallel_matvec_into_invalid_weights_ext_cov() {
    let weights = vec![0u8; 10]; // Invalid
    let activations = vec![1.0f32; 32];
    let mut output = vec![0.0f32; 1];
    let result = fused_q8_0_q8_0_parallel_matvec_into(&weights, &activations, 32, 32, &mut output);
    assert!(result.is_err());
}

// =========================================================================
// Extended Coverage Tests: FFN up/gate fused operations
// =========================================================================

#[test]
fn test_fused_q4k_q8k_ffn_up_gate_into_invalid_up_ext_cov() {
    let up_data = vec![0u8; 100]; // Invalid
    let gate_data = vec![0u8; 144];
    let scales = vec![1.0f32; 1];
    let quants = vec![0i8; 256];
    let mut up_output = vec![0.0f32; 1];
    let mut gate_output = vec![0.0f32; 1];
    let result = fused_q4k_q8k_ffn_up_gate_into(
        &up_data,
        &gate_data,
        &scales,
        &quants,
        1,
        256,
        &mut up_output,
        &mut gate_output,
    );
    assert!(result.is_err());
}

#[test]
fn test_fused_q4k_q8k_ffn_up_gate_into_invalid_gate_ext_cov() {
    let up_data = vec![0u8; 144];
    let gate_data = vec![0u8; 100]; // Invalid
    let scales = vec![1.0f32; 1];
    let quants = vec![0i8; 256];
    let mut up_output = vec![0.0f32; 1];
    let mut gate_output = vec![0.0f32; 1];
    let result = fused_q4k_q8k_ffn_up_gate_into(
        &up_data,
        &gate_data,
        &scales,
        &quants,
        1,
        256,
        &mut up_output,
        &mut gate_output,
    );
    assert!(result.is_err());
}

// =========================================================================
// Extended Coverage Tests: tiled matvec error paths
// =========================================================================

#[test]
fn test_fused_q4k_tiled_matvec_invalid_data_ext_cov() {
    let data = vec![0u8; 100]; // Invalid
    let activations = vec![0.0f32; 256];
    let result = fused_q4k_tiled_matvec(&data, &activations, 1, 256, Some(64));
    assert!(result.is_err());
}

#[test]
fn test_fused_q4k_tiled_matvec_dim_mismatch_ext_cov() {
    let data = vec![0u8; 144]; // Valid for 1 row
    let activations = vec![0.0f32; 128]; // Wrong dimension
    let result = fused_q4k_tiled_matvec(&data, &activations, 1, 256, Some(64));
    assert!(result.is_err());
}

#[test]
fn test_fused_q4k_parallel_matvec_invalid_data_ext_cov() {
    let data = vec![0u8; 100]; // Invalid
    let activations = vec![0.0f32; 256];
    let result = fused_q4k_parallel_matvec(&data, &activations, 1, 256);
    assert!(result.is_err());
}

#[test]
fn test_fused_q4k_parallel_matvec_dim_mismatch_ext_cov() {
    let data = vec![0u8; 144]; // Valid for 1 row
    let activations = vec![0.0f32; 128]; // Wrong dimension
    let result = fused_q4k_parallel_matvec(&data, &activations, 1, 256);
    assert!(result.is_err());
}

// =========================================================================
// Extended Coverage Tests: rmsnorm fused operations
// =========================================================================

#[test]
fn test_fused_rmsnorm_ffn_up_gate_weight_too_small_ext_cov() {
    let input = vec![1.0f32; 64];
    let norm_weight = vec![1.0f32; 64];
    let up_weights = vec![0u8; 10]; // Too small
    let gate_weights = vec![0u8; 10];

    let result = fused_rmsnorm_ffn_up_gate(
        &input,
        &norm_weight,
        0.00001,
        &up_weights,
        &gate_weights,
        64,
        1,
    );
    assert!(result.is_err());
}

#[test]
fn test_quantize_rmsnorm_q8_0_into_partial_block_ext_cov() {
    let input = vec![1.0f32; 48]; // Not multiple of 32
    let norm_weight = vec![1.0f32; 48];
    let eps = 1e-6;
    let (scales, quants) = quantize_rmsnorm_q8_0(&input, &norm_weight, eps);
    assert_eq!(scales.len(), 2); // 2 blocks needed for 48 values
    assert_eq!(quants.len(), 64); // 2 blocks * 32 quants
}

// =========================================================================
// Extended Coverage Tests: rope rotation edge cases
// =========================================================================

#[test]
fn test_apply_rope_rotation_simd_empty_ext_cov() {
    let mut x1: Vec<f32> = vec![];
    let mut x2: Vec<f32> = vec![];
    let cos_vals: Vec<f32> = vec![];
    let sin_vals: Vec<f32> = vec![];
    apply_rope_rotation_simd(&mut x1, &mut x2, &cos_vals, &sin_vals);
    assert!(x1.is_empty());
    assert!(x2.is_empty());
}

#[test]
fn test_apply_rope_rotation_simd_single_ext_cov() {
    let mut x1 = vec![1.0f32];
    let mut x2 = vec![0.0f32];
    let cos_vals = vec![1.0f32];
    let sin_vals = vec![0.0f32];
    apply_rope_rotation_simd(&mut x1, &mut x2, &cos_vals, &sin_vals);
    assert!((x1[0] - 1.0).abs() < 1e-6);
}

// =========================================================================
// Extended Coverage Tests: swiglu and softmax edge cases
// =========================================================================

#[test]
fn test_fused_swiglu_simd_empty_ext_cov() {
    let mut gate: Vec<f32> = vec![];
    let up: Vec<f32> = vec![];
    fused_swiglu_simd(&mut gate, &up);
    assert!(gate.is_empty());
}

#[test]
fn test_fused_swiglu_simd_single_ext_cov() {
    let mut gate = vec![0.0f32];
    let up = vec![1.0f32];
    fused_swiglu_simd(&mut gate, &up);
    // silu(0) = 0 / (1 + exp(-0)) = 0 / 2 = 0
    // result = 0 * 1 = 0
    assert!((gate[0]).abs() < 0.01);
}

#[test]
fn test_softmax_simd_empty_ext_cov() {
    let mut x: Vec<f32> = vec![];
    softmax_simd(&mut x);
    assert!(x.is_empty());
}

#[test]
fn test_softmax_simd_single_ext_cov() {
    let mut x = vec![5.0f32];
    softmax_simd(&mut x);
    assert!((x[0] - 1.0).abs() < 1e-6); // Single element should be 1.0
}

// =========================================================================
// Extended Coverage Tests: Q8_0Block and Q4_0Block operations
// =========================================================================

#[test]
fn test_q8_0_block_quantize_large_values_ext_cov() {
    let values: [f32; 32] = std::array::from_fn(|i| (i as f32) * 10.0);
    let block = Q8_0Block::quantize(&values);
    let dequant = block.dequantize();
    let error = values
        .iter()
        .zip(dequant.iter())
        .map(|(a, b)| (a - b).abs())
        .sum::<f32>()
        / 32.0;
    assert!(error < 2.0); // Acceptable quantization error
}

#[test]
fn test_q8_0_block_quantize_negative_large_ext_cov() {
    let values: [f32; 32] = std::array::from_fn(|i| -(i as f32) * 5.0);
    let block = Q8_0Block::quantize(&values);
    let dequant = block.dequantize();
    assert_eq!(dequant.len(), 32);
}

#[test]
fn test_interleaved_q4k_dot_multiple_blocks_ext_cov() {
    let q4k_data = vec![0u8; 288]; // 2 super-blocks
    let interleaved = InterleavedQ4K::from_q4k(&q4k_data).expect("valid");
    let activations = vec![0.0f32; 512];
    let result = interleaved.dot(&activations);
    assert!(result.is_ok());
}

// =========================================================================
// Extended Coverage Tests: Q8K SuperBlock operations
// =========================================================================

#[test]
fn test_q8k_superblock_quantize_large_ext_cov() {
    let values: [f32; 256] = std::array::from_fn(|i| (i as f32) * 0.5);
    let sb = Q8KSuperBlock::quantize(&values);
    let dequant = sb.dequantize();
    assert_eq!(dequant.len(), 256);
}

#[test]
fn test_q8k_superblock_quantize_all_same_ext_cov() {
    let values = [7.5f32; 256];
    let sb = Q8KSuperBlock::quantize(&values);
    assert!(sb.scale > 0.0);
    assert_eq!(sb.quants.len(), 256);
}

#[test]
fn test_q8k_superblock_quantize_into_multiple_ext_cov() {
    let values: [f32; 256] = std::array::from_fn(|i| if i < 128 { 1.0 } else { -1.0 });
    let values_slice = values.as_slice();
    let mut out_scale = 0.0f32;
    let mut out_quants = [0i8; 256];
    Q8KSuperBlock::quantize_into(values_slice, &mut out_scale, &mut out_quants);
    assert!(out_scale > 0.0);
}

// =========================================================================
// Deep Coverage Tests: InterleavedQ4K
// =========================================================================
