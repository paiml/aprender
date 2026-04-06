use crate::quantize::*;
/// Test Q8KSuperBlock quantize_into
#[test]
fn test_q8k_superblock_quantize_into() {
    let values = [2.0f32; 256];
    let mut scale = 0.0f32;
    let mut quants = [0i8; 256];
    Q8KSuperBlock::quantize_into(&values, &mut scale, &mut quants);
    assert!(scale > 0.0);
    // All values should be quantized to max (127)
    for &q in &quants {
        assert_eq!(q, 127);
    }
}

/// Test Q8KSuperBlock dequantize round-trip
#[test]
fn test_q8k_superblock_dequantize_roundtrip() {
    let mut original = [0.0f32; 256];
    for i in 0..256 {
        original[i] = (i as f32 - 128.0) / 10.0;
    }
    let block = Q8KSuperBlock::quantize(&original);
    let recovered = block.dequantize();

    // Check round-trip error is reasonable
    let max_error: f32 = original
        .iter()
        .zip(recovered.iter())
        .map(|(o, r)| (o - r).abs())
        .fold(0.0f32, f32::max);
    assert!(max_error < 0.2); // Q8 has ~1% error
}

/// Test Q8KSuperBlock Debug and Clone traits
#[test]
fn test_q8k_superblock_traits() {
    let values = [1.0f32; 256];
    let block = Q8KSuperBlock::quantize(&values);
    let debug_str = format!("{:?}", block);
    assert!(debug_str.contains("scale"));
    let cloned = block.clone();
    assert_eq!(cloned.scale, block.scale);
}

// =========================================================================
// Coverage Tests: quantize_activations_q8k_into error paths
// =========================================================================

/// Test quantize_activations_q8k_into with invalid length
#[test]
fn test_quantize_activations_q8k_into_invalid_length() {
    let activations = vec![1.0f32; 100]; // Not multiple of 256
    let mut scales = vec![0.0f32; 1];
    let mut quants = vec![0i8; 256];
    let result = quantize_activations_q8k_into(&activations, &mut scales, &mut quants);
    assert!(result.is_err());
}

/// Test quantize_activations_q8k_into with too small scales buffer
#[test]
fn test_quantize_activations_q8k_into_small_scales() {
    let activations = vec![1.0f32; 512]; // 2 super-blocks
    let mut scales = vec![0.0f32; 1]; // Need 2
    let mut quants = vec![0i8; 512];
    let result = quantize_activations_q8k_into(&activations, &mut scales, &mut quants);
    assert!(result.is_err());
}

/// Test quantize_activations_q8k_into with too small quants buffer
#[test]
fn test_quantize_activations_q8k_into_small_quants() {
    let activations = vec![1.0f32; 256];
    let mut scales = vec![0.0f32; 1];
    let mut quants = vec![0i8; 100]; // Need 256
    let result = quantize_activations_q8k_into(&activations, &mut scales, &mut quants);
    assert!(result.is_err());
}

/// Test quantize_activations_q8k_into success case
#[test]
fn test_quantize_activations_q8k_into_success() {
    let activations = vec![1.0f32; 256];
    let mut scales = vec![0.0f32; 1];
    let mut quants = vec![0i8; 256];
    let result = quantize_activations_q8k_into(&activations, &mut scales, &mut quants);
    assert!(result.is_ok());
    assert!(scales[0] > 0.0);
}

// =========================================================================
// Coverage Tests: fused_rmsnorm_ffn_up_gate error paths
// =========================================================================

/// Test fused_rmsnorm_ffn_up_gate input dimension error
#[test]
fn test_fused_rmsnorm_ffn_up_gate_input_dim_error() {
    let input = vec![1.0f32; 16]; // Wrong size
    let norm_weight = vec![1.0f32; 32];
    let up_weight = vec![0u8; 36];
    let gate_weight = vec![0u8; 36];
    let result =
        fused_rmsnorm_ffn_up_gate(&input, &norm_weight, 1e-5, &up_weight, &gate_weight, 32, 1);
    assert!(result.is_err());
}

/// Test fused_rmsnorm_ffn_up_gate up_weight too small
#[test]
fn test_fused_rmsnorm_ffn_up_gate_up_weight_error() {
    let input = vec![1.0f32; 32];
    let norm_weight = vec![1.0f32; 32];
    let up_weight = vec![0u8; 10]; // Too small
    let gate_weight = vec![0u8; 36];
    let result =
        fused_rmsnorm_ffn_up_gate(&input, &norm_weight, 1e-5, &up_weight, &gate_weight, 32, 2);
    assert!(result.is_err());
}

/// Test fused_rmsnorm_ffn_up_gate gate_weight too small
#[test]
fn test_fused_rmsnorm_ffn_up_gate_gate_weight_error() {
    let input = vec![1.0f32; 32];
    let norm_weight = vec![1.0f32; 32];
    let up_weight = vec![0u8; 36];
    let gate_weight = vec![0u8; 10]; // Too small
    let result =
        fused_rmsnorm_ffn_up_gate(&input, &norm_weight, 1e-5, &up_weight, &gate_weight, 32, 2);
    assert!(result.is_err());
}

// =========================================================================
// Coverage Tests: fused_rmsnorm_q4_0_matmul error paths
// =========================================================================

/// Test fused_rmsnorm_q4_0_matmul input dimension error
#[test]
fn test_fused_rmsnorm_q4_0_matmul_input_dim_error() {
    let input = vec![1.0f32; 16]; // Wrong size
    let norm_weight = vec![1.0f32; 32];
    let weight_data = vec![0u8; 36];
    let result = fused_rmsnorm_q4_0_matmul(&input, &norm_weight, 1e-5, &weight_data, 32, 1);
    assert!(result.is_err());
}

/// Test fused_rmsnorm_q4_0_matmul weight data too small
#[test]
fn test_fused_rmsnorm_q4_0_matmul_weight_data_error() {
    let input = vec![1.0f32; 32];
    let norm_weight = vec![1.0f32; 32];
    let weight_data = vec![0u8; 10]; // Too small
    let result = fused_rmsnorm_q4_0_matmul(&input, &norm_weight, 1e-5, &weight_data, 32, 2);
    assert!(result.is_err());
}

// =========================================================================
// Coverage Tests: Q8_0Block methods
// =========================================================================

/// Test Q8_0Block quantize basic
#[test]
fn test_q8_0_block_quantize_basic() {
    let values = [1.0f32; 32];
    let block = Q8_0Block::quantize(&values);
    assert!(block.scale > 0.0);
    for &q in &block.quants {
        assert_eq!(q, 127);
    }
}

/// Test Q8_0Block quantize near-zero
#[test]
fn test_q8_0_block_quantize_near_zero() {
    let values = [1e-12f32; 32];
    let block = Q8_0Block::quantize(&values);
    assert!((block.scale - 1.0 / 127.0).abs() < 1e-6);
}

/// Test Q8_0Block dequantize
#[test]
fn test_q8_0_block_dequantize() {
    let mut original = [0.0f32; 32];
    for i in 0..32 {
        original[i] = (i as f32 - 16.0) / 10.0;
    }
    let block = Q8_0Block::quantize(&original);
    let recovered = block.dequantize();

    let max_error: f32 = original
        .iter()
        .zip(recovered.iter())
        .map(|(o, r)| (o - r).abs())
        .fold(0.0f32, f32::max);
    assert!(max_error < 0.1);
}

/// Test Q8_0Block quantization_error
#[test]
fn test_q8_0_block_quantization_error() {
    let values = [1.5f32; 32];
    let block = Q8_0Block::quantize(&values);
    let error = block.quantization_error(&values);
    assert!(error >= 0.0);
    assert!(error < 0.1);
}

/// Test Q8_0Block relative_error
#[test]
fn test_q8_0_block_relative_error() {
    let values = [2.0f32; 32];
    let block = Q8_0Block::quantize(&values);
    let error = block.relative_error(&values);
    assert!(error >= 0.0);
    assert!(error < 0.05);
}

// =========================================================================
// Coverage Tests: fused_q5k_dot_simd
// =========================================================================

/// Test fused_q5k_dot_simd basic
#[test]
fn test_fused_q5k_dot_simd_basic() {
    // Q5_K super-block is 176 bytes for 256 values
    let mut q5k_data = vec![0u8; 176];
    q5k_data[0..2].copy_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
    let activations = vec![1.0f32; 256];
    let result = fused_q5k_dot_simd(&q5k_data, &activations);
    assert!(result.is_ok());
}

/// Test fused_q5k_dot_simd invalid length
#[test]
fn test_fused_q5k_dot_simd_invalid_length() {
    let q5k_data = vec![0u8; 100]; // Not multiple of 176
    let activations = vec![1.0f32; 256];
    let result = fused_q5k_dot_simd(&q5k_data, &activations);
    assert!(result.is_err());
}

// =========================================================================
// Coverage Tests: fused_q6k_dot_simd error handling
// =========================================================================

/// Test fused_q6k_dot_simd invalid length
#[test]
fn test_fused_q6k_dot_simd_invalid_length() {
    let q6k_data = vec![0u8; 100]; // Not multiple of 210
    let activations = vec![1.0f32; 256];
    let result = fused_q6k_dot_simd(&q6k_data, &activations);
    assert!(result.is_err());
}

/// Test fused_q6k_dot_simd dimension mismatch
#[test]
fn test_fused_q6k_dot_simd_dim_mismatch() {
    let q6k_data = vec![0u8; 210];
    let activations = vec![1.0f32; 128]; // Should be 256
    let result = fused_q6k_dot_simd(&q6k_data, &activations);
    assert!(result.is_err());
}

// =========================================================================
// Coverage Tests: quantize_rmsnorm_q8_0 with various sizes
// =========================================================================

/// Test quantize_rmsnorm_q8_0 with partial block
#[test]
fn test_quantize_rmsnorm_q8_0_partial_block() {
    let input = vec![1.0f32; 48]; // 1 full block + 16 partial
    let norm_weight = vec![1.0f32; 48];
    let (scales, quants) = quantize_rmsnorm_q8_0(&input, &norm_weight, 1e-5);
    assert_eq!(scales.len(), 2);
    assert_eq!(quants.len(), 64); // Padded to 32-element blocks
}

/// Test quantize_rmsnorm_q8_0 with large input
#[test]
fn test_quantize_rmsnorm_q8_0_large() {
    let input: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) / 64.0).collect();
    let norm_weight = vec![1.0f32; 256];
    let (scales, quants) = quantize_rmsnorm_q8_0(&input, &norm_weight, 1e-5);
    assert_eq!(scales.len(), 8);
    assert_eq!(quants.len(), 256);
    // Verify quants array is populated (values in i8 range by type)
    assert!(!quants.is_empty());
    // Ensure we have some non-zero values
    assert!(quants.iter().any(|&q| q != 0));
}

/// Test quantize_rmsnorm_q8_0 with near-zero max abs
#[test]
fn test_quantize_rmsnorm_q8_0_near_zero_max() {
    let input = vec![1e-12f32; 32];
    let norm_weight = vec![1.0f32; 32];
    let (scales, _quants) = quantize_rmsnorm_q8_0(&input, &norm_weight, 1e-5);
    // Should use fallback scale
    assert!(scales[0] > 0.0);
}

// =========================================================================
// Coverage Tests: dequantize_q8_blocks
// =========================================================================

/// Test dequantize_q8_blocks with multiple blocks
#[test]
fn test_dequantize_q8_blocks_multiple() {
    let blocks = vec![
        Q8_0Block {
            scale: 1.0,
            quants: [64i8; 32],
        },
        Q8_0Block {
            scale: 2.0,
            quants: [32i8; 32],
        },
    ];
    let result = dequantize_q8_blocks(&blocks);
    assert_eq!(result.len(), 64);
    assert!((result[0] - 64.0).abs() < 1e-5);
    assert!((result[32] - 64.0).abs() < 1e-5); // 32 * 2.0
}

// =========================================================================
// Coverage Tests: fused_q5k_dot scalar
// =========================================================================

/// Test fused_q5k_dot basic
#[test]
fn test_fused_q5k_dot_basic() {
    let mut q5k_data = vec![0u8; 176];
    q5k_data[0..2].copy_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
    let activations = vec![1.0f32; 256];
    let result = fused_q5k_dot(&q5k_data, &activations);
    assert!(result.is_ok());
}

/// Test fused_q5k_dot invalid length
#[test]
fn test_fused_q5k_dot_invalid_length() {
    let q5k_data = vec![0u8; 100];
    let activations = vec![1.0f32; 256];
    let result = fused_q5k_dot(&q5k_data, &activations);
    assert!(result.is_err());
}

// =========================================================================
// Coverage Tests: fused_q6k_dot scalar
// =========================================================================

/// Test fused_q6k_dot dimension mismatch
#[test]
fn test_fused_q6k_dot_dim_mismatch() {
    let q6k_data = vec![0u8; 210];
    let activations = vec![1.0f32; 128]; // Should be 256
    let result = fused_q6k_dot(&q6k_data, &activations);
    assert!(result.is_err());
}

// =========================================================================
// Coverage Tests: Tiled matvec error handling
// =========================================================================

/// Test fused_q4k_tiled_matvec dimension mismatch
#[test]
fn test_fused_q4k_tiled_matvec_dim_mismatch() {
    let q4k_data = vec![0u8; 144]; // 1 super-block
    let activations = vec![1.0f32; 128]; // Should be 256
    let result = fused_q4k_tiled_matvec(&q4k_data, &activations, 256, 1, Some(64));
    assert!(result.is_err());
}

// =========================================================================
// Coverage Tests: Parallel matvec into variants
// =========================================================================

/// Test fused_q4k_parallel_matvec_into dimension validation
#[test]
fn test_fused_q4k_parallel_matvec_into_dim_error() {
    let q4k_data = vec![0u8; 144];
    let activations = vec![1.0f32; 128]; // Wrong size
    let mut output = vec![0.0f32; 1];
    let result = fused_q4k_parallel_matvec_into(&q4k_data, &activations, 256, 1, &mut output);
    assert!(result.is_err());
}

/// Test fused_q5k_parallel_matvec_into dimension validation
#[test]
fn test_fused_q5k_parallel_matvec_into_dim_error() {
    let q5k_data = vec![0u8; 176];
    let activations = vec![1.0f32; 128];
    let mut output = vec![0.0f32; 1];
    let result = fused_q5k_parallel_matvec_into(&q5k_data, &activations, 256, 1, &mut output);
    assert!(result.is_err());
}

/// Test fused_q6k_parallel_matvec_into dimension validation
#[test]
fn test_fused_q6k_parallel_matvec_into_dim_error() {
    let q6k_data = vec![0u8; 210];
    let activations = vec![1.0f32; 128];
    let mut output = vec![0.0f32; 1];
    let result = fused_q6k_parallel_matvec_into(&q6k_data, &activations, 256, 1, &mut output);
    assert!(result.is_err());
}

// =========================================================================
// Coverage Tests: InterleavedQ4K
// =========================================================================

/// Test InterleavedQ4K from_q4k invalid length
#[test]
fn test_interleaved_q4k_invalid_length() {
    let data = vec![0u8; 100]; // Not multiple of 144
    let result = InterleavedQ4K::from_q4k(&data);
    assert!(result.is_err());
}

/// Test InterleavedQ4K from_q4k success
#[test]
fn test_interleaved_q4k_success() {
    let mut data = vec![0u8; 144];
    // Set d = 1.0 (f16)
    data[0..2].copy_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
    let result = InterleavedQ4K::from_q4k(&data);
    assert!(result.is_ok());
    let interleaved = result.expect("quantization failed");
    assert_eq!(interleaved.num_values(), 256);
}

/// Test InterleavedQ4K dot product
#[test]
fn test_interleaved_q4k_dot() {
    let data = vec![0u8; 144];
    let interleaved = InterleavedQ4K::from_q4k(&data).expect("quantization failed");
    let activations = vec![1.0f32; 256];
    let result = interleaved.dot(&activations);
    assert!(result.is_ok());
}

include!("detect_simd.rs");
include!("fused_swiglu_quantize.rs");
include!("quantize.rs");
