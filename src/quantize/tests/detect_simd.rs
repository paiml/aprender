
// =========================================================================
// Coverage Tests: detect_simd_backend
// =========================================================================

/// Test detect_simd_backend returns valid backend
#[test]
fn test_detect_simd_backend_returns_valid() {
    let backend = detect_simd_backend();
    // Backend should be one of the valid variants
    match backend {
        SimdBackend::Scalar | SimdBackend::Avx2 | SimdBackend::Sse2 | SimdBackend::Neon => {},
    }
}

// =========================================================================
// Coverage Tests: apply_rope_rotation_simd
// =========================================================================

/// Test apply_rope_rotation_simd basic
#[test]
fn test_apply_rope_rotation_simd_basic_coverage() {
    // x1, x2, cos, sin must all have same length
    let mut x1 = vec![1.0f32; 32];
    let mut x2 = vec![1.0f32; 32];
    let cos = vec![1.0f32; 32];
    let sin = vec![0.0f32; 32];
    apply_rope_rotation_simd(&mut x1, &mut x2, &cos, &sin);
    // With cos=1, sin=0, values should be unchanged
    for &v in &x1 {
        assert!((v - 1.0).abs() < 1e-5);
    }
}

/// Test apply_rope_rotation_simd with rotation
#[test]
fn test_apply_rope_rotation_simd_with_rotation() {
    let mut x1 = vec![1.0f32; 32];
    let mut x2 = vec![0.0f32; 32];
    let cos = vec![0.0f32; 32]; // cos(90 degrees)
    let sin = vec![1.0f32; 32]; // sin(90 degrees)
    apply_rope_rotation_simd(&mut x1, &mut x2, &cos, &sin);
    // x1_new = x1 * cos - x2 * sin = 1.0 * 0 - 0 * 1 = 0
    // x2_new = x1 * sin + x2 * cos = 1.0 * 1 + 0 * 0 = 1
    assert!(x1[0].abs() < 1e-4);
    assert!((x2[0] - 1.0).abs() < 1e-4);
}

// =========================================================================
// Coverage Tests: fused_swiglu_simd
// =========================================================================

/// Test fused_swiglu_simd various sizes
#[test]
fn test_fused_swiglu_simd_various_sizes() {
    for size in [16, 32, 64, 100] {
        let mut gate = vec![1.0f32; size];
        let up = vec![2.0f32; size];
        fused_swiglu_simd(&mut gate, &up);
        // Check all values are modified
        for &v in &gate {
            assert!(v.is_finite());
        }
    }
}

/// Test fused_swiglu_simd negative values
#[test]
fn test_fused_swiglu_simd_negative_values() {
    let mut gate = vec![-1.0f32; 32];
    let up = vec![2.0f32; 32];
    fused_swiglu_simd(&mut gate, &up);
    // Check SiLU(-1) * 2.0 is computed
    for &v in &gate {
        assert!(v.is_finite());
    }
}

// =========================================================================
// Coverage Tests: softmax_simd
// =========================================================================

/// Test softmax_simd with negative values
#[test]
fn test_softmax_simd_negative_values() {
    let mut x = vec![-1.0f32, -2.0, -3.0, -4.0];
    softmax_simd(&mut x);
    let sum: f32 = x.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);
}

/// Test softmax_simd with large values
#[test]
fn test_softmax_simd_large_values() {
    let mut x = vec![100.0f32, 200.0, 300.0, 400.0];
    softmax_simd(&mut x);
    let sum: f32 = x.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);
}

/// Test softmax_simd with mixed values
#[test]
fn test_softmax_simd_mixed_values() {
    let mut x = vec![-10.0f32, 0.0, 10.0, 20.0];
    softmax_simd(&mut x);
    let sum: f32 = x.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);
    // Largest input should have largest probability
    assert!(x[3] > x[2]);
    assert!(x[2] > x[1]);
    assert!(x[1] > x[0]);
}

// =========================================================================
// Coverage Tests: quantize_activations_q8_0
// =========================================================================

/// Test quantize_activations_q8_0 various sizes
#[test]
fn test_quantize_activations_q8_0_various_sizes() {
    for size in [32, 64, 128, 256] {
        let activations: Vec<f32> = (0..size).map(|i| i as f32 / 10.0).collect();
        let (scales, quants) = quantize_activations_q8_0(&activations);
        assert_eq!(scales.len(), size / 32);
        assert_eq!(quants.len(), size);
    }
}

/// Test quantize_activations_q8_0 negative values
#[test]
fn test_quantize_activations_q8_0_negative_values() {
    let activations: Vec<f32> = (0..32).map(|i| -i as f32 / 10.0).collect();
    let (scales, quants) = quantize_activations_q8_0(&activations);
    assert_eq!(scales.len(), 1);
    assert!(scales[0] > 0.0);
    // First value should be 0, others negative
    for &q in &quants[1..] {
        assert!(q <= 0);
    }
}

// =========================================================================
// Coverage Tests: fused_q4_0_q8_0_parallel_matvec
// =========================================================================

/// Test fused_q4_0_q8_0_parallel_matvec weight too small
#[test]
fn test_fused_q4_0_q8_0_parallel_matvec_weight_error() {
    let weight_data = vec![0u8; 10]; // Too small
    let activations = vec![1.0f32; 32];
    let result = fused_q4_0_q8_0_parallel_matvec(&weight_data, &activations, 32, 2);
    assert!(result.is_err());
}

/// Test fused_q4_0_q8_0_parallel_matvec_into weight error
#[test]
fn test_fused_q4_0_q8_0_parallel_matvec_into_weight_error() {
    let weight_data = vec![0u8; 10]; // Too small
    let activations = vec![1.0f32; 32];
    let mut output = vec![0.0f32; 2];
    let result = fused_q4_0_q8_0_parallel_matvec_into(&weight_data, &activations, 32, &mut output);
    assert!(result.is_err());
}

// =========================================================================
// Coverage Tests: fused_q8_0_q8_0_parallel_matvec
// =========================================================================

/// Test fused_q8_0_q8_0_parallel_matvec weight error
#[test]
fn test_fused_q8_0_q8_0_parallel_matvec_weight_error() {
    let weight_data = vec![0u8; 10]; // Too small
    let activations = vec![1.0f32; 32];
    let result = fused_q8_0_q8_0_parallel_matvec(&weight_data, &activations, 32, 2);
    assert!(result.is_err());
}

/// Test fused_q8_0_q8_0_parallel_matvec_into weight error
#[test]
fn test_fused_q8_0_q8_0_parallel_matvec_into_weight_error() {
    let weight_data = vec![0u8; 10]; // Too small
    let activations = vec![1.0f32; 32];
    let mut output = vec![0.0f32; 2];
    let result =
        fused_q8_0_q8_0_parallel_matvec_into(&weight_data, &activations, 32, 2, &mut output);
    assert!(result.is_err());
}

// =========================================================================
// Coverage Tests: dequantize parallel/simd variants
// =========================================================================

/// Test dequantize_q4_k_parallel invalid length
#[test]
fn test_dequantize_q4_k_parallel_invalid() {
    let data = vec![0u8; 100]; // Not multiple of 144
    let result = dequantize_q4_k_parallel(&data);
    assert!(result.is_err());
}

/// Test dequantize_q4_k_simd invalid length
#[test]
fn test_dequantize_q4_k_simd_invalid() {
    let data = vec![0u8; 100];
    let result = dequantize_q4_k_simd(&data);
    assert!(result.is_err());
}

/// Test dequantize_q8_0_parallel invalid length
#[test]
fn test_dequantize_q8_0_parallel_invalid() {
    let data = vec![0u8; 10]; // Not multiple of 34
    let result = dequantize_q8_0_parallel(&data);
    assert!(result.is_err());
}

/// Test dequantize_q8_0_simd invalid length
#[test]
fn test_dequantize_q8_0_simd_invalid() {
    let data = vec![0u8; 10];
    let result = dequantize_q8_0_simd(&data);
    assert!(result.is_err());
}

// =========================================================================
// Coverage Tests: Q8_0Block methods (cov suffix)
// =========================================================================

#[test]
fn test_q8_0_block_quantize_cov() {
    let values: [f32; 32] = [1.0; 32];
    let block = Q8_0Block::quantize(&values);
    assert!(block.scale > 0.0);
    assert_eq!(block.quants.len(), 32);
}

#[test]
fn test_q8_0_block_quantize_zeros_cov() {
    let values: [f32; 32] = [0.0; 32];
    let block = Q8_0Block::quantize(&values);
    // Should use minimal scale for near-zero blocks
    assert!(block.scale > 0.0);
    for q in &block.quants {
        assert_eq!(*q, 0);
    }
}

#[test]
fn test_q8_0_block_quantize_mixed_cov() {
    let mut values: [f32; 32] = [0.0; 32];
    for i in 0..32 {
        values[i] = (i as f32 - 16.0) * 0.1;
    }
    let block = Q8_0Block::quantize(&values);
    assert!(block.scale > 0.0);
}

#[test]
fn test_q8_0_block_dequantize_cov() {
    let values: [f32; 32] = [1.0; 32];
    let block = Q8_0Block::quantize(&values);
    let dequantized = block.dequantize();
    assert_eq!(dequantized.len(), 32);
    for v in &dequantized {
        assert!((*v - 1.0).abs() < 0.1); // Within quantization error
    }
}

#[test]
fn test_q8_0_block_quantization_error_cov() {
    let values: [f32; 32] = [1.0; 32];
    let block = Q8_0Block::quantize(&values);
    let error = block.quantization_error(&values);
    // Q8_0 should have small error for uniform values
    assert!(error < 0.1);
}

#[test]
fn test_q8_0_block_relative_error_cov() {
    let values: [f32; 32] = [1.0; 32];
    let block = Q8_0Block::quantize(&values);
    let rel_error = block.relative_error(&values);
    assert!(rel_error < 0.1);
}

#[test]
fn test_q8_0_block_relative_error_zeros_cov() {
    let values: [f32; 32] = [0.0; 32];
    let block = Q8_0Block::quantize(&values);
    let rel_error = block.relative_error(&values);
    assert_eq!(rel_error, 0.0);
}

// =========================================================================
// Coverage Tests: f16_to_f32 (cov suffix)
// =========================================================================

#[test]
fn test_f16_to_f32_zero_cov() {
    let result = f16_to_f32(0);
    assert_eq!(result, 0.0);
}

#[test]
fn test_f16_to_f32_one_cov() {
    // f16 1.0 = 0x3C00
    let result = f16_to_f32(0x3C00);
    assert!((result - 1.0).abs() < 1e-3);
}

#[test]
fn test_f16_to_f32_negative_cov() {
    // f16 -1.0 = 0xBC00
    let result = f16_to_f32(0xBC00);
    assert!((result + 1.0).abs() < 1e-3);
}

// =========================================================================
// Coverage Tests: dequantize valid paths (cov suffix)
// =========================================================================

#[test]
fn test_dequantize_q4_0_valid_cov() {
    // Q4_0: 18 bytes per block (2 for f16 scale + 16 for quants)
    let mut data = vec![0u8; 18];
    // Set scale to 1.0 (f16 = 0x3C00)
    data[0..2].copy_from_slice(&0x3C00u16.to_le_bytes());
    let result = dequantize_q4_0(&data);
    assert!(result.is_ok());
    let values = result.expect("test");
    assert_eq!(values.len(), 32);
}

#[test]
fn test_dequantize_q8_0_valid_cov() {
    // Q8_0: 34 bytes per block (2 for f16 scale + 32 for quants)
    let mut data = vec![0u8; 34];
    // Set scale to 1.0 as f16 (0x3C00)
    data[0..2].copy_from_slice(&0x3C00u16.to_le_bytes());
    let result = dequantize_q8_0(&data);
    assert!(result.is_ok());
    let values = result.expect("test");
    assert_eq!(values.len(), 32);
}

#[test]
fn test_dequantize_f16_valid_cov() {
    // f16: 2 bytes per value
    let mut data = vec![0u8; 4];
    data[0..2].copy_from_slice(&0x3C00u16.to_le_bytes()); // 1.0
    data[2..4].copy_from_slice(&0x4000u16.to_le_bytes()); // 2.0
    let result = dequantize_f16(&data);
    assert!(result.is_ok());
    let values = result.expect("test");
    assert_eq!(values.len(), 2);
}

#[test]
fn test_dequantize_f16_invalid_odd_length_cov() {
    let data = vec![0u8; 3]; // Not multiple of 2
    let result = dequantize_f16(&data);
    assert!(result.is_err());
}

// =========================================================================
// Coverage Tests: quantize_to_q8_blocks (cov suffix)
// =========================================================================

#[test]
fn test_quantize_to_q8_blocks_valid_cov() {
    let values = vec![1.0f32; 64]; // 2 blocks
    let result = quantize_to_q8_blocks(&values);
    assert!(result.is_ok());
    let blocks = result.expect("test");
    assert_eq!(blocks.len(), 2);
}

#[test]
fn test_quantize_to_q8_blocks_invalid_length_cov() {
    let values = vec![1.0f32; 33]; // Not multiple of 32
    let result = quantize_to_q8_blocks(&values);
    assert!(result.is_err());
}

#[test]
fn test_dequantize_q8_blocks_cov() {
    let values = vec![1.0f32; 32];
    let blocks = quantize_to_q8_blocks(&values).expect("test");
    let dequantized = dequantize_q8_blocks(&blocks);
    assert_eq!(dequantized.len(), 32);
}

// =========================================================================
// Coverage Tests: softmax_simd (cov suffix)
// =========================================================================

#[test]
fn test_softmax_simd_basic_cov() {
    let mut x = vec![1.0f32, 2.0, 3.0];
    softmax_simd(&mut x);
    // Sum should be ~1
    let sum: f32 = x.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);
    // x[2] should be largest
    assert!(x[2] > x[1]);
    assert!(x[1] > x[0]);
}

#[test]
fn test_softmax_simd_uniform_cov() {
    let mut x = vec![1.0f32; 4];
    softmax_simd(&mut x);
    // All values should be equal (0.25)
    for v in &x {
        assert!((*v - 0.25).abs() < 1e-5);
    }
}

#[test]
fn test_softmax_simd_large_values_cov() {
    let mut x = vec![100.0f32, 200.0, 300.0];
    softmax_simd(&mut x);
    // Should not overflow
    for v in &x {
        assert!(v.is_finite());
    }
    assert!((x.iter().sum::<f32>() - 1.0).abs() < 1e-5);
}

// =========================================================================
// Coverage Tests: fused_swiglu_simd (cov suffix)
// =========================================================================

#[test]
fn test_fused_swiglu_simd_basic_cov() {
    let mut gate = vec![0.0f32, 1.0, 2.0, -1.0];
    let up = vec![1.0f32, 1.0, 1.0, 1.0];
    fused_swiglu_simd(&mut gate, &up);
    // gate[0] = 0 * sigmoid(0) * 1 = 0 * 0.5 * 1 = 0
    assert!(gate[0].abs() < 0.01);
    // gate[1] = 1 * sigmoid(1) * 1 ≈ 0.731
    assert!((gate[1] - 0.731).abs() < 0.05);
}
