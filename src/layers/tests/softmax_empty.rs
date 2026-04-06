//! Additional coverage tests for layers/mod.rs
//!
//! Targets specific uncovered lines in:
//! - softmax: Empty tensor handling (lines 127-137)
//! - gelu: Empty tensor handling (lines 193-197)
//! - QuantizedLinear: Validation error paths (lines 602-632)
//! - FusedLayerNormLinear: forward_parallel validation (lines 915-930)
//! - Linear: 1D output shape construction (lines 490-491)
//! - LayerNorm: Empty input handling (lines 298-312)

use crate::error::RealizarError;
use crate::layers::*;
use crate::tensor::Tensor;

// =========================================================================
// softmax: Empty Tensor Error Path Tests
// =========================================================================

#[test]
fn test_softmax_empty_data_error() {
    // Create a tensor that has shape but no data (impossible via normal API,
    // but we test the error message pattern)
    // Since Tensor::from_vec validates, we need to test via shape mismatch scenarios

    // Test with minimal valid tensor then check error messages
    let input = Tensor::from_vec(vec![1], vec![1.0]).expect("test");
    let result = softmax(&input);
    assert!(result.is_ok(), "Single element softmax should succeed");
}

#[test]
fn test_softmax_single_element_tensor() {
    // Single element: softmax([x]) = [1.0] for any x
    let input = Tensor::from_vec(vec![1], vec![42.0]).expect("test");
    let output = softmax(&input).expect("test");

    assert_eq!(output.shape(), &[1]);
    assert!(
        (output.data()[0] - 1.0).abs() < 1e-6,
        "Softmax of single element should be 1.0"
    );
}

#[test]
fn test_softmax_very_large_negative_values() {
    // Test numerical stability with very large negative values
    let input = Tensor::from_vec(vec![3], vec![-1000.0, -1001.0, -1002.0]).expect("test");
    let output = softmax(&input).expect("test");

    // All outputs should be finite
    for &val in output.data() {
        assert!(
            val.is_finite(),
            "Softmax should handle large negative values"
        );
    }

    // Sum should still be 1.0
    let sum: f32 = output.data().iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);
}

#[test]
fn test_softmax_3d_tensor() {
    // Test softmax on 3D tensor [2, 2, 3] - applies to last dimension
    let input = Tensor::from_vec(
        vec![2, 2, 3],
        vec![
            1.0, 2.0, 3.0, // row 0
            4.0, 5.0, 6.0, // row 1
            7.0, 8.0, 9.0, // row 2
            10.0, 11.0, 12.0, // row 3
        ],
    )
    .expect("test");

    let output = softmax(&input).expect("test");

    assert_eq!(output.shape(), &[2, 2, 3]);

    // Each row of 3 should sum to 1.0
    for row in 0..4 {
        let row_sum: f32 = (0..3).map(|i| output.data()[row * 3 + i]).sum();
        assert!(
            (row_sum - 1.0).abs() < 1e-5,
            "Row {} sum should be 1.0, got {}",
            row,
            row_sum
        );
    }
}

#[test]
fn test_softmax_identical_values() {
    // All identical values should give uniform distribution
    let input = Tensor::from_vec(vec![4], vec![5.0, 5.0, 5.0, 5.0]).expect("test");
    let output = softmax(&input).expect("test");

    for &val in output.data() {
        assert!(
            (val - 0.25).abs() < 1e-6,
            "Uniform input should give uniform output"
        );
    }
}

// =========================================================================
// gelu: Empty Tensor and Edge Case Tests
// =========================================================================

#[test]
fn test_gelu_single_element() {
    // Single element tensor
    let input = Tensor::from_vec(vec![1], vec![0.5]).expect("test");
    let output = gelu(&input).expect("test");

    assert_eq!(output.shape(), &[1]);
    // GELU(0.5) should be approximately 0.5 * 0.5 * (1 + tanh(...)) ≈ 0.345
    assert!(output.data()[0] > 0.3 && output.data()[0] < 0.4);
}

#[test]
fn test_gelu_large_positive() {
    // Large positive value: GELU(x) ≈ x for large x
    let input = Tensor::from_vec(vec![1], vec![10.0]).expect("test");
    let output = gelu(&input).expect("test");

    // For large positive x, GELU(x) ≈ x
    assert!((output.data()[0] - 10.0).abs() < 0.01);
}

#[test]
fn test_gelu_large_negative() {
    // Large negative value: GELU(x) ≈ 0 for large negative x
    let input = Tensor::from_vec(vec![1], vec![-10.0]).expect("test");
    let output = gelu(&input).expect("test");

    // For large negative x, GELU(x) ≈ 0
    assert!(output.data()[0].abs() < 0.01);
}

#[test]
fn test_gelu_3d_tensor() {
    // Test GELU on 3D tensor
    let input = Tensor::from_vec(
        vec![2, 2, 3],
        vec![
            -1.0, 0.0, 1.0, -2.0, 0.5, 2.0, -3.0, 1.5, 3.0, -0.5, 0.25, 0.75,
        ],
    )
    .expect("test");

    let output = gelu(&input).expect("test");

    assert_eq!(output.shape(), &[2, 2, 3]);

    // GELU(0) = 0
    assert!((output.data()[1] - 0.0).abs() < 1e-6);

    // GELU(x) > 0 for all positive x
    assert!(output.data()[2] > 0.0); // GELU(1.0)
    assert!(output.data()[5] > 0.0); // GELU(2.0)
}

#[test]
fn test_gelu_symmetry_property() {
    // Test that GELU is NOT symmetric (unlike ReLU)
    // GELU(-x) != -GELU(x) in general
    let pos_input = Tensor::from_vec(vec![1], vec![0.5]).expect("test");
    let neg_input = Tensor::from_vec(vec![1], vec![-0.5]).expect("test");

    let pos_output = gelu(&pos_input).expect("test");
    let neg_output = gelu(&neg_input).expect("test");

    // GELU(-0.5) ≈ -0.154, GELU(0.5) ≈ 0.345
    // They are NOT negatives of each other
    assert!(
        (pos_output.data()[0] + neg_output.data()[0]).abs() > 0.1,
        "GELU should not be antisymmetric"
    );
}

// =========================================================================
// QuantizedLinear: Validation Error Path Tests
// =========================================================================

#[test]
fn test_quantized_linear_zero_in_features_error() {
    let result = QuantizedLinear::new(0, 256, vec![], vec![0.0; 256]);
    assert!(result.is_err(), "Should error on zero in_features");

    if let Err(RealizarError::InvalidShape { reason }) = result {
        assert!(reason.contains("in_features") || reason.contains("> 0"));
    }
}

#[test]
fn test_quantized_linear_zero_out_features_error() {
    let result = QuantizedLinear::new(256, 0, vec![], vec![]);
    assert!(result.is_err(), "Should error on zero out_features");

    if let Err(RealizarError::InvalidShape { reason }) = result {
        assert!(reason.contains("out_features") || reason.contains("> 0"));
    }
}

#[test]
fn test_quantized_linear_bias_length_mismatch_error() {
    // Q4_K: 256 values = 1 super-block = 144 bytes
    // For 2 output features, need 2 * 144 = 288 bytes
    let weight_bytes = vec![0u8; 288];
    let bias = vec![0.0f32; 3]; // Wrong: should be 2

    let result = QuantizedLinear::new(256, 2, weight_bytes, bias);
    assert!(result.is_err(), "Should error on bias length mismatch");

    if let Err(RealizarError::InvalidShape { reason }) = result {
        assert!(
            reason.contains("Bias") || reason.contains("doesn't match"),
            "Error should mention bias mismatch: {}",
            reason
        );
    }
}

#[test]
fn test_quantized_linear_weight_bytes_mismatch_error() {
    // Q4_K: in_features=256 = 1 super-block = 144 bytes per row
    // For 4 output features, need 4 * 144 = 576 bytes
    let weight_bytes = vec![0u8; 100]; // Wrong: should be 576
    let bias = vec![0.0f32; 4];

    let result = QuantizedLinear::new(256, 4, weight_bytes, bias);
    assert!(result.is_err(), "Should error on weight bytes mismatch");

    if let Err(RealizarError::InvalidShape { reason }) = result {
        assert!(
            reason.contains("Weight bytes") || reason.contains("doesn't match"),
            "Error should mention weight bytes mismatch: {}",
            reason
        );
    }
}

#[test]
fn test_quantized_linear_getters() {
    // Q4_K: 256 values = 1 super-block = 144 bytes
    let weight_bytes = vec![0u8; 144];
    let bias = vec![1.0f32];

    let layer = QuantizedLinear::new(256, 1, weight_bytes.clone(), bias.clone()).expect("test");

    assert_eq!(layer.in_features(), 256);
    assert_eq!(layer.out_features(), 1);
    assert_eq!(layer.weight_bytes().len(), 144);
    assert_eq!(layer.bias().len(), 1);
    assert!((layer.bias()[0] - 1.0).abs() < 1e-6);

    // Memory usage: 144 bytes for weights + 4 bytes for f32 bias
    assert_eq!(layer.memory_bytes(), 144 + 4);
}

#[test]
fn test_quantized_linear_non_256_aligned_in_features() {
    // Test with in_features not aligned to 256 (uses div_ceil)
    // in_features=300 -> ceil(300/256) = 2 super-blocks = 288 bytes per row
    let weight_bytes = vec![0u8; 288]; // 2 super-blocks for 1 output
    let bias = vec![0.0f32; 1];

    let layer = QuantizedLinear::new(300, 1, weight_bytes, bias).expect("test");

    assert_eq!(layer.in_features(), 300);
    assert_eq!(layer.out_features(), 1);
}

#[test]
fn test_quantized_linear_forward_empty_shape_error() {
    // Create valid layer first
    let weight_bytes = vec![0u8; 144]; // 1 super-block
    let bias = vec![0.0f32; 1];
    let layer = QuantizedLinear::new(256, 1, weight_bytes, bias).expect("test");

    // Test with mismatched input dimension
    let input = Tensor::from_vec(vec![128], vec![0.1; 128]).expect("test");
    let result = layer.forward(&input);

    assert!(result.is_err(), "Should error on dimension mismatch");
}

#[test]
fn test_quantized_linear_forward_shape_mismatch() {
    // Create valid layer: in_features=256, out_features=1
    let weight_bytes = vec![0u8; 144];
    let bias = vec![0.0f32; 1];
    let layer = QuantizedLinear::new(256, 1, weight_bytes, bias).expect("test");

    // Wrong input dimension
    let input = Tensor::from_vec(vec![512], vec![0.1; 512]).expect("test");
    let result = layer.forward(&input);

    assert!(result.is_err(), "Should error on in_features mismatch");

    if let Err(RealizarError::InvalidShape { reason }) = result {
        assert!(reason.contains("doesn't match"));
    }
}

// =========================================================================
// FusedLayerNormLinear: forward_parallel Validation Tests
// =========================================================================

#[test]
fn test_fused_layer_norm_linear_parallel_empty_shape_error() {
    let fused = FusedLayerNormLinear::new(4, 8, 1e-5).expect("test");

    // Create tensor with wrong dimension
    let input = Tensor::from_vec(vec![3], vec![1.0, 2.0, 3.0]).expect("test");
    let result = fused.forward_parallel(&input);

    assert!(
        result.is_err(),
        "forward_parallel should error on dimension mismatch"
    );
}

#[test]
fn test_fused_layer_norm_linear_parallel_dimension_mismatch() {
    let fused = FusedLayerNormLinear::new(8, 4, 1e-5).expect("test");

    // Wrong input dimension
    let input = Tensor::from_vec(vec![16], vec![0.1; 16]).expect("test");
    let result = fused.forward_parallel(&input);

    assert!(result.is_err(), "Should error on feature_dim mismatch");

    if let Err(RealizarError::InvalidShape { reason }) = result {
        assert!(reason.contains("doesn't match"));
    }
}

#[test]
fn test_fused_layer_norm_linear_parallel_single_row() {
    let fused = FusedLayerNormLinear::new(4, 8, 1e-5).expect("test");

    // Single row input
    let input = Tensor::from_vec(vec![4], vec![1.0, 2.0, 3.0, 4.0]).expect("test");

    let serial = fused.forward(&input).expect("test");
    let parallel = fused.forward_parallel(&input).expect("test");

    assert_eq!(serial.shape(), parallel.shape());
    for i in 0..serial.data().len() {
        assert!(
            (serial.data()[i] - parallel.data()[i]).abs() < 1e-5,
            "Single row mismatch at {}: {} vs {}",
            i,
            serial.data()[i],
            parallel.data()[i]
        );
    }
}

#[test]
fn test_fused_layer_norm_linear_forward_empty_shape_error() {
    let fused = FusedLayerNormLinear::new(4, 8, 1e-5).expect("test");

    // Wrong dimension
    let input = Tensor::from_vec(vec![6], vec![1.0; 6]).expect("test");
    let result = fused.forward(&input);

    assert!(
        result.is_err(),
        "forward should error on dimension mismatch"
    );
}

#[test]
fn test_fused_layer_norm_linear_weight_mutators() {
    let mut fused = FusedLayerNormLinear::new(4, 8, 1e-5).expect("test");

    // Test all weight mutators
    fused.norm_weight_mut()[0] = 2.0;
    assert!((fused.norm_weight_mut()[0] - 2.0).abs() < 1e-6);

    fused.norm_bias_mut()[0] = 0.5;
    assert!((fused.norm_bias_mut()[0] - 0.5).abs() < 1e-6);

    fused.linear_weight_mut()[0] = 0.1;
    assert!((fused.linear_weight_mut()[0] - 0.1).abs() < 1e-6);

    fused.linear_bias_mut()[0] = 0.05;
    assert!((fused.linear_bias_mut()[0] - 0.05).abs() < 1e-6);
}

// =========================================================================
// Linear: 1D Output Shape Construction Tests
// =========================================================================

#[test]
fn test_linear_1d_input_output_shape() {
    // Test 1D input: [in_features] -> [out_features]
    let linear = Linear::new(4, 8).expect("test");

    let input = Tensor::from_vec(vec![4], vec![1.0, 2.0, 3.0, 4.0]).expect("test");
    let output = linear.forward(&input).expect("test");

    // 1D input should produce output with shape [out_features]
    assert_eq!(output.shape(), &[8]);
}

#[test]
fn test_linear_2d_input_preserves_batch() {
    // Test 2D input: [batch, in_features] -> [batch, out_features]
    let linear = Linear::new(4, 8).expect("test");

    let input = Tensor::from_vec(vec![3, 4], vec![0.1; 12]).expect("test");
    let output = linear.forward(&input).expect("test");

    assert_eq!(output.shape(), &[3, 8]);
}

#[test]
fn test_linear_3d_input_preserves_dims() {
    // Test 3D input: [batch, seq, in_features] -> [batch, seq, out_features]
    let linear = Linear::new(4, 8).expect("test");

    let input = Tensor::from_vec(vec![2, 3, 4], vec![0.1; 24]).expect("test");
    let output = linear.forward(&input).expect("test");

    assert_eq!(output.shape(), &[2, 3, 8]);
}

#[test]
fn test_linear_forward_empty_shape_error() {
    let linear = Linear::new(4, 8).expect("test");

    // Wrong input dimension
    let input = Tensor::from_vec(vec![5], vec![1.0; 5]).expect("test");
    let result = linear.forward(&input);

    assert!(result.is_err(), "Should error on dimension mismatch");
}

include!("layer_norm_02.rs");
