//! Additional coverage tests for layers/mod.rs - Part 10
//!
//! Targets remaining uncovered code paths in:
//! - QuantizedLinear: 1D output shape construction (lines 710-712)
//! - Linear: weight_mut, bias_mut accessors
//! - LayerNorm: normalized_shape(), eps() getters
//! - FusedLayerNormLinear: feature_dim(), out_features() getters
//! - FeedForward: fc1_mut, fc2_mut with actual weight loading
//! - Attention submodules: additional edge cases
//! - Position embeddings: additional RoPE/ALiBi edge cases

use crate::error::RealizarError;
use crate::layers::*;
use crate::tensor::Tensor;

// =============================================================================
// QuantizedLinear: 1D Input/Output Shape Handling (lines 710-712)
// =============================================================================

#[test]
fn test_quantized_linear_forward_produces_correct_output_size() {
    // Create valid Q4_K layer: 256 in_features, 2 out_features
    // 256 values = 1 super-block = 144 bytes per row
    // 2 output rows = 2 * 144 = 288 bytes
    let weight_bytes = vec![0u8; 288];
    let bias = vec![0.5f32, -0.5f32];

    let layer = QuantizedLinear::new(256, 2, weight_bytes, bias).expect("create layer");

    // 2D input: [batch=3, in_features=256]
    let input = Tensor::from_vec(vec![3, 256], vec![0.1f32; 768]).expect("input");
    let output = layer.forward(&input).expect("forward");

    // Output shape should be [3, 2]
    assert_eq!(output.shape(), &[3, 2]);
    assert_eq!(output.data().len(), 6);
}

#[test]
fn test_quantized_linear_multi_batch_forward() {
    // Q4_K: 256 in_features, 4 out_features
    // 256 values = 1 super-block = 144 bytes per row
    // 4 output rows = 4 * 144 = 576 bytes
    let weight_bytes = vec![0u8; 576];
    let bias = vec![0.1f32, 0.2f32, 0.3f32, 0.4f32];

    let layer = QuantizedLinear::new(256, 4, weight_bytes, bias).expect("create layer");

    // 3D input: [2, 3, 256] - batch of 2, seq of 3
    let input = Tensor::from_vec(vec![2, 3, 256], vec![0.05f32; 1536]).expect("input");
    let output = layer.forward(&input).expect("forward");

    // Output shape should be [2, 3, 4]
    assert_eq!(output.shape(), &[2, 3, 4]);
    assert_eq!(output.data().len(), 24);
}

// =============================================================================
// Linear: Weight and Bias Accessor Tests
// =============================================================================

#[test]
fn test_linear_weight_mut_accessor() {
    let mut linear = Linear::new(4, 8).expect("create layer");

    // Access and modify weights
    let weights = linear.weight_mut();
    assert_eq!(weights.len(), 4 * 8, "Weights should have in*out elements");

    // Set specific pattern
    for (i, w) in weights.iter_mut().enumerate() {
        *w = (i as f32) * 0.1;
    }

    // Verify modification persists
    assert!((linear.weight_mut()[0] - 0.0).abs() < 1e-6);
    assert!((linear.weight_mut()[1] - 0.1).abs() < 1e-6);
    assert!((linear.weight_mut()[31] - 3.1).abs() < 1e-6);
}

#[test]
fn test_linear_bias_mut_accessor() {
    let mut linear = Linear::new(4, 8).expect("create layer");

    // Access and modify bias
    let bias = linear.bias_mut();
    assert_eq!(bias.len(), 8, "Bias should have out_features elements");

    // Set specific values
    for (i, b) in bias.iter_mut().enumerate() {
        *b = (i as f32) * -0.5;
    }

    // Verify modification persists
    assert!((linear.bias_mut()[0] - 0.0).abs() < 1e-6);
    assert!((linear.bias_mut()[1] - (-0.5)).abs() < 1e-6);
    assert!((linear.bias_mut()[7] - (-3.5)).abs() < 1e-6);
}

#[test]
fn test_linear_forward_with_custom_weights() {
    let mut linear = Linear::new(2, 3).expect("create layer");

    // Set identity-like weights: weight[i][j] = 1 if i==j else 0
    // For 2x3: column-major (out_features = 3)
    // weight[0,0]=1, weight[1,1]=1, rest=0
    for w in linear.weight_mut().iter_mut() {
        *w = 0.0;
    }
    linear.weight_mut()[0] = 1.0; // in=0, out=0
    linear.weight_mut()[4] = 1.0; // in=1, out=1

    // Zero bias
    for b in linear.bias_mut().iter_mut() {
        *b = 0.0;
    }

    let input = Tensor::from_vec(vec![2], vec![5.0, 3.0]).expect("input");
    let output = linear.forward(&input).expect("forward");

    // output[0] = input[0]*1 + input[1]*0 = 5.0
    // output[1] = input[0]*0 + input[1]*1 = 3.0
    // output[2] = input[0]*0 + input[1]*0 = 0.0
    assert_eq!(output.shape(), &[3]);
    assert!((output.data()[0] - 5.0).abs() < 1e-5);
    assert!((output.data()[1] - 3.0).abs() < 1e-5);
    assert!((output.data()[2] - 0.0).abs() < 1e-5);
}

#[test]
fn test_linear_forward_with_bias() {
    let mut linear = Linear::new(2, 2).expect("create layer");

    // Set weights to identity
    for w in linear.weight_mut().iter_mut() {
        *w = 0.0;
    }
    linear.weight_mut()[0] = 1.0; // in=0, out=0
    linear.weight_mut()[3] = 1.0; // in=1, out=1

    // Set non-zero bias
    linear.bias_mut()[0] = 10.0;
    linear.bias_mut()[1] = 20.0;

    let input = Tensor::from_vec(vec![2], vec![1.0, 2.0]).expect("input");
    let output = linear.forward(&input).expect("forward");

    // output = input * W + bias
    // output[0] = 1.0*1 + 2.0*0 + 10.0 = 11.0
    // output[1] = 1.0*0 + 2.0*1 + 20.0 = 22.0
    assert!((output.data()[0] - 11.0).abs() < 1e-5);
    assert!((output.data()[1] - 22.0).abs() < 1e-5);
}

// =============================================================================
// LayerNorm: Getter Methods Coverage
// =============================================================================

#[test]
fn test_layer_norm_normalized_shape_getter() {
    for size in [1, 64, 128, 512, 4096] {
        let layer_norm = LayerNorm::new(size, 1e-5).expect("create");
        assert_eq!(
            layer_norm.normalized_shape(),
            size,
            "normalized_shape() should return construction parameter"
        );
    }
}

#[test]
fn test_layer_norm_eps_getter() {
    let test_cases = [1e-12, 1e-6, 1e-5, 1e-3, 0.1, 1.0];

    for eps in test_cases {
        let layer_norm = LayerNorm::new(64, eps).expect("create");
        assert!(
            (layer_norm.eps() - eps).abs() < 1e-12,
            "eps() should return construction parameter, expected {} got {}",
            eps,
            layer_norm.eps()
        );
    }
}

#[test]
fn test_layer_norm_zero_variance_handling() {
    let layer_norm = LayerNorm::new(4, 1e-5).expect("create");

    // All same values -> variance = 0, but eps prevents division by zero
    let input = Tensor::from_vec(vec![4], vec![5.0, 5.0, 5.0, 5.0]).expect("input");
    let output = layer_norm.forward(&input).expect("forward");

    // With zero variance and eps=1e-5:
    // normalized = (x - mean) / sqrt(0 + eps) = 0 / sqrt(eps) = 0
    // With gamma=1 and beta=0: output = 0
    for &val in output.data() {
        assert!(val.is_finite(), "Should handle zero variance");
        assert!(
            val.abs() < 1e-2,
            "Zero variance should produce near-zero output"
        );
    }
}

#[test]
fn test_layer_norm_negative_values() {
    let layer_norm = LayerNorm::new(4, 1e-5).expect("create");

    let input = Tensor::from_vec(vec![4], vec![-10.0, -5.0, 5.0, 10.0]).expect("input");
    let output = layer_norm.forward(&input).expect("forward");

    assert_eq!(output.shape(), &[4]);

    // Mean should be 0, so outputs should be symmetric around 0
    let mean: f32 = output.data().iter().sum::<f32>() / 4.0;
    assert!(mean.abs() < 1e-5, "LayerNorm output mean should be ~0");
}

// =============================================================================
// FusedLayerNormLinear: Getter Methods Coverage
// =============================================================================

#[test]
fn test_fused_layer_norm_linear_feature_dim_getter() {
    for dim in [4, 64, 256, 512] {
        let fused = FusedLayerNormLinear::new(dim, 128, 1e-5).expect("create");
        assert_eq!(
            fused.feature_dim(),
            dim,
            "feature_dim() should return construction parameter"
        );
    }
}

#[test]
fn test_fused_layer_norm_linear_out_features_getter() {
    for out in [4, 64, 256, 512] {
        let fused = FusedLayerNormLinear::new(128, out, 1e-5).expect("create");
        assert_eq!(
            fused.out_features(),
            out,
            "out_features() should return construction parameter"
        );
    }
}

#[test]
fn test_fused_layer_norm_linear_combined_operation() {
    let mut fused = FusedLayerNormLinear::new(4, 2, 1e-5).expect("create");

    // Set norm weights to 2.0 (scale by 2)
    for w in fused.norm_weight_mut().iter_mut() {
        *w = 2.0;
    }

    // Set linear weights (identity-like for first 2 dims)
    for w in fused.linear_weight_mut().iter_mut() {
        *w = 0.0;
    }
    fused.linear_weight_mut()[0] = 0.5; // in=0, out=0
    fused.linear_weight_mut()[3] = 0.5; // in=1, out=1

    let input = Tensor::from_vec(vec![4], vec![1.0, 2.0, 3.0, 4.0]).expect("input");
    let output = fused.forward(&input).expect("forward");

    assert_eq!(output.shape(), &[2]);
    // Output should be finite
    for &val in output.data() {
        assert!(val.is_finite());
    }
}

#[test]
fn test_fused_layer_norm_linear_parallel_matches_serial_different_sizes() {
    for (feature_dim, out_features, batch_size) in [(8, 4, 10), (16, 8, 50), (32, 16, 100)] {
        let fused = FusedLayerNormLinear::new(feature_dim, out_features, 1e-5).expect("create");

        let input = Tensor::from_vec(
            vec![batch_size, feature_dim],
            vec![0.1f32; batch_size * feature_dim],
        )
        .expect("input");

        let serial = fused.forward(&input).expect("serial");
        let parallel = fused.forward_parallel(&input).expect("parallel");

        assert_eq!(serial.shape(), parallel.shape());

        for i in 0..serial.data().len() {
            assert!(
                (serial.data()[i] - parallel.data()[i]).abs() < 1e-4,
                "Mismatch at {} for dims ({}, {}, {}): {} vs {}",
                i,
                feature_dim,
                out_features,
                batch_size,
                serial.data()[i],
                parallel.data()[i]
            );
        }
    }
}

// =============================================================================
// FeedForward: fc1_mut and fc2_mut Accessor Tests
// =============================================================================

#[test]
fn test_ffn_fc1_mut_accessor() {
    let mut ffn = FeedForward::new(4, 16).expect("create");

    let fc1 = ffn.fc1_mut();
    assert_eq!(fc1.in_features(), 4);
    assert_eq!(fc1.out_features(), 16);

    // Modify weights
    fc1.weight_mut()[0] = 42.0;
    assert!((ffn.fc1_mut().weight_mut()[0] - 42.0).abs() < 1e-6);
}

#[test]
fn test_ffn_fc2_mut_accessor() {
    let mut ffn = FeedForward::new(4, 16).expect("create");

    let fc2 = ffn.fc2_mut();
    assert_eq!(fc2.in_features(), 16);
    assert_eq!(fc2.out_features(), 4);

    // Modify weights
    fc2.weight_mut()[0] = 99.0;
    assert!((ffn.fc2_mut().weight_mut()[0] - 99.0).abs() < 1e-6);
}

#[test]
fn test_ffn_with_loaded_weights_integration() {
    let mut ffn = FeedForward::new(2, 4).expect("create");

    // Set fc1: 2->4 expansion
    // Simple pattern: each output is sum of inputs
    for w in ffn.fc1_mut().weight_mut().iter_mut() {
        *w = 1.0;
    }
    for b in ffn.fc1_mut().bias_mut().iter_mut() {
        *b = 0.0;
    }

    // Set fc2: 4->2 projection
    // Average of intermediate outputs
    for w in ffn.fc2_mut().weight_mut().iter_mut() {
        *w = 0.25;
    }
    for b in ffn.fc2_mut().bias_mut().iter_mut() {
        *b = 0.0;
    }

    let input = Tensor::from_vec(vec![2], vec![1.0, 2.0]).expect("input");
    let output = ffn.forward(&input).expect("forward");

    assert_eq!(output.shape(), &[2]);
    // Output should be finite and reasonable
    for &val in output.data() {
        assert!(val.is_finite());
    }
}

// =============================================================================
// softmax: Additional Edge Cases
// =============================================================================

#[test]
fn test_softmax_2d_multiple_rows() {
    // Test 2D tensor with multiple rows, verifying each row sums to 1
    let input = Tensor::from_vec(
        vec![3, 4],
        vec![
            1.0, 2.0, 3.0, 4.0, // row 0
            0.0, 0.0, 0.0, 0.0, // row 1 (uniform)
            -1.0, -2.0, -3.0, -4.0, // row 2 (negative)
        ],
    )
    .expect("input");

    let output = softmax(&input).expect("softmax");

    assert_eq!(output.shape(), &[3, 4]);

    // Each row should sum to 1.0
    for row in 0..3 {
        let row_sum: f32 = (0..4).map(|col| output.data()[row * 4 + col]).sum();
        assert!(
            (row_sum - 1.0).abs() < 1e-5,
            "Row {} should sum to 1.0, got {}",
            row,
            row_sum
        );
    }

    // Row 1 (all zeros) should be uniform
    for col in 0..4 {
        assert!(
            (output.data()[4 + col] - 0.25).abs() < 1e-5,
            "Uniform row element should be 0.25"
        );
    }
}

#[test]
fn test_softmax_numerical_stability_extreme_range() {
    // Mix of very small and very large values
    let input = Tensor::from_vec(vec![4], vec![-500.0, 0.0, 500.0, 1000.0]).expect("input");

    let output = softmax(&input).expect("softmax");

    // Should produce finite values
    for &val in output.data() {
        assert!(val.is_finite(), "Should handle extreme range");
    }

    // Sum should be 1.0
    let sum: f32 = output.data().iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);

    // Largest input should dominate
    assert!(output.data()[3] > 0.9, "Largest input should dominate");
}

// =============================================================================
// gelu: Additional Edge Cases
// =============================================================================

#[test]
fn test_gelu_preserves_shape_multi_dimensional() {
    // Test that GELU preserves shape for various dimensions
    let shapes = [vec![4], vec![2, 4], vec![2, 3, 4], vec![2, 3, 4, 5]];

    for shape in shapes {
        let size: usize = shape.iter().product();
        let input = Tensor::from_vec(shape.clone(), vec![0.5f32; size]).expect("input");
        let output = gelu(&input).expect("gelu");

        assert_eq!(output.shape(), &shape[..], "GELU should preserve shape");
    }
}

include!("gelu_monotonicity_approximation.rs");
