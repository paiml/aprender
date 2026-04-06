use crate::layers::*;

#[test]
fn test_layer_norm_creation() {
    let layer_norm = LayerNorm::new(512, 1e-5).expect("test");
    assert_eq!(layer_norm.normalized_shape(), 512);
    assert!((layer_norm.eps() - 1e-5).abs() < 1e-10);
}

#[test]
fn test_layer_norm_zero_shape_error() {
    let result = LayerNorm::new(0, 1e-5);
    assert!(result.is_err());
}

#[test]
fn test_layer_norm_forward_simple() {
    // Simple test with known values
    let layer_norm = LayerNorm::new(3, 1e-5).expect("test");

    // Input: [1.0, 2.0, 3.0]
    // Mean: 2.0
    // Variance: ((1-2)^2 + (2-2)^2 + (3-2)^2) / 3 = 2/3
    // Std: sqrt(2/3 + 1e-5) ≈ 0.8165
    // Normalized: [(1-2)/0.8165, (2-2)/0.8165, (3-2)/0.8165]
    //           ≈ [-1.2247, 0.0, 1.2247]
    let input = Tensor::from_vec(vec![3], vec![1.0, 2.0, 3.0]).expect("test");
    let output = layer_norm.forward(&input).expect("test");

    let output_data = output.data();
    assert_eq!(output_data.len(), 3);

    // Check that mean is approximately 0
    let mean: f32 = output_data.iter().sum::<f32>() / 3.0;
    assert!((mean - 0.0).abs() < 1e-5);

    // Check that variance is approximately 1
    let variance: f32 = output_data
        .iter()
        .map(|&x| {
            let diff = x - mean;
            diff * diff
        })
        .sum::<f32>()
        / 3.0;
    assert!((variance - 1.0).abs() < 1e-4);
}

#[test]
fn test_layer_norm_forward_batched() {
    // Test with batch dimension
    let layer_norm = LayerNorm::new(2, 1e-5).expect("test");

    // Input: [[1.0, 3.0], [2.0, 4.0]]
    let input = Tensor::from_vec(vec![2, 2], vec![1.0, 3.0, 2.0, 4.0]).expect("test");
    let output = layer_norm.forward(&input).expect("test");

    assert_eq!(output.shape(), &[2, 2]);

    let output_data = output.data();
    assert_eq!(output_data.len(), 4);

    // First group [1.0, 3.0]: mean=2.0, normalized should have mean≈0, var≈1
    let group1_mean = f32::midpoint(output_data[0], output_data[1]);
    assert!((group1_mean - 0.0).abs() < 1e-5);

    // Second group [2.0, 4.0]: mean=3.0, normalized should have mean≈0, var≈1
    let group2_mean = f32::midpoint(output_data[2], output_data[3]);
    assert!((group2_mean - 0.0).abs() < 1e-5);
}

#[test]
fn test_layer_norm_empty_shape_handling() {
    // LayerNorm should handle validation properly
    // Since Tensor itself doesn't allow empty shapes, we test
    // that the normalized_shape validation works
    let result = LayerNorm::new(0, 1e-5);
    assert!(result.is_err());
}

#[test]
fn test_layer_norm_shape_mismatch_error() {
    let layer_norm = LayerNorm::new(3, 1e-5).expect("test");
    let input = Tensor::from_vec(vec![2], vec![1.0, 2.0]).expect("test"); // Wrong size
    let result = layer_norm.forward(&input);
    assert!(result.is_err());
}

#[test]
fn test_layer_norm_zero_variance() {
    // Test with constant input (zero variance)
    let layer_norm = LayerNorm::new(3, 1e-5).expect("test");
    let input = Tensor::from_vec(vec![3], vec![2.0, 2.0, 2.0]).expect("test");
    let output = layer_norm.forward(&input).expect("test");

    // With zero variance, normalized values should be close to 0
    // (since eps prevents division by zero)
    let output_data = output.data();
    for &val in output_data {
        assert!(val.abs() < 1e-2); // Should be near 0
    }
}

// Linear layer tests

#[test]
fn test_linear_creation() {
    let linear = Linear::new(128, 256).expect("test");
    assert_eq!(linear.in_features(), 128);
    assert_eq!(linear.out_features(), 256);
}

#[test]
fn test_linear_zero_dimensions_error() {
    let result = Linear::new(0, 256);
    assert!(result.is_err());

    let result = Linear::new(128, 0);
    assert!(result.is_err());
}

#[test]
fn test_linear_forward_simple() {
    // Simple test: 2 in_features, 3 out_features
    let mut linear = Linear::new(2, 3).expect("test");

    // Set identity-like weights for testing
    // weight[i][j] = 1.0 if i==j, 0.0 otherwise (extended for different dimensions)
    linear.weight_mut()[0] = 1.0; // weight[0][0]
    linear.weight_mut()[1] = 0.0; // weight[0][1]
    linear.weight_mut()[2] = 0.0; // weight[0][2]
    linear.weight_mut()[3] = 0.0; // weight[1][0]
    linear.weight_mut()[4] = 1.0; // weight[1][1]
    linear.weight_mut()[5] = 0.0; // weight[1][2]

    // Bias: all 0.5
    linear.bias_mut()[0] = 0.5;
    linear.bias_mut()[1] = 0.5;
    linear.bias_mut()[2] = 0.5;

    // Input: [2.0, 3.0]
    let input = Tensor::from_vec(vec![2], vec![2.0, 3.0]).expect("test");
    let output = linear.forward(&input).expect("test");

    assert_eq!(output.shape(), &[3]);
    let output_data = output.data();

    // Expected: [2.0*1.0 + 3.0*0.0 + 0.5, 2.0*0.0 + 3.0*1.0 + 0.5, 2.0*0.0 + 3.0*0.0 + 0.5]
    //         = [2.5, 3.5, 0.5]
    assert!((output_data[0] - 2.5).abs() < 1e-5);
    assert!((output_data[1] - 3.5).abs() < 1e-5);
    assert!((output_data[2] - 0.5).abs() < 1e-5);
}

#[test]
fn test_linear_forward_batched() {
    // Test with batch dimension: [2, 2] -> [2, 3]
    let mut linear = Linear::new(2, 3).expect("test");

    // Simple weights: all 1.0
    for i in 0..6 {
        linear.weight_mut()[i] = 1.0;
    }
    // Bias: all 0.0
    for i in 0..3 {
        linear.bias_mut()[i] = 0.0;
    }

    // Input: [[1.0, 2.0], [3.0, 4.0]]
    let input = Tensor::from_vec(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).expect("test");
    let output = linear.forward(&input).expect("test");

    assert_eq!(output.shape(), &[2, 3]);
    let output_data = output.data();

    // First row: [1.0, 2.0] * all-ones weight + zero bias = [3.0, 3.0, 3.0]
    assert!((output_data[0] - 3.0).abs() < 1e-5);
    assert!((output_data[1] - 3.0).abs() < 1e-5);
    assert!((output_data[2] - 3.0).abs() < 1e-5);

    // Second row: [3.0, 4.0] * all-ones weight + zero bias = [7.0, 7.0, 7.0]
    assert!((output_data[3] - 7.0).abs() < 1e-5);
    assert!((output_data[4] - 7.0).abs() < 1e-5);
    assert!((output_data[5] - 7.0).abs() < 1e-5);
}

#[test]
fn test_linear_shape_mismatch_error() {
    let linear = Linear::new(3, 2).expect("test");
    let input = Tensor::from_vec(vec![2], vec![1.0, 2.0]).expect("test"); // Wrong size (2 vs 3)
    let result = linear.forward(&input);
    assert!(result.is_err());
}

#[test]
fn test_linear_weight_bias_mut() {
    let mut linear = Linear::new(2, 3).expect("test");

    // Modify weights
    linear.weight_mut()[0] = 42.0;
    assert!((linear.weight_mut()[0] - 42.0).abs() < 1e-6);

    // Modify bias
    linear.bias_mut()[0] = 7.0;
    assert!((linear.bias_mut()[0] - 7.0).abs() < 1e-6);
}

// FusedLayerNormLinear tests

#[test]
fn test_fused_layer_norm_linear_creation() {
    let fused = FusedLayerNormLinear::new(4, 8, 1e-5).expect("test");
    assert_eq!(fused.feature_dim(), 4);
    assert_eq!(fused.out_features(), 8);
}

#[test]
fn test_fused_layer_norm_linear_zero_dims_error() {
    let result = FusedLayerNormLinear::new(0, 8, 1e-5);
    assert!(result.is_err());

    let result = FusedLayerNormLinear::new(4, 0, 1e-5);
    assert!(result.is_err());
}

#[test]
fn test_fused_layer_norm_linear_matches_separate() {
    // Test that fused implementation matches separate LayerNorm + Linear
    let feature_dim = 4;
    let out_features = 3;

    // Create fused layer
    let mut fused = FusedLayerNormLinear::new(feature_dim, out_features, 1e-5).expect("test");

    // Set weights
    for (i, weight) in fused.linear_weight_mut().iter_mut().enumerate() {
        #[allow(clippy::cast_precision_loss)]
        {
            *weight = (i as f32) * 0.1;
        }
    }
    for (i, bias) in fused.linear_bias_mut().iter_mut().enumerate() {
        #[allow(clippy::cast_precision_loss)]
        {
            *bias = (i as f32) * 0.05;
        }
    }

    // Create separate layers with same weights
    let layer_norm = LayerNorm::new(feature_dim, 1e-5).expect("test");
    let mut linear = Linear::new(feature_dim, out_features).expect("test");
    for (i, weight) in linear.weight_mut().iter_mut().enumerate() {
        #[allow(clippy::cast_precision_loss)]
        {
            *weight = (i as f32) * 0.1;
        }
    }
    for (i, bias) in linear.bias_mut().iter_mut().enumerate() {
        #[allow(clippy::cast_precision_loss)]
        {
            *bias = (i as f32) * 0.05;
        }
    }

    // Test input
    let input = Tensor::from_vec(vec![feature_dim], vec![1.0, 2.0, 3.0, 4.0]).expect("test");

    // Fused forward
    let fused_output = fused.forward(&input).expect("test");

    // Separate forward
    let norm_output = layer_norm.forward(&input).expect("test");
    let separate_output = linear.forward(&norm_output).expect("test");

    // Results should match
    assert_eq!(fused_output.shape(), separate_output.shape());
    for i in 0..fused_output.data().len() {
        assert!(
            (fused_output.data()[i] - separate_output.data()[i]).abs() < 1e-4,
            "Mismatch at {}: fused={} vs separate={}",
            i,
            fused_output.data()[i],
            separate_output.data()[i]
        );
    }
}

#[test]
fn test_fused_layer_norm_linear_batched() {
    let feature_dim = 4;
    let out_features = 2;

    let mut fused = FusedLayerNormLinear::new(feature_dim, out_features, 1e-5).expect("test");

    // Set simple weights
    for weight in fused.linear_weight_mut().iter_mut() {
        *weight = 1.0;
    }

    // Batched input [2, 4]
    let input = Tensor::from_vec(
        vec![2, feature_dim],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    )
    .expect("test");

    let output = fused.forward(&input).expect("test");
    assert_eq!(output.shape(), &[2, out_features]);
}

#[test]
fn test_fused_layer_norm_linear_parallel_matches_serial() {
    let feature_dim = 8;
    let out_features = 4;

    let mut fused = FusedLayerNormLinear::new(feature_dim, out_features, 1e-5).expect("test");

    // Set random-ish weights
    for (i, weight) in fused.linear_weight_mut().iter_mut().enumerate() {
        #[allow(clippy::cast_precision_loss)]
        {
            *weight = ((i * 7 + 3) % 11) as f32 * 0.1;
        }
    }
    for (i, bias) in fused.linear_bias_mut().iter_mut().enumerate() {
        #[allow(clippy::cast_precision_loss)]
        {
            *bias = ((i * 5 + 2) % 7) as f32 * 0.1;
        }
    }

    // Large batch
    let mut input_data = Vec::new();
    for i in 0..32 {
        for j in 0..feature_dim {
            #[allow(clippy::cast_precision_loss)]
            {
                input_data.push(((i * feature_dim + j) % 17) as f32 * 0.2);
            }
        }
    }
    let input = Tensor::from_vec(vec![32, feature_dim], input_data).expect("test");

    // Serial
    let serial_output = fused.forward(&input).expect("test");

    // Parallel
    let parallel_output = fused.forward_parallel(&input).expect("test");

    assert_eq!(serial_output.shape(), parallel_output.shape());
    for i in 0..serial_output.data().len() {
        assert!(
            (serial_output.data()[i] - parallel_output.data()[i]).abs() < 1e-5,
            "Mismatch at {}: serial={} vs parallel={}",
            i,
            serial_output.data()[i],
            parallel_output.data()[i]
        );
    }
}

#[test]
fn test_fused_layer_norm_linear_dimension_mismatch_error() {
    let fused = FusedLayerNormLinear::new(4, 8, 1e-5).expect("test");

    // Wrong input dimension
    let input = Tensor::from_vec(vec![3], vec![1.0, 2.0, 3.0]).expect("test");
    let result = fused.forward(&input);
    assert!(result.is_err());
}

// Softmax activation tests

#[test]
fn test_softmax_simple() {
    // Simple softmax: [0, 0, 0] -> [1/3, 1/3, 1/3]
    let input = Tensor::from_vec(vec![3], vec![0.0, 0.0, 0.0]).expect("test");
    let output = softmax(&input).expect("test");

    assert_eq!(output.shape(), &[3]);
    // All equal inputs -> equal probabilities
    assert!((output.data()[0] - 0.333_333).abs() < 1e-5);
    assert!((output.data()[1] - 0.333_333).abs() < 1e-5);
    assert!((output.data()[2] - 0.333_333).abs() < 1e-5);

    // Sum should be 1.0
    let sum: f32 = output.data().iter().sum();
    assert!((sum - 1.0).abs() < 1e-6);
}

#[test]
fn test_softmax_probabilities_sum_to_one() {
    let input = Tensor::from_vec(vec![4], vec![1.0, 2.0, 3.0, 4.0]).expect("test");
    let output = softmax(&input).expect("test");

    // Sum should be exactly 1.0
    let sum: f32 = output.data().iter().sum();
    assert!((sum - 1.0).abs() < 1e-6);

    // All values should be positive
    for &val in output.data() {
        assert!(val > 0.0);
        assert!(val < 1.0);
    }
}

#[test]
fn test_softmax_max_dominates() {
    // When one value is much larger, it should dominate
    let input = Tensor::from_vec(vec![3], vec![0.0, 0.0, 10.0]).expect("test");
    let output = softmax(&input).expect("test");

    // Last element should be close to 1.0
    assert!(output.data()[2] > 0.999);
    // Others should be very small
    assert!(output.data()[0] < 0.001);
    assert!(output.data()[1] < 0.001);
}

#[test]
fn test_softmax_batched() {
    // Batched: [[1, 2], [3, 4]]
    let input = Tensor::from_vec(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).expect("test");
    let output = softmax(&input).expect("test");

    assert_eq!(output.shape(), &[2, 2]);

    // Each row should sum to 1.0
    let row1_sum = output.data()[0] + output.data()[1];
    let row2_sum = output.data()[2] + output.data()[3];
    assert!((row1_sum - 1.0).abs() < 1e-6);
    assert!((row2_sum - 1.0).abs() < 1e-6);
}

#[test]
fn test_softmax_numerical_stability() {
    // Large values should not overflow
    let input = Tensor::from_vec(vec![3], vec![1000.0, 1001.0, 1002.0]).expect("test");
    let output = softmax(&input).expect("test");

    // Should not be NaN or Inf
    for &val in output.data() {
        assert!(val.is_finite());
    }

    // Sum should still be 1.0
    let sum: f32 = output.data().iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);
}

include!("softmax_preserves_gelu.rs");
include!("flash_attention.rs");
include!("scaled_rope.rs");
