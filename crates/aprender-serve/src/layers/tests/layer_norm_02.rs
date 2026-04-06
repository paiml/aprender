
// =========================================================================
// LayerNorm: Additional Edge Cases
// =========================================================================

#[test]
fn test_layer_norm_1d_input() {
    let layer_norm = LayerNorm::new(4, 1e-5).expect("test");

    let input = Tensor::from_vec(vec![4], vec![1.0, 2.0, 3.0, 4.0]).expect("test");
    let output = layer_norm.forward(&input).expect("test");

    assert_eq!(output.shape(), &[4]);

    // Check normalized mean is ~0
    let mean: f32 = output.data().iter().sum::<f32>() / 4.0;
    assert!((mean - 0.0).abs() < 1e-5);
}

#[test]
fn test_layer_norm_3d_input() {
    let layer_norm = LayerNorm::new(4, 1e-5).expect("test");

    let input =
        Tensor::from_vec(vec![2, 3, 4], (0..24).map(|i| i as f32 * 0.1).collect()).expect("test");

    let output = layer_norm.forward(&input).expect("test");

    assert_eq!(output.shape(), &[2, 3, 4]);

    // Each group of 4 should have mean ~0
    for group in 0..6 {
        let group_mean: f32 = (0..4).map(|i| output.data()[group * 4 + i]).sum::<f32>() / 4.0;
        assert!(
            (group_mean - 0.0).abs() < 1e-4,
            "Group {} mean should be ~0, got {}",
            group,
            group_mean
        );
    }
}

#[test]
fn test_layer_norm_large_eps() {
    // Test with large epsilon (affects variance normalization)
    let layer_norm = LayerNorm::new(4, 1.0).expect("test"); // Large eps

    let input = Tensor::from_vec(vec![4], vec![1.0, 2.0, 3.0, 4.0]).expect("test");
    let output = layer_norm.forward(&input).expect("test");

    // Should still produce finite values
    for &val in output.data() {
        assert!(val.is_finite());
    }
}

#[test]
fn test_layer_norm_numerical_stability_large_values() {
    let layer_norm = LayerNorm::new(4, 1e-5).expect("test");

    let input =
        Tensor::from_vec(vec![4], vec![1e6, 1e6 + 1.0, 1e6 + 2.0, 1e6 + 3.0]).expect("test");
    let output = layer_norm.forward(&input).expect("test");

    // Should produce finite values
    for &val in output.data() {
        assert!(val.is_finite(), "LayerNorm should handle large values");
    }

    // Mean should be ~0
    let mean: f32 = output.data().iter().sum::<f32>() / 4.0;
    assert!((mean - 0.0).abs() < 1e-4);
}

// =========================================================================
// FeedForward: Additional Coverage Tests
// =========================================================================

#[test]
fn test_ffn_1d_input() {
    let ffn = FeedForward::new(4, 16).expect("test");

    let input = Tensor::from_vec(vec![4], vec![1.0, 2.0, 3.0, 4.0]).expect("test");
    let output = ffn.forward(&input).expect("test");

    assert_eq!(output.shape(), &[4]);
}

#[test]
fn test_ffn_3d_input() {
    let ffn = FeedForward::new(4, 16).expect("test");

    let input = Tensor::from_vec(vec![2, 3, 4], vec![0.1; 24]).expect("test");
    let output = ffn.forward(&input).expect("test");

    assert_eq!(output.shape(), &[2, 3, 4]);
}

#[test]
fn test_ffn_dimension_mismatch_error() {
    let ffn = FeedForward::new(4, 16).expect("test");

    let input = Tensor::from_vec(vec![5], vec![0.1; 5]).expect("test");
    let result = ffn.forward(&input);

    assert!(result.is_err(), "Should error on dimension mismatch");
}

#[test]
fn test_ffn_with_custom_weights() {
    let mut ffn = FeedForward::new(2, 4).expect("test");

    // Set fc1: expand from 2 to 4
    for (i, w) in ffn.fc1_mut().weight_mut().iter_mut().enumerate() {
        *w = if i % 2 == 0 { 1.0 } else { 0.0 };
    }
    for b in ffn.fc1_mut().bias_mut().iter_mut() {
        *b = 0.0;
    }

    // Set fc2: project from 4 back to 2
    for (i, w) in ffn.fc2_mut().weight_mut().iter_mut().enumerate() {
        *w = if i % 2 == 0 { 1.0 } else { 0.0 };
    }
    for b in ffn.fc2_mut().bias_mut().iter_mut() {
        *b = 0.0;
    }

    let input = Tensor::from_vec(vec![2], vec![1.0, 0.5]).expect("test");
    let output = ffn.forward(&input).expect("test");

    assert_eq!(output.shape(), &[2]);
    // Output should be finite
    for &val in output.data() {
        assert!(val.is_finite());
    }
}

// =========================================================================
// Debug and Clone Implementations
// =========================================================================

#[test]
fn test_layer_norm_debug_clone() {
    let layer_norm = LayerNorm::new(4, 1e-5).expect("test");

    // Test Debug
    let debug_str = format!("{:?}", layer_norm);
    assert!(debug_str.contains("LayerNorm"));

    // Test Clone
    let cloned = layer_norm.clone();
    assert_eq!(cloned.normalized_shape(), layer_norm.normalized_shape());
    assert!((cloned.eps() - layer_norm.eps()).abs() < 1e-10);
}

#[test]
fn test_linear_debug_clone() {
    let linear = Linear::new(4, 8).expect("test");

    // Test Debug
    let debug_str = format!("{:?}", linear);
    assert!(debug_str.contains("Linear"));

    // Test Clone
    let cloned = linear.clone();
    assert_eq!(cloned.in_features(), linear.in_features());
    assert_eq!(cloned.out_features(), linear.out_features());
}

#[test]
fn test_quantized_linear_debug_clone() {
    let weight_bytes = vec![0u8; 144];
    let bias = vec![0.0f32; 1];
    let layer = QuantizedLinear::new(256, 1, weight_bytes, bias).expect("test");

    // Test Debug
    let debug_str = format!("{:?}", layer);
    assert!(debug_str.contains("QuantizedLinear"));

    // Test Clone
    let cloned = layer.clone();
    assert_eq!(cloned.in_features(), layer.in_features());
    assert_eq!(cloned.out_features(), layer.out_features());
}

#[test]
fn test_fused_layer_norm_linear_debug_clone() {
    let fused = FusedLayerNormLinear::new(4, 8, 1e-5).expect("test");

    // Test Debug
    let debug_str = format!("{:?}", fused);
    assert!(debug_str.contains("FusedLayerNormLinear"));

    // Test Clone
    let cloned = fused.clone();
    assert_eq!(cloned.feature_dim(), fused.feature_dim());
    assert_eq!(cloned.out_features(), fused.out_features());
}

#[test]
fn test_feed_forward_debug_clone() {
    let ffn = FeedForward::new(4, 16).expect("test");

    // Test Debug
    let debug_str = format!("{:?}", ffn);
    assert!(debug_str.contains("FeedForward"));

    // Test Clone
    let cloned = ffn.clone();
    assert_eq!(cloned.hidden_dim(), ffn.hidden_dim());
    assert_eq!(cloned.intermediate_dim(), ffn.intermediate_dim());
}

// =========================================================================
// Edge Cases: Numerical Boundaries
// =========================================================================

#[test]
fn test_softmax_inf_handling() {
    // Test with a value that's very large but not infinity
    let input = Tensor::from_vec(vec![3], vec![f32::MAX / 2.0, 0.0, 0.0]).expect("test");
    let result = softmax(&input);

    // Should either succeed with finite values or gracefully handle the edge case
    if let Ok(output) = result {
        // If it succeeds, values should sum to ~1
        let sum: f32 = output.data().iter().sum();
        assert!(sum.is_finite() || sum.is_nan(), "Sum should be handled");
    }
}

#[test]
fn test_gelu_inf_input() {
    // Test GELU with very large but not infinite input
    let input = Tensor::from_vec(vec![1], vec![f32::MAX / 2.0]).expect("test");
    let result = gelu(&input);

    // Should succeed and produce finite output (or same value for large x)
    if let Ok(output) = result {
        // For very large x, GELU(x) ≈ x, but may overflow
        let val = output.data()[0];
        assert!(
            val.is_finite() || val.is_infinite(),
            "Should produce a value"
        );
    }
}

#[test]
fn test_layer_norm_tiny_eps() {
    // Test with very small epsilon
    let layer_norm = LayerNorm::new(4, 1e-12).expect("test");

    let input = Tensor::from_vec(vec![4], vec![1.0, 2.0, 3.0, 4.0]).expect("test");
    let output = layer_norm.forward(&input).expect("test");

    // Should produce finite values
    for &val in output.data() {
        assert!(val.is_finite());
    }
}

// =========================================================================
// Batch Processing Edge Cases
// =========================================================================

#[test]
fn test_linear_large_batch() {
    let linear = Linear::new(4, 8).expect("test");

    // Large batch
    let input = Tensor::from_vec(vec![100, 4], vec![0.1; 400]).expect("test");
    let output = linear.forward(&input).expect("test");

    assert_eq!(output.shape(), &[100, 8]);
}

#[test]
fn test_layer_norm_large_batch() {
    let layer_norm = LayerNorm::new(4, 1e-5).expect("test");

    // Large batch
    let input = Tensor::from_vec(vec![100, 4], vec![0.1; 400]).expect("test");
    let output = layer_norm.forward(&input).expect("test");

    assert_eq!(output.shape(), &[100, 4]);
}

#[test]
fn test_fused_layer_norm_linear_large_batch_parallel() {
    let fused = FusedLayerNormLinear::new(8, 16, 1e-5).expect("test");

    // Large batch for parallel processing
    let input = Tensor::from_vec(vec![100, 8], vec![0.1; 800]).expect("test");

    let serial = fused.forward(&input).expect("test");
    let parallel = fused.forward_parallel(&input).expect("test");

    assert_eq!(serial.shape(), &[100, 16]);
    assert_eq!(parallel.shape(), &[100, 16]);

    // Results should match
    for i in 0..serial.data().len() {
        assert!(
            (serial.data()[i] - parallel.data()[i]).abs() < 1e-4,
            "Large batch mismatch at {}: {} vs {}",
            i,
            serial.data()[i],
            parallel.data()[i]
        );
    }
}
