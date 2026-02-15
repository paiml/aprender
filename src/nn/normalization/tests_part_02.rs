
#[test]
fn test_batch_norm_1d_3d_input_training() {
    // 3D input: [batch, features, length]
    let norm = BatchNorm1d::new(4);
    let x = Tensor::ones(&[2, 4, 8]);
    let y = norm.forward(&x);
    assert_eq!(y.shape(), &[2, 4, 8]);
}

#[test]
fn test_batch_norm_1d_3d_input_eval() {
    // 3D input in eval mode (uses running statistics)
    let mut norm = BatchNorm1d::new(4);
    norm.eval();
    let x = Tensor::ones(&[2, 4, 8]);
    let y = norm.forward(&x);
    assert_eq!(y.shape(), &[2, 4, 8]);
}

#[test]
fn test_batch_norm_1d_eval_mode() {
    // Test eval mode uses running statistics
    let mut norm = BatchNorm1d::new(4);
    norm.eval();
    assert!(!norm.training());

    let x = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 4]);
    let y = norm.forward(&x);
    assert_eq!(y.shape(), &[2, 4]);

    // Running mean defaults to 0, running var to 1
    // So output should be (x - 0) / 1 * gamma + beta = x * 1 + 0 = x
    for (i, &v) in y.data().iter().enumerate() {
        let expected = (i + 1) as f32;
        assert!((v - expected).abs() < 1e-4, "Expected {expected}, got {v}");
    }
}

#[test]
fn test_batch_norm_1d_debug() {
    let norm = BatchNorm1d::new(32);
    let debug_str = format!("{:?}", norm);
    assert!(debug_str.contains("BatchNorm1d"));
}

// ==========================================================================
// Additional GroupNorm Tests
// ==========================================================================

#[test]
fn test_group_norm_with_eps() {
    let norm = GroupNorm::with_eps(8, 64, 1e-3);
    let x = Tensor::ones(&[4, 64]);
    let y = norm.forward(&x);
    assert_eq!(y.shape(), &[4, 64]);
}

#[test]
fn test_group_norm_num_groups_getter() {
    let norm = GroupNorm::new(8, 64);
    assert_eq!(norm.num_groups(), 8);
}

#[test]
fn test_group_norm_num_channels_getter() {
    let norm = GroupNorm::new(8, 64);
    assert_eq!(norm.num_channels(), 64);
}

#[test]
fn test_group_norm_parameters_mut() {
    let mut norm = GroupNorm::new(8, 64);
    let params = norm.parameters_mut();
    assert_eq!(params.len(), 2);
    assert_eq!(params[0].numel(), 64);
}

#[test]
fn test_group_norm_without_affine_parameters_mut() {
    let mut norm = GroupNorm::without_affine(8, 64);
    let params = norm.parameters_mut();
    assert!(params.is_empty());
}

#[test]
fn test_group_norm_debug() {
    let norm = GroupNorm::new(8, 64);
    let debug_str = format!("{:?}", norm);
    assert!(debug_str.contains("GroupNorm"));
}

// ==========================================================================
// Additional RMSNorm Tests
// ==========================================================================

#[test]
fn test_rms_norm_normalized_shape_getter() {
    let norm = RMSNorm::new(&[128]);
    assert_eq!(norm.normalized_shape(), &[128]);
}

#[test]
fn test_rms_norm_set_weight() {
    let mut norm = RMSNorm::new(&[4]);
    let new_weight = Tensor::new(&[2.0, 2.0, 2.0, 2.0], &[4]);
    norm.set_weight(new_weight);

    let x = Tensor::new(&[1.0, 1.0, 1.0, 1.0], &[1, 4]);
    let y = norm.forward(&x);

    // RMS of all 1s = 1, normalized = 1, scaled by 2 = 2
    for &v in y.data() {
        assert!((v - 2.0).abs() < 1e-5, "Expected 2.0, got {v}");
    }
}

#[test]
fn test_rms_norm_weight_getter() {
    let norm = RMSNorm::new(&[64]);
    let weight = norm.weight();
    assert_eq!(weight.numel(), 64);
}

#[test]
fn test_rms_norm_placeholder() {
    let norm = RMSNorm::placeholder(&[256]);
    assert_eq!(norm.normalized_shape(), &[256]);
    // Placeholder has minimal weight (1 element)
    assert_eq!(norm.weight().numel(), 1);
}

#[test]
fn test_rms_norm_parameters_mut() {
    let mut norm = RMSNorm::new(&[64]);
    let params = norm.parameters_mut();
    assert_eq!(params.len(), 1);
    assert_eq!(params[0].numel(), 64);
}

#[test]
fn test_rms_norm_without_affine_parameters_mut() {
    let mut norm = RMSNorm::without_affine(&[64]);
    let params = norm.parameters_mut();
    assert!(params.is_empty());
}

#[test]
fn test_rms_norm_debug() {
    let norm = RMSNorm::new(&[64]);
    let debug_str = format!("{:?}", norm);
    assert!(debug_str.contains("RMSNorm"));
}

// ==========================================================================
// Additional InstanceNorm Tests
// ==========================================================================

#[test]
fn test_instance_norm_parameters() {
    let norm = InstanceNorm::new(32);
    let params = norm.parameters();
    assert_eq!(params.len(), 2);
    assert_eq!(params[0].numel(), 32);
}

#[test]
fn test_instance_norm_parameters_mut() {
    let mut norm = InstanceNorm::new(32);
    let params = norm.parameters_mut();
    assert_eq!(params.len(), 2);
    assert_eq!(params[0].numel(), 32);
}

#[test]
fn test_instance_norm_without_affine_parameters() {
    let norm = InstanceNorm::without_affine(32);
    let params = norm.parameters();
    assert!(params.is_empty());
}

#[test]
fn test_instance_norm_debug() {
    let norm = InstanceNorm::new(32);
    let debug_str = format!("{:?}", norm);
    assert!(debug_str.contains("InstanceNorm"));
}

#[test]
fn test_instance_norm_2d_input() {
    let norm = InstanceNorm::new(4);
    let x = Tensor::ones(&[2, 4]);
    let y = norm.forward(&x);
    assert_eq!(y.shape(), &[2, 4]);
}

// ==========================================================================
// Edge Case Tests
// ==========================================================================

#[test]
fn test_layer_norm_single_element() {
    let norm = LayerNorm::new(&[1]);
    let x = Tensor::new(&[5.0], &[1, 1]);
    let y = norm.forward(&x);
    // Single element: normalized = 0 (after mean subtraction), then scaled
    assert_eq!(y.shape(), &[1, 1]);
}

#[test]
fn test_group_norm_single_group() {
    // num_groups = 1 means normalize all channels together
    let norm = GroupNorm::new(1, 4);
    let x = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[1, 4]);
    let y = norm.forward(&x);
    assert_eq!(y.shape(), &[1, 4]);

    // All channels in one group = LayerNorm behavior
    let y_data = y.data();
    let mean: f32 = y_data.iter().sum::<f32>() / 4.0;
    assert!(mean.abs() < 1e-4, "Single group should center to 0");
}

#[test]
fn test_rms_norm_negative_values() {
    let norm = RMSNorm::without_affine(&[4]);
    let x = Tensor::new(&[-1.0, -2.0, 1.0, 2.0], &[1, 4]);
    let y = norm.forward(&x);

    // RMS ignores sign (squares values)
    let y_data = y.data();
    assert!(y_data.iter().all(|&v| v.is_finite()));
}

#[test]
fn test_batch_norm_varied_values() {
    let norm = BatchNorm1d::new(4);
    // Varied input to ensure proper batch statistics computation
    let x = Tensor::new(
        &[
            1.0, 2.0, 3.0, 4.0, // batch 0
            5.0, 6.0, 7.0, 8.0, // batch 1
        ],
        &[2, 4],
    );
    let y = norm.forward(&x);
    assert_eq!(y.shape(), &[2, 4]);

    // Each feature normalized across batch
    let y_data = y.data();
    // Feature 0: [1, 5] -> mean=3, std=2 -> [-1, 1]
    assert!((y_data[0] - y_data[4]).abs() > 0.5, "Should be different");
}

// ==========================================================================
// Additional Coverage Tests
// ==========================================================================

#[test]
fn test_layer_norm_debug() {
    let norm = LayerNorm::new(&[64]);
    let debug_str = format!("{:?}", norm);
    assert!(debug_str.contains("LayerNorm"));
}

#[test]
fn test_layer_norm_batch_independence() {
    // Each sample in batch is normalized independently
    let norm = LayerNorm::without_affine(&[4]);

    let x = Tensor::new(&[1.0, 1.0, 1.0, 1.0, 10.0, 10.0, 10.0, 10.0], &[2, 4]);
    let y = norm.forward(&x);
    let y_data = y.data();

    // Batch 0: all 1s -> variance = 0 -> all zeros after normalization (div by eps)
    // Batch 1: all 10s -> same behavior

    // For uniform values, normalized output should be 0 (mean subtraction)
    // when variance is nearly 0
    for &v in &y_data[0..4] {
        assert!(v.abs() < 1e-3, "Uniform input should normalize to ~0");
    }
}

#[test]
fn test_layer_norm_affine_transform() {
    // Test that affine parameters (weight, bias) are applied
    let norm = LayerNorm::new(&[4]);

    // With default weight=1 and bias=0, output should be same as without_affine
    let x = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[1, 4]);
    let y = norm.forward(&x);

    let norm_no_affine = LayerNorm::without_affine(&[4]);
    let y_no_affine = norm_no_affine.forward(&x);

    for (a, b) in y.data().iter().zip(y_no_affine.data().iter()) {
        assert!((a - b).abs() < 1e-5);
    }
}

#[test]
fn test_batch_norm_1d_chained_builders() {
    let norm = BatchNorm1d::new(32).with_momentum(0.2).with_eps(1e-4);
    let x = Tensor::ones(&[8, 32]);
    let y = norm.forward(&x);
    assert_eq!(y.shape(), &[8, 32]);
}

#[test]
fn test_batch_norm_1d_3d_varied_input_training() {
    // Test 3D input with varied values in training mode
    let norm = BatchNorm1d::new(2);
    let x = Tensor::new(
        &[
            // batch 0, feature 0, length=2
            1.0, 2.0, // batch 0, feature 1, length=2
            3.0, 4.0, // batch 1, feature 0, length=2
            5.0, 6.0, // batch 1, feature 1, length=2
            7.0, 8.0,
        ],
        &[2, 2, 2],
    );
    let y = norm.forward(&x);
    assert_eq!(y.shape(), &[2, 2, 2]);
}

#[test]
fn test_batch_norm_1d_3d_varied_input_eval() {
    // Test 3D input with varied values in eval mode
    let mut norm = BatchNorm1d::new(2);
    norm.eval();
    let x = Tensor::new(
        &[
            1.0, 2.0, 3.0, 4.0, // batch 0
            5.0, 6.0, 7.0, 8.0, // batch 1
        ],
        &[2, 2, 2],
    );
    let y = norm.forward(&x);
    assert_eq!(y.shape(), &[2, 2, 2]);
}

#[test]
fn test_group_norm_larger_spatial() {
    // Test with larger spatial dimensions
    let norm = GroupNorm::new(4, 8);
    let x = Tensor::ones(&[2, 8, 4, 4]);
    let y = norm.forward(&x);
    assert_eq!(y.shape(), &[2, 8, 4, 4]);
}

#[test]
fn test_group_norm_varying_values() {
    // Test with varying values to ensure proper group normalization
    let norm = GroupNorm::new(2, 4);
    let x = Tensor::new(
        &[
            1.0, 2.0, 3.0, 4.0, // sample 0
            5.0, 6.0, 7.0, 8.0, // sample 1
        ],
        &[2, 4],
    );
    let y = norm.forward(&x);

    // Check output is valid
    assert!(y.data().iter().all(|&v| v.is_finite()));
}

#[test]
fn test_group_norm_with_affine_output() {
    // Test that affine parameters affect output
    let norm_with_affine = GroupNorm::new(2, 4);
    let norm_without_affine = GroupNorm::without_affine(2, 4);

    let x = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[1, 4]);

    let y1 = norm_with_affine.forward(&x);
    let y2 = norm_without_affine.forward(&x);

    // With default weight=1, bias=0, outputs should be same
    for (a, b) in y1.data().iter().zip(y2.data().iter()) {
        assert!((a - b).abs() < 1e-5);
    }
}

#[test]
fn test_rms_norm_4d_input() {
    // Test with 4D input
    let norm = RMSNorm::new(&[8]);
    let x = Tensor::ones(&[2, 4, 4, 8]);
    let y = norm.forward(&x);
    assert_eq!(y.shape(), &[2, 4, 4, 8]);
}

#[test]
fn test_rms_norm_multi_dim_normalized_shape() {
    // Test with multi-dimensional normalized shape
    let norm = RMSNorm::new(&[4, 8]);
    let x = Tensor::ones(&[2, 4, 8]);
    let y = norm.forward(&x);
    assert_eq!(y.shape(), &[2, 4, 8]);
}

#[test]
fn test_rms_norm_eps_effect() {
    // Test that eps prevents division by zero
    let norm_small_eps = RMSNorm::with_eps(&[4], 1e-10);
    let norm_large_eps = RMSNorm::with_eps(&[4], 1.0);

    let x = Tensor::new(&[1e-8, 1e-8, 1e-8, 1e-8], &[1, 4]);

    let y_small = norm_small_eps.forward(&x);
    let y_large = norm_large_eps.forward(&x);

    // Both should be finite
    assert!(y_small.data().iter().all(|&v| v.is_finite()));
    assert!(y_large.data().iter().all(|&v| v.is_finite()));
}

#[test]
fn test_instance_norm_3d_varied_values() {
    let norm = InstanceNorm::new(4);
    let x = Tensor::new(
        &[
            1.0, 2.0, 3.0, 4.0, // batch 0, channels 0-3
            5.0, 6.0, 7.0, 8.0, // batch 1, channels 0-3
        ],
        &[2, 4],
    );
    let y = norm.forward(&x);
    assert_eq!(y.shape(), &[2, 4]);
}

#[test]
fn test_instance_norm_without_affine_forward() {
    let norm = InstanceNorm::without_affine(4);
    let x = Tensor::ones(&[2, 4, 8, 8]);
    let y = norm.forward(&x);
    assert_eq!(y.shape(), &[2, 4, 8, 8]);
}

#[test]
fn test_layer_norm_large_batch() {
    let norm = LayerNorm::new(&[64]);
    let x = Tensor::ones(&[128, 64]);
    let y = norm.forward(&x);
    assert_eq!(y.shape(), &[128, 64]);
}
