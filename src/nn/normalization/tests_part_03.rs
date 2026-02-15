
#[test]
fn test_batch_norm_single_sample_training() {
    // Single sample batch in training mode
    let norm = BatchNorm1d::new(4);
    let x = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[1, 4]);
    let y = norm.forward(&x);
    assert_eq!(y.shape(), &[1, 4]);
}

#[test]
fn test_group_norm_all_zeros() {
    // Test with all zeros input
    let norm = GroupNorm::new(2, 4);
    let x = Tensor::zeros(&[1, 4]);
    let y = norm.forward(&x);

    // Normalized zeros should still be zeros (0 - 0) / eps
    assert!(y.data().iter().all(|&v| v.abs() < 1e-3));
}

#[test]
fn test_rms_norm_all_zeros() {
    // Test with all zeros input - should be handled by eps
    let norm = RMSNorm::without_affine(&[4]);
    let x = Tensor::zeros(&[1, 4]);
    let y = norm.forward(&x);

    // 0 / sqrt(eps) should be finite
    assert!(y.data().iter().all(|&v| v.is_finite()));
}

#[test]
fn test_layer_norm_negative_values() {
    let norm = LayerNorm::without_affine(&[4]);
    let x = Tensor::new(&[-1.0, -2.0, -3.0, -4.0], &[1, 4]);
    let y = norm.forward(&x);

    // Mean should be centered
    let mean: f32 = y.data().iter().sum::<f32>() / 4.0;
    assert!(mean.abs() < 1e-5);
}

#[test]
fn test_batch_norm_large_values() {
    let norm = BatchNorm1d::new(4);
    let x = Tensor::new(
        &[
            1000.0, 2000.0, 3000.0, 4000.0, 1001.0, 2001.0, 3001.0, 4001.0,
        ],
        &[2, 4],
    );
    let y = norm.forward(&x);

    // Should normalize large values
    assert!(y.data().iter().all(|&v| v.is_finite()));
}

#[test]
fn test_group_norm_batch_size_one() {
    let norm = GroupNorm::new(4, 8);
    let x = Tensor::ones(&[1, 8]);
    let y = norm.forward(&x);
    assert_eq!(y.shape(), &[1, 8]);
}

#[test]
fn test_rms_norm_large_values() {
    let norm = RMSNorm::without_affine(&[4]);
    let x = Tensor::new(&[1000.0, 2000.0, 3000.0, 4000.0], &[1, 4]);
    let y = norm.forward(&x);

    // Should normalize large values without overflow
    assert!(y.data().iter().all(|&v| v.is_finite()));
}

#[test]
fn test_layer_norm_mixed_signs() {
    let norm = LayerNorm::without_affine(&[4]);
    let x = Tensor::new(&[-2.0, -1.0, 1.0, 2.0], &[1, 4]);
    let y = norm.forward(&x);

    // Mean should be 0 (input mean is 0)
    let mean: f32 = y.data().iter().sum::<f32>() / 4.0;
    assert!(mean.abs() < 1e-5);
}

#[test]
fn test_instance_norm_parameters_mut_without_affine() {
    let mut norm = InstanceNorm::without_affine(32);
    let params = norm.parameters_mut();
    assert!(params.is_empty());
}

#[test]
fn test_batch_norm_training_flag_persistence() {
    let mut norm = BatchNorm1d::new(64);

    // Start in training mode
    assert!(norm.training());

    // Switch to eval
    norm.eval();
    assert!(!norm.training());

    // Process some data
    let x = Tensor::ones(&[4, 64]);
    let _ = norm.forward(&x);

    // Should stay in eval mode
    assert!(!norm.training());

    // Switch back to train
    norm.train();
    assert!(norm.training());
}

#[test]
fn test_group_norm_3d_batch_processing() {
    // Test 3D input (no spatial dims) with multiple batches
    let norm = GroupNorm::new(2, 8);

    let data: Vec<f32> = (0..32).map(|x| x as f32).collect();
    let x = Tensor::new(&data, &[4, 8]);
    let y = norm.forward(&x);

    assert_eq!(y.shape(), &[4, 8]);
    assert!(y.data().iter().all(|&v| v.is_finite()));
}

#[test]
fn test_rms_norm_with_affine_vs_without() {
    let norm_affine = RMSNorm::new(&[4]);
    let norm_no_affine = RMSNorm::without_affine(&[4]);

    let x = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[1, 4]);

    // Default weights are 1.0, so should produce same results
    let y1 = norm_affine.forward(&x);
    let y2 = norm_no_affine.forward(&x);

    for (a, b) in y1.data().iter().zip(y2.data().iter()) {
        assert!((a - b).abs() < 1e-5);
    }
}

#[test]
fn test_layer_norm_4d_batch() {
    // Test 4D input
    let norm = LayerNorm::new(&[8]);
    let x = Tensor::ones(&[2, 4, 4, 8]);
    let y = norm.forward(&x);
    assert_eq!(y.shape(), &[2, 4, 4, 8]);
}
