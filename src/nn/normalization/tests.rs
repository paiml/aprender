//\! Normalization Tests - Extreme TDD
//\! PMAT-085: Extracted from mod.rs for PMAT file health compliance

pub(crate) use super::*;

#[test]
fn test_layer_norm_shape() {
    let norm = LayerNorm::new(&[256]);
    let x = Tensor::ones(&[32, 10, 256]);
    let y = norm.forward(&x);

    assert_eq!(y.shape(), x.shape());
}

#[test]
fn test_layer_norm_normalization() {
    let norm = LayerNorm::without_affine(&[4]);

    // Input: single sample with known values
    let x = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[1, 4]);
    let y = norm.forward(&x);

    // After normalization, mean should be ~0, std ~1
    let y_data = y.data();
    let mean: f32 = y_data.iter().sum::<f32>() / 4.0;
    let var: f32 = y_data.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / 4.0;

    assert!((mean).abs() < 1e-5, "Mean should be ~0, got {mean}");
    assert!((var - 1.0).abs() < 0.1, "Var should be ~1, got {var}");
}

#[test]
fn test_layer_norm_parameters() {
    let norm = LayerNorm::new(&[64]);
    let params = norm.parameters();

    assert_eq!(params.len(), 2); // weight and bias
    assert_eq!(params[0].numel(), 64); // weight
    assert_eq!(params[1].numel(), 64); // bias
}

#[test]
fn test_layer_norm_without_affine() {
    let norm = LayerNorm::without_affine(&[64]);
    let params = norm.parameters();

    assert!(params.is_empty());
}

#[test]
fn test_batch_norm_1d_shape() {
    let norm = BatchNorm1d::new(64);
    let x = Tensor::ones(&[32, 64]);
    let y = norm.forward(&x);

    assert_eq!(y.shape(), x.shape());
}

#[test]
fn test_batch_norm_1d_train_eval() {
    let mut norm = BatchNorm1d::new(64);

    assert!(norm.training());

    norm.eval();
    assert!(!norm.training());

    norm.train();
    assert!(norm.training());
}

#[test]
fn test_group_norm_shape() {
    let norm = GroupNorm::new(32, 256);
    let x = Tensor::ones(&[4, 256, 14, 14]);
    let y = norm.forward(&x);

    assert_eq!(y.shape(), x.shape());
}

#[test]
fn test_group_norm_2d_input() {
    // GroupNorm should also work with 2D input (no spatial dims)
    let norm = GroupNorm::new(8, 64);
    let x = Tensor::ones(&[4, 64]);
    let y = norm.forward(&x);

    assert_eq!(y.shape(), &[4, 64]);
}

#[test]
fn test_group_norm_parameters() {
    let norm = GroupNorm::new(32, 256);
    let params = norm.parameters();

    assert_eq!(params.len(), 2);
    assert_eq!(params[0].numel(), 256); // weight
    assert_eq!(params[1].numel(), 256); // bias
}

#[test]
fn test_group_norm_without_affine() {
    let norm = GroupNorm::without_affine(32, 256);
    let params = norm.parameters();

    assert!(params.is_empty());
}

#[test]
fn test_group_norm_normalization() {
    let norm = GroupNorm::without_affine(2, 4);

    // 2 groups of 2 channels each
    let x = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[1, 4]);
    let y = norm.forward(&x);

    // Each group should be normalized independently
    // Group 0: [1, 2] -> mean=1.5, std=0.5 -> [-1, 1] (approx)
    // Group 1: [3, 4] -> mean=3.5, std=0.5 -> [-1, 1] (approx)
    let y_data = y.data();

    // Check first group normalized
    let g0_mean = (y_data[0] + y_data[1]) / 2.0;
    assert!(
        g0_mean.abs() < 1e-5,
        "Group 0 mean should be ~0, got {g0_mean}"
    );

    // Check second group normalized
    let g1_mean = (y_data[2] + y_data[3]) / 2.0;
    assert!(
        g1_mean.abs() < 1e-5,
        "Group 1 mean should be ~0, got {g1_mean}"
    );
}

#[test]
fn test_instance_norm_shape() {
    let norm = InstanceNorm::new(64);
    let x = Tensor::ones(&[4, 64, 8, 8]);
    let y = norm.forward(&x);

    assert_eq!(y.shape(), x.shape());
}

#[test]
fn test_instance_norm_is_group_norm_with_num_groups_equal_channels() {
    // InstanceNorm is equivalent to GroupNorm with num_groups = num_channels
    let instance_norm = InstanceNorm::without_affine(4);
    let group_norm = GroupNorm::without_affine(4, 4);

    let x = Tensor::new(
        &[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ],
        &[1, 4, 2, 2],
    );

    let y_instance = instance_norm.forward(&x);
    let y_group = group_norm.forward(&x);

    // Should produce identical results
    for (a, b) in y_instance.data().iter().zip(y_group.data().iter()) {
        assert!(
            (a - b).abs() < 1e-5,
            "InstanceNorm and GroupNorm should match"
        );
    }
}

// ==========================================================================
// RMSNorm Tests
// ==========================================================================

#[test]
fn test_rms_norm_shape() {
    let norm = RMSNorm::new(&[256]);
    let x = Tensor::ones(&[32, 10, 256]);
    let y = norm.forward(&x);

    assert_eq!(y.shape(), x.shape());
}

#[test]
fn test_rms_norm_basic_normalization() {
    let norm = RMSNorm::without_affine(&[4]);

    // Input: single sample with known values
    let x = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[1, 4]);
    let y = norm.forward(&x);

    // RMS = sqrt(mean(x^2)) = sqrt((1+4+9+16)/4) = sqrt(7.5) ≈ 2.739
    // Normalized values: x / RMS
    let expected_rms = (7.5_f32 + 1e-6).sqrt();
    let y_data = y.data();

    for i in 0..4 {
        let expected = (i + 1) as f32 / expected_rms;
        assert!(
            (y_data[i] - expected).abs() < 1e-5,
            "Element {i}: expected {expected}, got {}",
            y_data[i]
        );
    }
}

#[test]
fn test_rms_norm_unit_vector_preserved() {
    // A unit vector should be nearly preserved (scaled by ~1)
    let norm = RMSNorm::without_affine(&[3]);

    // Unit vector: [1/sqrt(3), 1/sqrt(3), 1/sqrt(3)]
    let val = 1.0 / 3.0_f32.sqrt();
    let x = Tensor::new(&[val, val, val], &[1, 3]);
    let y = norm.forward(&x);

    // RMS of unit vector is 1/sqrt(3) ≈ 0.577
    // Dividing by RMS gives [1, 1, 1]
    let y_data = y.data();
    for &v in y_data {
        assert!(
            (v - 1.0).abs() < 1e-5,
            "Unit vector should normalize to 1s, got {v}"
        );
    }
}

#[test]
fn test_rms_norm_vs_layer_norm_no_centering() {
    // RMSNorm doesn't center, so mean of output is NOT zero in general
    let rms_norm = RMSNorm::without_affine(&[4]);
    let layer_norm = LayerNorm::without_affine(&[4]);

    let x = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[1, 4]);

    let y_rms = rms_norm.forward(&x);
    let y_layer = layer_norm.forward(&x);

    // LayerNorm output mean should be ~0
    let layer_mean: f32 = y_layer.data().iter().sum::<f32>() / 4.0;
    assert!(layer_mean.abs() < 1e-5, "LayerNorm should have zero mean");

    // RMSNorm output mean is NOT zero (no centering)
    let rms_mean: f32 = y_rms.data().iter().sum::<f32>() / 4.0;
    assert!(
        rms_mean > 0.1,
        "RMSNorm should NOT center, mean should be > 0, got {rms_mean}"
    );

    // Both should produce different outputs
    let diff: f32 = y_rms
        .data()
        .iter()
        .zip(y_layer.data().iter())
        .map(|(a, b)| (a - b).abs())
        .sum();
    assert!(
        diff > 0.1,
        "RMSNorm and LayerNorm should produce different outputs"
    );
}

#[test]
fn test_rms_norm_parameters() {
    let norm = RMSNorm::new(&[64]);
    let params = norm.parameters();

    // RMSNorm has only weight (no bias like LayerNorm)
    assert_eq!(params.len(), 1);
    assert_eq!(params[0].numel(), 64);
}

#[test]
fn test_rms_norm_without_affine() {
    let norm = RMSNorm::without_affine(&[64]);
    let params = norm.parameters();

    assert!(params.is_empty());
}

#[test]
fn test_rms_norm_with_custom_eps() {
    let norm = RMSNorm::with_eps(&[4], 1e-3);
    assert!((norm.eps() - 1e-3).abs() < 1e-8);
}

#[test]
fn test_rms_norm_batch_processing() {
    let norm = RMSNorm::without_affine(&[4]);

    // Two samples
    let x = Tensor::new(&[1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0], &[2, 4]);
    let y = norm.forward(&x);
    let y_data = y.data();

    // First sample: all 1s -> RMS = 1 -> output = 1s
    for i in 0..4 {
        assert!((y_data[i] - 1.0).abs() < 1e-5);
    }

    // Second sample: all 2s -> RMS = 2 -> output = 1s
    for i in 4..8 {
        assert!((y_data[i] - 1.0).abs() < 1e-5);
    }
}

#[test]
fn test_rms_norm_3d_input() {
    let norm = RMSNorm::new(&[256]);
    let x = Tensor::ones(&[4, 10, 256]); // [batch, seq, features]
    let y = norm.forward(&x);

    assert_eq!(y.shape(), &[4, 10, 256]);
}

#[test]
fn test_rms_norm_scaling_factor() {
    // RMSNorm scales input by 1/RMS, verify this is consistent
    let norm = RMSNorm::without_affine(&[4]);

    // If we scale input by 2, RMS doubles, output stays same
    let x1 = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[1, 4]);
    let x2 = Tensor::new(&[2.0, 4.0, 6.0, 8.0], &[1, 4]);

    let y1 = norm.forward(&x1);
    let y2 = norm.forward(&x2);

    // Outputs should be identical (RMSNorm is scale-invariant)
    for (a, b) in y1.data().iter().zip(y2.data().iter()) {
        assert!((a - b).abs() < 1e-5, "RMSNorm should be scale-invariant");
    }
}

#[test]
fn test_rms_norm_with_learnable_weight() {
    let norm = RMSNorm::new(&[4]);

    let x = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[1, 4]);
    let y = norm.forward(&x);

    // Default weight is 1.0, so output should be same as without_affine
    let norm_no_affine = RMSNorm::without_affine(&[4]);
    let y_no_affine = norm_no_affine.forward(&x);

    for (a, b) in y.data().iter().zip(y_no_affine.data().iter()) {
        assert!(
            (a - b).abs() < 1e-5,
            "Default weights should produce same result as no affine"
        );
    }
}

#[test]
fn test_rms_norm_numerical_stability() {
    // Test with very small values
    let norm = RMSNorm::without_affine(&[4]);

    let x = Tensor::new(&[1e-6, 1e-6, 1e-6, 1e-6], &[1, 4]);
    let y = norm.forward(&x);

    // Should not produce NaN or Inf
    for &v in y.data() {
        assert!(v.is_finite(), "Output should be finite");
    }
}

// ==========================================================================
// Additional LayerNorm Tests
// ==========================================================================

#[test]
fn test_layer_norm_with_eps() {
    let norm = LayerNorm::with_eps(&[64], 1e-3);
    let x = Tensor::ones(&[4, 64]);
    let y = norm.forward(&x);
    assert_eq!(y.shape(), &[4, 64]);
}

#[test]
fn test_layer_norm_normalized_shape_getter() {
    let norm = LayerNorm::new(&[32, 64]);
    assert_eq!(norm.normalized_shape(), &[32, 64]);
}

#[test]
fn test_layer_norm_parameters_mut() {
    let mut norm = LayerNorm::new(&[64]);
    let params = norm.parameters_mut();
    assert_eq!(params.len(), 2);
    // Can mutate parameters
    assert_eq!(params[0].numel(), 64);
}

#[test]
fn test_layer_norm_without_affine_parameters_mut() {
    let mut norm = LayerNorm::without_affine(&[64]);
    let params = norm.parameters_mut();
    assert!(params.is_empty());
}

#[test]
fn test_layer_norm_multi_dim_shape() {
    // Normalize over last 2 dimensions
    let norm = LayerNorm::new(&[8, 16]);
    let x = Tensor::ones(&[4, 8, 16]);
    let y = norm.forward(&x);
    assert_eq!(y.shape(), &[4, 8, 16]);
}

// ==========================================================================
// Additional BatchNorm1d Tests
// ==========================================================================

#[test]
fn test_batch_norm_1d_with_momentum() {
    let norm = BatchNorm1d::new(64).with_momentum(0.2);
    let x = Tensor::ones(&[32, 64]);
    let y = norm.forward(&x);
    assert_eq!(y.shape(), x.shape());
}

#[test]
fn test_batch_norm_1d_with_eps() {
    let norm = BatchNorm1d::new(64).with_eps(1e-3);
    let x = Tensor::ones(&[32, 64]);
    let y = norm.forward(&x);
    assert_eq!(y.shape(), x.shape());
}

#[test]
fn test_batch_norm_1d_parameters() {
    let norm = BatchNorm1d::new(32);
    let params = norm.parameters();
    assert_eq!(params.len(), 2);
    assert_eq!(params[0].numel(), 32); // weight
    assert_eq!(params[1].numel(), 32); // bias
}

#[test]
fn test_batch_norm_1d_parameters_mut() {
    let mut norm = BatchNorm1d::new(32);
    let params = norm.parameters_mut();
    assert_eq!(params.len(), 2);
    assert_eq!(params[0].numel(), 32);
}

#[path = "tests_part_02.rs"]
mod tests_part_02;
#[path = "tests_part_03.rs"]
mod tests_part_03;
