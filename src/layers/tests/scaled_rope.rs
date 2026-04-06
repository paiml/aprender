
#[test]
fn test_scaled_rope_linear_scaling() {
    // Linear scaling (Code Llama style)
    let scaling = RopeScalingType::Linear { scale: 4.0 };
    let scaled = ScaledRoPE::new(64, 10000.0, scaling).expect("test");

    assert!((scaled.context_length_multiplier() - 4.0).abs() < 1e-6);
    // Linear scaling doesn't change base frequency
    assert!((scaled.scaled_base() - 10000.0).abs() < 1e-6);
    assert!((scaled.mscale() - 1.0).abs() < 1e-6);
}

#[test]
fn test_scaled_rope_ntk_scaling() {
    // NTK-aware scaling
    let scaling = RopeScalingType::Ntk { scale: 4.0 };
    let scaled = ScaledRoPE::new(64, 10000.0, scaling).expect("test");

    assert!((scaled.context_length_multiplier() - 4.0).abs() < 1e-6);
    // NTK should increase base: base' = base * scale^(dim/(dim-2))
    // For dim=64: exponent = 64/62 ≈ 1.032
    // scaled_base = 10000 * 4^1.032 ≈ 41,376
    assert!(scaled.scaled_base() > 10000.0);
    assert!(scaled.scaled_base() > 40000.0);
    assert!((scaled.mscale() - 1.0).abs() < 1e-6);
}

#[test]
fn test_scaled_rope_dynamic_ntk() {
    // Dynamic NTK scaling
    let scaling = RopeScalingType::DynamicNtk {
        original_max_len: 2048,
        target_max_len: 8192,
    };
    let scaled = ScaledRoPE::new(64, 10000.0, scaling).expect("test");

    assert!((scaled.context_length_multiplier() - 4.0).abs() < 1e-6);
    // Should behave like NTK with scale = 4.0
    assert!(scaled.scaled_base() > 40000.0);
}

#[test]
fn test_scaled_rope_yarn() {
    // YaRN scaling
    let scaling = RopeScalingType::Yarn {
        original_max_len: 2048,
        target_max_len: 32768,
        attn_factor: 0.0, // Compute automatically
        beta_fast: 32.0,
        beta_slow: 1.0,
    };
    let scaled = ScaledRoPE::new(64, 10000.0, scaling).expect("test");

    // Context multiplier = 32768 / 2048 = 16
    assert!((scaled.context_length_multiplier() - 16.0).abs() < 1e-6);
    // YaRN should have mscale > 1.0 for large extensions
    assert!(scaled.mscale() > 1.0);
    // YaRN should have modified base
    assert!(scaled.scaled_base() > 10000.0);
}

#[test]
fn test_scaled_rope_yarn_custom_attn_factor() {
    // YaRN with custom attention factor
    let scaling = RopeScalingType::Yarn {
        original_max_len: 2048,
        target_max_len: 8192,
        attn_factor: 1.5, // Custom value
        beta_fast: 32.0,
        beta_slow: 1.0,
    };
    let scaled = ScaledRoPE::new(64, 10000.0, scaling).expect("test");

    // Should use custom attn_factor
    assert!((scaled.mscale() - 1.5).abs() < 1e-6);
}

#[test]
fn test_scaled_rope_forward_no_scaling() {
    let scaled = ScaledRoPE::new(4, 10000.0, RopeScalingType::None).expect("test");
    let input = Tensor::from_vec(vec![4], vec![1.0, 0.0, 0.0, 1.0]).expect("test");
    let output = scaled.forward(&input, 0).expect("test");

    // At position 0, rotation should be identity-like
    assert_eq!(output.shape(), &[4]);
}

#[test]
fn test_scaled_rope_forward_linear() {
    let scaling = RopeScalingType::Linear { scale: 2.0 };
    let scaled = ScaledRoPE::new(4, 10000.0, scaling).expect("test");
    let input = Tensor::from_vec(vec![4], vec![1.0, 0.0, 0.0, 1.0]).expect("test");

    // Position 10 with scale 2 should behave like position 5
    let output = scaled.forward(&input, 10).expect("test");
    assert_eq!(output.shape(), &[4]);
}

#[test]
fn test_scaled_rope_forward_ntk() {
    let scaling = RopeScalingType::Ntk { scale: 4.0 };
    let scaled = ScaledRoPE::new(4, 10000.0, scaling).expect("test");
    let input = Tensor::from_vec(vec![4], vec![1.0, 0.0, 0.0, 1.0]).expect("test");

    let output = scaled.forward(&input, 100).expect("test");
    assert_eq!(output.shape(), &[4]);
    // Output should preserve norm (rotation is norm-preserving)
    let norm: f32 = output.data().iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((norm - 2.0_f32.sqrt()).abs() < 0.1);
}

#[test]
fn test_scaled_rope_forward_yarn() {
    let scaling = RopeScalingType::Yarn {
        original_max_len: 2048,
        target_max_len: 8192,
        attn_factor: 1.0,
        beta_fast: 32.0,
        beta_slow: 1.0,
    };
    let scaled = ScaledRoPE::new(4, 10000.0, scaling).expect("test");
    let input = Tensor::from_vec(vec![4], vec![1.0, 0.0, 0.0, 1.0]).expect("test");

    let output = scaled.forward(&input, 5000).expect("test");
    assert_eq!(output.shape(), &[4]);
}

#[test]
fn test_scaled_rope_zero_dim_error() {
    let result = ScaledRoPE::new(0, 10000.0, RopeScalingType::None);
    assert!(result.is_err());
}

#[test]
fn test_scaled_rope_odd_dim_error() {
    let result = ScaledRoPE::new(63, 10000.0, RopeScalingType::None);
    assert!(result.is_err());
}

#[test]
fn test_scaled_rope_dimension_mismatch() {
    let scaled = ScaledRoPE::new(4, 10000.0, RopeScalingType::None).expect("test");
    let input = Tensor::from_vec(vec![8], vec![0.0; 8]).expect("test");

    let result = scaled.forward(&input, 0);
    assert!(result.is_err());
}

#[test]
fn test_rope_scaling_type_default() {
    let scaling = RopeScalingType::default();
    assert_eq!(scaling, RopeScalingType::None);
}

#[test]
fn test_scaled_rope_with_default_base() {
    let scaled = ScaledRoPE::with_default_base(64, RopeScalingType::None).expect("test");
    assert!((scaled.original_base() - 10000.0).abs() < 1e-6);
}

#[test]
fn test_scaled_rope_inv_freq_length() {
    let scaled = ScaledRoPE::new(128, 10000.0, RopeScalingType::None).expect("test");
    assert_eq!(scaled.inv_freq().len(), 64); // dim / 2
}

// ALiBi (Attention with Linear Biases) tests

#[test]
fn test_alibi_creation() {
    let alibi = ALiBi::new(8).expect("test");
    assert_eq!(alibi.num_heads(), 8);
    assert_eq!(alibi.slopes().len(), 8);
}

#[test]
fn test_alibi_zero_heads_error() {
    let result = ALiBi::new(0);
    assert!(result.is_err());
}

#[test]
fn test_alibi_slopes_power_of_2() {
    // For 8 heads (power of 2), slopes should follow: 2^(-8h/8) = 2^(-h)
    let alibi = ALiBi::new(8).expect("test");
    let slopes = alibi.slopes();

    // Expected slopes: 2^0, 2^-1, 2^-2, 2^-3, 2^-4, 2^-5, 2^-6, 2^-7
    assert!((slopes[0] - 1.0).abs() < 1e-6); // 2^0 = 1.0
    assert!((slopes[1] - 0.5).abs() < 1e-6); // 2^-1 = 0.5
    assert!((slopes[2] - 0.25).abs() < 1e-6); // 2^-2 = 0.25
    assert!((slopes[3] - 0.125).abs() < 1e-6); // 2^-3 = 0.125
}

#[test]
fn test_alibi_slopes_non_power_of_2() {
    // For 6 heads (not power of 2)
    let alibi = ALiBi::new(6).expect("test");
    let slopes = alibi.slopes();

    assert_eq!(slopes.len(), 6);

    // First 4 slopes follow 2^(-8h/4) = 2^(-2h)
    assert!((slopes[0] - 1.0).abs() < 1e-6); // 2^0
    assert!((slopes[1] - 0.25).abs() < 1e-6); // 2^-2
    assert!((slopes[2] - 0.0625).abs() < 1e-6); // 2^-4
    assert!((slopes[3] - 0.015_625).abs() < 1e-6); // 2^-6

    // Extra 2 slopes follow 2^(-4h/4) with step=2
    // slopes[4] = 2^(-1) = 0.5
    // slopes[5] = 2^(-3) = 0.125
    assert!((slopes[4] - 0.5).abs() < 1e-6);
    assert!((slopes[5] - 0.125).abs() < 1e-6);
}

#[test]
fn test_alibi_bias_shape() {
    let alibi = ALiBi::new(4).expect("test");
    let bias = alibi.get_bias(10).expect("test");

    // Shape should be [seq_len, seq_len, num_heads]
    assert_eq!(bias.shape(), &[10, 10, 4]);
}

#[test]
fn test_alibi_bias_zero_seq_len_error() {
    let alibi = ALiBi::new(4).expect("test");
    let result = alibi.get_bias(0);
    assert!(result.is_err());
}

#[test]
fn test_alibi_bias_diagonal_zero() {
    // Diagonal elements (same position) should be zero
    let alibi = ALiBi::new(4).expect("test");
    let bias = alibi.get_bias(5).expect("test");

    for i in 0..5 {
        for h in 0..4 {
            let idx = i * 5 * 4 + i * 4 + h; // [i, i, h]
            let value = bias.data()[idx];
            assert!(
                value.abs() < 1e-6,
                "Diagonal bias[{i}, {i}, {h}] should be 0, got {value}"
            );
        }
    }
}

#[test]
fn test_alibi_bias_symmetry() {
    // |i - j| = |j - i|, so bias[i,j,h] should equal bias[j,i,h]
    let alibi = ALiBi::new(2).expect("test");
    let bias = alibi.get_bias(4).expect("test");

    for i in 0..4 {
        for j in 0..4 {
            for h in 0..2 {
                let idx_ij = i * 4 * 2 + j * 2 + h;
                let idx_ji = j * 4 * 2 + i * 2 + h;
                let bias_ij = bias.data()[idx_ij];
                let bias_ji = bias.data()[idx_ji];
                assert!(
                    (bias_ij - bias_ji).abs() < 1e-6,
                    "Bias should be symmetric: [{i},{j},{h}]={bias_ij} vs [{j},{i},{h}]={bias_ji}"
                );
            }
        }
    }
}

#[test]
fn test_alibi_bias_computation() {
    // Test exact bias values
    let alibi = ALiBi::new(2).expect("test");
    let slopes = alibi.slopes();
    let bias = alibi.get_bias(3).expect("test");

    // For 2 heads: slopes = [1.0, 0.0625]
    // bias[0, 2, 0] = -slopes[0] * |0 - 2| = -1.0 * 2 = -2.0
    let idx = 2 * 2;
    assert!(
        (bias.data()[idx] - (-2.0)).abs() < 1e-6,
        "Expected -2.0, got {}",
        bias.data()[idx]
    );

    // bias[1, 2, 1] = -slopes[1] * |1 - 2| = -0.0625 * 1 = -0.0625
    let idx = 3 * 2 + 2 * 2 + 1;
    let expected = -slopes[1];
    assert!(
        (bias.data()[idx] - expected).abs() < 1e-6,
        "Expected {expected}, got {}",
        bias.data()[idx]
    );
}

#[test]
fn test_alibi_bias_negative() {
    // All bias values should be <= 0 (except diagonal which is 0)
    let alibi = ALiBi::new(4).expect("test");
    let bias = alibi.get_bias(10).expect("test");

    for &value in bias.data() {
        assert!(value <= 1e-6, "Bias should be non-positive, got {value}");
    }
}

#[test]
fn test_alibi_bias_distance_proportional() {
    // Bias should be proportional to distance
    let alibi = ALiBi::new(1).expect("test");
    let bias = alibi.get_bias(5).expect("test");

    // For head 0, slope is 1.0
    // bias[0, 1] = -1.0 * 1 = -1.0
    // bias[0, 2] = -1.0 * 2 = -2.0
    // bias[0, 3] = -1.0 * 3 = -3.0

    let bias_01 = bias.data()[1];
    let bias_02 = bias.data()[2];
    let bias_03 = bias.data()[3];

    assert!((bias_01 - (-1.0)).abs() < 1e-6);
    assert!((bias_02 - (-2.0)).abs() < 1e-6);
    assert!((bias_03 - (-3.0)).abs() < 1e-6);
}

#[test]
fn test_alibi_single_head() {
    let alibi = ALiBi::new(1).expect("test");
    assert_eq!(alibi.num_heads(), 1);
    assert_eq!(alibi.slopes().len(), 1);
    assert!((alibi.slopes()[0] - 1.0).abs() < 1e-6); // First slope is 2^0 = 1.0
}

#[test]
fn test_alibi_large_num_heads() {
    // Test with large number of heads (non-power of 2)
    let alibi = ALiBi::new(12).expect("test");
    assert_eq!(alibi.num_heads(), 12);
    assert_eq!(alibi.slopes().len(), 12);

    // All slopes should be positive
    for slope in alibi.slopes() {
        assert!(*slope > 0.0, "Slope should be positive, got {slope}");
    }

    // First head should have largest slope (1.0)
    assert!((alibi.slopes()[0] - 1.0).abs() < 1e-6);
}

#[test]
fn test_alibi_bias_long_sequence() {
    // Test with longer sequence
    let alibi = ALiBi::new(8).expect("test");
    let bias = alibi.get_bias(128).expect("test");

    assert_eq!(bias.shape(), &[128, 128, 8]);

    // Check that far positions have larger negative bias
    let near_bias = bias.data()[8]; // distance 1
    let far_bias = bias.data()[100 * 8]; // distance 100

    assert!(near_bias > far_bias); // near should be less negative
}

// KVCache tests
