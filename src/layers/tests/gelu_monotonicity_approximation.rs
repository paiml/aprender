
#[test]
fn test_gelu_monotonicity_positive_range() {
    // GELU should be monotonically increasing for x > 0
    let values: Vec<f32> = (0..10).map(|i| i as f32 * 0.5).collect();
    let input = Tensor::from_vec(vec![10], values).expect("input");
    let output = gelu(&input).expect("gelu");

    for i in 1..10 {
        assert!(
            output.data()[i] >= output.data()[i - 1],
            "GELU should be monotonic for positive x"
        );
    }
}

#[test]
fn test_gelu_approximation_accuracy() {
    // Test that GELU approximation is reasonably accurate
    let test_points = [0.0, 0.5, 1.0, 2.0, -0.5, -1.0];

    for &x in &test_points {
        let input = Tensor::from_vec(vec![1], vec![x]).expect("input");
        let output = gelu(&input).expect("gelu");

        // GELU(0) = 0
        if x == 0.0 {
            assert!(output.data()[0].abs() < 1e-6);
        }
        // GELU(x) > 0 for x > 0
        if x > 0.0 {
            assert!(output.data()[0] > 0.0);
        }
        // GELU should be finite
        assert!(output.data()[0].is_finite());
    }
}

// =============================================================================
// Error Path Coverage: Construction Errors
// =============================================================================

#[test]
fn test_linear_zero_in_features_error() {
    let result = Linear::new(0, 8);
    assert!(result.is_err());

    if let Err(RealizarError::InvalidShape { reason }) = result {
        assert!(reason.contains("> 0") || reason.contains("in_features"));
    }
}

#[test]
fn test_linear_zero_out_features_error() {
    let result = Linear::new(8, 0);
    assert!(result.is_err());

    if let Err(RealizarError::InvalidShape { reason }) = result {
        assert!(reason.contains("> 0") || reason.contains("out_features"));
    }
}

#[test]
fn test_layer_norm_zero_normalized_shape_error() {
    let result = LayerNorm::new(0, 1e-5);
    assert!(result.is_err());

    if let Err(RealizarError::InvalidShape { reason }) = result {
        assert!(reason.contains("> 0") || reason.contains("normalized_shape"));
    }
}

#[test]
fn test_fused_layer_norm_linear_zero_feature_dim_error() {
    let result = FusedLayerNormLinear::new(0, 8, 1e-5);
    assert!(result.is_err());

    if let Err(RealizarError::InvalidShape { reason }) = result {
        assert!(reason.contains("> 0") || reason.contains("feature_dim"));
    }
}

#[test]
fn test_fused_layer_norm_linear_zero_out_features_error() {
    let result = FusedLayerNormLinear::new(8, 0, 1e-5);
    assert!(result.is_err());

    if let Err(RealizarError::InvalidShape { reason }) = result {
        assert!(reason.contains("> 0") || reason.contains("out_features"));
    }
}

#[test]
fn test_ffn_zero_hidden_dim_error() {
    let result = FeedForward::new(0, 16);
    assert!(result.is_err());
}

#[test]
fn test_ffn_zero_intermediate_dim_error() {
    let result = FeedForward::new(4, 0);
    assert!(result.is_err());
}

// =============================================================================
// Batched Input Edge Cases
// =============================================================================

#[test]
fn test_linear_single_batch_dimension() {
    let linear = Linear::new(4, 8).expect("create");

    // Single item batch: [1, 4]
    let input = Tensor::from_vec(vec![1, 4], vec![1.0, 2.0, 3.0, 4.0]).expect("input");
    let output = linear.forward(&input).expect("forward");

    assert_eq!(output.shape(), &[1, 8]);
}

#[test]
fn test_layer_norm_single_batch() {
    let layer_norm = LayerNorm::new(4, 1e-5).expect("create");

    // Single item batch: [1, 4]
    let input = Tensor::from_vec(vec![1, 4], vec![1.0, 2.0, 3.0, 4.0]).expect("input");
    let output = layer_norm.forward(&input).expect("forward");

    assert_eq!(output.shape(), &[1, 4]);
}

#[test]
fn test_fused_layer_norm_linear_single_batch() {
    let fused = FusedLayerNormLinear::new(4, 8, 1e-5).expect("create");

    // Single item batch
    let input = Tensor::from_vec(vec![1, 4], vec![1.0, 2.0, 3.0, 4.0]).expect("input");

    let serial = fused.forward(&input).expect("serial");
    let parallel = fused.forward_parallel(&input).expect("parallel");

    assert_eq!(serial.shape(), &[1, 8]);
    assert_eq!(parallel.shape(), &[1, 8]);
}

// =============================================================================
// Property-Based Style Tests
// =============================================================================

#[test]
fn test_softmax_output_bounds() {
    // Softmax output should be in [0, 1] for all inputs
    let test_values: Vec<f32> = (-10..10).map(|i| i as f32 * 0.5).collect();
    let input = Tensor::from_vec(vec![20], test_values).expect("input");
    let output = softmax(&input).expect("softmax");

    for &val in output.data() {
        assert!(val >= 0.0, "Softmax output should be >= 0");
        assert!(val <= 1.0, "Softmax output should be <= 1");
    }
}

#[test]
fn test_layer_norm_output_mean_zero() {
    // LayerNorm with default gamma=1, beta=0 should produce mean ~= 0
    let layer_norm = LayerNorm::new(8, 1e-5).expect("create");

    // Various input distributions
    let test_inputs = [
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        vec![-4.0, -3.0, -2.0, -1.0, 1.0, 2.0, 3.0, 4.0],
        vec![100.0, 100.1, 100.2, 100.3, 100.4, 100.5, 100.6, 100.7],
    ];

    for values in test_inputs {
        let input = Tensor::from_vec(vec![8], values).expect("input");
        let output = layer_norm.forward(&input).expect("forward");

        let mean: f32 = output.data().iter().sum::<f32>() / 8.0;
        assert!(
            mean.abs() < 1e-4,
            "LayerNorm output mean should be ~0, got {}",
            mean
        );
    }
}

#[test]
fn test_linear_zero_input_produces_bias() {
    let mut linear = Linear::new(4, 3).expect("create");

    // Set all weights to 0, set specific bias
    for w in linear.weight_mut().iter_mut() {
        *w = 0.0;
    }
    linear.bias_mut()[0] = 1.0;
    linear.bias_mut()[1] = 2.0;
    linear.bias_mut()[2] = 3.0;

    // Zero input should produce bias
    let input = Tensor::from_vec(vec![4], vec![0.0, 0.0, 0.0, 0.0]).expect("input");
    let output = linear.forward(&input).expect("forward");

    assert!((output.data()[0] - 1.0).abs() < 1e-6);
    assert!((output.data()[1] - 2.0).abs() < 1e-6);
    assert!((output.data()[2] - 3.0).abs() < 1e-6);
}

// =============================================================================
// RoPE and Position Embedding Coverage
// =============================================================================

#[test]
fn test_rope_creation_various_dimensions() {
    for dim in [4, 8, 16, 32, 64, 128] {
        let rope = RoPE::new(dim, 512).expect("create");
        assert_eq!(rope.head_dim(), dim);
        assert_eq!(rope.max_seq_len(), 512);
    }
}

#[test]
fn test_scaled_rope_scaling_types() {
    // Test all scaling types can be created
    let linear = ScaledRoPE::new(64, 2048, RopeScalingType::Linear { scale: 2.0 }).expect("linear");
    assert_eq!(linear.head_dim(), 64);

    let dynamic =
        ScaledRoPE::new(64, 2048, RopeScalingType::Dynamic { scale: 2.0 }).expect("dynamic");
    assert_eq!(dynamic.head_dim(), 64);

    let ntk = ScaledRoPE::new(
        64,
        2048,
        RopeScalingType::NTKAware {
            scale: 2.0,
            alpha: 1.0,
        },
    )
    .expect("ntk");
    assert_eq!(ntk.head_dim(), 64);

    let yarn = ScaledRoPE::new(
        64,
        2048,
        RopeScalingType::YaRN {
            scale: 2.0,
            original_max_seq_len: 4096,
            attention_factor: 1.0,
            beta_fast: 32.0,
            beta_slow: 1.0,
        },
    )
    .expect("yarn");
    assert_eq!(yarn.head_dim(), 64);
}

#[test]
fn test_alibi_creation_various_heads() {
    for num_heads in [1, 2, 4, 8, 16, 32] {
        let alibi = ALiBi::new(num_heads).expect("create");
        assert_eq!(alibi.num_heads(), num_heads);
    }
}

#[test]
fn test_rope_forward_shape_preservation() {
    let rope = RoPE::new(8, 512).expect("create");

    // Test various input shapes
    let shapes = [vec![4, 8], vec![2, 4, 8], vec![1, 1, 8]];

    for shape in shapes {
        let size: usize = shape.iter().product();
        let input = Tensor::from_vec(shape.clone(), vec![0.1f32; size]).expect("input");
        let output = rope.forward(&input, 0).expect("forward");

        assert_eq!(output.shape(), &shape[..], "RoPE should preserve shape");
    }
}

#[test]
fn test_alibi_get_bias_various_seq_lengths() {
    let alibi = ALiBi::new(4).expect("create");

    for seq_len in [1, 4, 16, 64, 128] {
        let bias = alibi.get_bias(seq_len).expect("get_bias");

        // Bias shape should be [num_heads, seq_len, seq_len]
        assert_eq!(bias.shape(), &[4, seq_len, seq_len]);
    }
}

// =============================================================================
// Attention Submodule Additional Coverage
// =============================================================================

#[test]
fn test_attention_zero_head_dim_error() {
    let result = Attention::new(0);
    assert!(result.is_err(), "Should error on zero head_dim");
}

#[test]
fn test_attention_large_head_dim() {
    let attn = Attention::new(256).expect("create");

    let q = Tensor::from_vec(vec![2, 256], vec![0.1f32; 512]).expect("q");
    let k = Tensor::from_vec(vec![2, 256], vec![0.1f32; 512]).expect("k");
    let v = Tensor::from_vec(vec![2, 256], vec![0.1f32; 512]).expect("v");

    let output = attn.forward(&q, &k, &v).expect("forward");
    assert_eq!(output.shape(), &[2, 256]);
}

#[test]
fn test_sliding_window_attention_window_larger_than_seq() {
    let swa = SlidingWindowAttention::new(8, 100).expect("create");

    // seq_len=4 < window_size=100
    let q = Tensor::from_vec(vec![4, 8], vec![0.1f32; 32]).expect("q");
    let k = Tensor::from_vec(vec![4, 8], vec![0.1f32; 32]).expect("k");
    let v = Tensor::from_vec(vec![4, 8], vec![0.1f32; 32]).expect("v");

    // Should work - effectively full attention
    let output = swa.forward(&q, &k, &v).expect("forward");
    assert_eq!(output.shape(), &[4, 8]);
}

#[test]
fn test_multi_head_attention_weight_accessors() {
    let mut mha = MultiHeadAttention::mha(64, 8).expect("create");

    // Test all weight accessors are accessible and writable
    let w_q = mha.w_q_mut();
    assert!(!w_q.is_empty());
    w_q[0] = 1.0;

    let w_k = mha.w_k_mut();
    assert!(!w_k.is_empty());
    w_k[0] = 2.0;

    let w_v = mha.w_v_mut();
    assert!(!w_v.is_empty());
    w_v[0] = 3.0;

    let w_o = mha.w_o_mut();
    assert!(!w_o.is_empty());
    w_o[0] = 4.0;
}
