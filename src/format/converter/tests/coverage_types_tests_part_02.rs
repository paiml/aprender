
#[test]
fn test_compute_std_with_nan() {
    let data = vec![1.0, 2.0, f32::NAN, 4.0, 5.0];
    let mean = 3.0; // mean of valid values
    let std = compute_std(&data, mean, 4);
    // NaN is filtered out, so only 4 valid values
    assert!(std > 0.0);
    assert!(std.is_finite());
}

#[test]
fn test_compute_std_with_inf() {
    let data = vec![1.0, 2.0, f32::INFINITY, 4.0, 5.0];
    let mean = 3.0;
    let std = compute_std(&data, mean, 4);
    // Infinity is filtered out
    assert!(std.is_finite());
}

#[test]
fn test_compute_tensor_stats_basic() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let stats = compute_tensor_stats("test_tensor", &data);
    assert_eq!(stats.name, "test_tensor");
    assert_eq!(stats.count, 5);
    assert!((stats.mean - 3.0).abs() < 0.01);
    assert!((stats.min - 1.0).abs() < 0.01);
    assert!((stats.max - 5.0).abs() < 0.01);
    assert!(stats.std > 0.0);
    assert_eq!(stats.nan_count, 0);
    assert_eq!(stats.inf_count, 0);
}

#[test]
fn test_compute_tensor_stats_empty() {
    let data: Vec<f32> = vec![];
    let stats = compute_tensor_stats("empty_tensor", &data);
    assert_eq!(stats.name, "empty_tensor");
    assert_eq!(stats.count, 0);
    assert_eq!(stats.mean, 0.0);
    assert_eq!(stats.std, 0.0);
    assert_eq!(stats.min, 0.0);
    assert_eq!(stats.max, 0.0);
}

#[test]
fn test_compute_tensor_stats_with_nan() {
    let data = vec![1.0, f32::NAN, 3.0, f32::NAN, 5.0];
    let stats = compute_tensor_stats("nan_tensor", &data);
    assert_eq!(stats.count, 5);
    assert_eq!(stats.nan_count, 2);
    // Mean should be computed from valid values only
    assert!((stats.mean - 3.0).abs() < 0.01);
}

#[test]
fn test_compute_tensor_stats_with_inf() {
    let data = vec![1.0, f32::INFINITY, 3.0, f32::NEG_INFINITY, 5.0];
    let stats = compute_tensor_stats("inf_tensor", &data);
    assert_eq!(stats.count, 5);
    assert_eq!(stats.inf_count, 2);
}

#[test]
fn test_compute_tensor_stats_zeros() {
    let data = vec![0.0, 1.0, 0.0, 2.0, 0.0];
    let stats = compute_tensor_stats("sparse", &data);
    assert_eq!(stats.zero_count, 3);
}

#[test]
fn test_needs_transpose_2d_weight() {
    assert!(needs_transpose("layer.0.attn_q.weight", &[512, 512]));
    assert!(needs_transpose(
        "model.layers.0.self_attn.q_proj.weight",
        &[512, 512]
    ));
    assert!(needs_transpose("model.lm_head.weight", &[50257, 768]));
}

#[test]
fn test_needs_transpose_1d() {
    // 1D tensors should NOT be transposed
    assert!(!needs_transpose("layer.0.attn_q.bias", &[512]));
    assert!(!needs_transpose(
        "model.layers.0.self_attn.q_proj.weight",
        &[512]
    ));
}

#[test]
fn test_needs_transpose_3d() {
    // 3D tensors should NOT be transposed
    assert!(!needs_transpose("conv.weight", &[32, 64, 3]));
}

#[test]
fn test_needs_transpose_non_weight() {
    // Non-weight 2D tensors should NOT be transposed
    assert!(!needs_transpose("layer.0.attn_q.bias", &[512, 512]));
    assert!(!needs_transpose("embeddings", &[50257, 768]));
}

#[test]
fn test_needs_transpose_all_patterns() {
    // Test all weight patterns from the function
    let patterns = [
        "attn_output.weight",
        "attn_k.weight",
        "attn_q.weight",
        "attn_v.weight",
        "ffn_gate.weight",
        "ffn_up.weight",
        "ffn_down.weight",
        "output.weight",
        "lm_head.weight",
        "q_proj.weight",
        "k_proj.weight",
        "v_proj.weight",
        "o_proj.weight",
        "gate_proj.weight",
        "up_proj.weight",
        "down_proj.weight",
    ];
    for pattern in patterns {
        let name = format!("model.layers.0.{pattern}");
        assert!(
            needs_transpose(&name, &[512, 512]),
            "Pattern {pattern} should need transpose"
        );
    }
}

// ========================================================================
// TensorAccumulator Tests (ROSETTA-ML-001)
// ========================================================================

#[test]
fn test_tensor_accumulator_new() {
    let acc = TensorAccumulator::new();
    assert_eq!(acc.valid_count, 0);
    assert_eq!(acc.nan_count, 0);
    assert_eq!(acc.inf_count, 0);
    assert_eq!(acc.zero_count, 0);
}

#[test]
fn test_tensor_accumulator_basic_values() {
    let mut acc = TensorAccumulator::new();
    acc.accumulate(1.0);
    acc.accumulate(2.0);
    acc.accumulate(3.0);

    assert_eq!(acc.valid_count, 3);
    assert!((acc.mean() - 2.0).abs() < 0.001);
    assert!((acc.min - 1.0).abs() < 0.001);
    assert!((acc.max - 3.0).abs() < 0.001);
}

#[test]
fn test_tensor_accumulator_nan_tracking() {
    let mut acc = TensorAccumulator::new();
    acc.accumulate(1.0);
    acc.accumulate(f32::NAN);
    acc.accumulate(2.0);
    acc.accumulate(f32::NAN);
    acc.accumulate(3.0);

    assert_eq!(acc.valid_count, 3);
    assert_eq!(acc.nan_count, 2);
    assert!((acc.mean() - 2.0).abs() < 0.001);
}

#[test]
fn test_tensor_accumulator_inf_tracking() {
    let mut acc = TensorAccumulator::new();
    acc.accumulate(1.0);
    acc.accumulate(f32::INFINITY);
    acc.accumulate(2.0);
    acc.accumulate(f32::NEG_INFINITY);
    acc.accumulate(3.0);

    assert_eq!(acc.valid_count, 3);
    assert_eq!(acc.inf_count, 2);
}

#[test]
fn test_tensor_accumulator_zero_tracking() {
    let mut acc = TensorAccumulator::new();
    acc.accumulate(0.0);
    acc.accumulate(1.0);
    acc.accumulate(0.0);

    assert_eq!(acc.zero_count, 2);
    assert_eq!(acc.valid_count, 3);
}

#[test]
fn test_tensor_accumulator_mean_empty() {
    let acc = TensorAccumulator::new();
    assert_eq!(acc.mean(), 0.0);
}

#[test]
fn test_tensor_accumulator_safe_min_empty() {
    let acc = TensorAccumulator::new();
    assert_eq!(acc.safe_min(), 0.0);
}

#[test]
fn test_tensor_accumulator_safe_max_empty() {
    let acc = TensorAccumulator::new();
    assert_eq!(acc.safe_max(), 0.0);
}

#[test]
fn test_tensor_accumulator_negative_values() {
    let mut acc = TensorAccumulator::new();
    acc.accumulate(-5.0);
    acc.accumulate(-1.0);
    acc.accumulate(0.0);
    acc.accumulate(1.0);
    acc.accumulate(5.0);

    assert_eq!(acc.valid_count, 5);
    assert!((acc.safe_min() - (-5.0)).abs() < 0.001);
    assert!((acc.safe_max() - 5.0).abs() < 0.001);
    assert!((acc.mean() - 0.0).abs() < 0.001);
}

// ========================================================================
// Quantization Roundtrip Tests
// ========================================================================

#[test]
fn test_validate_single_tensor_no_issues() {
    // Use data that will pass validation: mean near 0, std reasonable
    let data = vec![-0.5, -0.25, 0.0, 0.25, 0.5];
    let mut validator = AprValidator::new();
    let mut errors = Vec::new();
    let mut options = ImportOptions::default();
    options.allow_no_config = true;
    // Use Basic validation which is less strict
    options.validation = ValidationConfig::Basic;

    validate_single_tensor("test_tensor", &data, &options, &mut validator, &mut errors);

    // Basic validation should not produce errors for reasonable data
    assert!(errors.is_empty(), "Unexpected errors: {:?}", errors);
}

#[test]
fn test_validate_single_tensor_with_nan() {
    let data = vec![0.1, f32::NAN, 0.3, 0.4];
    let mut validator = AprValidator::new();
    let mut errors = Vec::new();
    let mut options = ImportOptions::default();
    options.allow_no_config = true;
    options.validation = ValidationConfig::Strict;

    validate_single_tensor("test.weight", &data, &options, &mut validator, &mut errors);

    // Should have error for NaN
    assert!(errors.iter().any(|e| e.contains("NaN")));
}

#[test]
fn test_validate_single_tensor_none_validation() {
    let data = vec![0.1, f32::NAN, f32::INFINITY, 0.4];
    let mut validator = AprValidator::new();
    let mut errors = Vec::new();
    let mut options = ImportOptions::default();
    options.allow_no_config = true;
    options.validation = ValidationConfig::None;

    validate_single_tensor("test.weight", &data, &options, &mut validator, &mut errors);

    // ValidationConfig::None should not produce errors
    assert!(errors.is_empty());
}

#[test]
fn test_compression_variants() {
    let _zstd_default = Compression::ZstdDefault;
    let _zstd_max = Compression::ZstdMax;
    let _lz4 = Compression::Lz4;
    let _none = Compression::None;
}

// ========================================================================
// TensorExpectation Tests (ROSETTA-ML-001)
// ========================================================================

#[test]
fn test_tensor_expectation_for_tensor_rmsnorm() {
    // Test RMSNorm weight pattern detection
    let exp = TensorExpectation::for_tensor("model.layers.0.input_layernorm.weight");
    assert!(exp.is_some());
    let exp = exp.unwrap();
    assert!(exp.mean_range.0 < 1.0 && exp.mean_range.1 > 1.0);
}

#[test]
fn test_tensor_expectation_for_tensor_rmsnorm_post_attn() {
    let exp = TensorExpectation::for_tensor("model.layers.0.post_attention_layernorm.weight");
    assert!(exp.is_some());
}

#[test]
fn test_tensor_expectation_for_tensor_rms_norm() {
    let exp = TensorExpectation::for_tensor("model.layers.0.rms_norm.weight");
    assert!(exp.is_some());
}

#[test]
fn test_tensor_expectation_for_tensor_layer_norm_gamma() {
    let exp = TensorExpectation::for_tensor("bert.encoder.layer_norm.gamma");
    assert!(exp.is_some());
}

#[test]
fn test_tensor_expectation_for_tensor_layer_norm_beta() {
    let exp = TensorExpectation::for_tensor("bert.encoder.layer_norm.beta");
    assert!(exp.is_some());
}

#[test]
fn test_tensor_expectation_for_tensor_ln_weight() {
    let exp = TensorExpectation::for_tensor("transformer.ln_1.weight");
    assert!(exp.is_some());
}

#[test]
fn test_tensor_expectation_for_tensor_ln_bias() {
    let exp = TensorExpectation::for_tensor("transformer.ln_1.bias");
    assert!(exp.is_some());
}

#[test]
fn test_tensor_expectation_for_tensor_final_norm() {
    let exp = TensorExpectation::for_tensor("norm.weight");
    assert!(exp.is_some());
}

#[test]
fn test_tensor_expectation_for_tensor_embedding() {
    let exp = TensorExpectation::for_tensor("model.embed_tokens.weight");
    assert!(exp.is_some());
}

#[test]
fn test_tensor_expectation_for_tensor_linear_weight() {
    let exp = TensorExpectation::for_tensor("model.layers.0.fc1.weight");
    assert!(exp.is_some());
}

#[test]
fn test_tensor_expectation_check_passing() {
    let exp = TensorExpectation::EMBEDDING;
    let stats = TensorStats {
        name: "embed.weight".to_string(),
        count: 1000,
        mean: 0.001, // Near 0, within range
        std: 0.02,
        min: -0.1,
        max: 0.1,
        nan_count: 0,
        inf_count: 0,
        zero_count: 10,
    };
    assert!(exp.check(&stats).is_ok());
}

#[test]
fn test_tensor_expectation_check_mean_out_of_range() {
    let exp = TensorExpectation::EMBEDDING;
    let stats = TensorStats {
        name: "embed.weight".to_string(),
        count: 1000,
        mean: 5.0, // Way outside expected range
        std: 0.02,
        min: -0.1,
        max: 0.1,
        nan_count: 0,
        inf_count: 0,
        zero_count: 10,
    };
    assert!(exp.check(&stats).is_err());
}

#[test]
fn test_tensor_expectation_check_std_out_of_range() {
    // Use LAYER_NORM_WEIGHT which has std_range check
    let exp = TensorExpectation::LAYER_NORM_WEIGHT;
    let stats = TensorStats {
        name: "layer_norm.weight".to_string(),
        count: 1000,
        mean: 1.0,  // Within mean range for LayerNorm
        std: 100.0, // Way outside expected std range (0.0, 2.0)
        min: -0.1,
        max: 0.1,
        nan_count: 0,
        inf_count: 0,
        zero_count: 10,
    };
    assert!(exp.check(&stats).is_err());
}

#[test]
fn test_tensor_expectation_check_rmsnorm_passing() {
    let exp = TensorExpectation::RMSNORM_WEIGHT;
    let stats = TensorStats {
        name: "norm.weight".to_string(),
        count: 100,
        mean: 1.0, // Near 1.0 for RMSNorm
        std: 0.01,
        min: 0.99,
        max: 1.01,
        nan_count: 0,
        inf_count: 0,
        zero_count: 0,
    };
    assert!(exp.check(&stats).is_ok());
}
