use super::*;

// ========================================================================
// Additional TensorExpectation Coverage Tests
// ========================================================================

#[test]
fn test_tensor_expectation_input_layernorm() {
    let exp = TensorExpectation::for_tensor("model.layers.0.input_layernorm.weight");
    assert!(exp.is_some());
    let exp = exp.unwrap();
    assert_eq!(exp.description, "RMSNorm weight (gamma)");
}

#[test]
fn test_tensor_expectation_post_attention_layernorm() {
    let exp = TensorExpectation::for_tensor("model.layers.0.post_attention_layernorm.weight");
    assert!(exp.is_some());
    let exp = exp.unwrap();
    assert_eq!(exp.description, "RMSNorm weight (gamma)");
}

#[test]
fn test_tensor_expectation_rms_norm() {
    let exp = TensorExpectation::for_tensor("rms_norm.weight");
    assert!(exp.is_some());
}

/// Fix #163: GGUF attn_norm pattern should be recognized as RMSNorm
#[test]
fn test_tensor_expectation_gguf_attn_norm() {
    let exp = TensorExpectation::for_tensor("blk.0.attn_norm.weight");
    assert!(exp.is_some());
    let exp = exp.unwrap();
    assert_eq!(exp.description, "RMSNorm weight (gamma)");
    // Mean range should be wide enough for trained weights
    assert!(exp.mean_range.0 <= 0.0 && exp.mean_range.1 >= 2.0);
}

/// Fix #163: GGUF ffn_norm pattern should be recognized as RMSNorm
#[test]
fn test_tensor_expectation_gguf_ffn_norm() {
    let exp = TensorExpectation::for_tensor("blk.5.ffn_norm.weight");
    assert!(exp.is_some());
    let exp = exp.unwrap();
    assert_eq!(exp.description, "RMSNorm weight (gamma)");
}

#[test]
fn test_tensor_expectation_ln_weight() {
    let exp = TensorExpectation::for_tensor("ln_1.weight");
    assert!(exp.is_some());
    let exp = exp.unwrap();
    assert_eq!(exp.description, "LayerNorm weight (gamma)");
}

#[test]
fn test_tensor_expectation_ln_bias() {
    let exp = TensorExpectation::for_tensor("ln_1.bias");
    assert!(exp.is_some());
    let exp = exp.unwrap();
    assert_eq!(exp.description, "LayerNorm bias (beta)");
}

#[test]
fn test_tensor_expectation_gamma() {
    let exp = TensorExpectation::for_tensor("layer_norm.gamma");
    assert!(exp.is_some());
    let exp = exp.unwrap();
    assert_eq!(exp.description, "LayerNorm weight (gamma)");
}

#[test]
fn test_tensor_expectation_beta() {
    let exp = TensorExpectation::for_tensor("layer_norm.beta");
    assert!(exp.is_some());
    let exp = exp.unwrap();
    assert_eq!(exp.description, "LayerNorm bias (beta)");
}

#[test]
fn test_tensor_expectation_final_norm() {
    let exp = TensorExpectation::for_tensor("norm.weight");
    assert!(exp.is_some());
    let exp = exp.unwrap();
    assert_eq!(exp.description, "RMSNorm weight (gamma)");
}

#[test]
fn test_tensor_expectation_model_norm() {
    let exp = TensorExpectation::for_tensor("model.norm.weight");
    assert!(exp.is_some());
    let exp = exp.unwrap();
    assert_eq!(exp.description, "RMSNorm weight (gamma)");
}

#[test]
fn test_tensor_expectation_linear_weight() {
    let exp = TensorExpectation::for_tensor("fc1.weight");
    assert!(exp.is_some());
    let exp = exp.unwrap();
    assert_eq!(exp.description, "Linear/Attention weight");
}

#[test]
fn test_tensor_expectation_check_valid_layernorm() {
    let exp = TensorExpectation::LAYER_NORM_WEIGHT;
    let stats = TensorStats {
        name: "test.weight".to_string(),
        count: 1000,
        mean: 1.0,
        std: 0.5,
        min: 0.0,
        max: 2.0,
        nan_count: 0,
        inf_count: 0,
        zero_count: 0,
    };
    assert!(exp.check(&stats).is_ok());
}

#[test]
fn test_tensor_expectation_check_invalid_layernorm_std() {
    let exp = TensorExpectation::LAYER_NORM_WEIGHT;
    let stats = TensorStats {
        name: "test.weight".to_string(),
        count: 1000,
        mean: 1.0,
        std: 5.0, // Too high
        min: -10.0,
        max: 10.0,
        nan_count: 0,
        inf_count: 0,
        zero_count: 0,
    };
    // std range is Some((0.0, 2.0)), so 5.0 is outside
    assert!(exp.check(&stats).is_err());
}

#[test]
fn test_tensor_expectation_linear_no_std_range() {
    let exp = TensorExpectation::LINEAR_WEIGHT;
    assert!(exp.std_range.is_none());
}

#[test]
fn test_tensor_expectation_embedding_range() {
    let exp = TensorExpectation::EMBEDDING;
    assert!(exp.mean_range.0 < 0.0);
    assert!(exp.mean_range.1 > 0.0);
}

#[test]
fn test_tensor_expectation_rmsnorm_range() {
    let exp = TensorExpectation::RMSNORM_WEIGHT;
    // Wide range for trained models
    assert!(exp.mean_range.0 < 0.0);
    assert!(exp.mean_range.1 > 2.0);
}

// ========================================================================
// Additional Architecture Coverage Tests
// ========================================================================

#[test]
fn test_architecture_auto_preserves_model_prefix() {
    let arch = Architecture::Auto;
    assert_eq!(arch.map_name("model.weight"), "model.weight");
}

#[test]
fn test_architecture_whisper_preserves_prefix() {
    let arch = Architecture::Whisper;
    assert_eq!(
        arch.map_name("model.encoder.weight"),
        "model.encoder.weight"
    );
}

#[test]
fn test_architecture_llama_preserves_prefix() {
    let arch = Architecture::Llama;
    assert_eq!(
        arch.map_name("model.layers.0.weight"),
        "model.layers.0.weight"
    );
}

#[test]
fn test_architecture_bert_preserves_prefix() {
    let arch = Architecture::Bert;
    assert_eq!(arch.map_name("bert.encoder.weight"), "bert.encoder.weight");
}

#[test]
fn test_architecture_qwen2_preserves_prefix() {
    let arch = Architecture::Qwen2;
    assert_eq!(
        arch.map_name("model.embed_tokens.weight"),
        "model.embed_tokens.weight"
    );
}

#[test]
fn test_architecture_debug() {
    let arch = Architecture::Auto;
    assert!(format!("{:?}", arch).contains("Auto"));
}

#[test]
fn test_architecture_clone() {
    let arch1 = Architecture::Llama;
    let arch2 = arch1.clone();
    assert_eq!(arch1, arch2);
}

// ========================================================================
// Source Type Coverage Tests
// ========================================================================

#[test]
fn test_source_hf_with_file() {
    let source = Source::HuggingFace {
        org: "org".to_string(),
        repo: "repo".to_string(),
        file: Some("custom.safetensors".to_string()),
    };
    assert_eq!(source.default_file(), "custom.safetensors");
}

#[test]
fn test_source_url_default_file() {
    let source = Source::Url("https://example.com/path/to/model.gguf".to_string());
    assert_eq!(source.default_file(), "model.gguf");
}

#[test]
fn test_source_url_no_filename() {
    let source = Source::Url("https://example.com/".to_string());
    // URL without filename returns empty (edge case)
    let file = source.default_file();
    // Can be empty if no filename in URL
    let _ = file;
}

#[test]
fn test_source_debug() {
    let source = Source::Local("/path/to/model".into());
    assert!(format!("{:?}", source).contains("Local"));
}

#[test]
fn test_source_clone() {
    let source1 = Source::Url("https://test.com".to_string());
    let source2 = source1.clone();
    assert!(matches!(source2, Source::Url(_)));
}

// ========================================================================
// Validation Config Coverage Tests
// ========================================================================

#[test]
fn test_validation_config_none() {
    let config = ValidationConfig::None;
    assert!(matches!(config, ValidationConfig::None));
}

#[test]
fn test_validation_config_basic() {
    let config = ValidationConfig::Basic;
    assert!(matches!(config, ValidationConfig::Basic));
}

// ========================================================================
// QuantizationType Coverage Tests
// ========================================================================

#[test]
fn test_quantization_type_eq() {
    assert_eq!(QuantizationType::Fp16, QuantizationType::Fp16);
    assert_ne!(QuantizationType::Int8, QuantizationType::Int4);
}

#[test]
fn test_quantization_type_q4k() {
    let q = QuantizationType::Q4K;
    assert!(format!("{:?}", q).contains("Q4K"));
}

// ========================================================================
// Compression Coverage Tests
// ========================================================================

#[test]
fn test_compression_zstd_default() {
    let c = Compression::ZstdDefault;
    assert!(format!("{:?}", c).contains("Zstd"));
}

#[test]
fn test_compression_eq() {
    assert_eq!(Compression::Lz4, Compression::Lz4);
    assert_ne!(Compression::Lz4, Compression::ZstdDefault);
}

// ========================================================================
// Import Options Coverage Tests
// ========================================================================

#[test]
fn test_import_options_with_quantize() {
    let opts = ImportOptions {
        architecture: Architecture::Auto,
        validation: ValidationConfig::Basic,
        quantize: Some(QuantizationType::Int8),
        compress: Some(Compression::Lz4),
        strict: true,
        cache: false,
        tokenizer_path: None,
        allow_no_config: true,
    };
    assert!(opts.quantize.is_some());
    assert!(opts.compress.is_some());
    assert!(opts.strict);
    assert!(!opts.cache);
}

#[test]
fn test_import_options_debug() {
    let opts = ImportOptions::default();
    assert!(format!("{:?}", opts).contains("ImportOptions"));
}

#[test]
fn test_import_options_clone() {
    let opts1 = ImportOptions::default();
    let opts2 = opts1.clone();
    assert_eq!(opts1.validation, opts2.validation);
}

// ========================================================================
// ConvertOptions Coverage Tests
// ========================================================================

#[test]
fn test_convert_options_default() {
    let opts = ConvertOptions::default();
    assert!(opts.quantize.is_none());
    assert!(opts.compress.is_none());
}

#[test]
fn test_convert_options_with_all() {
    let opts = ConvertOptions {
        quantize: Some(QuantizationType::Q4K),
        compress: Some(Compression::ZstdDefault),
        validate: true,
    };
    assert!(opts.quantize.is_some());
    assert!(opts.compress.is_some());
    assert!(opts.validate);
}

#[test]
fn test_convert_options_debug() {
    let opts = ConvertOptions::default();
    assert!(format!("{:?}", opts).contains("ConvertOptions"));
}

#[test]
fn test_convert_options_clone() {
    let opts1 = ConvertOptions {
        quantize: Some(QuantizationType::Int8),
        compress: None,
        validate: false,
    };
    let opts2 = opts1.clone();
    assert_eq!(opts1.quantize, opts2.quantize);
    assert_eq!(opts1.validate, opts2.validate);
}

// ========================================================================
// TensorStats Coverage Tests
// ========================================================================

#[test]
fn test_tensor_stats_debug() {
    let stats = TensorStats {
        name: "test".to_string(),
        count: 100,
        mean: 0.0,
        std: 1.0,
        min: -3.0,
        max: 3.0,
        nan_count: 0,
        inf_count: 0,
        zero_count: 10,
    };
    assert!(format!("{:?}", stats).contains("TensorStats"));
}

#[test]
fn test_tensor_stats_clone() {
    let stats1 = TensorStats {
        name: "w".to_string(),
        count: 50,
        mean: 0.5,
        std: 0.1,
        min: 0.0,
        max: 1.0,
        nan_count: 0,
        inf_count: 0,
        zero_count: 5,
    };
    let stats2 = stats1.clone();
    assert_eq!(stats1.name, stats2.name);
    assert_eq!(stats1.count, stats2.count);
}

// ========================================================================
// Internal Helper Function Tests (ROSETTA-ML-001)
// ========================================================================

#[test]
fn test_compute_std_basic() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let mean = 3.0;
    let std = compute_std(&data, mean, 5);
    // Expected: sqrt(((1-3)^2 + (2-3)^2 + (3-3)^2 + (4-3)^2 + (5-3)^2) / 4)
    // = sqrt((4 + 1 + 0 + 1 + 4) / 4) = sqrt(10/4) = sqrt(2.5) ≈ 1.58
    assert!((std - 1.58).abs() < 0.01);
}

#[test]
fn test_compute_std_single_value() {
    let data = vec![42.0];
    let std = compute_std(&data, 42.0, 1);
    assert_eq!(std, 0.0);
}

#[test]
fn test_compute_std_empty() {
    let data: Vec<f32> = vec![];
    let std = compute_std(&data, 0.0, 0);
    assert_eq!(std, 0.0);
}

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

// ========================================================================
// Pygmy-Based Conversion Tests (T-COV-95)
// Using in-memory model builders to test conversion paths without real files
// ========================================================================

#[test]
fn test_pygmy_safetensors_to_apr_conversion() {
    use crate::format::test_factory::build_pygmy_safetensors;
    use std::fs;
    use tempfile::tempdir;

    // Build pygmy SafeTensors
    let st_data = build_pygmy_safetensors();

    // Write to temp file
    let dir = tempdir().expect("Failed to create temp dir");
    let st_path = dir.path().join("pygmy_model.safetensors");
    fs::write(&st_path, &st_data).expect("Failed to write SafeTensors");

    // Read back and verify
    let read_back = fs::read(&st_path).expect("Failed to read back");
    assert_eq!(read_back.len(), st_data.len());

    // Verify SafeTensors header is valid
    let header_len = u64::from_le_bytes(read_back[0..8].try_into().unwrap());
    assert!(header_len > 0);
}

#[test]
fn test_pygmy_apr_roundtrip() {
    use crate::format::test_factory::build_pygmy_apr;
    use crate::format::v2::AprV2Reader;
    use std::fs;
    use tempfile::tempdir;

    // Build pygmy APR
    let apr_data = build_pygmy_apr();

    // Write to temp file
    let dir = tempdir().expect("Failed to create temp dir");
    let apr_path = dir.path().join("pygmy_model.apr");
    fs::write(&apr_path, &apr_data).expect("Failed to write APR");

    // Read back and parse
    let read_back = fs::read(&apr_path).expect("Failed to read back");
    let reader = AprV2Reader::from_bytes(&read_back).expect("Failed to parse APR");

    // Verify metadata
    assert_eq!(reader.metadata().architecture, Some("llama".to_string()));
    assert!(!reader.tensor_names().is_empty());
}

#[test]
fn test_pygmy_apr_f16_roundtrip() {
    use crate::format::test_factory::build_pygmy_apr_f16;
    use crate::format::v2::AprV2Reader;
    use std::fs;
    use tempfile::tempdir;

    let apr_data = build_pygmy_apr_f16();
    let dir = tempdir().expect("Failed to create temp dir");
    let apr_path = dir.path().join("pygmy_f16.apr");
    fs::write(&apr_path, &apr_data).expect("Failed to write APR");

    let read_back = fs::read(&apr_path).expect("Failed to read back");
    let reader = AprV2Reader::from_bytes(&read_back).expect("Failed to parse APR");

    // Verify alignment (critical for mmap)
    assert!(reader.verify_alignment());
}

#[test]
fn test_pygmy_apr_q8_roundtrip() {
    use crate::format::test_factory::build_pygmy_apr_q8;
    use crate::format::v2::AprV2Reader;

    let apr_data = build_pygmy_apr_q8();
    let reader = AprV2Reader::from_bytes(&apr_data).expect("Failed to parse APR");

    // Verify has attention tensors
    let tensor_names = reader.tensor_names();
    let has_q_proj = tensor_names
        .iter()
        .any(|n| n.contains("self_attn.q_proj.weight"));
    assert!(has_q_proj);
}

#[test]
fn test_pygmy_apr_q4_roundtrip() {
    use crate::format::test_factory::build_pygmy_apr_q4;
    use crate::format::v2::AprV2Reader;

    let apr_data = build_pygmy_apr_q4();
    let reader = AprV2Reader::from_bytes(&apr_data).expect("Failed to parse APR");

    // Verify alignment with Q4 block-aligned tensors
    assert!(reader.verify_alignment());
}

#[test]
fn test_pygmy_config_llama_style_tensor_names() {
    use crate::format::test_factory::{build_pygmy_apr_with_config, PygmyConfig};
    use crate::format::v2::AprV2Reader;

    let config = PygmyConfig::llama_style();
    let apr_data = build_pygmy_apr_with_config(config);
    let reader = AprV2Reader::from_bytes(&apr_data).expect("Failed to parse APR");

    let names = reader.tensor_names();

    // LLaMA-style should have standard tensor names
    assert!(names.iter().any(|n| n.contains("embed_tokens")));
    assert!(names.iter().any(|n| n.contains("input_layernorm")));
    assert!(names.iter().any(|n| n.contains("self_attn.q_proj")));
    assert!(names.iter().any(|n| n.contains("mlp.gate_proj")));
}

#[test]
fn test_pygmy_multiple_configs_all_valid() {
    use crate::format::test_factory::{
        build_pygmy_apr_with_config, build_pygmy_safetensors_with_config, PygmyConfig,
    };
    use crate::format::v2::AprV2Reader;

    let configs = vec![
        PygmyConfig::default(),
        PygmyConfig::minimal(),
        PygmyConfig::embedding_only(),
        PygmyConfig::llama_style(),
    ];

    for config in configs {
        // Test SafeTensors generation
        let st_data = build_pygmy_safetensors_with_config(config.clone());
        assert!(st_data.len() > 8, "SafeTensors should have header");

        // Test APR generation
        let apr_data = build_pygmy_apr_with_config(config);
        let reader = AprV2Reader::from_bytes(&apr_data);
        assert!(reader.is_ok(), "APR should be parseable");
    }
}

#[test]
fn test_pygmy_safetensors_tensor_data_validity() {
    use crate::format::test_factory::build_pygmy_safetensors;

    let data = build_pygmy_safetensors();

    // Parse header
    let header_len = u64::from_le_bytes(data[0..8].try_into().unwrap()) as usize;
    let header_str = std::str::from_utf8(&data[8..8 + header_len]).unwrap();

    // Should be valid JSON
    let parsed: serde_json::Value = serde_json::from_str(header_str).unwrap();
    assert!(parsed.is_object());

    // Should have data_offsets for tensors
    if let Some(obj) = parsed.as_object() {
        for (key, val) in obj {
            if key != "__metadata__" {
                assert!(val.get("data_offsets").is_some());
                assert!(val.get("dtype").is_some());
                assert!(val.get("shape").is_some());
            }
        }
    }
}

#[test]
fn test_tensor_accumulator_basic() {
    let mut acc = TensorAccumulator::new();
    for &v in &[1.0, 2.0, 3.0, 4.0, 5.0] {
        acc.accumulate(v);
    }
    assert!((acc.mean() - 3.0).abs() < 0.01);
}

#[test]
fn test_tensor_accumulator_with_nan_and_inf() {
    let mut acc = TensorAccumulator::new();
    acc.accumulate(1.0);
    acc.accumulate(f32::NAN);
    acc.accumulate(3.0);
    acc.accumulate(f32::INFINITY);
    acc.accumulate(5.0);

    // safe_min/safe_max ignore NaN/Inf
    assert!((acc.safe_min() - 1.0).abs() < 0.01);
    assert!((acc.safe_max() - 5.0).abs() < 0.01);
}

#[test]
fn test_tensor_accumulator_empty() {
    let acc = TensorAccumulator::new();
    assert_eq!(acc.mean(), 0.0);
    assert_eq!(acc.safe_min(), 0.0);
    assert_eq!(acc.safe_max(), 0.0);
}

// ========================================================================
// ExportFormat and ExportOptions Tests (T-COV-95)
// ========================================================================

#[test]
fn test_export_format_from_str_safetensors() {
    use std::str::FromStr;
    assert_eq!(
        ExportFormat::from_str("safetensors").unwrap(),
        ExportFormat::SafeTensors
    );
    assert_eq!(
        ExportFormat::from_str("st").unwrap(),
        ExportFormat::SafeTensors
    );
    assert_eq!(
        ExportFormat::from_str("SAFETENSORS").unwrap(),
        ExportFormat::SafeTensors
    );
}

#[test]
fn test_export_format_from_str_gguf() {
    use std::str::FromStr;
    assert_eq!(ExportFormat::from_str("gguf").unwrap(), ExportFormat::Gguf);
    assert_eq!(ExportFormat::from_str("GGUF").unwrap(), ExportFormat::Gguf);
}

#[test]
fn test_export_format_from_str_onnx() {
    use std::str::FromStr;
    assert_eq!(ExportFormat::from_str("onnx").unwrap(), ExportFormat::Onnx);
}

#[test]
fn test_export_format_from_str_torchscript() {
    use std::str::FromStr;
    assert_eq!(
        ExportFormat::from_str("torchscript").unwrap(),
        ExportFormat::TorchScript
    );
    assert_eq!(
        ExportFormat::from_str("pt").unwrap(),
        ExportFormat::TorchScript
    );
    assert_eq!(
        ExportFormat::from_str("torch").unwrap(),
        ExportFormat::TorchScript
    );
}

#[test]
fn test_export_format_from_str_unknown() {
    use std::str::FromStr;
    let result = ExportFormat::from_str("unknown_format");
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("Unknown"));
}

#[test]
fn test_export_format_extension() {
    assert_eq!(ExportFormat::SafeTensors.extension(), "safetensors");
    assert_eq!(ExportFormat::Gguf.extension(), "gguf");
    assert_eq!(ExportFormat::Onnx.extension(), "onnx");
    assert_eq!(ExportFormat::TorchScript.extension(), "pt");
}

#[test]
fn test_export_format_is_supported() {
    assert!(ExportFormat::SafeTensors.is_supported());
    assert!(ExportFormat::Gguf.is_supported());
    assert!(!ExportFormat::Onnx.is_supported());
    assert!(!ExportFormat::TorchScript.is_supported());
}

#[test]
fn test_export_format_debug() {
    let format = ExportFormat::SafeTensors;
    let debug_str = format!("{:?}", format);
    assert!(debug_str.contains("SafeTensors"));
}

#[test]
fn test_export_format_clone() {
    let format = ExportFormat::Gguf;
    let cloned = format;
    assert_eq!(format, cloned);
}

#[test]
fn test_export_options_default() {
    let opts = ExportOptions::default();
    assert_eq!(opts.format, ExportFormat::SafeTensors);
    assert!(opts.quantize.is_none());
    assert!(opts.include_tokenizer);
    assert!(opts.include_config);
}

#[test]
fn test_export_options_custom() {
    let opts = ExportOptions {
        format: ExportFormat::Gguf,
        quantize: Some(QuantizationType::Int8),
        include_tokenizer: false,
        include_config: false,
    };
    assert_eq!(opts.format, ExportFormat::Gguf);
    assert_eq!(opts.quantize, Some(QuantizationType::Int8));
    assert!(!opts.include_tokenizer);
    assert!(!opts.include_config);
}

#[test]
fn test_export_options_debug() {
    let opts = ExportOptions::default();
    let debug_str = format!("{:?}", opts);
    assert!(debug_str.contains("ExportOptions"));
}

#[test]
fn test_export_options_clone() {
    let opts = ExportOptions {
        format: ExportFormat::Gguf,
        quantize: Some(QuantizationType::Int4),
        include_tokenizer: true,
        include_config: false,
    };
    let cloned = opts.clone();
    assert_eq!(opts.format, cloned.format);
    assert_eq!(opts.quantize, cloned.quantize);
}

#[test]
fn test_export_report_struct() {
    let report = ExportReport {
        original_size: 1000,
        exported_size: 500,
        tensor_count: 10,
        format: ExportFormat::SafeTensors,
        quantization: Some(QuantizationType::Int8),
    };
    assert_eq!(report.original_size, 1000);
    assert_eq!(report.exported_size, 500);
    assert_eq!(report.tensor_count, 10);
    assert_eq!(report.format, ExportFormat::SafeTensors);
    assert_eq!(report.quantization, Some(QuantizationType::Int8));
}

#[test]
fn test_export_report_debug() {
    let report = ExportReport {
        original_size: 100,
        exported_size: 50,
        tensor_count: 5,
        format: ExportFormat::Gguf,
        quantization: None,
    };
    let debug_str = format!("{:?}", report);
    assert!(debug_str.contains("ExportReport"));
}

// ========================================================================
// MergeOptions and MergeReport Tests (T-COV-95)
// ========================================================================

#[test]
fn test_merge_strategy_variants() {
    let _ = MergeStrategy::Average;
    let _ = MergeStrategy::Weighted;
    let _ = MergeStrategy::Slerp;
    let _ = MergeStrategy::Ties;
    let _ = MergeStrategy::Dare;
}

#[test]
fn test_merge_strategy_from_str() {
    use std::str::FromStr;
    assert_eq!(
        MergeStrategy::from_str("average").unwrap(),
        MergeStrategy::Average
    );
    assert_eq!(
        MergeStrategy::from_str("weighted").unwrap(),
        MergeStrategy::Weighted
    );
    assert_eq!(
        MergeStrategy::from_str("slerp").unwrap(),
        MergeStrategy::Slerp
    );
    assert_eq!(
        MergeStrategy::from_str("ties").unwrap(),
        MergeStrategy::Ties
    );
    assert_eq!(
        MergeStrategy::from_str("dare").unwrap(),
        MergeStrategy::Dare
    );
}

#[test]
fn test_merge_strategy_from_str_case_insensitive() {
    use std::str::FromStr;
    assert_eq!(
        MergeStrategy::from_str("AVERAGE").unwrap(),
        MergeStrategy::Average
    );
    assert_eq!(
        MergeStrategy::from_str("Average").unwrap(),
        MergeStrategy::Average
    );
}

#[test]
fn test_merge_strategy_from_str_unknown() {
    use std::str::FromStr;
    let result = MergeStrategy::from_str("unknown");
    assert!(result.is_err());
}

#[test]
fn test_merge_options_default() {
    let opts = MergeOptions::default();
    assert_eq!(opts.strategy, MergeStrategy::Average);
    assert!(opts.weights.is_none());
}

#[test]
fn test_merge_options_custom() {
    let opts = MergeOptions {
        strategy: MergeStrategy::Weighted,
        weights: Some(vec![0.7, 0.3]),
        ..Default::default()
    };
    assert_eq!(opts.strategy, MergeStrategy::Weighted);
    assert!(opts.weights.is_some());
    assert_eq!(opts.weights.as_ref().unwrap().len(), 2);
}

#[test]
fn test_merge_options_slerp() {
    let opts = MergeOptions {
        strategy: MergeStrategy::Slerp,
        weights: None,
        ..Default::default()
    };
    assert_eq!(opts.strategy, MergeStrategy::Slerp);
    assert!(opts.weights.is_none());
}

#[test]
fn test_merge_report_struct() {
    let report = MergeReport {
        model_count: 3,
        strategy: MergeStrategy::Average,
        tensor_count: 100,
        output_size: 10000,
        weights_used: None,
    };
    assert_eq!(report.model_count, 3);
    assert_eq!(report.tensor_count, 100);
}
