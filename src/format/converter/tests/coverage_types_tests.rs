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

#[path = "coverage_types_tests_part_02.rs"]
mod coverage_types_tests_part_02;
#[path = "coverage_types_tests_part_03.rs"]
mod coverage_types_tests_part_03;
#[path = "coverage_types_tests_part_04.rs"]
mod coverage_types_tests_part_04;
#[path = "coverage_types_tests_part_05.rs"]
mod coverage_types_tests_part_05;
