use super::*;

#[test]
fn test_import_error_not_found_message() {
    let err = ImportError::NotFound {
        resource: "openai/whisper-tiny".to_string(),
        status: 404,
    };
    let msg = err.to_string();
    assert!(msg.contains("404"), "Should include status code");
    assert!(msg.contains("whisper-tiny"), "Should include resource");
}

#[test]
fn test_import_error_rate_limited_message() {
    let err = ImportError::RateLimited {
        retry_after: Some(60),
    };
    let msg = err.to_string();
    assert!(
        msg.to_lowercase().contains("rate"),
        "Should mention rate limit"
    );
    assert!(msg.contains("60"), "Should include retry time");
}

#[test]
fn test_import_error_auth_required_message() {
    let err = ImportError::AuthRequired {
        resource: "meta-llama/Llama-2-7b".to_string(),
    };
    let msg = err.to_string();
    assert!(msg.contains("HF_TOKEN"), "Should suggest HF_TOKEN");
    assert!(msg.contains("Llama-2-7b"), "Should include resource");
}

#[test]
fn test_import_error_actionable_suggestions() {
    let err = ImportError::NotFound {
        resource: "openai/whisper-tiny".to_string(),
        status: 404,
    };

    // Error should provide actionable fix
    let msg = err.to_string();
    assert!(
        msg.contains("Fix:") || msg.contains("check") || msg.contains("verify"),
        "Error should be actionable"
    );
}

#[test]
fn test_import_error_sharding_oom() {
    let err = ImportError::ShardingRequired {
        model_size: 14_000_000_000, // 14GB
        shard_count: 7,
    };
    let msg = err.to_string();
    assert!(msg.contains("14"), "Should include size");
    assert!(msg.contains("7"), "Should include shard count");
}

// GH-129: Tests for parse_import_error (only when hf-hub-integration enabled)
#[cfg(feature = "hf-hub-integration")]
#[test]
fn test_parse_import_error_404() {
    let err = parse_import_error("HTTP 404: Repository not found", "openai/whisper-tiny");
    match err {
        ImportError::NotFound { resource, status } => {
            assert_eq!(resource, "openai/whisper-tiny");
            assert_eq!(status, 404);
        }
        _ => panic!("Expected NotFound error, got {:?}", err),
    }
}

#[cfg(feature = "hf-hub-integration")]
#[test]
fn test_parse_import_error_not_found_text() {
    let err = parse_import_error("The requested resource does not exist", "test/model");
    match err {
        ImportError::NotFound { .. } => {}
        _ => panic!("Expected NotFound error, got {:?}", err),
    }
}

#[cfg(feature = "hf-hub-integration")]
#[test]
fn test_parse_import_error_401() {
    let err = parse_import_error("HTTP 401: Unauthorized access", "meta-llama/Llama-2-7b");
    match err {
        ImportError::AuthRequired { resource } => {
            assert_eq!(resource, "meta-llama/Llama-2-7b");
        }
        _ => panic!("Expected AuthRequired error, got {:?}", err),
    }
}

#[cfg(feature = "hf-hub-integration")]
#[test]
fn test_parse_import_error_gated_model() {
    let err = parse_import_error(
        "This model is gated. Access requires acceptance.",
        "meta-llama/Llama-2-7b",
    );
    match err {
        ImportError::AuthRequired { .. } => {}
        _ => panic!("Expected AuthRequired error, got {:?}", err),
    }
}

#[cfg(feature = "hf-hub-integration")]
#[test]
fn test_parse_import_error_429() {
    let err = parse_import_error(
        "HTTP 429: Too many requests. Retry after 60 seconds.",
        "test/model",
    );
    match err {
        ImportError::RateLimited { retry_after } => {
            assert_eq!(retry_after, Some(60));
        }
        _ => panic!("Expected RateLimited error, got {:?}", err),
    }
}

#[cfg(feature = "hf-hub-integration")]
#[test]
fn test_parse_import_error_rate_limit_no_retry() {
    let err = parse_import_error("Rate limit exceeded", "test/model");
    match err {
        ImportError::RateLimited { retry_after } => {
            assert_eq!(retry_after, None);
        }
        _ => panic!("Expected RateLimited error, got {:?}", err),
    }
}

#[cfg(feature = "hf-hub-integration")]
#[test]
fn test_parse_import_error_generic() {
    let err = parse_import_error("Connection timeout", "test/model");
    match err {
        ImportError::DownloadFailed { source, reason } => {
            assert_eq!(source, "test/model");
            assert_eq!(reason, "Connection timeout");
        }
        _ => panic!("Expected DownloadFailed error, got {:?}", err),
    }
}

#[test]
fn test_import_error_from_conversion() {
    let import_err = ImportError::NotFound {
        resource: "test".to_string(),
        status: 404,
    };
    let aprender_err: AprenderError = import_err.into();
    let msg = aprender_err.to_string();
    assert!(msg.contains("404"));
    assert!(msg.contains("test"));
}

// =========================================================================
// Coverage boost: ExportFormat, MergeStrategy, and related APIs
// =========================================================================

#[test]
fn test_export_format_from_str() {
    assert!(matches!(
        "safetensors".parse::<ExportFormat>(),
        Ok(ExportFormat::SafeTensors)
    ));
    assert!(matches!(
        "st".parse::<ExportFormat>(),
        Ok(ExportFormat::SafeTensors)
    ));
    assert!(matches!(
        "gguf".parse::<ExportFormat>(),
        Ok(ExportFormat::Gguf)
    ));
    assert!(matches!(
        "onnx".parse::<ExportFormat>(),
        Ok(ExportFormat::Onnx)
    ));
    assert!(matches!(
        "torchscript".parse::<ExportFormat>(),
        Ok(ExportFormat::TorchScript)
    ));
    assert!(matches!(
        "pt".parse::<ExportFormat>(),
        Ok(ExportFormat::TorchScript)
    ));
    assert!(matches!(
        "torch".parse::<ExportFormat>(),
        Ok(ExportFormat::TorchScript)
    ));
    assert!("unknown".parse::<ExportFormat>().is_err());
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
fn test_export_options_default() {
    let opts = ExportOptions::default();
    assert!(matches!(opts.format, ExportFormat::SafeTensors));
    assert!(opts.quantize.is_none());
}

#[test]
fn test_export_options_with_quantize() {
    let opts = ExportOptions {
        format: ExportFormat::Gguf,
        quantize: Some(QuantizationType::Int8),
        ..Default::default()
    };
    assert!(matches!(opts.format, ExportFormat::Gguf));
    assert!(matches!(opts.quantize, Some(QuantizationType::Int8)));
}

#[test]
fn test_merge_strategy_from_str() {
    assert!(matches!(
        "average".parse::<MergeStrategy>(),
        Ok(MergeStrategy::Average)
    ));
    assert!(matches!(
        "avg".parse::<MergeStrategy>(),
        Ok(MergeStrategy::Average)
    ));
    assert!(matches!(
        "weighted".parse::<MergeStrategy>(),
        Ok(MergeStrategy::Weighted)
    ));
    assert!("unknown".parse::<MergeStrategy>().is_err());
}

#[test]
fn test_merge_strategy_is_supported() {
    // Average and Weighted are supported
    assert!(MergeStrategy::Average.is_supported());
    assert!(MergeStrategy::Weighted.is_supported());
    // Advanced strategies not yet implemented
    assert!(!MergeStrategy::Ties.is_supported());
    assert!(!MergeStrategy::Dare.is_supported());
    assert!(!MergeStrategy::Slerp.is_supported());
}

#[test]
fn test_merge_options_default() {
    let opts = MergeOptions::default();
    assert!(matches!(opts.strategy, MergeStrategy::Average));
    assert!(opts.weights.is_none());
}

#[test]
fn test_merge_options_weighted() {
    let opts = MergeOptions {
        strategy: MergeStrategy::Weighted,
        weights: Some(vec![0.7, 0.3]),
    };
    assert!(matches!(opts.strategy, MergeStrategy::Weighted));
    assert_eq!(opts.weights, Some(vec![0.7, 0.3]));
}

#[test]
fn test_merge_report_fields() {
    let report = MergeReport {
        model_count: 2,
        output_size: 1000,
        tensor_count: 10,
        strategy: MergeStrategy::Average,
        weights_used: None,
    };
    assert_eq!(report.model_count, 2);
    assert_eq!(report.output_size, 1000);
    assert_eq!(report.tensor_count, 10);
}

#[test]
fn test_merge_report_with_weights() {
    let report = MergeReport {
        model_count: 3,
        output_size: 2000,
        tensor_count: 15,
        strategy: MergeStrategy::Weighted,
        weights_used: Some(vec![0.5, 0.3, 0.2]),
    };
    assert_eq!(report.model_count, 3);
    assert!(matches!(report.strategy, MergeStrategy::Weighted));
    assert!(report.weights_used.is_some());
}

#[test]
fn test_export_report_fields() {
    let report = ExportReport {
        original_size: 2000,
        exported_size: 1000,
        tensor_count: 5,
        format: ExportFormat::Gguf,
        quantization: Some(QuantizationType::Int8),
    };
    assert_eq!(report.original_size, 2000);
    assert_eq!(report.exported_size, 1000);
    assert_eq!(report.tensor_count, 5);
}

#[test]
fn test_validation_config_strict() {
    let config = ValidationConfig::strict();
    assert_eq!(config, ValidationConfig::Strict);
}

#[test]
fn test_validation_config_default() {
    let config = ValidationConfig::default();
    assert_eq!(config, ValidationConfig::Strict);
}

#[test]
fn test_validation_config_variants() {
    let _none = ValidationConfig::None;
    let _basic = ValidationConfig::Basic;
    let _strict = ValidationConfig::Strict;
}

#[test]
fn test_import_options_default() {
    let opts = ImportOptions::default();
    assert_eq!(opts.validation, ValidationConfig::Strict);
    assert!(opts.quantize.is_none());
    assert!(opts.compress.is_none());
}

#[test]
fn test_architecture_mapping_auto() {
    let arch = Architecture::Auto;
    // PMAT-099: Preserve model. prefix for AprTransformer compatibility
    assert_eq!(
        arch.map_name("model.embed_tokens.weight"),
        "model.embed_tokens.weight"
    );
    // Pass through names without prefix
    assert_eq!(arch.map_name("layer.0.weight"), "layer.0.weight");
}

#[test]
fn test_architecture_mapping_whisper() {
    let arch = Architecture::Whisper;
    let name = arch.map_name("model.encoder.weight");
    assert!(!name.is_empty());
}

#[test]
fn test_architecture_mapping_llama() {
    let arch = Architecture::Llama;
    let name = arch.map_name("model.layers.0.weight");
    assert!(!name.is_empty());
}

#[test]
fn test_architecture_mapping_bert() {
    let arch = Architecture::Bert;
    let name = arch.map_name("bert.encoder.layer.0.weight");
    assert!(!name.is_empty());
}

#[test]
fn test_source_parse_local_absolute() {
    let source = Source::parse("/path/to/model.safetensors").unwrap();
    assert!(matches!(source, Source::Local(_)));
}

#[test]
fn test_source_parse_local_relative() {
    let source = Source::parse("./models/model.safetensors").unwrap();
    assert!(matches!(source, Source::Local(_)));
}

#[test]
fn test_source_default_file_hf() {
    let source = Source::HuggingFace {
        org: "openai".to_string(),
        repo: "whisper".to_string(),
        file: None,
    };
    assert_eq!(source.default_file(), "model.safetensors");
}

#[test]
fn test_source_default_file_local() {
    let source = Source::Local("/path/to/model.safetensors".into());
    // Local returns full path as the "file"
    assert!(source.default_file().ends_with("model.safetensors"));
}

#[test]
fn test_tensor_expectation_for_unknown() {
    let exp = TensorExpectation::for_tensor("unknown_tensor_name");
    assert!(exp.is_none());
}

#[test]
fn test_tensor_expectation_for_layer_norm_weight() {
    let exp = TensorExpectation::for_tensor("layer_norm.weight");
    assert!(exp.is_some());
    let exp = exp.unwrap();
    // LayerNorm weight should have mean near 1.0
    assert!(exp.mean_range.0 < 1.0 && exp.mean_range.1 > 1.0);
}

#[test]
fn test_tensor_expectation_for_embedding() {
    let exp = TensorExpectation::for_tensor("embed_tokens.weight");
    assert!(exp.is_some());
}

#[test]
fn test_import_error_display() {
    let err = ImportError::NotFound {
        resource: "model.safetensors".to_string(),
        status: 404,
    };
    let msg = format!("{}", err);
    assert!(msg.contains("404") || msg.contains("not found"));
}

#[test]
fn test_import_error_download_failed() {
    let err = ImportError::DownloadFailed {
        source: "huggingface".to_string(),
        reason: "timeout".to_string(),
    };
    let msg = format!("{}", err);
    assert!(msg.contains("timeout") || msg.contains("Download"));
}

#[test]
fn test_import_error_validation_failed() {
    let err = ImportError::ValidationFailed {
        name: "layer.weight".to_string(),
        reason: "NaN detected".to_string(),
    };
    let msg = format!("{}", err);
    assert!(msg.contains("layer.weight") || msg.contains("NaN"));
}

#[test]
fn test_import_error_unsupported_format() {
    let err = ImportError::UnsupportedFormat {
        extension: "pickle".to_string(),
    };
    let msg = format!("{}", err);
    assert!(msg.contains("pickle") || msg.contains("Unsupported"));
}

#[test]
fn test_import_error_unknown_tensor() {
    let err = ImportError::UnknownTensor {
        source_name: "weird.tensor".to_string(),
    };
    let msg = format!("{}", err);
    assert!(msg.contains("weird.tensor") || msg.contains("Unknown"));
}

#[test]
fn test_import_error_missing_tensor() {
    let err = ImportError::MissingTensor {
        name: "model.weight".to_string(),
    };
    let msg = format!("{}", err);
    assert!(msg.contains("model.weight") || msg.contains("Missing"));
}

#[test]
fn test_import_error_rate_limited() {
    let err = ImportError::RateLimited {
        retry_after: Some(60),
    };
    let msg = format!("{}", err);
    assert!(msg.contains("Rate") || msg.contains("limit") || msg.contains("60"));
}

#[test]
fn test_import_error_auth_required() {
    let err = ImportError::AuthRequired {
        resource: "gated-model".to_string(),
    };
    let msg = format!("{}", err);
    assert!(msg.contains("Auth") || msg.contains("gated-model"));
}

#[test]
fn test_import_error_sharding_required() {
    let err = ImportError::ShardingRequired {
        model_size: 14_000_000_000,
        shard_count: 7,
    };
    let msg = format!("{}", err);
    assert!(msg.contains("shard") || msg.contains("7"));
}

// =========================================================================
// ShardedIndex Tests
// =========================================================================

#[test]
fn test_sharded_index_parse() {
    let json = r#"{
        "metadata": {"total_size": 1000000},
        "weight_map": {
            "layer.0.weight": "model-00001-of-00002.safetensors",
            "layer.1.weight": "model-00002-of-00002.safetensors"
        }
    }"#;
    let index = ShardedIndex::parse(json).expect("parse should succeed");
    assert_eq!(index.tensor_count(), 2);
    assert_eq!(index.shard_count(), 2);
}

#[test]
fn test_sharded_index_shard_for_tensor() {
    let json = r#"{
        "weight_map": {
            "embed.weight": "model-00001.safetensors",
            "lm_head.weight": "model-00002.safetensors"
        }
    }"#;
    let index = ShardedIndex::parse(json).expect("parse");
    assert_eq!(
        index.shard_for_tensor("embed.weight"),
        Some("model-00001.safetensors")
    );
    assert_eq!(
        index.shard_for_tensor("lm_head.weight"),
        Some("model-00002.safetensors")
    );
    assert_eq!(index.shard_for_tensor("missing"), None);
}

#[test]
fn test_sharded_index_tensors_in_shard() {
    let json = r#"{
        "weight_map": {
            "a.weight": "shard1.safetensors",
            "b.weight": "shard1.safetensors",
            "c.weight": "shard2.safetensors"
        }
    }"#;
    let index = ShardedIndex::parse(json).expect("parse");
    let tensors = index.tensors_in_shard("shard1.safetensors");
    assert_eq!(tensors.len(), 2);
    assert!(tensors.contains(&"a.weight"));
    assert!(tensors.contains(&"b.weight"));
}

#[test]
fn test_sharded_index_shard_files() {
    let json = r#"{
        "weight_map": {
            "a": "z.safetensors",
            "b": "a.safetensors",
            "c": "m.safetensors"
        }
    }"#;
    let index = ShardedIndex::parse(json).expect("parse");
    let files = index.shard_files();
    // Should be sorted
    assert_eq!(
        files,
        vec!["a.safetensors", "m.safetensors", "z.safetensors"]
    );
}

#[test]
fn test_sharded_index_total_size() {
    let with_size = r#"{"metadata": {"total_size": 5000}, "weight_map": {}}"#;
    let without_size = r#"{"weight_map": {}}"#;

    let index1 = ShardedIndex::parse(with_size).expect("parse");
    let index2 = ShardedIndex::parse(without_size).expect("parse");

    assert_eq!(index1.total_size(), Some(5000));
    assert_eq!(index2.total_size(), None);
}

#[test]
fn test_sharded_index_parse_invalid_json() {
    let result = ShardedIndex::parse("not valid json");
    assert!(result.is_err());
}

// =========================================================================
// Source URL Tests
// =========================================================================

#[test]
fn test_source_parse_url() {
    let source = Source::parse("https://example.com/model.safetensors").unwrap();
    assert!(matches!(source, Source::Url(_)));
}

#[test]
fn test_source_parse_http_url() {
    let source = Source::parse("http://localhost:8080/model.bin").unwrap();
    assert!(matches!(source, Source::Url(_)));
}

#[test]
fn test_source_default_file_url() {
    let source = Source::Url("https://example.com/path/to/model.safetensors".to_string());
    assert_eq!(source.default_file(), "model.safetensors");
}

// =========================================================================
// ConvertReport Tests
// =========================================================================

#[test]
fn test_convert_report_reduction_percent() {
    let report = ConvertReport {
        original_size: 1000,
        converted_size: 500,
        tensor_count: 10,
        quantization: Some(QuantizationType::Int8),
        compression: None,
        reduction_ratio: 2.0,
    };
    let reduction = report.reduction_percent();
    assert!(reduction.contains("50"));
}

#[test]
fn test_convert_report_no_reduction() {
    let report = ConvertReport {
        original_size: 1000,
        converted_size: 1000,
        tensor_count: 5,
        quantization: None,
        compression: None,
        reduction_ratio: 1.0,
    };
    let reduction = report.reduction_percent();
    assert!(reduction.contains("0"));
}

// =========================================================================
// ExportFormat Tests
// =========================================================================

#[test]
fn test_export_format_safetensors() {
    let format = ExportFormat::SafeTensors;
    assert_eq!(format.extension(), "safetensors");
    assert!(format.is_supported());
}

#[test]
fn test_export_format_gguf() {
    let format = ExportFormat::Gguf;
    assert_eq!(format.extension(), "gguf");
    assert!(format.is_supported());
}

#[test]
fn test_export_format_onnx() {
    let format = ExportFormat::Onnx;
    assert_eq!(format.extension(), "onnx");
    // ONNX may or may not be supported
    let _ = format.is_supported();
}

#[test]
fn test_export_format_torchscript() {
    let format = ExportFormat::TorchScript;
    assert_eq!(format.extension(), "pt");
}

// =========================================================================
// Quantization Type Tests
// =========================================================================

#[test]
fn test_quantization_type_debug() {
    let q = QuantizationType::Int8;
    let debug = format!("{:?}", q);
    assert!(debug.contains("Int8"));
}

#[test]
fn test_quantization_type_clone() {
    let q1 = QuantizationType::Int4;
    let q2 = q1.clone();
    assert_eq!(q1, q2);
}

#[test]
fn test_q4k_quantization_roundtrip() {
    // Test data: 512 f32 values (2 super-blocks of 256)
    // Use realistic weight distribution: centered around 0, mostly negative to positive
    let mut original: Vec<f32> = Vec::with_capacity(512);
    for i in 0..512 {
        // Simulate typical weight distribution: values mostly in [-0.1, 0.1]
        // with some outliers in [-0.3, 0.3]
        let base = ((i as f32) / 512.0 - 0.5) * 0.2; // -0.1 to 0.1
        let noise = (i as f32 * 0.1).sin() * 0.05;
        original.push(base + noise);
    }

    // Quantize to Q4K bytes
    let q4k_bytes = quantize_q4_k(&original);

    // Expected size: 2 super-blocks * 144 bytes each = 288 bytes
    assert_eq!(
        q4k_bytes.len(),
        288,
        "Q4K output should be 144 bytes per 256-element super-block"
    );

    // Dequantize back to f32
    let reconstructed = dequantize_q4_k_to_f32(&q4k_bytes, 512);
    assert_eq!(reconstructed.len(), 512);

    // Check reconstruction error
    let mut max_error = 0.0f32;
    let mut total_error = 0.0f32;
    for (orig, recon) in original.iter().zip(reconstructed.iter()) {
        let error = (orig - recon).abs();
        max_error = max_error.max(error);
        total_error += error;
    }
    let avg_error = total_error / 512.0;

    // Q4_K should have reasonable reconstruction quality for typical weights
    // With 4-bit quantization (15 levels) + nested 6-bit scale quantization,
    // max error is approximately: value_range * (1/15 + 1/63) ≈ range * 0.08
    // For our data range of ~0.3, max error ~0.024, but f16 quantization
    // of d/dmin adds additional error, so we allow up to 0.06
    assert!(
        max_error < 0.06,
        "Q4K max reconstruction error too high: {max_error}"
    );
    assert!(
        avg_error < 0.02,
        "Q4K avg reconstruction error too high: {avg_error}"
    );
}

#[test]
fn test_q4k_empty_data() {
    let empty: Vec<f32> = vec![];
    let q4k_bytes = quantize_q4_k(&empty);
    assert!(q4k_bytes.is_empty());

    let reconstructed = dequantize_q4_k_to_f32(&q4k_bytes, 0);
    assert!(reconstructed.is_empty());
}

#[test]
fn test_q4k_partial_block() {
    // Test with 100 elements (less than one 256-element super-block)
    let original: Vec<f32> = (0..100).map(|i| i as f32 * 0.01 - 0.5).collect();

    let q4k_bytes = quantize_q4_k(&original);
    // Should have 1 super-block (144 bytes) since we pad to 256
    assert_eq!(q4k_bytes.len(), 144);

    let reconstructed = dequantize_q4_k_to_f32(&q4k_bytes, 100);
    assert_eq!(reconstructed.len(), 100);

    // Verify reasonable reconstruction
    for (orig, recon) in original.iter().zip(reconstructed.iter()) {
        let error = (orig - recon).abs();
        assert!(error < 0.2, "Reconstruction error too high: {error}");
    }
}

#[test]
fn test_quantize_tensors_q4k() {
    let mut tensors = BTreeMap::new();
    tensors.insert(
        "test".to_string(),
        (
            (0..512).map(|i| i as f32 * 0.001 - 0.256).collect(),
            vec![512],
        ),
    );

    let native = NativeF32Tensors::new(tensors);
    let result = quantize_tensors(&native, &QuantizationType::Q4K).unwrap();
    let result = result.as_ref();

    assert_eq!(result.len(), 1);
    assert!(result.contains_key("test"));
    let (data, shape) = &result["test"];
    assert_eq!(shape, &vec![512]);
    assert_eq!(data.len(), 512); // Dequantized back to f32
}

// =========================================================================
// Compression Type Tests
// =========================================================================

#[test]
fn test_compression_debug() {
    let c = Compression::ZstdDefault;
    let debug = format!("{:?}", c);
    assert!(debug.contains("Zstd"));
}

#[test]
fn test_compression_clone() {
    let c1 = Compression::Lz4;
    let c2 = c1;
    assert_eq!(c1, c2);
}

// =========================================================================
// TensorExpectation Check Tests
// =========================================================================

#[test]
fn test_tensor_expectation_check_valid() {
    let exp = TensorExpectation::LAYER_NORM_WEIGHT;
    let stats = TensorStats {
        name: "layer_norm.weight".to_string(),
        count: 768,
        mean: 1.0,
        std: 0.1,
        min: 0.5,
        max: 1.5,
        nan_count: 0,
        inf_count: 0,
        zero_count: 0,
    };
    assert!(exp.check(&stats).is_ok());
}

#[test]
fn test_tensor_expectation_check_invalid_mean() {
    let exp = TensorExpectation::LAYER_NORM_WEIGHT;
    let stats = TensorStats {
        name: "layer_norm.weight".to_string(),
        count: 768,
        mean: 100.0, // Way outside expected range
        std: 0.1,
        min: 99.0,
        max: 101.0,
        nan_count: 0,
        inf_count: 0,
        zero_count: 0,
    };
    assert!(exp.check(&stats).is_err());
}

// =========================================================================
// TensorStats Creation Tests
// =========================================================================

#[test]
fn test_tensor_stats_fields() {
    let stats = TensorStats {
        name: "test.weight".to_string(),
        count: 100,
        mean: 0.5,
        std: 0.2,
        min: 0.0,
        max: 1.0,
        nan_count: 0,
        inf_count: 0,
        zero_count: 5,
    };
    assert!((stats.mean - 0.5).abs() < 1e-6);
    assert!((stats.std - 0.2).abs() < 1e-6);
    assert!((stats.min - 0.0).abs() < 1e-6);
    assert!((stats.max - 1.0).abs() < 1e-6);
    assert_eq!(stats.count, 100);
    assert_eq!(stats.zero_count, 5);
}

// =========================================================================
// Quantization and Internal Function Tests (Coverage Boost)
// =========================================================================

#[test]
fn test_calculate_tensor_size() {
    let mut tensors = BTreeMap::new();
    tensors.insert("a".to_string(), (vec![1.0f32; 100], vec![10, 10]));
    tensors.insert("b".to_string(), (vec![2.0f32; 50], vec![50]));
    let size = calculate_tensor_size(&tensors);
    // 100 * 4 + 50 * 4 = 600
    assert_eq!(size, 600);
}

#[test]
fn test_calculate_tensor_size_empty() {
    let tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
    assert_eq!(calculate_tensor_size(&tensors), 0);
}

#[test]
fn test_quantize_fp16_roundtrip() {
    let data = vec![1.0, 2.0, 3.0, -1.0, 0.0, 0.5];
    let quantized = quantize_fp16(&data);
    // Should preserve values with f16 precision
    assert_eq!(quantized.len(), data.len());
    for (orig, quant) in data.iter().zip(quantized.iter()) {
        // f16 has limited precision
        assert!((orig - quant).abs() < 0.01, "fp16 should preserve value");
    }
}

#[test]
fn test_quantize_fp16_large_values() {
    let data = vec![65504.0, -65504.0]; // max f16 values
    let quantized = quantize_fp16(&data);
    assert!((quantized[0] - 65504.0).abs() < 1.0);
}

#[test]
fn test_quantize_int8_roundtrip() {
    let data = vec![1.0, -1.0, 0.5, -0.5, 0.0, 0.25];
    let quantized = quantize_int8(&data);
    assert_eq!(quantized.len(), data.len());
    // int8 quantization scales to -127..127
    for (orig, quant) in data.iter().zip(quantized.iter()) {
        assert!(
            (orig - quant).abs() < 0.05,
            "int8 should preserve value within tolerance"
        );
    }
}

#[test]
fn test_quantize_int8_all_zeros() {
    let data = vec![0.0, 0.0, 0.0];
    let quantized = quantize_int8(&data);
    for v in &quantized {
        assert_eq!(*v, 0.0);
    }
}

#[test]
fn test_quantize_int4_roundtrip() {
    let data = vec![1.0, -1.0, 0.5, -0.5, 0.0, 0.25];
    let quantized = quantize_int4(&data);
    assert_eq!(quantized.len(), data.len());
    // int4 has only 16 levels so lower precision
    for (orig, quant) in data.iter().zip(quantized.iter()) {
        assert!(
            (orig - quant).abs() < 0.15,
            "int4 should preserve value within tolerance"
        );
    }
}

#[test]
fn test_quantize_int4_all_zeros() {
    let data = vec![0.0, 0.0, 0.0];
    let quantized = quantize_int4(&data);
    for v in &quantized {
        assert_eq!(*v, 0.0);
    }
}

#[test]
fn test_f16_to_f32_zero() {
    assert_eq!(f16_to_f32(0x0000), 0.0);
}

#[test]
fn test_f16_to_f32_one() {
    let result = f16_to_f32(0x3C00);
    assert!((result - 1.0).abs() < 0.001);
}

#[test]
fn test_f16_to_f32_negative() {
    let result = f16_to_f32(0xBC00);
    assert!((result + 1.0).abs() < 0.001);
}

#[test]
fn test_f16_to_f32_subnormal() {
    let result = f16_to_f32(0x0001);
    assert!(result > 0.0 && result < 0.001);
}

#[test]
fn test_f16_to_f32_max() {
    // Max f16 is 65504
    let result = f16_to_f32(0x7BFF);
    assert!((result - 65504.0).abs() < 1.0);
}

#[test]
fn test_convert_report_zero_sizes() {
    let report = ConvertReport {
        original_size: 0,
        converted_size: 0,
        tensor_count: 0,
        quantization: None,
        compression: None,
        reduction_ratio: 0.0,
    };
    assert_eq!(report.reduction_percent(), "N/A");
}

#[test]
fn test_convert_report_debug() {
    let report = ConvertReport {
        original_size: 1000,
        converted_size: 500,
        tensor_count: 10,
        quantization: Some(QuantizationType::Int8),
        compression: Some(Compression::Lz4),
        reduction_ratio: 2.0,
    };
    assert!(format!("{:?}", report).contains("ConvertReport"));
}

#[test]
fn test_quantize_tensors_fp16() {
    let mut tensors = BTreeMap::new();
    tensors.insert("w".to_string(), (vec![1.0, 2.0, 3.0], vec![3]));
    let native = NativeF32Tensors::new(tensors);
    let result = quantize_tensors(&native, &QuantizationType::Fp16).expect("quantize");
    assert!(result.as_ref().contains_key("w"));
}

#[test]
fn test_quantize_tensors_int8() {
    let mut tensors = BTreeMap::new();
    tensors.insert("w".to_string(), (vec![1.0, -1.0, 0.5], vec![3]));
    let native = NativeF32Tensors::new(tensors);
    let result = quantize_tensors(&native, &QuantizationType::Int8).expect("quantize");
    assert!(result.as_ref().contains_key("w"));
}

#[test]
fn test_quantize_tensors_int4() {
    let mut tensors = BTreeMap::new();
    tensors.insert("w".to_string(), (vec![0.5, -0.5, 0.0], vec![3]));
    let native = NativeF32Tensors::new(tensors);
    let result = quantize_tensors(&native, &QuantizationType::Int4).expect("quantize");
    assert!(result.as_ref().contains_key("w"));
}

#[test]
fn test_dequantize_q4k_to_f32_basic() {
    // Create a minimal Q4K block (144 bytes for 256 elements)
    let mut data = vec![0u8; 144];
    // Set d = 1.0 in f16 (0x3C00)
    data[0] = 0x00;
    data[1] = 0x3C;
    // Set dmin = 0.0
    data[2] = 0x00;
    data[3] = 0x00;
    let result = dequantize_q4_k_to_f32(&data, 256);
    assert_eq!(result.len(), 256);
}

#[test]
fn test_dequantize_q4k_to_f32_truncated() {
    // Data smaller than one block
    let data = vec![0u8; 50];
    let result = dequantize_q4_k_to_f32(&data, 256);
    // Should produce zero-filled result
    assert_eq!(result.len(), 256);
}

/// PMAT-177: Test that NaN/Inf scale factors are replaced with safe values
#[test]
fn test_dequantize_q4k_nan_inf_protection_pmat177() {
    // Create a Q4K block with NaN d value (f16 NaN = 0x7E00)
    let mut data = vec![0u8; 144];
    // Set d = NaN in f16 (0x7E00)
    data[0] = 0x00;
    data[1] = 0x7E;
    // Set dmin = Inf in f16 (0x7C00)
    data[2] = 0x00;
    data[3] = 0x7C;

    let result = dequantize_q4_k_to_f32(&data, 256);

    // PMAT-177: Result should contain NO NaN or Inf values
    let nan_count = result.iter().filter(|v| v.is_nan()).count();
    let inf_count = result.iter().filter(|v| v.is_infinite()).count();

    assert_eq!(
        nan_count, 0,
        "PMAT-177: dequantize_q4_k should not produce NaN"
    );
    assert_eq!(
        inf_count, 0,
        "PMAT-177: dequantize_q4_k should not produce Inf"
    );
}

/// PMAT-177: Test that subnormal f16 scales are clamped to zero
#[test]
fn test_dequantize_q4k_subnormal_protection_pmat177() {
    // Create a Q4K block with subnormal d value (f16 subnormal = 0x0001)
    let mut data = vec![0u8; 144];
    // Set d = subnormal in f16 (0x0001 - smallest subnormal)
    data[0] = 0x01;
    data[1] = 0x00;
    // Set dmin = 0.0
    data[2] = 0x00;
    data[3] = 0x00;

    let result = dequantize_q4_k_to_f32(&data, 256);

    // PMAT-177: Subnormal should be treated as zero, result should be all zeros
    let non_zero_count = result.iter().filter(|&&v| v != 0.0).count();
    assert_eq!(
        non_zero_count, 0,
        "PMAT-177: subnormal f16 scales should be clamped to zero"
    );
}

#[test]
fn test_calculate_merge_weights_average() {
    let options = MergeOptions {
        strategy: MergeStrategy::Average,
        weights: None,
    };
    let weights = calculate_merge_weights(3, &options).expect("weights");
    assert_eq!(weights.len(), 3);
    for w in &weights {
        assert!((*w - 1.0 / 3.0).abs() < 0.001);
    }
}

#[test]
fn test_calculate_merge_weights_custom() {
    let options = MergeOptions {
        strategy: MergeStrategy::Weighted,
        weights: Some(vec![0.5, 0.3, 0.2]),
    };
    let weights = calculate_merge_weights(3, &options).expect("weights");
    // Weighted merging always normalizes
    let sum: f32 = weights.iter().sum();
    assert!((sum - 1.0).abs() < 0.001);
}

#[test]
fn test_calculate_merge_weights_normalize() {
    let options = MergeOptions {
        strategy: MergeStrategy::Weighted,
        weights: Some(vec![2.0, 2.0, 1.0]),
    };
    let weights = calculate_merge_weights(3, &options).expect("weights");
    let sum: f32 = weights.iter().sum();
    assert!((sum - 1.0).abs() < 0.001);
    // Check relative proportions: 2:2:1
    assert!((weights[0] - 0.4).abs() < 0.001);
    assert!((weights[1] - 0.4).abs() < 0.001);
    assert!((weights[2] - 0.2).abs() < 0.001);
}

#[test]
fn test_calculate_merge_weights_zero_sum() {
    let options = MergeOptions {
        strategy: MergeStrategy::Weighted,
        weights: Some(vec![0.0, 0.0, 0.0]),
    };
    let result = calculate_merge_weights(3, &options);
    assert!(result.is_err());
}
