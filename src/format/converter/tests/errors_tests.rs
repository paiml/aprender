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
    // All 5 strategies are supported
    assert!(MergeStrategy::Average.is_supported());
    assert!(MergeStrategy::Weighted.is_supported());
    assert!(MergeStrategy::Ties.is_supported());
    assert!(MergeStrategy::Dare.is_supported());
    assert!(MergeStrategy::Slerp.is_supported());
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
        ..Default::default()
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

include!("errors_tests_part_02.rs");
include!("errors_tests_part_03.rs");
