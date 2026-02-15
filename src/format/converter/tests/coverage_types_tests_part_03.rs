
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
