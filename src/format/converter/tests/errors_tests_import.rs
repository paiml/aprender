use super::*;

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
