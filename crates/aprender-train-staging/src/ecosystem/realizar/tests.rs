//! Tests for Realizar GGUF export integration.

use super::*;

#[test]
fn test_quantization_type_as_str() {
    assert_eq!(QuantizationType::Q4KM.as_str(), "Q4_K_M");
    assert_eq!(QuantizationType::Q5KM.as_str(), "Q5_K_M");
    assert_eq!(QuantizationType::Q80.as_str(), "Q8_0");
    assert_eq!(QuantizationType::F16.as_str(), "F16");
    assert_eq!(QuantizationType::F32.as_str(), "F32");
}

#[test]
fn test_quantization_type_bits() {
    assert!((QuantizationType::Q4KM.bits_per_weight() - 4.5).abs() < f32::EPSILON);
    assert!((QuantizationType::Q80.bits_per_weight() - 8.0).abs() < f32::EPSILON);
    assert!((QuantizationType::F16.bits_per_weight() - 16.0).abs() < f32::EPSILON);
}

#[test]
fn test_quantization_type_quality() {
    assert!(QuantizationType::Q2K.quality_score() < QuantizationType::Q4KM.quality_score());
    assert!(QuantizationType::Q4KM.quality_score() < QuantizationType::Q80.quality_score());
    assert!(QuantizationType::Q80.quality_score() < QuantizationType::F16.quality_score());
}

#[test]
fn test_quantization_type_estimate_size() {
    let original = 1_000_000_000u64; // 1GB (F32 weights)

    let q4_size = QuantizationType::Q4KM.estimate_size(original);
    let f16_size = QuantizationType::F16.estimate_size(original);

    assert!(q4_size < f16_size);
    assert!(f16_size < original);
}

#[test]
fn test_quantization_type_parse() {
    assert_eq!(QuantizationType::parse("Q4_K_M"), Some(QuantizationType::Q4KM));
    assert_eq!(QuantizationType::parse("q4km"), Some(QuantizationType::Q4KM));
    assert_eq!(QuantizationType::parse("F16"), Some(QuantizationType::F16));
    assert_eq!(QuantizationType::parse("fp16"), Some(QuantizationType::F16));
    assert_eq!(QuantizationType::parse("invalid"), None);
}

#[test]
fn test_experiment_provenance_creation() {
    let prov = ExperimentProvenance::new("exp-001", "run-123")
        .with_config_hash("abc123")
        .with_dataset("imagenet-1k")
        .with_base_model("llama-7b")
        .with_metric("loss", 0.125)
        .with_metric("accuracy", 0.92)
        .with_git_commit("deadbeef")
        .with_custom("framework", "entrenar");

    assert_eq!(prov.experiment_id, "exp-001");
    assert_eq!(prov.run_id, "run-123");
    assert_eq!(prov.config_hash, "abc123");
    assert_eq!(prov.dataset_id, Some("imagenet-1k".to_string()));
    assert_eq!(prov.base_model_id, Some("llama-7b".to_string()));
    assert_eq!(prov.metrics.get("loss"), Some(&0.125));
    assert_eq!(prov.metrics.get("accuracy"), Some(&0.92));
    assert_eq!(prov.git_commit, Some("deadbeef".to_string()));
    assert_eq!(prov.custom.get("framework"), Some(&"entrenar".to_string()));
}

#[test]
fn test_experiment_provenance_to_metadata() {
    let prov = ExperimentProvenance::new("exp-001", "run-123").with_metric("loss", 0.125);

    let pairs = prov.to_metadata_pairs();

    assert!(pairs.iter().any(|(k, v)| k == "entrenar.experiment_id" && v == "exp-001"));
    assert!(pairs.iter().any(|(k, v)| k == "entrenar.run_id" && v == "run-123"));
    assert!(pairs.iter().any(|(k, _)| k == "entrenar.timestamp"));
    assert!(pairs.iter().any(|(k, _)| k == "entrenar.metric.loss"));
}

#[test]
fn test_general_metadata_creation() {
    let general = GeneralMetadata::new("llama", "my-model")
        .with_author("PAIML")
        .with_description("Fine-tuned LLaMA model")
        .with_license("MIT");

    assert_eq!(general.architecture, "llama");
    assert_eq!(general.name, "my-model");
    assert_eq!(general.author, Some("PAIML".to_string()));
    assert_eq!(general.description, Some("Fine-tuned LLaMA model".to_string()));
    assert_eq!(general.license, Some("MIT".to_string()));
}

#[test]
fn test_gguf_exporter_creation() {
    let exporter = GgufExporter::new(QuantizationType::Q5KM).with_threads(8).without_validation();

    assert_eq!(exporter.quantization(), QuantizationType::Q5KM);
}

#[test]
fn test_gguf_exporter_with_provenance() {
    let prov = ExperimentProvenance::new("exp-001", "run-123");
    let general = GeneralMetadata::new("llama", "test-model");

    let exporter =
        GgufExporter::new(QuantizationType::Q4KM).with_general(general).with_provenance(prov);

    assert!(exporter.metadata().provenance.is_some());
    assert_eq!(exporter.metadata().general.architecture, "llama");
}

#[test]
fn test_gguf_exporter_collect_metadata() {
    let prov = ExperimentProvenance::new("exp-001", "run-123").with_metric("loss", 0.1);
    let general = GeneralMetadata::new("llama", "test-model").with_author("PAIML");

    let exporter =
        GgufExporter::new(QuantizationType::Q4KM).with_general(general).with_provenance(prov);

    let pairs = exporter.collect_metadata();

    assert!(pairs.iter().any(|(k, v)| k == "general.architecture" && v == "llama"));
    assert!(pairs.iter().any(|(k, v)| k == "general.name" && v == "test-model"));
    assert!(pairs.iter().any(|(k, v)| k == "general.author" && v == "PAIML"));
    assert!(pairs.iter().any(|(k, v)| k == "general.file_type" && v == "Q4_K_M"));
    assert!(pairs.iter().any(|(k, _)| k == "entrenar.experiment_id"));
}

#[test]
fn test_gguf_export_result() {
    let result = GgufExportResult {
        output_path: std::path::PathBuf::from("/tmp/model.gguf"),
        quantization: QuantizationType::Q4KM,
        metadata_keys: 10,
        estimated_size_bytes: 4_000_000_000,
    };

    assert_eq!(result.quantization, QuantizationType::Q4KM);
    assert_eq!(result.metadata_keys, 10);
}

#[test]
fn test_quantization_type_display() {
    assert_eq!(format!("{}", QuantizationType::Q4KM), "Q4_K_M");
    assert_eq!(format!("{}", QuantizationType::Q5KM), "Q5_K_M");
    assert_eq!(format!("{}", QuantizationType::Q80), "Q8_0");
    assert_eq!(format!("{}", QuantizationType::F16), "F16");
    assert_eq!(format!("{}", QuantizationType::F32), "F32");
    assert_eq!(format!("{}", QuantizationType::Q2K), "Q2_K");
    assert_eq!(format!("{}", QuantizationType::Q3KM), "Q3_K_M");
    assert_eq!(format!("{}", QuantizationType::Q6K), "Q6_K");
}

#[test]
fn test_quantization_type_all_bits() {
    assert!((QuantizationType::Q2K.bits_per_weight() - 2.5).abs() < f32::EPSILON);
    assert!((QuantizationType::Q3KM.bits_per_weight() - 3.5).abs() < f32::EPSILON);
    assert!((QuantizationType::Q5KM.bits_per_weight() - 5.5).abs() < f32::EPSILON);
    assert!((QuantizationType::Q6K.bits_per_weight() - 6.5).abs() < f32::EPSILON);
    assert!((QuantizationType::F32.bits_per_weight() - 32.0).abs() < f32::EPSILON);
}

#[test]
fn test_quantization_type_all_quality() {
    assert_eq!(QuantizationType::Q2K.quality_score(), 50);
    assert_eq!(QuantizationType::Q3KM.quality_score(), 65);
    assert_eq!(QuantizationType::Q4KM.quality_score(), 78);
    assert_eq!(QuantizationType::Q5KM.quality_score(), 85);
    assert_eq!(QuantizationType::Q6K.quality_score(), 92);
    assert_eq!(QuantizationType::Q80.quality_score(), 97);
    assert_eq!(QuantizationType::F16.quality_score(), 100);
    assert_eq!(QuantizationType::F32.quality_score(), 100);
}

#[test]
fn test_quantization_type_parse_all_variants() {
    assert_eq!(QuantizationType::parse("Q2K"), Some(QuantizationType::Q2K));
    assert_eq!(QuantizationType::parse("Q2"), Some(QuantizationType::Q2K));
    assert_eq!(QuantizationType::parse("Q3KM"), Some(QuantizationType::Q3KM));
    assert_eq!(QuantizationType::parse("Q3K"), Some(QuantizationType::Q3KM));
    assert_eq!(QuantizationType::parse("Q5KM"), Some(QuantizationType::Q5KM));
    assert_eq!(QuantizationType::parse("Q5K"), Some(QuantizationType::Q5KM));
    assert_eq!(QuantizationType::parse("Q6K"), Some(QuantizationType::Q6K));
    assert_eq!(QuantizationType::parse("Q6"), Some(QuantizationType::Q6K));
    assert_eq!(QuantizationType::parse("Q80"), Some(QuantizationType::Q80));
    assert_eq!(QuantizationType::parse("Q8"), Some(QuantizationType::Q80));
    assert_eq!(QuantizationType::parse("FP32"), Some(QuantizationType::F32));
}

#[test]
fn test_gguf_export_error_display() {
    let err = GgufExportError::InvalidQuantization("bad config".to_string());
    assert!(format!("{err}").contains("Invalid quantization"));
    assert!(format!("{err}").contains("bad config"));

    let err = GgufExportError::ValidationFailed("model check failed".to_string());
    assert!(format!("{err}").contains("validation failed"));

    let err = GgufExportError::IoError("write error".to_string());
    assert!(format!("{err}").contains("I/O error"));

    let err = GgufExportError::UnsupportedArchitecture("unknown_arch".to_string());
    assert!(format!("{err}").contains("Unsupported architecture"));

    let err = GgufExportError::MetadataError("serialization failed".to_string());
    assert!(format!("{err}").contains("Metadata error"));
}

#[test]
fn test_experiment_provenance_with_metrics_iterator() {
    let metrics = vec![
        ("loss".to_string(), 0.1),
        ("accuracy".to_string(), 0.95),
        ("perplexity".to_string(), 12.5),
    ];

    let prov = ExperimentProvenance::new("exp-001", "run-123").with_metrics(metrics);

    assert_eq!(prov.metrics.len(), 3);
    assert_eq!(prov.metrics.get("loss"), Some(&0.1));
    assert_eq!(prov.metrics.get("accuracy"), Some(&0.95));
    assert_eq!(prov.metrics.get("perplexity"), Some(&12.5));
}

#[test]
fn test_experiment_provenance_to_metadata_with_all_fields() {
    let prov = ExperimentProvenance::new("exp-001", "run-123")
        .with_config_hash("abc123")
        .with_dataset("dataset-id")
        .with_base_model("base-model")
        .with_git_commit("deadbeef")
        .with_metric("loss", 0.1)
        .with_custom("key", "value");

    let pairs = prov.to_metadata_pairs();

    assert!(pairs.iter().any(|(k, v)| k == "entrenar.config_hash" && v == "abc123"));
    assert!(pairs.iter().any(|(k, v)| k == "entrenar.dataset_id" && v == "dataset-id"));
    assert!(pairs.iter().any(|(k, v)| k == "entrenar.base_model_id" && v == "base-model"));
    assert!(pairs.iter().any(|(k, v)| k == "entrenar.git_commit" && v == "deadbeef"));
    assert!(pairs.iter().any(|(k, _)| k == "entrenar.metric.loss"));
    assert!(pairs.iter().any(|(k, v)| k == "entrenar.custom.key" && v == "value"));
}

#[test]
fn test_gguf_exporter_default() {
    let exporter = GgufExporter::default();
    assert_eq!(exporter.quantization(), QuantizationType::Q4KM);
}

#[test]
fn test_gguf_exporter_with_metadata() {
    let mut metadata = GgufMetadata::default();
    metadata.custom.insert("key".to_string(), "value".to_string());

    let exporter = GgufExporter::new(QuantizationType::F16).with_metadata(metadata);

    assert_eq!(exporter.metadata().custom.get("key"), Some(&"value".to_string()));
}

#[test]
fn test_gguf_exporter_export_missing_parent_dir() {
    let exporter = GgufExporter::new(QuantizationType::Q4KM);
    let result = exporter.export("input.gguf", "/nonexistent/dir/output.gguf");
    assert!(result.is_err());
    match result {
        Err(GgufExportError::IoError(msg)) => {
            assert!(msg.contains("does not exist"));
        }
        _ => panic!("Expected IoError"),
    }
}

#[test]
fn test_gguf_exporter_export_success() {
    let prov = ExperimentProvenance::new("exp-001", "run-123").with_metric("loss", 0.1);
    let exporter = GgufExporter::new(QuantizationType::Q4KM).with_provenance(prov);

    // Export to a valid directory (current dir)
    let result = exporter.export("input.gguf", "./output.gguf");
    assert!(result.is_ok());

    let res = result.expect("operation should succeed");
    assert_eq!(res.quantization, QuantizationType::Q4KM);
    assert!(res.metadata_keys > 0);
}

#[test]
fn test_gguf_exporter_collect_metadata_with_all_fields() {
    let prov = ExperimentProvenance::new("exp-001", "run-123").with_metric("loss", 0.1);
    let general = GeneralMetadata::new("llama", "test-model")
        .with_author("PAIML")
        .with_description("Test description")
        .with_license("MIT");

    let mut metadata = GgufMetadata::default();
    metadata.general = general;
    metadata.general.url = Some("https://example.com".to_string());
    metadata.provenance = Some(prov);
    metadata.custom.insert("custom_key".to_string(), "custom_value".to_string());

    let exporter = GgufExporter::new(QuantizationType::Q4KM).with_metadata(metadata);

    let pairs = exporter.collect_metadata();

    assert!(pairs.iter().any(|(k, v)| k == "general.description" && v == "Test description"));
    assert!(pairs.iter().any(|(k, v)| k == "general.license" && v == "MIT"));
    assert!(pairs.iter().any(|(k, v)| k == "general.url" && v == "https://example.com"));
    assert!(pairs.iter().any(|(k, v)| k == "custom.custom_key" && v == "custom_value"));
}

#[test]
fn test_quantization_type_serde() {
    let q = QuantizationType::Q4KM;
    let json = serde_json::to_string(&q).expect("JSON serialization should succeed");
    let parsed: QuantizationType =
        serde_json::from_str(&json).expect("JSON deserialization should succeed");
    assert_eq!(q, parsed);
}

#[test]
fn test_experiment_provenance_serde() {
    let prov = ExperimentProvenance::new("exp-001", "run-123")
        .with_metric("loss", 0.1)
        .with_custom("key", "value");

    let json = serde_json::to_string(&prov).expect("JSON serialization should succeed");
    let parsed: ExperimentProvenance =
        serde_json::from_str(&json).expect("JSON deserialization should succeed");

    assert_eq!(prov.experiment_id, parsed.experiment_id);
    assert_eq!(prov.run_id, parsed.run_id);
    assert_eq!(prov.metrics.get("loss"), parsed.metrics.get("loss"));
}

#[test]
fn test_gguf_metadata_serde() {
    let mut metadata = GgufMetadata::default();
    metadata.general = GeneralMetadata::new("llama", "test");
    metadata.custom.insert("key".to_string(), "value".to_string());

    let json = serde_json::to_string(&metadata).expect("JSON serialization should succeed");
    let parsed: GgufMetadata =
        serde_json::from_str(&json).expect("JSON deserialization should succeed");

    assert_eq!(metadata.general.architecture, parsed.general.architecture);
    assert_eq!(metadata.custom.get("key"), parsed.custom.get("key"));
}

#[test]
fn test_general_metadata_default() {
    let general = GeneralMetadata::default();
    assert!(general.architecture.is_empty());
    assert!(general.name.is_empty());
    assert!(general.author.is_none());
    assert!(general.description.is_none());
    assert!(general.license.is_none());
    assert!(general.url.is_none());
    assert!(general.file_type.is_none());
}

#[test]
fn test_gguf_metadata_default() {
    let metadata = GgufMetadata::default();
    assert!(metadata.provenance.is_none());
    assert!(metadata.custom.is_empty());
}

#[test]
fn test_quantization_type_hash() {
    use std::collections::HashSet;
    let mut set = HashSet::new();
    set.insert(QuantizationType::Q4KM);
    set.insert(QuantizationType::Q5KM);
    set.insert(QuantizationType::Q4KM); // Duplicate

    assert_eq!(set.len(), 2);
}
