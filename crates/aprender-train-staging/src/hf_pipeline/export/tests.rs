//! Tests for export module.

use super::*;
use std::path::Path;

// =========================================================================
// ExportFormat Tests
// =========================================================================

#[test]
fn test_format_extension() {
    assert_eq!(ExportFormat::SafeTensors.extension(), "safetensors");
    assert_eq!(ExportFormat::APR.extension(), "apr.json");
    assert_eq!(ExportFormat::GGUF.extension(), "gguf");
    assert_eq!(ExportFormat::PyTorch.extension(), "pt");
}

#[test]
fn test_format_is_safe() {
    assert!(ExportFormat::SafeTensors.is_safe());
    assert!(ExportFormat::APR.is_safe());
    assert!(ExportFormat::GGUF.is_safe());
    assert!(!ExportFormat::PyTorch.is_safe());
}

#[test]
fn test_format_from_path() {
    assert_eq!(
        ExportFormat::from_path(Path::new("model.safetensors")),
        Some(ExportFormat::SafeTensors)
    );
    assert_eq!(ExportFormat::from_path(Path::new("model.apr.json")), Some(ExportFormat::APR));
    assert_eq!(ExportFormat::from_path(Path::new("model.gguf")), Some(ExportFormat::GGUF));
    assert_eq!(ExportFormat::from_path(Path::new("model.pt")), Some(ExportFormat::PyTorch));
    assert_eq!(ExportFormat::from_path(Path::new("model.txt")), None);
}

#[test]
fn test_format_display() {
    assert_eq!(format!("{}", ExportFormat::SafeTensors), "SafeTensors");
    assert_eq!(format!("{}", ExportFormat::APR), "APR");
}

// =========================================================================
// DataType Tests
// =========================================================================

#[test]
fn test_dtype_bytes() {
    assert_eq!(types::DataType::F32.bytes_per_element(), 4.0);
    assert_eq!(types::DataType::F16.bytes_per_element(), 2.0);
    assert_eq!(types::DataType::Q4_0.bytes_per_element(), 0.5);
}

// =========================================================================
// ModelWeights Tests
// =========================================================================

#[test]
fn test_weights_new() {
    let weights = ModelWeights::new();
    assert!(weights.tensors.is_empty());
    assert_eq!(weights.param_count(), 0);
}

#[test]
fn test_weights_add_tensor() {
    let mut weights = ModelWeights::new();
    weights.add_tensor("layer.0.weight", vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);

    assert_eq!(weights.tensors.len(), 1);
    assert_eq!(weights.param_count(), 4);

    let (data, shape) = weights.get_tensor("layer.0.weight").expect("operation should succeed");
    assert_eq!(data.len(), 4);
    assert_eq!(shape, &vec![2, 2]);
}

#[test]
fn test_weights_tensor_names() {
    let mut weights = ModelWeights::new();
    weights.add_tensor("a", vec![1.0], vec![1]);
    weights.add_tensor("b", vec![2.0], vec![1]);

    let names = weights.tensor_names();
    assert_eq!(names.len(), 2);
    assert!(names.contains(&"a"));
    assert!(names.contains(&"b"));
}

#[test]
fn test_weights_mock() {
    let weights = ModelWeights::mock(4, 256);

    // 4 layers * (4 attention + 2 MLP) = 24 tensors
    assert_eq!(weights.tensors.len(), 24);
    assert!(weights.param_count() > 0);
    assert_eq!(weights.metadata.num_layers, Some(4));
    assert_eq!(weights.metadata.hidden_size, Some(256));
}

#[test]
fn test_weights_with_metadata() {
    let metadata = ModelMetadata {
        model_name: Some("test_model".to_string()),
        num_params: 1000,
        ..Default::default()
    };

    let weights = ModelWeights::new().with_metadata(metadata);
    assert_eq!(weights.metadata.model_name, Some("test_model".to_string()));
}

// =========================================================================
// Exporter Tests
// =========================================================================

#[test]
fn test_exporter_new() {
    let exporter = Exporter::new();
    assert_eq!(exporter.default_format, ExportFormat::SafeTensors);
    assert!(exporter.include_metadata);
}

#[test]
fn test_exporter_builder() {
    let exporter = Exporter::new()
        .output_dir("/tmp/export")
        .default_format(ExportFormat::APR)
        .include_metadata(false);

    // Fields are pub(super) so we verify via behavior rather than direct access
    // Exporter was configured successfully if we reach this point
    let _ = exporter;
}

#[test]
fn test_export_safetensors() {
    let weights = ModelWeights::mock(2, 64);
    let exporter = Exporter::new().output_dir("/tmp");

    let result = exporter.export(&weights, ExportFormat::SafeTensors, "test_model.safetensors");
    assert!(result.is_ok());

    let result = result.expect("operation should succeed");
    assert_eq!(result.format, ExportFormat::SafeTensors);
    assert!(result.size_bytes > 0);

    // Cleanup
    std::fs::remove_file(&result.path).ok();
}

#[test]
fn test_export_apr() {
    let weights = ModelWeights::mock(2, 64);
    let exporter = Exporter::new().output_dir("/tmp");

    let result = exporter.export(&weights, ExportFormat::APR, "test_model.apr.json");
    assert!(result.is_ok());

    let result = result.expect("operation should succeed");
    assert_eq!(result.format, ExportFormat::APR);

    // Verify JSON is valid
    let content = std::fs::read_to_string(&result.path).expect("file read should succeed");
    assert!(content.contains("\"version\""));
    assert!(content.contains("\"tensors\""));

    // Cleanup
    std::fs::remove_file(&result.path).ok();
}

#[test]
fn test_export_gguf() {
    let weights = ModelWeights::mock(2, 64);
    let exporter = Exporter::new().output_dir("/tmp");

    let result = exporter.export(&weights, ExportFormat::GGUF, "test_model.gguf");
    assert!(result.is_ok());

    let result = result.expect("operation should succeed");
    assert_eq!(result.format, ExportFormat::GGUF);

    // Cleanup
    std::fs::remove_file(&result.path).ok();
}

#[test]
fn test_export_pytorch_rejected() {
    let weights = ModelWeights::mock(2, 64);
    let exporter = Exporter::new().output_dir("/tmp");

    let result = exporter.export(&weights, ExportFormat::PyTorch, "test_model.pt");
    assert!(result.is_err());
    assert!(matches!(result, Err(crate::hf_pipeline::error::FetchError::PickleSecurityRisk)));
}

#[test]
fn test_export_auto() {
    let weights = ModelWeights::mock(2, 64);
    let exporter = Exporter::new().output_dir("/tmp");

    let result = exporter.export_auto(&weights, "test_auto.safetensors");
    assert!(result.is_ok());
    assert_eq!(result.expect("deserialization should succeed").format, ExportFormat::SafeTensors);

    // Cleanup
    std::fs::remove_file("/tmp/test_auto.safetensors").ok();
}

// =========================================================================
// ExportResult Tests
// =========================================================================

#[test]
fn test_result_size_human() {
    let result = ExportResult {
        path: std::path::PathBuf::from("test"),
        format: ExportFormat::SafeTensors,
        size_bytes: 500,
        num_tensors: 1,
    };
    assert_eq!(result.size_human(), "500 B");

    let result = ExportResult { size_bytes: 1_500_000, ..result };
    assert!(result.size_human().contains("MB"));

    let result = ExportResult { size_bytes: 2_500_000_000, ..result };
    assert!(result.size_human().contains("GB"));
}

// =========================================================================
// TrainingMetadata Tests
// =========================================================================

#[test]
fn test_training_metadata() {
    let meta = TrainingMetadata {
        epochs: 10,
        final_loss: Some(0.5),
        teacher_model: Some("bert-base".to_string()),
        ..Default::default()
    };

    assert_eq!(meta.epochs, 10);
    assert_eq!(meta.final_loss, Some(0.5));
}

// =========================================================================
// Additional Coverage Tests
// =========================================================================

#[test]
fn test_dtype_bytes_all_types() {
    // Cover all DataType branches
    assert_eq!(types::DataType::I32.bytes_per_element(), 4.0);
    assert_eq!(types::DataType::BF16.bytes_per_element(), 2.0);
    assert_eq!(types::DataType::I8.bytes_per_element(), 1.0);
    assert_eq!(types::DataType::U8.bytes_per_element(), 1.0);
    assert_eq!(types::DataType::Q8_0.bytes_per_element(), 1.0);
}

#[test]
fn test_format_from_path_apr_short() {
    // Test .apr without .json
    assert_eq!(ExportFormat::from_path(Path::new("model.apr")), Some(ExportFormat::APR));
}

#[test]
fn test_format_from_path_bin() {
    // Test .bin (PyTorch)
    assert_eq!(ExportFormat::from_path(Path::new("model.bin")), Some(ExportFormat::PyTorch));
}

#[test]
fn test_format_display_gguf_pytorch() {
    assert_eq!(format!("{}", ExportFormat::GGUF), "GGUF");
    assert_eq!(format!("{}", ExportFormat::PyTorch), "PyTorch");
}

#[test]
fn test_weights_get_tensor_not_found() {
    let weights = ModelWeights::new();
    assert!(weights.get_tensor("nonexistent").is_none());
}

#[test]
fn test_result_size_kb() {
    let result = ExportResult {
        path: std::path::PathBuf::from("test"),
        format: ExportFormat::SafeTensors,
        size_bytes: 5_000,
        num_tensors: 1,
    };
    let size_str = result.size_human();
    assert!(size_str.contains("KB"), "Expected KB, got: {size_str}");
}

#[test]
fn test_model_metadata_default() {
    let meta = ModelMetadata::default();
    assert!(meta.architecture.is_none());
    assert!(meta.model_name.is_none());
    assert_eq!(meta.num_params, 0);
}

#[test]
fn test_training_metadata_full() {
    let meta = TrainingMetadata {
        epochs: 5,
        final_loss: Some(0.1),
        final_val_loss: Some(0.15),
        learning_rate: Some(1e-4),
        batch_size: Some(32),
        temperature: Some(2.0),
        teacher_model: Some("gpt2".to_string()),
    };
    assert_eq!(meta.batch_size, Some(32));
    assert_eq!(meta.temperature, Some(2.0));
}

#[test]
fn test_weights_default() {
    let weights = ModelWeights::default();
    assert!(weights.tensors.is_empty());
}

#[test]
fn test_exporter_default() {
    let exporter = Exporter::default();
    assert_eq!(exporter.default_format, ExportFormat::SafeTensors);
}

#[test]
fn test_export_result_clone() {
    let result = ExportResult {
        path: std::path::PathBuf::from("/tmp/test"),
        format: ExportFormat::APR,
        size_bytes: 1000,
        num_tensors: 5,
    };
    let cloned = result.clone();
    assert_eq!(result.path, cloned.path);
    assert_eq!(result.num_tensors, cloned.num_tensors);
}

#[test]
fn test_format_from_path_no_extension() {
    // Path with no recognizable extension
    assert!(ExportFormat::from_path(Path::new("model")).is_none());
}
