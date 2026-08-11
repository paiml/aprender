//! Tests for training batch loading (non-WASM only)

use crate::config::schema::{DataConfig, ModelRef, OptimSpec, TrainSpec, TrainingParams};
use crate::config::train::load_training_batches;
use std::collections::HashMap;
use tempfile::NamedTempFile;

#[test]
fn test_load_training_batches_missing_file() {
    let spec = TrainSpec {
        model: ModelRef { path: std::path::PathBuf::from("model.gguf"), ..Default::default() },
        data: DataConfig {
            train: std::path::PathBuf::from("/nonexistent/data.parquet"),
            batch_size: 4,
            ..Default::default()
        },
        optimizer: OptimSpec { name: "adam".to_string(), lr: 0.001, params: HashMap::new() },
        lora: None,
        quantize: None,
        merge: None,
        training: TrainingParams::default(),
        publish: None,
    };

    // A dataset that is not there is a hard error, never demo data.
    // (This test previously asserted is_ok(), locking the fabricated-data defect in.)
    let err = load_training_batches(&spec).expect_err("missing dataset must be rejected");
    let msg = err.to_string();
    assert!(msg.contains("/nonexistent/data.parquet"), "error must name the dataset: {msg}");
}

#[test]
fn test_load_training_batches_unsupported_extension() {
    let temp_file = NamedTempFile::with_suffix(".txt").expect("temp file creation should succeed");
    std::fs::write(temp_file.path(), "test data").expect("file write should succeed");

    let spec = TrainSpec {
        model: ModelRef { path: std::path::PathBuf::from("model.gguf"), ..Default::default() },
        data: DataConfig {
            train: temp_file.path().to_path_buf(),
            batch_size: 4,
            ..Default::default()
        },
        optimizer: OptimSpec { name: "adam".to_string(), lr: 0.001, params: HashMap::new() },
        lora: None,
        quantize: None,
        merge: None,
        training: TrainingParams::default(),
        publish: None,
    };

    // An unreadable format is a hard error naming the format and the alternatives.
    // (This test previously asserted is_ok(), locking the fabricated-data defect in.)
    let err = load_training_batches(&spec).expect_err("unsupported format must be rejected");
    let msg = err.to_string();
    assert!(msg.contains("txt"), "error must name the offending extension: {msg}");
    assert!(msg.contains("json"), "error must name a supported format: {msg}");
}

#[test]
fn test_load_training_batches_json() {
    let json = r#"[
    {"input": [1.0, 2.0, 3.0, 4.0], "target": [5.0, 6.0, 7.0, 8.0]},
    {"input": [9.0, 10.0, 11.0, 12.0], "target": [13.0, 14.0, 15.0, 16.0]}
]"#;
    let temp_file = NamedTempFile::with_suffix(".json").expect("temp file creation should succeed");
    std::fs::write(temp_file.path(), json).expect("file write should succeed");

    let spec = TrainSpec {
        model: ModelRef { path: std::path::PathBuf::from("model.gguf"), ..Default::default() },
        data: DataConfig {
            train: temp_file.path().to_path_buf(),
            batch_size: 1,
            ..Default::default()
        },
        optimizer: OptimSpec { name: "adam".to_string(), lr: 0.001, params: HashMap::new() },
        lora: None,
        quantize: None,
        merge: None,
        training: TrainingParams::default(),
        publish: None,
    };

    let result = load_training_batches(&spec);
    assert!(result.is_ok());
    let batches = result.expect("operation should succeed");
    assert_eq!(batches.len(), 2);
}
