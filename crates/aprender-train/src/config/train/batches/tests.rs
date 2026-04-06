//! Tests for batch loading

use super::*;
use crate::train::Batch;
use crate::Tensor;

#[cfg(not(target_arch = "wasm32"))]
use std::io::Write;
#[cfg(not(target_arch = "wasm32"))]
use tempfile::NamedTempFile;

fn make_batch(input: Vec<f32>, target: Vec<f32>) -> Batch {
    Batch::new(Tensor::from_vec(input, false), Tensor::from_vec(target, false))
}

#[test]
fn test_rebatch_empty() {
    let batches: Vec<Batch> = Vec::new();
    let result = rebatch(batches, 4);
    assert!(result.is_empty());
}

#[test]
fn test_rebatch_single_batch() {
    // 4 inputs with input_dim=2 means 2 examples
    // batch_size=2 -> 1 batch
    let batches = vec![make_batch(vec![1.0, 2.0, 3.0, 4.0], vec![0.1, 0.2])];
    let result = rebatch(batches, 2);
    assert_eq!(result.len(), 1);
}

#[test]
fn test_rebatch_multiple_batches() {
    // 3 batches, each with input_dim=2, total 6 inputs = 3 examples
    // batch_size=2 -> 2 batches (2 + 1)
    let batches = vec![
        make_batch(vec![1.0, 2.0], vec![0.1]),
        make_batch(vec![3.0, 4.0], vec![0.2]),
        make_batch(vec![5.0, 6.0], vec![0.3]),
    ];
    let result = rebatch(batches, 2);
    assert_eq!(result.len(), 2); // 3 examples -> 2 batches
}

#[test]
fn test_rebatch_batch_size_one() {
    // 3 inputs with input_dim=3 means 1 example
    // batch_size=1 -> 1 batch
    let batches = vec![make_batch(vec![1.0, 2.0, 3.0], vec![0.1, 0.2, 0.3])];
    let result = rebatch(batches, 1);
    assert_eq!(result.len(), 1);
}

#[test]
fn test_rebatch_large_batch_size() {
    let batches = vec![make_batch(vec![1.0], vec![0.1]), make_batch(vec![2.0], vec![0.2])];
    let result = rebatch(batches, 100);
    assert_eq!(result.len(), 1); // All fit in one batch
}

#[test]
fn test_load_training_batches_nonexistent_file() {
    use crate::config::schema::TrainSpec;
    use std::path::PathBuf;

    let spec = TrainSpec {
        model: crate::config::ModelRef {
            path: PathBuf::from("model.bin"),
            layers: vec![],
            ..Default::default()
        },
        data: crate::config::DataConfig {
            train: PathBuf::from("/nonexistent/path/data.parquet"),
            val: None,
            batch_size: 8,
            auto_infer_types: false,
            seq_len: None,
            ..Default::default()
        },
        optimizer: crate::config::OptimSpec {
            name: "adam".to_string(),
            lr: 0.001,
            params: Default::default(),
        },
        lora: None,
        quantize: None,
        merge: None,
        training: Default::default(),
        publish: None,
    };

    let result = load_training_batches(&spec);
    assert!(result.is_ok());
    let batches = result.expect("operation should succeed");
    assert!(!batches.is_empty()); // Should return demo batches
}

#[test]
#[cfg(not(target_arch = "wasm32"))]
fn test_load_training_batches_unsupported_format() {
    use crate::config::schema::TrainSpec;
    use std::path::PathBuf;

    // Create a temp file with unsupported extension
    let mut temp_file =
        NamedTempFile::with_suffix(".xyz").expect("temp file creation should succeed");
    writeln!(temp_file, "test data").expect("file write should succeed");

    let spec = TrainSpec {
        model: crate::config::ModelRef {
            path: PathBuf::from("model.bin"),
            layers: vec![],
            ..Default::default()
        },
        data: crate::config::DataConfig {
            train: temp_file.path().to_path_buf(),
            val: None,
            batch_size: 8,
            auto_infer_types: false,
            seq_len: None,
            ..Default::default()
        },
        optimizer: crate::config::OptimSpec {
            name: "adam".to_string(),
            lr: 0.001,
            params: Default::default(),
        },
        lora: None,
        quantize: None,
        merge: None,
        training: Default::default(),
        publish: None,
    };

    let result = load_training_batches(&spec);
    assert!(result.is_ok());
    let batches = result.expect("operation should succeed");
    assert!(!batches.is_empty()); // Should return demo batches
}

#[test]
#[cfg(not(target_arch = "wasm32"))]
fn test_load_json_batches_structured_format() {
    let mut temp_file =
        NamedTempFile::with_suffix(".json").expect("temp file creation should succeed");
    writeln!(
        temp_file,
        r#"{{
            "examples": [
                {{"input": [1.0, 2.0], "target": [0.1]}},
                {{"input": [3.0, 4.0], "target": [0.2]}}
            ]
        }}"#
    )
    .expect("operation should succeed");

    let result = load_json_batches(temp_file.path(), 2);
    assert!(result.is_ok());
    let batches = result.expect("operation should succeed");
    assert!(!batches.is_empty());
}

#[test]
#[cfg(not(target_arch = "wasm32"))]
fn test_load_json_batches_array_format() {
    let mut temp_file =
        NamedTempFile::with_suffix(".json").expect("temp file creation should succeed");
    writeln!(
        temp_file,
        r#"[
            {{"input": [1.0, 2.0], "target": [0.1]}},
            {{"input": [3.0, 4.0], "target": [0.2]}},
            {{"input": [5.0, 6.0], "target": [0.3]}}
        ]"#
    )
    .expect("operation should succeed");

    let result = load_json_batches(temp_file.path(), 2);
    assert!(result.is_ok());
    let batches = result.expect("operation should succeed");
    assert!(!batches.is_empty());
}

#[test]
#[cfg(not(target_arch = "wasm32"))]
fn test_load_json_batches_invalid_format() {
    let mut temp_file =
        NamedTempFile::with_suffix(".json").expect("temp file creation should succeed");
    writeln!(temp_file, r#"{{"some": "other", "format": true}}"#)
        .expect("file write should succeed");

    let result = load_json_batches(temp_file.path(), 2);
    assert!(result.is_ok());
    // Should fall back to demo batches
    let batches = result.expect("operation should succeed");
    assert!(!batches.is_empty());
}

#[test]
#[cfg(not(target_arch = "wasm32"))]
fn test_load_json_batches_file_not_found() {
    use std::path::Path;
    let result = load_json_batches(Path::new("/nonexistent/file.json"), 2);
    assert!(result.is_err());
}

#[test]
#[cfg(not(target_arch = "wasm32"))]
fn test_load_json_batches_batch_size_one() {
    let mut temp_file =
        NamedTempFile::with_suffix(".json").expect("temp file creation should succeed");
    writeln!(
        temp_file,
        r#"[
            {{"input": [1.0], "target": [0.1]}},
            {{"input": [2.0], "target": [0.2]}}
        ]"#
    )
    .expect("operation should succeed");

    let result = load_json_batches(temp_file.path(), 1);
    assert!(result.is_ok());
    let batches = result.expect("operation should succeed");
    assert_eq!(batches.len(), 2);
}

#[test]
#[cfg(not(target_arch = "wasm32"))]
fn test_load_json_batches_batch_size_zero() {
    let mut temp_file =
        NamedTempFile::with_suffix(".json").expect("temp file creation should succeed");
    writeln!(temp_file, r#"[{{"input": [1.0], "target": [0.1]}}]"#)
        .expect("file write should succeed");

    let result = load_json_batches(temp_file.path(), 0);
    assert!(result.is_ok());
    // batch_size.max(1) ensures at least 1
}

#[test]
#[cfg(not(target_arch = "wasm32"))]
fn test_load_training_batches_json_file() {
    use crate::config::schema::TrainSpec;
    use std::path::PathBuf;

    let mut temp_file =
        NamedTempFile::with_suffix(".json").expect("temp file creation should succeed");
    writeln!(temp_file, r#"[{{"input": [1.0, 2.0], "target": [0.5]}}]"#)
        .expect("file write should succeed");

    let spec = TrainSpec {
        model: crate::config::ModelRef {
            path: PathBuf::from("model.bin"),
            layers: vec![],
            ..Default::default()
        },
        data: crate::config::DataConfig {
            train: temp_file.path().to_path_buf(),
            val: None,
            batch_size: 4,
            auto_infer_types: false,
            seq_len: None,
            ..Default::default()
        },
        optimizer: crate::config::OptimSpec {
            name: "adam".to_string(),
            lr: 0.001,
            params: Default::default(),
        },
        lora: None,
        quantize: None,
        merge: None,
        training: Default::default(),
        publish: None,
    };

    let result = load_training_batches(&spec);
    assert!(result.is_ok());
}

#[test]
fn test_rebatch_preserves_data() {
    // 2 batches, each with input_dim=2, total 4 inputs = 2 examples
    let batches =
        vec![make_batch(vec![1.0, 2.0], vec![10.0]), make_batch(vec![3.0, 4.0], vec![20.0])];

    // batch_size=1, 2 examples -> 2 batches
    let result = rebatch(batches, 1);
    assert_eq!(result.len(), 2);

    // Check data is preserved
    let all_inputs: Vec<f32> =
        result.iter().flat_map(|b| b.inputs.data().iter().copied()).collect();
    assert_eq!(all_inputs, vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_rebatch_exact_batch_size() {
    // 4 batches, each with input_dim=1, total 4 inputs = 4 examples
    // batch_size=2 -> 2 batches
    let batches = vec![
        make_batch(vec![1.0], vec![0.1]),
        make_batch(vec![2.0], vec![0.2]),
        make_batch(vec![3.0], vec![0.3]),
        make_batch(vec![4.0], vec![0.4]),
    ];
    let result = rebatch(batches, 2);
    assert_eq!(result.len(), 2);
}

#[test]
fn test_rebatch_remainder() {
    // 5 batches, each with input_dim=1, total 5 inputs = 5 examples
    // batch_size=2 -> 3 batches (2 + 2 + 1)
    let batches = vec![
        make_batch(vec![1.0], vec![0.1]),
        make_batch(vec![2.0], vec![0.2]),
        make_batch(vec![3.0], vec![0.3]),
        make_batch(vec![4.0], vec![0.4]),
        make_batch(vec![5.0], vec![0.5]),
    ];
    let result = rebatch(batches, 2);
    assert_eq!(result.len(), 3);
}
