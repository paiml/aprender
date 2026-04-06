//! ENT-118: End-to-end LLM training integration tests
//!
//! Tests the complete LLM training pipeline from YAML config to model output.

use entrenar::config::train_from_yaml;
use std::fs;
use std::io::Write;
use tempfile::{NamedTempFile, TempDir};

/// Create a dummy SafeTensors file (minimal valid header)
fn create_dummy_safetensors(dir: &TempDir, name: &str) -> std::path::PathBuf {
    let path = dir.path().join(name);
    // Minimal SafeTensors format: 8 bytes header length + empty JSON header
    let header = b"{}";
    let header_len = (header.len() as u64).to_le_bytes();
    let mut content = Vec::new();
    content.extend_from_slice(&header_len);
    content.extend_from_slice(header);
    fs::write(&path, content).expect("file write should succeed");
    path
}

/// Create a minimal model config.json file for testing
fn create_minimal_config(config_dir: &TempDir) -> std::path::PathBuf {
    let config_path = config_dir.path().join("config.json");
    fs::write(
        &config_path,
        r#"{
            "hidden_size": 32,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "intermediate_size": 64,
            "num_hidden_layers": 1,
            "vocab_size": 100,
            "max_position_embeddings": 64,
            "rms_norm_eps": 1e-6,
            "rope_theta": 10000.0,
            "attention_bias": false
        }"#,
    )
    .expect("operation should succeed");
    config_path
}

/// Create pre-tokenized training data
fn create_pretokenized_data(data_dir: &TempDir) -> std::path::PathBuf {
    let data_path = data_dir.path().join("train.json");
    fs::write(
        &data_path,
        r#"{
            "examples": [
                {"input_ids": [1, 2, 3, 4, 5, 6, 7, 8]},
                {"input_ids": [10, 11, 12, 13, 14, 15, 16, 17]},
                {"input_ids": [20, 21, 22, 23, 24, 25, 26, 27]},
                {"input_ids": [30, 31, 32, 33, 34, 35, 36, 37]}
            ]
        }"#,
    )
    .expect("operation should succeed");
    data_path
}

/// Test: Complete transformer training pipeline
#[test]
fn test_e2e_transformer_training() {
    let output_dir = TempDir::new().expect("temp file creation should succeed");
    let config_dir = TempDir::new().expect("temp file creation should succeed");
    let data_dir = TempDir::new().expect("temp file creation should succeed");
    let model_dir = TempDir::new().expect("temp file creation should succeed");

    let config_path = create_minimal_config(&config_dir);
    let data_path = create_pretokenized_data(&data_dir);
    let model_path = create_dummy_safetensors(&model_dir, "model.safetensors");

    let yaml = format!(
        r#"
model:
  path: "{}"
  mode: transformer
  config: "{}"

data:
  train: "{}"
  batch_size: 2
  seq_len: 8

optimizer:
  name: adam
  lr: 0.01

training:
  epochs: 3
  mode: causal_lm
  output_dir: "{}"
"#,
        model_path.display(),
        config_path.display(),
        data_path.display(),
        output_dir.path().display()
    );

    let mut temp_file = NamedTempFile::new().expect("temp file creation should succeed");
    temp_file.write_all(yaml.as_bytes()).expect("file write should succeed");

    let result = train_from_yaml(temp_file.path());
    assert!(result.is_ok(), "Training should succeed: {:?}", result.err());

    // Verify output file was created
    let output_path = output_dir.path().join("final_model.json");
    assert!(output_path.exists(), "Output model file should exist");

    // Verify output contains expected metadata
    let content = fs::read_to_string(&output_path).expect("file read should succeed");
    assert!(content.contains("transformer"), "Should indicate transformer mode");
    assert!(content.contains("epochs_completed"), "Should have epoch count");
    assert!(content.contains("final_loss"), "Should have final loss");
}

/// Test: Training with gradient accumulation
#[test]
fn test_e2e_training_with_gradient_accumulation() {
    let output_dir = TempDir::new().expect("temp file creation should succeed");
    let config_dir = TempDir::new().expect("temp file creation should succeed");
    let data_dir = TempDir::new().expect("temp file creation should succeed");
    let model_dir = TempDir::new().expect("temp file creation should succeed");

    let config_path = create_minimal_config(&config_dir);
    let data_path = create_pretokenized_data(&data_dir);
    let model_path = create_dummy_safetensors(&model_dir, "model.safetensors");

    let yaml = format!(
        r#"
model:
  path: "{}"
  mode: transformer
  config: "{}"

data:
  train: "{}"
  batch_size: 1
  seq_len: 8

optimizer:
  name: adamw
  lr: 0.005

training:
  epochs: 2
  mode: causal_lm
  gradient_accumulation: 4
  warmup_steps: 2
  output_dir: "{}"
"#,
        model_path.display(),
        config_path.display(),
        data_path.display(),
        output_dir.path().display()
    );

    let mut temp_file = NamedTempFile::new().expect("temp file creation should succeed");
    temp_file.write_all(yaml.as_bytes()).expect("file write should succeed");

    let result = train_from_yaml(temp_file.path());
    assert!(
        result.is_ok(),
        "Training with gradient accumulation should succeed: {:?}",
        result.err()
    );
}

/// Test: Training with mixed precision
#[test]
fn test_e2e_training_with_mixed_precision() {
    let output_dir = TempDir::new().expect("temp file creation should succeed");
    let config_dir = TempDir::new().expect("temp file creation should succeed");
    let data_dir = TempDir::new().expect("temp file creation should succeed");
    let model_dir = TempDir::new().expect("temp file creation should succeed");

    let config_path = create_minimal_config(&config_dir);
    let data_path = create_pretokenized_data(&data_dir);
    let model_path = create_dummy_safetensors(&model_dir, "model.safetensors");

    let yaml = format!(
        r#"
model:
  path: "{}"
  mode: transformer
  config: "{}"

data:
  train: "{}"
  batch_size: 2
  seq_len: 8

optimizer:
  name: adam
  lr: 0.01

training:
  epochs: 1
  mode: causal_lm
  mixed_precision: bf16
  output_dir: "{}"
"#,
        model_path.display(),
        config_path.display(),
        data_path.display(),
        output_dir.path().display()
    );

    let mut temp_file = NamedTempFile::new().expect("temp file creation should succeed");
    temp_file.write_all(yaml.as_bytes()).expect("file write should succeed");

    let result = train_from_yaml(temp_file.path());
    assert!(result.is_ok(), "Training with mixed precision should succeed: {:?}", result.err());
}

/// Test: Multiple epochs show loss progression
#[test]
fn test_e2e_loss_tracking() {
    let output_dir = TempDir::new().expect("temp file creation should succeed");
    let config_dir = TempDir::new().expect("temp file creation should succeed");
    let data_dir = TempDir::new().expect("temp file creation should succeed");
    let model_dir = TempDir::new().expect("temp file creation should succeed");

    let config_path = create_minimal_config(&config_dir);
    let data_path = create_pretokenized_data(&data_dir);
    let model_path = create_dummy_safetensors(&model_dir, "model.safetensors");

    let yaml = format!(
        r#"
model:
  path: "{}"
  mode: transformer
  config: "{}"

data:
  train: "{}"
  batch_size: 4
  seq_len: 8

optimizer:
  name: adam
  lr: 0.1

training:
  epochs: 5
  mode: causal_lm
  output_dir: "{}"
"#,
        model_path.display(),
        config_path.display(),
        data_path.display(),
        output_dir.path().display()
    );

    let mut temp_file = NamedTempFile::new().expect("temp file creation should succeed");
    temp_file.write_all(yaml.as_bytes()).expect("file write should succeed");

    let result = train_from_yaml(temp_file.path());
    assert!(result.is_ok(), "Training should succeed: {:?}", result.err());

    // Verify output metadata
    let output_path = output_dir.path().join("final_model.json");
    let content = fs::read_to_string(&output_path).expect("file read should succeed");
    let metadata: serde_json::Value =
        serde_json::from_str(&content).expect("JSON deserialization should succeed");

    // Check epochs completed
    assert_eq!(metadata["epochs_completed"].as_u64().expect("operation should succeed"), 5);

    // Check that we have loss values
    assert!(metadata["final_loss"].as_f64().is_some(), "Should have final_loss");
    assert!(metadata["best_loss"].as_f64().is_some(), "Should have best_loss");
}

/// Test: Text data with tokenization
#[test]
fn test_e2e_text_tokenization() {
    let output_dir = TempDir::new().expect("temp file creation should succeed");
    let config_dir = TempDir::new().expect("temp file creation should succeed");
    let data_dir = TempDir::new().expect("temp file creation should succeed");
    let model_dir = TempDir::new().expect("temp file creation should succeed");

    let config_path = create_minimal_config(&config_dir);
    let model_path = create_dummy_safetensors(&model_dir, "model.safetensors");

    // Create text training data
    let data_path = data_dir.path().join("train.json");
    fs::write(
        &data_path,
        r#"[
            {"text": "Hello world"},
            {"text": "Testing the tokenizer"},
            {"text": "Another example"}
        ]"#,
    )
    .expect("operation should succeed");

    let yaml = format!(
        r#"
model:
  path: "{}"
  mode: transformer
  config: "{}"

data:
  train: "{}"
  batch_size: 2
  seq_len: 16
  input_column: text

optimizer:
  name: adam
  lr: 0.01

training:
  epochs: 1
  mode: causal_lm
  output_dir: "{}"
"#,
        model_path.display(),
        config_path.display(),
        data_path.display(),
        output_dir.path().display()
    );

    let mut temp_file = NamedTempFile::new().expect("temp file creation should succeed");
    temp_file.write_all(yaml.as_bytes()).expect("file write should succeed");

    let result = train_from_yaml(temp_file.path());
    assert!(result.is_ok(), "Training with text tokenization should succeed: {:?}", result.err());
}

/// Test: JSONL format support
#[test]
fn test_e2e_jsonl_format() {
    let output_dir = TempDir::new().expect("temp file creation should succeed");
    let config_dir = TempDir::new().expect("temp file creation should succeed");
    let data_dir = TempDir::new().expect("temp file creation should succeed");
    let model_dir = TempDir::new().expect("temp file creation should succeed");

    let config_path = create_minimal_config(&config_dir);
    let model_path = create_dummy_safetensors(&model_dir, "model.safetensors");

    // Create JSONL training data
    let data_path = data_dir.path().join("train.jsonl");
    fs::write(
        &data_path,
        r#"{"text": "First training example"}
{"text": "Second training example"}
{"text": "Third training example"}"#,
    )
    .expect("operation should succeed");

    let yaml = format!(
        r#"
model:
  path: "{}"
  mode: transformer
  config: "{}"

data:
  train: "{}"
  batch_size: 2
  seq_len: 16

optimizer:
  name: adam
  lr: 0.01

training:
  epochs: 1
  mode: causal_lm
  output_dir: "{}"
"#,
        model_path.display(),
        config_path.display(),
        data_path.display(),
        output_dir.path().display()
    );

    let mut temp_file = NamedTempFile::new().expect("temp file creation should succeed");
    temp_file.write_all(yaml.as_bytes()).expect("file write should succeed");

    let result = train_from_yaml(temp_file.path());
    assert!(result.is_ok(), "Training with JSONL format should succeed: {:?}", result.err());
}

// Note: Tabular mode is tested in src/config/train/tests/train_yaml.rs
// ENT-118 focuses on LLM/Transformer training integration
