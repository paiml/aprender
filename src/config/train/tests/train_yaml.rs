//! Tests for train_from_yaml functionality

use crate::config::train::train_from_yaml;
use std::io::Write;
use tempfile::{NamedTempFile, TempDir};

#[test]
fn test_train_from_yaml_nonexistent() {
    let result = train_from_yaml("/nonexistent/config.yaml");
    assert!(result.is_err());
}

#[test]
fn test_train_from_yaml_success() {
    let output_dir = TempDir::new().expect("temp file creation should succeed");
    let yaml = format!(
        r#"
model:
  path: nonexistent_test_model.gguf
  layers: []

data:
  train: train.parquet
  batch_size: 8

optimizer:
  name: adam
  lr: 0.001

training:
  epochs: 2
  output_dir: "{}"
"#,
        output_dir.path().display()
    );

    let mut temp_file = NamedTempFile::new().expect("temp file creation should succeed");
    temp_file.write_all(yaml.as_bytes()).expect("file write should succeed");

    let result = train_from_yaml(temp_file.path());
    assert!(result.is_ok());

    // Check that model file was saved
    let output_path = output_dir.path().join("final_model.json");
    assert!(output_path.exists());
}

#[test]
fn test_train_from_yaml_with_grad_clip() {
    let output_dir = TempDir::new().expect("temp file creation should succeed");
    let yaml = format!(
        r#"
model:
  path: nonexistent_test_model.gguf
  layers: []

data:
  train: train.parquet
  batch_size: 4

optimizer:
  name: sgd
  lr: 0.01

training:
  epochs: 1
  grad_clip: 1.0
  output_dir: "{}"
"#,
        output_dir.path().display()
    );

    let mut temp_file = NamedTempFile::new().expect("temp file creation should succeed");
    temp_file.write_all(yaml.as_bytes()).expect("file write should succeed");

    let result = train_from_yaml(temp_file.path());
    assert!(result.is_ok());
}

#[test]
fn test_train_from_yaml_with_lora() {
    let output_dir = TempDir::new().expect("temp file creation should succeed");
    let yaml = format!(
        r#"
model:
  path: nonexistent_test_model.gguf
  layers: [q_proj, v_proj]

data:
  train: train.parquet
  batch_size: 4

optimizer:
  name: adamw
  lr: 0.0001

lora:
  rank: 16
  alpha: 32
  target_modules: [q_proj, v_proj]

training:
  epochs: 1
  output_dir: "{}"
"#,
        output_dir.path().display()
    );

    let mut temp_file = NamedTempFile::new().expect("temp file creation should succeed");
    temp_file.write_all(yaml.as_bytes()).expect("file write should succeed");

    let result = train_from_yaml(temp_file.path());
    assert!(result.is_ok());
}

#[test]
fn test_train_from_yaml_with_quantize() {
    let output_dir = TempDir::new().expect("temp file creation should succeed");
    let yaml = format!(
        r#"
model:
  path: nonexistent_test_model.gguf
  layers: []

data:
  train: train.parquet
  batch_size: 4

optimizer:
  name: adam
  lr: 0.001

quantize:
  bits: 4
  symmetric: true

training:
  epochs: 1
  output_dir: "{}"
"#,
        output_dir.path().display()
    );

    let mut temp_file = NamedTempFile::new().expect("temp file creation should succeed");
    temp_file.write_all(yaml.as_bytes()).expect("file write should succeed");

    let result = train_from_yaml(temp_file.path());
    assert!(result.is_ok());
}

#[test]
fn test_train_from_yaml_malformed() {
    let yaml = "not: [valid yaml";
    let mut temp_file = NamedTempFile::new().expect("temp file creation should succeed");
    temp_file.write_all(yaml.as_bytes()).expect("file write should succeed");

    let result = train_from_yaml(temp_file.path());
    assert!(result.is_err());
}

#[test]
fn test_train_from_yaml_invalid_config() {
    let yaml = r"
model:
  path: nonexistent_test_model.gguf

data:
  train: train.parquet
  batch_size: 0

optimizer:
  name: adam
  lr: 0.001
";
    let mut temp_file = NamedTempFile::new().expect("temp file creation should succeed");
    temp_file.write_all(yaml.as_bytes()).expect("file write should succeed");

    let result = train_from_yaml(temp_file.path());
    assert!(result.is_err());
}

// =========================================================================
// ENT-115: Transformer Mode Tests
// =========================================================================

/// Test that transformer mode is correctly detected and uses TransformerTrainer
/// Uses a minimal config file to keep tests fast
#[test]
fn test_train_from_yaml_transformer_mode() {
    let output_dir = TempDir::new().expect("temp file creation should succeed");
    let config_dir = TempDir::new().expect("temp file creation should succeed");

    // Create a minimal model config for fast testing
    let config_path = config_dir.path().join("config.json");
    std::fs::write(
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

    let yaml = format!(
        r#"
model:
  path: model.safetensors
  mode: transformer
  config: "{}"

data:
  train: train.json
  batch_size: 2
  seq_len: 8

optimizer:
  name: adam
  lr: 0.0001

training:
  epochs: 1
  mode: causal_lm
  output_dir: "{}"
"#,
        config_path.display(),
        output_dir.path().display()
    );

    let mut temp_file = NamedTempFile::new().expect("temp file creation should succeed");
    temp_file.write_all(yaml.as_bytes()).expect("file write should succeed");

    let result = train_from_yaml(temp_file.path());
    assert!(result.is_ok());

    // Check that model file was saved
    let output_path = output_dir.path().join("final_model.json");
    assert!(output_path.exists());

    // Verify output contains transformer-specific metadata
    let output_content = std::fs::read_to_string(&output_path).expect("file read should succeed");
    assert!(output_content.contains("transformer"));
}

/// Test gradient accumulation works in transformer mode
#[test]
fn test_train_from_yaml_transformer_with_gradient_accumulation() {
    let output_dir = TempDir::new().expect("temp file creation should succeed");
    let config_dir = TempDir::new().expect("temp file creation should succeed");

    // Create a minimal model config
    let config_path = config_dir.path().join("config.json");
    std::fs::write(
        &config_path,
        r#"{
            "hidden_size": 32,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "intermediate_size": 64,
            "num_hidden_layers": 1,
            "vocab_size": 100,
            "max_position_embeddings": 64
        }"#,
    )
    .expect("operation should succeed");

    let yaml = format!(
        r#"
model:
  path: model.safetensors
  mode: transformer
  config: "{}"

data:
  train: train.json
  batch_size: 2
  seq_len: 8

optimizer:
  name: adamw
  lr: 0.00001

training:
  epochs: 1
  mode: causal_lm
  gradient_accumulation: 2
  warmup_steps: 1
  output_dir: "{}"
"#,
        config_path.display(),
        output_dir.path().display()
    );

    let mut temp_file = NamedTempFile::new().expect("temp file creation should succeed");
    temp_file.write_all(yaml.as_bytes()).expect("file write should succeed");

    let result = train_from_yaml(temp_file.path());
    assert!(result.is_ok());
}

/// Test mixed precision is applied correctly
#[test]
fn test_train_from_yaml_transformer_with_mixed_precision() {
    let output_dir = TempDir::new().expect("temp file creation should succeed");
    let config_dir = TempDir::new().expect("temp file creation should succeed");

    // Create a minimal model config
    let config_path = config_dir.path().join("config.json");
    std::fs::write(
        &config_path,
        r#"{
            "hidden_size": 32,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "intermediate_size": 64,
            "num_hidden_layers": 1,
            "vocab_size": 100,
            "max_position_embeddings": 64
        }"#,
    )
    .expect("operation should succeed");

    let yaml = format!(
        r#"
model:
  path: model.safetensors
  mode: transformer
  config: "{}"

data:
  train: train.json
  batch_size: 2
  seq_len: 8

optimizer:
  name: adam
  lr: 0.0001

training:
  epochs: 1
  mode: causal_lm
  mixed_precision: bf16
  output_dir: "{}"
"#,
        config_path.display(),
        output_dir.path().display()
    );

    let mut temp_file = NamedTempFile::new().expect("temp file creation should succeed");
    temp_file.write_all(yaml.as_bytes()).expect("file write should succeed");

    let result = train_from_yaml(temp_file.path());
    assert!(result.is_ok());
}

#[test]
fn test_train_from_yaml_default_mode_is_tabular() {
    // When mode is not specified, should default to tabular
    let output_dir = TempDir::new().expect("temp file creation should succeed");
    let yaml = format!(
        r#"
model:
  path: nonexistent_test_model.gguf

data:
  train: train.parquet
  batch_size: 4

optimizer:
  name: adam
  lr: 0.001

training:
  epochs: 1
  output_dir: "{}"
"#,
        output_dir.path().display()
    );

    let mut temp_file = NamedTempFile::new().expect("temp file creation should succeed");
    temp_file.write_all(yaml.as_bytes()).expect("file write should succeed");

    let result = train_from_yaml(temp_file.path());
    assert!(result.is_ok());
}

#[test]
fn test_train_from_yaml_transformer_with_config_file() {
    let output_dir = TempDir::new().expect("temp file creation should succeed");

    // Create a mock config.json file
    let config_dir = TempDir::new().expect("temp file creation should succeed");
    let config_path = config_dir.path().join("config.json");
    std::fs::write(
        &config_path,
        r#"{
            "hidden_size": 256,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "intermediate_size": 512,
            "num_hidden_layers": 2,
            "vocab_size": 1000,
            "max_position_embeddings": 512,
            "rms_norm_eps": 1e-6,
            "rope_theta": 10000.0,
            "attention_bias": false
        }"#,
    )
    .expect("operation should succeed");

    let yaml = format!(
        r#"
model:
  path: model.safetensors
  mode: transformer
  config: "{}"

data:
  train: train.json
  batch_size: 2
  seq_len: 64

optimizer:
  name: adam
  lr: 0.0001

training:
  epochs: 1
  mode: causal_lm
  output_dir: "{}"
"#,
        config_path.display(),
        output_dir.path().display()
    );

    let mut temp_file = NamedTempFile::new().expect("temp file creation should succeed");
    temp_file.write_all(yaml.as_bytes()).expect("file write should succeed");

    let result = train_from_yaml(temp_file.path());
    assert!(result.is_ok());
}

// =========================================================================
// ENT-116: Text Tokenization Tests
// =========================================================================

#[test]
fn test_train_from_yaml_transformer_with_text_json() {
    let output_dir = TempDir::new().expect("temp file creation should succeed");
    let config_dir = TempDir::new().expect("temp file creation should succeed");
    let data_dir = TempDir::new().expect("temp file creation should succeed");

    // Create a minimal model config
    let config_path = config_dir.path().join("config.json");
    std::fs::write(
        &config_path,
        r#"{
            "hidden_size": 32,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "intermediate_size": 64,
            "num_hidden_layers": 1,
            "vocab_size": 100,
            "max_position_embeddings": 64
        }"#,
    )
    .expect("operation should succeed");

    // Create text training data in JSON format
    let data_path = data_dir.path().join("train.json");
    std::fs::write(
        &data_path,
        r#"[
            {"text": "Hello world, this is a test."},
            {"text": "Another example for training."},
            {"text": "More data for the language model."}
        ]"#,
    )
    .expect("operation should succeed");

    let yaml = format!(
        r#"
model:
  path: model.safetensors
  mode: transformer
  config: "{}"

data:
  train: "{}"
  batch_size: 2
  seq_len: 32
  input_column: text

optimizer:
  name: adam
  lr: 0.0001

training:
  epochs: 1
  mode: causal_lm
  output_dir: "{}"
"#,
        config_path.display(),
        data_path.display(),
        output_dir.path().display()
    );

    let mut temp_file = NamedTempFile::new().expect("temp file creation should succeed");
    temp_file.write_all(yaml.as_bytes()).expect("file write should succeed");

    let result = train_from_yaml(temp_file.path());
    assert!(result.is_ok());
}

#[test]
fn test_train_from_yaml_transformer_with_jsonl() {
    let output_dir = TempDir::new().expect("temp file creation should succeed");
    let config_dir = TempDir::new().expect("temp file creation should succeed");
    let data_dir = TempDir::new().expect("temp file creation should succeed");

    // Create a minimal model config
    let config_path = config_dir.path().join("config.json");
    std::fs::write(
        &config_path,
        r#"{"hidden_size": 32, "num_attention_heads": 2, "num_key_value_heads": 1, "intermediate_size": 64, "num_hidden_layers": 1, "vocab_size": 100, "max_position_embeddings": 64}"#,
    )
    .expect("operation should succeed");

    // Create text training data in JSONL format
    let data_path = data_dir.path().join("train.jsonl");
    std::fs::write(
        &data_path,
        r#"{"text": "First line of training data."}
{"text": "Second line of training data."}
{"text": "Third line of training data."}"#,
    )
    .expect("operation should succeed");

    let yaml = format!(
        r#"
model:
  path: model.safetensors
  mode: transformer
  config: "{}"

data:
  train: "{}"
  batch_size: 2
  seq_len: 32

optimizer:
  name: adam
  lr: 0.0001

training:
  epochs: 1
  mode: causal_lm
  output_dir: "{}"
"#,
        config_path.display(),
        data_path.display(),
        output_dir.path().display()
    );

    let mut temp_file = NamedTempFile::new().expect("temp file creation should succeed");
    temp_file.write_all(yaml.as_bytes()).expect("file write should succeed");

    let result = train_from_yaml(temp_file.path());
    assert!(result.is_ok());
}

#[test]
fn test_train_from_yaml_transformer_with_pretokenized() {
    let output_dir = TempDir::new().expect("temp file creation should succeed");
    let config_dir = TempDir::new().expect("temp file creation should succeed");
    let data_dir = TempDir::new().expect("temp file creation should succeed");

    // Create a minimal model config
    let config_path = config_dir.path().join("config.json");
    std::fs::write(
        &config_path,
        r#"{"hidden_size": 32, "num_attention_heads": 2, "num_key_value_heads": 1, "intermediate_size": 64, "num_hidden_layers": 1, "vocab_size": 100, "max_position_embeddings": 64}"#,
    )
    .expect("operation should succeed");

    // Create pre-tokenized training data
    let data_path = data_dir.path().join("train.json");
    std::fs::write(
        &data_path,
        r#"{"examples": [
            {"input_ids": [1, 2, 3, 4, 5, 6, 7, 8]},
            {"input_ids": [10, 20, 30, 40, 50, 60]},
            {"input_ids": [100, 200, 300, 400]}
        ]}"#,
    )
    .expect("operation should succeed");

    let yaml = format!(
        r#"
model:
  path: model.safetensors
  mode: transformer
  config: "{}"

data:
  train: "{}"
  batch_size: 2
  seq_len: 16

optimizer:
  name: adam
  lr: 0.0001

training:
  epochs: 1
  mode: causal_lm
  output_dir: "{}"
"#,
        config_path.display(),
        data_path.display(),
        output_dir.path().display()
    );

    let mut temp_file = NamedTempFile::new().expect("temp file creation should succeed");
    temp_file.write_all(yaml.as_bytes()).expect("file write should succeed");

    let result = train_from_yaml(temp_file.path());
    assert!(result.is_ok());
}
