//! Tests for configuration file loading

use crate::config::train::load_config;
use std::io::Write;
use tempfile::NamedTempFile;

#[test]
fn test_load_valid_config() {
    let yaml = r"
model:
  path: nonexistent_test_model.gguf
  layers: []

data:
  train: train.parquet
  batch_size: 8

optimizer:
  name: adam
  lr: 0.001
";

    let mut temp_file = NamedTempFile::new().expect("temp file creation should succeed");
    temp_file.write_all(yaml.as_bytes()).expect("file write should succeed");

    let spec = load_config(temp_file.path()).expect("load should succeed");
    assert_eq!(spec.optimizer.name, "adam");
    assert_eq!(spec.data.batch_size, 8);
}

#[test]
fn test_load_invalid_config() {
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

    let result = load_config(temp_file.path());
    assert!(result.is_err());
}

#[test]
fn test_load_malformed_yaml() {
    let yaml = "this is not valid yaml: [}";

    let mut temp_file = NamedTempFile::new().expect("temp file creation should succeed");
    temp_file.write_all(yaml.as_bytes()).expect("file write should succeed");

    let result = load_config(temp_file.path());
    assert!(result.is_err());
}

#[test]
fn test_load_config_nonexistent_file() {
    let result = load_config("/nonexistent/path/to/config.yaml");
    assert!(result.is_err());
}

#[test]
fn test_load_config_with_lora() {
    let yaml = r"
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
";
    let mut temp_file = NamedTempFile::new().expect("temp file creation should succeed");
    temp_file.write_all(yaml.as_bytes()).expect("file write should succeed");

    let spec = load_config(temp_file.path()).expect("load should succeed");
    assert!(spec.lora.is_some());
    let lora = spec.lora.expect("operation should succeed");
    assert_eq!(lora.rank, 16);
    assert_eq!(lora.alpha, 32.0);
}

#[test]
fn test_load_config_with_quantize() {
    let yaml = r"
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
";
    let mut temp_file = NamedTempFile::new().expect("temp file creation should succeed");
    temp_file.write_all(yaml.as_bytes()).expect("file write should succeed");

    let spec = load_config(temp_file.path()).expect("load should succeed");
    assert!(spec.quantize.is_some());
    let quant = spec.quantize.expect("operation should succeed");
    assert_eq!(quant.bits, 4);
}

#[test]
fn test_load_config_with_training_options() {
    let yaml = r"
model:
  path: nonexistent_test_model.gguf
  layers: []

data:
  train: train.parquet
  batch_size: 8

optimizer:
  name: sgd
  lr: 0.01

training:
  epochs: 5
  grad_clip: 1.0
";
    let mut temp_file = NamedTempFile::new().expect("temp file creation should succeed");
    temp_file.write_all(yaml.as_bytes()).expect("file write should succeed");

    let spec = load_config(temp_file.path()).expect("load should succeed");
    assert_eq!(spec.training.epochs, 5);
    assert_eq!(spec.training.grad_clip, Some(1.0));
}
