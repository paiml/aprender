//! Integration tests for config module

use super::*;
use std::io::Write;
use tempfile::NamedTempFile;

#[test]
fn test_end_to_end_config_loading() {
    let yaml = r"
model:
  path: llama-7b.gguf
  layers: [q_proj, v_proj]

data:
  train: train.parquet
  val: val.parquet
  batch_size: 16
  auto_infer_types: true

optimizer:
  name: adamw
  lr: 0.0001
  beta1: 0.9
  beta2: 0.999
  weight_decay: 0.01

lora:
  rank: 64
  alpha: 16
  target_modules: [q_proj, v_proj]
  dropout: 0.1

training:
  epochs: 3
  grad_clip: 1.0
  lr_scheduler: cosine
  warmup_steps: 100
";

    let mut temp_file = NamedTempFile::new().expect("temp file creation should succeed");
    temp_file.write_all(yaml.as_bytes()).expect("file write should succeed");

    // Should parse and validate successfully
    let spec = load_config(temp_file.path()).expect("load should succeed");

    assert_eq!(spec.model.layers.len(), 2);
    assert_eq!(spec.data.batch_size, 16);
    assert_eq!(spec.optimizer.name, "adamw");
    assert!(spec.lora.is_some());
    assert_eq!(spec.lora.as_ref().expect("operation should succeed").rank, 64);
    assert_eq!(spec.training.epochs, 3);
}

#[test]
fn test_minimal_config() {
    let yaml = r"
model:
  path: model.gguf

data:
  train: data.parquet
  batch_size: 8

optimizer:
  name: adam
  lr: 0.001
";

    let mut temp_file = NamedTempFile::new().expect("temp file creation should succeed");
    temp_file.write_all(yaml.as_bytes()).expect("file write should succeed");

    let spec = load_config(temp_file.path()).expect("load should succeed");

    // Check defaults are applied
    assert_eq!(spec.training.epochs, 10); // Default
    assert!(spec.lora.is_none());
    assert!(spec.quantize.is_none());
}
