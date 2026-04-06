//! Serialization roundtrip tests - verify YAML serialize/deserialize

use crate::yaml_mode::*;

#[test]
fn test_roundtrip_minimal_manifest() {
    let yaml = r#"
entrenar: "1.0"
name: "roundtrip-test"
version: "1.0.0"
"#;
    let manifest: TrainingManifest = serde_yaml::from_str(yaml).expect("operation should succeed");
    let serialized = serde_yaml::to_string(&manifest).expect("operation should succeed");
    let deserialized: TrainingManifest =
        serde_yaml::from_str(&serialized).expect("deserialization should succeed");

    assert_eq!(manifest.entrenar, deserialized.entrenar);
    assert_eq!(manifest.name, deserialized.name);
    assert_eq!(manifest.version, deserialized.version);
}

#[test]
fn test_roundtrip_complete_manifest() {
    let yaml = r#"
entrenar: "1.0"
name: "complete-test"
version: "1.0.0"
description: "Test description"
seed: 42

data:
  source: "./train.parquet"
  split:
    train: 0.8
    val: 0.2

model:
  source: "hf://test/model"
  dtype: "float16"

optimizer:
  name: "adamw"
  lr: 0.001
  weight_decay: 0.01

training:
  epochs: 5
  gradient:
    clip_norm: 1.0

lora:
  enabled: true
  rank: 16
  alpha: 32
  target_modules:
    - q_proj
    - v_proj

output:
  dir: "./output"
"#;
    let manifest: TrainingManifest = serde_yaml::from_str(yaml).expect("operation should succeed");
    let serialized = serde_yaml::to_string(&manifest).expect("operation should succeed");
    let deserialized: TrainingManifest =
        serde_yaml::from_str(&serialized).expect("deserialization should succeed");

    assert_eq!(manifest.seed, deserialized.seed);
    assert_eq!(
        manifest.lora.as_ref().expect("deserialization should succeed").rank,
        deserialized.lora.as_ref().expect("deserialization should succeed").rank
    );
}
