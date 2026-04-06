//! LoRA, quantization, optimizer, and scheduler validation tests

use crate::yaml_mode::*;

#[test]
fn test_validate_rejects_invalid_quantization_bits() {
    let yaml = r#"
entrenar: "1.0"
name: "test"
version: "1.0.0"

quantize:
  enabled: true
  bits: 5
"#;
    let manifest: TrainingManifest = serde_yaml::from_str(yaml).expect("operation should succeed");
    let result = validate_manifest(&manifest);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(matches!(err, ManifestError::InvalidQuantBits { .. }));
}

#[test]
fn test_validate_accepts_valid_lora_config() {
    let yaml = r#"
entrenar: "1.0"
name: "test"
version: "1.0.0"

lora:
  enabled: true
  rank: 16
  alpha: 32
  target_modules:
    - q_proj
    - v_proj
"#;
    let manifest: TrainingManifest = serde_yaml::from_str(yaml).expect("operation should succeed");
    let result = validate_manifest(&manifest);
    assert!(result.is_ok());
}

#[test]
fn test_validate_rejects_lora_without_target_modules() {
    let yaml = r#"
entrenar: "1.0"
name: "test"
version: "1.0.0"

lora:
  enabled: true
  rank: 16
  alpha: 32
  target_modules: []
"#;
    let manifest: TrainingManifest = serde_yaml::from_str(yaml).expect("operation should succeed");
    let result = validate_manifest(&manifest);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(matches!(err, ManifestError::EmptyRequiredField(_)));
}

#[test]
fn test_validate_accepts_disabled_lora_without_modules() {
    let yaml = r#"
entrenar: "1.0"
name: "test"
version: "1.0.0"

lora:
  enabled: false
  rank: 16
  alpha: 32
  target_modules: []
"#;
    let manifest: TrainingManifest = serde_yaml::from_str(yaml).expect("operation should succeed");
    let result = validate_manifest(&manifest);
    assert!(result.is_ok(), "Disabled LoRA should not require target_modules");
}

#[test]
fn test_validate_rejects_zero_lora_rank() {
    let yaml = r#"
entrenar: "1.0"
name: "test"
version: "1.0.0"

lora:
  enabled: true
  rank: 0
  alpha: 32
  target_modules:
    - q_proj
"#;
    let manifest: TrainingManifest = serde_yaml::from_str(yaml).expect("operation should succeed");
    let result = validate_manifest(&manifest);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(matches!(err, ManifestError::InvalidRange { .. }));
}

#[test]
fn test_validate_rejects_negative_lora_alpha() {
    let yaml = r#"
entrenar: "1.0"
name: "test"
version: "1.0.0"

lora:
  enabled: true
  rank: 16
  alpha: -1.0
  target_modules:
    - q_proj
"#;
    let manifest: TrainingManifest = serde_yaml::from_str(yaml).expect("operation should succeed");
    let result = validate_manifest(&manifest);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(matches!(err, ManifestError::InvalidRange { .. }));
}

#[test]
fn test_validate_rejects_invalid_lora_dropout() {
    let yaml = r#"
entrenar: "1.0"
name: "test"
version: "1.0.0"

lora:
  enabled: true
  rank: 16
  alpha: 32
  dropout: 1.5
  target_modules:
    - q_proj
"#;
    let manifest: TrainingManifest = serde_yaml::from_str(yaml).expect("operation should succeed");
    let result = validate_manifest(&manifest);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(matches!(err, ManifestError::InvalidRange { .. }));
}

#[test]
fn test_validate_rejects_invalid_qlora_bits() {
    let yaml = r#"
entrenar: "1.0"
name: "test"
version: "1.0.0"

lora:
  enabled: true
  rank: 16
  alpha: 32
  target_modules:
    - q_proj
  quantize_bits: 3
"#;
    let manifest: TrainingManifest = serde_yaml::from_str(yaml).expect("operation should succeed");
    let result = validate_manifest(&manifest);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(matches!(err, ManifestError::InvalidQuantBits { .. }));
}

#[test]
fn test_validate_rejects_negative_weight_decay() {
    let yaml = r#"
entrenar: "1.0"
name: "test"
version: "1.0.0"

optimizer:
  name: "adam"
  lr: 0.001
  weight_decay: -0.01
"#;
    let manifest: TrainingManifest = serde_yaml::from_str(yaml).expect("operation should succeed");
    let result = validate_manifest(&manifest);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(matches!(err, ManifestError::InvalidRange { .. }));
}

#[test]
fn test_validate_rejects_invalid_betas() {
    let yaml = r#"
entrenar: "1.0"
name: "test"
version: "1.0.0"

optimizer:
  name: "adam"
  lr: 0.001
  betas: [1.5, 0.999]
"#;
    let manifest: TrainingManifest = serde_yaml::from_str(yaml).expect("operation should succeed");
    let result = validate_manifest(&manifest);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(matches!(err, ManifestError::InvalidRange { .. }));
}

#[test]
fn test_validate_rejects_invalid_scheduler() {
    let yaml = r#"
entrenar: "1.0"
name: "test"
version: "1.0.0"

scheduler:
  name: "invalid_scheduler"
"#;
    let manifest: TrainingManifest = serde_yaml::from_str(yaml).expect("operation should succeed");
    let result = validate_manifest(&manifest);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(matches!(err, ManifestError::InvalidScheduler(_)));
}

#[test]
fn test_validate_rejects_invalid_optimizer() {
    let yaml = r#"
entrenar: "1.0"
name: "test"
version: "1.0.0"

optimizer:
  name: "invalid_optimizer"
  lr: 0.001
"#;
    let manifest: TrainingManifest = serde_yaml::from_str(yaml).expect("operation should succeed");
    let result = validate_manifest(&manifest);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(matches!(err, ManifestError::InvalidOptimizer(_)));
}

#[test]
fn test_validate_rejects_zero_learning_rate() {
    let yaml = r#"
entrenar: "1.0"
name: "test"
version: "1.0.0"

optimizer:
  name: "adam"
  lr: 0.0
"#;
    let manifest: TrainingManifest = serde_yaml::from_str(yaml).expect("operation should succeed");
    let result = validate_manifest(&manifest);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(matches!(err, ManifestError::InvalidRange { .. }));
}

#[test]
fn test_validate_accepts_lora_with_pattern() {
    let yaml = r#"
entrenar: "1.0"
name: "test"
version: "1.0.0"

lora:
  enabled: true
  rank: 16
  alpha: 32
  target_modules: []
  target_modules_pattern: ".*proj$"
"#;
    let manifest: TrainingManifest = serde_yaml::from_str(yaml).expect("operation should succeed");
    let result = validate_manifest(&manifest);
    assert!(result.is_ok());
}

#[test]
fn test_validate_accepts_disabled_quantize() {
    let yaml = r#"
entrenar: "1.0"
name: "test"
version: "1.0.0"

quantize:
  enabled: false
  bits: 99
"#;
    let manifest: TrainingManifest = serde_yaml::from_str(yaml).expect("operation should succeed");
    let result = validate_manifest(&manifest);
    assert!(result.is_ok(), "Disabled quantize should skip bit validation");
}
