//! Tests for YAML configuration

use super::*;

const MINIMAL_CONFIG: &str = r#"
teacher:
  model_id: "microsoft/codebert-base"
student:
  model_id: "distilbert-base-uncased"
dataset:
  path: "wikitext"
"#;

const FULL_CONFIG: &str = r#"
teacher:
  model_id: "microsoft/codebert-base"
  revision: "main"
  load_in_8bit: false

student:
  model_id: "distilbert-base-uncased"
  lora:
    rank: 16
    alpha: 32.0
    target_modules: ["q_proj", "v_proj"]
  load_in_4bit: true

distillation:
  temperature: 6.0
  alpha: 0.8
  progressive:
    layer_mapping: [[0, 3], [1, 7], [2, 11]]
    hidden_weight: 0.5
  attention_transfer:
    weight: 0.1

training:
  epochs: 5
  batch_size: 32
  learning_rate: 1.0e-4
  weight_decay: 0.01
  gradient_checkpointing: true
  mixed_precision: "bf16"

dataset:
  path: "wikitext"
  max_seq_length: 256
  max_train_examples: 10000

output:
  dir: "./distillation_output"
  save_steps: 1000
  eval_steps: 200
"#;

// =========================================================================
// Parsing Tests
// =========================================================================

#[test]
fn test_parse_minimal_config() {
    let config = DistillationYamlConfig::from_yaml(MINIMAL_CONFIG);
    assert!(config.is_ok());
    let config = config.expect("config should be valid");
    assert_eq!(config.teacher.model_id, "microsoft/codebert-base");
    assert_eq!(config.student.model_id, "distilbert-base-uncased");
}

#[test]
fn test_parse_full_config() {
    let config = DistillationYamlConfig::from_yaml(FULL_CONFIG);
    assert!(config.is_ok());
    let config = config.expect("config should be valid");

    // Teacher
    assert_eq!(config.teacher.model_id, "microsoft/codebert-base");
    assert!(!config.teacher.load_in_8bit);

    // Student
    assert!(config.student.lora.is_some());
    let lora = config.student.lora.as_ref().expect("config should be valid");
    assert_eq!(lora.rank, 16);
    assert_eq!(lora.alpha, 32.0);
    assert!(config.student.load_in_4bit);

    // Distillation
    assert_eq!(config.distillation.temperature, 6.0);
    assert_eq!(config.distillation.alpha, 0.8);
    assert!(config.distillation.progressive.is_some());
    assert!(config.distillation.attention_transfer.is_some());

    // Training
    assert_eq!(config.training.epochs, 5);
    assert_eq!(config.training.batch_size, 32);
    assert!(config.training.gradient_checkpointing);
}

#[test]
fn test_defaults() {
    let config = DistillationYamlConfig::from_yaml(MINIMAL_CONFIG).expect("config should be valid");

    // Check defaults
    assert_eq!(config.distillation.temperature, 4.0);
    assert_eq!(config.distillation.alpha, 0.7);
    assert_eq!(config.training.epochs, 3);
    assert_eq!(config.training.batch_size, 16);
    assert_eq!(config.teacher.revision, "main");
}

// =========================================================================
// Validation Tests
// =========================================================================

#[test]
fn test_validate_minimal() {
    let config = DistillationYamlConfig::from_yaml(MINIMAL_CONFIG).expect("config should be valid");
    assert!(config.validate().is_ok());
}

#[test]
fn test_validate_empty_teacher() {
    let yaml = r#"
teacher:
  model_id: ""
student:
  model_id: "test"
dataset:
  path: "test"
"#;
    let config = DistillationYamlConfig::from_yaml(yaml).expect("config should be valid");
    assert!(config.validate().is_err());
}

#[test]
fn test_validate_invalid_temperature() {
    let yaml = r#"
teacher:
  model_id: "test"
student:
  model_id: "test"
distillation:
  temperature: -1.0
dataset:
  path: "test"
"#;
    let config = DistillationYamlConfig::from_yaml(yaml).expect("config should be valid");
    assert!(config.validate().is_err());
}

#[test]
fn test_validate_invalid_alpha() {
    let yaml = r#"
teacher:
  model_id: "test"
student:
  model_id: "test"
distillation:
  alpha: 1.5
dataset:
  path: "test"
"#;
    let config = DistillationYamlConfig::from_yaml(yaml).expect("config should be valid");
    assert!(config.validate().is_err());
}

// =========================================================================
// Conversion Tests
// =========================================================================

#[test]
fn test_to_trainer_config() {
    let config = DistillationYamlConfig::from_yaml(FULL_CONFIG).expect("config should be valid");
    let trainer_config = config.to_trainer_config();
    assert!(trainer_config.is_ok());

    let trainer = trainer_config.expect("config should be valid");
    assert_eq!(trainer.teacher_model, "microsoft/codebert-base");
    assert_eq!(trainer.epochs, 5);
    assert!(trainer.progressive.is_some());
}

#[test]
fn test_lora_config_conversion() {
    let yaml = LoRAYamlConfig {
        rank: 8,
        alpha: 16.0,
        target_modules: vec!["q_proj".to_string(), "k_proj".to_string()],
        layers: Some(vec![0, 1, 2]),
    };

    let lora_config = crate::lora::LoRAConfig::from(&yaml);
    assert_eq!(lora_config.rank, 8);
    assert_eq!(lora_config.alpha, 16.0);
}

// =========================================================================
// Serialization Tests
// =========================================================================

#[test]
fn test_roundtrip() {
    let config = DistillationYamlConfig::from_yaml(FULL_CONFIG).expect("config should be valid");
    let yaml = config.to_yaml().expect("config should be valid");
    let config2 = DistillationYamlConfig::from_yaml(&yaml).expect("config should be valid");

    assert_eq!(config.teacher.model_id, config2.teacher.model_id);
    assert_eq!(config.training.epochs, config2.training.epochs);
}

#[test]
fn test_save_load() {
    let config = DistillationYamlConfig::from_yaml(MINIMAL_CONFIG).expect("config should be valid");
    let path = "/tmp/test_distill_config.yaml";

    config.save(path).expect("save should succeed");
    let loaded = DistillationYamlConfig::load(path).expect("load should succeed");

    assert_eq!(config.teacher.model_id, loaded.teacher.model_id);
    std::fs::remove_file(path).ok();
}

// =========================================================================
// Progressive Config Tests
// =========================================================================

#[test]
fn test_progressive_config() {
    let yaml = r#"
teacher:
  model_id: "test"
student:
  model_id: "test"
distillation:
  progressive:
    layer_mapping: [[0, 2], [1, 5], [2, 8]]
dataset:
  path: "test"
"#;
    let config = DistillationYamlConfig::from_yaml(yaml).expect("config should be valid");
    let prog = config.distillation.progressive.expect("config should be valid");
    assert_eq!(prog.layer_mapping.len(), 3);
    assert_eq!(prog.layer_mapping[0], [0, 2]);
}

// =========================================================================
// Additional Coverage Tests
// =========================================================================

#[test]
fn test_validate_empty_student() {
    let yaml = r#"
teacher:
  model_id: "teacher_model"
student:
  model_id: ""
dataset:
  path: "test"
"#;
    let config = DistillationYamlConfig::from_yaml(yaml).expect("config should be valid");
    assert!(config.validate().is_err());
}

#[test]
fn test_validate_zero_batch_size() {
    let yaml = r#"
teacher:
  model_id: "test"
student:
  model_id: "test"
training:
  batch_size: 0
dataset:
  path: "test"
"#;
    let config = DistillationYamlConfig::from_yaml(yaml).expect("config should be valid");
    assert!(config.validate().is_err());
}

#[test]
fn test_validate_negative_learning_rate() {
    let yaml = r#"
teacher:
  model_id: "test"
student:
  model_id: "test"
training:
  learning_rate: -0.001
dataset:
  path: "test"
"#;
    let config = DistillationYamlConfig::from_yaml(yaml).expect("config should be valid");
    assert!(config.validate().is_err());
}

#[test]
fn test_validate_empty_dataset_path() {
    let yaml = r#"
teacher:
  model_id: "test"
student:
  model_id: "test"
dataset:
  path: ""
"#;
    let config = DistillationYamlConfig::from_yaml(yaml).expect("config should be valid");
    assert!(config.validate().is_err());
}

#[test]
fn test_attention_transfer_config() {
    let yaml = r#"
teacher:
  model_id: "test"
student:
  model_id: "test"
distillation:
  attention_transfer:
    weight: 0.3
dataset:
  path: "test"
"#;
    let config = DistillationYamlConfig::from_yaml(yaml).expect("config should be valid");
    let at = config.distillation.attention_transfer.expect("config should be valid");
    assert_eq!(at.weight, 0.3);
}

#[test]
fn test_mixed_precision_fp16() {
    let yaml = r#"
teacher:
  model_id: "test"
student:
  model_id: "test"
training:
  mixed_precision: "fp16"
dataset:
  path: "test"
"#;
    let config = DistillationYamlConfig::from_yaml(yaml).expect("config should be valid");
    assert_eq!(config.training.mixed_precision, Some("fp16".to_string()));
}

#[test]
fn test_mixed_precision_bf16() {
    let yaml = r#"
teacher:
  model_id: "test"
student:
  model_id: "test"
training:
  mixed_precision: "bf16"
dataset:
  path: "test"
"#;
    let config = DistillationYamlConfig::from_yaml(yaml).expect("config should be valid");
    assert_eq!(config.training.mixed_precision, Some("bf16".to_string()));
}

#[test]
fn test_full_fine_tune_no_lora() {
    let yaml = r#"
teacher:
  model_id: "teacher"
student:
  model_id: "student"
  load_in_4bit: false
dataset:
  path: "data"
"#;
    let config = DistillationYamlConfig::from_yaml(yaml).expect("config should be valid");
    let trainer_config = config.to_trainer_config().expect("config should be valid");
    // Should have fine_tune set for full fine-tuning
    assert!(!trainer_config.fine_tune.model_id.is_empty());
}

#[test]
fn test_teacher_revision() {
    let yaml = r#"
teacher:
  model_id: "test"
  revision: "v1.0.0"
student:
  model_id: "test"
dataset:
  path: "test"
"#;
    let config = DistillationYamlConfig::from_yaml(yaml).expect("config should be valid");
    assert_eq!(config.teacher.revision, "v1.0.0");
}

#[test]
fn test_output_config() {
    let yaml = r#"
teacher:
  model_id: "test"
student:
  model_id: "test"
dataset:
  path: "test"
output:
  dir: "/custom/output"
  save_steps: 500
  eval_steps: 250
"#;
    let config = DistillationYamlConfig::from_yaml(yaml).expect("config should be valid");
    assert_eq!(config.output.dir, "/custom/output");
    assert_eq!(config.output.save_steps, 500);
    assert_eq!(config.output.eval_steps, 250);
}

#[test]
fn test_lora_with_layers() {
    let yaml = LoRAYamlConfig {
        rank: 16,
        alpha: 32.0,
        target_modules: vec!["q_proj".to_string()],
        layers: Some(vec![0, 1, 2, 3]),
    };
    assert_eq!(yaml.layers.as_ref().expect("operation should succeed").len(), 4);
}

#[test]
fn test_lora_without_layers() {
    let yaml = LoRAYamlConfig {
        rank: 8,
        alpha: 16.0,
        target_modules: vec!["v_proj".to_string()],
        layers: None,
    };
    assert!(yaml.layers.is_none());
}
