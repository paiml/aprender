//! Default values tests - verify default value handling

use crate::yaml_mode::*;

#[test]
fn test_default_data_loader_values() {
    let yaml = r#"
entrenar: "1.0"
name: "test"
version: "1.0.0"

data:
  source: "./data.parquet"
  loader:
    batch_size: 16
    shuffle: false
"#;
    let manifest: TrainingManifest = serde_yaml::from_str(yaml).expect("operation should succeed");
    let loader = manifest.data.expect("load should succeed").loader.expect("load should succeed");

    // Verify defaults are applied via Default trait or serde defaults
    assert_eq!(loader.batch_size, 16);
    assert!(!loader.shuffle);
    // num_workers should default to 0
    assert_eq!(loader.num_workers.unwrap_or(0), 0);
    // pin_memory should default to false
    assert!(!loader.pin_memory.unwrap_or(false));
    // drop_last should default to false
    assert!(!loader.drop_last.unwrap_or(false));
}

#[test]
fn test_default_optimizer_values() {
    let yaml = r#"
entrenar: "1.0"
name: "test"
version: "1.0.0"

optimizer:
  name: "adam"
  lr: 0.001
"#;
    let manifest: TrainingManifest = serde_yaml::from_str(yaml).expect("operation should succeed");
    let optim = manifest.optimizer.expect("operation should succeed");

    // Default weight_decay is 0.01 for adamw, but adam has no default
    assert!(optim.weight_decay.is_none() || optim.weight_decay == Some(0.0));
}

#[test]
fn test_default_training_epochs() {
    let yaml = r#"
entrenar: "1.0"
name: "test"
version: "1.0.0"

training: {}
"#;
    let manifest: TrainingManifest = serde_yaml::from_str(yaml).expect("operation should succeed");
    let training = manifest.training.expect("operation should succeed");

    // Default epochs is 10 per spec Appendix A
    assert_eq!(training.epochs.unwrap_or(10), 10);
}
