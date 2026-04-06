//! Tests for trainer integration module.

use super::*;
use crate::prune::config::{PruneMethod, PruningConfig};
use crate::prune::data_loader::CalibrationDataConfig;
use crate::prune::pipeline::PruningStage;

fn default_config() -> PruneTrainerConfig {
    PruneTrainerConfig::new()
        .with_pruning(PruningConfig::default().with_target_sparsity(0.5))
        .with_calibration(CalibrationDataConfig::new().with_num_samples(5))
}

// =============================================================================
// PruneTrainerConfig Tests
// =============================================================================

#[test]
fn test_config_default() {
    // TEST_ID: TI-001
    let config = PruneTrainerConfig::default();
    assert_eq!(config.finetune_epochs, 1);
    assert!((config.finetune_lr - 1e-5).abs() < 1e-10);
    assert!(config.evaluate_pre_post);
    assert!(!config.save_checkpoints);
}

#[test]
fn test_config_builder() {
    // TEST_ID: TI-002
    let config = PruneTrainerConfig::new()
        .with_finetune_epochs(5)
        .with_finetune_lr(1e-4)
        .with_evaluate(false)
        .with_checkpoint_dir("/tmp/checkpoints")
        .with_save_checkpoints(true);

    assert_eq!(config.finetune_epochs, 5);
    assert!((config.finetune_lr - 1e-4).abs() < 1e-10);
    assert!(!config.evaluate_pre_post);
    assert_eq!(config.checkpoint_dir, Some("/tmp/checkpoints".to_string()));
    assert!(config.save_checkpoints);
}

#[test]
fn test_config_validate_valid() {
    // TEST_ID: TI-003
    let config = default_config();
    assert!(config.validate().is_ok());
}

#[test]
fn test_config_validate_invalid_lr() {
    // TEST_ID: TI-004
    let config = PruneTrainerConfig::new().with_finetune_lr(0.0);
    assert!(config.validate().is_err(), "TI-004 FALSIFIED: Zero LR should be invalid");
}

#[test]
fn test_config_serialize() {
    // TEST_ID: TI-005
    let config = default_config();
    let json = serde_json::to_string(&config).expect("JSON serialization should succeed");
    let deserialized: PruneTrainerConfig =
        serde_json::from_str(&json).expect("JSON deserialization should succeed");
    assert_eq!(config.finetune_epochs, deserialized.finetune_epochs);
}

// =============================================================================
// PruneTrainer Tests
// =============================================================================

#[test]
fn test_trainer_new() {
    // TEST_ID: TI-010
    let config = default_config();
    let trainer = PruneTrainer::new(config);

    assert_eq!(trainer.stage(), PruningStage::Idle);
    assert!(!trainer.is_complete());
    assert_eq!(trainer.current_epoch(), 0);
}

#[test]
fn test_trainer_initialize() {
    // TEST_ID: TI-011
    let config = default_config();
    let mut trainer = PruneTrainer::new(config);

    let result = trainer.initialize();
    assert!(result.is_ok(), "TI-011 FALSIFIED: Initialize should succeed");
}

#[test]
fn test_trainer_calibrate() {
    // TEST_ID: TI-012
    let config = default_config();
    let mut trainer = PruneTrainer::new(config);

    trainer.initialize().expect("operation should succeed");
    let result = trainer.calibrate();
    assert!(result.is_ok(), "TI-012 FALSIFIED: Calibrate should succeed");
}

#[test]
fn test_trainer_prune() {
    // TEST_ID: TI-013
    let config = default_config();
    let mut trainer = PruneTrainer::new(config);

    trainer.initialize().expect("operation should succeed");
    trainer.calibrate().expect("operation should succeed");
    let result = trainer.prune();
    assert!(result.is_ok(), "TI-013 FALSIFIED: Prune should succeed");
}

#[test]
fn test_trainer_finetune() {
    // TEST_ID: TI-014
    let config = default_config().with_finetune_epochs(3);
    let mut trainer = PruneTrainer::new(config);

    trainer.initialize().expect("operation should succeed");
    trainer.calibrate().expect("operation should succeed");
    trainer.prune().expect("operation should succeed");
    let result = trainer.finetune();
    assert!(result.is_ok(), "TI-014 FALSIFIED: Finetune should succeed");

    assert_eq!(
        trainer.metrics().finetune_losses.len(),
        3,
        "TI-014 FALSIFIED: Should have 3 loss entries"
    );
}

#[test]
fn test_trainer_evaluate() {
    // TEST_ID: TI-015
    let config = default_config()
        .with_pruning(PruningConfig::default().with_target_sparsity(0.5).with_fine_tune(false));
    let mut trainer = PruneTrainer::new(config);

    trainer.initialize().expect("operation should succeed");
    trainer.calibrate().expect("operation should succeed");
    trainer.prune().expect("operation should succeed");
    let result = trainer.evaluate();
    assert!(result.is_ok(), "TI-015 FALSIFIED: Evaluate should succeed");
}

#[test]
fn test_trainer_full_run() {
    // TEST_ID: TI-016
    let config = default_config().with_finetune_epochs(2);
    let mut trainer = PruneTrainer::new(config);

    let result = trainer.run();
    assert!(result.is_ok(), "TI-016 FALSIFIED: Full run should succeed");
    assert!(trainer.is_complete());
    assert!(trainer.succeeded());

    let metrics = result.expect("operation should succeed");
    assert!((metrics.target_sparsity - 0.5).abs() < 1e-6);
}

#[test]
fn test_trainer_skip_finetune() {
    // TEST_ID: TI-017
    let config = default_config()
        .with_pruning(PruningConfig::default().with_target_sparsity(0.5).with_fine_tune(false));
    let mut trainer = PruneTrainer::new(config);

    let result = trainer.run();
    assert!(result.is_ok());

    // Should not have fine-tuning losses
    assert!(
        trainer.metrics().finetune_losses.is_empty(),
        "TI-017 FALSIFIED: Should skip fine-tuning"
    );
}

#[test]
fn test_trainer_reset() {
    // TEST_ID: TI-018
    let config = default_config();
    let mut trainer = PruneTrainer::new(config);

    trainer.run().expect("operation should succeed");
    assert!(trainer.is_complete());

    trainer.reset();
    assert!(!trainer.is_complete());
    assert_eq!(trainer.stage(), PruningStage::Idle);
    assert_eq!(trainer.current_epoch(), 0);
}

#[test]
fn test_trainer_metrics_access() {
    // TEST_ID: TI-019
    let config = default_config();
    let mut trainer = PruneTrainer::new(config);

    trainer.run().expect("operation should succeed");
    let metrics = trainer.metrics();
    assert!((metrics.target_sparsity - 0.5).abs() < 1e-6);
}

#[test]
fn test_trainer_pipeline_access() {
    // TEST_ID: TI-020
    let config = default_config();
    let mut trainer = PruneTrainer::new(config);

    trainer.run().expect("operation should succeed");
    assert_eq!(trainer.pipeline().stage(), PruningStage::Complete);
}

#[test]
fn test_trainer_clone() {
    // TEST_ID: TI-021
    let config = default_config();
    let trainer = PruneTrainer::new(config);
    let cloned = trainer.clone();

    assert_eq!(trainer.stage(), cloned.stage());
    assert_eq!(trainer.current_epoch(), cloned.current_epoch());
}

// =============================================================================
// Error Handling Tests
// =============================================================================

#[test]
fn test_trainer_prune_wrong_stage() {
    // TEST_ID: TI-030
    let config = default_config();
    let mut trainer = PruneTrainer::new(config);

    // Try to prune without initialization
    let result = trainer.prune();
    assert!(result.is_err(), "TI-030 FALSIFIED: Should fail when pruning in wrong stage");
}

#[test]
fn test_trainer_finetune_wrong_stage() {
    // TEST_ID: TI-031
    let config = default_config();
    let mut trainer = PruneTrainer::new(config);

    // Try to finetune without pruning
    let result = trainer.finetune();
    assert!(result.is_err(), "TI-031 FALSIFIED: Should fail when finetuning in wrong stage");
}

#[test]
fn test_trainer_evaluate_wrong_stage() {
    // TEST_ID: TI-032
    let config = default_config();
    let mut trainer = PruneTrainer::new(config);

    let result = trainer.evaluate();
    assert!(result.is_err(), "TI-032 FALSIFIED: Should fail when evaluating in wrong stage");
}

#[test]
fn test_trainer_export_wrong_stage() {
    // TEST_ID: TI-033
    let config = default_config();
    let mut trainer = PruneTrainer::new(config);

    let result = trainer.export();
    assert!(result.is_err(), "TI-033 FALSIFIED: Should fail when exporting in wrong stage");
}

// =============================================================================
// Calibration Tests
// =============================================================================

#[test]
fn test_trainer_calibration_required_for_wanda() {
    // TEST_ID: TI-040
    let config = default_config().with_pruning(
        PruningConfig::default().with_method(PruneMethod::Wanda).with_target_sparsity(0.5),
    );
    let mut trainer = PruneTrainer::new(config);

    trainer.initialize().expect("operation should succeed");
    assert!(trainer.calibration.is_some(), "TI-040 FALSIFIED: Wanda should require calibration");
}

#[test]
fn test_trainer_no_calibration_for_magnitude() {
    // TEST_ID: TI-041
    let config = default_config().with_pruning(
        PruningConfig::default().with_method(PruneMethod::Magnitude).with_target_sparsity(0.5),
    );
    let mut trainer = PruneTrainer::new(config);

    trainer.initialize().expect("operation should succeed");
    // Magnitude doesn't require calibration
    assert!(
        trainer.calibration.is_none(),
        "TI-041 FALSIFIED: Magnitude should not require calibration"
    );
}
