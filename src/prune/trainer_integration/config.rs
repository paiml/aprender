//! Configuration for the prune-finetune trainer.
//!
//! # Toyota Way: Kaizen (Continuous Improvement)
//! Fine-tuning allows the model to recover from pruning-induced accuracy loss.

use crate::prune::config::PruningConfig;
use crate::prune::data_loader::CalibrationDataConfig;
use serde::{Deserialize, Serialize};

/// Configuration for the prune-finetune trainer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PruneTrainerConfig {
    /// Pruning configuration.
    pub pruning: PruningConfig,
    /// Calibration data configuration.
    pub calibration: CalibrationDataConfig,
    /// Number of fine-tuning epochs after pruning.
    pub finetune_epochs: usize,
    /// Learning rate for fine-tuning.
    pub finetune_lr: f32,
    /// Whether to evaluate before and after pruning.
    pub evaluate_pre_post: bool,
    /// Checkpoint directory.
    pub checkpoint_dir: Option<String>,
    /// Whether to save intermediate checkpoints.
    pub save_checkpoints: bool,
}

impl Default for PruneTrainerConfig {
    fn default() -> Self {
        Self {
            pruning: PruningConfig::default(),
            calibration: CalibrationDataConfig::default(),
            finetune_epochs: 1,
            finetune_lr: 1e-5,
            evaluate_pre_post: true,
            checkpoint_dir: None,
            save_checkpoints: false,
        }
    }
}

impl PruneTrainerConfig {
    /// Create a new configuration with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the pruning configuration.
    pub fn with_pruning(mut self, config: PruningConfig) -> Self {
        self.pruning = config;
        self
    }

    /// Set the calibration configuration.
    pub fn with_calibration(mut self, config: CalibrationDataConfig) -> Self {
        self.calibration = config;
        self
    }

    /// Set the number of fine-tuning epochs.
    pub fn with_finetune_epochs(mut self, epochs: usize) -> Self {
        self.finetune_epochs = epochs;
        self
    }

    /// Set the fine-tuning learning rate.
    pub fn with_finetune_lr(mut self, lr: f32) -> Self {
        self.finetune_lr = lr;
        self
    }

    /// Enable or disable pre/post evaluation.
    pub fn with_evaluate(mut self, enabled: bool) -> Self {
        self.evaluate_pre_post = enabled;
        self
    }

    /// Set the checkpoint directory.
    pub fn with_checkpoint_dir(mut self, dir: impl Into<String>) -> Self {
        self.checkpoint_dir = Some(dir.into());
        self
    }

    /// Enable or disable checkpoint saving.
    pub fn with_save_checkpoints(mut self, enabled: bool) -> Self {
        self.save_checkpoints = enabled;
        self
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<(), String> {
        self.pruning.validate()?;

        if self.finetune_lr <= 0.0 {
            return Err("finetune_lr must be positive".to_string());
        }

        Ok(())
    }
}
