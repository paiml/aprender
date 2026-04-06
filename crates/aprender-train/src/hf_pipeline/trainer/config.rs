//! Distillation training configuration.

use crate::hf_pipeline::distillation::{
    AttentionTransfer, DistillationLoss, ProgressiveDistillation,
};
use crate::hf_pipeline::fine_tune::FineTuneConfig;
use std::path::PathBuf;

/// Default number of training steps between checkpoint saves.
const DEFAULT_SAVE_STEPS: usize = 500;

/// Distillation training configuration
#[derive(Debug, Clone)]
pub struct TrainerConfig {
    /// Teacher model ID or path
    pub teacher_model: String,
    /// Student model ID or path
    pub student_model: String,
    /// Output directory for checkpoints and logs
    pub output_dir: PathBuf,
    /// Distillation loss configuration
    pub distillation_loss: DistillationLoss,
    /// Progressive distillation (hidden state matching)
    pub progressive: Option<ProgressiveDistillation>,
    /// Attention transfer
    pub attention_transfer: Option<AttentionTransfer>,
    /// Fine-tuning configuration for student
    pub fine_tune: FineTuneConfig,
    /// Number of training epochs
    pub epochs: usize,
    /// Steps per epoch (0 = auto-detect from dataset)
    pub steps_per_epoch: usize,
    /// Logging frequency (steps)
    pub log_every_n_steps: usize,
    /// Checkpoint frequency (steps)
    pub save_every_n_steps: usize,
    /// Evaluation frequency (steps)
    pub eval_every_n_steps: usize,
    /// Maximum gradient norm for clipping
    pub max_grad_norm: f32,
    /// Random seed
    pub seed: u64,
}

impl Default for TrainerConfig {
    fn default() -> Self {
        Self {
            teacher_model: String::new(),
            student_model: String::new(),
            output_dir: PathBuf::from("./distillation_output"),
            distillation_loss: DistillationLoss::default(),
            progressive: None,
            attention_transfer: None,
            fine_tune: FineTuneConfig::default(),
            epochs: 3,
            steps_per_epoch: 0,
            log_every_n_steps: 10,
            save_every_n_steps: DEFAULT_SAVE_STEPS,
            eval_every_n_steps: 100,
            max_grad_norm: 1.0,
            seed: 42,
        }
    }
}

impl TrainerConfig {
    /// Create new trainer config with teacher and student models
    #[must_use]
    pub fn new(teacher: impl Into<String>, student: impl Into<String>) -> Self {
        Self { teacher_model: teacher.into(), student_model: student.into(), ..Default::default() }
    }

    /// Set temperature for distillation
    #[must_use]
    pub fn temperature(mut self, temp: f32) -> Self {
        contract_pre_temperature!();
        self.distillation_loss = DistillationLoss::new(temp, self.distillation_loss.alpha);
        contract_post_temperature_bounds!(temp);
        self
    }

    /// Set alpha for soft vs hard loss weight
    #[must_use]
    pub fn alpha(mut self, alpha: f32) -> Self {
        self.distillation_loss = DistillationLoss::new(self.distillation_loss.temperature, alpha);
        self
    }

    /// Enable progressive distillation with layer mapping
    #[must_use]
    pub fn with_progressive(mut self, layer_mapping: Vec<(usize, usize)>) -> Self {
        self.progressive = Some(ProgressiveDistillation::new(layer_mapping));
        self
    }

    /// Enable attention transfer
    #[must_use]
    pub fn with_attention_transfer(mut self, weight: f32) -> Self {
        self.attention_transfer = Some(AttentionTransfer::new(weight));
        self
    }

    /// Set output directory
    #[must_use]
    pub fn output_dir(mut self, path: impl Into<PathBuf>) -> Self {
        self.output_dir = path.into();
        self
    }

    /// Set number of epochs
    #[must_use]
    pub fn epochs(mut self, n: usize) -> Self {
        self.epochs = n;
        self
    }
}
