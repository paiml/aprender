//! Training hyperparameters configuration

use serde::{Deserialize, Serialize};

/// Training hyperparameters
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct TrainingConfig {
    /// Number of epochs
    pub epochs: usize,
    /// Batch size
    pub batch_size: usize,
    /// Learning rate
    pub learning_rate: f64,
    /// Weight decay
    pub weight_decay: f64,
    /// Warmup ratio
    pub warmup_ratio: f32,
    /// Gradient accumulation steps
    pub gradient_accumulation_steps: usize,
    /// Maximum gradient norm
    pub max_grad_norm: f32,
    /// Enable gradient checkpointing
    pub gradient_checkpointing: bool,
    /// Mixed precision mode
    pub mixed_precision: Option<String>,
    /// Random seed
    pub seed: u64,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            epochs: 3,
            batch_size: 16,
            learning_rate: 2e-4,
            weight_decay: 0.01,
            warmup_ratio: 0.03,
            gradient_accumulation_steps: 1,
            max_grad_norm: 1.0,
            gradient_checkpointing: false,
            mixed_precision: None,
            seed: 42,
        }
    }
}
