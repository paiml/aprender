//! Main pruning configuration struct.

use crate::prune::schedule::PruningSchedule;
use serde::{Deserialize, Serialize};

use super::{PruneMethod, SparsityPatternConfig};

/// Configuration for pruning operations.
///
/// # Example
///
/// ```
/// use entrenar::prune::{PruningConfig, PruningSchedule, PruneMethod};
///
/// let config = PruningConfig::default()
///     .with_method(PruneMethod::Wanda)
///     .with_schedule(PruningSchedule::Gradual {
///         start_step: 1000,
///         end_step: 5000,
///         initial_sparsity: 0.0,
///         final_sparsity: 0.5,
///         frequency: 100,
///     })
///     .with_target_sparsity(0.5);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PruningConfig {
    /// Pruning method to use.
    method: PruneMethod,

    /// Target sparsity (0.0 to 1.0).
    target_sparsity: f32,

    /// Sparsity pattern.
    pattern: SparsityPatternConfig,

    /// Pruning schedule.
    schedule: PruningSchedule,

    /// Whether to fine-tune after pruning.
    fine_tune_after_pruning: bool,

    /// Number of fine-tuning steps.
    fine_tune_steps: usize,

    /// Learning rate for fine-tuning.
    fine_tune_lr: f32,

    /// Whether to skip first and last layers (recommended for LLMs).
    skip_embed_layers: bool,
}

impl Default for PruningConfig {
    fn default() -> Self {
        Self {
            method: PruneMethod::default(),
            target_sparsity: 0.5,
            pattern: SparsityPatternConfig::default(),
            schedule: PruningSchedule::default(),
            fine_tune_after_pruning: true,
            fine_tune_steps: 1000,
            fine_tune_lr: 1e-5,
            skip_embed_layers: true,
        }
    }
}

impl PruningConfig {
    /// Create a new configuration with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the pruning method.
    pub fn with_method(mut self, method: PruneMethod) -> Self {
        self.method = method;
        self
    }

    /// Set the target sparsity.
    pub fn with_target_sparsity(mut self, sparsity: f32) -> Self {
        self.target_sparsity = sparsity.clamp(0.0, 1.0);
        self
    }

    /// Set the sparsity pattern.
    pub fn with_pattern(mut self, pattern: SparsityPatternConfig) -> Self {
        self.pattern = pattern;
        self
    }

    /// Set the pruning schedule.
    pub fn with_schedule(mut self, schedule: PruningSchedule) -> Self {
        self.schedule = schedule;
        self
    }

    /// Enable or disable fine-tuning after pruning.
    pub fn with_fine_tune(mut self, enabled: bool) -> Self {
        self.fine_tune_after_pruning = enabled;
        self
    }

    /// Set the number of fine-tuning steps.
    pub fn with_fine_tune_steps(mut self, steps: usize) -> Self {
        self.fine_tune_steps = steps;
        self
    }

    /// Set the fine-tuning learning rate.
    pub fn with_fine_tune_lr(mut self, lr: f32) -> Self {
        self.fine_tune_lr = lr;
        self
    }

    /// Enable or disable skipping embedding layers.
    pub fn with_skip_embed_layers(mut self, skip: bool) -> Self {
        self.skip_embed_layers = skip;
        self
    }

    /// Get the pruning method.
    pub fn method(&self) -> PruneMethod {
        self.method
    }

    /// Get the target sparsity.
    pub fn target_sparsity(&self) -> f32 {
        self.target_sparsity
    }

    /// Get the sparsity pattern.
    pub fn pattern(&self) -> &SparsityPatternConfig {
        &self.pattern
    }

    /// Get the pruning schedule.
    pub fn schedule(&self) -> &PruningSchedule {
        &self.schedule
    }

    /// Check if fine-tuning is enabled.
    pub fn fine_tune_after_pruning(&self) -> bool {
        self.fine_tune_after_pruning
    }

    /// Get fine-tuning steps.
    pub fn fine_tune_steps(&self) -> usize {
        self.fine_tune_steps
    }

    /// Get fine-tuning learning rate.
    pub fn fine_tune_lr(&self) -> f32 {
        self.fine_tune_lr
    }

    /// Check if embedding layers should be skipped.
    pub fn skip_embed_layers(&self) -> bool {
        self.skip_embed_layers
    }

    /// Check if this configuration requires calibration data.
    pub fn requires_calibration(&self) -> bool {
        self.method.requires_calibration()
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<(), String> {
        // Validate schedule
        self.schedule.validate()?;

        // Validate target sparsity
        if self.target_sparsity < 0.0 || self.target_sparsity > 1.0 {
            return Err(format!(
                "target_sparsity ({}) must be between 0.0 and 1.0",
                self.target_sparsity
            ));
        }

        // Validate N:M pattern
        if let SparsityPatternConfig::NM { n, m } = &self.pattern {
            if *n >= *m {
                return Err(format!("N ({n}) must be less than M ({m})"));
            }
            if *m == 0 {
                return Err("M cannot be 0".to_string());
            }
        }

        // Validate block pattern
        if let SparsityPatternConfig::Block { height, width } = &self.pattern {
            if *height == 0 || *width == 0 {
                return Err("Block dimensions must be non-zero".to_string());
            }
        }

        Ok(())
    }
}
