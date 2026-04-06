//! Pruning callback implementation
//!
//! This module contains the `PruningCallback` struct that integrates with
//! Entrenar's training callback system to apply pruning during training.

#![allow(clippy::field_reassign_with_default)]

use crate::prune::calibrate::{CalibrationCollector, CalibrationConfig};
use crate::prune::config::PruningConfig;
use crate::prune::schedule::PruningSchedule;
use crate::train::callback::{CallbackAction, CallbackContext, TrainerCallback};

/// Callback for applying pruning during training.
///
/// Integrates with the training loop to apply pruning at scheduled steps,
/// collect calibration data for activation-weighted methods, and log
/// pruning metrics.
///
/// # Toyota Way: Kaizen (Continuous Improvement)
/// Gradual pruning allows the model to adapt incrementally to sparsity.
///
/// # Example
///
/// ```ignore
/// use entrenar::prune::{PruningCallback, PruningConfig, PruningSchedule};
///
/// let config = PruningConfig::new()
///     .with_schedule(PruningSchedule::Gradual {
///         start_step: 1000,
///         end_step: 5000,
///         initial_sparsity: 0.0,
///         final_sparsity: 0.5,
///         frequency: 100,
///     })
///     .with_target_sparsity(0.5);
///
/// let callback = PruningCallback::new(config);
/// trainer.add_callback(callback);
/// ```
#[derive(Debug)]
pub struct PruningCallback {
    /// Configuration for pruning
    config: PruningConfig,
    /// Current achieved sparsity
    current_sparsity: f32,
    /// Total parameters pruned so far
    parameters_pruned: usize,
    /// Calibration data collector
    pub(crate) calibration: Option<CalibrationCollector>,
    /// Whether pruning is enabled
    enabled: bool,
    /// Step when last pruning occurred
    pub(crate) last_prune_step: Option<usize>,
}

impl PruningCallback {
    /// Create a new pruning callback with the given configuration.
    ///
    /// # Arguments
    ///
    /// * `config` - Pruning configuration
    pub fn new(config: PruningConfig) -> Self {
        let calibration = if config.requires_calibration() {
            Some(CalibrationCollector::new(CalibrationConfig::default()))
        } else {
            None
        };

        Self {
            config,
            current_sparsity: 0.0,
            parameters_pruned: 0,
            calibration,
            enabled: true,
            last_prune_step: None,
        }
    }

    /// Create a pruning callback with custom calibration configuration.
    pub fn with_calibration(config: PruningConfig, cal_config: CalibrationConfig) -> Self {
        Self { calibration: Some(CalibrationCollector::new(cal_config)), ..Self::new(config) }
    }

    /// Enable or disable the callback.
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Check if the callback is enabled.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Get the current achieved sparsity.
    pub fn current_sparsity(&self) -> f32 {
        self.current_sparsity
    }

    /// Get the target sparsity from the configuration.
    pub fn target_sparsity(&self) -> f32 {
        self.config.target_sparsity()
    }

    /// Get the total number of parameters pruned.
    pub fn parameters_pruned(&self) -> usize {
        self.parameters_pruned
    }

    /// Get the pruning schedule.
    pub fn schedule(&self) -> &PruningSchedule {
        self.config.schedule()
    }

    /// Check if pruning is complete.
    pub fn is_complete(&self) -> bool {
        self.last_prune_step.is_some_and(|step| self.config.schedule().is_complete(step))
    }

    /// Get the step at which pruning last occurred.
    pub fn last_prune_step(&self) -> Option<usize> {
        self.last_prune_step
    }

    /// Update current sparsity (for testing or manual updates).
    pub fn set_current_sparsity(&mut self, sparsity: f32) {
        self.current_sparsity = sparsity.clamp(0.0, 1.0);
    }

    /// Get the configuration.
    pub fn config(&self) -> &PruningConfig {
        &self.config
    }

    /// Check if pruning should be applied at the given step.
    pub(crate) fn should_prune(&self, step: usize) -> bool {
        if !self.enabled {
            return false;
        }
        let target = self.config.schedule().sparsity_at_step(step);
        target > self.current_sparsity && self.config.schedule().should_prune_at_step(step)
    }

    /// Compute progress through the pruning schedule (0.0 to 1.0).
    pub fn progress(&self) -> f32 {
        let target = self.config.target_sparsity();
        if target <= 0.0 {
            return 1.0;
        }
        (self.current_sparsity / target).clamp(0.0, 1.0)
    }
}

impl TrainerCallback for PruningCallback {
    fn on_train_begin(&mut self, _ctx: &CallbackContext) -> CallbackAction {
        // Validate configuration at training start
        if let Err(e) = self.config.schedule().validate() {
            eprintln!("[PruningCallback] Invalid schedule configuration: {e}");
            return CallbackAction::Stop;
        }
        CallbackAction::Continue
    }

    fn on_step_end(&mut self, ctx: &CallbackContext) -> CallbackAction {
        if !self.enabled {
            return CallbackAction::Continue;
        }

        let step = ctx.global_step;
        let target_sparsity = self.config.schedule().sparsity_at_step(step);

        // Check if we should prune at this step
        if self.should_prune(step) {
            // In a real implementation, we would:
            // 1. Collect calibration data if needed
            // 2. Compute importance scores
            // 3. Apply pruning masks
            // 4. Update metrics

            // For now, simulate pruning progress
            self.current_sparsity = target_sparsity;
            self.last_prune_step = Some(step);

            // Log pruning event (placeholder for actual logging)
            // In production, this would integrate with the monitoring system
        }

        CallbackAction::Continue
    }

    fn on_train_end(&mut self, _ctx: &CallbackContext) {
        // Log final pruning summary
        if self.parameters_pruned > 0 || self.current_sparsity > 0.0 {
            eprintln!(
                "[PruningCallback] Training complete. Final sparsity: {:.2}%, Parameters pruned: {}",
                self.current_sparsity * 100.0,
                self.parameters_pruned
            );
        }
    }

    fn name(&self) -> &'static str {
        "PruningCallback"
    }
}

impl Clone for PruningCallback {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            current_sparsity: self.current_sparsity,
            parameters_pruned: self.parameters_pruned,
            calibration: self.calibration.clone(),
            enabled: self.enabled,
            last_prune_step: self.last_prune_step,
        }
    }
}
