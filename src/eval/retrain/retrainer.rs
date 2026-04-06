//! Auto-retrainer implementation.

use super::action::Action;
use super::config::RetrainConfig;
use super::policy::RetrainPolicy;
use crate::error::Result;
use crate::eval::drift::{DriftDetector, DriftResult, DriftSummary, Severity};

/// Callback type for retrain triggers
pub type RetrainCallback = Box<dyn Fn(&[DriftResult]) -> Result<String> + Send + Sync>;

/// Auto-retrainer that monitors drift and triggers retraining
pub struct AutoRetrainer {
    detector: DriftDetector,
    config: RetrainConfig,
    retrain_callback: Option<RetrainCallback>,
    batches_since_retrain: usize,
    total_retrains: usize,
}

impl AutoRetrainer {
    /// Create a new auto-retrainer with given detector and config
    pub fn new(detector: DriftDetector, config: RetrainConfig) -> Self {
        Self {
            detector,
            config,
            retrain_callback: None,
            batches_since_retrain: 0,
            total_retrains: 0,
        }
    }

    /// Set the callback to invoke when retraining is triggered
    ///
    /// The callback receives the drift results and should return a job ID
    /// or an error if retraining failed to start.
    pub fn on_retrain<F>(&mut self, callback: F)
    where
        F: Fn(&[DriftResult]) -> Result<String> + Send + Sync + 'static,
    {
        self.retrain_callback = Some(Box::new(callback));
    }

    /// Process a batch of data and check for drift
    ///
    /// Returns the action taken based on drift detection and policy.
    pub fn process_batch(&mut self, batch: &[Vec<f64>]) -> Result<Action> {
        self.batches_since_retrain += 1;

        // Check for drift
        let results = self.detector.check(batch);

        if results.is_empty() {
            return Ok(Action::None);
        }

        let summary = DriftDetector::summary(&results);

        // Check if we're in cooldown
        if self.batches_since_retrain < self.config.cooldown_batches {
            if summary.has_drift() && self.config.log_warnings {
                return Ok(Action::WarningLogged);
            }
            return Ok(Action::None);
        }

        // Check max retrains limit
        if self.config.max_retrains > 0 && self.total_retrains >= self.config.max_retrains {
            if summary.has_drift() && self.config.log_warnings {
                return Ok(Action::WarningLogged);
            }
            return Ok(Action::None);
        }

        // Evaluate policy
        let should_retrain = self.evaluate_policy(&results, &summary);

        if should_retrain {
            if let Some(ref callback) = self.retrain_callback {
                let job_id = callback(&results)?;
                self.batches_since_retrain = 0;
                self.total_retrains += 1;
                return Ok(Action::RetrainTriggered(job_id));
            }
            // No callback set but policy says retrain
            return Ok(Action::WarningLogged);
        }

        if summary.warnings > 0 && self.config.log_warnings {
            return Ok(Action::WarningLogged);
        }

        Ok(Action::None)
    }

    /// Evaluate whether retraining should be triggered based on policy
    fn evaluate_policy(&self, results: &[DriftResult], summary: &DriftSummary) -> bool {
        match &self.config.policy {
            RetrainPolicy::FeatureCount { count } => summary.drifted_features >= *count,

            RetrainPolicy::CriticalFeature { names } => {
                results.iter().any(|r| r.drifted && names.contains(&r.feature))
            }

            RetrainPolicy::DriftPercentage { threshold } => {
                summary.drift_percentage() >= *threshold
            }

            RetrainPolicy::AnyCritical => results.iter().any(|r| r.severity == Severity::Critical),
        }
    }

    /// Get the underlying drift detector
    pub fn detector(&self) -> &DriftDetector {
        &self.detector
    }

    /// Get mutable reference to drift detector (for setting baseline)
    pub fn detector_mut(&mut self) -> &mut DriftDetector {
        &mut self.detector
    }

    /// Get statistics about retraining
    pub fn stats(&self) -> RetrainerStats {
        RetrainerStats {
            total_retrains: self.total_retrains,
            batches_since_retrain: self.batches_since_retrain,
        }
    }

    /// Reset the cooldown counter
    pub fn reset_cooldown(&mut self) {
        self.batches_since_retrain = self.config.cooldown_batches;
    }
}

/// Statistics about the auto-retrainer
#[derive(Clone, Debug)]
pub struct RetrainerStats {
    /// Total number of retrains triggered
    pub total_retrains: usize,
    /// Batches processed since last retrain
    pub batches_since_retrain: usize,
}
