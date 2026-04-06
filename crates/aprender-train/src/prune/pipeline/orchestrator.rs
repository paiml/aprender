//! Prune-Finetune pipeline orchestrator
//!
//! Manages the full pruning workflow from calibration through export.

use super::metrics::PruningMetrics;
use super::stage::PruningStage;
use crate::prune::calibrate::CalibrationCollector;
use crate::prune::config::PruningConfig;

/// Prune-Finetune pipeline orchestrator.
///
/// Manages the full pruning workflow from calibration through export.
#[derive(Debug)]
pub struct PruneFinetunePipeline {
    /// Configuration.
    config: PruningConfig,
    /// Current stage.
    stage: PruningStage,
    /// Collected metrics.
    metrics: PruningMetrics,
    /// Calibration collector.
    calibration: Option<CalibrationCollector>,
    /// Error message if failed.
    error: Option<String>,
}

impl PruneFinetunePipeline {
    /// Create a new pipeline with the given configuration.
    pub fn new(config: PruningConfig) -> Self {
        let metrics = PruningMetrics::new(config.target_sparsity());
        Self { config, stage: PruningStage::Idle, metrics, calibration: None, error: None }
    }

    /// Get the current stage.
    pub fn stage(&self) -> PruningStage {
        self.stage
    }

    /// Get the configuration.
    pub fn config(&self) -> &PruningConfig {
        &self.config
    }

    /// Get the collected metrics.
    pub fn metrics(&self) -> &PruningMetrics {
        &self.metrics
    }

    /// Get mutable access to metrics.
    pub fn metrics_mut(&mut self) -> &mut PruningMetrics {
        &mut self.metrics
    }

    /// Get the error message if failed.
    pub fn error(&self) -> Option<&str> {
        self.error.as_deref()
    }

    /// Start the calibration stage.
    pub fn start_calibration(&mut self, calibration: CalibrationCollector) {
        if self.stage != PruningStage::Idle {
            return;
        }
        self.calibration = Some(calibration);
        self.stage = PruningStage::Calibrating;
    }

    /// Advance to the next stage.
    pub fn advance(&mut self) {
        self.stage = match self.stage {
            PruningStage::Idle => PruningStage::Calibrating,
            PruningStage::Calibrating => PruningStage::ComputingImportance,
            PruningStage::ComputingImportance => PruningStage::Pruning,
            PruningStage::Pruning => {
                if self.config.fine_tune_after_pruning() {
                    PruningStage::FineTuning
                } else {
                    PruningStage::Evaluating
                }
            }
            PruningStage::FineTuning => PruningStage::Evaluating,
            PruningStage::Evaluating => PruningStage::Exporting,
            PruningStage::Exporting => PruningStage::Complete,
            // Terminal states don't advance
            PruningStage::Complete | PruningStage::Failed => self.stage,
        };
    }

    /// Mark the pipeline as failed with an error message.
    pub fn fail(&mut self, error: impl Into<String>) {
        self.error = Some(error.into());
        self.stage = PruningStage::Failed;
    }

    /// Execute the export stage.
    ///
    /// This is called when the pipeline reaches the `Exporting` stage.
    /// Exports the pruned model weights and sparsity metadata.
    ///
    /// Returns `Ok(())` and advances to `Complete` on success, or
    /// sets the pipeline to `Failed` on error.
    pub fn execute_export(
        &mut self,
        weights: &std::collections::HashMap<String, Vec<f32>>,
        shapes: &std::collections::HashMap<String, Vec<usize>>,
        output_dir: impl AsRef<std::path::Path>,
        filename: &str,
    ) -> Result<super::sparse_export::SparseExportResult, String> {
        if self.stage != PruningStage::Exporting {
            return Err(format!("Cannot export in stage {:?}, expected Exporting", self.stage));
        }

        match super::sparse_export::export_sparse_model(
            weights,
            shapes,
            &self.metrics,
            output_dir,
            filename,
        ) {
            Ok(result) => {
                self.advance(); // -> Complete
                Ok(result)
            }
            Err(e) => {
                self.fail(format!("Export failed: {e}"));
                Err(format!("Export failed: {e}"))
            }
        }
    }

    /// Reset the pipeline to idle state.
    pub fn reset(&mut self) {
        self.stage = PruningStage::Idle;
        self.metrics = PruningMetrics::new(self.config.target_sparsity());
        self.calibration = None;
        self.error = None;
    }

    /// Check if the pipeline is complete (success or failure).
    pub fn is_complete(&self) -> bool {
        self.stage.is_terminal()
    }

    /// Check if the pipeline succeeded.
    pub fn succeeded(&self) -> bool {
        self.stage == PruningStage::Complete
    }

    /// Check if the pipeline failed.
    pub fn failed(&self) -> bool {
        self.stage == PruningStage::Failed
    }

    /// Get calibration collector if available.
    pub fn calibration(&self) -> Option<&CalibrationCollector> {
        self.calibration.as_ref()
    }

    /// Get calibration progress (0.0 to 1.0).
    pub fn calibration_progress(&self) -> f32 {
        self.calibration.as_ref().map_or(0.0, CalibrationCollector::progress)
    }

    /// Get overall pipeline progress (0.0 to 1.0).
    pub fn overall_progress(&self) -> f32 {
        match self.stage {
            PruningStage::Idle => 0.0,
            PruningStage::Calibrating => 0.1 + 0.1 * self.calibration_progress(),
            PruningStage::ComputingImportance => 0.25,
            PruningStage::Pruning => 0.4,
            PruningStage::FineTuning => 0.6,
            PruningStage::Evaluating => 0.8,
            PruningStage::Exporting => 0.95,
            PruningStage::Complete => 1.0,
            PruningStage::Failed => 0.0, // Reset on failure
        }
    }
}

impl Clone for PruneFinetunePipeline {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            stage: self.stage,
            metrics: self.metrics.clone(),
            calibration: self.calibration.clone(),
            error: self.error.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_pipeline() -> PruneFinetunePipeline {
        PruneFinetunePipeline::new(PruningConfig::new())
    }

    #[test]
    fn test_advance_from_idle() {
        let mut p = make_pipeline();
        assert_eq!(p.stage(), PruningStage::Idle);
        p.advance();
        assert_eq!(p.stage(), PruningStage::Calibrating);
    }

    #[test]
    fn test_advance_full_pipeline_with_finetune() {
        // Default config has fine_tune_after_pruning=true
        let mut p = make_pipeline();

        // Idle → Calibrating
        p.advance();
        assert_eq!(p.stage(), PruningStage::Calibrating);

        // Calibrating → ComputingImportance
        p.advance();
        assert_eq!(p.stage(), PruningStage::ComputingImportance);

        // ComputingImportance → Pruning
        p.advance();
        assert_eq!(p.stage(), PruningStage::Pruning);

        // Pruning → FineTuning (fine_tune_after_pruning=true)
        p.advance();
        assert_eq!(p.stage(), PruningStage::FineTuning);

        // FineTuning → Evaluating
        p.advance();
        assert_eq!(p.stage(), PruningStage::Evaluating);

        // Evaluating → Exporting
        p.advance();
        assert_eq!(p.stage(), PruningStage::Exporting);

        // Exporting → Complete
        p.advance();
        assert_eq!(p.stage(), PruningStage::Complete);

        // Complete stays Complete
        p.advance();
        assert_eq!(p.stage(), PruningStage::Complete);
    }

    #[test]
    fn test_advance_skip_finetune() {
        let config = PruningConfig::new().with_fine_tune(false);
        let mut p = PruneFinetunePipeline::new(config);
        // Advance to Pruning
        p.advance(); // Calibrating
        p.advance(); // ComputingImportance
        p.advance(); // Pruning
                     // Pruning → Evaluating (fine_tune_after_pruning=false)
        p.advance();
        assert_eq!(p.stage(), PruningStage::Evaluating);
    }

    #[test]
    fn test_advance_failed_stays_failed() {
        let mut p = make_pipeline();
        p.fail("test error");
        assert_eq!(p.stage(), PruningStage::Failed);
        p.advance();
        assert_eq!(p.stage(), PruningStage::Failed);
    }

    #[test]
    fn test_overall_progress_all_stages() {
        // Default config has fine_tune_after_pruning=true
        let mut p = make_pipeline();

        // Idle → 0.0
        assert_eq!(p.overall_progress(), 0.0);

        // Calibrating → ~0.1
        p.advance();
        assert!(p.overall_progress() >= 0.1);

        // ComputingImportance → 0.25
        p.advance();
        assert_eq!(p.overall_progress(), 0.25);

        // Pruning → 0.4
        p.advance();
        assert_eq!(p.overall_progress(), 0.4);

        // FineTuning → 0.6
        p.advance();
        assert_eq!(p.overall_progress(), 0.6);

        // Evaluating → 0.8
        p.advance();
        assert_eq!(p.overall_progress(), 0.8);

        // Exporting → 0.95
        p.advance();
        assert_eq!(p.overall_progress(), 0.95);

        // Complete → 1.0
        p.advance();
        assert_eq!(p.overall_progress(), 1.0);
    }

    #[test]
    fn test_overall_progress_failed() {
        let mut p = make_pipeline();
        p.fail("test");
        assert_eq!(p.overall_progress(), 0.0);
    }

    #[test]
    fn test_reset_to_idle() {
        let mut p = make_pipeline();
        p.advance();
        p.advance();
        p.reset();
        assert_eq!(p.stage(), PruningStage::Idle);
        assert!(p.error().is_none());
    }
}
