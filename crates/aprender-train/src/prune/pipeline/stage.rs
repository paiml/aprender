//! Pruning pipeline stage enum
//!
//! Defines the stages of the pruning pipeline workflow.

use serde::{Deserialize, Serialize};

/// Current stage of the pruning pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum PruningStage {
    /// Not started.
    #[default]
    Idle,
    /// Collecting calibration data.
    Calibrating,
    /// Computing importance scores.
    ComputingImportance,
    /// Applying pruning masks.
    Pruning,
    /// Fine-tuning after pruning.
    FineTuning,
    /// Evaluating pruned model.
    Evaluating,
    /// Exporting pruned model.
    Exporting,
    /// Pipeline complete.
    Complete,
    /// Pipeline failed.
    Failed,
}

impl PruningStage {
    /// Check if the pipeline is in an active (non-terminal) state.
    pub fn is_active(&self) -> bool {
        matches!(
            self,
            PruningStage::Calibrating
                | PruningStage::ComputingImportance
                | PruningStage::Pruning
                | PruningStage::FineTuning
                | PruningStage::Evaluating
                | PruningStage::Exporting
        )
    }

    /// Check if the pipeline is complete (success or failure).
    pub fn is_terminal(&self) -> bool {
        matches!(self, PruningStage::Complete | PruningStage::Failed)
    }

    /// Get display name for the stage.
    pub fn display_name(&self) -> &'static str {
        match self {
            PruningStage::Idle => "Idle",
            PruningStage::Calibrating => "Calibrating",
            PruningStage::ComputingImportance => "Computing Importance",
            PruningStage::Pruning => "Pruning",
            PruningStage::FineTuning => "Fine-Tuning",
            PruningStage::Evaluating => "Evaluating",
            PruningStage::Exporting => "Exporting",
            PruningStage::Complete => "Complete",
            PruningStage::Failed => "Failed",
        }
    }
}
