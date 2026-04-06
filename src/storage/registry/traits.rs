//! Model registry trait definition

use std::collections::HashMap;

use super::comparison::VersionComparison;
use super::error::Result;
use super::policy::{PolicyCheckResult, PromotionPolicy};
use super::stage::ModelStage;
use super::transition::StageTransition;
use super::version::ModelVersion;

/// Model registry trait
pub trait ModelRegistry: Send + Sync {
    /// Register a new model version
    fn register_model(&mut self, name: &str, artifact_uri: &str) -> Result<ModelVersion>;

    /// Get a model version
    fn get_model(&self, name: &str, version: u32) -> Result<ModelVersion>;

    /// Get latest version of a model
    fn get_latest(&self, name: &str) -> Result<ModelVersion>;

    /// Get latest version at a specific stage
    fn get_latest_by_stage(&self, name: &str, stage: ModelStage) -> Option<ModelVersion>;

    /// List all versions of a model
    fn list_versions(&self, name: &str) -> Result<Vec<ModelVersion>>;

    /// Transition model to new stage
    fn transition_stage(
        &mut self,
        name: &str,
        version: u32,
        target_stage: ModelStage,
        user: Option<&str>,
    ) -> Result<()>;

    /// Compare two versions
    fn compare_versions(&self, name: &str, v1: u32, v2: u32) -> Result<VersionComparison>;

    /// Log metrics for a model version
    fn log_metrics(
        &mut self,
        name: &str,
        version: u32,
        metrics: HashMap<String, f64>,
    ) -> Result<()>;

    /// Get transition history for a model
    fn get_transition_history(&self, name: &str) -> Result<Vec<StageTransition>>;

    /// Set promotion policy for a stage
    fn set_policy(&mut self, policy: PromotionPolicy);

    /// Get promotion policy for a stage
    fn get_policy(&self, stage: ModelStage) -> Option<&PromotionPolicy>;

    /// Check if model can be promoted (with policy check)
    fn can_promote(
        &self,
        name: &str,
        version: u32,
        target_stage: ModelStage,
        approvals: u32,
    ) -> Result<PolicyCheckResult>;
}
