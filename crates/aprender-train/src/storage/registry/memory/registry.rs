//! InMemoryRegistry struct and core methods

use std::collections::HashMap;

use super::super::policy::PromotionPolicy;
use super::super::stage::ModelStage;
use super::super::transition::StageTransition;
use super::super::version::ModelVersion;

/// In-memory model registry for testing
#[derive(Debug, Default)]
pub struct InMemoryRegistry {
    /// Models by name -> version -> ModelVersion
    pub(crate) models: HashMap<String, HashMap<u32, ModelVersion>>,
    /// Stage transition history
    pub(crate) transitions: Vec<StageTransition>,
    /// Promotion policies by stage
    pub(crate) policies: HashMap<ModelStage, PromotionPolicy>,
    /// Auto-rollback configuration
    pub(crate) rollback_enabled: HashMap<String, (String, f64)>, // model -> (metric, threshold)
}

impl InMemoryRegistry {
    /// Create a new in-memory registry
    pub fn new() -> Self {
        Self::default()
    }

    /// Enable auto-rollback for a model
    pub fn enable_auto_rollback(&mut self, model: &str, metric: &str, threshold: f64) {
        self.rollback_enabled.insert(model.to_string(), (metric.to_string(), threshold));
    }

    /// Check if rollback is needed based on metrics
    pub fn check_rollback(&self, model: &str, current_metric: f64) -> bool {
        if let Some((_, threshold)) = self.rollback_enabled.get(model) {
            current_metric < *threshold
        } else {
            false
        }
    }

    /// Get next version number for a model
    pub(crate) fn next_version(&self, name: &str) -> u32 {
        self.models.get(name).map_or(1, |versions| versions.keys().max().copied().unwrap_or(0) + 1)
    }
}
