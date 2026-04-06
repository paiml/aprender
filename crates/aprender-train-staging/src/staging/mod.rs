//! Model Staging Workflows (GH-70)
//!
//! Provides a lightweight model staging registry for managing model versions
//! through lifecycle stages: Dev -> Staging -> Production.
//!
//! Transition rules enforce no skipping: a model must progress through each
//! stage sequentially, and demotion follows the reverse path.
//!
//! # Example
//!
//! ```
//! use entrenar::staging::{Stage, StagingRegistry};
//!
//! let mut registry = StagingRegistry::new();
//! let mv = registry.register_model("llama-7b", "1.0.0", "/models/llama-7b-v1");
//! assert_eq!(mv.stage, Stage::Dev);
//!
//! registry.promote("llama-7b", "1.0.0", Stage::Staging).expect("promote to staging");
//! registry.promote("llama-7b", "1.0.0", Stage::Production).expect("promote to production");
//! ```

#[cfg(test)]
mod tests;

use std::collections::HashMap;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Model lifecycle stage.
///
/// Models progress linearly: Dev -> Staging -> Production.
/// No stage may be skipped during promotion or demotion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Stage {
    /// In active development
    Dev,
    /// Under validation/testing
    Staging,
    /// Deployed to production
    Production,
}

impl Stage {
    /// Numeric ordering for stage progression.
    fn ordinal(self) -> u8 {
        match self {
            Stage::Dev => 0,
            Stage::Staging => 1,
            Stage::Production => 2,
        }
    }

    /// Display name for the stage.
    pub fn as_str(self) -> &'static str {
        match self {
            Stage::Dev => "Dev",
            Stage::Staging => "Staging",
            Stage::Production => "Production",
        }
    }
}

impl std::fmt::Display for Stage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Metadata for a registered model version.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelVersion {
    /// Model name (e.g., "llama-7b-finetuned")
    pub name: String,
    /// Semantic version string (e.g., "1.0.0")
    pub version: String,
    /// Current lifecycle stage
    pub stage: Stage,
    /// Arbitrary key-value metadata
    pub metadata: HashMap<String, String>,
    /// When this version was registered
    pub created_at: DateTime<Utc>,
    /// When this version was last promoted or demoted (None if still at initial stage)
    pub promoted_at: Option<DateTime<Utc>>,
    /// Path to model artifacts
    pub path: String,
}

/// Errors from staging operations.
#[derive(Debug, Error)]
pub enum StagingError {
    /// The requested model/version was not found in the registry.
    #[error("model not found: {name} v{version}")]
    NotFound { name: String, version: String },

    /// The requested stage transition is not allowed.
    /// Only single-step transitions (adjacent stages) are permitted.
    #[error("invalid transition from {from} to {to} for {name} v{version}")]
    InvalidTransition { name: String, version: String, from: Stage, to: Stage },

    /// A model with this name and version already exists.
    #[error("model already exists: {name} v{version}")]
    AlreadyExists { name: String, version: String },
}

/// Result type for staging operations.
pub type Result<T> = std::result::Result<T, StagingError>;

/// Registry managing model versions and their lifecycle stages.
///
/// Models are keyed by (name, version) pairs. The registry enforces
/// sequential stage transitions: Dev -> Staging -> Production for
/// promotion, and the reverse for demotion.
#[derive(Debug, Default)]
pub struct StagingRegistry {
    /// All registered model versions, keyed by (name, version).
    models: HashMap<(String, String), ModelVersion>,
}

impl StagingRegistry {
    /// Create an empty staging registry.
    pub fn new() -> Self {
        Self { models: HashMap::new() }
    }

    /// Register a new model version at the Dev stage.
    ///
    /// # Panics
    ///
    /// Does not panic. Returns the existing version if already registered
    /// (idempotent behavior for re-registration at the same key).
    pub fn register_model(&mut self, name: &str, version: &str, path: &str) -> ModelVersion {
        let key = (name.to_string(), version.to_string());
        let mv = ModelVersion {
            name: name.to_string(),
            version: version.to_string(),
            stage: Stage::Dev,
            metadata: HashMap::new(),
            created_at: Utc::now(),
            promoted_at: None,
            path: path.to_string(),
        };
        self.models.entry(key).or_insert(mv).clone()
    }

    /// Promote a model version to the given target stage.
    ///
    /// The target must be exactly one stage above the current stage:
    /// - Dev -> Staging
    /// - Staging -> Production
    ///
    /// Skipping stages (e.g., Dev -> Production) is rejected.
    pub fn promote(&mut self, name: &str, version: &str, target: Stage) -> Result<ModelVersion> {
        let key = (name.to_string(), version.to_string());
        let mv = self.models.get_mut(&key).ok_or_else(|| StagingError::NotFound {
            name: name.to_string(),
            version: version.to_string(),
        })?;

        let current_ord = mv.stage.ordinal();
        let target_ord = target.ordinal();

        // Promote must go exactly one step up
        if target_ord != current_ord + 1 {
            return Err(StagingError::InvalidTransition {
                name: name.to_string(),
                version: version.to_string(),
                from: mv.stage,
                to: target,
            });
        }

        mv.stage = target;
        mv.promoted_at = Some(Utc::now());
        Ok(mv.clone())
    }

    /// Demote a model version to the given target stage.
    ///
    /// The target must be exactly one stage below the current stage:
    /// - Production -> Staging
    /// - Staging -> Dev
    ///
    /// Skipping stages (e.g., Production -> Dev) is rejected.
    pub fn demote(&mut self, name: &str, version: &str, target: Stage) -> Result<ModelVersion> {
        let key = (name.to_string(), version.to_string());
        let mv = self.models.get_mut(&key).ok_or_else(|| StagingError::NotFound {
            name: name.to_string(),
            version: version.to_string(),
        })?;

        let current_ord = mv.stage.ordinal();
        let target_ord = target.ordinal();

        // Demote must go exactly one step down
        if current_ord == 0 || target_ord != current_ord - 1 {
            return Err(StagingError::InvalidTransition {
                name: name.to_string(),
                version: version.to_string(),
                from: mv.stage,
                to: target,
            });
        }

        mv.stage = target;
        mv.promoted_at = Some(Utc::now());
        Ok(mv.clone())
    }

    /// Get the latest version of a model at the given stage.
    ///
    /// "Latest" is determined by `created_at` timestamp. Returns `None`
    /// if no version of the model exists at that stage.
    pub fn get_latest(&self, name: &str, stage: Stage) -> Option<&ModelVersion> {
        self.models
            .values()
            .filter(|mv| mv.name == name && mv.stage == stage)
            .max_by_key(|mv| mv.created_at)
    }

    /// List all versions of a model, sorted by creation time (oldest first).
    pub fn list_versions(&self, name: &str) -> Vec<&ModelVersion> {
        let mut versions: Vec<&ModelVersion> =
            self.models.values().filter(|mv| mv.name == name).collect();
        versions.sort_by_key(|mv| mv.created_at);
        versions
    }
}
