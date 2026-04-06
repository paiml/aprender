//! Model registry types and operations.
//!
//! Provides model versioning, model cards, and lifecycle management.

mod card;
mod stage;
mod version;

pub use card::ModelCard;
pub use stage::ModelStage;
pub use version::ModelVersion;

use crate::storage::ContentAddress;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Unique identifier for a registered model.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ModelId(Uuid);

impl ModelId {
    /// Create a new random model ID.
    #[must_use]
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }

    /// Create from a UUID.
    #[must_use]
    pub fn from_uuid(uuid: Uuid) -> Self {
        Self(uuid)
    }

    /// Get the underlying UUID.
    #[must_use]
    pub fn as_uuid(&self) -> &Uuid {
        &self.0
    }
}

impl Default for ModelId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for ModelId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::str::FromStr for ModelId {
    type Err = uuid::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(Self(Uuid::parse_str(s)?))
    }
}

/// Reference to a model (name + version).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelReference {
    /// Model name.
    pub name: String,
    /// Model version.
    pub version: ModelVersion,
}

impl ModelReference {
    /// Create a new model reference.
    #[must_use]
    pub fn new(name: impl Into<String>, version: ModelVersion) -> Self {
        Self { name: name.into(), version }
    }
}

impl std::fmt::Display for ModelReference {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}:{}", self.name, self.version)
    }
}

/// A registered model in the registry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Model {
    /// Unique identifier.
    pub id: ModelId,
    /// Model name.
    pub name: String,
    /// Model version.
    pub version: ModelVersion,
    /// Content address of the artifact.
    pub content_address: ContentAddress,
    /// Model card with metadata.
    pub card: ModelCard,
    /// Current lifecycle stage.
    pub stage: ModelStage,
    /// Registration timestamp.
    pub created_at: DateTime<Utc>,
    /// Last updated timestamp.
    pub updated_at: DateTime<Utc>,
}

impl Model {
    /// Create a reference to this model.
    #[must_use]
    pub fn reference(&self) -> ModelReference {
        ModelReference::new(&self.name, self.version.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_id_generation() {
        let id1 = ModelId::new();
        let id2 = ModelId::new();
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_model_id_from_str() {
        let id = ModelId::new();
        let s = id.to_string();
        let parsed: ModelId = s.parse().unwrap();
        assert_eq!(id, parsed);
    }

    #[test]
    fn test_model_reference_display() {
        let reference = ModelReference::new("fraud-detector", ModelVersion::new(1, 2, 3));
        assert_eq!(reference.to_string(), "fraud-detector:1.2.3");
    }
}
