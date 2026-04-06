//! Model version metadata

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::stage::ModelStage;

/// Model version metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelVersion {
    /// Model name
    pub name: String,
    /// Version number (monotonically increasing)
    pub version: u32,
    /// Current stage
    pub stage: ModelStage,
    /// URI to model artifacts
    pub artifact_uri: String,
    /// Performance metrics
    pub metrics: HashMap<String, f64>,
    /// Tags for organization
    pub tags: HashMap<String, String>,
    /// Description
    pub description: Option<String>,
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
    /// Last promotion timestamp
    pub promoted_at: Option<DateTime<Utc>>,
    /// User who last promoted
    pub promoted_by: Option<String>,
}

impl ModelVersion {
    /// Create a new model version
    pub fn new(name: &str, version: u32, artifact_uri: &str) -> Self {
        Self {
            name: name.to_string(),
            version,
            stage: ModelStage::None,
            artifact_uri: artifact_uri.to_string(),
            metrics: HashMap::new(),
            tags: HashMap::new(),
            description: None,
            created_at: Utc::now(),
            promoted_at: None,
            promoted_by: None,
        }
    }

    /// Add a metric
    pub fn with_metric(mut self, name: &str, value: f64) -> Self {
        self.metrics.insert(name.to_string(), value);
        self
    }

    /// Add a tag
    pub fn with_tag(mut self, key: &str, value: &str) -> Self {
        self.tags.insert(key.to_string(), value.to_string());
        self
    }

    /// Set description
    pub fn with_description(mut self, desc: &str) -> Self {
        self.description = Some(desc.to_string());
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_version_new() {
        let model = ModelVersion::new("test-model", 1, "/path/to/model");
        assert_eq!(model.name, "test-model");
        assert_eq!(model.version, 1);
        assert_eq!(model.stage, ModelStage::None);
    }

    #[test]
    fn test_model_version_with_metric() {
        let model = ModelVersion::new("test", 1, "/path").with_metric("accuracy", 0.95);
        assert_eq!(model.metrics.get("accuracy"), Some(&0.95));
    }

    #[test]
    fn test_model_version_with_tag() {
        let model = ModelVersion::new("test", 1, "/path").with_tag("framework", "pytorch");
        assert_eq!(model.tags.get("framework"), Some(&"pytorch".to_string()));
    }

    #[test]
    fn test_model_version_with_description() {
        let model = ModelVersion::new("test", 1, "/path").with_description("A test model");
        assert_eq!(model.description, Some("A test model".to_string()));
    }
}
