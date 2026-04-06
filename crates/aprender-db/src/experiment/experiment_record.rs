//! Experiment Record - root entity for experiment tracking

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Experiment Record represents a tracked experiment.
///
/// This is the root entity in the experiment tracking schema.
/// Each experiment can have multiple runs.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ExperimentRecord {
    experiment_id: String,
    name: String,
    created_at: DateTime<Utc>,
    config: Option<serde_json::Value>,
}

impl ExperimentRecord {
    /// Create a new experiment record with the given ID and name.
    ///
    /// # Arguments
    ///
    /// * `experiment_id` - Unique identifier for the experiment
    /// * `name` - Human-readable name for the experiment
    ///
    /// # Returns
    ///
    /// A new `ExperimentRecord` with the current timestamp.
    #[must_use]
    pub fn new(experiment_id: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            experiment_id: experiment_id.into(),
            name: name.into(),
            created_at: Utc::now(),
            config: None,
        }
    }

    /// Create a builder for constructing an experiment record with optional fields.
    #[must_use]
    pub fn builder(
        experiment_id: impl Into<String>,
        name: impl Into<String>,
    ) -> ExperimentRecordBuilder {
        ExperimentRecordBuilder::new(experiment_id, name)
    }

    /// Get the experiment ID.
    #[must_use]
    pub fn experiment_id(&self) -> &str {
        &self.experiment_id
    }

    /// Get the experiment name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get the creation timestamp.
    #[must_use]
    pub const fn created_at(&self) -> DateTime<Utc> {
        self.created_at
    }

    /// Get the experiment configuration, if any.
    #[must_use]
    pub const fn config(&self) -> Option<&serde_json::Value> {
        self.config.as_ref()
    }
}

/// Builder for `ExperimentRecord`.
#[derive(Debug)]
pub struct ExperimentRecordBuilder {
    experiment_id: String,
    name: String,
    created_at: DateTime<Utc>,
    config: Option<serde_json::Value>,
}

impl ExperimentRecordBuilder {
    /// Create a new builder with required fields.
    #[must_use]
    pub fn new(experiment_id: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            experiment_id: experiment_id.into(),
            name: name.into(),
            created_at: Utc::now(),
            config: None,
        }
    }

    /// Set the experiment configuration.
    #[must_use]
    pub fn config(mut self, config: serde_json::Value) -> Self {
        self.config = Some(config);
        self
    }

    /// Set a custom creation timestamp (useful for deserialization/testing).
    #[must_use]
    pub const fn created_at(mut self, created_at: DateTime<Utc>) -> Self {
        self.created_at = created_at;
        self
    }

    /// Build the `ExperimentRecord`.
    #[must_use]
    pub fn build(self) -> ExperimentRecord {
        ExperimentRecord {
            experiment_id: self.experiment_id,
            name: self.name,
            created_at: self.created_at,
            config: self.config,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_experiment_record_new() {
        let record = ExperimentRecord::new("test-id", "test-name");
        assert_eq!(record.experiment_id(), "test-id");
        assert_eq!(record.name(), "test-name");
    }

    #[test]
    fn test_experiment_record_builder() {
        let config = serde_json::json!({"key": "value"});
        let record =
            ExperimentRecord::builder("test-id", "test-name").config(config.clone()).build();

        assert_eq!(record.config(), Some(&config));
    }
}
