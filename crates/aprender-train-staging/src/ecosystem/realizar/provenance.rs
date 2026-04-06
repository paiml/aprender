//! Experiment provenance tracking.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Experiment provenance for tracking model lineage.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExperimentProvenance {
    /// Experiment identifier
    pub experiment_id: String,
    /// Run identifier within experiment
    pub run_id: String,
    /// Training configuration hash
    pub config_hash: String,
    /// Dataset identifier
    pub dataset_id: Option<String>,
    /// Base model identifier (for fine-tuned models)
    pub base_model_id: Option<String>,
    /// Training metrics at export time
    pub metrics: HashMap<String, f64>,
    /// Timestamp of export
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Git commit hash (if available)
    pub git_commit: Option<String>,
    /// Additional custom metadata
    pub custom: HashMap<String, String>,
}

impl ExperimentProvenance {
    /// Create new provenance with required fields.
    pub fn new(experiment_id: impl Into<String>, run_id: impl Into<String>) -> Self {
        Self {
            experiment_id: experiment_id.into(),
            run_id: run_id.into(),
            config_hash: String::new(),
            dataset_id: None,
            base_model_id: None,
            metrics: HashMap::new(),
            timestamp: chrono::Utc::now(),
            git_commit: None,
            custom: HashMap::new(),
        }
    }

    /// Set configuration hash.
    pub fn with_config_hash(mut self, hash: impl Into<String>) -> Self {
        self.config_hash = hash.into();
        self
    }

    /// Set dataset identifier.
    pub fn with_dataset(mut self, dataset_id: impl Into<String>) -> Self {
        self.dataset_id = Some(dataset_id.into());
        self
    }

    /// Set base model identifier.
    pub fn with_base_model(mut self, model_id: impl Into<String>) -> Self {
        self.base_model_id = Some(model_id.into());
        self
    }

    /// Add a metric.
    pub fn with_metric(mut self, name: impl Into<String>, value: f64) -> Self {
        self.metrics.insert(name.into(), value);
        self
    }

    /// Add multiple metrics.
    pub fn with_metrics(mut self, metrics: impl IntoIterator<Item = (String, f64)>) -> Self {
        self.metrics.extend(metrics);
        self
    }

    /// Set git commit hash.
    pub fn with_git_commit(mut self, commit: impl Into<String>) -> Self {
        self.git_commit = Some(commit.into());
        self
    }

    /// Add custom metadata.
    pub fn with_custom(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.custom.insert(key.into(), value.into());
        self
    }

    /// Convert to GGUF metadata key-value pairs.
    pub fn to_metadata_pairs(&self) -> Vec<(String, String)> {
        let mut pairs = vec![
            ("entrenar.experiment_id".to_string(), self.experiment_id.clone()),
            ("entrenar.run_id".to_string(), self.run_id.clone()),
            ("entrenar.timestamp".to_string(), self.timestamp.to_rfc3339()),
        ];

        if !self.config_hash.is_empty() {
            pairs.push(("entrenar.config_hash".to_string(), self.config_hash.clone()));
        }

        if let Some(ref dataset) = self.dataset_id {
            pairs.push(("entrenar.dataset_id".to_string(), dataset.clone()));
        }

        if let Some(ref base) = self.base_model_id {
            pairs.push(("entrenar.base_model_id".to_string(), base.clone()));
        }

        if let Some(ref commit) = self.git_commit {
            pairs.push(("entrenar.git_commit".to_string(), commit.clone()));
        }

        for (key, value) in &self.metrics {
            pairs.push((format!("entrenar.metric.{key}"), value.to_string()));
        }

        for (key, value) in &self.custom {
            pairs.push((format!("entrenar.custom.{key}"), value.clone()));
        }

        pairs
    }
}
