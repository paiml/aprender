//! Experiment tracking for training runs.

use crate::recipe::{Hyperparameters, RecipeReference};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Unique identifier for an experiment run.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RunId(Uuid);

impl RunId {
    /// Create a new random run ID.
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

impl Default for RunId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for RunId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::str::FromStr for RunId {
    type Err = uuid::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(Self(Uuid::parse_str(s)?))
    }
}

/// Status of an experiment run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RunStatus {
    /// Run is pending start.
    Pending,
    /// Run is currently executing.
    Running,
    /// Run completed successfully.
    Completed,
    /// Run failed with an error.
    Failed,
    /// Run was cancelled.
    Cancelled,
}

impl std::fmt::Display for RunStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::Pending => "pending",
            Self::Running => "running",
            Self::Completed => "completed",
            Self::Failed => "failed",
            Self::Cancelled => "cancelled",
        };
        write!(f, "{s}")
    }
}

/// Information about hardware used for a run.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HardwareInfo {
    /// CPU model.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cpu_model: Option<String>,
    /// Number of CPU cores used.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cpu_cores: Option<usize>,
    /// RAM in GB.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ram_gb: Option<usize>,
    /// GPU model.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gpu_model: Option<String>,
    /// Number of GPUs used.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gpu_count: Option<usize>,
}

/// A metric recorded during training.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricRecord {
    /// Metric name.
    pub name: String,
    /// Metric value.
    pub value: f64,
    /// Training step.
    pub step: u64,
    /// Timestamp.
    pub timestamp: DateTime<Utc>,
}

impl MetricRecord {
    /// Create a new metric record.
    #[must_use]
    pub fn new(name: impl Into<String>, value: f64, step: u64) -> Self {
        Self { name: name.into(), value, step, timestamp: Utc::now() }
    }
}

/// Reference to an artifact produced by a run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtifactReference {
    /// Artifact type (e.g., "model", "checkpoint").
    pub artifact_type: String,
    /// Artifact name.
    pub name: String,
    /// Content hash.
    pub content_hash: String,
}

/// An experiment run tracking a training execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentRun {
    /// Unique run identifier.
    pub run_id: RunId,
    /// Recipe used for this run.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub recipe: Option<RecipeReference>,
    /// Actual hyperparameters used (may override recipe).
    pub hyperparameters: Hyperparameters,

    /// When the run started.
    pub started_at: DateTime<Utc>,
    /// When the run finished.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finished_at: Option<DateTime<Utc>>,
    /// Current status.
    pub status: RunStatus,
    /// Hardware used.
    pub hardware: HardwareInfo,

    /// Metrics recorded during training.
    #[serde(default)]
    pub metrics: Vec<MetricRecord>,
    /// Artifacts produced.
    #[serde(default)]
    pub artifacts: Vec<ArtifactReference>,
    /// Log URI.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub log_uri: Option<String>,

    /// Git commit hash.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub git_commit: Option<String>,
    /// Whether the git working directory was dirty.
    #[serde(default)]
    pub git_dirty: bool,

    /// Error message if failed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error_message: Option<String>,

    /// Additional metadata.
    #[serde(default)]
    pub extra: HashMap<String, serde_json::Value>,
}

impl ExperimentRun {
    /// Create a new experiment run.
    #[must_use]
    pub fn new(hyperparameters: Hyperparameters) -> Self {
        Self {
            run_id: RunId::new(),
            recipe: None,
            hyperparameters,
            started_at: Utc::now(),
            finished_at: None,
            status: RunStatus::Pending,
            hardware: HardwareInfo::default(),
            metrics: Vec::new(),
            artifacts: Vec::new(),
            log_uri: None,
            git_commit: None,
            git_dirty: false,
            error_message: None,
            extra: HashMap::new(),
        }
    }

    /// Create a run from a recipe.
    #[must_use]
    pub fn from_recipe(recipe: RecipeReference, hyperparameters: Hyperparameters) -> Self {
        let mut run = Self::new(hyperparameters);
        run.recipe = Some(recipe);
        run
    }

    /// Start the run.
    pub fn start(&mut self) {
        self.status = RunStatus::Running;
        self.started_at = Utc::now();
    }

    /// Complete the run successfully.
    pub fn complete(&mut self) {
        self.status = RunStatus::Completed;
        self.finished_at = Some(Utc::now());
    }

    /// Mark the run as failed.
    pub fn fail(&mut self, error: impl Into<String>) {
        self.status = RunStatus::Failed;
        self.finished_at = Some(Utc::now());
        self.error_message = Some(error.into());
    }

    /// Cancel the run.
    pub fn cancel(&mut self) {
        self.status = RunStatus::Cancelled;
        self.finished_at = Some(Utc::now());
    }

    /// Log a metric.
    pub fn log_metric(&mut self, name: impl Into<String>, value: f64, step: u64) {
        self.metrics.push(MetricRecord::new(name, value, step));
    }

    /// Get the latest value for a metric.
    #[must_use]
    pub fn get_metric(&self, name: &str) -> Option<f64> {
        self.metrics.iter().filter(|m| m.name == name).max_by_key(|m| m.step).map(|m| m.value)
    }

    /// Get duration in seconds.
    #[must_use]
    pub fn duration_secs(&self) -> Option<i64> {
        self.finished_at.map(|end| (end - self.started_at).num_seconds())
    }

    /// Check if the run is finished.
    #[must_use]
    pub fn is_finished(&self) -> bool {
        matches!(self.status, RunStatus::Completed | RunStatus::Failed | RunStatus::Cancelled)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_run_id_generation() {
        let id1 = RunId::new();
        let id2 = RunId::new();
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_run_status_display() {
        assert_eq!(RunStatus::Running.to_string(), "running");
        assert_eq!(RunStatus::Completed.to_string(), "completed");
        assert_eq!(RunStatus::Failed.to_string(), "failed");
    }

    #[test]
    fn test_experiment_run_lifecycle() {
        let params = Hyperparameters::default();
        let mut run = ExperimentRun::new(params);

        assert_eq!(run.status, RunStatus::Pending);
        assert!(!run.is_finished());

        run.start();
        assert_eq!(run.status, RunStatus::Running);

        run.log_metric("loss", 0.5, 100);
        run.log_metric("loss", 0.3, 200);
        run.log_metric("accuracy", 0.8, 200);

        assert_eq!(run.get_metric("loss"), Some(0.3));
        assert_eq!(run.get_metric("accuracy"), Some(0.8));
        assert_eq!(run.get_metric("nonexistent"), None);

        run.complete();
        assert_eq!(run.status, RunStatus::Completed);
        assert!(run.is_finished());
        assert!(run.duration_secs().is_some());
    }

    #[test]
    fn test_experiment_run_failure() {
        let params = Hyperparameters::default();
        let mut run = ExperimentRun::new(params);

        run.start();
        run.fail("Out of memory");

        assert_eq!(run.status, RunStatus::Failed);
        assert_eq!(run.error_message, Some("Out of memory".to_string()));
        assert!(run.is_finished());
    }

    #[test]
    fn test_experiment_run_cancel() {
        let params = Hyperparameters::default();
        let mut run = ExperimentRun::new(params);

        run.start();
        run.cancel();

        assert_eq!(run.status, RunStatus::Cancelled);
        assert!(run.is_finished());
    }

    #[test]
    fn test_metric_record() {
        let metric = MetricRecord::new("val_loss", 0.25, 1000);
        assert_eq!(metric.name, "val_loss");
        assert!((metric.value - 0.25).abs() < 1e-10);
        assert_eq!(metric.step, 1000);
    }

    #[test]
    fn test_experiment_run_serialization() {
        let params = Hyperparameters::default();
        let mut run = ExperimentRun::new(params);
        run.log_metric("loss", 0.5, 100);

        let json = serde_json::to_string(&run).unwrap();
        let deserialized: ExperimentRun = serde_json::from_str(&json).unwrap();

        assert_eq!(run.run_id, deserialized.run_id);
        assert_eq!(run.metrics.len(), deserialized.metrics.len());
    }
}
