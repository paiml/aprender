//! Experiment Storage Module (ENT-001)
//!
//! Provides the `ExperimentStorage` trait and backends for persisting
//! experiment tracking data.
//!
//! # Backends
//!
//! - `TruenoBackend`: Production backend using trueno-db (feature: "monitor")
//! - `InMemoryStorage`: In-memory backend for testing and fuzzing
//!
//! # Example
//!
//! ```
//! use entrenar::storage::{ExperimentStorage, InMemoryStorage, RunStatus};
//!
//! let mut storage = InMemoryStorage::new();
//! let exp_id = storage.create_experiment("my-experiment", None).expect("create experiment");
//! let run_id = storage.create_run(&exp_id).expect("create run");
//! storage.start_run(&run_id).expect("start run");
//! storage.log_metric(&run_id, "loss", 0, 0.5).expect("log metric");
//! storage.complete_run(&run_id, RunStatus::Success).expect("complete run");
//! ```

pub mod cloud;
pub mod memory;
pub mod preflight;
pub mod registry;
pub mod sqlite;
#[cfg(feature = "monitor")]
pub mod trueno;

pub use cloud::{
    ArtifactBackend, ArtifactMetadata, AzureConfig, BackendConfig, CloudError, GCSConfig,
    InMemoryBackend, LocalBackend, MockS3Backend, S3Config,
};
pub use memory::InMemoryStorage;
pub use preflight::{
    CheckMetadata, CheckResult, CheckType, Preflight, PreflightCheck, PreflightContext,
    PreflightError, PreflightResults,
};
pub use registry::{
    Comparison, InMemoryRegistry, MetricRequirement, ModelRegistry, ModelStage, ModelVersion,
    PolicyCheckResult, PromotionPolicy, RegistryError, StageTransition, VersionComparison,
};
pub use sqlite::{
    ArtifactRef, Experiment, FilterOp, ParamFilter, ParameterValue, Run, SqliteBackend,
};
#[cfg(feature = "monitor")]
pub use trueno::TruenoBackend;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Storage errors
#[derive(Debug, Error)]
pub enum StorageError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Experiment not found: {0}")]
    ExperimentNotFound(String),

    #[error("Run not found: {0}")]
    RunNotFound(String),

    #[error("Invalid state transition: {0}")]
    InvalidState(String),

    #[error("Storage backend error: {0}")]
    Backend(String),
}

/// Result type for storage operations
pub type Result<T> = std::result::Result<T, StorageError>;

/// Status of a run
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RunStatus {
    /// Run is created but not yet started
    Pending,
    /// Run is currently executing
    Running,
    /// Run completed successfully
    Success,
    /// Run failed with an error
    Failed,
    /// Run was cancelled
    Cancelled,
}

/// A single metric data point
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MetricPoint {
    /// Training step
    pub step: u64,
    /// Metric value
    pub value: f64,
    /// Timestamp when recorded
    pub timestamp: DateTime<Utc>,
}

impl MetricPoint {
    /// Create a new metric point with current timestamp
    pub fn new(step: u64, value: f64) -> Self {
        Self { step, value, timestamp: Utc::now() }
    }

    /// Create a metric point with specific timestamp
    pub fn with_timestamp(step: u64, value: f64, timestamp: DateTime<Utc>) -> Self {
        Self { step, value, timestamp }
    }
}

/// Trait for experiment storage backends
///
/// This trait abstracts over different storage implementations, allowing
/// for production use with TruenoDB and testing with in-memory storage.
///
/// # Thread Safety
///
/// All implementations must be `Send + Sync` to support concurrent access
/// from multiple training threads.
pub trait ExperimentStorage: Send + Sync {
    /// Create a new experiment
    ///
    /// # Arguments
    ///
    /// * `name` - Human-readable experiment name
    /// * `config` - Optional JSON configuration for the experiment
    ///
    /// # Returns
    ///
    /// Unique experiment ID
    fn create_experiment(
        &mut self,
        name: &str,
        config: Option<serde_json::Value>,
    ) -> Result<String>;

    /// Create a new run within an experiment
    ///
    /// The run starts in `Pending` status.
    ///
    /// # Arguments
    ///
    /// * `experiment_id` - ID of the parent experiment
    ///
    /// # Returns
    ///
    /// Unique run ID
    fn create_run(&mut self, experiment_id: &str) -> Result<String>;

    /// Start a run, transitioning from Pending to Running
    ///
    /// # Arguments
    ///
    /// * `run_id` - ID of the run to start
    fn start_run(&mut self, run_id: &str) -> Result<()>;

    /// Complete a run with the given status
    ///
    /// # Arguments
    ///
    /// * `run_id` - ID of the run
    /// * `status` - Final status (Success, Failed, or Cancelled)
    fn complete_run(&mut self, run_id: &str, status: RunStatus) -> Result<()>;

    /// Log a metric value for a run
    ///
    /// # Arguments
    ///
    /// * `run_id` - ID of the run
    /// * `key` - Metric name (e.g., "loss", "accuracy")
    /// * `step` - Training step or epoch number
    /// * `value` - Metric value
    fn log_metric(&mut self, run_id: &str, key: &str, step: u64, value: f64) -> Result<()>;

    /// Log an artifact for a run
    ///
    /// # Arguments
    ///
    /// * `run_id` - ID of the run
    /// * `key` - Artifact name (e.g., "model.safetensors")
    /// * `data` - Artifact data bytes
    ///
    /// # Returns
    ///
    /// Content-addressable hash of the artifact
    fn log_artifact(&mut self, run_id: &str, key: &str, data: &[u8]) -> Result<String>;

    /// Get metrics for a specific run and key
    ///
    /// # Arguments
    ///
    /// * `run_id` - ID of the run
    /// * `key` - Metric name to retrieve
    ///
    /// # Returns
    ///
    /// Vector of metric points, ordered by step
    fn get_metrics(&self, run_id: &str, key: &str) -> Result<Vec<MetricPoint>>;

    /// Get the current status of a run
    fn get_run_status(&self, run_id: &str) -> Result<RunStatus>;

    /// Set renacer span ID for distributed tracing
    fn set_span_id(&mut self, run_id: &str, span_id: &str) -> Result<()>;

    /// Get renacer span ID for a run
    fn get_span_id(&self, run_id: &str) -> Result<Option<String>>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metric_point_new() {
        let point = MetricPoint::new(10, 0.5);
        assert_eq!(point.step, 10);
        assert!((point.value - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_metric_point_with_timestamp() {
        let ts = Utc::now();
        let point = MetricPoint::with_timestamp(5, 0.3, ts);
        assert_eq!(point.step, 5);
        assert_eq!(point.timestamp, ts);
    }

    #[test]
    fn test_run_status_variants() {
        assert_ne!(RunStatus::Pending, RunStatus::Running);
        assert_ne!(RunStatus::Success, RunStatus::Failed);
    }

    #[test]
    fn test_storage_error_display() {
        let err = StorageError::ExperimentNotFound("exp-1".to_string());
        assert!(err.to_string().contains("exp-1"));

        let err = StorageError::RunNotFound("run-1".to_string());
        assert!(err.to_string().contains("run-1"));

        let err = StorageError::InvalidState("cannot start".to_string());
        assert!(err.to_string().contains("cannot start"));
    }
}
