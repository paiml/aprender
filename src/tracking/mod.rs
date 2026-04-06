//! Experiment Tracking Module (GH-31)
//!
//! Provides high-level experiment tracking with parameter logging, metric
//! recording, and artifact management. Backed by pluggable storage via the
//! [`TrackingBackend`](storage::TrackingBackend) trait.
//!
//! # Architecture
//!
//! - **`ExperimentTracker`**: Top-level handle that manages runs for a named experiment
//! - **`Run`**: A single training run with parameters, metrics, and artifacts
//! - **`TrackingBackend`**: Pluggable persistence (JSON files, in-memory)
//!
//! # Example
//!
//! ```
//! use entrenar::tracking::{ExperimentTracker, RunStatus};
//! use entrenar::tracking::storage::InMemoryBackend;
//!
//! # fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
//! let backend = InMemoryBackend::new();
//! let mut tracker = ExperimentTracker::new("my-experiment", backend);
//! tracker.add_tag("team", "ml-infra");
//!
//! let run_id = tracker.start_run(Some("baseline-v1"))?;
//! tracker.log_param(&run_id, "lr", "0.001")?;
//! tracker.log_metric(&run_id, "loss", 0.5, 1)?;
//! tracker.log_metric(&run_id, "loss", 0.3, 2)?;
//! tracker.log_artifact(&run_id, "model.safetensors")?;
//! tracker.end_run(&run_id, RunStatus::Completed)?;
//!
//! let run = tracker.get_run(&run_id)?;
//! assert_eq!(run.params.get("lr").unwrap_or(&String::new()), "0.001");
//!
//! let all = tracker.list_runs()?;
//! assert_eq!(all.len(), 1);
//! # Ok(())
//! # }
//! ```

pub mod storage;

#[cfg(test)]
mod tests;

use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use storage::{TrackingBackend, TrackingStorageError};

/// Status of a tracking run
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RunStatus {
    /// Run is actively recording
    Active,
    /// Run completed successfully
    Completed,
    /// Run failed
    Failed,
    /// Run was cancelled
    Cancelled,
}

/// A single experiment run
///
/// Tracks parameters (hyperparameters), metrics (per-step values),
/// artifacts (file paths), and tags (key-value metadata).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Run {
    /// Unique identifier for the run
    pub run_id: String,
    /// Optional human-readable name
    pub run_name: Option<String>,
    /// Parent experiment name
    pub experiment_name: String,
    /// Current status
    pub status: RunStatus,
    /// Hyperparameters: key -> value (string-encoded)
    pub params: HashMap<String, String>,
    /// Metrics: key -> list of (value, step)
    pub metrics: HashMap<String, Vec<(f64, u64)>>,
    /// Artifact paths
    pub artifacts: Vec<String>,
    /// Tags: key -> value
    pub tags: HashMap<String, String>,
    /// Unix timestamp (ms) when the run started
    pub start_time_ms: Option<u64>,
    /// Unix timestamp (ms) when the run ended
    pub end_time_ms: Option<u64>,
}

impl Run {
    fn new(run_id: String, run_name: Option<String>, experiment_name: String) -> Self {
        let now_ms =
            SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_millis() as u64;

        Self {
            run_id,
            run_name,
            experiment_name,
            status: RunStatus::Active,
            params: HashMap::new(),
            metrics: HashMap::new(),
            artifacts: Vec::new(),
            tags: HashMap::new(),
            start_time_ms: Some(now_ms),
            end_time_ms: None,
        }
    }
}

/// Errors from experiment tracking operations
#[derive(Debug, thiserror::Error)]
pub enum TrackingError {
    #[error("Run not found: {0}")]
    RunNotFound(String),

    #[error("Run is not active: {0}")]
    RunNotActive(String),

    #[error("Storage error: {0}")]
    Storage(#[from] TrackingStorageError),
}

/// Result alias for tracking operations
pub type Result<T> = std::result::Result<T, TrackingError>;

/// Experiment tracker
///
/// Manages multiple runs under a single experiment name. Persists run data
/// through a pluggable [`TrackingBackend`].
#[derive(Debug)]
pub struct ExperimentTracker<B: TrackingBackend> {
    experiment_name: String,
    tags: HashMap<String, String>,
    backend: B,
    /// Active runs held in memory for fast mutation
    active_runs: HashMap<String, Run>,
    next_run_id: u64,
}

impl<B: TrackingBackend> ExperimentTracker<B> {
    /// Create a new tracker for the given experiment name
    pub fn new(experiment_name: impl Into<String>, backend: B) -> Self {
        Self {
            experiment_name: experiment_name.into(),
            tags: HashMap::new(),
            backend,
            active_runs: HashMap::new(),
            next_run_id: 1,
        }
    }

    /// Add an experiment-level tag
    pub fn add_tag(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.tags.insert(key.into(), value.into());
    }

    /// Get the experiment name
    #[must_use]
    pub fn experiment_name(&self) -> &str {
        &self.experiment_name
    }

    /// Get experiment-level tags
    #[must_use]
    pub fn tags(&self) -> &HashMap<String, String> {
        &self.tags
    }

    /// Start a new run, optionally with a human-readable name
    ///
    /// Returns the run ID.
    pub fn start_run(&mut self, run_name: Option<&str>) -> Result<String> {
        let run_id = format!("run-{}", self.next_run_id);
        self.next_run_id += 1;

        let mut run =
            Run::new(run_id.clone(), run_name.map(String::from), self.experiment_name.clone());
        // Inherit experiment-level tags
        for (k, v) in &self.tags {
            run.tags.insert(k.clone(), v.clone());
        }

        self.active_runs.insert(run_id.clone(), run);
        Ok(run_id)
    }

    /// End a run with the given status, persisting it to the backend
    pub fn end_run(&mut self, run_id: &str, status: RunStatus) -> Result<()> {
        let mut run = self
            .active_runs
            .remove(run_id)
            .ok_or_else(|| TrackingError::RunNotFound(run_id.to_string()))?;

        let now_ms =
            SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_millis() as u64;

        run.status = status;
        run.end_time_ms = Some(now_ms);

        self.backend.save_run(&run)?;
        Ok(())
    }

    /// Log a single parameter (hyperparameter)
    pub fn log_param(&mut self, run_id: &str, key: &str, value: &str) -> Result<()> {
        let run = self
            .active_runs
            .get_mut(run_id)
            .ok_or_else(|| TrackingError::RunNotActive(run_id.to_string()))?;

        run.params.insert(key.to_string(), value.to_string());
        Ok(())
    }

    /// Log multiple parameters at once
    pub fn log_params(&mut self, run_id: &str, params: &HashMap<String, String>) -> Result<()> {
        let run = self
            .active_runs
            .get_mut(run_id)
            .ok_or_else(|| TrackingError::RunNotActive(run_id.to_string()))?;

        for (k, v) in params {
            run.params.insert(k.clone(), v.clone());
        }
        Ok(())
    }

    /// Log a metric value at a given step
    pub fn log_metric(&mut self, run_id: &str, key: &str, value: f64, step: u64) -> Result<()> {
        let run = self
            .active_runs
            .get_mut(run_id)
            .ok_or_else(|| TrackingError::RunNotActive(run_id.to_string()))?;

        run.metrics.entry(key.to_string()).or_default().push((value, step));
        Ok(())
    }

    /// Log an artifact path
    pub fn log_artifact(&mut self, run_id: &str, path: &str) -> Result<()> {
        let run = self
            .active_runs
            .get_mut(run_id)
            .ok_or_else(|| TrackingError::RunNotActive(run_id.to_string()))?;

        run.artifacts.push(path.to_string());
        Ok(())
    }

    /// Retrieve a run by ID
    ///
    /// Checks active (in-memory) runs first, then falls back to the backend.
    pub fn get_run(&self, run_id: &str) -> Result<Run> {
        if let Some(run) = self.active_runs.get(run_id) {
            return Ok(run.clone());
        }
        self.backend
            .load_run(run_id)
            .map_err(|e| TrackingError::RunNotFound(format!("{run_id}: {e}")))
    }

    /// List all runs (active + persisted)
    pub fn list_runs(&self) -> Result<Vec<Run>> {
        let mut runs: Vec<Run> = self.active_runs.values().cloned().collect();
        let persisted = self.backend.list_runs()?;
        // Avoid duplicates: only add persisted runs whose IDs are not active
        for r in persisted {
            if !self.active_runs.contains_key(&r.run_id) {
                runs.push(r);
            }
        }
        runs.sort_by(|a, b| a.run_id.cmp(&b.run_id));
        Ok(runs)
    }
}
