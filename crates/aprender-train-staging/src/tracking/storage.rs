//! Tracking storage backends
//!
//! Provides the `TrackingBackend` trait and a JSON file-based implementation
//! for persisting experiment runs to disk.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use super::{Run, RunStatus};

/// Errors from tracking storage operations
#[derive(Debug, thiserror::Error)]
pub enum TrackingStorageError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON serialization error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Run not found: {0}")]
    RunNotFound(String),
}

/// Result alias for tracking storage operations
pub type Result<T> = std::result::Result<T, TrackingStorageError>;

/// Serializable snapshot of a run for persistence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunRecord {
    pub run_id: String,
    pub run_name: Option<String>,
    pub experiment_name: String,
    pub status: RunStatus,
    pub params: HashMap<String, String>,
    pub metrics: HashMap<String, Vec<MetricEntry>>,
    pub artifacts: Vec<String>,
    pub tags: HashMap<String, String>,
    pub start_time_ms: Option<u64>,
    pub end_time_ms: Option<u64>,
}

/// A single metric data point for serialization
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MetricEntry {
    pub value: f64,
    pub step: u64,
}

impl From<&Run> for RunRecord {
    fn from(run: &Run) -> Self {
        Self {
            run_id: run.run_id.clone(),
            run_name: run.run_name.clone(),
            experiment_name: run.experiment_name.clone(),
            status: run.status,
            params: run.params.clone(),
            metrics: run
                .metrics
                .iter()
                .map(|(k, v)| {
                    (
                        k.clone(),
                        v.iter()
                            .map(|(val, step)| MetricEntry { value: *val, step: *step })
                            .collect(),
                    )
                })
                .collect(),
            artifacts: run.artifacts.clone(),
            tags: run.tags.clone(),
            start_time_ms: run.start_time_ms,
            end_time_ms: run.end_time_ms,
        }
    }
}

impl RunRecord {
    /// Convert back into a `Run`
    pub fn into_run(self) -> Run {
        Run {
            run_id: self.run_id,
            run_name: self.run_name,
            experiment_name: self.experiment_name,
            status: self.status,
            params: self.params,
            metrics: self
                .metrics
                .into_iter()
                .map(|(k, v)| (k, v.into_iter().map(|e| (e.value, e.step)).collect()))
                .collect(),
            artifacts: self.artifacts,
            tags: self.tags,
            start_time_ms: self.start_time_ms,
            end_time_ms: self.end_time_ms,
        }
    }
}

/// Trait for tracking storage backends
///
/// Implementations persist and retrieve experiment runs.
pub trait TrackingBackend {
    /// Save a run to the backend
    fn save_run(&mut self, run: &Run) -> Result<()>;

    /// Load a run by its ID
    fn load_run(&self, run_id: &str) -> Result<Run>;

    /// List all stored runs
    fn list_runs(&self) -> Result<Vec<Run>>;

    /// Delete a run by its ID
    fn delete_run(&mut self, run_id: &str) -> Result<()>;
}

/// JSON file-based tracking backend
///
/// Stores each run as a separate JSON file in a directory.
/// File names are `{run_id}.json`.
///
/// # Example
///
/// ```no_run
/// use entrenar::tracking::storage::JsonFileBackend;
/// use entrenar::tracking::storage::TrackingBackend;
///
/// let mut backend = JsonFileBackend::new("/tmp/runs");
/// ```
#[derive(Debug)]
pub struct JsonFileBackend {
    dir: PathBuf,
}

impl JsonFileBackend {
    /// Create a new JSON file backend, creating the directory if it does not exist
    pub fn new(dir: impl AsRef<Path>) -> Self {
        Self { dir: dir.as_ref().to_path_buf() }
    }

    fn run_path(&self, run_id: &str) -> PathBuf {
        self.dir.join(format!("{run_id}.json"))
    }

    fn ensure_dir(&self) -> Result<()> {
        if !self.dir.exists() {
            fs::create_dir_all(&self.dir)?;
        }
        Ok(())
    }
}

impl TrackingBackend for JsonFileBackend {
    fn save_run(&mut self, run: &Run) -> Result<()> {
        self.ensure_dir()?;
        let record = RunRecord::from(run);
        let json = serde_json::to_string_pretty(&record)?;
        fs::write(self.run_path(&run.run_id), json)?;
        Ok(())
    }

    fn load_run(&self, run_id: &str) -> Result<Run> {
        let path = self.run_path(run_id);
        if !path.exists() {
            return Err(TrackingStorageError::RunNotFound(run_id.to_string()));
        }
        let json = fs::read_to_string(path)?;
        let record: RunRecord = serde_json::from_str(&json)?;
        Ok(record.into_run())
    }

    fn list_runs(&self) -> Result<Vec<Run>> {
        if !self.dir.exists() {
            return Ok(Vec::new());
        }
        let mut runs = Vec::new();
        for entry in fs::read_dir(&self.dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("json") {
                let json = fs::read_to_string(&path)?;
                let record: RunRecord = serde_json::from_str(&json)?;
                runs.push(record.into_run());
            }
        }
        runs.sort_by(|a, b| a.run_id.cmp(&b.run_id));
        Ok(runs)
    }

    fn delete_run(&mut self, run_id: &str) -> Result<()> {
        let path = self.run_path(run_id);
        if !path.exists() {
            return Err(TrackingStorageError::RunNotFound(run_id.to_string()));
        }
        fs::remove_file(path)?;
        Ok(())
    }
}

/// In-memory tracking backend for testing
///
/// Stores runs in a `HashMap`. No persistence.
#[derive(Debug, Default)]
pub struct InMemoryBackend {
    runs: HashMap<String, RunRecord>,
}

impl InMemoryBackend {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

impl TrackingBackend for InMemoryBackend {
    fn save_run(&mut self, run: &Run) -> Result<()> {
        self.runs.insert(run.run_id.clone(), RunRecord::from(run));
        Ok(())
    }

    fn load_run(&self, run_id: &str) -> Result<Run> {
        self.runs
            .get(run_id)
            .map(|r| r.clone().into_run())
            .ok_or_else(|| TrackingStorageError::RunNotFound(run_id.to_string()))
    }

    fn list_runs(&self) -> Result<Vec<Run>> {
        let mut runs: Vec<Run> = self.runs.values().map(|r| r.clone().into_run()).collect();
        runs.sort_by(|a, b| a.run_id.cmp(&b.run_id));
        Ok(runs)
    }

    fn delete_run(&mut self, run_id: &str) -> Result<()> {
        self.runs
            .remove(run_id)
            .map(|_| ())
            .ok_or_else(|| TrackingStorageError::RunNotFound(run_id.to_string()))
    }
}
