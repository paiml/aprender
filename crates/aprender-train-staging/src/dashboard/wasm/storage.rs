//! IndexedDB-backed storage implementation.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::storage::{ExperimentStorage, MetricPoint, Result, RunStatus, StorageError};

/// Internal experiment data structure.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct ExperimentData {
    pub id: String,
    pub name: String,
    pub config: Option<serde_json::Value>,
    pub created_at: DateTime<Utc>,
}

/// Internal run data structure.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct RunData {
    pub id: String,
    pub experiment_id: String,
    pub status: RunStatus,
    pub span_id: Option<String>,
    pub started_at: Option<DateTime<Utc>>,
    pub completed_at: Option<DateTime<Utc>>,
}

/// IndexedDB-backed storage for browser environments.
///
/// This is a simplified in-memory implementation that mimics
/// IndexedDB behavior. A full implementation would use web-sys
/// to interact with the actual IndexedDB API.
#[derive(Debug, Default)]
pub struct IndexedDbStorage {
    /// Experiments by ID
    experiments: HashMap<String, ExperimentData>,
    /// Runs by ID
    runs: HashMap<String, RunData>,
    /// Metrics by run_id -> key -> points
    metrics: HashMap<String, HashMap<String, Vec<MetricPoint>>>,
    /// Artifacts by hash
    artifacts: HashMap<String, Vec<u8>>,
    /// Next experiment ID counter
    next_exp_id: u64,
    /// Next run ID counter
    next_run_id: u64,
}

impl IndexedDbStorage {
    /// Create a new IndexedDB storage instance.
    pub fn new() -> Self {
        Self::default()
    }

    /// Get all experiments.
    pub fn list_experiments(&self) -> Vec<String> {
        self.experiments.keys().cloned().collect()
    }

    /// Get all runs for an experiment.
    pub fn list_runs(&self, experiment_id: &str) -> Vec<String> {
        self.runs
            .values()
            .filter(|r| r.experiment_id == experiment_id)
            .map(|r| r.id.clone())
            .collect()
    }

    /// Get all metric keys for a run.
    pub fn list_metric_keys(&self, run_id: &str) -> Vec<String> {
        self.metrics.get(run_id).map(|m| m.keys().cloned().collect()).unwrap_or_default()
    }
}

impl ExperimentStorage for IndexedDbStorage {
    fn create_experiment(
        &mut self,
        name: &str,
        config: Option<serde_json::Value>,
    ) -> Result<String> {
        let id = format!("exp-{}", self.next_exp_id);
        self.next_exp_id += 1;

        let experiment = ExperimentData {
            id: id.clone(),
            name: name.to_string(),
            config,
            created_at: Utc::now(),
        };

        self.experiments.insert(id.clone(), experiment);
        Ok(id)
    }

    fn create_run(&mut self, experiment_id: &str) -> Result<String> {
        if !self.experiments.contains_key(experiment_id) {
            return Err(StorageError::ExperimentNotFound(experiment_id.to_string()));
        }

        let id = format!("run-{}", self.next_run_id);
        self.next_run_id += 1;

        let run = RunData {
            id: id.clone(),
            experiment_id: experiment_id.to_string(),
            status: RunStatus::Pending,
            span_id: None,
            started_at: None,
            completed_at: None,
        };

        self.runs.insert(id.clone(), run);
        self.metrics.insert(id.clone(), HashMap::new());
        Ok(id)
    }

    fn start_run(&mut self, run_id: &str) -> Result<()> {
        let run = self
            .runs
            .get_mut(run_id)
            .ok_or_else(|| StorageError::RunNotFound(run_id.to_string()))?;

        if run.status != RunStatus::Pending {
            return Err(StorageError::InvalidState(format!(
                "Run {run_id} is not in Pending state"
            )));
        }

        run.status = RunStatus::Running;
        run.started_at = Some(Utc::now());
        Ok(())
    }

    fn complete_run(&mut self, run_id: &str, status: RunStatus) -> Result<()> {
        let run = self
            .runs
            .get_mut(run_id)
            .ok_or_else(|| StorageError::RunNotFound(run_id.to_string()))?;

        if run.status != RunStatus::Running {
            return Err(StorageError::InvalidState(format!(
                "Run {run_id} is not in Running state"
            )));
        }

        run.status = status;
        run.completed_at = Some(Utc::now());
        Ok(())
    }

    fn log_metric(&mut self, run_id: &str, key: &str, step: u64, value: f64) -> Result<()> {
        let metrics = self
            .metrics
            .get_mut(run_id)
            .ok_or_else(|| StorageError::RunNotFound(run_id.to_string()))?;

        let points = metrics.entry(key.to_string()).or_default();
        points.push(MetricPoint::new(step, value));
        Ok(())
    }

    fn log_artifact(&mut self, run_id: &str, key: &str, data: &[u8]) -> Result<String> {
        if !self.runs.contains_key(run_id) {
            return Err(StorageError::RunNotFound(run_id.to_string()));
        }

        // Compute SHA-256 hash
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(data);
        let hash = hex::encode(hasher.finalize());

        // Store artifact
        let artifact_key = format!("{run_id}/{key}/{hash}");
        self.artifacts.insert(artifact_key, data.to_vec());

        Ok(hash)
    }

    fn get_metrics(&self, run_id: &str, key: &str) -> Result<Vec<MetricPoint>> {
        let metrics = self
            .metrics
            .get(run_id)
            .ok_or_else(|| StorageError::RunNotFound(run_id.to_string()))?;

        Ok(metrics.get(key).cloned().unwrap_or_default())
    }

    fn get_run_status(&self, run_id: &str) -> Result<RunStatus> {
        let run =
            self.runs.get(run_id).ok_or_else(|| StorageError::RunNotFound(run_id.to_string()))?;

        Ok(run.status)
    }

    fn set_span_id(&mut self, run_id: &str, span_id: &str) -> Result<()> {
        let run = self
            .runs
            .get_mut(run_id)
            .ok_or_else(|| StorageError::RunNotFound(run_id.to_string()))?;

        run.span_id = Some(span_id.to_string());
        Ok(())
    }

    fn get_span_id(&self, run_id: &str) -> Result<Option<String>> {
        let run =
            self.runs.get(run_id).ok_or_else(|| StorageError::RunNotFound(run_id.to_string()))?;

        Ok(run.span_id.clone())
    }
}
