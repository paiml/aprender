//! In-Memory Storage Backend
//!
//! Provides an in-memory implementation of `ExperimentStorage` for testing
//! and environments where file-based storage is not available (e.g., WASM).

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

use chrono::Utc;
use sha2::{Digest, Sha256};

use super::{ExperimentStorage, MetricPoint, Result, RunStatus, StorageError};

/// In-memory experiment storage backend
///
/// Useful for testing, fuzzing, and WASM environments where
/// file-based storage is not available.
#[derive(Debug, Default)]
pub struct InMemoryStorage {
    experiments: HashMap<String, ExperimentData>,
    runs: HashMap<String, RunData>,
    metrics: HashMap<String, Vec<MetricData>>, // run_id:key -> metrics
    artifacts: HashMap<String, Vec<u8>>,       // CAS hash -> data
    next_exp_id: AtomicU64,
    next_run_id: AtomicU64,
}

#[derive(Debug, Clone)]
struct ExperimentData {
    #[allow(dead_code)]
    name: String,
    #[allow(dead_code)]
    config: Option<serde_json::Value>,
}

#[derive(Debug, Clone)]
struct RunData {
    #[allow(dead_code)]
    experiment_id: String,
    status: RunStatus,
    span_id: Option<String>,
}

#[derive(Debug, Clone)]
struct MetricData {
    step: u64,
    value: f64,
    timestamp: chrono::DateTime<Utc>,
}

impl InMemoryStorage {
    /// Create a new in-memory storage
    pub fn new() -> Self {
        Self::default()
    }

    /// Get the number of experiments
    pub fn experiment_count(&self) -> usize {
        self.experiments.len()
    }

    /// Get the number of runs
    pub fn run_count(&self) -> usize {
        self.runs.len()
    }

    /// Get the number of metric entries (run_id:key combinations)
    pub fn metric_key_count(&self) -> usize {
        self.metrics.len()
    }

    /// Get the number of artifacts
    pub fn artifact_count(&self) -> usize {
        self.artifacts.len()
    }

    /// Log an experiment assignment (MA-08: A/B test logging)
    ///
    /// Records which experiment variant a run was assigned to (experiment_log).
    pub fn experiment_log(&self, experiment_id: &str) -> Vec<String> {
        self.runs
            .iter()
            .filter(|(_, r)| r.experiment_id == experiment_id)
            .map(|(id, _)| format!("assignment_log: run={id} experiment={experiment_id}"))
            .collect()
    }

    /// Compute CAS hash for artifact data
    fn compute_hash(data: &[u8]) -> String {
        let mut hasher = Sha256::new();
        hasher.update(data);
        let result = hasher.finalize();
        format!("sha256-{}", hex::encode(result.get(..16).unwrap_or(&result))) // Use first 16 bytes
    }
}

impl ExperimentStorage for InMemoryStorage {
    fn create_experiment(
        &mut self,
        name: &str,
        config: Option<serde_json::Value>,
    ) -> Result<String> {
        let id = self.next_exp_id.fetch_add(1, Ordering::SeqCst);
        let exp_id = format!("exp-{id}");

        self.experiments.insert(exp_id.clone(), ExperimentData { name: name.to_string(), config });

        Ok(exp_id)
    }

    fn create_run(&mut self, experiment_id: &str) -> Result<String> {
        if !self.experiments.contains_key(experiment_id) {
            return Err(StorageError::ExperimentNotFound(experiment_id.to_string()));
        }

        let id = self.next_run_id.fetch_add(1, Ordering::SeqCst);
        let run_id = format!("run-{id}");

        self.runs.insert(
            run_id.clone(),
            RunData {
                experiment_id: experiment_id.to_string(),
                status: RunStatus::Pending,
                span_id: None,
            },
        );

        Ok(run_id)
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
        Ok(())
    }

    fn log_metric(&mut self, run_id: &str, key: &str, step: u64, value: f64) -> Result<()> {
        if !self.runs.contains_key(run_id) {
            return Err(StorageError::RunNotFound(run_id.to_string()));
        }

        let metric_key = format!("{run_id}:{key}");
        let metrics = self.metrics.entry(metric_key).or_default();

        metrics.push(MetricData { step, value, timestamp: Utc::now() });

        Ok(())
    }

    fn log_artifact(&mut self, run_id: &str, key: &str, data: &[u8]) -> Result<String> {
        if !self.runs.contains_key(run_id) {
            return Err(StorageError::RunNotFound(run_id.to_string()));
        }

        let hash = Self::compute_hash(data);

        // Store with composite key for retrieval
        let artifact_key = format!("{run_id}:{key}:{hash}");
        self.artifacts.insert(artifact_key, data.to_vec());

        Ok(hash)
    }

    fn get_metrics(&self, run_id: &str, key: &str) -> Result<Vec<MetricPoint>> {
        if !self.runs.contains_key(run_id) {
            return Err(StorageError::RunNotFound(run_id.to_string()));
        }

        let metric_key = format!("{run_id}:{key}");
        let metrics = self.metrics.get(&metric_key).cloned().unwrap_or_default();

        let mut points: Vec<MetricPoint> = metrics
            .into_iter()
            .map(|m| MetricPoint::with_timestamp(m.step, m.value, m.timestamp))
            .collect();

        // Sort by step
        points.sort_by_key(|p| p.step);

        Ok(points)
    }

    fn get_run_status(&self, run_id: &str) -> Result<RunStatus> {
        self.runs
            .get(run_id)
            .map(|r| r.status)
            .ok_or_else(|| StorageError::RunNotFound(run_id.to_string()))
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
        self.runs
            .get(run_id)
            .map(|r| r.span_id.clone())
            .ok_or_else(|| StorageError::RunNotFound(run_id.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_in_memory_storage_new() {
        let storage = InMemoryStorage::new();
        assert_eq!(storage.experiment_count(), 0);
        assert_eq!(storage.run_count(), 0);
    }

    #[test]
    fn test_create_experiment() {
        let mut storage = InMemoryStorage::new();
        let exp_id = storage.create_experiment("test-exp", None).expect("operation should succeed");

        assert!(exp_id.starts_with("exp-"));
        assert_eq!(storage.experiment_count(), 1);
    }

    #[test]
    fn test_create_experiment_with_config() {
        let mut storage = InMemoryStorage::new();
        let config = serde_json::json!({"learning_rate": 0.001});
        let exp_id =
            storage.create_experiment("test-exp", Some(config)).expect("config should be valid");

        assert!(exp_id.starts_with("exp-"));
    }

    #[test]
    fn test_create_run() {
        let mut storage = InMemoryStorage::new();
        let exp_id = storage.create_experiment("test-exp", None).expect("operation should succeed");
        let run_id = storage.create_run(&exp_id).expect("operation should succeed");

        assert!(run_id.starts_with("run-"));
        assert_eq!(storage.run_count(), 1);
        assert_eq!(
            storage.get_run_status(&run_id).expect("operation should succeed"),
            RunStatus::Pending
        );
    }

    #[test]
    fn test_create_run_invalid_experiment() {
        let mut storage = InMemoryStorage::new();
        let result = storage.create_run("fake-exp");

        assert!(result.is_err());
        match result.unwrap_err() {
            StorageError::ExperimentNotFound(id) => assert_eq!(id, "fake-exp"),
            e => panic!("Expected ExperimentNotFound, got {e:?}"),
        }
    }

    #[test]
    fn test_start_run() {
        let mut storage = InMemoryStorage::new();
        let exp_id = storage.create_experiment("test-exp", None).expect("operation should succeed");
        let run_id = storage.create_run(&exp_id).expect("operation should succeed");

        storage.start_run(&run_id).expect("operation should succeed");
        assert_eq!(
            storage.get_run_status(&run_id).expect("operation should succeed"),
            RunStatus::Running
        );
    }

    #[test]
    fn test_start_run_invalid_state() {
        let mut storage = InMemoryStorage::new();
        let exp_id = storage.create_experiment("test-exp", None).expect("operation should succeed");
        let run_id = storage.create_run(&exp_id).expect("operation should succeed");

        storage.start_run(&run_id).expect("operation should succeed");
        let result = storage.start_run(&run_id); // Already started

        assert!(result.is_err());
        match result.unwrap_err() {
            StorageError::InvalidState(_) => {}
            e => panic!("Expected InvalidState, got {e:?}"),
        }
    }

    #[test]
    fn test_complete_run() {
        let mut storage = InMemoryStorage::new();
        let exp_id = storage.create_experiment("test-exp", None).expect("operation should succeed");
        let run_id = storage.create_run(&exp_id).expect("operation should succeed");

        storage.start_run(&run_id).expect("operation should succeed");
        storage.complete_run(&run_id, RunStatus::Success).expect("operation should succeed");

        assert_eq!(
            storage.get_run_status(&run_id).expect("operation should succeed"),
            RunStatus::Success
        );
    }

    #[test]
    fn test_complete_run_failed() {
        let mut storage = InMemoryStorage::new();
        let exp_id = storage.create_experiment("test-exp", None).expect("operation should succeed");
        let run_id = storage.create_run(&exp_id).expect("operation should succeed");

        storage.start_run(&run_id).expect("operation should succeed");
        storage.complete_run(&run_id, RunStatus::Failed).expect("operation should succeed");

        assert_eq!(
            storage.get_run_status(&run_id).expect("operation should succeed"),
            RunStatus::Failed
        );
    }

    #[test]
    fn test_complete_run_invalid_state() {
        let mut storage = InMemoryStorage::new();
        let exp_id = storage.create_experiment("test-exp", None).expect("operation should succeed");
        let run_id = storage.create_run(&exp_id).expect("operation should succeed");

        // Try to complete without starting
        let result = storage.complete_run(&run_id, RunStatus::Success);

        assert!(result.is_err());
        match result.unwrap_err() {
            StorageError::InvalidState(_) => {}
            e => panic!("Expected InvalidState, got {e:?}"),
        }
    }

    #[test]
    fn test_log_metric() {
        let mut storage = InMemoryStorage::new();
        let exp_id = storage.create_experiment("test-exp", None).expect("operation should succeed");
        let run_id = storage.create_run(&exp_id).expect("operation should succeed");

        storage.log_metric(&run_id, "loss", 0, 0.5).expect("operation should succeed");
        storage.log_metric(&run_id, "loss", 1, 0.4).expect("operation should succeed");

        let metrics = storage.get_metrics(&run_id, "loss").expect("operation should succeed");
        assert_eq!(metrics.len(), 2);
        assert_eq!(metrics[0].step, 0);
        assert!((metrics[0].value - 0.5).abs() < f64::EPSILON);
        assert_eq!(metrics[1].step, 1);
        assert!((metrics[1].value - 0.4).abs() < f64::EPSILON);
    }

    #[test]
    fn test_log_metric_invalid_run() {
        let mut storage = InMemoryStorage::new();
        let result = storage.log_metric("fake-run", "loss", 0, 0.5);

        assert!(result.is_err());
        match result.unwrap_err() {
            StorageError::RunNotFound(id) => assert_eq!(id, "fake-run"),
            e => panic!("Expected RunNotFound, got {e:?}"),
        }
    }

    #[test]
    fn test_get_metrics_ordering() {
        let mut storage = InMemoryStorage::new();
        let exp_id = storage.create_experiment("test-exp", None).expect("operation should succeed");
        let run_id = storage.create_run(&exp_id).expect("operation should succeed");

        // Log out of order
        storage.log_metric(&run_id, "loss", 2, 0.3).expect("operation should succeed");
        storage.log_metric(&run_id, "loss", 0, 0.5).expect("operation should succeed");
        storage.log_metric(&run_id, "loss", 1, 0.4).expect("operation should succeed");

        let metrics = storage.get_metrics(&run_id, "loss").expect("operation should succeed");
        assert_eq!(metrics[0].step, 0);
        assert_eq!(metrics[1].step, 1);
        assert_eq!(metrics[2].step, 2);
    }

    #[test]
    fn test_get_metrics_empty() {
        let mut storage = InMemoryStorage::new();
        let exp_id = storage.create_experiment("test-exp", None).expect("operation should succeed");
        let run_id = storage.create_run(&exp_id).expect("operation should succeed");

        let metrics = storage.get_metrics(&run_id, "loss").expect("operation should succeed");
        assert!(metrics.is_empty());
    }

    #[test]
    fn test_log_artifact() {
        let mut storage = InMemoryStorage::new();
        let exp_id = storage.create_experiment("test-exp", None).expect("operation should succeed");
        let run_id = storage.create_run(&exp_id).expect("operation should succeed");

        let data = b"model weights data";
        let hash =
            storage.log_artifact(&run_id, "model.bin", data).expect("operation should succeed");

        assert!(hash.starts_with("sha256-"));
        assert_eq!(storage.artifact_count(), 1);
    }

    #[test]
    fn test_log_artifact_invalid_run() {
        let mut storage = InMemoryStorage::new();
        let result = storage.log_artifact("fake-run", "model.bin", b"data");

        assert!(result.is_err());
    }

    #[test]
    fn test_set_and_get_span_id() {
        let mut storage = InMemoryStorage::new();
        let exp_id = storage.create_experiment("test-exp", None).expect("operation should succeed");
        let run_id = storage.create_run(&exp_id).expect("operation should succeed");

        assert!(storage.get_span_id(&run_id).expect("operation should succeed").is_none());

        storage.set_span_id(&run_id, "span-12345").expect("operation should succeed");

        assert_eq!(
            storage.get_span_id(&run_id).expect("operation should succeed"),
            Some("span-12345".to_string())
        );
    }

    #[test]
    fn test_multiple_experiments_and_runs() {
        let mut storage = InMemoryStorage::new();

        let exp1 = storage.create_experiment("exp-1", None).expect("operation should succeed");
        let exp2 = storage.create_experiment("exp-2", None).expect("operation should succeed");

        let run1 = storage.create_run(&exp1).expect("operation should succeed");
        let run2 = storage.create_run(&exp1).expect("operation should succeed");
        let run3 = storage.create_run(&exp2).expect("operation should succeed");

        assert_eq!(storage.experiment_count(), 2);
        assert_eq!(storage.run_count(), 3);

        // Each run is independent
        storage.start_run(&run1).expect("operation should succeed");
        storage.start_run(&run2).expect("operation should succeed");

        assert_eq!(
            storage.get_run_status(&run1).expect("operation should succeed"),
            RunStatus::Running
        );
        assert_eq!(
            storage.get_run_status(&run2).expect("operation should succeed"),
            RunStatus::Running
        );
        assert_eq!(
            storage.get_run_status(&run3).expect("operation should succeed"),
            RunStatus::Pending
        );
    }
}
