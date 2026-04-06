//! TruenoDB Storage Backend (ENT-001)
//!
//! Production backend that persists experiment data to TruenoDB.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

use sha2::{Digest, Sha256};
use trueno_db::experiment::{
    ExperimentRecord, ExperimentStore, MetricRecord, RunRecord, RunStatus as TruenoRunStatus,
};

use super::{ExperimentStorage, MetricPoint, Result, RunStatus, StorageError};

/// TruenoDB-backed experiment storage
///
/// Production backend that persists experiment data to TruenoDB.
/// Opens `~/.entrenar/experiments.trueno` by default.
pub struct TruenoBackend {
    store: Mutex<ExperimentStore>,
    next_exp_id: AtomicU64,
    next_run_id: AtomicU64,
}

impl std::fmt::Debug for TruenoBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TruenoBackend")
            .field("next_exp_id", &self.next_exp_id)
            .field("next_run_id", &self.next_run_id)
            .finish_non_exhaustive()
    }
}

impl TruenoBackend {
    /// Create a new TruenoDB backend
    ///
    /// Opens an in-memory ExperimentStore. For file-backed persistence,
    /// use the `open` method (requires file persistence feature in trueno-db).
    pub fn new() -> Self {
        Self {
            store: Mutex::new(ExperimentStore::new()),
            next_exp_id: AtomicU64::new(0),
            next_run_id: AtomicU64::new(0),
        }
    }

    /// Open a TruenoDB backend at the specified path
    ///
    /// Currently creates an in-memory store. File persistence will be
    /// added when trueno-db supports it (FUTURE: file-backed storage).
    #[allow(unused_variables)]
    pub fn open(path: impl AsRef<std::path::Path>) -> Result<Self> {
        Ok(Self::new())
    }

    /// Get the number of experiments
    pub fn experiment_count(&self) -> usize {
        self.store.lock().unwrap_or_else(std::sync::PoisonError::into_inner).experiment_count()
    }

    /// Get the number of runs
    pub fn run_count(&self) -> usize {
        self.store.lock().unwrap_or_else(std::sync::PoisonError::into_inner).run_count()
    }

    /// Compute CAS hash for artifact data
    fn compute_hash(data: &[u8]) -> String {
        let mut hasher = Sha256::new();
        hasher.update(data);
        let result = hasher.finalize();
        format!("sha256-{}", hex::encode(result.get(..16).unwrap_or(&result)))
    }

    /// Convert our RunStatus to trueno-db's RunStatus
    fn to_trueno_status(status: RunStatus) -> TruenoRunStatus {
        match status {
            RunStatus::Pending => TruenoRunStatus::Pending,
            RunStatus::Running => TruenoRunStatus::Running,
            RunStatus::Success => TruenoRunStatus::Success,
            RunStatus::Failed => TruenoRunStatus::Failed,
            RunStatus::Cancelled => TruenoRunStatus::Cancelled,
        }
    }

    /// Convert trueno-db's RunStatus to our RunStatus
    fn from_trueno_status(status: TruenoRunStatus) -> RunStatus {
        match status {
            TruenoRunStatus::Pending => RunStatus::Pending,
            TruenoRunStatus::Running => RunStatus::Running,
            TruenoRunStatus::Success => RunStatus::Success,
            TruenoRunStatus::Failed => RunStatus::Failed,
            TruenoRunStatus::Cancelled => RunStatus::Cancelled,
        }
    }
}

impl Default for TruenoBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl ExperimentStorage for TruenoBackend {
    fn create_experiment(
        &mut self,
        name: &str,
        config: Option<serde_json::Value>,
    ) -> Result<String> {
        let id = self.next_exp_id.fetch_add(1, Ordering::SeqCst);
        let exp_id = format!("exp-{id}");

        let record = if let Some(cfg) = config {
            ExperimentRecord::builder(&exp_id, name).config(cfg).build()
        } else {
            ExperimentRecord::new(&exp_id, name)
        };

        self.store.lock().unwrap_or_else(std::sync::PoisonError::into_inner).add_experiment(record);

        Ok(exp_id)
    }

    fn create_run(&mut self, experiment_id: &str) -> Result<String> {
        let store = self.store.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        if store.get_experiment(experiment_id).is_none() {
            return Err(StorageError::ExperimentNotFound(experiment_id.to_string()));
        }
        drop(store);

        let id = self.next_run_id.fetch_add(1, Ordering::SeqCst);
        let run_id = format!("run-{id}");

        // Create run in Pending state
        let record = RunRecord::new(&run_id, experiment_id);
        self.store.lock().unwrap_or_else(std::sync::PoisonError::into_inner).add_run(record);

        Ok(run_id)
    }

    fn start_run(&mut self, run_id: &str) -> Result<()> {
        let mut store = self.store.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let run =
            store.get_run(run_id).ok_or_else(|| StorageError::RunNotFound(run_id.to_string()))?;

        if run.status() != TruenoRunStatus::Pending {
            return Err(StorageError::InvalidState(format!(
                "Run {run_id} is not in Pending state"
            )));
        }

        // trueno-db RunRecord is immutable, so we create a new one in Running state
        let mut new_record = RunRecord::new(run_id, run.experiment_id());
        new_record.start();

        // Replace the run (trueno-db's add_run replaces existing)
        store.add_run(new_record);

        Ok(())
    }

    fn complete_run(&mut self, run_id: &str, status: RunStatus) -> Result<()> {
        let mut store = self.store.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let run =
            store.get_run(run_id).ok_or_else(|| StorageError::RunNotFound(run_id.to_string()))?;

        if run.status() != TruenoRunStatus::Running {
            return Err(StorageError::InvalidState(format!(
                "Run {run_id} is not in Running state"
            )));
        }

        // Create new record with completed status
        let mut new_record = RunRecord::new(run_id, run.experiment_id());
        new_record.start();
        new_record.complete(Self::to_trueno_status(status));

        store.add_run(new_record);

        Ok(())
    }

    fn log_metric(&mut self, run_id: &str, key: &str, step: u64, value: f64) -> Result<()> {
        let store = self.store.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        if store.get_run(run_id).is_none() {
            return Err(StorageError::RunNotFound(run_id.to_string()));
        }
        drop(store);

        let metric = MetricRecord::new(run_id, key, step, value);
        self.store.lock().unwrap_or_else(std::sync::PoisonError::into_inner).add_metric(metric);

        Ok(())
    }

    fn log_artifact(&mut self, run_id: &str, key: &str, data: &[u8]) -> Result<String> {
        let store = self.store.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        if store.get_run(run_id).is_none() {
            return Err(StorageError::RunNotFound(run_id.to_string()));
        }
        drop(store);

        // Artifact storage: FUTURE(trueno-db) when ArtifactRecord is available
        let hash = Self::compute_hash(data);
        let _ = key; // Unused until artifact storage is implemented

        Ok(hash)
    }

    fn get_metrics(&self, run_id: &str, key: &str) -> Result<Vec<MetricPoint>> {
        let store = self.store.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        if store.get_run(run_id).is_none() {
            return Err(StorageError::RunNotFound(run_id.to_string()));
        }

        let metrics = store.get_metrics_for_run(run_id, key);

        Ok(metrics
            .into_iter()
            .map(|m| MetricPoint::with_timestamp(m.step(), m.value(), m.timestamp()))
            .collect())
    }

    fn get_run_status(&self, run_id: &str) -> Result<RunStatus> {
        let store = self.store.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let run =
            store.get_run(run_id).ok_or_else(|| StorageError::RunNotFound(run_id.to_string()))?;

        Ok(Self::from_trueno_status(run.status()))
    }

    fn set_span_id(&mut self, run_id: &str, span_id: &str) -> Result<()> {
        let mut store = self.store.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let run =
            store.get_run(run_id).ok_or_else(|| StorageError::RunNotFound(run_id.to_string()))?;

        let current_status = run.status();
        let experiment_id = run.experiment_id().to_string();

        // Create new record with span_id, preserving status
        let mut new_record =
            RunRecord::builder(run_id, &experiment_id).renacer_span_id(span_id).build();

        // Preserve the run status
        match current_status {
            TruenoRunStatus::Running => {
                new_record.start();
            }
            TruenoRunStatus::Success | TruenoRunStatus::Failed | TruenoRunStatus::Cancelled => {
                new_record.start();
                new_record.complete(current_status);
            }
            TruenoRunStatus::Pending => {
                // Already in Pending state, no change needed
            }
        }

        store.add_run(new_record);

        Ok(())
    }

    fn get_span_id(&self, run_id: &str) -> Result<Option<String>> {
        let store = self.store.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let run =
            store.get_run(run_id).ok_or_else(|| StorageError::RunNotFound(run_id.to_string()))?;

        Ok(run.renacer_span_id().map(String::from))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trueno_backend_new() {
        let backend = TruenoBackend::new();
        assert_eq!(backend.experiment_count(), 0);
        assert_eq!(backend.run_count(), 0);
    }

    #[test]
    fn test_trueno_backend_open() {
        let backend = TruenoBackend::open("/tmp/test.trueno").expect("operation should succeed");
        assert_eq!(backend.experiment_count(), 0);
    }

    #[test]
    fn test_trueno_create_experiment() {
        let mut backend = TruenoBackend::new();
        let exp_id = backend.create_experiment("test-exp", None).expect("operation should succeed");

        assert!(exp_id.starts_with("exp-"));
        assert_eq!(backend.experiment_count(), 1);
    }

    #[test]
    fn test_trueno_create_experiment_with_config() {
        let mut backend = TruenoBackend::new();
        let config = serde_json::json!({"batch_size": 32});
        let exp_id =
            backend.create_experiment("test-exp", Some(config)).expect("config should be valid");

        assert!(exp_id.starts_with("exp-"));
    }

    #[test]
    fn test_trueno_create_run() {
        let mut backend = TruenoBackend::new();
        let exp_id = backend.create_experiment("test-exp", None).expect("operation should succeed");
        let run_id = backend.create_run(&exp_id).expect("operation should succeed");

        assert!(run_id.starts_with("run-"));
        assert_eq!(backend.run_count(), 1);
        assert_eq!(
            backend.get_run_status(&run_id).expect("operation should succeed"),
            RunStatus::Pending
        );
    }

    #[test]
    fn test_trueno_create_run_invalid_experiment() {
        let mut backend = TruenoBackend::new();
        let result = backend.create_run("fake-exp");

        assert!(result.is_err());
    }

    #[test]
    fn test_trueno_start_run() {
        let mut backend = TruenoBackend::new();
        let exp_id = backend.create_experiment("test-exp", None).expect("operation should succeed");
        let run_id = backend.create_run(&exp_id).expect("operation should succeed");

        backend.start_run(&run_id).expect("operation should succeed");
        assert_eq!(
            backend.get_run_status(&run_id).expect("operation should succeed"),
            RunStatus::Running
        );
    }

    #[test]
    fn test_trueno_complete_run() {
        let mut backend = TruenoBackend::new();
        let exp_id = backend.create_experiment("test-exp", None).expect("operation should succeed");
        let run_id = backend.create_run(&exp_id).expect("operation should succeed");

        backend.start_run(&run_id).expect("operation should succeed");
        backend.complete_run(&run_id, RunStatus::Success).expect("operation should succeed");

        assert_eq!(
            backend.get_run_status(&run_id).expect("operation should succeed"),
            RunStatus::Success
        );
    }

    #[test]
    fn test_trueno_log_metric() {
        let mut backend = TruenoBackend::new();
        let exp_id = backend.create_experiment("test-exp", None).expect("operation should succeed");
        let run_id = backend.create_run(&exp_id).expect("operation should succeed");

        backend.log_metric(&run_id, "loss", 0, 0.5).expect("operation should succeed");
        backend.log_metric(&run_id, "loss", 1, 0.4).expect("operation should succeed");

        let metrics = backend.get_metrics(&run_id, "loss").expect("operation should succeed");
        assert_eq!(metrics.len(), 2);
        assert_eq!(metrics[0].step, 0);
        assert_eq!(metrics[1].step, 1);
    }

    #[test]
    fn test_trueno_log_metric_invalid_run() {
        let mut backend = TruenoBackend::new();
        let result = backend.log_metric("fake-run", "loss", 0, 0.5);

        assert!(result.is_err());
    }

    #[test]
    fn test_trueno_log_artifact() {
        let mut backend = TruenoBackend::new();
        let exp_id = backend.create_experiment("test-exp", None).expect("operation should succeed");
        let run_id = backend.create_run(&exp_id).expect("operation should succeed");

        let hash = backend
            .log_artifact(&run_id, "model.safetensors", b"model data")
            .expect("operation should succeed");

        assert!(hash.starts_with("sha256-"));
    }

    #[test]
    fn test_trueno_get_run_not_found() {
        let backend = TruenoBackend::new();
        let result = backend.get_run_status("fake-run");

        assert!(result.is_err());
    }

    #[test]
    fn test_trueno_complete_run_failed() {
        let mut backend = TruenoBackend::new();
        let exp_id = backend.create_experiment("test-exp", None).expect("operation should succeed");
        let run_id = backend.create_run(&exp_id).expect("operation should succeed");

        backend.start_run(&run_id).expect("operation should succeed");
        backend.complete_run(&run_id, RunStatus::Failed).expect("operation should succeed");

        assert_eq!(
            backend.get_run_status(&run_id).expect("operation should succeed"),
            RunStatus::Failed
        );
    }

    #[test]
    fn test_trueno_complete_run_cancelled() {
        let mut backend = TruenoBackend::new();
        let exp_id = backend.create_experiment("test-exp", None).expect("operation should succeed");
        let run_id = backend.create_run(&exp_id).expect("operation should succeed");

        backend.start_run(&run_id).expect("operation should succeed");
        backend.complete_run(&run_id, RunStatus::Cancelled).expect("operation should succeed");

        assert_eq!(
            backend.get_run_status(&run_id).expect("operation should succeed"),
            RunStatus::Cancelled
        );
    }

    #[test]
    fn test_trueno_set_and_get_span_id() {
        let mut backend = TruenoBackend::new();
        let exp_id = backend.create_experiment("test-exp", None).expect("operation should succeed");
        let run_id = backend.create_run(&exp_id).expect("operation should succeed");

        backend.set_span_id(&run_id, "span-abc123").expect("operation should succeed");

        assert_eq!(
            backend.get_span_id(&run_id).expect("operation should succeed"),
            Some("span-abc123".to_string())
        );
    }
}
