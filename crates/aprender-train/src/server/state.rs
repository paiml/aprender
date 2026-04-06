//! Server application state
//!
//! Shared state for the tracking server with thread-safe storage.

use crate::server::{ExperimentResponse, Result, RunResponse, ServerConfig, ServerError};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::Instant;

/// Experiment data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Experiment {
    pub id: String,
    pub name: String,
    pub description: Option<String>,
    pub created_at: DateTime<Utc>,
    pub tags: HashMap<String, String>,
}

impl From<Experiment> for ExperimentResponse {
    fn from(exp: Experiment) -> Self {
        Self {
            id: exp.id,
            name: exp.name,
            description: exp.description,
            created_at: exp.created_at.to_rfc3339(),
            tags: exp.tags,
        }
    }
}

/// Run status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RunStatus {
    Running,
    Completed,
    Failed,
    Killed,
}

impl std::fmt::Display for RunStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RunStatus::Running => write!(f, "running"),
            RunStatus::Completed => write!(f, "completed"),
            RunStatus::Failed => write!(f, "failed"),
            RunStatus::Killed => write!(f, "killed"),
        }
    }
}

impl std::str::FromStr for RunStatus {
    type Err = ServerError;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "running" => Ok(RunStatus::Running),
            "completed" => Ok(RunStatus::Completed),
            "failed" => Ok(RunStatus::Failed),
            "killed" => Ok(RunStatus::Killed),
            _ => Err(ServerError::Validation(format!("Invalid status: {s}"))),
        }
    }
}

/// Run data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Run {
    pub id: String,
    pub experiment_id: String,
    pub name: Option<String>,
    pub status: RunStatus,
    pub start_time: DateTime<Utc>,
    pub end_time: Option<DateTime<Utc>>,
    pub params: HashMap<String, serde_json::Value>,
    pub metrics: HashMap<String, f64>,
    pub tags: HashMap<String, String>,
}

impl From<Run> for RunResponse {
    fn from(run: Run) -> Self {
        Self {
            id: run.id,
            experiment_id: run.experiment_id,
            name: run.name,
            status: run.status.to_string(),
            start_time: run.start_time.to_rfc3339(),
            end_time: run.end_time.map(|t| t.to_rfc3339()),
            params: run.params,
            metrics: run.metrics,
            tags: run.tags,
        }
    }
}

/// In-memory storage for experiments and runs
#[derive(Debug, Default)]
pub struct InMemoryStorage {
    experiments: RwLock<HashMap<String, Experiment>>,
    runs: RwLock<HashMap<String, Run>>,
    counter: RwLock<u64>,
}

impl InMemoryStorage {
    pub fn new() -> Self {
        Self::default()
    }

    /// Generate a unique ID
    pub fn generate_id(&self, prefix: &str) -> String {
        let mut counter = self.counter.write().expect("counter RwLock must not be poisoned");
        *counter += 1;
        format!("{}-{:08x}", prefix, *counter)
    }

    /// Create a new experiment
    pub fn create_experiment(
        &self,
        name: &str,
        description: Option<String>,
        tags: Option<HashMap<String, String>>,
    ) -> Result<Experiment> {
        let id = self.generate_id("exp");
        let experiment = Experiment {
            id: id.clone(),
            name: name.to_string(),
            description,
            created_at: Utc::now(),
            tags: tags.unwrap_or_default(),
        };

        let mut experiments = self
            .experiments
            .write()
            .map_err(|e| ServerError::Internal(format!("Lock error: {e}")))?;
        experiments.insert(id, experiment.clone());

        Ok(experiment)
    }

    /// Get an experiment by ID
    pub fn get_experiment(&self, id: &str) -> Result<Experiment> {
        let experiments = self
            .experiments
            .read()
            .map_err(|e| ServerError::Internal(format!("Lock error: {e}")))?;

        experiments
            .get(id)
            .cloned()
            .ok_or_else(|| ServerError::NotFound(format!("Experiment not found: {id}")))
    }

    /// List all experiments
    pub fn list_experiments(&self) -> Result<Vec<Experiment>> {
        let experiments = self
            .experiments
            .read()
            .map_err(|e| ServerError::Internal(format!("Lock error: {e}")))?;

        Ok(experiments.values().cloned().collect())
    }

    /// Create a new run
    pub fn create_run(
        &self,
        experiment_id: &str,
        name: Option<String>,
        tags: Option<HashMap<String, String>>,
    ) -> Result<Run> {
        // Verify experiment exists
        self.get_experiment(experiment_id)?;

        let id = self.generate_id("run");
        let run = Run {
            id: id.clone(),
            experiment_id: experiment_id.to_string(),
            name,
            status: RunStatus::Running,
            start_time: Utc::now(),
            end_time: None,
            params: HashMap::new(),
            metrics: HashMap::new(),
            tags: tags.unwrap_or_default(),
        };

        let mut runs =
            self.runs.write().map_err(|e| ServerError::Internal(format!("Lock error: {e}")))?;
        runs.insert(id, run.clone());

        Ok(run)
    }

    /// Get a run by ID
    pub fn get_run(&self, id: &str) -> Result<Run> {
        let runs =
            self.runs.read().map_err(|e| ServerError::Internal(format!("Lock error: {e}")))?;

        runs.get(id).cloned().ok_or_else(|| ServerError::NotFound(format!("Run not found: {id}")))
    }

    /// Update run status
    pub fn update_run(
        &self,
        id: &str,
        status: Option<RunStatus>,
        end_time: Option<DateTime<Utc>>,
    ) -> Result<Run> {
        let mut runs =
            self.runs.write().map_err(|e| ServerError::Internal(format!("Lock error: {e}")))?;

        let run = runs
            .get_mut(id)
            .ok_or_else(|| ServerError::NotFound(format!("Run not found: {id}")))?;

        if let Some(s) = status {
            run.status = s;
        }
        if let Some(t) = end_time {
            run.end_time = Some(t);
        }

        Ok(run.clone())
    }

    /// Log parameters for a run
    pub fn log_params(
        &self,
        run_id: &str,
        params: HashMap<String, serde_json::Value>,
    ) -> Result<()> {
        let mut runs =
            self.runs.write().map_err(|e| ServerError::Internal(format!("Lock error: {e}")))?;

        let run = runs
            .get_mut(run_id)
            .ok_or_else(|| ServerError::NotFound(format!("Run not found: {run_id}")))?;

        run.params.extend(params);
        Ok(())
    }

    /// Log metrics for a run
    pub fn log_metrics(&self, run_id: &str, metrics: HashMap<String, f64>) -> Result<()> {
        let mut runs =
            self.runs.write().map_err(|e| ServerError::Internal(format!("Lock error: {e}")))?;

        let run = runs
            .get_mut(run_id)
            .ok_or_else(|| ServerError::NotFound(format!("Run not found: {run_id}")))?;

        run.metrics.extend(metrics);
        Ok(())
    }

    /// Count experiments
    pub fn experiments_count(&self) -> usize {
        self.experiments.read().map(|e| e.len()).unwrap_or(0)
    }

    /// Count runs
    pub fn runs_count(&self) -> usize {
        self.runs.read().map(|r| r.len()).unwrap_or(0)
    }
}

/// Application state shared across handlers
#[derive(Clone)]
pub struct AppState {
    pub storage: Arc<InMemoryStorage>,
    pub config: ServerConfig,
    pub start_time: Instant,
}

impl AppState {
    pub fn new(config: ServerConfig) -> Self {
        Self { storage: Arc::new(InMemoryStorage::new()), config, start_time: Instant::now() }
    }

    /// Get uptime in seconds
    pub fn uptime_secs(&self) -> u64 {
        self.start_time.elapsed().as_secs()
    }

    /// Create with an explicit start time (for deterministic testing)
    #[cfg(test)]
    pub fn with_start_time(config: ServerConfig, start_time: Instant) -> Self {
        Self { storage: Arc::new(InMemoryStorage::new()), config, start_time }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_in_memory_storage_new() {
        let storage = InMemoryStorage::new();
        assert_eq!(storage.experiments_count(), 0);
        assert_eq!(storage.runs_count(), 0);
    }

    #[test]
    fn test_generate_id() {
        let storage = InMemoryStorage::new();
        let id1 = storage.generate_id("test");
        let id2 = storage.generate_id("test");
        assert!(id1.starts_with("test-"));
        assert!(id2.starts_with("test-"));
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_create_experiment() {
        let storage = InMemoryStorage::new();
        let exp = storage
            .create_experiment("my-exp", Some("desc".into()), None)
            .expect("operation should succeed");
        assert!(exp.id.starts_with("exp-"));
        assert_eq!(exp.name, "my-exp");
        assert_eq!(storage.experiments_count(), 1);
    }

    #[test]
    fn test_get_experiment() {
        let storage = InMemoryStorage::new();
        let exp = storage.create_experiment("test", None, None).expect("operation should succeed");
        let retrieved = storage.get_experiment(&exp.id).expect("operation should succeed");
        assert_eq!(retrieved.name, "test");
    }

    #[test]
    fn test_get_experiment_not_found() {
        let storage = InMemoryStorage::new();
        let result = storage.get_experiment("nonexistent");
        assert!(result.is_err());
    }

    #[test]
    fn test_list_experiments() {
        let storage = InMemoryStorage::new();
        storage.create_experiment("exp1", None, None).expect("operation should succeed");
        storage.create_experiment("exp2", None, None).expect("operation should succeed");
        let list = storage.list_experiments().expect("operation should succeed");
        assert_eq!(list.len(), 2);
    }

    #[test]
    fn test_create_run() {
        let storage = InMemoryStorage::new();
        let exp = storage.create_experiment("test", None, None).expect("operation should succeed");
        let run = storage
            .create_run(&exp.id, Some("run-1".into()), None)
            .expect("operation should succeed");
        assert!(run.id.starts_with("run-"));
        assert_eq!(run.experiment_id, exp.id);
        assert_eq!(run.status, RunStatus::Running);
    }

    #[test]
    fn test_create_run_invalid_experiment() {
        let storage = InMemoryStorage::new();
        let result = storage.create_run("nonexistent", None, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_update_run() {
        let storage = InMemoryStorage::new();
        let exp = storage.create_experiment("test", None, None).expect("operation should succeed");
        let run = storage.create_run(&exp.id, None, None).expect("operation should succeed");

        let updated = storage
            .update_run(&run.id, Some(RunStatus::Completed), None)
            .expect("operation should succeed");
        assert_eq!(updated.status, RunStatus::Completed);
    }

    #[test]
    fn test_log_params() {
        let storage = InMemoryStorage::new();
        let exp = storage.create_experiment("test", None, None).expect("operation should succeed");
        let run = storage.create_run(&exp.id, None, None).expect("operation should succeed");

        let mut params = HashMap::new();
        params.insert("lr".to_string(), serde_json::json!(0.001));
        storage.log_params(&run.id, params).expect("operation should succeed");

        let updated = storage.get_run(&run.id).expect("operation should succeed");
        assert!(updated.params.contains_key("lr"));
    }

    #[test]
    fn test_log_metrics() {
        let storage = InMemoryStorage::new();
        let exp = storage.create_experiment("test", None, None).expect("operation should succeed");
        let run = storage.create_run(&exp.id, None, None).expect("operation should succeed");

        let mut metrics = HashMap::new();
        metrics.insert("loss".to_string(), 0.5);
        storage.log_metrics(&run.id, metrics).expect("operation should succeed");

        let updated = storage.get_run(&run.id).expect("operation should succeed");
        assert_eq!(updated.metrics.get("loss"), Some(&0.5));
    }

    #[test]
    fn test_run_status_from_str() {
        assert_eq!(
            "running".parse::<RunStatus>().expect("parsing should succeed"),
            RunStatus::Running
        );
        assert_eq!(
            "completed".parse::<RunStatus>().expect("parsing should succeed"),
            RunStatus::Completed
        );
        assert_eq!(
            "failed".parse::<RunStatus>().expect("parsing should succeed"),
            RunStatus::Failed
        );
        assert_eq!(
            "killed".parse::<RunStatus>().expect("parsing should succeed"),
            RunStatus::Killed
        );
        assert!("invalid".parse::<RunStatus>().is_err());
    }

    #[test]
    fn test_run_status_display() {
        assert_eq!(RunStatus::Running.to_string(), "running");
        assert_eq!(RunStatus::Completed.to_string(), "completed");
    }

    #[test]
    fn test_app_state_new() {
        let config = ServerConfig::default();
        let state = AppState::new(config);
        assert_eq!(state.storage.experiments_count(), 0);
    }

    #[test]
    fn test_app_state_uptime_deterministic() {
        // Use with_start_time to avoid flaky Instant::now() timing assertions
        let config = ServerConfig::default();
        let state = AppState::with_start_time(config, Instant::now());
        // uptime_secs returns u64 truncated seconds; just verify it doesn't panic
        let _uptime = state.uptime_secs();
    }

    #[test]
    fn test_experiment_to_response() {
        let exp = Experiment {
            id: "exp-1".to_string(),
            name: "test".to_string(),
            description: None,
            created_at: Utc::now(),
            tags: HashMap::new(),
        };
        let resp: ExperimentResponse = exp.into();
        assert_eq!(resp.id, "exp-1");
    }

    #[test]
    fn test_run_to_response() {
        let run = Run {
            id: "run-1".to_string(),
            experiment_id: "exp-1".to_string(),
            name: None,
            status: RunStatus::Running,
            start_time: Utc::now(),
            end_time: None,
            params: HashMap::new(),
            metrics: HashMap::new(),
            tags: HashMap::new(),
        };
        let resp: RunResponse = run.into();
        assert_eq!(resp.id, "run-1");
        assert_eq!(resp.status, "running");
    }
}
