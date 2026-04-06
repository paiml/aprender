//! Metrics operations for SQLite Backend.
//!
//! Contains metric logging and retrieval methods (ExperimentStorage trait implementation).

use super::backend::SqliteBackend;
use crate::storage::{ExperimentStorage, MetricPoint, Result, RunStatus, StorageError};
use chrono::{DateTime, Utc};
use rusqlite::params;
use sha2::{Digest, Sha256};

/// Map a RunStatus to its SQLite TEXT representation
fn status_to_str(status: RunStatus) -> &'static str {
    match status {
        RunStatus::Pending => "pending",
        RunStatus::Running => "running",
        RunStatus::Success => "completed",
        RunStatus::Failed => "failed",
        RunStatus::Cancelled => "cancelled",
    }
}

/// Parse a SQLite TEXT status back to RunStatus
pub(crate) fn str_to_status(s: &str) -> RunStatus {
    match s {
        "pending" => RunStatus::Pending,
        "running" => RunStatus::Running,
        "completed" => RunStatus::Success,
        "failed" => RunStatus::Failed,
        "cancelled" => RunStatus::Cancelled,
        _ => RunStatus::Failed,
    }
}

impl ExperimentStorage for SqliteBackend {
    fn create_experiment(
        &mut self,
        name: &str,
        config: Option<serde_json::Value>,
    ) -> Result<String> {
        let id = Self::generate_id();
        let config_json = config.map(|c| c.to_string());
        let now = Utc::now().to_rfc3339();

        let conn = self.lock_conn()?;
        conn.execute(
            "INSERT INTO experiments (id, name, config, created_at, updated_at) VALUES (?1, ?2, ?3, ?4, ?5)",
            params![id, name, config_json, now, now],
        )
        .map_err(|e| StorageError::Backend(format!("Failed to create experiment: {e}")))?;

        Ok(id)
    }

    fn create_run(&mut self, experiment_id: &str) -> Result<String> {
        let conn = self.lock_conn()?;

        // Verify experiment exists
        let exists: bool = conn
            .query_row(
                "SELECT EXISTS(SELECT 1 FROM experiments WHERE id = ?1)",
                [experiment_id],
                |row| row.get(0),
            )
            .map_err(|e| StorageError::Backend(format!("Failed to check experiment: {e}")))?;

        if !exists {
            return Err(StorageError::ExperimentNotFound(experiment_id.to_string()));
        }

        let id = Self::generate_id();
        let now = Utc::now().to_rfc3339();

        conn.execute(
            "INSERT INTO runs (id, experiment_id, status, start_time) VALUES (?1, ?2, 'pending', ?3)",
            params![id, experiment_id, now],
        )
        .map_err(|e| StorageError::Backend(format!("Failed to create run: {e}")))?;

        Ok(id)
    }

    fn start_run(&mut self, run_id: &str) -> Result<()> {
        let conn = self.lock_conn()?;

        let current_status: String = conn
            .query_row("SELECT status FROM runs WHERE id = ?1", [run_id], |row| row.get(0))
            .map_err(|e| match e {
                rusqlite::Error::QueryReturnedNoRows => {
                    StorageError::RunNotFound(run_id.to_string())
                }
                _ => StorageError::Backend(format!("Failed to get run status: {e}")),
            })?;

        if current_status != "pending" {
            return Err(StorageError::InvalidState(format!(
                "Cannot start run in {current_status} status"
            )));
        }

        let now = Utc::now().to_rfc3339();
        conn.execute(
            "UPDATE runs SET status = 'running', start_time = ?1 WHERE id = ?2",
            params![now, run_id],
        )
        .map_err(|e| StorageError::Backend(format!("Failed to start run: {e}")))?;

        Ok(())
    }

    fn complete_run(&mut self, run_id: &str, status: RunStatus) -> Result<()> {
        let conn = self.lock_conn()?;

        let current_status: String = conn
            .query_row("SELECT status FROM runs WHERE id = ?1", [run_id], |row| row.get(0))
            .map_err(|e| match e {
                rusqlite::Error::QueryReturnedNoRows => {
                    StorageError::RunNotFound(run_id.to_string())
                }
                _ => StorageError::Backend(format!("Failed to get run status: {e}")),
            })?;

        if current_status != "running" {
            return Err(StorageError::InvalidState(format!(
                "Cannot complete run in {current_status} status"
            )));
        }

        let now = Utc::now().to_rfc3339();
        conn.execute(
            "UPDATE runs SET status = ?1, end_time = ?2 WHERE id = ?3",
            params![status_to_str(status), now, run_id],
        )
        .map_err(|e| StorageError::Backend(format!("Failed to complete run: {e}")))?;

        Ok(())
    }

    fn log_metric(&mut self, run_id: &str, key: &str, step: u64, value: f64) -> Result<()> {
        let conn = self.lock_conn()?;

        // Verify run exists
        let exists: bool = conn
            .query_row("SELECT EXISTS(SELECT 1 FROM runs WHERE id = ?1)", [run_id], |row| {
                row.get(0)
            })
            .map_err(|e| StorageError::Backend(format!("Failed to check run: {e}")))?;

        if !exists {
            return Err(StorageError::RunNotFound(run_id.to_string()));
        }

        let now = Utc::now().to_rfc3339();
        conn.execute(
            "INSERT INTO metrics (run_id, key, step, value, timestamp) VALUES (?1, ?2, ?3, ?4, ?5)",
            params![run_id, key, step as i64, value, now],
        )
        .map_err(|e| StorageError::Backend(format!("Failed to log metric: {e}")))?;

        Ok(())
    }

    fn log_artifact(&mut self, run_id: &str, key: &str, data: &[u8]) -> Result<String> {
        let conn = self.lock_conn()?;

        // Verify run exists
        let exists: bool = conn
            .query_row("SELECT EXISTS(SELECT 1 FROM runs WHERE id = ?1)", [run_id], |row| {
                row.get(0)
            })
            .map_err(|e| StorageError::Backend(format!("Failed to check run: {e}")))?;

        if !exists {
            return Err(StorageError::RunNotFound(run_id.to_string()));
        }

        // Compute SHA-256 for content-addressable storage
        let mut hasher = Sha256::new();
        hasher.update(data);
        let sha256 = format!("{:x}", hasher.finalize());

        let id = Self::generate_id();
        let size = data.len() as i64;

        conn.execute(
            "INSERT INTO artifacts (id, run_id, path, size_bytes, sha256, data) VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![id, run_id, key, size, sha256, data],
        )
        .map_err(|e| StorageError::Backend(format!("Failed to log artifact: {e}")))?;

        Ok(sha256)
    }

    fn get_metrics(&self, run_id: &str, key: &str) -> Result<Vec<MetricPoint>> {
        let conn = self.lock_conn()?;

        // Verify run exists
        let exists: bool = conn
            .query_row("SELECT EXISTS(SELECT 1 FROM runs WHERE id = ?1)", [run_id], |row| {
                row.get(0)
            })
            .map_err(|e| StorageError::Backend(format!("Failed to check run: {e}")))?;

        if !exists {
            return Err(StorageError::RunNotFound(run_id.to_string()));
        }

        let mut stmt = conn
            .prepare("SELECT step, value, timestamp FROM metrics WHERE run_id = ?1 AND key = ?2 ORDER BY step")
            .map_err(|e| StorageError::Backend(format!("Failed to prepare metrics query: {e}")))?;

        let points = stmt
            .query_map(params![run_id, key], |row| {
                let step: i64 = row.get(0)?;
                let value: f64 = row.get(1)?;
                let ts_str: String = row.get(2)?;
                let timestamp: DateTime<Utc> = ts_str.parse().unwrap_or_else(|_| Utc::now());
                Ok(MetricPoint::with_timestamp(step as u64, value, timestamp))
            })
            .map_err(|e| StorageError::Backend(format!("Failed to query metrics: {e}")))?
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| StorageError::Backend(format!("Failed to read metric row: {e}")))?;

        Ok(points)
    }

    fn get_run_status(&self, run_id: &str) -> Result<RunStatus> {
        let conn = self.lock_conn()?;

        let status_str: String = conn
            .query_row("SELECT status FROM runs WHERE id = ?1", [run_id], |row| row.get(0))
            .map_err(|e| match e {
                rusqlite::Error::QueryReturnedNoRows => {
                    StorageError::RunNotFound(run_id.to_string())
                }
                _ => StorageError::Backend(format!("Failed to get run status: {e}")),
            })?;

        Ok(str_to_status(&status_str))
    }

    fn set_span_id(&mut self, run_id: &str, span_id: &str) -> Result<()> {
        let conn = self.lock_conn()?;

        // Verify run exists
        let exists: bool = conn
            .query_row("SELECT EXISTS(SELECT 1 FROM runs WHERE id = ?1)", [run_id], |row| {
                row.get(0)
            })
            .map_err(|e| StorageError::Backend(format!("Failed to check run: {e}")))?;

        if !exists {
            return Err(StorageError::RunNotFound(run_id.to_string()));
        }

        conn.execute(
            "INSERT OR REPLACE INTO span_ids (run_id, span_id) VALUES (?1, ?2)",
            params![run_id, span_id],
        )
        .map_err(|e| StorageError::Backend(format!("Failed to set span ID: {e}")))?;

        Ok(())
    }

    fn get_span_id(&self, run_id: &str) -> Result<Option<String>> {
        let conn = self.lock_conn()?;

        // Verify run exists
        let exists: bool = conn
            .query_row("SELECT EXISTS(SELECT 1 FROM runs WHERE id = ?1)", [run_id], |row| {
                row.get(0)
            })
            .map_err(|e| StorageError::Backend(format!("Failed to check run: {e}")))?;

        if !exists {
            return Err(StorageError::RunNotFound(run_id.to_string()));
        }

        let result =
            conn.query_row("SELECT span_id FROM span_ids WHERE run_id = ?1", [run_id], |row| {
                row.get(0)
            });

        match result {
            Ok(span_id) => Ok(Some(span_id)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(StorageError::Backend(format!("Failed to get span ID: {e}"))),
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::storage::ExperimentStorage;

    fn test_backend() -> SqliteBackend {
        SqliteBackend::open_in_memory().expect("in-memory db should succeed")
    }

    #[test]
    fn test_status_to_str_all_variants() {
        assert_eq!(status_to_str(RunStatus::Pending), "pending");
        assert_eq!(status_to_str(RunStatus::Running), "running");
        assert_eq!(status_to_str(RunStatus::Success), "completed");
        assert_eq!(status_to_str(RunStatus::Failed), "failed");
        assert_eq!(status_to_str(RunStatus::Cancelled), "cancelled");
    }

    #[test]
    fn test_str_to_status_all_variants() {
        assert_eq!(str_to_status("pending"), RunStatus::Pending);
        assert_eq!(str_to_status("running"), RunStatus::Running);
        assert_eq!(str_to_status("completed"), RunStatus::Success);
        assert_eq!(str_to_status("failed"), RunStatus::Failed);
        assert_eq!(str_to_status("cancelled"), RunStatus::Cancelled);
    }

    #[test]
    fn test_str_to_status_unknown_defaults_failed() {
        assert_eq!(str_to_status("xyz"), RunStatus::Failed);
        assert_eq!(str_to_status(""), RunStatus::Failed);
    }

    #[test]
    fn test_create_experiment() {
        let mut backend = test_backend();
        let id = backend.create_experiment("test-exp", None).unwrap();
        assert!(!id.is_empty());
    }

    #[test]
    fn test_create_experiment_with_config() {
        let mut backend = test_backend();
        let config = serde_json::json!({"lr": 0.001, "epochs": 10});
        let id = backend.create_experiment("config-exp", Some(config)).unwrap();
        assert!(!id.is_empty());
    }

    #[test]
    fn test_create_run() {
        let mut backend = test_backend();
        let exp_id = backend.create_experiment("test", None).unwrap();
        let run_id = backend.create_run(&exp_id).unwrap();
        assert!(!run_id.is_empty());
    }

    #[test]
    fn test_create_run_nonexistent_experiment() {
        let mut backend = test_backend();
        let result = backend.create_run("nonexistent-exp");
        assert!(result.is_err());
    }

    #[test]
    fn test_start_run() {
        let mut backend = test_backend();
        let exp_id = backend.create_experiment("test", None).unwrap();
        let run_id = backend.create_run(&exp_id).unwrap();
        backend.start_run(&run_id).unwrap();
        let status = backend.get_run_status(&run_id).unwrap();
        assert_eq!(status, RunStatus::Running);
    }

    #[test]
    fn test_start_run_nonexistent() {
        let mut backend = test_backend();
        let result = backend.start_run("nonexistent-run");
        assert!(result.is_err());
    }

    #[test]
    fn test_start_run_not_pending() {
        let mut backend = test_backend();
        let exp_id = backend.create_experiment("test", None).unwrap();
        let run_id = backend.create_run(&exp_id).unwrap();
        backend.start_run(&run_id).unwrap();
        // Starting again should fail (status is "running", not "pending")
        let result = backend.start_run(&run_id);
        assert!(result.is_err());
    }

    #[test]
    fn test_complete_run_success() {
        let mut backend = test_backend();
        let exp_id = backend.create_experiment("test", None).unwrap();
        let run_id = backend.create_run(&exp_id).unwrap();
        backend.start_run(&run_id).unwrap();
        backend.complete_run(&run_id, RunStatus::Success).unwrap();
        let status = backend.get_run_status(&run_id).unwrap();
        assert_eq!(status, RunStatus::Success);
    }

    #[test]
    fn test_complete_run_failed() {
        let mut backend = test_backend();
        let exp_id = backend.create_experiment("test", None).unwrap();
        let run_id = backend.create_run(&exp_id).unwrap();
        backend.start_run(&run_id).unwrap();
        backend.complete_run(&run_id, RunStatus::Failed).unwrap();
        let status = backend.get_run_status(&run_id).unwrap();
        assert_eq!(status, RunStatus::Failed);
    }

    #[test]
    fn test_complete_run_cancelled() {
        let mut backend = test_backend();
        let exp_id = backend.create_experiment("test", None).unwrap();
        let run_id = backend.create_run(&exp_id).unwrap();
        backend.start_run(&run_id).unwrap();
        backend.complete_run(&run_id, RunStatus::Cancelled).unwrap();
        let status = backend.get_run_status(&run_id).unwrap();
        assert_eq!(status, RunStatus::Cancelled);
    }

    #[test]
    fn test_complete_run_not_running() {
        let mut backend = test_backend();
        let exp_id = backend.create_experiment("test", None).unwrap();
        let run_id = backend.create_run(&exp_id).unwrap();
        // Completing a "pending" run should fail
        let result = backend.complete_run(&run_id, RunStatus::Success);
        assert!(result.is_err());
    }

    #[test]
    fn test_complete_run_nonexistent() {
        let mut backend = test_backend();
        let result = backend.complete_run("nonexistent-run", RunStatus::Success);
        assert!(result.is_err());
    }

    #[test]
    fn test_log_metric() {
        let mut backend = test_backend();
        let exp_id = backend.create_experiment("test", None).unwrap();
        let run_id = backend.create_run(&exp_id).unwrap();
        backend.log_metric(&run_id, "loss", 0, 0.5).unwrap();
        backend.log_metric(&run_id, "loss", 1, 0.4).unwrap();
        backend.log_metric(&run_id, "loss", 2, 0.3).unwrap();
    }

    #[test]
    fn test_log_metric_nonexistent_run() {
        let mut backend = test_backend();
        let result = backend.log_metric("nonexistent-run", "loss", 0, 0.5);
        assert!(result.is_err());
    }

    #[test]
    fn test_get_metrics() {
        let mut backend = test_backend();
        let exp_id = backend.create_experiment("test", None).unwrap();
        let run_id = backend.create_run(&exp_id).unwrap();
        backend.log_metric(&run_id, "loss", 0, 0.5).unwrap();
        backend.log_metric(&run_id, "loss", 1, 0.4).unwrap();
        backend.log_metric(&run_id, "accuracy", 0, 0.8).unwrap();

        let loss_metrics = backend.get_metrics(&run_id, "loss").unwrap();
        assert_eq!(loss_metrics.len(), 2);
        assert_eq!(loss_metrics[0].step, 0);
        assert!((loss_metrics[0].value - 0.5).abs() < f64::EPSILON);
        assert_eq!(loss_metrics[1].step, 1);

        let acc_metrics = backend.get_metrics(&run_id, "accuracy").unwrap();
        assert_eq!(acc_metrics.len(), 1);
    }

    #[test]
    fn test_get_metrics_nonexistent_run() {
        let backend = test_backend();
        let result = backend.get_metrics("nonexistent-run", "loss");
        assert!(result.is_err());
    }

    #[test]
    fn test_get_metrics_empty() {
        let mut backend = test_backend();
        let exp_id = backend.create_experiment("test", None).unwrap();
        let run_id = backend.create_run(&exp_id).unwrap();
        let metrics = backend.get_metrics(&run_id, "loss").unwrap();
        assert!(metrics.is_empty());
    }

    #[test]
    fn test_get_run_status() {
        let mut backend = test_backend();
        let exp_id = backend.create_experiment("test", None).unwrap();
        let run_id = backend.create_run(&exp_id).unwrap();
        let status = backend.get_run_status(&run_id).unwrap();
        assert_eq!(status, RunStatus::Pending);
    }

    #[test]
    fn test_get_run_status_nonexistent() {
        let backend = test_backend();
        let result = backend.get_run_status("nonexistent-run");
        assert!(result.is_err());
    }

    #[test]
    fn test_log_artifact() {
        let mut backend = test_backend();
        let exp_id = backend.create_experiment("test", None).unwrap();
        let run_id = backend.create_run(&exp_id).unwrap();
        let sha = backend.log_artifact(&run_id, "model.bin", b"fake model data").unwrap();
        assert!(!sha.is_empty());
        // SHA-256 should be 64 hex chars
        assert_eq!(sha.len(), 64);
    }

    #[test]
    fn test_log_artifact_nonexistent_run() {
        let mut backend = test_backend();
        let result = backend.log_artifact("nonexistent-run", "file.bin", b"data");
        assert!(result.is_err());
    }

    #[test]
    fn test_log_artifact_deterministic_hash() {
        let mut backend = test_backend();
        let exp_id = backend.create_experiment("test", None).unwrap();
        let run_id1 = backend.create_run(&exp_id).unwrap();
        let run_id2 = backend.create_run(&exp_id).unwrap();
        let sha1 = backend.log_artifact(&run_id1, "file.bin", b"same data").unwrap();
        let sha2 = backend.log_artifact(&run_id2, "file.bin", b"same data").unwrap();
        // Same data -> same SHA-256
        assert_eq!(sha1, sha2);
    }

    #[test]
    fn test_set_and_get_span_id() {
        let mut backend = test_backend();
        let exp_id = backend.create_experiment("test", None).unwrap();
        let run_id = backend.create_run(&exp_id).unwrap();

        // Initially no span ID
        let span = backend.get_span_id(&run_id).unwrap();
        assert!(span.is_none());

        // Set span ID
        backend.set_span_id(&run_id, "span-12345").unwrap();
        let span = backend.get_span_id(&run_id).unwrap();
        assert_eq!(span, Some("span-12345".to_string()));
    }

    #[test]
    fn test_set_span_id_nonexistent_run() {
        let mut backend = test_backend();
        let result = backend.set_span_id("nonexistent-run", "span-123");
        assert!(result.is_err());
    }

    #[test]
    fn test_get_span_id_nonexistent_run() {
        let backend = test_backend();
        let result = backend.get_span_id("nonexistent-run");
        assert!(result.is_err());
    }

    #[test]
    fn test_set_span_id_overwrite() {
        let mut backend = test_backend();
        let exp_id = backend.create_experiment("test", None).unwrap();
        let run_id = backend.create_run(&exp_id).unwrap();

        backend.set_span_id(&run_id, "span-1").unwrap();
        backend.set_span_id(&run_id, "span-2").unwrap();
        let span = backend.get_span_id(&run_id).unwrap();
        assert_eq!(span, Some("span-2".to_string()));
    }

    #[test]
    fn test_full_lifecycle() {
        let mut backend = test_backend();
        let exp_id = backend.create_experiment("lifecycle-test", None).unwrap();
        let run_id = backend.create_run(&exp_id).unwrap();

        assert_eq!(backend.get_run_status(&run_id).unwrap(), RunStatus::Pending);

        backend.start_run(&run_id).unwrap();
        assert_eq!(backend.get_run_status(&run_id).unwrap(), RunStatus::Running);

        backend.log_metric(&run_id, "loss", 0, 1.0).unwrap();
        backend.log_metric(&run_id, "loss", 1, 0.5).unwrap();

        backend.complete_run(&run_id, RunStatus::Success).unwrap();
        assert_eq!(backend.get_run_status(&run_id).unwrap(), RunStatus::Success);

        let metrics = backend.get_metrics(&run_id, "loss").unwrap();
        assert_eq!(metrics.len(), 2);
    }

    #[test]
    fn test_metrics_ordered_by_step() {
        let mut backend = test_backend();
        let exp_id = backend.create_experiment("test", None).unwrap();
        let run_id = backend.create_run(&exp_id).unwrap();
        // Insert out of order
        backend.log_metric(&run_id, "loss", 5, 0.1).unwrap();
        backend.log_metric(&run_id, "loss", 1, 0.5).unwrap();
        backend.log_metric(&run_id, "loss", 3, 0.3).unwrap();

        let metrics = backend.get_metrics(&run_id, "loss").unwrap();
        assert_eq!(metrics.len(), 3);
        assert_eq!(metrics[0].step, 1);
        assert_eq!(metrics[1].step, 3);
        assert_eq!(metrics[2].step, 5);
    }
}
