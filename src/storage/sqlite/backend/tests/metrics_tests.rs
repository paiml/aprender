//! Metrics logging and retrieval tests.

use crate::storage::sqlite::backend::SqliteBackend;
use crate::storage::{ExperimentStorage, StorageError};

#[test]
fn test_log_metric() {
    let mut backend = SqliteBackend::open_in_memory().expect("operation should succeed");
    let exp_id = backend.create_experiment("test-exp", None).expect("operation should succeed");
    let run_id = backend.create_run(&exp_id).expect("operation should succeed");

    backend.log_metric(&run_id, "loss", 0, 0.5).expect("operation should succeed");
    backend.log_metric(&run_id, "loss", 1, 0.4).expect("operation should succeed");
    backend.log_metric(&run_id, "loss", 2, 0.3).expect("operation should succeed");

    let metrics = backend.get_metrics(&run_id, "loss").expect("operation should succeed");
    assert_eq!(metrics.len(), 3);
    assert!((metrics[0].value - 0.5).abs() < f64::EPSILON);
    assert!((metrics[1].value - 0.4).abs() < f64::EPSILON);
    assert!((metrics[2].value - 0.3).abs() < f64::EPSILON);
}

#[test]
fn test_log_metric_run_not_found() {
    let mut backend = SqliteBackend::open_in_memory().expect("operation should succeed");
    let result = backend.log_metric("nonexistent", "loss", 0, 0.5);
    assert!(matches!(result, Err(StorageError::RunNotFound(_))));
}

#[test]
fn test_get_metrics_empty() {
    let mut backend = SqliteBackend::open_in_memory().expect("operation should succeed");
    let exp_id = backend.create_experiment("test-exp", None).expect("operation should succeed");
    let run_id = backend.create_run(&exp_id).expect("operation should succeed");

    let metrics = backend.get_metrics(&run_id, "loss").expect("operation should succeed");
    assert!(metrics.is_empty());
}
