//! Run lifecycle tests.

use crate::storage::sqlite::backend::SqliteBackend;
use crate::storage::{ExperimentStorage, RunStatus, StorageError};

#[test]
fn test_create_run() {
    let mut backend = SqliteBackend::open_in_memory().expect("operation should succeed");
    let exp_id = backend.create_experiment("test-exp", None).expect("operation should succeed");
    let run_id = backend.create_run(&exp_id).expect("operation should succeed");

    assert!(!run_id.is_empty());
    assert_eq!(
        backend.get_run_status(&run_id).expect("operation should succeed"),
        RunStatus::Pending
    );
}

#[test]
fn test_create_run_experiment_not_found() {
    let mut backend = SqliteBackend::open_in_memory().expect("operation should succeed");
    let result = backend.create_run("nonexistent");
    assert!(matches!(result, Err(StorageError::ExperimentNotFound(_))));
}

#[test]
fn test_start_run() {
    let mut backend = SqliteBackend::open_in_memory().expect("operation should succeed");
    let exp_id = backend.create_experiment("test-exp", None).expect("operation should succeed");
    let run_id = backend.create_run(&exp_id).expect("operation should succeed");

    backend.start_run(&run_id).expect("operation should succeed");
    assert_eq!(
        backend.get_run_status(&run_id).expect("operation should succeed"),
        RunStatus::Running
    );
}

#[test]
fn test_start_run_not_found() {
    let mut backend = SqliteBackend::open_in_memory().expect("operation should succeed");
    let result = backend.start_run("nonexistent");
    assert!(matches!(result, Err(StorageError::RunNotFound(_))));
}

#[test]
fn test_start_run_invalid_state() {
    let mut backend = SqliteBackend::open_in_memory().expect("operation should succeed");
    let exp_id = backend.create_experiment("test-exp", None).expect("operation should succeed");
    let run_id = backend.create_run(&exp_id).expect("operation should succeed");

    backend.start_run(&run_id).expect("operation should succeed");
    let result = backend.start_run(&run_id);
    assert!(matches!(result, Err(StorageError::InvalidState(_))));
}

#[test]
fn test_complete_run_success() {
    let mut backend = SqliteBackend::open_in_memory().expect("operation should succeed");
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
fn test_complete_run_failed() {
    let mut backend = SqliteBackend::open_in_memory().expect("operation should succeed");
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
fn test_complete_run_invalid_state() {
    let mut backend = SqliteBackend::open_in_memory().expect("operation should succeed");
    let exp_id = backend.create_experiment("test-exp", None).expect("operation should succeed");
    let run_id = backend.create_run(&exp_id).expect("operation should succeed");

    // Try to complete without starting
    let result = backend.complete_run(&run_id, RunStatus::Success);
    assert!(matches!(result, Err(StorageError::InvalidState(_))));
}

#[test]
fn test_list_runs() {
    let mut backend = SqliteBackend::open_in_memory().expect("operation should succeed");
    let exp_id = backend.create_experiment("test-exp", None).expect("operation should succeed");

    backend.create_run(&exp_id).expect("operation should succeed");
    backend.create_run(&exp_id).expect("operation should succeed");

    let runs = backend.list_runs(&exp_id).expect("operation should succeed");
    assert_eq!(runs.len(), 2);
}
