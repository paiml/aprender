//! Experiment CRUD tests.

use crate::storage::sqlite::backend::SqliteBackend;
use crate::storage::{ExperimentStorage, StorageError};

#[test]
fn test_create_experiment() {
    let mut backend = SqliteBackend::open_in_memory().expect("operation should succeed");
    let exp_id = backend.create_experiment("test-exp", None).expect("operation should succeed");
    assert!(!exp_id.is_empty());
}

#[test]
fn test_create_experiment_with_config() {
    let mut backend = SqliteBackend::open_in_memory().expect("operation should succeed");
    let config = serde_json::json!({"lr": 0.001, "epochs": 10});
    let exp_id = backend
        .create_experiment("test-exp", Some(config.clone()))
        .expect("config should be valid");

    let exp = backend.get_experiment(&exp_id).expect("operation should succeed");
    assert_eq!(exp.name, "test-exp");
    assert_eq!(exp.config, Some(config));
}

#[test]
fn test_get_experiment_not_found() {
    let backend = SqliteBackend::open_in_memory().expect("operation should succeed");
    let result = backend.get_experiment("nonexistent");
    assert!(matches!(result, Err(StorageError::ExperimentNotFound(_))));
}

#[test]
fn test_list_experiments() {
    let mut backend = SqliteBackend::open_in_memory().expect("operation should succeed");
    backend.create_experiment("exp-1", None).expect("operation should succeed");
    backend.create_experiment("exp-2", None).expect("operation should succeed");

    let experiments = backend.list_experiments().expect("operation should succeed");
    assert_eq!(experiments.len(), 2);
}
