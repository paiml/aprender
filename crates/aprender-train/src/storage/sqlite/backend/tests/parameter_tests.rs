//! Parameter logging tests (MLOPS-003).

use std::collections::HashMap;

use crate::storage::sqlite::backend::SqliteBackend;
use crate::storage::sqlite::types::ParameterValue;
use crate::storage::{ExperimentStorage, StorageError};

#[test]
fn test_log_param_string() {
    let mut backend = SqliteBackend::open_in_memory().expect("operation should succeed");
    let exp_id = backend.create_experiment("test-exp", None).expect("operation should succeed");
    let run_id = backend.create_run(&exp_id).expect("operation should succeed");

    backend
        .log_param(&run_id, "optimizer", ParameterValue::String("adam".to_string()))
        .expect("operation should succeed");

    let params = backend.get_params(&run_id).expect("operation should succeed");
    assert_eq!(params.get("optimizer"), Some(&ParameterValue::String("adam".to_string())));
}

#[test]
fn test_log_param_float() {
    let mut backend = SqliteBackend::open_in_memory().expect("operation should succeed");
    let exp_id = backend.create_experiment("test-exp", None).expect("operation should succeed");
    let run_id = backend.create_run(&exp_id).expect("operation should succeed");

    backend
        .log_param(&run_id, "lr", ParameterValue::Float(0.001))
        .expect("operation should succeed");

    let params = backend.get_params(&run_id).expect("operation should succeed");
    assert_eq!(params.get("lr"), Some(&ParameterValue::Float(0.001)));
}

#[test]
fn test_log_param_int() {
    let mut backend = SqliteBackend::open_in_memory().expect("operation should succeed");
    let exp_id = backend.create_experiment("test-exp", None).expect("operation should succeed");
    let run_id = backend.create_run(&exp_id).expect("operation should succeed");

    backend
        .log_param(&run_id, "epochs", ParameterValue::Int(100))
        .expect("operation should succeed");

    let params = backend.get_params(&run_id).expect("operation should succeed");
    assert_eq!(params.get("epochs"), Some(&ParameterValue::Int(100)));
}

#[test]
fn test_log_param_bool() {
    let mut backend = SqliteBackend::open_in_memory().expect("operation should succeed");
    let exp_id = backend.create_experiment("test-exp", None).expect("operation should succeed");
    let run_id = backend.create_run(&exp_id).expect("operation should succeed");

    backend
        .log_param(&run_id, "use_cuda", ParameterValue::Bool(true))
        .expect("operation should succeed");

    let params = backend.get_params(&run_id).expect("operation should succeed");
    assert_eq!(params.get("use_cuda"), Some(&ParameterValue::Bool(true)));
}

#[test]
fn test_log_params_batch() {
    let mut backend = SqliteBackend::open_in_memory().expect("operation should succeed");
    let exp_id = backend.create_experiment("test-exp", None).expect("operation should succeed");
    let run_id = backend.create_run(&exp_id).expect("operation should succeed");

    let mut params = HashMap::new();
    params.insert("lr".to_string(), ParameterValue::Float(0.001));
    params.insert("epochs".to_string(), ParameterValue::Int(100));
    params.insert("optimizer".to_string(), ParameterValue::String("adam".to_string()));

    backend.log_params(&run_id, params).expect("operation should succeed");

    let retrieved = backend.get_params(&run_id).expect("operation should succeed");
    assert_eq!(retrieved.len(), 3);
}

#[test]
fn test_log_param_run_not_found() {
    let backend = SqliteBackend::open_in_memory().expect("operation should succeed");
    let result = backend.log_param("nonexistent", "key", ParameterValue::Int(1));
    assert!(matches!(result, Err(StorageError::RunNotFound(_))));
}
