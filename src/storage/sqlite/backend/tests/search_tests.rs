//! Parameter search and filter tests.

use crate::storage::sqlite::backend::SqliteBackend;
use crate::storage::sqlite::types::{FilterOp, ParamFilter, ParameterValue};
use crate::storage::ExperimentStorage;

#[test]
fn test_search_runs_by_params_eq() {
    let mut backend = SqliteBackend::open_in_memory().expect("operation should succeed");
    let exp_id = backend.create_experiment("test-exp", None).expect("operation should succeed");

    let run1 = backend.create_run(&exp_id).expect("operation should succeed");
    backend.log_param(&run1, "lr", ParameterValue::Float(0.001)).expect("operation should succeed");

    let run2 = backend.create_run(&exp_id).expect("operation should succeed");
    backend.log_param(&run2, "lr", ParameterValue::Float(0.01)).expect("operation should succeed");

    let filters = vec![ParamFilter {
        key: "lr".to_string(),
        op: FilterOp::Eq,
        value: ParameterValue::Float(0.001),
    }];

    let results = backend.search_runs_by_params(&filters).expect("operation should succeed");
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].id, run1);
}

#[test]
fn test_search_runs_by_params_gt() {
    let mut backend = SqliteBackend::open_in_memory().expect("operation should succeed");
    let exp_id = backend.create_experiment("test-exp", None).expect("operation should succeed");

    let run1 = backend.create_run(&exp_id).expect("operation should succeed");
    backend.log_param(&run1, "lr", ParameterValue::Float(0.001)).expect("operation should succeed");

    let run2 = backend.create_run(&exp_id).expect("operation should succeed");
    backend.log_param(&run2, "lr", ParameterValue::Float(0.01)).expect("operation should succeed");

    let filters = vec![ParamFilter {
        key: "lr".to_string(),
        op: FilterOp::Gt,
        value: ParameterValue::Float(0.005),
    }];

    let results = backend.search_runs_by_params(&filters).expect("operation should succeed");
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].id, run2);
}

#[test]
fn test_search_runs_by_params_contains() {
    let mut backend = SqliteBackend::open_in_memory().expect("operation should succeed");
    let exp_id = backend.create_experiment("test-exp", None).expect("operation should succeed");

    let run1 = backend.create_run(&exp_id).expect("operation should succeed");
    backend
        .log_param(&run1, "model", ParameterValue::String("llama-7b".to_string()))
        .expect("operation should succeed");

    let run2 = backend.create_run(&exp_id).expect("operation should succeed");
    backend
        .log_param(&run2, "model", ParameterValue::String("gpt-3.5".to_string()))
        .expect("operation should succeed");

    let filters = vec![ParamFilter {
        key: "model".to_string(),
        op: FilterOp::Contains,
        value: ParameterValue::String("llama".to_string()),
    }];

    let results = backend.search_runs_by_params(&filters).expect("operation should succeed");
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].id, run1);
}

#[test]
fn test_search_runs_no_filters() {
    let mut backend = SqliteBackend::open_in_memory().expect("operation should succeed");
    let exp_id = backend.create_experiment("test-exp", None).expect("operation should succeed");

    backend.create_run(&exp_id).expect("operation should succeed");
    backend.create_run(&exp_id).expect("operation should succeed");

    let results = backend.search_runs_by_params(&[]).expect("operation should succeed");
    assert_eq!(results.len(), 2);
}
