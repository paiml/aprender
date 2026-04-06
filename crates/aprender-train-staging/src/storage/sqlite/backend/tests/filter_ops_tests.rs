//! Filter operations tests for float, int, string, and bool parameter types.

use crate::storage::sqlite::backend::SqliteBackend;
use crate::storage::sqlite::types::{FilterOp, ParamFilter, ParameterValue};
use crate::storage::ExperimentStorage;

// -------------------------------------------------------------------------
// Float filter operations
// -------------------------------------------------------------------------

#[test]
fn test_filter_op_float_ne() {
    let mut backend = SqliteBackend::open_in_memory().expect("operation should succeed");
    let exp_id = backend.create_experiment("test", None).expect("operation should succeed");

    let run1 = backend.create_run(&exp_id).expect("operation should succeed");
    backend.log_param(&run1, "lr", ParameterValue::Float(0.001)).expect("operation should succeed");

    let run2 = backend.create_run(&exp_id).expect("operation should succeed");
    backend.log_param(&run2, "lr", ParameterValue::Float(0.01)).expect("operation should succeed");

    let filters = vec![ParamFilter {
        key: "lr".to_string(),
        op: FilterOp::Ne,
        value: ParameterValue::Float(0.001),
    }];

    let results = backend.search_runs_by_params(&filters).expect("operation should succeed");
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].id, run2);
}

#[test]
fn test_filter_op_float_lt() {
    let mut backend = SqliteBackend::open_in_memory().expect("operation should succeed");
    let exp_id = backend.create_experiment("test", None).expect("operation should succeed");

    let run1 = backend.create_run(&exp_id).expect("operation should succeed");
    backend.log_param(&run1, "lr", ParameterValue::Float(0.001)).expect("operation should succeed");

    let run2 = backend.create_run(&exp_id).expect("operation should succeed");
    backend.log_param(&run2, "lr", ParameterValue::Float(0.01)).expect("operation should succeed");

    let filters = vec![ParamFilter {
        key: "lr".to_string(),
        op: FilterOp::Lt,
        value: ParameterValue::Float(0.005),
    }];

    let results = backend.search_runs_by_params(&filters).expect("operation should succeed");
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].id, run1);
}

#[test]
fn test_filter_op_float_gte() {
    let mut backend = SqliteBackend::open_in_memory().expect("operation should succeed");
    let exp_id = backend.create_experiment("test", None).expect("operation should succeed");

    let run1 = backend.create_run(&exp_id).expect("operation should succeed");
    backend.log_param(&run1, "lr", ParameterValue::Float(0.01)).expect("operation should succeed");

    let filters = vec![ParamFilter {
        key: "lr".to_string(),
        op: FilterOp::Gte,
        value: ParameterValue::Float(0.01),
    }];

    let results = backend.search_runs_by_params(&filters).expect("operation should succeed");
    assert_eq!(results.len(), 1);
}

#[test]
fn test_filter_op_float_lte() {
    let mut backend = SqliteBackend::open_in_memory().expect("operation should succeed");
    let exp_id = backend.create_experiment("test", None).expect("operation should succeed");

    let run1 = backend.create_run(&exp_id).expect("operation should succeed");
    backend.log_param(&run1, "lr", ParameterValue::Float(0.01)).expect("operation should succeed");

    let filters = vec![ParamFilter {
        key: "lr".to_string(),
        op: FilterOp::Lte,
        value: ParameterValue::Float(0.01),
    }];

    let results = backend.search_runs_by_params(&filters).expect("operation should succeed");
    assert_eq!(results.len(), 1);
}

// -------------------------------------------------------------------------
// Int filter operations
// -------------------------------------------------------------------------

#[test]
fn test_filter_op_int_eq() {
    let mut backend = SqliteBackend::open_in_memory().expect("operation should succeed");
    let exp_id = backend.create_experiment("test", None).expect("operation should succeed");

    let run1 = backend.create_run(&exp_id).expect("operation should succeed");
    backend.log_param(&run1, "epochs", ParameterValue::Int(100)).expect("operation should succeed");

    let filters = vec![ParamFilter {
        key: "epochs".to_string(),
        op: FilterOp::Eq,
        value: ParameterValue::Int(100),
    }];

    let results = backend.search_runs_by_params(&filters).expect("operation should succeed");
    assert_eq!(results.len(), 1);
}

#[test]
fn test_filter_op_int_ne() {
    let mut backend = SqliteBackend::open_in_memory().expect("operation should succeed");
    let exp_id = backend.create_experiment("test", None).expect("operation should succeed");

    let run1 = backend.create_run(&exp_id).expect("operation should succeed");
    backend.log_param(&run1, "epochs", ParameterValue::Int(100)).expect("operation should succeed");

    let run2 = backend.create_run(&exp_id).expect("operation should succeed");
    backend.log_param(&run2, "epochs", ParameterValue::Int(200)).expect("operation should succeed");

    let filters = vec![ParamFilter {
        key: "epochs".to_string(),
        op: FilterOp::Ne,
        value: ParameterValue::Int(100),
    }];

    let results = backend.search_runs_by_params(&filters).expect("operation should succeed");
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].id, run2);
}

#[test]
fn test_filter_op_int_gt() {
    let mut backend = SqliteBackend::open_in_memory().expect("operation should succeed");
    let exp_id = backend.create_experiment("test", None).expect("operation should succeed");

    let run1 = backend.create_run(&exp_id).expect("operation should succeed");
    backend.log_param(&run1, "epochs", ParameterValue::Int(100)).expect("operation should succeed");

    let run2 = backend.create_run(&exp_id).expect("operation should succeed");
    backend.log_param(&run2, "epochs", ParameterValue::Int(200)).expect("operation should succeed");

    let filters = vec![ParamFilter {
        key: "epochs".to_string(),
        op: FilterOp::Gt,
        value: ParameterValue::Int(150),
    }];

    let results = backend.search_runs_by_params(&filters).expect("operation should succeed");
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].id, run2);
}

#[test]
fn test_filter_op_int_lt() {
    let mut backend = SqliteBackend::open_in_memory().expect("operation should succeed");
    let exp_id = backend.create_experiment("test", None).expect("operation should succeed");

    let run1 = backend.create_run(&exp_id).expect("operation should succeed");
    backend.log_param(&run1, "epochs", ParameterValue::Int(100)).expect("operation should succeed");

    let filters = vec![ParamFilter {
        key: "epochs".to_string(),
        op: FilterOp::Lt,
        value: ParameterValue::Int(150),
    }];

    let results = backend.search_runs_by_params(&filters).expect("operation should succeed");
    assert_eq!(results.len(), 1);
}

#[test]
fn test_filter_op_int_gte() {
    let mut backend = SqliteBackend::open_in_memory().expect("operation should succeed");
    let exp_id = backend.create_experiment("test", None).expect("operation should succeed");

    let run1 = backend.create_run(&exp_id).expect("operation should succeed");
    backend.log_param(&run1, "epochs", ParameterValue::Int(100)).expect("operation should succeed");

    let filters = vec![ParamFilter {
        key: "epochs".to_string(),
        op: FilterOp::Gte,
        value: ParameterValue::Int(100),
    }];

    let results = backend.search_runs_by_params(&filters).expect("operation should succeed");
    assert_eq!(results.len(), 1);
}

#[test]
fn test_filter_op_int_lte() {
    let mut backend = SqliteBackend::open_in_memory().expect("operation should succeed");
    let exp_id = backend.create_experiment("test", None).expect("operation should succeed");

    let run1 = backend.create_run(&exp_id).expect("operation should succeed");
    backend.log_param(&run1, "epochs", ParameterValue::Int(100)).expect("operation should succeed");

    let filters = vec![ParamFilter {
        key: "epochs".to_string(),
        op: FilterOp::Lte,
        value: ParameterValue::Int(100),
    }];

    let results = backend.search_runs_by_params(&filters).expect("operation should succeed");
    assert_eq!(results.len(), 1);
}

// -------------------------------------------------------------------------
// String filter operations
// -------------------------------------------------------------------------

#[test]
fn test_filter_op_string_eq() {
    let mut backend = SqliteBackend::open_in_memory().expect("operation should succeed");
    let exp_id = backend.create_experiment("test", None).expect("operation should succeed");

    let run1 = backend.create_run(&exp_id).expect("operation should succeed");
    backend
        .log_param(&run1, "model", ParameterValue::String("llama".to_string()))
        .expect("operation should succeed");

    let filters = vec![ParamFilter {
        key: "model".to_string(),
        op: FilterOp::Eq,
        value: ParameterValue::String("llama".to_string()),
    }];

    let results = backend.search_runs_by_params(&filters).expect("operation should succeed");
    assert_eq!(results.len(), 1);
}

#[test]
fn test_filter_op_string_ne() {
    let mut backend = SqliteBackend::open_in_memory().expect("operation should succeed");
    let exp_id = backend.create_experiment("test", None).expect("operation should succeed");

    let run1 = backend.create_run(&exp_id).expect("operation should succeed");
    backend
        .log_param(&run1, "model", ParameterValue::String("llama".to_string()))
        .expect("operation should succeed");

    let run2 = backend.create_run(&exp_id).expect("operation should succeed");
    backend
        .log_param(&run2, "model", ParameterValue::String("gpt".to_string()))
        .expect("operation should succeed");

    let filters = vec![ParamFilter {
        key: "model".to_string(),
        op: FilterOp::Ne,
        value: ParameterValue::String("llama".to_string()),
    }];

    let results = backend.search_runs_by_params(&filters).expect("operation should succeed");
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].id, run2);
}
