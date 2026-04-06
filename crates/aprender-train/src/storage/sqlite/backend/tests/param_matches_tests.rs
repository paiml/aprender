//! Tests for cross-type and unsupported parameter filter operations.
//!
//! These tests cover the untested match arms in param_matches:
//! - Cross-type comparisons (e.g., Float vs Int, String vs Bool, etc.)
//! - Unsupported operations on same-type pairs (e.g., Contains/StartsWith on Float/Int/Bool)
//! - List and Dict same-type combinations

use crate::storage::sqlite::backend::SqliteBackend;
use crate::storage::sqlite::types::{FilterOp, ParamFilter, ParameterValue};
use crate::storage::ExperimentStorage;
use std::collections::HashMap;

// -------------------------------------------------------------------------
// Cross-type comparisons (all should return false/no matches)
// -------------------------------------------------------------------------

#[test]
fn test_param_filter_cross_type_float_vs_int_eq() {
    let mut backend = SqliteBackend::open_in_memory().expect("operation should succeed");
    let exp_id = backend.create_experiment("test", None).expect("operation should succeed");

    let run = backend.create_run(&exp_id).expect("operation should succeed");
    backend
        .log_param(&run, "value", ParameterValue::Float(42.0))
        .expect("operation should succeed");

    let filters = vec![ParamFilter {
        key: "value".to_string(),
        op: FilterOp::Eq,
        value: ParameterValue::Int(42),
    }];

    let results = backend.search_runs_by_params(&filters).expect("operation should succeed");
    assert_eq!(results.len(), 0, "Float 42.0 should not match Int 42 with Eq");
}

#[test]
fn test_param_filter_cross_type_int_vs_float_eq() {
    let mut backend = SqliteBackend::open_in_memory().expect("operation should succeed");
    let exp_id = backend.create_experiment("test", None).expect("operation should succeed");

    let run = backend.create_run(&exp_id).expect("operation should succeed");
    backend.log_param(&run, "value", ParameterValue::Int(42)).expect("operation should succeed");

    let filters = vec![ParamFilter {
        key: "value".to_string(),
        op: FilterOp::Eq,
        value: ParameterValue::Float(42.0),
    }];

    let results = backend.search_runs_by_params(&filters).expect("operation should succeed");
    assert_eq!(results.len(), 0, "Int 42 should not match Float 42.0 with Eq");
}

#[test]
fn test_param_filter_cross_type_float_vs_string() {
    let mut backend = SqliteBackend::open_in_memory().expect("operation should succeed");
    let exp_id = backend.create_experiment("test", None).expect("operation should succeed");

    let run = backend.create_run(&exp_id).expect("operation should succeed");
    backend
        .log_param(&run, "value", ParameterValue::Float(3.14))
        .expect("operation should succeed");

    let filters = vec![ParamFilter {
        key: "value".to_string(),
        op: FilterOp::Eq,
        value: ParameterValue::String("3.14".to_string()),
    }];

    let results = backend.search_runs_by_params(&filters).expect("operation should succeed");
    assert_eq!(results.len(), 0, "Float 3.14 should not match String '3.14' with Eq");
}

#[test]
fn test_param_filter_cross_type_float_vs_bool() {
    let mut backend = SqliteBackend::open_in_memory().expect("operation should succeed");
    let exp_id = backend.create_experiment("test", None).expect("operation should succeed");

    let run = backend.create_run(&exp_id).expect("operation should succeed");
    backend.log_param(&run, "value", ParameterValue::Float(1.0)).expect("operation should succeed");

    let filters = vec![ParamFilter {
        key: "value".to_string(),
        op: FilterOp::Eq,
        value: ParameterValue::Bool(true),
    }];

    let results = backend.search_runs_by_params(&filters).expect("operation should succeed");
    assert_eq!(results.len(), 0, "Float 1.0 should not match Bool true with Eq");
}

#[test]
fn test_param_filter_cross_type_int_vs_string() {
    let mut backend = SqliteBackend::open_in_memory().expect("operation should succeed");
    let exp_id = backend.create_experiment("test", None).expect("operation should succeed");

    let run = backend.create_run(&exp_id).expect("operation should succeed");
    backend.log_param(&run, "value", ParameterValue::Int(42)).expect("operation should succeed");

    let filters = vec![ParamFilter {
        key: "value".to_string(),
        op: FilterOp::Eq,
        value: ParameterValue::String("42".to_string()),
    }];

    let results = backend.search_runs_by_params(&filters).expect("operation should succeed");
    assert_eq!(results.len(), 0, "Int 42 should not match String '42' with Eq");
}

#[test]
fn test_param_filter_cross_type_int_vs_bool() {
    let mut backend = SqliteBackend::open_in_memory().expect("operation should succeed");
    let exp_id = backend.create_experiment("test", None).expect("operation should succeed");

    let run = backend.create_run(&exp_id).expect("operation should succeed");
    backend.log_param(&run, "value", ParameterValue::Int(1)).expect("operation should succeed");

    let filters = vec![ParamFilter {
        key: "value".to_string(),
        op: FilterOp::Eq,
        value: ParameterValue::Bool(true),
    }];

    let results = backend.search_runs_by_params(&filters).expect("operation should succeed");
    assert_eq!(results.len(), 0, "Int 1 should not match Bool true with Eq");
}

#[test]
fn test_param_filter_cross_type_string_vs_bool() {
    let mut backend = SqliteBackend::open_in_memory().expect("operation should succeed");
    let exp_id = backend.create_experiment("test", None).expect("operation should succeed");

    let run = backend.create_run(&exp_id).expect("operation should succeed");
    backend
        .log_param(&run, "value", ParameterValue::String("true".to_string()))
        .expect("operation should succeed");

    let filters = vec![ParamFilter {
        key: "value".to_string(),
        op: FilterOp::Eq,
        value: ParameterValue::Bool(true),
    }];

    let results = backend.search_runs_by_params(&filters).expect("operation should succeed");
    assert_eq!(results.len(), 0, "String 'true' should not match Bool true with Eq");
}

#[test]
fn test_param_filter_cross_type_float_vs_list() {
    let mut backend = SqliteBackend::open_in_memory().expect("operation should succeed");
    let exp_id = backend.create_experiment("test", None).expect("operation should succeed");

    let run = backend.create_run(&exp_id).expect("operation should succeed");
    backend
        .log_param(&run, "value", ParameterValue::Float(3.14))
        .expect("operation should succeed");

    let filters = vec![ParamFilter {
        key: "value".to_string(),
        op: FilterOp::Eq,
        value: ParameterValue::List(vec![ParameterValue::Float(3.14)]),
    }];

    let results = backend.search_runs_by_params(&filters).expect("operation should succeed");
    assert_eq!(results.len(), 0, "Float 3.14 should not match List with Eq");
}

#[test]
fn test_param_filter_cross_type_int_vs_list() {
    let mut backend = SqliteBackend::open_in_memory().expect("operation should succeed");
    let exp_id = backend.create_experiment("test", None).expect("operation should succeed");

    let run = backend.create_run(&exp_id).expect("operation should succeed");
    backend.log_param(&run, "value", ParameterValue::Int(42)).expect("operation should succeed");

    let filters = vec![ParamFilter {
        key: "value".to_string(),
        op: FilterOp::Eq,
        value: ParameterValue::List(vec![ParameterValue::Int(42)]),
    }];

    let results = backend.search_runs_by_params(&filters).expect("operation should succeed");
    assert_eq!(results.len(), 0, "Int 42 should not match List with Eq");
}

#[test]
fn test_param_filter_cross_type_string_vs_list() {
    let mut backend = SqliteBackend::open_in_memory().expect("operation should succeed");
    let exp_id = backend.create_experiment("test", None).expect("operation should succeed");

    let run = backend.create_run(&exp_id).expect("operation should succeed");
    backend
        .log_param(&run, "value", ParameterValue::String("hello".to_string()))
        .expect("operation should succeed");

    let filters = vec![ParamFilter {
        key: "value".to_string(),
        op: FilterOp::Eq,
        value: ParameterValue::List(vec![ParameterValue::String("hello".to_string())]),
    }];

    let results = backend.search_runs_by_params(&filters).expect("operation should succeed");
    assert_eq!(results.len(), 0, "String 'hello' should not match List with Eq");
}

#[test]
fn test_param_filter_cross_type_bool_vs_list() {
    let mut backend = SqliteBackend::open_in_memory().expect("operation should succeed");
    let exp_id = backend.create_experiment("test", None).expect("operation should succeed");

    let run = backend.create_run(&exp_id).expect("operation should succeed");
    backend.log_param(&run, "value", ParameterValue::Bool(true)).expect("operation should succeed");

    let filters = vec![ParamFilter {
        key: "value".to_string(),
        op: FilterOp::Eq,
        value: ParameterValue::List(vec![ParameterValue::Bool(true)]),
    }];

    let results = backend.search_runs_by_params(&filters).expect("operation should succeed");
    assert_eq!(results.len(), 0, "Bool true should not match List with Eq");
}

#[test]
fn test_param_filter_cross_type_float_vs_dict() {
    let mut backend = SqliteBackend::open_in_memory().expect("operation should succeed");
    let exp_id = backend.create_experiment("test", None).expect("operation should succeed");

    let run = backend.create_run(&exp_id).expect("operation should succeed");
    backend
        .log_param(&run, "value", ParameterValue::Float(3.14))
        .expect("operation should succeed");

    let mut dict = HashMap::new();
    dict.insert("pi".to_string(), ParameterValue::Float(3.14));

    let filters = vec![ParamFilter {
        key: "value".to_string(),
        op: FilterOp::Eq,
        value: ParameterValue::Dict(dict),
    }];

    let results = backend.search_runs_by_params(&filters).expect("operation should succeed");
    assert_eq!(results.len(), 0, "Float 3.14 should not match Dict with Eq");
}

#[test]
fn test_param_filter_cross_type_ne_float_vs_int() {
    let mut backend = SqliteBackend::open_in_memory().expect("operation should succeed");
    let exp_id = backend.create_experiment("test", None).expect("operation should succeed");

    let run = backend.create_run(&exp_id).expect("operation should succeed");
    backend
        .log_param(&run, "value", ParameterValue::Float(42.0))
        .expect("operation should succeed");

    let filters = vec![ParamFilter {
        key: "value".to_string(),
        op: FilterOp::Ne,
        value: ParameterValue::Int(42),
    }];

    let results = backend.search_runs_by_params(&filters).expect("operation should succeed");
    assert_eq!(results.len(), 0, "Float 42.0 should not match Int 42 with Ne");
}

#[test]
fn test_param_filter_cross_type_gt_float_vs_int() {
    let mut backend = SqliteBackend::open_in_memory().expect("operation should succeed");
    let exp_id = backend.create_experiment("test", None).expect("operation should succeed");

    let run = backend.create_run(&exp_id).expect("operation should succeed");
    backend
        .log_param(&run, "value", ParameterValue::Float(100.0))
        .expect("operation should succeed");

    let filters = vec![ParamFilter {
        key: "value".to_string(),
        op: FilterOp::Gt,
        value: ParameterValue::Int(50),
    }];

    let results = backend.search_runs_by_params(&filters).expect("operation should succeed");
    assert_eq!(results.len(), 0, "Float 100.0 should not match Int 50 with Gt");
}

#[test]
fn test_param_filter_cross_type_lt_int_vs_float() {
    let mut backend = SqliteBackend::open_in_memory().expect("operation should succeed");
    let exp_id = backend.create_experiment("test", None).expect("operation should succeed");

    let run = backend.create_run(&exp_id).expect("operation should succeed");
    backend.log_param(&run, "value", ParameterValue::Int(10)).expect("operation should succeed");

    let filters = vec![ParamFilter {
        key: "value".to_string(),
        op: FilterOp::Lt,
        value: ParameterValue::Float(50.0),
    }];

    let results = backend.search_runs_by_params(&filters).expect("operation should succeed");
    assert_eq!(results.len(), 0, "Int 10 should not match Float 50.0 with Lt");
}

#[test]
fn test_param_filter_cross_type_gte_string_vs_int() {
    let mut backend = SqliteBackend::open_in_memory().expect("operation should succeed");
    let exp_id = backend.create_experiment("test", None).expect("operation should succeed");

    let run = backend.create_run(&exp_id).expect("operation should succeed");
    backend
        .log_param(&run, "value", ParameterValue::String("hello".to_string()))
        .expect("operation should succeed");

    let filters = vec![ParamFilter {
        key: "value".to_string(),
        op: FilterOp::Gte,
        value: ParameterValue::Int(5),
    }];

    let results = backend.search_runs_by_params(&filters).expect("operation should succeed");
    assert_eq!(results.len(), 0, "String 'hello' should not match Int 5 with Gte");
}

#[test]
fn test_param_filter_cross_type_lte_bool_vs_float() {
    let mut backend = SqliteBackend::open_in_memory().expect("operation should succeed");
    let exp_id = backend.create_experiment("test", None).expect("operation should succeed");

    let run = backend.create_run(&exp_id).expect("operation should succeed");
    backend.log_param(&run, "value", ParameterValue::Bool(true)).expect("operation should succeed");

    let filters = vec![ParamFilter {
        key: "value".to_string(),
        op: FilterOp::Lte,
        value: ParameterValue::Float(1.0),
    }];

    let results = backend.search_runs_by_params(&filters).expect("operation should succeed");
    assert_eq!(results.len(), 0, "Bool true should not match Float 1.0 with Lte");
}

// -------------------------------------------------------------------------
// Unsupported operations on same types (all should return false/no matches)
// -------------------------------------------------------------------------

#[test]
fn test_param_filter_unsupported_contains_on_float() {
    let mut backend = SqliteBackend::open_in_memory().expect("operation should succeed");
    let exp_id = backend.create_experiment("test", None).expect("operation should succeed");

    let run = backend.create_run(&exp_id).expect("operation should succeed");
    backend
        .log_param(&run, "value", ParameterValue::Float(3.14159))
        .expect("operation should succeed");

    let filters = vec![ParamFilter {
        key: "value".to_string(),
        op: FilterOp::Contains,
        value: ParameterValue::Float(3.14),
    }];

    let results = backend.search_runs_by_params(&filters).expect("operation should succeed");
    assert_eq!(results.len(), 0, "Contains operation should not work on Float type");
}

#[test]
fn test_param_filter_unsupported_startswith_on_float() {
    let mut backend = SqliteBackend::open_in_memory().expect("operation should succeed");
    let exp_id = backend.create_experiment("test", None).expect("operation should succeed");

    let run = backend.create_run(&exp_id).expect("operation should succeed");
    backend
        .log_param(&run, "value", ParameterValue::Float(3.14159))
        .expect("operation should succeed");

    let filters = vec![ParamFilter {
        key: "value".to_string(),
        op: FilterOp::StartsWith,
        value: ParameterValue::Float(3.0),
    }];

    let results = backend.search_runs_by_params(&filters).expect("operation should succeed");
    assert_eq!(results.len(), 0, "StartsWith operation should not work on Float type");
}

#[test]
fn test_param_filter_unsupported_contains_on_int() {
    let mut backend = SqliteBackend::open_in_memory().expect("operation should succeed");
    let exp_id = backend.create_experiment("test", None).expect("operation should succeed");

    let run = backend.create_run(&exp_id).expect("operation should succeed");
    backend.log_param(&run, "value", ParameterValue::Int(12345)).expect("operation should succeed");

    let filters = vec![ParamFilter {
        key: "value".to_string(),
        op: FilterOp::Contains,
        value: ParameterValue::Int(123),
    }];

    let results = backend.search_runs_by_params(&filters).expect("operation should succeed");
    assert_eq!(results.len(), 0, "Contains operation should not work on Int type");
}

#[test]
fn test_param_filter_unsupported_startswith_on_int() {
    let mut backend = SqliteBackend::open_in_memory().expect("operation should succeed");
    let exp_id = backend.create_experiment("test", None).expect("operation should succeed");

    let run = backend.create_run(&exp_id).expect("operation should succeed");
    backend.log_param(&run, "value", ParameterValue::Int(12345)).expect("operation should succeed");

    let filters = vec![ParamFilter {
        key: "value".to_string(),
        op: FilterOp::StartsWith,
        value: ParameterValue::Int(1),
    }];

    let results = backend.search_runs_by_params(&filters).expect("operation should succeed");
    assert_eq!(results.len(), 0, "StartsWith operation should not work on Int type");
}

#[test]
fn test_param_filter_unsupported_gt_on_string() {
    let mut backend = SqliteBackend::open_in_memory().expect("operation should succeed");
    let exp_id = backend.create_experiment("test", None).expect("operation should succeed");

    let run = backend.create_run(&exp_id).expect("operation should succeed");
    backend
        .log_param(&run, "value", ParameterValue::String("hello".to_string()))
        .expect("operation should succeed");

    let filters = vec![ParamFilter {
        key: "value".to_string(),
        op: FilterOp::Gt,
        value: ParameterValue::String("abc".to_string()),
    }];

    let results = backend.search_runs_by_params(&filters).expect("operation should succeed");
    assert_eq!(results.len(), 0, "Gt operation should not work on String type");
}

#[test]
fn test_param_filter_unsupported_lt_on_string() {
    let mut backend = SqliteBackend::open_in_memory().expect("operation should succeed");
    let exp_id = backend.create_experiment("test", None).expect("operation should succeed");

    let run = backend.create_run(&exp_id).expect("operation should succeed");
    backend
        .log_param(&run, "value", ParameterValue::String("hello".to_string()))
        .expect("operation should succeed");

    let filters = vec![ParamFilter {
        key: "value".to_string(),
        op: FilterOp::Lt,
        value: ParameterValue::String("xyz".to_string()),
    }];

    let results = backend.search_runs_by_params(&filters).expect("operation should succeed");
    assert_eq!(results.len(), 0, "Lt operation should not work on String type");
}

#[test]
fn test_param_filter_unsupported_gte_on_string() {
    let mut backend = SqliteBackend::open_in_memory().expect("operation should succeed");
    let exp_id = backend.create_experiment("test", None).expect("operation should succeed");

    let run = backend.create_run(&exp_id).expect("operation should succeed");
    backend
        .log_param(&run, "value", ParameterValue::String("hello".to_string()))
        .expect("operation should succeed");

    let filters = vec![ParamFilter {
        key: "value".to_string(),
        op: FilterOp::Gte,
        value: ParameterValue::String("abc".to_string()),
    }];

    let results = backend.search_runs_by_params(&filters).expect("operation should succeed");
    assert_eq!(results.len(), 0, "Gte operation should not work on String type");
}

#[test]
fn test_param_filter_unsupported_lte_on_string() {
    let mut backend = SqliteBackend::open_in_memory().expect("operation should succeed");
    let exp_id = backend.create_experiment("test", None).expect("operation should succeed");

    let run = backend.create_run(&exp_id).expect("operation should succeed");
    backend
        .log_param(&run, "value", ParameterValue::String("hello".to_string()))
        .expect("operation should succeed");

    let filters = vec![ParamFilter {
        key: "value".to_string(),
        op: FilterOp::Lte,
        value: ParameterValue::String("xyz".to_string()),
    }];

    let results = backend.search_runs_by_params(&filters).expect("operation should succeed");
    assert_eq!(results.len(), 0, "Lte operation should not work on String type");
}

#[test]
fn test_param_filter_unsupported_gt_on_bool() {
    let mut backend = SqliteBackend::open_in_memory().expect("operation should succeed");
    let exp_id = backend.create_experiment("test", None).expect("operation should succeed");

    let run = backend.create_run(&exp_id).expect("operation should succeed");
    backend.log_param(&run, "value", ParameterValue::Bool(true)).expect("operation should succeed");

    let filters = vec![ParamFilter {
        key: "value".to_string(),
        op: FilterOp::Gt,
        value: ParameterValue::Bool(false),
    }];

    let results = backend.search_runs_by_params(&filters).expect("operation should succeed");
    assert_eq!(results.len(), 0, "Gt operation should not work on Bool type");
}

#[test]
fn test_param_filter_unsupported_lt_on_bool() {
    let mut backend = SqliteBackend::open_in_memory().expect("operation should succeed");
    let exp_id = backend.create_experiment("test", None).expect("operation should succeed");

    let run = backend.create_run(&exp_id).expect("operation should succeed");
    backend.log_param(&run, "value", ParameterValue::Bool(true)).expect("operation should succeed");

    let filters = vec![ParamFilter {
        key: "value".to_string(),
        op: FilterOp::Lt,
        value: ParameterValue::Bool(false),
    }];

    let results = backend.search_runs_by_params(&filters).expect("operation should succeed");
    assert_eq!(results.len(), 0, "Lt operation should not work on Bool type");
}

#[test]
fn test_param_filter_unsupported_gte_on_bool() {
    let mut backend = SqliteBackend::open_in_memory().expect("operation should succeed");
    let exp_id = backend.create_experiment("test", None).expect("operation should succeed");

    let run = backend.create_run(&exp_id).expect("operation should succeed");
    backend.log_param(&run, "value", ParameterValue::Bool(true)).expect("operation should succeed");

    let filters = vec![ParamFilter {
        key: "value".to_string(),
        op: FilterOp::Gte,
        value: ParameterValue::Bool(false),
    }];

    let results = backend.search_runs_by_params(&filters).expect("operation should succeed");
    assert_eq!(results.len(), 0, "Gte operation should not work on Bool type");
}

#[test]
fn test_param_filter_unsupported_lte_on_bool() {
    let mut backend = SqliteBackend::open_in_memory().expect("operation should succeed");
    let exp_id = backend.create_experiment("test", None).expect("operation should succeed");

    let run = backend.create_run(&exp_id).expect("operation should succeed");
    backend.log_param(&run, "value", ParameterValue::Bool(true)).expect("operation should succeed");

    let filters = vec![ParamFilter {
        key: "value".to_string(),
        op: FilterOp::Lte,
        value: ParameterValue::Bool(false),
    }];

    let results = backend.search_runs_by_params(&filters).expect("operation should succeed");
    assert_eq!(results.len(), 0, "Lte operation should not work on Bool type");
}

#[test]
fn test_param_filter_unsupported_contains_on_bool() {
    let mut backend = SqliteBackend::open_in_memory().expect("operation should succeed");
    let exp_id = backend.create_experiment("test", None).expect("operation should succeed");

    let run = backend.create_run(&exp_id).expect("operation should succeed");
    backend.log_param(&run, "value", ParameterValue::Bool(true)).expect("operation should succeed");

    let filters = vec![ParamFilter {
        key: "value".to_string(),
        op: FilterOp::Contains,
        value: ParameterValue::Bool(false),
    }];

    let results = backend.search_runs_by_params(&filters).expect("operation should succeed");
    assert_eq!(results.len(), 0, "Contains operation should not work on Bool type");
}

#[test]
fn test_param_filter_unsupported_startswith_on_bool() {
    let mut backend = SqliteBackend::open_in_memory().expect("operation should succeed");
    let exp_id = backend.create_experiment("test", None).expect("operation should succeed");

    let run = backend.create_run(&exp_id).expect("operation should succeed");
    backend.log_param(&run, "value", ParameterValue::Bool(true)).expect("operation should succeed");

    let filters = vec![ParamFilter {
        key: "value".to_string(),
        op: FilterOp::StartsWith,
        value: ParameterValue::Bool(false),
    }];

    let results = backend.search_runs_by_params(&filters).expect("operation should succeed");
    assert_eq!(results.len(), 0, "StartsWith operation should not work on Bool type");
}

// -------------------------------------------------------------------------
// List and Dict same-type combinations (all operations should return false)
// -------------------------------------------------------------------------

#[test]
fn test_param_filter_list_eq_list() {
    let mut backend = SqliteBackend::open_in_memory().expect("operation should succeed");
    let exp_id = backend.create_experiment("test", None).expect("operation should succeed");

    let run = backend.create_run(&exp_id).expect("operation should succeed");
    backend
        .log_param(
            &run,
            "value",
            ParameterValue::List(vec![ParameterValue::Int(1), ParameterValue::Int(2)]),
        )
        .expect("operation should succeed");

    let filters = vec![ParamFilter {
        key: "value".to_string(),
        op: FilterOp::Eq,
        value: ParameterValue::List(vec![ParameterValue::Int(1), ParameterValue::Int(2)]),
    }];

    let results = backend.search_runs_by_params(&filters).expect("operation should succeed");
    assert_eq!(results.len(), 0, "List-to-list Eq should not be supported");
}

#[test]
fn test_param_filter_list_ne_list() {
    let mut backend = SqliteBackend::open_in_memory().expect("operation should succeed");
    let exp_id = backend.create_experiment("test", None).expect("operation should succeed");

    let run = backend.create_run(&exp_id).expect("operation should succeed");
    backend
        .log_param(&run, "value", ParameterValue::List(vec![ParameterValue::Int(1)]))
        .expect("operation should succeed");

    let filters = vec![ParamFilter {
        key: "value".to_string(),
        op: FilterOp::Ne,
        value: ParameterValue::List(vec![ParameterValue::Int(2)]),
    }];

    let results = backend.search_runs_by_params(&filters).expect("operation should succeed");
    assert_eq!(results.len(), 0, "List-to-list Ne should not be supported");
}

#[test]
fn test_param_filter_list_gt_list() {
    let mut backend = SqliteBackend::open_in_memory().expect("operation should succeed");
    let exp_id = backend.create_experiment("test", None).expect("operation should succeed");

    let run = backend.create_run(&exp_id).expect("operation should succeed");
    backend
        .log_param(&run, "value", ParameterValue::List(vec![ParameterValue::Int(1)]))
        .expect("operation should succeed");

    let filters = vec![ParamFilter {
        key: "value".to_string(),
        op: FilterOp::Gt,
        value: ParameterValue::List(vec![ParameterValue::Int(0)]),
    }];

    let results = backend.search_runs_by_params(&filters).expect("operation should succeed");
    assert_eq!(results.len(), 0, "List-to-list Gt should not be supported");
}

#[test]
fn test_param_filter_dict_eq_dict() {
    let mut backend = SqliteBackend::open_in_memory().expect("operation should succeed");
    let exp_id = backend.create_experiment("test", None).expect("operation should succeed");

    let run = backend.create_run(&exp_id).expect("operation should succeed");
    let mut dict = HashMap::new();
    dict.insert("key".to_string(), ParameterValue::Int(1));
    backend
        .log_param(&run, "value", ParameterValue::Dict(dict.clone()))
        .expect("operation should succeed");

    let filters = vec![ParamFilter {
        key: "value".to_string(),
        op: FilterOp::Eq,
        value: ParameterValue::Dict(dict),
    }];

    let results = backend.search_runs_by_params(&filters).expect("operation should succeed");
    assert_eq!(results.len(), 0, "Dict-to-dict Eq should not be supported");
}

#[test]
fn test_param_filter_dict_ne_dict() {
    let mut backend = SqliteBackend::open_in_memory().expect("operation should succeed");
    let exp_id = backend.create_experiment("test", None).expect("operation should succeed");

    let run = backend.create_run(&exp_id).expect("operation should succeed");
    let mut dict1 = HashMap::new();
    dict1.insert("key".to_string(), ParameterValue::Int(1));
    backend
        .log_param(&run, "value", ParameterValue::Dict(dict1))
        .expect("operation should succeed");

    let mut dict2 = HashMap::new();
    dict2.insert("key".to_string(), ParameterValue::Int(2));

    let filters = vec![ParamFilter {
        key: "value".to_string(),
        op: FilterOp::Ne,
        value: ParameterValue::Dict(dict2),
    }];

    let results = backend.search_runs_by_params(&filters).expect("operation should succeed");
    assert_eq!(results.len(), 0, "Dict-to-dict Ne should not be supported");
}

#[test]
fn test_param_filter_dict_contains_dict() {
    let mut backend = SqliteBackend::open_in_memory().expect("operation should succeed");
    let exp_id = backend.create_experiment("test", None).expect("operation should succeed");

    let run = backend.create_run(&exp_id).expect("operation should succeed");
    let mut dict = HashMap::new();
    dict.insert("key".to_string(), ParameterValue::Int(1));
    backend
        .log_param(&run, "value", ParameterValue::Dict(dict.clone()))
        .expect("operation should succeed");

    let filters = vec![ParamFilter {
        key: "value".to_string(),
        op: FilterOp::Contains,
        value: ParameterValue::Dict(dict),
    }];

    let results = backend.search_runs_by_params(&filters).expect("operation should succeed");
    assert_eq!(results.len(), 0, "Dict-to-dict Contains should not be supported");
}

#[test]
fn test_param_filter_dict_startswith_dict() {
    let mut backend = SqliteBackend::open_in_memory().expect("operation should succeed");
    let exp_id = backend.create_experiment("test", None).expect("operation should succeed");

    let run = backend.create_run(&exp_id).expect("operation should succeed");
    let mut dict = HashMap::new();
    dict.insert("key".to_string(), ParameterValue::Int(1));
    backend
        .log_param(&run, "value", ParameterValue::Dict(dict.clone()))
        .expect("operation should succeed");

    let filters = vec![ParamFilter {
        key: "value".to_string(),
        op: FilterOp::StartsWith,
        value: ParameterValue::Dict(dict),
    }];

    let results = backend.search_runs_by_params(&filters).expect("operation should succeed");
    assert_eq!(results.len(), 0, "Dict-to-dict StartsWith should not be supported");
}

// -------------------------------------------------------------------------
// List and Dict cross-type combinations (all should return false)
// -------------------------------------------------------------------------

#[test]
fn test_param_filter_list_vs_dict() {
    let mut backend = SqliteBackend::open_in_memory().expect("operation should succeed");
    let exp_id = backend.create_experiment("test", None).expect("operation should succeed");

    let run = backend.create_run(&exp_id).expect("operation should succeed");
    backend
        .log_param(&run, "value", ParameterValue::List(vec![ParameterValue::Int(1)]))
        .expect("operation should succeed");

    let mut dict = HashMap::new();
    dict.insert("key".to_string(), ParameterValue::Int(1));

    let filters = vec![ParamFilter {
        key: "value".to_string(),
        op: FilterOp::Eq,
        value: ParameterValue::Dict(dict),
    }];

    let results = backend.search_runs_by_params(&filters).expect("operation should succeed");
    assert_eq!(results.len(), 0, "List should not match Dict with Eq");
}

#[test]
fn test_param_filter_dict_vs_list() {
    let mut backend = SqliteBackend::open_in_memory().expect("operation should succeed");
    let exp_id = backend.create_experiment("test", None).expect("operation should succeed");

    let run = backend.create_run(&exp_id).expect("operation should succeed");
    let mut dict = HashMap::new();
    dict.insert("key".to_string(), ParameterValue::Int(1));
    backend.log_param(&run, "value", ParameterValue::Dict(dict)).expect("operation should succeed");

    let filters = vec![ParamFilter {
        key: "value".to_string(),
        op: FilterOp::Eq,
        value: ParameterValue::List(vec![ParameterValue::Int(1)]),
    }];

    let results = backend.search_runs_by_params(&filters).expect("operation should succeed");
    assert_eq!(results.len(), 0, "Dict should not match List with Eq");
}

// -------------------------------------------------------------------------
// Syntactic match coverage for FilterOp variants (satisfies variant scanner)
// -------------------------------------------------------------------------

#[test]
fn test_filter_op_variant_coverage() {
    let mut backend = SqliteBackend::open_in_memory().expect("operation should succeed");
    let exp_id = backend.create_experiment("test", None).expect("operation should succeed");

    let run = backend.create_run(&exp_id).expect("operation should succeed");
    backend.log_param(&run, "x", ParameterValue::Float(5.0)).expect("operation should succeed");

    // Exercise every FilterOp variant through param_matches
    let ops = [
        FilterOp::Eq,
        FilterOp::Ne,
        FilterOp::Gt,
        FilterOp::Lt,
        FilterOp::Gte,
        FilterOp::Lte,
        FilterOp::Contains,
        FilterOp::StartsWith,
    ];

    for op in &ops {
        let expected = match op {
            FilterOp::Eq => true,          // 5.0 == 5.0
            FilterOp::Ne => false,         // 5.0 != 5.0 is false
            FilterOp::Gt => false,         // 5.0 > 5.0 is false
            FilterOp::Lt => false,         // 5.0 < 5.0 is false
            FilterOp::Gte => true,         // 5.0 >= 5.0
            FilterOp::Lte => true,         // 5.0 <= 5.0
            FilterOp::Contains => false,   // unsupported for Float
            FilterOp::StartsWith => false, // unsupported for Float
        };

        let filters = vec![ParamFilter {
            key: "x".to_string(),
            op: op.clone(),
            value: ParameterValue::Float(5.0),
        }];

        let results = backend.search_runs_by_params(&filters).expect("operation should succeed");
        let matched = !results.is_empty();
        assert_eq!(matched, expected, "FilterOp::{op:?} on Float(5.0) vs Float(5.0)");
    }
}
