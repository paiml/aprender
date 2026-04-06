//! Property-based tests.

use crate::storage::sqlite::backend::SqliteBackend;
use crate::storage::sqlite::types::ParameterValue;
use crate::storage::{ExperimentStorage, RunStatus};
use proptest::prelude::*;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    #[test]
    fn prop_experiment_name_preserved(name in "[a-zA-Z][a-zA-Z0-9_-]{0,50}") {
        let mut backend = SqliteBackend::open_in_memory().expect("operation should succeed");
        let exp_id = backend.create_experiment(&name, None).expect("operation should succeed");
        let exp = backend.get_experiment(&exp_id).expect("operation should succeed");
        prop_assert_eq!(exp.name, name);
    }

    #[test]
    fn prop_metric_values_preserved(values in prop::collection::vec(-1e10f64..1e10f64, 1..100)) {
        let mut backend = SqliteBackend::open_in_memory().expect("operation should succeed");
        let exp_id = backend.create_experiment("test", None).expect("operation should succeed");
        let run_id = backend.create_run(&exp_id).expect("operation should succeed");

        for (step, value) in values.iter().enumerate() {
            if !value.is_nan() && !value.is_infinite() {
                backend.log_metric(&run_id, "metric", step as u64, *value).expect("operation should succeed");
            }
        }

        let metrics = backend.get_metrics(&run_id, "metric").expect("operation should succeed");
        for (i, metric) in metrics.iter().enumerate() {
            let original = values[i];
            if !original.is_nan() && !original.is_infinite() {
                prop_assert!((metric.value - original).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn prop_artifact_sha256_deterministic(data in prop::collection::vec(any::<u8>(), 1..1000)) {
        let mut backend = SqliteBackend::open_in_memory().expect("operation should succeed");
        let exp_id = backend.create_experiment("test", None).expect("operation should succeed");
        let run_id = backend.create_run(&exp_id).expect("operation should succeed");

        let sha1 = backend.log_artifact(&run_id, "file1", &data).expect("operation should succeed");
        let sha2 = backend.log_artifact(&run_id, "file2", &data).expect("operation should succeed");

        prop_assert_eq!(sha1, sha2);
    }

    #[test]
    fn prop_parameter_int_roundtrip(value in any::<i64>()) {
        let param = ParameterValue::Int(value);
        let json = param.to_json();
        let parsed = ParameterValue::from_json(&json).expect("parsing should succeed");
        prop_assert_eq!(param, parsed);
    }

    #[test]
    fn prop_parameter_float_roundtrip(value in -1e15f64..1e15f64) {
        if !value.is_nan() && !value.is_infinite() {
            let param = ParameterValue::Float(value);
            let json = param.to_json();
            let parsed = ParameterValue::from_json(&json).expect("parsing should succeed");
            if let ParameterValue::Float(v) = parsed {
                // Use relative tolerance for large values
                let tol = if value.abs() > 1.0 {
                    value.abs() * 1e-14
                } else {
                    1e-14
                };
                prop_assert!((v - value).abs() < tol, "Expected {} == {} within tolerance", v, value);
            } else {
                prop_assert!(false, "Expected Float");
            }
        }
    }

    #[test]
    fn prop_run_status_transitions_valid(
        complete_success in any::<bool>()
    ) {
        let mut backend = SqliteBackend::open_in_memory().expect("operation should succeed");
        let exp_id = backend.create_experiment("test", None).expect("operation should succeed");
        let run_id = backend.create_run(&exp_id).expect("operation should succeed");

        // Pending -> Running
        prop_assert!(backend.start_run(&run_id).is_ok());
        prop_assert_eq!(backend.get_run_status(&run_id).expect("operation should succeed"), RunStatus::Running);

        // Running -> Success/Failed
        let final_status = if complete_success {
            RunStatus::Success
        } else {
            RunStatus::Failed
        };
        prop_assert!(backend.complete_run(&run_id, final_status).is_ok());
        prop_assert_eq!(backend.get_run_status(&run_id).expect("operation should succeed"), final_status);
    }
}
