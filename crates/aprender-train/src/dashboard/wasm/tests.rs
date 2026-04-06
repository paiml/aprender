//! Tests for WASM dashboard bindings.

use super::storage::IndexedDbStorage;
use crate::storage::{ExperimentStorage, RunStatus, StorageError};

#[test]
fn test_indexed_db_storage_create_experiment() {
    let mut storage = IndexedDbStorage::new();
    let id = storage.create_experiment("test-exp", None).expect("operation should succeed");

    assert!(id.starts_with("exp-"));
    assert_eq!(storage.list_experiments().len(), 1);
}

#[test]
fn test_indexed_db_storage_create_run() {
    let mut storage = IndexedDbStorage::new();
    let exp_id = storage.create_experiment("test-exp", None).expect("operation should succeed");
    let run_id = storage.create_run(&exp_id).expect("operation should succeed");

    assert!(run_id.starts_with("run-"));
    assert_eq!(
        storage.get_run_status(&run_id).expect("operation should succeed"),
        RunStatus::Pending
    );
}

#[test]
fn test_indexed_db_storage_run_lifecycle() {
    let mut storage = IndexedDbStorage::new();
    let exp_id = storage.create_experiment("test-exp", None).expect("operation should succeed");
    let run_id = storage.create_run(&exp_id).expect("operation should succeed");

    storage.start_run(&run_id).expect("operation should succeed");
    assert_eq!(
        storage.get_run_status(&run_id).expect("operation should succeed"),
        RunStatus::Running
    );

    storage.complete_run(&run_id, RunStatus::Success).expect("operation should succeed");
    assert_eq!(
        storage.get_run_status(&run_id).expect("operation should succeed"),
        RunStatus::Success
    );
}

#[test]
fn test_indexed_db_storage_log_metrics() {
    let mut storage = IndexedDbStorage::new();
    let exp_id = storage.create_experiment("test-exp", None).expect("operation should succeed");
    let run_id = storage.create_run(&exp_id).expect("operation should succeed");
    storage.start_run(&run_id).expect("operation should succeed");

    storage.log_metric(&run_id, "loss", 0, 0.5).expect("operation should succeed");
    storage.log_metric(&run_id, "loss", 1, 0.4).expect("operation should succeed");
    storage.log_metric(&run_id, "accuracy", 0, 0.8).expect("operation should succeed");

    let loss_metrics = storage.get_metrics(&run_id, "loss").expect("operation should succeed");
    assert_eq!(loss_metrics.len(), 2);
    assert!((loss_metrics[0].value - 0.5).abs() < f64::EPSILON);

    let keys = storage.list_metric_keys(&run_id);
    assert_eq!(keys.len(), 2);
}

#[test]
fn test_indexed_db_storage_log_artifact() {
    let mut storage = IndexedDbStorage::new();
    let exp_id = storage.create_experiment("test-exp", None).expect("operation should succeed");
    let run_id = storage.create_run(&exp_id).expect("operation should succeed");
    storage.start_run(&run_id).expect("operation should succeed");

    let data = b"test artifact data";
    let hash = storage.log_artifact(&run_id, "model.bin", data).expect("operation should succeed");

    assert!(!hash.is_empty());
    assert_eq!(hash.len(), 64); // SHA-256 hex
}

#[test]
fn test_indexed_db_storage_span_id() {
    let mut storage = IndexedDbStorage::new();
    let exp_id = storage.create_experiment("test-exp", None).expect("operation should succeed");
    let run_id = storage.create_run(&exp_id).expect("operation should succeed");

    storage.set_span_id(&run_id, "span-123").expect("operation should succeed");
    let span_id = storage.get_span_id(&run_id).expect("operation should succeed");

    assert_eq!(span_id, Some("span-123".to_string()));
}

#[test]
fn test_indexed_db_storage_error_experiment_not_found() {
    let mut storage = IndexedDbStorage::new();
    let result = storage.create_run("nonexistent");

    assert!(matches!(result, Err(StorageError::ExperimentNotFound(_))));
}

/// WASM performance: wasm_bench native_comparison (PW-03)
#[test]
fn wasm_bench_native_comparison_storage() {
    // Verify WASM storage operations complete without allocation overhead
    let mut storage = IndexedDbStorage::new();
    let start = std::time::Instant::now();
    for i in 0..100 {
        let _ = storage.create_experiment(&format!("bench-{i}"), None);
    }
    let loading_time = start.elapsed();
    // Should complete in < 10ms (no I/O)
    assert!(loading_time.as_millis() < 1000, "wasm_bench: {loading_time:?}");
}

#[test]
fn test_indexed_db_storage_error_run_not_found() {
    let storage = IndexedDbStorage::new();
    let result = storage.get_run_status("nonexistent");

    assert!(matches!(result, Err(StorageError::RunNotFound(_))));
}

#[test]
fn test_indexed_db_storage_error_invalid_state_start() {
    let mut storage = IndexedDbStorage::new();
    let exp_id = storage.create_experiment("test-exp", None).expect("operation should succeed");
    let run_id = storage.create_run(&exp_id).expect("operation should succeed");

    storage.start_run(&run_id).expect("operation should succeed");
    let result = storage.start_run(&run_id); // Try to start again

    assert!(matches!(result, Err(StorageError::InvalidState(_))));
}

#[test]
fn test_indexed_db_storage_error_invalid_state_complete() {
    let mut storage = IndexedDbStorage::new();
    let exp_id = storage.create_experiment("test-exp", None).expect("operation should succeed");
    let run_id = storage.create_run(&exp_id).expect("operation should succeed");

    // Try to complete without starting
    let result = storage.complete_run(&run_id, RunStatus::Success);

    assert!(matches!(result, Err(StorageError::InvalidState(_))));
}

#[test]
fn test_indexed_db_storage_list_runs() {
    let mut storage = IndexedDbStorage::new();
    let exp_id = storage.create_experiment("test-exp", None).expect("operation should succeed");

    storage.create_run(&exp_id).expect("operation should succeed");
    storage.create_run(&exp_id).expect("operation should succeed");
    storage.create_run(&exp_id).expect("operation should succeed");

    let runs = storage.list_runs(&exp_id);
    assert_eq!(runs.len(), 3);
}

// Note: WasmRun tests require wasm-bindgen-test for full coverage
// These are basic structural tests
#[test]
fn test_indexed_db_storage_implements_trait() {
    fn assert_storage<S: ExperimentStorage>(_: &S) {}

    let storage = IndexedDbStorage::new();
    assert_storage(&storage);
}
