//! Thread safety tests.

use crate::storage::sqlite::backend::SqliteBackend;
use crate::storage::ExperimentStorage;

/// Verify SqliteBackend satisfies Send + Sync bounds required by ExperimentStorage
#[test]
fn test_sqlite_backend_is_send_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<SqliteBackend>();
}

/// Verify sequential multi-step workflows produce consistent results.
/// With `&mut self` trait methods, concurrent writes go through the Mutex
/// internally — this test validates the full lifecycle works correctly.
#[test]
fn test_sequential_metric_logging() {
    let mut backend = SqliteBackend::open_in_memory().expect("operation should succeed");
    let exp_id = backend.create_experiment("test-exp", None).expect("operation should succeed");
    let run_id = backend.create_run(&exp_id).expect("operation should succeed");

    backend.start_run(&run_id).expect("operation should succeed");

    for i in 0..10 {
        backend.log_metric(&run_id, "loss", i, i as f64 * 0.1).expect("operation should succeed");
    }

    let metrics = backend.get_metrics(&run_id, "loss").expect("operation should succeed");
    assert_eq!(metrics.len(), 10);

    // Verify ordering
    for (i, point) in metrics.iter().enumerate() {
        assert_eq!(point.step, i as u64);
    }
}
