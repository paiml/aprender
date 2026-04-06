//! Span ID tests.

use crate::storage::sqlite::backend::SqliteBackend;
use crate::storage::ExperimentStorage;

#[test]
fn test_set_and_get_span_id() {
    let mut backend = SqliteBackend::open_in_memory().expect("operation should succeed");
    let exp_id = backend.create_experiment("test-exp", None).expect("operation should succeed");
    let run_id = backend.create_run(&exp_id).expect("operation should succeed");

    backend.set_span_id(&run_id, "span-123").expect("operation should succeed");
    let span_id = backend.get_span_id(&run_id).expect("operation should succeed");
    assert_eq!(span_id, Some("span-123".to_string()));
}

#[test]
fn test_get_span_id_none() {
    let mut backend = SqliteBackend::open_in_memory().expect("operation should succeed");
    let exp_id = backend.create_experiment("test-exp", None).expect("operation should succeed");
    let run_id = backend.create_run(&exp_id).expect("operation should succeed");

    let span_id = backend.get_span_id(&run_id).expect("operation should succeed");
    assert_eq!(span_id, None);
}
