//! Artifact tests.

use crate::storage::sqlite::backend::SqliteBackend;
use crate::storage::ExperimentStorage;

#[test]
fn test_log_artifact() {
    let mut backend = SqliteBackend::open_in_memory().expect("operation should succeed");
    let exp_id = backend.create_experiment("test-exp", None).expect("operation should succeed");
    let run_id = backend.create_run(&exp_id).expect("operation should succeed");

    let data = b"model weights data";
    let sha256 =
        backend.log_artifact(&run_id, "model.bin", data).expect("operation should succeed");

    assert!(!sha256.is_empty());
    assert_eq!(sha256.len(), 64); // SHA-256 hex length
}

#[test]
fn test_artifact_deduplication() {
    let mut backend = SqliteBackend::open_in_memory().expect("operation should succeed");
    let exp_id = backend.create_experiment("test-exp", None).expect("operation should succeed");
    let run_id = backend.create_run(&exp_id).expect("operation should succeed");

    let data = b"same data";
    let sha1 = backend.log_artifact(&run_id, "file1.bin", data).expect("operation should succeed");
    let sha2 = backend.log_artifact(&run_id, "file2.bin", data).expect("operation should succeed");

    // Same data should produce same hash
    assert_eq!(sha1, sha2);
}

#[test]
fn test_get_artifact_data() {
    let mut backend = SqliteBackend::open_in_memory().expect("operation should succeed");
    let exp_id = backend.create_experiment("test-exp", None).expect("operation should succeed");
    let run_id = backend.create_run(&exp_id).expect("operation should succeed");

    let data = b"test artifact data";
    let sha256 = backend.log_artifact(&run_id, "file.bin", data).expect("operation should succeed");

    let retrieved = backend.get_artifact_data(&sha256).expect("operation should succeed");
    assert_eq!(retrieved, data);
}

#[test]
fn test_list_artifacts() {
    let mut backend = SqliteBackend::open_in_memory().expect("operation should succeed");
    let exp_id = backend.create_experiment("test-exp", None).expect("operation should succeed");
    let run_id = backend.create_run(&exp_id).expect("operation should succeed");

    backend.log_artifact(&run_id, "model.bin", b"model").expect("operation should succeed");
    backend.log_artifact(&run_id, "config.json", b"config").expect("config should be valid");

    let artifacts = backend.list_artifacts(&run_id).expect("operation should succeed");
    assert_eq!(artifacts.len(), 2);
}
