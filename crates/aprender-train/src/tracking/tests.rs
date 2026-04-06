//! Tests for the tracking module

use std::collections::HashMap;

use super::storage::{InMemoryBackend, JsonFileBackend, TrackingBackend, TrackingStorageError};
use super::{ExperimentTracker, Run, RunStatus, TrackingError};

// ---------------------------------------------------------------------------
// RunStatus tests
// ---------------------------------------------------------------------------

#[test]
fn test_run_status_equality() {
    assert_eq!(RunStatus::Active, RunStatus::Active);
    assert_eq!(RunStatus::Completed, RunStatus::Completed);
    assert_eq!(RunStatus::Failed, RunStatus::Failed);
    assert_eq!(RunStatus::Cancelled, RunStatus::Cancelled);
    assert_ne!(RunStatus::Active, RunStatus::Completed);
}

#[test]
fn test_run_status_clone() {
    let s = RunStatus::Active;
    let s2 = s;
    assert_eq!(s, s2);
}

#[test]
fn test_run_status_serde_roundtrip() {
    for status in [RunStatus::Active, RunStatus::Completed, RunStatus::Failed, RunStatus::Cancelled]
    {
        let json = serde_json::to_string(&status).expect("JSON serialization should succeed");
        let deserialized: RunStatus =
            serde_json::from_str(&json).expect("JSON deserialization should succeed");
        assert_eq!(status, deserialized);
    }
}

// ---------------------------------------------------------------------------
// Run struct tests
// ---------------------------------------------------------------------------

#[test]
fn test_run_new_defaults() {
    let run = Run::new("r-1".into(), Some("my run".into()), "exp-1".into());
    assert_eq!(run.run_id, "r-1");
    assert_eq!(run.run_name.as_deref(), Some("my run"));
    assert_eq!(run.experiment_name, "exp-1");
    assert_eq!(run.status, RunStatus::Active);
    assert!(run.params.is_empty());
    assert!(run.metrics.is_empty());
    assert!(run.artifacts.is_empty());
    assert!(run.tags.is_empty());
    assert!(run.start_time_ms.is_some());
    assert!(run.end_time_ms.is_none());
}

#[test]
fn test_run_new_no_name() {
    let run = Run::new("r-2".into(), None, "exp-2".into());
    assert!(run.run_name.is_none());
}

#[test]
fn test_run_serde_roundtrip() {
    let mut run = Run::new("r-3".into(), Some("test".into()), "exp-3".into());
    run.params.insert("lr".into(), "0.01".into());
    run.metrics.insert("loss".into(), vec![(0.5, 1), (0.3, 2)]);
    run.artifacts.push("model.bin".into());
    run.tags.insert("team".into(), "infra".into());

    let json = serde_json::to_string(&run).expect("JSON serialization should succeed");
    let deserialized: Run =
        serde_json::from_str(&json).expect("JSON deserialization should succeed");

    assert_eq!(deserialized.run_id, "r-3");
    assert_eq!(deserialized.params.get("lr").expect("key should exist"), "0.01");
    assert_eq!(deserialized.metrics["loss"].len(), 2);
    assert_eq!(deserialized.artifacts, vec!["model.bin"]);
}

// ---------------------------------------------------------------------------
// ExperimentTracker core tests
// ---------------------------------------------------------------------------

fn make_tracker() -> ExperimentTracker<InMemoryBackend> {
    ExperimentTracker::new("test-experiment", InMemoryBackend::new())
}

#[test]
fn test_tracker_creation() {
    let tracker = make_tracker();
    assert_eq!(tracker.experiment_name(), "test-experiment");
    assert!(tracker.tags().is_empty());
}

#[test]
fn test_tracker_tags() {
    let mut tracker = make_tracker();
    tracker.add_tag("env", "staging");
    tracker.add_tag("team", "ml");
    assert_eq!(tracker.tags().get("env").expect("key should exist"), "staging");
    assert_eq!(tracker.tags().get("team").expect("key should exist"), "ml");
}

#[test]
fn test_start_run_assigns_sequential_ids() {
    let mut tracker = make_tracker();
    let id1 = tracker.start_run(Some("first")).expect("operation should succeed");
    let id2 = tracker.start_run(Some("second")).expect("operation should succeed");
    assert_eq!(id1, "run-1");
    assert_eq!(id2, "run-2");
}

#[test]
fn test_start_run_inherits_tags() {
    let mut tracker = make_tracker();
    tracker.add_tag("env", "prod");

    let run_id = tracker.start_run(None).expect("operation should succeed");
    let run = tracker.get_run(&run_id).expect("operation should succeed");
    assert_eq!(run.tags.get("env").expect("key should exist"), "prod");
}

#[test]
fn test_start_run_without_name() {
    let mut tracker = make_tracker();
    let run_id = tracker.start_run(None).expect("operation should succeed");
    let run = tracker.get_run(&run_id).expect("operation should succeed");
    assert!(run.run_name.is_none());
}

// ---------------------------------------------------------------------------
// Parameter logging
// ---------------------------------------------------------------------------

#[test]
fn test_log_param() {
    let mut tracker = make_tracker();
    let run_id = tracker.start_run(None).expect("operation should succeed");

    tracker.log_param(&run_id, "lr", "0.001").expect("operation should succeed");
    tracker.log_param(&run_id, "batch_size", "32").expect("operation should succeed");

    let run = tracker.get_run(&run_id).expect("operation should succeed");
    assert_eq!(run.params.get("lr").expect("key should exist"), "0.001");
    assert_eq!(run.params.get("batch_size").expect("key should exist"), "32");
}

#[test]
fn test_log_param_overwrite() {
    let mut tracker = make_tracker();
    let run_id = tracker.start_run(None).expect("operation should succeed");

    tracker.log_param(&run_id, "lr", "0.001").expect("operation should succeed");
    tracker.log_param(&run_id, "lr", "0.01").expect("operation should succeed");

    let run = tracker.get_run(&run_id).expect("operation should succeed");
    assert_eq!(run.params.get("lr").expect("key should exist"), "0.01");
}

#[test]
fn test_log_params_batch() {
    let mut tracker = make_tracker();
    let run_id = tracker.start_run(None).expect("operation should succeed");

    let mut params = HashMap::new();
    params.insert("lr".into(), "0.001".into());
    params.insert("epochs".into(), "10".into());
    params.insert("optimizer".into(), "adam".into());

    tracker.log_params(&run_id, &params).expect("operation should succeed");

    let run = tracker.get_run(&run_id).expect("operation should succeed");
    assert_eq!(run.params.len(), 3);
    assert_eq!(run.params.get("optimizer").expect("key should exist"), "adam");
}

#[test]
fn test_log_param_on_nonexistent_run() {
    let mut tracker = make_tracker();
    let result = tracker.log_param("nonexistent", "lr", "0.001");
    assert!(result.is_err());
    match result.unwrap_err() {
        TrackingError::RunNotActive(id) => assert_eq!(id, "nonexistent"),
        other => panic!("Expected RunNotActive, got {other:?}"),
    }
}

#[test]
fn test_log_params_on_nonexistent_run() {
    let mut tracker = make_tracker();
    let params = HashMap::new();
    let result = tracker.log_params("nonexistent", &params);
    assert!(result.is_err());
}

// ---------------------------------------------------------------------------
// Metric logging
// ---------------------------------------------------------------------------

#[test]
fn test_log_metric_single() {
    let mut tracker = make_tracker();
    let run_id = tracker.start_run(None).expect("operation should succeed");

    tracker.log_metric(&run_id, "loss", 0.5, 1).expect("operation should succeed");

    let run = tracker.get_run(&run_id).expect("operation should succeed");
    let loss = &run.metrics["loss"];
    assert_eq!(loss.len(), 1);
    assert!((loss[0].0 - 0.5).abs() < f64::EPSILON);
    assert_eq!(loss[0].1, 1);
}

#[test]
fn test_log_metric_multiple_steps() {
    let mut tracker = make_tracker();
    let run_id = tracker.start_run(None).expect("operation should succeed");

    tracker.log_metric(&run_id, "loss", 0.5, 1).expect("operation should succeed");
    tracker.log_metric(&run_id, "loss", 0.3, 2).expect("operation should succeed");
    tracker.log_metric(&run_id, "loss", 0.1, 3).expect("operation should succeed");

    let run = tracker.get_run(&run_id).expect("operation should succeed");
    let loss = &run.metrics["loss"];
    assert_eq!(loss.len(), 3);
    assert!((loss[2].0 - 0.1).abs() < f64::EPSILON);
}

#[test]
fn test_log_metric_multiple_keys() {
    let mut tracker = make_tracker();
    let run_id = tracker.start_run(None).expect("operation should succeed");

    tracker.log_metric(&run_id, "loss", 0.5, 1).expect("operation should succeed");
    tracker.log_metric(&run_id, "accuracy", 0.8, 1).expect("operation should succeed");

    let run = tracker.get_run(&run_id).expect("operation should succeed");
    assert_eq!(run.metrics.len(), 2);
    assert!(run.metrics.contains_key("loss"));
    assert!(run.metrics.contains_key("accuracy"));
}

#[test]
fn test_log_metric_on_nonexistent_run() {
    let mut tracker = make_tracker();
    let result = tracker.log_metric("nonexistent", "loss", 0.5, 1);
    assert!(result.is_err());
}

// ---------------------------------------------------------------------------
// Artifact logging
// ---------------------------------------------------------------------------

#[test]
fn test_log_artifact() {
    let mut tracker = make_tracker();
    let run_id = tracker.start_run(None).expect("operation should succeed");

    tracker.log_artifact(&run_id, "model.safetensors").expect("operation should succeed");
    tracker.log_artifact(&run_id, "config.yaml").expect("config should be valid");

    let run = tracker.get_run(&run_id).expect("operation should succeed");
    assert_eq!(run.artifacts.len(), 2);
    assert_eq!(run.artifacts[0], "model.safetensors");
    assert_eq!(run.artifacts[1], "config.yaml");
}

#[test]
fn test_log_artifact_on_nonexistent_run() {
    let mut tracker = make_tracker();
    let result = tracker.log_artifact("nonexistent", "model.bin");
    assert!(result.is_err());
}

// ---------------------------------------------------------------------------
// End run
// ---------------------------------------------------------------------------

#[test]
fn test_end_run_completed() {
    let mut tracker = make_tracker();
    let run_id = tracker.start_run(Some("test")).expect("operation should succeed");
    tracker.log_param(&run_id, "lr", "0.001").expect("operation should succeed");
    tracker.log_metric(&run_id, "loss", 0.5, 1).expect("operation should succeed");

    tracker.end_run(&run_id, RunStatus::Completed).expect("operation should succeed");

    // Run should now be in the backend, not active
    let run = tracker.get_run(&run_id).expect("operation should succeed");
    assert_eq!(run.status, RunStatus::Completed);
    assert!(run.end_time_ms.is_some());
}

#[test]
fn test_end_run_failed() {
    let mut tracker = make_tracker();
    let run_id = tracker.start_run(None).expect("operation should succeed");
    tracker.end_run(&run_id, RunStatus::Failed).expect("operation should succeed");

    let run = tracker.get_run(&run_id).expect("operation should succeed");
    assert_eq!(run.status, RunStatus::Failed);
}

#[test]
fn test_end_run_cancelled() {
    let mut tracker = make_tracker();
    let run_id = tracker.start_run(None).expect("operation should succeed");
    tracker.end_run(&run_id, RunStatus::Cancelled).expect("operation should succeed");

    let run = tracker.get_run(&run_id).expect("operation should succeed");
    assert_eq!(run.status, RunStatus::Cancelled);
}

#[test]
fn test_end_run_nonexistent() {
    let mut tracker = make_tracker();
    let result = tracker.end_run("nonexistent", RunStatus::Completed);
    assert!(result.is_err());
    match result.unwrap_err() {
        TrackingError::RunNotFound(id) => assert_eq!(id, "nonexistent"),
        other => panic!("Expected RunNotFound, got {other:?}"),
    }
}

#[test]
fn test_end_run_sets_end_time() {
    let mut tracker = make_tracker();
    let run_id = tracker.start_run(None).expect("operation should succeed");
    tracker.end_run(&run_id, RunStatus::Completed).expect("operation should succeed");

    let run = tracker.get_run(&run_id).expect("operation should succeed");
    assert!(
        run.end_time_ms.expect("operation should succeed")
            >= run.start_time_ms.expect("operation should succeed")
    );
}

#[test]
fn test_cannot_log_after_end() {
    let mut tracker = make_tracker();
    let run_id = tracker.start_run(None).expect("operation should succeed");
    tracker.end_run(&run_id, RunStatus::Completed).expect("operation should succeed");

    // Run is no longer active -- logging should fail
    let result = tracker.log_param(&run_id, "lr", "0.001");
    assert!(result.is_err());

    let result = tracker.log_metric(&run_id, "loss", 0.5, 1);
    assert!(result.is_err());

    let result = tracker.log_artifact(&run_id, "model.bin");
    assert!(result.is_err());
}

// ---------------------------------------------------------------------------
// get_run / list_runs
// ---------------------------------------------------------------------------

#[test]
fn test_get_run_active() {
    let mut tracker = make_tracker();
    let run_id = tracker.start_run(Some("active")).expect("operation should succeed");

    let run = tracker.get_run(&run_id).expect("operation should succeed");
    assert_eq!(run.status, RunStatus::Active);
    assert_eq!(run.run_name.as_deref(), Some("active"));
}

#[test]
fn test_get_run_persisted() {
    let mut tracker = make_tracker();
    let run_id = tracker.start_run(None).expect("operation should succeed");
    tracker.end_run(&run_id, RunStatus::Completed).expect("operation should succeed");

    let run = tracker.get_run(&run_id).expect("operation should succeed");
    assert_eq!(run.status, RunStatus::Completed);
}

#[test]
fn test_get_run_not_found() {
    let tracker = make_tracker();
    let result = tracker.get_run("nonexistent");
    assert!(result.is_err());
}

#[test]
fn test_list_runs_empty() {
    let tracker = make_tracker();
    let runs = tracker.list_runs().expect("operation should succeed");
    assert!(runs.is_empty());
}

#[test]
fn test_list_runs_mixed() {
    let mut tracker = make_tracker();

    // One active run
    let _active = tracker.start_run(Some("active")).expect("operation should succeed");

    // One completed run
    let completed_id = tracker.start_run(Some("done")).expect("operation should succeed");
    tracker.end_run(&completed_id, RunStatus::Completed).expect("operation should succeed");

    let runs = tracker.list_runs().expect("operation should succeed");
    assert_eq!(runs.len(), 2);
}

#[test]
fn test_list_runs_sorted_by_id() {
    let mut tracker = make_tracker();
    let id1 = tracker.start_run(None).expect("operation should succeed");
    let id2 = tracker.start_run(None).expect("operation should succeed");
    let id3 = tracker.start_run(None).expect("operation should succeed");

    tracker.end_run(&id2, RunStatus::Completed).expect("operation should succeed");

    let runs = tracker.list_runs().expect("operation should succeed");
    assert_eq!(runs.len(), 3);
    assert_eq!(runs[0].run_id, id1);
    assert_eq!(runs[1].run_id, id2);
    assert_eq!(runs[2].run_id, id3);
}

// ---------------------------------------------------------------------------
// InMemoryBackend tests
// ---------------------------------------------------------------------------

#[test]
fn test_in_memory_backend_save_and_load() {
    let mut backend = InMemoryBackend::new();
    let run = Run::new("r-1".into(), None, "exp".into());

    backend.save_run(&run).expect("save should succeed");
    let loaded = backend.load_run("r-1").expect("load should succeed");
    assert_eq!(loaded.run_id, "r-1");
}

#[test]
fn test_in_memory_backend_load_not_found() {
    let backend = InMemoryBackend::new();
    let result = backend.load_run("nonexistent");
    assert!(result.is_err());
    match result.unwrap_err() {
        TrackingStorageError::RunNotFound(id) => assert_eq!(id, "nonexistent"),
        other => panic!("Expected RunNotFound, got {other:?}"),
    }
}

#[test]
fn test_in_memory_backend_list() {
    let mut backend = InMemoryBackend::new();

    backend.save_run(&Run::new("r-2".into(), None, "exp".into())).expect("save should succeed");
    backend.save_run(&Run::new("r-1".into(), None, "exp".into())).expect("save should succeed");

    let runs = backend.list_runs().expect("operation should succeed");
    assert_eq!(runs.len(), 2);
    // Sorted by run_id
    assert_eq!(runs[0].run_id, "r-1");
    assert_eq!(runs[1].run_id, "r-2");
}

#[test]
fn test_in_memory_backend_list_empty() {
    let backend = InMemoryBackend::new();
    let runs = backend.list_runs().expect("operation should succeed");
    assert!(runs.is_empty());
}

#[test]
fn test_in_memory_backend_delete() {
    let mut backend = InMemoryBackend::new();
    backend.save_run(&Run::new("r-1".into(), None, "exp".into())).expect("save should succeed");

    backend.delete_run("r-1").expect("operation should succeed");
    assert!(backend.load_run("r-1").is_err());
}

#[test]
fn test_in_memory_backend_delete_not_found() {
    let mut backend = InMemoryBackend::new();
    let result = backend.delete_run("nonexistent");
    assert!(result.is_err());
}

#[test]
fn test_in_memory_backend_overwrite() {
    let mut backend = InMemoryBackend::new();
    let mut run = Run::new("r-1".into(), None, "exp".into());
    backend.save_run(&run).expect("save should succeed");

    run.params.insert("lr".into(), "0.001".into());
    backend.save_run(&run).expect("save should succeed");

    let loaded = backend.load_run("r-1").expect("load should succeed");
    assert_eq!(loaded.params.get("lr").expect("key should exist"), "0.001");
}

// ---------------------------------------------------------------------------
// JsonFileBackend tests
// ---------------------------------------------------------------------------

#[test]
fn test_json_file_backend_save_and_load() {
    let dir = tempfile::tempdir().expect("temp file creation should succeed");
    let mut backend = JsonFileBackend::new(dir.path());

    let mut run = Run::new("r-1".into(), Some("test".into()), "exp".into());
    run.params.insert("lr".into(), "0.001".into());
    run.metrics.insert("loss".into(), vec![(0.5, 1), (0.3, 2)]);
    run.artifacts.push("model.bin".into());

    backend.save_run(&run).expect("save should succeed");

    let loaded = backend.load_run("r-1").expect("load should succeed");
    assert_eq!(loaded.run_id, "r-1");
    assert_eq!(loaded.run_name.as_deref(), Some("test"));
    assert_eq!(loaded.params.get("lr").expect("key should exist"), "0.001");
    assert_eq!(loaded.metrics["loss"].len(), 2);
    assert_eq!(loaded.artifacts, vec!["model.bin"]);
}

#[test]
fn test_json_file_backend_load_not_found() {
    let dir = tempfile::tempdir().expect("temp file creation should succeed");
    let backend = JsonFileBackend::new(dir.path());
    let result = backend.load_run("nonexistent");
    assert!(result.is_err());
}

#[test]
fn test_json_file_backend_list() {
    let dir = tempfile::tempdir().expect("temp file creation should succeed");
    let mut backend = JsonFileBackend::new(dir.path());

    backend.save_run(&Run::new("r-2".into(), None, "exp".into())).expect("save should succeed");
    backend.save_run(&Run::new("r-1".into(), None, "exp".into())).expect("save should succeed");

    let runs = backend.list_runs().expect("operation should succeed");
    assert_eq!(runs.len(), 2);
    assert_eq!(runs[0].run_id, "r-1");
    assert_eq!(runs[1].run_id, "r-2");
}

#[test]
fn test_json_file_backend_list_empty_nonexistent_dir() {
    let dir = tempfile::tempdir().expect("temp file creation should succeed");
    let backend = JsonFileBackend::new(dir.path().join("nonexistent"));
    let runs = backend.list_runs().expect("operation should succeed");
    assert!(runs.is_empty());
}

#[test]
fn test_json_file_backend_delete() {
    let dir = tempfile::tempdir().expect("temp file creation should succeed");
    let mut backend = JsonFileBackend::new(dir.path());

    backend.save_run(&Run::new("r-1".into(), None, "exp".into())).expect("save should succeed");
    backend.delete_run("r-1").expect("operation should succeed");
    assert!(backend.load_run("r-1").is_err());
}

#[test]
fn test_json_file_backend_delete_not_found() {
    let dir = tempfile::tempdir().expect("temp file creation should succeed");
    let mut backend = JsonFileBackend::new(dir.path());
    let result = backend.delete_run("nonexistent");
    assert!(result.is_err());
}

#[test]
fn test_json_file_backend_creates_dir() {
    let dir = tempfile::tempdir().expect("temp file creation should succeed");
    let nested = dir.path().join("a").join("b").join("c");
    let mut backend = JsonFileBackend::new(&nested);

    backend.save_run(&Run::new("r-1".into(), None, "exp".into())).expect("save should succeed");
    assert!(nested.exists());

    let loaded = backend.load_run("r-1").expect("load should succeed");
    assert_eq!(loaded.run_id, "r-1");
}

// ---------------------------------------------------------------------------
// Integration: full workflow
// ---------------------------------------------------------------------------

#[test]
fn test_full_tracking_workflow() {
    let dir = tempfile::tempdir().expect("temp file creation should succeed");
    let backend = JsonFileBackend::new(dir.path());
    let mut tracker = ExperimentTracker::new("lora-finetune", backend);

    tracker.add_tag("model", "llama-7b");
    tracker.add_tag("method", "qlora");

    // Run 1: baseline
    let run1 = tracker.start_run(Some("baseline")).expect("operation should succeed");
    tracker.log_param(&run1, "lr", "1e-4").expect("operation should succeed");
    tracker.log_param(&run1, "rank", "64").expect("operation should succeed");

    let mut batch_params = HashMap::new();
    batch_params.insert("batch_size".into(), "8".into());
    batch_params.insert("epochs".into(), "3".into());
    tracker.log_params(&run1, &batch_params).expect("operation should succeed");

    for step in 1u64..=5 {
        let loss = 1.0 / step as f64;
        tracker.log_metric(&run1, "loss", loss, step).expect("operation should succeed");
        tracker
            .log_metric(&run1, "accuracy", 0.5 + step as f64 * 0.1, step)
            .expect("operation should succeed");
    }

    tracker
        .log_artifact(&run1, "checkpoints/epoch_3.safetensors")
        .expect("operation should succeed");
    tracker.end_run(&run1, RunStatus::Completed).expect("operation should succeed");

    // Run 2: failed early
    let run2 = tracker.start_run(Some("failed-run")).expect("operation should succeed");
    tracker.log_metric(&run2, "loss", 999.0, 1).expect("operation should succeed");
    tracker.end_run(&run2, RunStatus::Failed).expect("operation should succeed");

    // Verify
    let runs = tracker.list_runs().expect("operation should succeed");
    assert_eq!(runs.len(), 2);

    let loaded1 = tracker.get_run(&run1).expect("load should succeed");
    assert_eq!(loaded1.status, RunStatus::Completed);
    assert_eq!(loaded1.params.len(), 4);
    assert_eq!(loaded1.metrics["loss"].len(), 5);
    assert_eq!(loaded1.metrics["accuracy"].len(), 5);
    assert_eq!(loaded1.artifacts.len(), 1);
    assert_eq!(loaded1.tags.get("model").expect("key should exist"), "llama-7b");

    let loaded2 = tracker.get_run(&run2).expect("load should succeed");
    assert_eq!(loaded2.status, RunStatus::Failed);
}

// ---------------------------------------------------------------------------
// Error display tests
// ---------------------------------------------------------------------------

#[test]
fn test_tracking_error_display() {
    let err = TrackingError::RunNotFound("r-42".into());
    assert!(err.to_string().contains("r-42"));

    let err = TrackingError::RunNotActive("r-99".into());
    assert!(err.to_string().contains("r-99"));
}

#[test]
fn test_storage_error_display() {
    let err = TrackingStorageError::RunNotFound("r-1".into());
    assert!(err.to_string().contains("r-1"));
}

// ---------------------------------------------------------------------------
// RunRecord conversion tests
// ---------------------------------------------------------------------------

#[test]
fn test_run_record_roundtrip() {
    use super::storage::RunRecord;

    let mut run = Run::new("r-1".into(), Some("test".into()), "exp".into());
    run.params.insert("lr".into(), "0.001".into());
    run.metrics.insert("loss".into(), vec![(0.5, 1), (0.3, 2)]);
    run.artifacts.push("model.bin".into());
    run.tags.insert("env".into(), "test".into());

    let record = RunRecord::from(&run);
    let restored = record.into_run();

    assert_eq!(restored.run_id, "r-1");
    assert_eq!(restored.run_name.as_deref(), Some("test"));
    assert_eq!(restored.params.get("lr").expect("key should exist"), "0.001");
    assert_eq!(restored.metrics["loss"].len(), 2);
    assert_eq!(restored.artifacts, vec!["model.bin"]);
    assert_eq!(restored.tags.get("env").expect("key should exist"), "test");
}
