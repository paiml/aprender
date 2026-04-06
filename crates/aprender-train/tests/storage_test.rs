//! Integration tests for ExperimentStorage (ENT-001)

use entrenar::storage::{ExperimentStorage, InMemoryStorage, MetricPoint, RunStatus};

#[test]
fn test_experiment_storage_trait_exists() {
    fn assert_storage<S: ExperimentStorage>() {}
    assert_storage::<InMemoryStorage>();
}

#[test]
fn test_full_experiment_lifecycle() {
    let mut storage = InMemoryStorage::new();

    // Create experiment
    let exp_id = storage
        .create_experiment("integration-test", Some(serde_json::json!({"lr": 0.001})))
        .expect("operation should succeed");
    assert!(exp_id.starts_with("exp-"));

    // Create and start run
    let run_id = storage.create_run(&exp_id).expect("operation should succeed");
    assert_eq!(
        storage.get_run_status(&run_id).expect("operation should succeed"),
        RunStatus::Pending
    );

    storage.start_run(&run_id).expect("operation should succeed");
    assert_eq!(
        storage.get_run_status(&run_id).expect("operation should succeed"),
        RunStatus::Running
    );

    // Log metrics
    for step in 0..10 {
        let loss = 1.0 / (step as f64 + 1.0);
        storage.log_metric(&run_id, "loss", step, loss).expect("operation should succeed");
    }

    // Verify metrics
    let metrics = storage.get_metrics(&run_id, "loss").expect("operation should succeed");
    assert_eq!(metrics.len(), 10);
    assert_eq!(metrics[0].step, 0);
    assert_eq!(metrics[9].step, 9);

    // Log artifact
    let artifact_data = b"model weights binary data";
    let hash = storage
        .log_artifact(&run_id, "model.bin", artifact_data)
        .expect("operation should succeed");
    assert!(hash.starts_with("sha256-"));

    // Complete run
    storage.complete_run(&run_id, RunStatus::Success).expect("operation should succeed");
    assert_eq!(
        storage.get_run_status(&run_id).expect("operation should succeed"),
        RunStatus::Success
    );
}

#[test]
fn test_metric_point_struct() {
    let point = MetricPoint::new(5, 0.123);
    assert_eq!(point.step, 5);
    assert!((point.value - 0.123).abs() < f64::EPSILON);
}

#[test]
fn test_run_status_re_exported() {
    // RunStatus should be re-exported from storage module
    let _pending = RunStatus::Pending;
    let _running = RunStatus::Running;
    let _success = RunStatus::Success;
    let _failed = RunStatus::Failed;
    let _cancelled = RunStatus::Cancelled;
}

#[test]
fn test_multiple_runs_same_experiment() {
    let mut storage = InMemoryStorage::new();
    let exp_id = storage.create_experiment("multi-run", None).expect("operation should succeed");

    let run1 = storage.create_run(&exp_id).expect("operation should succeed");
    let run2 = storage.create_run(&exp_id).expect("operation should succeed");
    let run3 = storage.create_run(&exp_id).expect("operation should succeed");

    // All runs are independent
    storage.start_run(&run1).expect("operation should succeed");
    storage.start_run(&run2).expect("operation should succeed");

    assert_eq!(
        storage.get_run_status(&run1).expect("operation should succeed"),
        RunStatus::Running
    );
    assert_eq!(
        storage.get_run_status(&run2).expect("operation should succeed"),
        RunStatus::Running
    );
    assert_eq!(
        storage.get_run_status(&run3).expect("operation should succeed"),
        RunStatus::Pending
    );

    // Complete runs with different statuses
    storage.complete_run(&run1, RunStatus::Success).expect("operation should succeed");
    storage.complete_run(&run2, RunStatus::Failed).expect("operation should succeed");

    assert_eq!(
        storage.get_run_status(&run1).expect("operation should succeed"),
        RunStatus::Success
    );
    assert_eq!(storage.get_run_status(&run2).expect("operation should succeed"), RunStatus::Failed);
}

#[test]
fn test_span_id_tracking() {
    let mut storage = InMemoryStorage::new();
    let exp_id = storage.create_experiment("span-test", None).expect("operation should succeed");
    let run_id = storage.create_run(&exp_id).expect("operation should succeed");

    // Initially no span
    assert!(storage.get_span_id(&run_id).expect("operation should succeed").is_none());

    // Set span ID
    storage.set_span_id(&run_id, "renacer-span-123").expect("operation should succeed");

    // Verify span ID
    assert_eq!(
        storage.get_span_id(&run_id).expect("operation should succeed"),
        Some("renacer-span-123".to_string())
    );
}

#[cfg(feature = "monitor")]
mod trueno_tests {
    use entrenar::storage::{ExperimentStorage, RunStatus, TruenoBackend};

    #[test]
    fn test_trueno_backend_implements_trait() {
        fn assert_storage<S: ExperimentStorage>() {}
        assert_storage::<TruenoBackend>();
    }

    #[test]
    fn test_trueno_full_lifecycle() {
        let mut backend = TruenoBackend::new();

        let exp_id =
            backend.create_experiment("trueno-test", None).expect("operation should succeed");
        let run_id = backend.create_run(&exp_id).expect("operation should succeed");

        backend.start_run(&run_id).expect("operation should succeed");
        backend.log_metric(&run_id, "loss", 0, 0.5).expect("operation should succeed");
        backend.log_metric(&run_id, "loss", 1, 0.4).expect("operation should succeed");

        let metrics = backend.get_metrics(&run_id, "loss").expect("operation should succeed");
        assert_eq!(metrics.len(), 2);

        backend.complete_run(&run_id, RunStatus::Success).expect("operation should succeed");
        assert_eq!(
            backend.get_run_status(&run_id).expect("operation should succeed"),
            RunStatus::Success
        );
    }
}
