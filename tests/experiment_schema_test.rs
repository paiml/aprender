//! Experiment Schema Tests (Phase 5: ENT-EPIC-001)
//!
//! EXTREME TDD: These tests were written BEFORE the implementation.
//! Run `cargo test experiment_schema` to confirm RED phase.

use trueno_db::experiment::{
    ArtifactRecord, ExperimentRecord, ExperimentStore, MetricRecord, RunRecord, RunStatus,
};

// =============================================================================
// ExperimentRecord Tests
// =============================================================================

#[test]
fn test_experiment_record_creation() {
    let record = ExperimentRecord::new("exp-001", "My Experiment");

    assert_eq!(record.experiment_id(), "exp-001");
    assert_eq!(record.name(), "My Experiment");
    assert!(record.created_at().timestamp() > 0);
    assert!(record.config().is_none());
}

#[test]
fn test_experiment_record_with_config() {
    let config = serde_json::json!({
        "learning_rate": 0.01,
        "batch_size": 32,
        "model": "resnet50"
    });

    let record =
        ExperimentRecord::builder("exp-002", "Training Run").config(config.clone()).build();

    assert_eq!(record.experiment_id(), "exp-002");
    assert_eq!(record.config(), Some(&config));
}

#[test]
fn test_experiment_record_serialization() {
    let record = ExperimentRecord::new("exp-003", "Serialization Test");

    let json = serde_json::to_string(&record).expect("serialization failed");
    let deserialized: ExperimentRecord =
        serde_json::from_str(&json).expect("deserialization failed");

    assert_eq!(record.experiment_id(), deserialized.experiment_id());
    assert_eq!(record.name(), deserialized.name());
    assert_eq!(record.created_at(), deserialized.created_at());
}

#[test]
fn test_experiment_record_equality() {
    let record1 = ExperimentRecord::new("exp-004", "Test");
    let record2 = ExperimentRecord::new("exp-004", "Test");

    // Different timestamps mean different records
    assert_ne!(record1, record2);

    // But same experiment_id and name
    assert_eq!(record1.experiment_id(), record2.experiment_id());
    assert_eq!(record1.name(), record2.name());
}

// =============================================================================
// RunRecord Tests
// =============================================================================

#[test]
fn test_run_record_creation() {
    let run = RunRecord::new("run-001", "exp-001");

    assert_eq!(run.run_id(), "run-001");
    assert_eq!(run.experiment_id(), "exp-001");
    assert_eq!(run.status(), RunStatus::Pending);
    assert!(run.started_at().is_none());
    assert!(run.ended_at().is_none());
    assert!(run.renacer_span_id().is_none());
}

#[test]
fn test_run_record_start() {
    let mut run = RunRecord::new("run-002", "exp-001");
    run.start();

    assert_eq!(run.status(), RunStatus::Running);
    assert!(run.started_at().is_some());
    assert!(run.ended_at().is_none());
}

#[test]
fn test_run_record_complete_success() {
    let mut run = RunRecord::new("run-003", "exp-001");
    run.start();
    run.complete(RunStatus::Success);

    assert_eq!(run.status(), RunStatus::Success);
    assert!(run.started_at().is_some());
    assert!(run.ended_at().is_some());
    assert!(run.ended_at().unwrap() >= run.started_at().unwrap());
}

#[test]
fn test_run_record_complete_failed() {
    let mut run = RunRecord::new("run-004", "exp-001");
    run.start();
    run.complete(RunStatus::Failed);

    assert_eq!(run.status(), RunStatus::Failed);
}

#[test]
fn test_run_record_with_renacer_span() {
    let run = RunRecord::builder("run-005", "exp-001").renacer_span_id("span-abc-123").build();

    assert_eq!(run.renacer_span_id(), Some("span-abc-123"));
}

#[test]
fn test_run_record_serialization() {
    let mut run = RunRecord::new("run-006", "exp-001");
    run.start();

    let json = serde_json::to_string(&run).expect("serialization failed");
    let deserialized: RunRecord = serde_json::from_str(&json).expect("deserialization failed");

    assert_eq!(run.run_id(), deserialized.run_id());
    assert_eq!(run.experiment_id(), deserialized.experiment_id());
    assert_eq!(run.status(), deserialized.status());
}

#[test]
fn test_run_status_variants() {
    // Ensure all status variants are covered
    assert_eq!(format!("{:?}", RunStatus::Pending), "Pending");
    assert_eq!(format!("{:?}", RunStatus::Running), "Running");
    assert_eq!(format!("{:?}", RunStatus::Success), "Success");
    assert_eq!(format!("{:?}", RunStatus::Failed), "Failed");
    assert_eq!(format!("{:?}", RunStatus::Cancelled), "Cancelled");
}

// =============================================================================
// MetricRecord Tests (Time-Series Optimized)
// =============================================================================

#[test]
fn test_metric_record_creation() {
    let metric = MetricRecord::new("run-001", "loss", 0, 0.5);

    assert_eq!(metric.run_id(), "run-001");
    assert_eq!(metric.key(), "loss");
    assert_eq!(metric.step(), 0);
    assert!((metric.value() - 0.5).abs() < f64::EPSILON);
    assert!(metric.timestamp().timestamp() > 0);
}

#[test]
fn test_metric_record_with_explicit_timestamp() {
    use chrono::{TimeZone, Utc};
    let ts = Utc.with_ymd_and_hms(2025, 1, 15, 12, 0, 0).unwrap();

    let metric = MetricRecord::builder("run-001", "accuracy", 100, 0.95).timestamp(ts).build();

    assert_eq!(metric.timestamp(), ts);
}

#[test]
fn test_metric_record_serialization() {
    let metric = MetricRecord::new("run-001", "loss", 50, 0.25);

    let json = serde_json::to_string(&metric).expect("serialization failed");
    let deserialized: MetricRecord = serde_json::from_str(&json).expect("deserialization failed");

    assert_eq!(metric.run_id(), deserialized.run_id());
    assert_eq!(metric.key(), deserialized.key());
    assert_eq!(metric.step(), deserialized.step());
    assert!((metric.value() - deserialized.value()).abs() < f64::EPSILON);
}

#[test]
fn test_metric_record_ordering_by_step() {
    let m1 = MetricRecord::new("run-001", "loss", 0, 1.0);
    let m2 = MetricRecord::new("run-001", "loss", 1, 0.9);
    let m3 = MetricRecord::new("run-001", "loss", 2, 0.8);

    // Metrics should be orderable by step for time-series queries
    assert!(m1.step() < m2.step());
    assert!(m2.step() < m3.step());
}

#[test]
fn test_metric_record_time_series_batch() {
    // Simulate a training loop writing metrics
    let metrics: Vec<MetricRecord> = (0..100)
        .map(|step| {
            let loss = 1.0 / (step as f64 + 1.0);
            MetricRecord::new("run-001", "loss", step, loss)
        })
        .collect();

    assert_eq!(metrics.len(), 100);
    assert_eq!(metrics[0].step(), 0);
    assert_eq!(metrics[99].step(), 99);

    // Loss should decrease over steps
    assert!(metrics[0].value() > metrics[99].value());
}

// =============================================================================
// ArtifactRecord Tests
// =============================================================================

#[test]
fn test_artifact_record_creation() {
    let artifact = ArtifactRecord::new(
        "run-001",
        "model.pt",
        "sha256:abc123def456",
        1024 * 1024 * 100, // 100MB
    );

    assert_eq!(artifact.run_id(), "run-001");
    assert_eq!(artifact.key(), "model.pt");
    assert_eq!(artifact.cas_hash(), "sha256:abc123def456");
    assert_eq!(artifact.size_bytes(), 100 * 1024 * 1024);
}

#[test]
fn test_artifact_record_serialization() {
    let artifact = ArtifactRecord::new("run-001", "checkpoint.ckpt", "sha256:xyz789", 5000);

    let json = serde_json::to_string(&artifact).expect("serialization failed");
    let deserialized: ArtifactRecord = serde_json::from_str(&json).expect("deserialization failed");

    assert_eq!(artifact.run_id(), deserialized.run_id());
    assert_eq!(artifact.key(), deserialized.key());
    assert_eq!(artifact.cas_hash(), deserialized.cas_hash());
    assert_eq!(artifact.size_bytes(), deserialized.size_bytes());
}

#[test]
fn test_artifact_record_cas_hash_format() {
    // CAS hash should follow content-addressable storage format
    let artifact = ArtifactRecord::new("run-001", "data.bin", "sha256:e3b0c44298fc1c14", 0);

    assert!(artifact.cas_hash().starts_with("sha256:"));
}

// =============================================================================
// Cross-Record Integration Tests
// =============================================================================

#[test]
fn test_experiment_run_metric_relationship() {
    let experiment = ExperimentRecord::new("exp-001", "Integration Test");
    let run = RunRecord::new("run-001", experiment.experiment_id());
    let metric = MetricRecord::new(run.run_id(), "accuracy", 0, 0.95);

    // Foreign key relationships
    assert_eq!(run.experiment_id(), experiment.experiment_id());
    assert_eq!(metric.run_id(), run.run_id());
}

#[test]
fn test_full_experiment_lifecycle() {
    // 1. Create experiment
    let config = serde_json::json!({"epochs": 10});
    let experiment = ExperimentRecord::builder("exp-lifecycle", "Full Test").config(config).build();

    // 2. Start a run
    let mut run = RunRecord::builder("run-lifecycle", experiment.experiment_id())
        .renacer_span_id("span-lifecycle")
        .build();
    run.start();

    // 3. Log metrics during training
    let metrics: Vec<MetricRecord> = (0..10)
        .map(|epoch| {
            MetricRecord::new(run.run_id(), "epoch_loss", epoch, 1.0 / (epoch as f64 + 1.0))
        })
        .collect();

    // 4. Save artifact
    let artifact = ArtifactRecord::new(run.run_id(), "final_model.pt", "sha256:final", 500_000);

    // 5. Complete run
    run.complete(RunStatus::Success);

    // Verify relationships
    assert_eq!(run.status(), RunStatus::Success);
    assert_eq!(metrics.len(), 10);
    assert_eq!(artifact.run_id(), run.run_id());
}

// =============================================================================
// ExperimentStore Tests
// =============================================================================

#[test]
fn test_experiment_store_creation() {
    let store = ExperimentStore::new();
    assert!(store.is_empty());
}

#[test]
fn test_experiment_store_add_metric() {
    let mut store = ExperimentStore::new();
    let metric = MetricRecord::new("run-001", "loss", 0, 0.5);

    store.add_metric(metric);
    assert!(!store.is_empty());
    assert_eq!(store.metric_count(), 1);
}

#[test]
fn test_experiment_store_add_multiple_metrics() {
    let mut store = ExperimentStore::new();

    // Add metrics for multiple runs and keys
    for step in 0..10 {
        store.add_metric(MetricRecord::new("run-001", "loss", step, 1.0 / (step as f64 + 1.0)));
        store.add_metric(MetricRecord::new("run-001", "accuracy", step, step as f64 / 10.0));
        store.add_metric(MetricRecord::new("run-002", "loss", step, 0.5 / (step as f64 + 1.0)));
    }

    assert_eq!(store.metric_count(), 30);
}

#[test]
fn test_get_metrics_for_run_single_key() {
    let mut store = ExperimentStore::new();

    // Add metrics for run-001
    for step in 0..5 {
        store.add_metric(MetricRecord::new("run-001", "loss", step, 1.0 / (step as f64 + 1.0)));
    }

    // Add metrics for different run (should not be returned)
    for step in 0..5 {
        store.add_metric(MetricRecord::new("run-002", "loss", step, 0.5));
    }

    // Query metrics for run-001, key "loss"
    let metrics = store.get_metrics_for_run("run-001", "loss");

    assert_eq!(metrics.len(), 5);
    for metric in &metrics {
        assert_eq!(metric.run_id(), "run-001");
        assert_eq!(metric.key(), "loss");
    }
}

#[test]
fn test_get_metrics_for_run_ordered_by_step() {
    let mut store = ExperimentStore::new();

    // Add metrics out of order
    store.add_metric(MetricRecord::new("run-001", "loss", 3, 0.3));
    store.add_metric(MetricRecord::new("run-001", "loss", 1, 0.1));
    store.add_metric(MetricRecord::new("run-001", "loss", 4, 0.4));
    store.add_metric(MetricRecord::new("run-001", "loss", 0, 0.0));
    store.add_metric(MetricRecord::new("run-001", "loss", 2, 0.2));

    // Query should return metrics ordered by step
    let metrics = store.get_metrics_for_run("run-001", "loss");

    assert_eq!(metrics.len(), 5);
    for (i, metric) in metrics.iter().enumerate() {
        assert_eq!(metric.step(), i as u64);
    }
}

#[test]
fn test_get_metrics_for_run_empty_result() {
    let mut store = ExperimentStore::new();

    // Add metrics for different run
    store.add_metric(MetricRecord::new("run-001", "loss", 0, 0.5));

    // Query for non-existent run
    let metrics = store.get_metrics_for_run("run-999", "loss");
    assert!(metrics.is_empty());

    // Query for non-existent key
    let metrics = store.get_metrics_for_run("run-001", "accuracy");
    assert!(metrics.is_empty());
}

#[test]
fn test_get_metrics_for_run_multiple_keys() {
    let mut store = ExperimentStore::new();

    // Add multiple metric types
    for step in 0..5 {
        store.add_metric(MetricRecord::new("run-001", "loss", step, 1.0));
        store.add_metric(MetricRecord::new("run-001", "accuracy", step, 0.9));
        store.add_metric(MetricRecord::new("run-001", "f1_score", step, 0.85));
    }

    // Query each key separately
    let loss_metrics = store.get_metrics_for_run("run-001", "loss");
    let acc_metrics = store.get_metrics_for_run("run-001", "accuracy");
    let f1_metrics = store.get_metrics_for_run("run-001", "f1_score");

    assert_eq!(loss_metrics.len(), 5);
    assert_eq!(acc_metrics.len(), 5);
    assert_eq!(f1_metrics.len(), 5);
}

#[test]
fn test_experiment_store_with_experiments_and_runs() {
    let mut store = ExperimentStore::new();

    // Add experiment
    let experiment = ExperimentRecord::new("exp-001", "Test Experiment");
    store.add_experiment(experiment.clone());

    // Add runs
    let run1 = RunRecord::new("run-001", "exp-001");
    let run2 = RunRecord::new("run-002", "exp-001");
    store.add_run(run1.clone());
    store.add_run(run2.clone());

    // Add metrics
    store.add_metric(MetricRecord::new("run-001", "loss", 0, 0.5));
    store.add_metric(MetricRecord::new("run-002", "loss", 0, 0.6));

    // Verify counts
    assert_eq!(store.experiment_count(), 1);
    assert_eq!(store.run_count(), 2);
    assert_eq!(store.metric_count(), 2);
}

#[test]
fn test_experiment_store_get_experiment() {
    let mut store = ExperimentStore::new();

    let experiment = ExperimentRecord::new("exp-001", "Test Experiment");
    store.add_experiment(experiment);

    let retrieved = store.get_experiment("exp-001");
    assert!(retrieved.is_some());
    assert_eq!(retrieved.unwrap().name(), "Test Experiment");

    let not_found = store.get_experiment("exp-999");
    assert!(not_found.is_none());
}

#[test]
fn test_experiment_store_get_runs_for_experiment() {
    let mut store = ExperimentStore::new();

    // Add runs for multiple experiments
    store.add_run(RunRecord::new("run-001", "exp-001"));
    store.add_run(RunRecord::new("run-002", "exp-001"));
    store.add_run(RunRecord::new("run-003", "exp-002"));

    let runs = store.get_runs_for_experiment("exp-001");
    assert_eq!(runs.len(), 2);
    for run in runs {
        assert_eq!(run.experiment_id(), "exp-001");
    }
}
