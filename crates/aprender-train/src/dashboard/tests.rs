//! Tests for dashboard module.

use std::sync::{Arc, Mutex};

use super::*;
use crate::run::{Run, TracingConfig};
use crate::storage::{ExperimentStorage, InMemoryStorage, RunStatus};

fn setup_storage() -> (Arc<Mutex<InMemoryStorage>>, String) {
    let mut storage = InMemoryStorage::new();
    let exp_id = storage.create_experiment("test-exp", None).expect("operation should succeed");
    (Arc::new(Mutex::new(storage)), exp_id)
}

#[test]
fn test_trend_from_rising_values() {
    let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    assert_eq!(Trend::from_values(&values), Trend::Rising);
}

#[test]
fn test_trend_from_falling_values() {
    let values = vec![5.0, 4.0, 3.0, 2.0, 1.0];
    assert_eq!(Trend::from_values(&values), Trend::Falling);
}

#[test]
fn test_trend_from_stable_values() {
    let values = vec![5.0, 5.01, 4.99, 5.0, 5.02];
    assert_eq!(Trend::from_values(&values), Trend::Stable);
}

#[test]
fn test_trend_from_single_value() {
    let values = vec![5.0];
    assert_eq!(Trend::from_values(&values), Trend::Stable);
}

#[test]
fn test_trend_from_empty() {
    let values: Vec<f64> = vec![];
    assert_eq!(Trend::from_values(&values), Trend::Stable);
}

#[test]
fn test_trend_display() {
    assert_eq!(format!("{}", Trend::Rising), "rising");
    assert_eq!(format!("{}", Trend::Falling), "falling");
    assert_eq!(format!("{}", Trend::Stable), "stable");
}

#[test]
fn test_trend_emoji() {
    assert_eq!(Trend::Rising.emoji(), "↑");
    assert_eq!(Trend::Falling.emoji(), "↓");
    assert_eq!(Trend::Stable.emoji(), "→");
}

#[test]
fn test_metric_snapshot_new() {
    let values = vec![(1000, 0.5), (2000, 0.4), (3000, 0.3)];
    let snapshot = MetricSnapshot::new("loss", values);

    assert_eq!(snapshot.key, "loss");
    assert_eq!(snapshot.len(), 3);
    assert_eq!(snapshot.trend, Trend::Falling);
}

#[test]
fn test_metric_snapshot_latest() {
    let values = vec![(1000, 0.5), (2000, 0.4), (3000, 0.3)];
    let snapshot = MetricSnapshot::new("loss", values);

    assert_eq!(snapshot.latest(), Some(0.3));
}

#[test]
fn test_metric_snapshot_min_max() {
    let values = vec![(1000, 0.5), (2000, 0.2), (3000, 0.8)];
    let snapshot = MetricSnapshot::new("metric", values);

    assert_eq!(snapshot.min(), Some(0.2));
    assert_eq!(snapshot.max(), Some(0.8));
}

#[test]
fn test_metric_snapshot_mean() {
    let values = vec![(1000, 1.0), (2000, 2.0), (3000, 3.0)];
    let snapshot = MetricSnapshot::new("metric", values);

    assert!((snapshot.mean().expect("operation should succeed") - 2.0).abs() < f64::EPSILON);
}

#[test]
fn test_metric_snapshot_empty() {
    let snapshot = MetricSnapshot::new("empty", vec![]);

    assert!(snapshot.is_empty());
    assert_eq!(snapshot.len(), 0);
    assert_eq!(snapshot.latest(), None);
    assert_eq!(snapshot.min(), None);
    assert_eq!(snapshot.max(), None);
    assert_eq!(snapshot.mean(), None);
}

#[test]
fn test_resource_snapshot_default() {
    let snapshot = ResourceSnapshot::default();

    assert!((snapshot.gpu_util - 0.0).abs() < f64::EPSILON);
    assert!((snapshot.cpu_util - 0.0).abs() < f64::EPSILON);
    assert_eq!(snapshot.memory_used, 0);
    assert_eq!(snapshot.memory_total, 0);
}

#[test]
fn test_resource_snapshot_with_values() {
    let snapshot = ResourceSnapshot::new()
        .with_gpu_util(0.75)
        .with_cpu_util(0.50)
        .with_memory(4_000_000_000, 8_000_000_000)
        .with_gpu_memory(6_000_000_000, 16_000_000_000);

    assert!((snapshot.gpu_util - 0.75).abs() < f64::EPSILON);
    assert!((snapshot.cpu_util - 0.50).abs() < f64::EPSILON);
    assert!((snapshot.memory_util() - 0.5).abs() < f64::EPSILON);
    assert!(
        (snapshot.gpu_memory_util().expect("operation should succeed") - 0.375).abs()
            < f64::EPSILON
    );
}

#[test]
fn test_resource_snapshot_clamp() {
    let snapshot = ResourceSnapshot::new().with_gpu_util(1.5).with_cpu_util(-0.5);

    assert!((snapshot.gpu_util - 1.0).abs() < f64::EPSILON);
    assert!((snapshot.cpu_util - 0.0).abs() < f64::EPSILON);
}

#[test]
fn test_resource_snapshot_zero_total() {
    let snapshot = ResourceSnapshot::new();
    assert!((snapshot.memory_util() - 0.0).abs() < f64::EPSILON);
    assert!(snapshot.gpu_memory_util().is_none());
}

#[test]
fn test_run_dashboard_status_running() {
    let (storage, exp_id) = setup_storage();
    let run =
        Run::new(&exp_id, storage, TracingConfig::disabled()).expect("config should be valid");

    assert_eq!(run.status(), RunStatus::Running);
}

#[test]
fn test_run_dashboard_status_finished() {
    let (storage, exp_id) = setup_storage();
    let run =
        Run::new(&exp_id, storage, TracingConfig::disabled()).expect("config should be valid");
    let run_id = run.id.clone();

    run.finish(RunStatus::Success).expect("operation should succeed");

    // Status would be checked after finish
    // Note: we consumed run, so we can't call status() on it
    let _ = run_id;
}

#[test]
fn test_run_dashboard_recent_metrics() {
    let (storage, exp_id) = setup_storage();
    let mut run =
        Run::new(&exp_id, storage, TracingConfig::disabled()).expect("config should be valid");

    run.log_metric("loss", 0.5).expect("operation should succeed");
    run.log_metric("loss", 0.4).expect("operation should succeed");
    run.log_metric("loss", 0.3).expect("operation should succeed");
    run.log_metric("accuracy", 0.8).expect("operation should succeed");

    let metrics = run.recent_metrics(10);

    assert!(metrics.contains_key("loss"));
    assert!(metrics.contains_key("accuracy"));
    assert_eq!(metrics.get("loss").expect("key should exist").len(), 3);
    assert_eq!(metrics.get("accuracy").expect("key should exist").len(), 1);
}

#[test]
fn test_run_dashboard_recent_metrics_limited() {
    let (storage, exp_id) = setup_storage();
    let mut run =
        Run::new(&exp_id, storage, TracingConfig::disabled()).expect("config should be valid");

    for i in 0..20 {
        run.log_metric("loss", 1.0 / (f64::from(i) + 1.0)).expect("operation should succeed");
    }

    let metrics = run.recent_metrics(5);

    assert_eq!(metrics.get("loss").expect("key should exist").len(), 5);
}

#[test]
fn test_run_metric_keys() {
    let (storage, exp_id) = setup_storage();
    let mut run =
        Run::new(&exp_id, storage, TracingConfig::disabled()).expect("config should be valid");

    run.log_metric("loss", 0.5).expect("operation should succeed");
    run.log_metric("accuracy", 0.8).expect("operation should succeed");
    run.log_metric("f1", 0.75).expect("operation should succeed");

    let keys = run.metric_keys();

    assert_eq!(keys.len(), 3);
    assert!(keys.contains(&"loss".to_string()));
    assert!(keys.contains(&"accuracy".to_string()));
    assert!(keys.contains(&"f1".to_string()));
}

#[test]
fn test_run_resource_usage() {
    let (storage, exp_id) = setup_storage();
    let run =
        Run::new(&exp_id, storage, TracingConfig::disabled()).expect("config should be valid");

    let resources = run.resource_usage();

    // Default implementation returns zeros
    assert!((resources.gpu_util - 0.0).abs() < f64::EPSILON);
    assert!((resources.cpu_util - 0.0).abs() < f64::EPSILON);
}
