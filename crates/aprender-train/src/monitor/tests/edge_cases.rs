//! Edge case tests for monitor module

use crate::monitor::*;

#[test]
fn test_empty_summary() {
    let collector = MetricsCollector::new();
    let summary = collector.summary();
    assert!(summary.is_empty());
}

#[test]
fn test_nan_handling() {
    let mut collector = MetricsCollector::new();
    collector.record(Metric::Loss, f64::NAN);

    let summary = collector.summary();
    let stats = summary.get(&Metric::Loss).expect("key should exist");

    // NaN values should be detected
    assert!(stats.has_nan);
}

#[test]
fn test_inf_handling() {
    let mut collector = MetricsCollector::new();
    collector.record(Metric::Loss, f64::INFINITY);

    let summary = collector.summary();
    let stats = summary.get(&Metric::Loss).expect("key should exist");

    // Inf values should be detected
    assert!(stats.has_inf);
}

#[test]
fn test_custom_metric() {
    let mut collector = MetricsCollector::new();
    collector.record(Metric::Custom("my_metric".to_string()), 42.0);

    let summary = collector.summary();
    assert!(summary.contains_key(&Metric::Custom("my_metric".to_string())));
}
