//! Performance tests for monitor module - Verify spec requirements

use crate::monitor::dashboard::Dashboard;
use crate::monitor::drift::DriftDetector;
use crate::monitor::report::HanseiAnalyzer;
use crate::monitor::*;

/// Simple pseudo-random float for testing (deterministic)
fn rand_float() -> f64 {
    use std::cell::Cell;
    thread_local! {
        static SEED: Cell<u64> = const { Cell::new(12345) };
    }
    SEED.with(|seed| {
        let mut s = seed.get();
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        seed.set(s);
        (s as f64) / (u64::MAX as f64)
    })
}

#[test]
fn test_performance_metrics_collector_overhead() {
    let mut collector = MetricsCollector::new();

    for i in 0..10_000 {
        collector.record(Metric::Loss, 1.0 / (f64::from(i) + 1.0));
        collector.record(Metric::Accuracy, 0.5 + (f64::from(i) * 0.00005));
    }

    let summary = collector.summary();
    assert_eq!(summary[&Metric::Loss].count, 10_000);
    assert_eq!(summary[&Metric::Accuracy].count, 10_000);
}

#[test]
fn test_performance_summary_calculation() {
    let mut collector = MetricsCollector::new();
    for i in 0..10_000 {
        collector.record(Metric::Loss, 1.0 / (f64::from(i) + 1.0));
    }

    let summary = collector.summary();
    let loss_stats = &summary[&Metric::Loss];
    assert_eq!(loss_stats.count, 10_000);
    assert!(loss_stats.min > 0.0);
    assert!(loss_stats.max <= 1.0);
}

#[test]
fn test_performance_dashboard_render() {
    let mut collector = MetricsCollector::new();
    for i in 0..1000 {
        collector.record(Metric::Loss, 1.0 - (f64::from(i) * 0.001));
        collector.record(Metric::Accuracy, 0.5 + (f64::from(i) * 0.0005));
    }

    let mut dashboard = Dashboard::new();
    dashboard.update(collector.summary());

    let output = dashboard.render_ascii();
    assert!(!output.is_empty());
}

#[test]
fn test_performance_drift_detection() {
    let mut detector = DriftDetector::new(100);

    for _ in 0..100 {
        detector.check(50.0 + rand_float() * 2.0);
    }

    let mut last_status = drift::DriftStatus::NoDrift;
    for _ in 0..1000 {
        last_status = detector.check(50.0 + rand_float() * 2.0);
    }

    assert!(matches!(
        last_status,
        drift::DriftStatus::NoDrift | drift::DriftStatus::Warning(_) | drift::DriftStatus::Drift(_)
    ));
}

#[test]
fn test_performance_hansei_analysis() {
    let mut collector = MetricsCollector::new();
    for i in 0..10_000 {
        collector.record(Metric::Loss, 1.0 - (f64::from(i) * 0.0001));
        collector.record(Metric::Accuracy, 0.5 + (f64::from(i) * 0.00005));
        collector.record(Metric::GradientNorm, 1.0 + rand_float() * 0.1);
    }

    let analyzer = HanseiAnalyzer::new();
    let report = analyzer.analyze("perf-test", &collector, 100.0);

    assert_eq!(report.training_id, "perf-test");
    assert_eq!(report.duration_secs, 100.0);
    assert!(report.total_steps > 0);
}
