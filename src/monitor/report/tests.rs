//! Tests for the Hansei report module.

use super::*;
use crate::monitor::{Metric, MetricsCollector};

#[test]
fn test_hansei_analyzer_default() {
    let analyzer = HanseiAnalyzer::default();
    assert_eq!(analyzer.loss_increase_threshold, 0.1);
    assert_eq!(analyzer.gradient_explosion_threshold, 100.0);
}

#[test]
fn test_analyze_healthy_training() {
    let analyzer = HanseiAnalyzer::new();
    let mut collector = MetricsCollector::new();

    // Simulate healthy training: decreasing loss, increasing accuracy
    for i in 0..100 {
        let loss = 1.0 - (f64::from(i) * 0.008); // 1.0 -> 0.2
        let accuracy = 0.5 + (f64::from(i) * 0.004); // 0.5 -> 0.9
        collector.record(Metric::Loss, loss);
        collector.record(Metric::Accuracy, accuracy);
    }

    let report = analyzer.analyze("test-run-1", &collector, 120.0);

    assert_eq!(report.training_id, "test-run-1");
    assert_eq!(report.total_steps, 200); // 100 loss + 100 accuracy
    assert!(report.duration_secs == 120.0);
}

#[test]
fn test_detect_nan_loss() {
    let analyzer = HanseiAnalyzer::new();
    let mut collector = MetricsCollector::new();

    collector.record(Metric::Loss, 1.0);
    collector.record(Metric::Loss, f64::NAN);

    let report = analyzer.analyze("nan-test", &collector, 10.0);

    let critical_issues: Vec<_> =
        report.issues.iter().filter(|i| i.severity == IssueSeverity::Critical).collect();

    assert!(!critical_issues.is_empty());
    assert!(critical_issues[0].description.contains("NaN"));
}

#[test]
fn test_detect_inf_loss() {
    let analyzer = HanseiAnalyzer::new();
    let mut collector = MetricsCollector::new();

    collector.record(Metric::Loss, 1.0);
    collector.record(Metric::Loss, f64::INFINITY);

    let report = analyzer.analyze("inf-test", &collector, 10.0);

    let critical_issues: Vec<_> =
        report.issues.iter().filter(|i| i.severity == IssueSeverity::Critical).collect();

    assert!(!critical_issues.is_empty());
    assert!(critical_issues[0].description.contains("Infinity"));
}

#[test]
fn test_detect_gradient_explosion() {
    let analyzer = HanseiAnalyzer::new();
    let mut collector = MetricsCollector::new();

    collector.record(Metric::GradientNorm, 1.0);
    collector.record(Metric::GradientNorm, 500.0); // Explosion!

    let report = analyzer.analyze("grad-explosion", &collector, 10.0);

    let gradient_issues: Vec<_> =
        report.issues.iter().filter(|i| i.category == "Gradient Health").collect();

    assert!(!gradient_issues.is_empty());
    assert!(gradient_issues[0].description.contains("explosion"));
}

#[test]
fn test_detect_vanishing_gradients() {
    let analyzer = HanseiAnalyzer::new();
    let mut collector = MetricsCollector::new();

    // Very small gradients
    for _ in 0..20 {
        collector.record(Metric::GradientNorm, 1e-10);
    }

    let report = analyzer.analyze("vanishing-grad", &collector, 10.0);

    let gradient_issues: Vec<_> =
        report.issues.iter().filter(|i| i.description.contains("vanishing")).collect();

    assert!(!gradient_issues.is_empty());
}

#[test]
fn test_missing_loss_metric() {
    let analyzer = HanseiAnalyzer::new();
    let mut collector = MetricsCollector::new();

    // Only record accuracy, no loss
    collector.record(Metric::Accuracy, 0.5);

    let report = analyzer.analyze("no-loss", &collector, 10.0);

    let observability_issues: Vec<_> =
        report.issues.iter().filter(|i| i.category == "Observability").collect();

    assert!(!observability_issues.is_empty());
}

#[test]
fn test_format_report() {
    let analyzer = HanseiAnalyzer::new();
    let mut collector = MetricsCollector::new();

    collector.record(Metric::Loss, 1.0);
    collector.record(Metric::Loss, 0.5);
    collector.record(Metric::Accuracy, 0.8);

    let report = analyzer.analyze("format-test", &collector, 60.0);
    let formatted = analyzer.format_report(&report);

    assert!(formatted.contains("HANSEI POST-TRAINING REPORT"));
    assert!(formatted.contains("format-test"));
    assert!(formatted.contains("Duration: 60.00s"));
}

#[test]
fn test_trend_detection_improving_loss() {
    let analyzer = HanseiAnalyzer::new();
    let mut collector = MetricsCollector::new();

    // Loss that is biased toward lower values (mean < midpoint)
    // Use values with low CV to avoid oscillation detection
    // Range: 1.0 to 2.0, mean around 1.2 (below midpoint of 1.5)
    for _ in 0..40 {
        collector.record(Metric::Loss, 1.0);
    }
    for _ in 0..10 {
        collector.record(Metric::Loss, 2.0);
    }

    let report = analyzer.analyze("improving", &collector, 10.0);
    let loss_summary = report.metric_summaries.get(&Metric::Loss).expect("key should exist");

    // Mean = 1.2, midpoint = 1.5, so mean < midpoint → Improving
    assert!(
        loss_summary.trend == Trend::Improving,
        "Expected Improving, got {:?} (mean={:.2}, mid={:.2})",
        loss_summary.trend,
        loss_summary.mean,
        f64::midpoint(loss_summary.min, loss_summary.max)
    );
}

#[test]
fn test_trend_detection_oscillating() {
    let analyzer = HanseiAnalyzer::new();
    let mut collector = MetricsCollector::new();

    // Highly oscillating values
    for i in 0..50 {
        let value = if i % 2 == 0 { 10.0 } else { 1.0 };
        collector.record(Metric::Loss, value);
    }

    let report = analyzer.analyze("oscillating", &collector, 10.0);
    let loss_summary = report.metric_summaries.get(&Metric::Loss).expect("key should exist");

    assert_eq!(loss_summary.trend, Trend::Oscillating);
}

#[test]
fn test_issue_severity_ordering() {
    assert!(IssueSeverity::Critical > IssueSeverity::Error);
    assert!(IssueSeverity::Error > IssueSeverity::Warning);
    assert!(IssueSeverity::Warning > IssueSeverity::Info);
}

#[test]
fn test_recommendations_generated() {
    let analyzer = HanseiAnalyzer::new();
    let mut collector = MetricsCollector::new();

    collector.record(Metric::Loss, f64::NAN);

    let report = analyzer.analyze("rec-test", &collector, 10.0);

    assert!(!report.recommendations.is_empty());
    assert!(report.recommendations[0].contains("numerical stability"));
}

#[test]
fn test_low_accuracy_warning() {
    let analyzer = HanseiAnalyzer::new();
    let mut collector = MetricsCollector::new();

    // Low accuracy over many steps
    for _ in 0..150 {
        collector.record(Metric::Accuracy, 0.3);
    }

    let report = analyzer.analyze("low-acc", &collector, 100.0);

    let perf_issues: Vec<_> =
        report.issues.iter().filter(|i| i.category == "Performance").collect();

    assert!(!perf_issues.is_empty());
}

#[test]
fn test_empty_collector() {
    let analyzer = HanseiAnalyzer::new();
    let collector = MetricsCollector::new();

    let report = analyzer.analyze("empty", &collector, 0.0);

    assert_eq!(report.total_steps, 0);
    assert!(report.metric_summaries.is_empty());
    // Should have warning about missing loss
    assert!(report.issues.iter().any(|i| i.category == "Observability"));
}

#[test]
fn test_issue_severity_display() {
    assert_eq!(format!("{}", IssueSeverity::Info), "INFO");
    assert_eq!(format!("{}", IssueSeverity::Warning), "WARNING");
    assert_eq!(format!("{}", IssueSeverity::Error), "ERROR");
    assert_eq!(format!("{}", IssueSeverity::Critical), "CRITICAL");
}

#[test]
fn test_trend_display() {
    assert_eq!(format!("{}", Trend::Improving), "↑ Improving");
    assert_eq!(format!("{}", Trend::Degrading), "↓ Degrading");
    assert_eq!(format!("{}", Trend::Stable), "→ Stable");
    assert_eq!(format!("{}", Trend::Oscillating), "~ Oscillating");
}

#[test]
fn test_trend_detection_degrading() {
    let analyzer = HanseiAnalyzer::new();
    let mut collector = MetricsCollector::new();

    // Loss that is biased toward higher values (mean > midpoint)
    for _ in 0..10 {
        collector.record(Metric::Loss, 1.0);
    }
    for _ in 0..40 {
        collector.record(Metric::Loss, 2.0);
    }

    let report = analyzer.analyze("degrading", &collector, 10.0);
    let loss_summary = report.metric_summaries.get(&Metric::Loss).expect("key should exist");

    // Mean = 1.8, midpoint = 1.5, so mean > midpoint → Degrading
    assert!(
        loss_summary.trend == Trend::Degrading,
        "Expected Degrading, got {:?}",
        loss_summary.trend
    );
}

#[test]
fn test_trend_detection_stable() {
    let analyzer = HanseiAnalyzer::new();
    let mut collector = MetricsCollector::new();

    // GradientNorm with low coefficient of variation (cv < 0.2) is stable
    // cv = std / mean, so mean=1.0 with values 0.95-1.05 gives cv ≈ 0.03
    for i in 0..50 {
        collector.record(Metric::GradientNorm, 1.0 + (f64::from(i) % 10.0 - 5.0) * 0.01);
    }

    let report = analyzer.analyze("stable", &collector, 10.0);
    let grad_summary =
        report.metric_summaries.get(&Metric::GradientNorm).expect("key should exist");

    // With low CV, gradient norm should be stable
    assert!(grad_summary.trend == Trend::Stable, "Expected Stable, got {:?}", grad_summary.trend);
}

#[test]
fn test_custom_thresholds() {
    let analyzer = HanseiAnalyzer {
        loss_increase_threshold: 0.5,
        gradient_explosion_threshold: 50.0,
        gradient_vanishing_threshold: 1e-8,
        min_accuracy_improvement: 0.2,
    };

    assert_eq!(analyzer.loss_increase_threshold, 0.5);
    assert_eq!(analyzer.gradient_explosion_threshold, 50.0);
}

#[test]
fn test_training_issue_clone() {
    let issue = TrainingIssue {
        severity: IssueSeverity::Warning,
        category: "Test".to_string(),
        description: "Test description".to_string(),
        recommendation: "Test recommendation".to_string(),
    };
    let cloned = issue.clone();
    assert_eq!(issue.severity, cloned.severity);
    assert_eq!(issue.category, cloned.category);
}

#[test]
fn test_metric_summary_clone() {
    let summary = MetricSummary {
        initial: 1.0,
        final_value: 0.5,
        min: 0.3,
        max: 1.2,
        mean: 0.6,
        std_dev: 0.2,
        trend: Trend::Improving,
    };
    let cloned = summary.clone();
    assert_eq!(summary.initial, cloned.initial);
    assert_eq!(summary.trend, cloned.trend);
}

#[test]
fn test_post_training_report_clone() {
    use std::collections::HashMap;

    let report = PostTrainingReport {
        training_id: "test".to_string(),
        duration_secs: 10.0,
        total_steps: 100,
        final_metrics: HashMap::new(),
        metric_summaries: HashMap::new(),
        issues: vec![],
        recommendations: vec![],
    };
    let cloned = report.clone();
    assert_eq!(report.training_id, cloned.training_id);
    assert_eq!(report.total_steps, cloned.total_steps);
}
