//! Additional coverage tests for monitor module

use crate::monitor::*;

#[test]
fn test_metric_as_str() {
    assert_eq!(Metric::Loss.as_str(), "loss");
    assert_eq!(Metric::Accuracy.as_str(), "accuracy");
    assert_eq!(Metric::LearningRate.as_str(), "learning_rate");
    assert_eq!(Metric::GradientNorm.as_str(), "gradient_norm");
    assert_eq!(Metric::Epoch.as_str(), "epoch");
    assert_eq!(Metric::Batch.as_str(), "batch");
    assert_eq!(Metric::Custom("test".to_string()).as_str(), "test");
}

#[test]
fn test_metric_from_str_all_variants() {
    assert_eq!(Metric::from_str("loss"), Some(Metric::Loss));
    assert_eq!(Metric::from_str("accuracy"), Some(Metric::Accuracy));
    assert_eq!(Metric::from_str("learning_rate"), Some(Metric::LearningRate));
    assert_eq!(Metric::from_str("gradient_norm"), Some(Metric::GradientNorm));
    assert_eq!(Metric::from_str("epoch"), Some(Metric::Epoch));
    assert_eq!(Metric::from_str("batch"), Some(Metric::Batch));
    assert_eq!(Metric::from_str("unknown"), None);
}

#[test]
fn test_metric_record_with_tag() {
    let record =
        MetricRecord::new(Metric::Loss, 0.5).with_tag("phase", "training").with_tag("epoch", "1");

    assert_eq!(record.tags.get("phase"), Some(&"training".to_string()));
    assert_eq!(record.tags.get("epoch"), Some(&"1".to_string()));
}

#[test]
fn test_metrics_collector_record_batch() {
    let mut collector = MetricsCollector::new();
    collector.record_batch(&[(Metric::Loss, 0.5), (Metric::Accuracy, 0.85)]);

    assert_eq!(collector.count(), 2);
    let summary = collector.summary();
    assert!(summary.contains_key(&Metric::Loss));
    assert!(summary.contains_key(&Metric::Accuracy));
}

#[test]
fn test_metrics_collector_clear() {
    let mut collector = MetricsCollector::new();
    collector.record(Metric::Loss, 0.5);
    assert!(!collector.is_empty());

    collector.clear();
    assert!(collector.is_empty());
    assert_eq!(collector.count(), 0);
}

#[test]
fn test_metrics_collector_to_records() {
    let mut collector = MetricsCollector::new();
    collector.record(Metric::Loss, 0.5);
    collector.record(Metric::Loss, 0.4);

    let records = collector.to_records();
    assert_eq!(records.len(), 2);
}

#[test]
fn test_metrics_collector_to_json() {
    let mut collector = MetricsCollector::new();
    collector.record(Metric::Loss, 0.5);

    let json = collector.to_json().expect("operation should succeed");
    assert!(json.contains("Loss") || json.contains("loss") || json.contains("metric"));
}

#[test]
fn test_metrics_collector_summary_to_json() {
    let mut collector = MetricsCollector::new();
    collector.record(Metric::Loss, 0.5);
    collector.record(Metric::Loss, 0.4);

    let json = collector.summary_to_json().expect("operation should succeed");
    assert!(json.contains("mean"));
}

#[test]
fn test_metric_stats_default() {
    let stats = MetricStats::default();
    assert_eq!(stats.count, 0);
    assert_eq!(stats.mean, 0.0);
    assert_eq!(stats.min, f64::INFINITY);
    assert_eq!(stats.max, f64::NEG_INFINITY);
}

#[test]
fn test_running_stats_std_with_one_value() {
    let mut collector = MetricsCollector::new();
    collector.record(Metric::Loss, 5.0);

    let summary = collector.summary();
    let stats = summary.get(&Metric::Loss).expect("key should exist");
    assert_eq!(stats.std, 0.0); // Single value has 0 std
}

#[test]
fn test_running_stats_negative_inf() {
    let mut collector = MetricsCollector::new();
    collector.record(Metric::Loss, f64::NEG_INFINITY);

    let summary = collector.summary();
    let stats = summary.get(&Metric::Loss).expect("key should exist");
    assert!(stats.has_inf);
    assert_eq!(stats.min, f64::NEG_INFINITY);
}

#[test]
fn test_metrics_collector_default() {
    let collector = MetricsCollector::default();
    assert!(collector.is_empty());
}

#[test]
fn test_metric_clone() {
    let metric = Metric::Loss;
    let cloned = metric.clone();
    assert_eq!(metric, cloned);
}

#[test]
fn test_metric_custom_clone() {
    let metric = Metric::Custom("mymetric".to_string());
    let cloned = metric.clone();
    assert_eq!(metric, cloned);
}

#[test]
fn test_metric_stats_clone() {
    let stats = MetricStats {
        count: 5,
        mean: 0.5,
        std: 0.1,
        min: 0.2,
        max: 0.8,
        sum: 2.5,
        has_nan: false,
        has_inf: false,
    };
    let cloned = stats.clone();
    assert_eq!(stats.count, cloned.count);
    assert_eq!(stats.mean, cloned.mean);
}

#[test]
fn test_metric_record_clone() {
    let record = MetricRecord::new(Metric::Loss, 0.5).with_tag("test", "value");
    let cloned = record.clone();
    assert_eq!(record.metric, cloned.metric);
    assert_eq!(record.value, cloned.value);
}
