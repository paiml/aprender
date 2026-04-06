use super::*;

#[test]
fn test_metric_type_names() {
    assert_eq!(MetricType::Gauge.name(), "gauge");
    assert_eq!(MetricType::Counter.name(), "counter");
    assert_eq!(MetricType::Histogram.name(), "histogram");
}

#[test]
fn test_labels() {
    let labels = Labels::new().add("host", "server1").add("env", "prod");

    assert_eq!(labels.len(), 2);
    assert!(labels.format().contains("host=\"server1\""));
}

#[test]
fn test_label_escaping() {
    assert_eq!(escape_label_value("hello"), "hello");
    assert_eq!(escape_label_value("hello\"world"), "hello\\\"world");
    assert_eq!(escape_label_value("line\nbreak"), "line\\nbreak");
}

#[test]
fn test_gauge_value() {
    let gauge = GaugeValue::new(42.5).with_labels(Labels::new().add("cpu", "0"));

    let formatted = gauge.format("cpu_usage");
    assert!(formatted.contains("cpu_usage"));
    assert!(formatted.contains("42.5"));
}

#[test]
fn test_counter_value() {
    let counter = CounterValue::new(100);
    let formatted = counter.format("requests_total");
    assert_eq!(formatted, "requests_total 100");
}

#[test]
fn test_histogram_observe() {
    let mut hist = HistogramValue::new();
    hist.observe(0.05);
    hist.observe(0.15);
    hist.observe(0.5);

    assert_eq!(hist.buckets.count, 3);
    assert!((hist.buckets.sum - 0.7).abs() < 0.001);
}

#[test]
fn test_registry_gauge() {
    let mut registry = MetricsRegistry::new();
    registry.register_gauge("cpu_temp", "CPU temperature in Celsius");

    let result = registry.set_gauge("cpu_temp", GaugeValue::new(65.0));
    assert!(result);

    let export = registry.export();
    assert!(export.contains("# HELP cpu_temp"));
    assert!(export.contains("# TYPE cpu_temp gauge"));
    assert!(export.contains("65"));
}

#[test]
fn test_registry_counter() {
    let mut registry = MetricsRegistry::new();
    registry.register_counter("requests_total", "Total requests");

    registry.inc_counter("requests_total", Labels::new());
    registry.add_counter("requests_total", 5, Labels::new());

    let export = registry.export();
    assert!(export.contains("# TYPE requests_total counter"));
    assert!(export.contains("6"));
}

#[test]
fn test_registry_histogram() {
    let mut registry = MetricsRegistry::new();
    registry.register_histogram("request_duration_seconds", "Request duration");

    registry.observe_histogram("request_duration_seconds", 0.05, Labels::new());

    let export = registry.export();
    assert!(export.contains("# TYPE request_duration_seconds histogram"));
    assert!(export.contains("_bucket"));
    assert!(export.contains("_sum"));
    assert!(export.contains("_count"));
}

#[test]
fn test_validate_metric_name() {
    assert!(validate_metric_name("cpu_usage"));
    assert!(validate_metric_name("requests_total"));
    assert!(validate_metric_name("_private_metric"));
    assert!(!validate_metric_name(""));
    assert!(!validate_metric_name("CpuUsage"));
    assert!(!validate_metric_name("123_invalid"));
}

#[test]
fn test_cardinality_limit() {
    let mut registry = MetricsRegistry::new().with_max_labels(2);
    registry.register_gauge("test", "Test metric");

    let labels_ok = Labels::new().add("a", "1").add("b", "2");
    let labels_too_many = Labels::new().add("a", "1").add("b", "2").add("c", "3");

    assert!(registry.set_gauge("test", GaugeValue::new(1.0).with_labels(labels_ok)));
    assert!(!registry.set_gauge("test", GaugeValue::new(1.0).with_labels(labels_too_many)));
}

#[test]
fn test_metric_def() {
    let def = MetricDef::new("test_metric", "A test metric", MetricType::Gauge);

    assert_eq!(def.format_help(), "# HELP test_metric A test metric");
    assert_eq!(def.format_type(), "# TYPE test_metric gauge");
}
