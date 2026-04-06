//! Tests for Prometheus metrics module.

// =============================================================================
// Tests (TDD - Written First)
// =============================================================================

use super::*;

// -------------------------------------------------------------------------
// MetricDef Tests
// -------------------------------------------------------------------------

#[test]
fn test_metric_def_gauge() {
    let def = MetricDef::gauge("test_metric", "A test metric");
    assert_eq!(def.name, "test_metric");
    assert_eq!(def.help, "A test metric");
    assert_eq!(def.metric_type, MetricType::Gauge);
}

#[test]
fn test_metric_def_counter() {
    let def = MetricDef::counter("test_counter", "A test counter");
    assert_eq!(def.metric_type, MetricType::Counter);
}

#[test]
fn test_metric_def_with_labels() {
    let def = MetricDef::gauge("test", "help").with_labels(&["foo", "bar"]);
    assert_eq!(def.labels, vec!["foo", "bar"]);
}

// -------------------------------------------------------------------------
// LabelSet Tests
// -------------------------------------------------------------------------

#[test]
fn test_label_set_new() {
    let labels = LabelSet::new();
    assert!(labels.values.is_empty());
}

#[test]
fn test_label_set_from_pairs() {
    let labels = LabelSet::from_pairs(&[("foo", "bar"), ("baz", "qux")]);
    assert_eq!(labels.values.len(), 2);
}

#[test]
fn test_label_set_add() {
    let labels = LabelSet::new().add("key", "value");
    assert_eq!(labels.values.len(), 1);
    assert_eq!(labels.values[0], ("key".to_string(), "value".to_string()));
}

#[test]
fn test_label_set_format_empty() {
    let labels = LabelSet::new();
    assert_eq!(labels.format(), "");
}

#[test]
fn test_label_set_format() {
    let labels = LabelSet::from_pairs(&[("foo", "bar"), ("baz", "qux")]);
    let formatted = labels.format();
    assert!(formatted.contains("foo=\"bar\""));
    assert!(formatted.contains("baz=\"qux\""));
}

#[test]
fn test_label_set_format_escaping() {
    let labels = LabelSet::from_pairs(&[("key", "value with \"quotes\"")]);
    let formatted = labels.format();
    assert!(formatted.contains("\\\""));
}

// -------------------------------------------------------------------------
// PrometheusExporter Tests
// -------------------------------------------------------------------------

#[test]
fn test_exporter_new() {
    let exporter = PrometheusExporter::new("test-exp", "run-1");
    assert!(!exporter.definitions.is_empty());
}

#[test]
fn test_exporter_record() {
    let exporter = PrometheusExporter::new("test-exp", "run-1");
    exporter.record("entrenar_epoch_loss", 0.5);
    assert_eq!(exporter.total_samples(), 1);
}

#[test]
fn test_exporter_record_epoch() {
    let exporter = PrometheusExporter::new("test-exp", "run-1");
    exporter.record_epoch(1, 0.5, 0.001);
    assert_eq!(exporter.total_samples(), 3); // loss, lr, steps
}

#[test]
fn test_exporter_record_validation() {
    let exporter = PrometheusExporter::new("test-exp", "run-1");
    exporter.record_validation(0.4, 0.85);
    assert_eq!(exporter.total_samples(), 2);
}

#[test]
fn test_exporter_record_batch() {
    let exporter = PrometheusExporter::new("test-exp", "run-1");
    exporter.record_batch(100.0);
    assert_eq!(exporter.total_samples(), 1);
}

#[test]
fn test_exporter_record_gpu() {
    let exporter = PrometheusExporter::new("test-exp", "run-1");
    exporter.record_gpu(0, 85.0, 8_000_000_000.0, 65.0, 250.0);
    assert_eq!(exporter.total_samples(), 4); // util, mem, temp, power
}

#[test]
fn test_exporter_record_memory() {
    let exporter = PrometheusExporter::new("test-exp", "run-1");
    exporter.record_memory(4_000_000_000.0);
    assert_eq!(exporter.total_samples(), 1);
}

#[test]
fn test_exporter_clear() {
    let exporter = PrometheusExporter::new("test-exp", "run-1");
    exporter.record("entrenar_epoch_loss", 0.5);
    exporter.clear();

    let output = exporter.export();
    // Should not have any metric values after clear
    assert!(!output.contains("entrenar_epoch_loss{"));
}

// -------------------------------------------------------------------------
// Export Format Tests
// -------------------------------------------------------------------------

#[test]
fn test_exporter_export_prometheus_format() {
    let exporter = PrometheusExporter::new("test-exp", "run-1");
    exporter.record("entrenar_epoch_loss", 0.5);

    let output = exporter.export();

    // Should have HELP line
    assert!(output.contains("# HELP entrenar_epoch_loss"));
    // Should have TYPE line
    assert!(output.contains("# TYPE entrenar_epoch_loss gauge"));
    // Should have metric line with labels
    assert!(output.contains("entrenar_epoch_loss{"));
    assert!(output.contains("experiment=\"test-exp\""));
    assert!(output.contains("run=\"run-1\""));
}

#[test]
fn test_exporter_export_with_timestamp() {
    let exporter = PrometheusExporter::new("test-exp", "run-1");
    exporter.record("entrenar_epoch_loss", 0.5);

    let output = exporter.export();

    // Should have timestamp at end of line
    let lines: Vec<&str> = output.lines().collect();
    let metric_line = lines.iter().find(|l| l.starts_with("entrenar_epoch_loss{"));
    assert!(metric_line.is_some());

    // Line format: metric{labels} value timestamp
    let parts: Vec<&str> =
        metric_line.expect("operation should succeed").split_whitespace().collect();
    assert!(parts.len() >= 2); // metric+labels, value, optional timestamp
}

#[test]
fn test_exporter_export_json() {
    let exporter = PrometheusExporter::new("test-exp", "run-1");
    exporter.record("entrenar_epoch_loss", 0.5);

    let json = exporter.export_json();

    // Should be valid JSON
    let parsed: serde_json::Value =
        serde_json::from_str(&json).expect("JSON deserialization should succeed");
    assert!(parsed.is_object());

    // Should have the metric
    let obj = parsed.as_object().expect("parsing should succeed");
    assert!(obj.contains_key("entrenar_epoch_loss"));
}

#[test]
fn test_exporter_export_multiple_values() {
    let exporter = PrometheusExporter::new("test-exp", "run-1");
    exporter.record("entrenar_epoch_loss", 0.5);
    exporter.record("entrenar_epoch_loss", 0.4);
    exporter.record("entrenar_epoch_loss", 0.3);

    let output = exporter.export();

    // Should only have latest value in export (gauges show current value)
    let loss_lines: Vec<&str> =
        output.lines().filter(|l| l.starts_with("entrenar_epoch_loss{")).collect();
    assert_eq!(loss_lines.len(), 1);
}

#[test]
fn test_exporter_export_multiple_gpus() {
    let exporter = PrometheusExporter::new("test-exp", "run-1");
    exporter.record_gpu(0, 80.0, 8e9, 60.0, 200.0);
    exporter.record_gpu(1, 90.0, 12e9, 70.0, 300.0);

    let output = exporter.export();

    // Should have metrics for both devices
    assert!(output.contains("device=\"0\""));
    assert!(output.contains("device=\"1\""));
}

// -------------------------------------------------------------------------
// Custom Label Tests
// -------------------------------------------------------------------------

#[test]
fn test_exporter_with_custom_labels() {
    let labels = LabelSet::from_pairs(&[("model", "llama-7b"), ("dataset", "alpaca")]);
    let exporter = PrometheusExporter::with_labels(labels);
    exporter.record("entrenar_epoch_loss", 0.5);

    let output = exporter.export();
    assert!(output.contains("model=\"llama-7b\""));
    assert!(output.contains("dataset=\"alpaca\""));
}

#[test]
fn test_exporter_record_with_extra_labels() {
    let exporter = PrometheusExporter::new("test-exp", "run-1");

    let extra_labels = LabelSet::from_pairs(&[("layer", "attention")]);
    exporter.record_with_labels("entrenar_epoch_loss", 0.5, extra_labels);

    let output = exporter.export();
    assert!(output.contains("layer=\"attention\""));
    assert!(output.contains("experiment=\"test-exp\"")); // Default labels preserved
}

// -------------------------------------------------------------------------
// Metric Registration Tests
// -------------------------------------------------------------------------

#[test]
fn test_exporter_register_custom_metric() {
    let mut exporter = PrometheusExporter::new("test-exp", "run-1");

    exporter.register(MetricDef::gauge("custom_metric", "A custom metric"));
    exporter.record("custom_metric", 42.0);

    let output = exporter.export();
    assert!(output.contains("# HELP custom_metric A custom metric"));
    assert!(output.contains("custom_metric{"));
}

#[test]
fn test_exporter_default_metrics_registered() {
    let exporter = PrometheusExporter::new("test-exp", "run-1");

    // Check that default metrics are registered
    assert!(exporter.definitions.contains_key("entrenar_epoch_loss"));
    assert!(exporter.definitions.contains_key("entrenar_learning_rate"));
    assert!(exporter.definitions.contains_key("entrenar_gpu_utilization"));
    assert!(exporter.definitions.contains_key("entrenar_validation_accuracy"));
}

// =============================================================================
// Property Tests
// =============================================================================

mod property_tests {
    use super::*;
    use crate::monitor::prometheus::types::escape_label_value;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(200))]

        #[test]
        fn prop_label_escape_roundtrip(s in "[a-zA-Z0-9 \"\\\\\\n]{0,50}") {
            let escaped = escape_label_value(&s);
            // Escaped string should not have unescaped quotes or newlines
            prop_assert!(!escaped.contains('"') || escaped.contains("\\\""));
        }

        #[test]
        fn prop_label_set_format_balanced_braces(
            pairs in prop::collection::vec(
                ("[a-z]{1,10}", "[a-zA-Z0-9]{1,20}"),
                1..5
            )
        ) {
            let labels = LabelSet::from_pairs(
                &pairs.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect::<Vec<_>>()
            );
            let formatted = labels.format();

            // Count braces
            let open = formatted.chars().filter(|&c| c == '{').count();
            let close = formatted.chars().filter(|&c| c == '}').count();
            prop_assert_eq!(open, close);
        }

        #[test]
        fn prop_export_valid_format(
            loss in 0.0f64..10.0,
            lr in 1e-10f64..1.0
        ) {
            let exporter = PrometheusExporter::new("test", "run");
            exporter.record_epoch(1, loss, lr);

            let output = exporter.export();

            // Should have required components
            prop_assert!(output.contains("# HELP"));
            prop_assert!(output.contains("# TYPE"));
        }

        #[test]
        fn prop_total_samples_accurate(count in 1usize..100) {
            let exporter = PrometheusExporter::new("test", "run");
            for i in 0..count {
                exporter.record("entrenar_epoch_loss", i as f64 * 0.1);
            }
            prop_assert_eq!(exporter.total_samples() as usize, count);
        }

        #[test]
        fn prop_gpu_metrics_all_recorded(device_id in 0u32..8) {
            let exporter = PrometheusExporter::new("test", "run");
            exporter.record_gpu(device_id, 80.0, 8e9, 65.0, 200.0);

            let output = exporter.export();
            let device_str = format!("device=\"{device_id}\"");
            prop_assert!(output.contains(&device_str));
        }
    }
}
