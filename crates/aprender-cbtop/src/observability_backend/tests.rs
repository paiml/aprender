use super::*;

#[test]
fn test_export_metric_creation() {
    let metric =
        ExportMetric::gauge("cpu_usage", 75.5).with_tag("host", "server1").with_unit("percent");

    assert_eq!(metric.name, "cpu_usage");
    assert_eq!(metric.value, 75.5);
    assert!(matches!(metric.metric_type, MetricExportType::Gauge));
    assert_eq!(metric.tags.get("host"), Some(&"server1".to_string()));
    assert_eq!(metric.unit, Some("percent".to_string()));
}

#[test]
fn test_counter_metric() {
    let metric = ExportMetric::counter("requests", 1000.0);

    assert_eq!(metric.name, "requests");
    assert!(matches!(metric.metric_type, MetricExportType::Counter));
}

#[test]
fn test_datadog_config() {
    let config = DatadogConfig::default();

    assert_eq!(config.host, "localhost");
    assert_eq!(config.port, 8125);
    assert_eq!(config.prefix, "cbtop");
}

#[test]
fn test_newrelic_config() {
    let config =
        NewRelicConfig::new("api-key", "12345").with_endpoint("https://eu-api.newrelic.com");

    assert_eq!(config.api_key, "api-key");
    assert_eq!(config.account_id, "12345");
    assert!(config.endpoint.contains("eu-api"));
}

#[test]
fn test_honeycomb_config() {
    let config = HoneycombConfig::new("api-key", "my-dataset");

    assert_eq!(config.api_key, "api-key");
    assert_eq!(config.dataset, "my-dataset");
    assert_eq!(config.service_name, "cbtop");
}

#[test]
fn test_webhook_config() {
    let config = WebhookConfig::new("https://example.com/metrics").with_auth("token123");

    assert_eq!(config.url, "https://example.com/metrics");
    assert_eq!(config.auth_token, Some("token123".to_string()));
}

#[test]
fn test_observability_config_enabled_backends() {
    let config = ObservabilityConfig::new()
        .with_datadog(DatadogConfig::default())
        .with_honeycomb(HoneycombConfig::new("key", "dataset"));

    let backends = config.enabled_backends();

    assert_eq!(backends.len(), 2);
    assert!(backends.contains(&ObservabilityBackend::Datadog));
    assert!(backends.contains(&ObservabilityBackend::Honeycomb));
}

#[test]
fn test_exporter_record_and_flush() {
    let config = ObservabilityConfig::new().with_datadog(DatadogConfig::default());

    let mut exporter = ObservabilityExporter::new(config);

    exporter.record(ExportMetric::gauge("metric1", 10.0));
    exporter.record(ExportMetric::gauge("metric2", 20.0));

    assert_eq!(exporter.buffer_size(), 2);

    let results = exporter.flush();

    assert_eq!(results.len(), 1); // One backend (Datadog)
    assert!(results[0].success);
    assert_eq!(results[0].metrics_exported, 2);
    assert_eq!(exporter.buffer_size(), 0);
}

#[test]
fn test_exporter_auto_flush() {
    let mut config = ObservabilityConfig::new().with_datadog(DatadogConfig::default());
    config.batch_size = 3;

    let mut exporter = ObservabilityExporter::new(config);

    exporter.record(ExportMetric::gauge("m1", 1.0));
    exporter.record(ExportMetric::gauge("m2", 2.0));
    assert_eq!(exporter.buffer_size(), 2);

    // Third record triggers auto-flush
    exporter.record(ExportMetric::gauge("m3", 3.0));
    assert_eq!(exporter.buffer_size(), 0); // Buffer flushed
}

#[test]
fn test_exporter_health_tracking() {
    let config = ObservabilityConfig::new().with_datadog(DatadogConfig::default());

    let mut exporter = ObservabilityExporter::new(config);

    exporter.record(ExportMetric::gauge("metric1", 10.0));
    exporter.flush();

    let health = exporter.get_health(ObservabilityBackend::Datadog).unwrap();
    assert!(health.healthy);
    assert!(health.last_success.is_some());
    assert_eq!(health.consecutive_failures, 0);
}

#[test]
fn test_export_count() {
    let config = ObservabilityConfig::new().with_datadog(DatadogConfig::default());

    let mut exporter = ObservabilityExporter::new(config);

    exporter.record(ExportMetric::gauge("m1", 1.0));
    exporter.record(ExportMetric::gauge("m2", 2.0));
    exporter.flush();

    assert_eq!(exporter.export_count(ObservabilityBackend::Datadog), 2);
}

#[test]
fn test_dogstatsd_format() {
    let metric = ExportMetric::gauge("cpu", 75.0).with_tag("host", "server1");

    let formatted = format_dogstatsd(&metric, "cbtop", &["env:prod".to_string()]);

    assert!(formatted.starts_with("cbtop.cpu:75"));
    assert!(formatted.contains("|g|"));
    assert!(formatted.contains("host:server1"));
    assert!(formatted.contains("env:prod"));
}

#[test]
fn test_backend_name() {
    assert_eq!(ObservabilityBackend::Datadog.name(), "Datadog");
    assert_eq!(ObservabilityBackend::NewRelic.name(), "NewRelic");
    assert_eq!(ObservabilityBackend::Honeycomb.name(), "Honeycomb");
    assert_eq!(ObservabilityBackend::Otlp.name(), "OTLP");
    assert_eq!(ObservabilityBackend::Webhook.name(), "Webhook");
}

#[test]
fn test_error_display() {
    let err = ObservabilityError::ConnectionFailed {
        backend: "Datadog".to_string(),
        reason: "timeout".to_string(),
    };
    assert!(err.to_string().contains("Datadog"));
    assert!(err.to_string().contains("timeout"));

    let err =
        ObservabilityError::RateLimited { backend: "NewRelic".to_string(), retry_after_sec: 60 };
    assert!(err.to_string().contains("60"));
}

#[test]
fn test_record_batch() {
    let config = ObservabilityConfig::new().with_datadog(DatadogConfig::default());

    let mut exporter = ObservabilityExporter::new(config);

    let metrics = vec![
        ExportMetric::gauge("m1", 1.0),
        ExportMetric::gauge("m2", 2.0),
        ExportMetric::gauge("m3", 3.0),
    ];

    exporter.record_batch(metrics);

    assert_eq!(exporter.buffer_size(), 3);
}

#[test]
fn test_should_flush_empty() {
    let config = ObservabilityConfig::new().with_datadog(DatadogConfig::default());

    let exporter = ObservabilityExporter::new(config);

    // Empty buffer should not need flush
    assert!(!exporter.should_flush());
}

// FKR-047: Multi-backend simultaneous export
#[test]
fn test_fkr_047_multi_backend_export() {
    let config = ObservabilityConfig::new()
        .with_datadog(DatadogConfig::default())
        .with_newrelic(NewRelicConfig::new("key", "account"))
        .with_honeycomb(HoneycombConfig::new("key", "dataset"))
        .with_otlp(OtlpConfig::default())
        .with_webhook(WebhookConfig::new("https://example.com"));

    let mut exporter = ObservabilityExporter::new(config);

    // Record test metrics
    for i in 0..10 {
        exporter.record(ExportMetric::gauge(format!("metric_{}", i), i as f64));
    }

    // Flush to all backends
    let results = exporter.flush();

    // Verify we have results from all 5 backends
    assert_eq!(results.len(), 5);

    // Verify each backend received metrics
    let successful_backends: Vec<_> =
        results.iter().filter(|r| r.success).map(|r| r.backend).collect();

    assert!(successful_backends.contains(&ObservabilityBackend::Datadog));
    assert!(successful_backends.contains(&ObservabilityBackend::NewRelic));
    assert!(successful_backends.contains(&ObservabilityBackend::Honeycomb));
    assert!(successful_backends.contains(&ObservabilityBackend::Otlp));
    assert!(successful_backends.contains(&ObservabilityBackend::Webhook));

    // Verify all backends exported all metrics
    for result in results {
        if result.success {
            assert_eq!(result.metrics_exported, 10);
        }
    }
}

#[test]
fn test_all_health() {
    let config = ObservabilityConfig::new()
        .with_datadog(DatadogConfig::default())
        .with_newrelic(NewRelicConfig::new("key", "account"));

    let exporter = ObservabilityExporter::new(config);

    let health_entries: Vec<_> = exporter.all_health().collect();

    assert_eq!(health_entries.len(), 2);
}
