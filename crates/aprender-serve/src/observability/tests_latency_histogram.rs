
#[test]
fn test_latency_histogram_percentile_invalid() {
    let mut hist = LatencyHistogram::new();
    hist.observe(1000);

    assert!(hist.percentile(-1.0).is_none());
    assert!(hist.percentile(101.0).is_none());
}

#[test]
fn test_latency_histogram_custom_buckets() {
    let buckets = vec![100, 500, 1000, 5000, 10000];
    let mut hist = LatencyHistogram::with_buckets(buckets);

    hist.observe(50); // bucket 0 (<=100)
    hist.observe(200); // bucket 1 (<=500)
    hist.observe(750); // bucket 2 (<=1000)
    hist.observe(20000); // overflow bucket

    assert_eq!(hist.count(), 4);
    assert_eq!(hist.min(), Some(50));
    assert_eq!(hist.max_val(), Some(20000));
}

#[test]
fn test_latency_histogram_to_prometheus() {
    let mut hist = LatencyHistogram::new();
    hist.observe(1000); // 1ms
    hist.observe(50000); // 50ms

    let prom = hist.to_prometheus("request_latency", "service=\"api\"");

    assert!(prom.contains("request_latency_bucket"));
    assert!(prom.contains("le="));
    assert!(prom.contains("service=\"api\""));
    assert!(prom.contains("request_latency_sum"));
    assert!(prom.contains("request_latency_count"));
    assert!(prom.contains("} 2")); // count = 2
}

// =========================================================================
// OpenTelemetry Export Tests
// =========================================================================

#[test]
fn test_span_to_otel_ok() {
    let mut span = Span::new("test-op", "trace123456789012345678901234");
    span.end_ok();

    let otel = span.to_otel();

    assert_eq!(otel.trace_id, "trace123456789012345678901234");
    assert_eq!(otel.operation_name, "test-op");
    assert_eq!(otel.service_name, "realizar");
    assert_eq!(otel.status.code, OtelStatusCode::Ok);
    assert!(otel.status.message.is_none());
}

#[test]
fn test_span_to_otel_error() {
    let mut span = Span::new("failing-op", "trace123456789012345678901234");
    span.end_error("Connection timeout");

    let otel = span.to_otel();

    assert_eq!(otel.status.code, OtelStatusCode::Error);
    assert_eq!(otel.status.message, Some("Connection timeout".to_string()));
}

#[test]
fn test_span_to_otel_with_parent() {
    let parent = Span::new("parent-op", "trace123456789012345678901234");
    let mut child = parent.child("child-op");
    child.end_ok();

    let otel = child.to_otel();

    assert_eq!(otel.parent_span_id, Some(parent.span_id.clone()));
}

#[test]
fn test_span_to_otel_with_attributes() {
    let mut span = Span::new("test-op", "trace123456789012345678901234")
        .with_attribute("model", "llama3")
        .with_attribute("tokens", "256");
    span.end_ok();

    let otel = span.to_otel();

    assert!(otel.attributes.iter().any(|a| a.key == "model"));
    assert!(otel.attributes.iter().any(|a| a.key == "tokens"));
}

#[test]
fn test_span_to_otel_with_kind() {
    let mut span =
        Span::new("server-op", "trace123456789012345678901234").with_kind(SpanKind::Server);
    span.end_ok();

    let otel = span.to_otel();
    assert_eq!(otel.kind, SpanKind::Server);
}

#[test]
fn test_span_to_otel_timestamps() {
    let mut span = Span::new("test-op", "trace123456789012345678901234");
    std::thread::sleep(Duration::from_millis(5));
    span.end_ok();

    let otel = span.to_otel();

    // End time should be >= start time
    assert!(otel.end_time >= otel.start_time);
    // Timestamps should be in nanoseconds
    assert!(otel.start_time > 0);
}

#[test]
fn test_span_trace_context() {
    let span = Span::new("test-op", "0af7651916cd43dd8448eb211c80319c");
    let ctx = span.trace_context();

    assert_eq!(ctx.trace_id, "0af7651916cd43dd8448eb211c80319c");
    assert_eq!(ctx.parent_span_id, Some(span.span_id.clone()));
    assert_eq!(ctx.trace_flags, 0x01);
}

#[test]
fn test_span_traceparent() {
    let span = Span::new("test-op", "0af7651916cd43dd8448eb211c80319c");
    let traceparent = span.traceparent();

    assert!(traceparent.starts_with("00-"));
    assert!(traceparent.contains("0af7651916cd43dd8448eb211c80319c"));
    assert!(traceparent.ends_with("-01"));

    // Should have format: 00-{trace_id}-{span_id}-01
    let parts: Vec<&str> = traceparent.split('-').collect();
    assert_eq!(parts.len(), 4);
    assert_eq!(parts[0], "00");
    assert_eq!(parts[1], "0af7651916cd43dd8448eb211c80319c");
    assert_eq!(parts[3], "01");
}

// =========================================================================
// OtelValue Tests
// =========================================================================

#[test]
fn test_otel_value_from_str() {
    let val: OtelValue = "test".into();
    match val {
        OtelValue::String { string_value } => assert_eq!(string_value, "test"),
        _ => panic!("Expected String variant"),
    }
}

#[test]
fn test_otel_value_from_string() {
    let val: OtelValue = String::from("test").into();
    match val {
        OtelValue::String { string_value } => assert_eq!(string_value, "test"),
        _ => panic!("Expected String variant"),
    }
}

#[test]
fn test_otel_value_from_i64() {
    let val: OtelValue = 42i64.into();
    match val {
        OtelValue::Int { int_value } => assert_eq!(int_value, 42),
        _ => panic!("Expected Int variant"),
    }
}

#[test]
fn test_otel_value_from_f64() {
    let val: OtelValue = 3.14f64.into();
    match val {
        OtelValue::Float { double_value } => assert!((double_value - 3.14).abs() < 0.001),
        _ => panic!("Expected Float variant"),
    }
}

#[test]
fn test_otel_value_from_bool() {
    let val: OtelValue = true.into();
    match val {
        OtelValue::Bool { bool_value } => assert!(bool_value),
        _ => panic!("Expected Bool variant"),
    }
}

// =========================================================================
// SpanKind Tests
// =========================================================================

#[test]
fn test_span_kind_default() {
    let kind = SpanKind::default();
    assert_eq!(kind, SpanKind::Internal);
}

#[test]
fn test_otel_status_code_default() {
    let code = OtelStatusCode::default();
    assert_eq!(code, OtelStatusCode::Unset);
}

// =========================================================================
// Generate Trace ID Tests
// =========================================================================

#[test]
fn test_generate_trace_id_length() {
    let id = generate_trace_id();
    assert_eq!(id.len(), 32); // 16 bytes = 32 hex chars
}

#[test]
fn test_generate_trace_id_hex() {
    let id = generate_trace_id();
    assert!(id.chars().all(|c| c.is_ascii_hexdigit()));
}

// =========================================================================
// Additional Coverage Tests - Edge Cases
// =========================================================================

#[test]
fn test_metric_line_protocol_no_labels() {
    let metric = MetricPoint::new("simple_metric", 100.0);
    let line = metric.to_line_protocol();

    // Should not contain commas (no labels)
    assert!(line.starts_with("simple_metric value=100"));
    // Verify no extra comma before "value"
    assert!(!line.contains(",value"));
}

#[test]
fn test_span_with_parent() {
    let span = Span::new("child-op", "trace-abc").with_parent("parent-span-123");

    assert_eq!(span.parent_id, Some("parent-span-123".to_string()));
}

#[test]
fn test_span_duration() {
    let mut span = Span::new("test", "trace");

    // Before ending, duration should be None
    assert!(span.duration().is_none());

    std::thread::sleep(Duration::from_millis(5));
    span.end_ok();

    // After ending, duration should be Some
    let dur = span.duration().expect("should have duration");
    assert!(dur >= Duration::from_millis(5));
}

#[test]
fn test_span_to_otel_in_progress() {
    let span = Span::new("in-progress-op", "trace123456789012345678901234");
    // Don't end the span - leave it in progress

    let otel = span.to_otel();

    assert_eq!(otel.status.code, OtelStatusCode::Unset);
    assert!(otel.status.message.is_none());
}

#[test]
fn test_span_to_otel_client_kind() {
    let mut span =
        Span::new("client-op", "trace123456789012345678901234").with_kind(SpanKind::Client);
    span.end_ok();

    let otel = span.to_otel();
    assert_eq!(otel.kind, SpanKind::Client);
}

#[test]
fn test_span_to_otel_producer_kind() {
    let mut span =
        Span::new("producer-op", "trace123456789012345678901234").with_kind(SpanKind::Producer);
    span.end_ok();

    let otel = span.to_otel();
    assert_eq!(otel.kind, SpanKind::Producer);
}

#[test]
fn test_span_to_otel_consumer_kind() {
    let mut span =
        Span::new("consumer-op", "trace123456789012345678901234").with_kind(SpanKind::Consumer);
    span.end_ok();

    let otel = span.to_otel();
    assert_eq!(otel.kind, SpanKind::Consumer);
}

#[test]
fn test_span_to_otel_unknown_kind() {
    // Test with an unknown kind string in attributes
    let mut span = Span::new("test-op", "trace123456789012345678901234");
    span.attributes
        .insert("span.kind".to_string(), "Unknown".to_string());
    span.end_ok();

    let otel = span.to_otel();
    assert_eq!(otel.kind, SpanKind::Internal); // Falls back to Internal
}

#[test]
fn test_variant_result_zero_requests() {
    let result = VariantResult::default();

    assert_eq!(result.requests, 0);
    assert!((result.success_rate() - 0.0).abs() < 0.001);
    assert!((result.avg_latency_ms() - 0.0).abs() < 0.001);
    assert!((result.tokens_per_request() - 0.0).abs() < 0.001);
}

#[test]
fn test_ab_test_select_inactive() {
    let mut test = ABTest::new("inactive-test")
        .with_variant("a", "model-a", 0.5)
        .with_variant("b", "model-b", 0.5);
    test.active = false;

    assert!(test.select("user-123").is_none());
}

#[test]
fn test_ab_test_select_empty_variants() {
    let test = ABTest::new("empty-test");

    assert!(test.select("user-123").is_none());
}

#[test]
fn test_ab_test_select_zero_weight() {
    let test = ABTest::new("zero-weight-test")
        .with_variant("a", "model-a", 0.0)
        .with_variant("b", "model-b", 0.0);

    // With zero total weight, should return first variant
    let variant = test.select("user-123");
    assert!(variant.is_some());
    assert_eq!(variant.unwrap().name, "a");
}

#[test]
fn test_ab_test_is_valid_empty() {
    let test = ABTest::new("empty-test");
    assert!(!test.is_valid());
}

#[test]
fn test_ab_test_weight_clamping() {
    let test = ABTest::new("clamped-test")
        .with_variant("a", "model-a", 1.5) // Should clamp to 1.0
        .with_variant("b", "model-b", -0.5); // Should clamp to 0.0

    assert!((test.variants[0].weight - 1.0).abs() < 0.001);
    assert!((test.variants[1].weight - 0.0).abs() < 0.001);
}

#[test]
fn test_observer_ab_testing_disabled() {
    let config = ObservabilityConfig {
        ab_testing_enabled: false,
        ..ObservabilityConfig::new()
    };
    let observer = Observer::new(config);

    observer.record_ab_result("test", "control", true, 50, 100);

    // Results should not be recorded
    assert!(observer.get_ab_results("test").is_none());
}

#[test]
fn test_observability_config_flush_interval() {
    let config = ObservabilityConfig::new().with_flush_interval(120);

    assert_eq!(config.flush_interval_secs, 120);
}

#[test]
fn test_observability_config_sample_rate_clamping() {
    let config_high = ObservabilityConfig::new().with_sample_rate(2.0);
    assert!((config_high.trace_sample_rate - 1.0).abs() < 0.001);

    let config_low = ObservabilityConfig::new().with_sample_rate(-0.5);
    assert!((config_low.trace_sample_rate - 0.0).abs() < 0.001);
}

#[test]
fn test_latency_histogram_overflow_bucket() {
    let mut hist = LatencyHistogram::new();
    // Add a value larger than all buckets (60s = 60_000_000 us)
    hist.observe(100_000_000); // 100s - definitely in overflow

    assert_eq!(hist.count(), 1);
    assert_eq!(hist.max_val(), Some(100_000_000));

    // p100 should return max value (overflow bucket)
    let p100 = hist.percentile(100.0);
    assert_eq!(p100, Some(100_000_000));
}

#[test]
fn test_latency_histogram_default() {
    let hist = LatencyHistogram::default();
    assert_eq!(hist.count(), 0);
}

#[test]
fn test_latency_histogram_with_buckets_unsorted() {
    // Buckets should be sorted internally
    let buckets = vec![1000, 100, 500, 200];
    let mut hist = LatencyHistogram::with_buckets(buckets);

    hist.observe(150);
    hist.observe(350);

    assert_eq!(hist.count(), 2);
}

#[test]
fn test_observer_prometheus_multiple_metrics_same_name() {
    let observer = Observer::default_observer();

    observer
        .record_metric(MetricPoint::new("request_count", 10.0).with_label("endpoint", "/api/v1"));
    observer
        .record_metric(MetricPoint::new("request_count", 20.0).with_label("endpoint", "/api/v2"));

    let prom = observer.prometheus_metrics();

    // Should have TYPE header
    assert!(prom.contains("# TYPE request_count gauge"));
    // Should have both metrics
    assert!(prom.contains("endpoint=\"/api/v1\""));
    assert!(prom.contains("endpoint=\"/api/v2\""));
}
