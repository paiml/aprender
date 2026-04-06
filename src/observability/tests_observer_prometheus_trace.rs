
#[test]
fn test_observer_prometheus_no_labels() {
    let observer = Observer::default_observer();
    observer.record_metric(MetricPoint::new("simple_count", 5.0));

    let prom = observer.prometheus_metrics();

    assert!(prom.contains("simple_count 5"));
}

#[test]
fn test_trace_context_from_traceparent_invalid_hex_flags() {
    // Invalid hex in flags field
    let header = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-GG";
    assert!(TraceContext::from_traceparent(header).is_none());
}

#[test]
fn test_trace_context_from_traceparent_wrong_trace_id_length() {
    // trace_id too short (only 16 chars instead of 32)
    let header = "00-0af7651916cd43dd-b7ad6b7169203331-01";
    assert!(TraceContext::from_traceparent(header).is_none());
}

#[test]
fn test_trace_context_from_traceparent_wrong_span_id_length() {
    // span_id too short (only 8 chars instead of 16)
    let header = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b71-01";
    assert!(TraceContext::from_traceparent(header).is_none());
}

#[test]
fn test_otel_span_serialization() {
    let mut span =
        Span::new("test-op", "0af7651916cd43dd8448eb211c80319c").with_attribute("key", "value");
    span.end_ok();

    let otel = span.to_otel();

    // Should be serializable to JSON
    let json = serde_json::to_string(&otel).expect("should serialize");
    assert!(json.contains("traceId"));
    assert!(json.contains("spanId"));
    assert!(json.contains("operationName"));
}

#[test]
fn test_ab_test_result_default() {
    let result = ABTestResult::default();
    assert!(result.test_name.is_empty());
    assert!(result.variants.is_empty());
}

#[test]
fn test_metric_point_with_multiple_labels() {
    let metric = MetricPoint::new("multi_label", 42.0)
        .with_label("region", "us-east")
        .with_label("env", "prod")
        .with_label("version", "1.0");

    let line = metric.to_line_protocol();

    assert!(line.contains("region=us-east"));
    assert!(line.contains("env=prod"));
    assert!(line.contains("version=1.0"));
}

#[test]
fn test_observer_flush_metrics_clears_buffer() {
    let observer = Observer::default_observer();

    observer.record_metric(MetricPoint::new("test", 1.0));
    let first_flush = observer.flush_metrics();
    assert_eq!(first_flush.len(), 1);

    // Second flush should return empty
    let second_flush = observer.flush_metrics();
    assert!(second_flush.is_empty());
}

#[test]
fn test_observer_flush_spans_clears_buffer() {
    let observer = Observer::default_observer();

    let mut span = observer.start_trace("test-op");
    span.end_ok();
    observer.record_span(span);

    let first_flush = observer.flush_spans();
    assert_eq!(first_flush.len(), 1);

    // Second flush should return empty
    let second_flush = observer.flush_spans();
    assert!(second_flush.is_empty());
}

#[test]
fn test_ab_test_select_last_variant() {
    // Test that selection can hit the last variant
    let test = ABTest::new("test")
        .with_variant("a", "model-a", 0.1)
        .with_variant("b", "model-b", 0.9);

    // With many users, we should hit both variants
    let mut hit_b = false;
    for i in 0..100 {
        let user_id = format!("user-{i}");
        if let Some(variant) = test.select(&user_id) {
            if variant.name == "b" {
                hit_b = true;
                break;
            }
        }
    }
    assert!(hit_b, "Should hit variant b with 90% weight");
}

#[test]
fn test_latency_histogram_single_value_percentiles() {
    let mut hist = LatencyHistogram::new();
    hist.observe(5000); // 5ms

    // All percentiles should return the same bucket
    assert!(hist.p50().is_some());
    assert!(hist.p95().is_some());
    assert!(hist.p99().is_some());
}

#[test]
fn test_span_with_attribute_chain() {
    let span = Span::new("test", "trace")
        .with_attribute("a", "1")
        .with_attribute("b", "2")
        .with_attribute("c", "3");

    assert_eq!(span.attributes.len(), 3);
    assert_eq!(span.attributes.get("a"), Some(&"1".to_string()));
    assert_eq!(span.attributes.get("b"), Some(&"2".to_string()));
    assert_eq!(span.attributes.get("c"), Some(&"3".to_string()));
}

#[test]
fn test_trace_context_set_sampled_preserves_other_flags() {
    let mut ctx = TraceContext {
        trace_id: "test".to_string(),
        parent_span_id: None,
        trace_flags: 0xFF, // All flags set
        trace_state: None,
    };

    // Setting sampled to false should only clear bit 0
    ctx.set_sampled(false);
    assert_eq!(ctx.trace_flags, 0xFE);

    // Setting sampled to true should only set bit 0
    ctx.set_sampled(true);
    assert_eq!(ctx.trace_flags, 0xFF);
}

#[test]
fn test_simple_hash_empty_string() {
    let hash = simple_hash("");
    // Should not panic and should return a consistent value
    let hash2 = simple_hash("");
    assert_eq!(hash, hash2);
}

#[test]
fn test_generate_id_format() {
    let id = generate_id();
    // Should be 16 hex characters
    assert_eq!(id.len(), 16);
    assert!(id.chars().all(|c| c.is_ascii_hexdigit()));
}

// =========================================================================
// Additional Edge Case Tests for Coverage
// =========================================================================

#[test]
fn test_latency_histogram_percentile_boundary() {
    // Test percentile at exact boundary values
    let mut hist = LatencyHistogram::new();
    hist.observe(1000);

    // p0 should work
    let p0 = hist.percentile(0.0);
    assert!(p0.is_some());

    // p100 should work
    let p100 = hist.percentile(100.0);
    assert!(p100.is_some());
}

#[test]
fn test_latency_histogram_to_prometheus_empty_labels() {
    let mut hist = LatencyHistogram::new();
    hist.observe(5000);

    // Empty labels string
    let prom = hist.to_prometheus("latency", "");
    assert!(prom.contains("latency_bucket"));
    assert!(prom.contains("le=\"+Inf\""));
}

#[test]
fn test_latency_histogram_min_max_update() {
    let mut hist = LatencyHistogram::new();

    // First observation sets both min and max
    hist.observe(5000);
    assert_eq!(hist.min(), Some(5000));
    assert_eq!(hist.max_val(), Some(5000));

    // Lower value updates min
    hist.observe(1000);
    assert_eq!(hist.min(), Some(1000));
    assert_eq!(hist.max_val(), Some(5000));

    // Higher value updates max
    hist.observe(10000);
    assert_eq!(hist.min(), Some(1000));
    assert_eq!(hist.max_val(), Some(10000));
}

#[test]
fn test_span_to_otel_internal_kind_default() {
    // Span without explicit kind should be Internal
    let mut span = Span::new("test-op", "trace123456789012345678901234");
    span.end_ok();

    let otel = span.to_otel();
    assert_eq!(otel.kind, SpanKind::Internal);
}

#[test]
fn test_otel_attribute_construction() {
    let attr = OtelAttribute {
        key: "test_key".to_string(),
        value: OtelValue::from("test_value"),
    };

    assert_eq!(attr.key, "test_key");
    match attr.value {
        OtelValue::String { string_value } => assert_eq!(string_value, "test_value"),
        _ => panic!("Expected String variant"),
    }
}

#[test]
fn test_otel_status_construction() {
    let status_ok = OtelStatus {
        code: OtelStatusCode::Ok,
        message: None,
    };
    assert_eq!(status_ok.code, OtelStatusCode::Ok);

    let status_err = OtelStatus {
        code: OtelStatusCode::Error,
        message: Some("failed".to_string()),
    };
    assert_eq!(status_err.message, Some("failed".to_string()));
}

#[test]
fn test_ab_variant_construction() {
    let variant = ABVariant {
        name: "control".to_string(),
        model: "model-v1".to_string(),
        weight: 0.5,
    };

    assert_eq!(variant.name, "control");
    assert_eq!(variant.model, "model-v1");
    assert!((variant.weight - 0.5).abs() < 0.001);
}

#[test]
fn test_observer_default_trait() {
    let observer = Observer::default();
    let id = observer.next_request_id();
    assert_eq!(id, 0);
}

#[test]
fn test_observer_record_metric_multiple() {
    let observer = Observer::default_observer();

    // Record multiple metrics of different types
    observer.record_metric(MetricPoint::new("counter", 1.0));
    observer.record_metric(MetricPoint::new("gauge", 50.0));
    observer.record_metric(MetricPoint::new("histogram", 100.0));

    let metrics = observer.flush_metrics();
    assert_eq!(metrics.len(), 3);
}

#[test]
fn test_ab_test_result_construction() {
    let mut result = ABTestResult {
        test_name: "my-test".to_string(),
        variants: HashMap::new(),
    };

    result.variants.insert(
        "control".to_string(),
        VariantResult {
            requests: 10,
            successes: 8,
            total_latency_ms: 500,
            total_tokens: 1000,
        },
    );

    assert_eq!(result.test_name, "my-test");
    assert_eq!(result.variants.len(), 1);
    assert!(result.variants.contains_key("control"));
}

#[test]
fn test_serde_trace_context_roundtrip() {
    let ctx = TraceContext {
        trace_id: "0af7651916cd43dd8448eb211c80319c".to_string(),
        parent_span_id: Some("b7ad6b7169203331".to_string()),
        trace_flags: 0x01,
        trace_state: Some("vendor=value".to_string()),
    };

    let json = serde_json::to_string(&ctx).expect("serialize");
    let ctx2: TraceContext = serde_json::from_str(&json).expect("deserialize");

    assert_eq!(ctx.trace_id, ctx2.trace_id);
    assert_eq!(ctx.parent_span_id, ctx2.parent_span_id);
    assert_eq!(ctx.trace_flags, ctx2.trace_flags);
    assert_eq!(ctx.trace_state, ctx2.trace_state);
}

#[test]
fn test_serde_latency_histogram_roundtrip() {
    let mut hist = LatencyHistogram::new();
    hist.observe(1000);
    hist.observe(5000);

    let json = serde_json::to_string(&hist).expect("serialize");
    let hist2: LatencyHistogram = serde_json::from_str(&json).expect("deserialize");

    assert_eq!(hist.count(), hist2.count());
    assert_eq!(hist.min(), hist2.min());
    assert_eq!(hist.max_val(), hist2.max_val());
}

#[test]
fn test_serde_metric_point_roundtrip() {
    let metric = MetricPoint::new("test_metric", 42.5).with_label("env", "prod");

    let json = serde_json::to_string(&metric).expect("serialize");
    let metric2: MetricPoint = serde_json::from_str(&json).expect("deserialize");

    assert_eq!(metric.name, metric2.name);
    assert!((metric.value - metric2.value).abs() < 0.001);
    assert_eq!(metric.labels.get("env"), metric2.labels.get("env"));
}

#[test]
fn test_serde_span_roundtrip() {
    let mut span = Span::new("test-op", "trace-123").with_attribute("key", "value");
    span.end_ok();

    let json = serde_json::to_string(&span).expect("serialize");
    let span2: Span = serde_json::from_str(&json).expect("deserialize");

    assert_eq!(span.trace_id, span2.trace_id);
    assert_eq!(span.operation, span2.operation);
    assert_eq!(span.status, span2.status);
}

#[test]
fn test_serde_ab_test_roundtrip() {
    let test = ABTest::new("test")
        .with_variant("a", "model-a", 0.5)
        .with_variant("b", "model-b", 0.5);

    let json = serde_json::to_string(&test).expect("serialize");
    let test2: ABTest = serde_json::from_str(&json).expect("deserialize");

    assert_eq!(test.name, test2.name);
    assert_eq!(test.variants.len(), test2.variants.len());
}

#[test]
fn test_serde_span_kind_roundtrip() {
    let kinds = vec![
        SpanKind::Internal,
        SpanKind::Server,
        SpanKind::Client,
        SpanKind::Producer,
        SpanKind::Consumer,
    ];

    for kind in kinds {
        let json = serde_json::to_string(&kind).expect("serialize");
        let kind2: SpanKind = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(kind, kind2);
    }
}

#[test]
fn test_serde_otel_status_code_roundtrip() {
    let codes = vec![
        OtelStatusCode::Unset,
        OtelStatusCode::Ok,
        OtelStatusCode::Error,
    ];

    for code in codes {
        let json = serde_json::to_string(&code).expect("serialize");
        let code2: OtelStatusCode = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(code, code2);
    }
}

#[test]
fn test_serde_span_status_roundtrip() {
    let statuses = vec![SpanStatus::InProgress, SpanStatus::Ok, SpanStatus::Error];

    for status in statuses {
        let json = serde_json::to_string(&status).expect("serialize");
        let status2: SpanStatus = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(status, status2);
    }
}
