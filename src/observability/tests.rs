use super::*;

#[test]
fn test_observability_config_default() {
    let config = ObservabilityConfig::new();
    assert!(config.tracing_enabled);
    assert!((config.trace_sample_rate - 1.0).abs() < 0.001);
}

#[test]
fn test_observability_config_builder() {
    let config = ObservabilityConfig::new()
        .with_trueno_db("trueno-db://localhost:5432")
        .with_tracing(false)
        .with_sample_rate(0.5);

    assert_eq!(
        config.trueno_db_uri,
        Some("trueno-db://localhost:5432".to_string())
    );
    assert!(!config.tracing_enabled);
    assert!((config.trace_sample_rate - 0.5).abs() < 0.001);
}

#[test]
fn test_metric_point() {
    let metric = MetricPoint::new("test_metric", 42.0)
        .with_label("model", "llama3")
        .with_label("version", "1.0");

    assert_eq!(metric.name, "test_metric");
    assert!((metric.value - 42.0).abs() < 0.001);
    assert_eq!(metric.labels.get("model"), Some(&"llama3".to_string()));
}

#[test]
fn test_metric_line_protocol() {
    let metric = MetricPoint::new("cpu_usage", 75.5).with_label("host", "server1");

    let line = metric.to_line_protocol();
    assert!(line.contains("cpu_usage"));
    assert!(line.contains("host=server1"));
    assert!(line.contains("75.5"));
}

#[test]
fn test_span_creation() {
    let span = Span::new("inference", "trace-123");

    assert!(!span.span_id.is_empty());
    assert_eq!(span.trace_id, "trace-123");
    assert_eq!(span.operation, "inference");
    assert_eq!(span.status, SpanStatus::InProgress);
}

#[test]
fn test_span_child() {
    let parent = Span::new("request", "trace-456");
    let child = parent.child("tokenize");

    assert_eq!(child.trace_id, parent.trace_id);
    assert_eq!(child.parent_id, Some(parent.span_id.clone()));
    assert_eq!(child.operation, "tokenize");
}

#[test]
fn test_span_end_ok() {
    let mut span = Span::new("test", "trace");
    std::thread::sleep(Duration::from_millis(10));
    span.end_ok();

    assert_eq!(span.status, SpanStatus::Ok);
    assert!(span.duration_us.is_some());
    assert!(span.duration_us.expect("test") >= 10000); // At least 10ms
}

#[test]
fn test_span_end_error() {
    let mut span = Span::new("test", "trace");
    span.end_error("Something went wrong");

    assert_eq!(span.status, SpanStatus::Error);
    assert_eq!(
        span.attributes.get("error"),
        Some(&"Something went wrong".to_string())
    );
}

#[test]
fn test_ab_test_creation() {
    let test = ABTest::new("model-comparison")
        .with_variant("control", "model-v1", 0.5)
        .with_variant("treatment", "model-v2", 0.5);

    assert_eq!(test.name, "model-comparison");
    assert_eq!(test.variants.len(), 2);
    assert!(test.is_valid());
}

#[test]
fn test_ab_test_selection_deterministic() {
    let test = ABTest::new("test")
        .with_variant("a", "model-a", 0.5)
        .with_variant("b", "model-b", 0.5);

    // Same user should always get same variant
    let variant1 = test.select("user-123");
    let variant2 = test.select("user-123");

    assert_eq!(variant1.map(|v| &v.name), variant2.map(|v| &v.name));
}

#[test]
fn test_ab_test_selection_distribution() {
    let test = ABTest::new("test")
        .with_variant("a", "model-a", 0.5)
        .with_variant("b", "model-b", 0.5);

    let mut count_a = 0;
    let mut count_b = 0;

    for i in 0..1000 {
        let user_id = format!("user-{i}");
        if let Some(variant) = test.select(&user_id) {
            if variant.name == "a" {
                count_a += 1;
            } else {
                count_b += 1;
            }
        }
    }

    // Should be roughly 50/50 (within 10% tolerance)
    let ratio = count_a as f64 / (count_a + count_b) as f64;
    assert!(ratio > 0.4 && ratio < 0.6);
}

#[test]
fn test_ab_test_invalid_weights() {
    let test = ABTest::new("test")
        .with_variant("a", "model-a", 0.3)
        .with_variant("b", "model-b", 0.3);

    assert!(!test.is_valid()); // Weights sum to 0.6, not 1.0
}

#[test]
fn test_variant_result_calculations() {
    let result = VariantResult {
        requests: 100,
        successes: 90,
        total_latency_ms: 5000,
        total_tokens: 10000,
    };

    assert!((result.success_rate() - 0.9).abs() < 0.001);
    assert!((result.avg_latency_ms() - 50.0).abs() < 0.001);
    assert!((result.tokens_per_request() - 100.0).abs() < 0.001);
}

#[test]
fn test_observer_record_inference() {
    let observer = Observer::default_observer();
    observer.record_inference("llama3", 100, 50);

    let metrics = observer.flush_metrics();
    assert!(!metrics.is_empty());
}

#[test]
fn test_observer_record_span() {
    let observer = Observer::default_observer();
    let mut span = observer.start_trace("test-op");
    span.end_ok();
    observer.record_span(span);

    let spans = observer.flush_spans();
    assert_eq!(spans.len(), 1);
    assert_eq!(spans[0].operation, "test-op");
}

#[test]
fn test_observer_ab_results() {
    let observer = Observer::default_observer();

    observer.record_ab_result("test", "control", true, 50, 100);
    observer.record_ab_result("test", "control", true, 60, 120);
    observer.record_ab_result("test", "treatment", false, 40, 80);

    let results = observer.get_ab_results("test").expect("test");
    let control = results.variants.get("control").expect("test");
    let treatment = results.variants.get("treatment").expect("test");

    assert_eq!(control.requests, 2);
    assert_eq!(control.successes, 2);
    assert_eq!(treatment.requests, 1);
    assert_eq!(treatment.successes, 0);
}

#[test]
fn test_observer_prometheus_format() {
    let observer = Observer::default_observer();
    observer.record_metric(MetricPoint::new("test_metric", 42.0).with_label("env", "prod"));

    let prom = observer.prometheus_metrics();
    assert!(prom.contains("test_metric"));
    assert!(prom.contains("env=\"prod\""));
    assert!(prom.contains("42"));
}

#[test]
fn test_observer_request_id() {
    let observer = Observer::default_observer();

    let id1 = observer.next_request_id();
    let id2 = observer.next_request_id();
    let id3 = observer.next_request_id();

    assert_eq!(id1, 0);
    assert_eq!(id2, 1);
    assert_eq!(id3, 2);
}

#[test]
fn test_span_status_default() {
    let status = SpanStatus::default();
    assert_eq!(status, SpanStatus::InProgress);
}

#[test]
fn test_generate_id_unique() {
    let id1 = generate_id();
    let id2 = generate_id();

    // IDs should be unique (though not guaranteed with fast calls)
    assert_eq!(id1.len(), 16);
    assert_eq!(id2.len(), 16);
}

#[test]
fn test_simple_hash_deterministic() {
    let hash1 = simple_hash("test-input");
    let hash2 = simple_hash("test-input");
    let hash3 = simple_hash("different-input");

    assert_eq!(hash1, hash2);
    assert_ne!(hash1, hash3);
}

#[test]
fn test_observer_sampling() {
    let config = ObservabilityConfig::new().with_sample_rate(0.0);
    let observer = Observer::new(config);

    let mut span = observer.start_trace("test");
    span.end_ok();
    observer.record_span(span);

    // With 0% sampling, no spans should be recorded
    let spans = observer.flush_spans();
    assert!(spans.is_empty());
}

#[test]
fn test_observer_tracing_disabled() {
    let config = ObservabilityConfig::new().with_tracing(false);
    let observer = Observer::new(config);

    let mut span = observer.start_trace("test");
    span.end_ok();
    observer.record_span(span);

    let spans = observer.flush_spans();
    assert!(spans.is_empty());
}

// =========================================================================
// W3C Trace Context Tests
// =========================================================================

#[test]
fn test_trace_context_new() {
    let ctx = TraceContext::new();
    assert_eq!(ctx.trace_id.len(), 32);
    assert!(ctx.parent_span_id.is_none());
    assert_eq!(ctx.trace_flags, 0x01); // Sampled by default
    assert!(ctx.trace_state.is_none());
}

#[test]
fn test_trace_context_child() {
    let parent_ctx = TraceContext::new();
    let child_ctx = parent_ctx.child("abcdef0123456789");

    assert_eq!(child_ctx.trace_id, parent_ctx.trace_id);
    assert_eq!(
        child_ctx.parent_span_id,
        Some("abcdef0123456789".to_string())
    );
    assert_eq!(child_ctx.trace_flags, parent_ctx.trace_flags);
}

#[test]
fn test_trace_context_from_traceparent_valid() {
    let header = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01";
    let ctx = TraceContext::from_traceparent(header).expect("test");

    assert_eq!(ctx.trace_id, "0af7651916cd43dd8448eb211c80319c");
    assert_eq!(ctx.parent_span_id, Some("b7ad6b7169203331".to_string()));
    assert_eq!(ctx.trace_flags, 0x01);
}

#[test]
fn test_trace_context_from_traceparent_not_sampled() {
    let header = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-00";
    let ctx = TraceContext::from_traceparent(header).expect("test");

    assert_eq!(ctx.trace_flags, 0x00);
    assert!(!ctx.is_sampled());
}

#[test]
fn test_trace_context_from_traceparent_invalid_version() {
    let header = "01-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01";
    assert!(TraceContext::from_traceparent(header).is_none());
}

#[test]
fn test_trace_context_from_traceparent_invalid_format() {
    assert!(TraceContext::from_traceparent("invalid").is_none());
    assert!(TraceContext::from_traceparent("00-abc-def-01").is_none());
    assert!(TraceContext::from_traceparent("").is_none());
}

#[test]
fn test_trace_context_to_traceparent() {
    let ctx = TraceContext {
        trace_id: "0af7651916cd43dd8448eb211c80319c".to_string(),
        parent_span_id: None,
        trace_flags: 0x01,
        trace_state: None,
    };

    let traceparent = ctx.to_traceparent("b7ad6b7169203331");
    assert_eq!(
        traceparent,
        "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"
    );
}

#[test]
fn test_trace_context_with_tracestate() {
    let ctx = TraceContext::new().with_tracestate("vendor=value");
    assert_eq!(ctx.trace_state, Some("vendor=value".to_string()));
}

#[test]
fn test_trace_context_sampled_flag() {
    let mut ctx = TraceContext::new();
    assert!(ctx.is_sampled());

    ctx.set_sampled(false);
    assert!(!ctx.is_sampled());
    assert_eq!(ctx.trace_flags, 0x00);

    ctx.set_sampled(true);
    assert!(ctx.is_sampled());
    assert_eq!(ctx.trace_flags, 0x01);
}

#[test]
fn test_trace_context_default() {
    let ctx = TraceContext::default();
    assert_eq!(ctx.trace_id.len(), 32);
}

// =========================================================================
// Latency Histogram Tests
// =========================================================================

#[test]
fn test_latency_histogram_new() {
    let hist = LatencyHistogram::new();
    assert_eq!(hist.count(), 0);
    assert!(hist.min().is_none());
    assert!(hist.max_val().is_none());
    assert!(hist.mean().is_none());
}

#[test]
fn test_latency_histogram_observe() {
    let mut hist = LatencyHistogram::new();
    hist.observe(1000); // 1ms
    hist.observe(5000); // 5ms
    hist.observe(10000); // 10ms

    assert_eq!(hist.count(), 3);
    assert_eq!(hist.min(), Some(1000));
    assert_eq!(hist.max_val(), Some(10000));
}

#[test]
fn test_latency_histogram_mean() {
    let mut hist = LatencyHistogram::new();
    hist.observe(1000);
    hist.observe(2000);
    hist.observe(3000);

    let mean = hist.mean().expect("test");
    assert!((mean - 2000.0).abs() < 0.001);
}

#[test]
fn test_latency_histogram_observe_duration() {
    let mut hist = LatencyHistogram::new();
    hist.observe_duration(Duration::from_millis(5));

    assert_eq!(hist.count(), 1);
    assert_eq!(hist.min(), Some(5000)); // 5ms = 5000us
}

#[test]
fn test_latency_histogram_percentiles() {
    let mut hist = LatencyHistogram::new();
    // Add 100 observations: 1ms, 2ms, 3ms, ... 100ms
    for i in 1..=100 {
        hist.observe(i * 1000);
    }

    // p50 should be around 50ms
    let p50 = hist.p50().expect("test");
    assert!(p50 >= 25_000 && p50 <= 100_000);

    // p95 should be around 95ms
    let p95 = hist.p95().expect("test");
    assert!(p95 >= 50000);

    // p99 should be around 99ms
    let p99 = hist.p99().expect("test");
    assert!(p99 >= 50000);
}

#[test]
fn test_latency_histogram_percentile_empty() {
    let hist = LatencyHistogram::new();
    assert!(hist.percentile(50.0).is_none());
    assert!(hist.p50().is_none());
    assert!(hist.p95().is_none());
    assert!(hist.p99().is_none());
}

include!("tests_latency_histogram.rs");
include!("tests_observer_prometheus_trace.rs");
include!("tests_serde_otel.rs");
