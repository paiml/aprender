
#[test]
fn test_serde_otel_value_variants() {
    // String variant
    let s: OtelValue = "test".into();
    let json = serde_json::to_string(&s).expect("serialize");
    assert!(json.contains("string_value"));

    // Int variant
    let i: OtelValue = 42i64.into();
    let json = serde_json::to_string(&i).expect("serialize");
    assert!(json.contains("int_value"));

    // Float variant
    let f: OtelValue = 3.14f64.into();
    let json = serde_json::to_string(&f).expect("serialize");
    assert!(json.contains("double_value"));

    // Bool variant
    let b: OtelValue = true.into();
    let json = serde_json::to_string(&b).expect("serialize");
    assert!(json.contains("bool_value"));
}

#[test]
fn test_observability_config_default_values() {
    let config = ObservabilityConfig::default();

    assert!(config.trueno_db_uri.is_none());
    assert!(!config.tracing_enabled); // Default is false
    assert!((config.trace_sample_rate - 0.0).abs() < 0.001);
    assert_eq!(config.flush_interval_secs, 0);
    assert!(!config.ab_testing_enabled);
}

#[test]
fn test_concurrent_request_id_generation() {
    use std::sync::Arc;
    use std::thread;

    let observer = Arc::new(Observer::default_observer());
    let mut handles = vec![];

    // Spawn multiple threads to get request IDs
    for _ in 0..10 {
        let obs = Arc::clone(&observer);
        handles.push(thread::spawn(move || obs.next_request_id()));
    }

    let mut ids: Vec<u64> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    ids.sort_unstable();

    // All IDs should be unique (0-9)
    for (i, id) in ids.iter().enumerate() {
        assert_eq!(*id, i as u64);
    }
}

#[test]
fn test_concurrent_metric_recording() {
    use std::sync::Arc;
    use std::thread;

    let observer = Arc::new(Observer::default_observer());
    let mut handles = vec![];

    for i in 0..5 {
        let obs = Arc::clone(&observer);
        handles.push(thread::spawn(move || {
            obs.record_metric(MetricPoint::new(format!("metric_{i}"), i as f64));
        }));
    }

    for h in handles {
        h.join().unwrap();
    }

    let metrics = observer.flush_metrics();
    assert_eq!(metrics.len(), 5);
}

#[test]
fn test_concurrent_span_recording() {
    use std::sync::Arc;
    use std::thread;

    let observer = Arc::new(Observer::default_observer());
    let mut handles = vec![];

    for i in 0..5 {
        let obs = Arc::clone(&observer);
        handles.push(thread::spawn(move || {
            let mut span = obs.start_trace(&format!("op-{i}"));
            span.end_ok();
            obs.record_span(span);
        }));
    }

    for h in handles {
        h.join().unwrap();
    }

    let spans = observer.flush_spans();
    assert_eq!(spans.len(), 5);
}

#[test]
fn test_observability_config_serde_roundtrip() {
    let config = ObservabilityConfig::new()
        .with_trueno_db("trueno://localhost")
        .with_tracing(true)
        .with_sample_rate(0.75)
        .with_flush_interval(30);

    let json = serde_json::to_string(&config).expect("serialize");
    let config2: ObservabilityConfig = serde_json::from_str(&json).expect("deserialize");

    assert_eq!(config.trueno_db_uri, config2.trueno_db_uri);
    assert_eq!(config.tracing_enabled, config2.tracing_enabled);
    assert!((config.trace_sample_rate - config2.trace_sample_rate).abs() < 0.001);
    assert_eq!(config.flush_interval_secs, config2.flush_interval_secs);
}

#[test]
fn test_metric_point_timestamp_is_set() {
    let metric = MetricPoint::new("test", 1.0);
    // Timestamp should be non-zero (set to current time)
    assert!(metric.timestamp > 0);
}

#[test]
fn test_span_start_time_is_set() {
    let span = Span::new("test", "trace-id");
    // Start time should be non-zero
    assert!(span.start_time > 0);
}

#[test]
fn test_ab_test_start_time_is_set() {
    let test = ABTest::new("test");
    // Start time should be non-zero
    assert!(test.start_time > 0);
}

#[test]
fn test_latency_histogram_with_empty_buckets() {
    let hist = LatencyHistogram::with_buckets(vec![]);

    // Empty buckets - only overflow bucket exists
    assert_eq!(hist.buckets.len(), 0);
    // counts should have 1 element (overflow bucket)
    assert_eq!(hist.counts.len(), 1);
}

#[test]
fn test_latency_histogram_observe_into_first_bucket() {
    let mut hist = LatencyHistogram::with_buckets(vec![1000, 5000, 10000]);

    // Value exactly at first bucket boundary
    hist.observe(1000);
    assert_eq!(hist.count(), 1);

    // p50 should return the first bucket
    let p50 = hist.p50();
    assert_eq!(p50, Some(1000));
}

#[test]
fn test_span_service_name_default() {
    let span = Span::new("op", "trace");
    assert_eq!(span.service, "realizar");
}

#[test]
fn test_otel_span_timestamps_in_nanoseconds() {
    let mut span = Span::new("test", "trace123456789012345678901234");
    span.end_ok();

    let otel = span.to_otel();

    // OTel timestamps should be in nanoseconds (start_time * 1000 from microseconds)
    // They should be much larger than the original microsecond values
    assert!(otel.start_time > span.start_time);
}

#[test]
fn test_observer_get_ab_results_missing() {
    let observer = Observer::default_observer();
    assert!(observer.get_ab_results("nonexistent").is_none());
}

#[test]
fn test_ab_test_select_full_weight_first_variant() {
    let test = ABTest::new("test").with_variant("only", "model-only", 1.0);

    // Should always select the only variant
    for i in 0..10 {
        let variant = test.select(&format!("user-{i}"));
        assert!(variant.is_some());
        assert_eq!(variant.unwrap().name, "only");
    }
}
