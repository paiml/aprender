use super::*;

/// Build a [`MetricEvent`] for testing with the given measurement name,
/// tag pairs, and field pairs.
fn make_test_event(
    measurement: &str,
    tags: &[(&str, &str)],
    fields: &[(&str, f64)],
) -> MetricEvent {
    let mut event = MetricEvent::new(measurement);
    for &(k, v) in tags {
        event = event.with_tag(k, v);
    }
    for &(k, v) in fields {
        event = event.with_field(k, v);
    }
    event
}

#[test]
fn test_sink_type_names() {
    assert_eq!(SinkType::InfluxDb.name(), "influxdb");
    assert_eq!(SinkType::JsonLines.name(), "jsonlines");
    assert_eq!(SinkType::Kafka.name(), "kafka");
}

#[test]
fn test_metric_event_creation() {
    let event = make_test_event("cpu_usage", &[("host", "server1")], &[("value", 85.5)]);

    assert_eq!(event.measurement, "cpu_usage");
    assert_eq!(event.tags.get("host"), Some(&"server1".to_string()));
    assert_eq!(event.fields.get("value"), Some(&85.5));
}

#[test]
fn test_influx_line_protocol() {
    let event = make_test_event("cpu", &[("host", "server1")], &[("usage", 85.5)])
        .with_timestamp(1234567890000000000);

    let line = event.to_influx_line();
    assert!(line.starts_with("cpu,host=server1"));
    assert!(line.contains("usage=85.5"));
    assert!(line.ends_with("1234567890000000000"));
}

#[test]
fn test_event_json() {
    let event = make_test_event("test", &[("env", "prod")], &[("value", 42.0)]);

    let json = event.to_json();
    assert!(json.contains("\"measurement\":\"test\""));
    assert!(json.contains("\"env\":\"prod\""));
    assert!(json.contains("\"value\":42"));
}

#[test]
fn test_correlation_id() {
    let event = MetricEvent::new("test").with_correlation_id("trace-123");

    assert_eq!(event.correlation_id, Some("trace-123".to_string()));
    assert!(event.to_json().contains("correlation_id"));
}

#[test]
fn test_event_batch() {
    let mut batch = EventBatch::new(1);
    batch.add(MetricEvent::new("test1"));
    batch.add(MetricEvent::new("test2"));

    assert_eq!(batch.len(), 2);
    assert!(!batch.is_empty());
}

#[test]
fn test_batch_influx_format() {
    let mut batch = EventBatch::new(1);
    batch.add(make_test_event("cpu", &[], &[("value", 50.0)]));
    batch.add(make_test_event("mem", &[], &[("value", 70.0)]));

    let output = batch.to_influx_batch();
    assert!(output.contains("cpu"));
    assert!(output.contains("mem"));
}

#[test]
fn test_streamer_send() {
    let mut streamer = EventStreamer::new(SinkType::JsonLines).with_batch_size(5);

    for i in 0..3 {
        streamer.send(make_test_event("test", &[], &[("i", i as f64)]));
    }

    assert_eq!(streamer.pending_count(), 3);
}

#[test]
fn test_streamer_flush() {
    let mut streamer = EventStreamer::new(SinkType::JsonLines).with_batch_size(2);

    streamer.send(MetricEvent::new("test"));
    streamer.send(MetricEvent::new("test")); // Triggers flush

    assert!(streamer.events_written() >= 2);
}

#[test]
fn test_streamer_health() {
    let streamer = EventStreamer::new(SinkType::Console);

    assert!(streamer.is_healthy());
    assert!(streamer.health().connected);
}

#[test]
fn test_retry_config() {
    let config = RetryConfig::default();

    assert_eq!(config.delay_for_attempt(0), 100);
    assert_eq!(config.delay_for_attempt(1), 200);
    assert_eq!(config.delay_for_attempt(2), 400);
}

#[test]
fn test_correlation_id_generation() {
    let mut streamer = EventStreamer::new(SinkType::Console);

    let id1 = streamer.generate_correlation_id();
    let id2 = streamer.generate_correlation_id();

    assert_ne!(id1, id2);
    assert!(id1.contains("cbtop"));
}

#[test]
fn test_graceful_shutdown() {
    let mut streamer = EventStreamer::new(SinkType::JsonLines);

    streamer.send(MetricEvent::new("test"));
    let result = streamer.shutdown();

    assert!(result);
    assert_eq!(streamer.pending_count(), 0);
}

#[test]
fn test_event_from_sample() {
    let event = event_from_sample(
        "latency_ms",
        42.5,
        &[("host", "server1"), ("region", "us-west")],
    );

    assert_eq!(event.measurement, "latency_ms");
    assert_eq!(event.fields.get("value"), Some(&42.5));
    assert_eq!(event.tags.get("host"), Some(&"server1".to_string()));
}

#[test]
fn test_escape_influx() {
    assert_eq!(escape_influx("hello world"), "hello\\ world");
    assert_eq!(escape_influx("key=value"), "key\\=value");
    assert_eq!(escape_influx("a,b,c"), "a\\,b\\,c");
}
