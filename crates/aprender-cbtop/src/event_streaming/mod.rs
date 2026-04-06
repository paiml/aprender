//! Structured Event Streaming (PMAT-043)
//!
//! Stream metrics to time-series databases and event systems.
//!
//! # Features
//!
//! - InfluxDB Line Protocol export
//! - JSON Lines file sink
//! - Batch buffering and compression
//! - Correlation ID tracking
//!
//! # Falsification Criteria (F1351-F1360)
//!
//! See `tests/event_streaming_f1351.rs` for falsification tests.

use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

/// Schema version for event format
pub const SCHEMA_VERSION: u32 = 1;

/// Return the current wall-clock time as nanoseconds since the Unix epoch.
///
/// Falls back to `0` if the system clock is before the epoch (should never
/// happen in practice).
#[inline]
fn now_nanos() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(0)
}

/// Format a slice of events by applying `formatter` to each element and
/// joining the results with newlines.
fn format_events<T, F: Fn(&T) -> String>(events: &[T], formatter: F) -> String {
    events.iter().map(formatter).collect::<Vec<_>>().join("\n")
}

/// Default batch size
pub const DEFAULT_BATCH_SIZE: usize = 100;

/// Event sink type
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SinkType {
    /// InfluxDB Line Protocol
    InfluxDb,
    /// JSON Lines file
    JsonLines,
    /// Kafka (simulated)
    Kafka,
    /// Console (for testing)
    Console,
}

impl SinkType {
    /// Get sink name
    pub fn name(&self) -> &'static str {
        match self {
            Self::InfluxDb => "influxdb",
            Self::JsonLines => "jsonlines",
            Self::Kafka => "kafka",
            Self::Console => "console",
        }
    }
}

/// Metric event
#[derive(Debug, Clone)]
pub struct MetricEvent {
    /// Measurement name
    pub measurement: String,
    /// Tags (indexed fields)
    pub tags: HashMap<String, String>,
    /// Fields (values)
    pub fields: HashMap<String, f64>,
    /// Timestamp (nanoseconds)
    pub timestamp_ns: u64,
    /// Correlation ID for tracing
    pub correlation_id: Option<String>,
    /// Schema version
    pub schema_version: u32,
}

impl MetricEvent {
    /// Create new event
    pub fn new(measurement: &str) -> Self {
        Self {
            measurement: measurement.to_string(),
            tags: HashMap::new(),
            fields: HashMap::new(),
            timestamp_ns: now_nanos(),
            correlation_id: None,
            schema_version: SCHEMA_VERSION,
        }
    }

    /// Add tag
    pub fn with_tag(mut self, key: &str, value: &str) -> Self {
        self.tags.insert(key.to_string(), value.to_string());
        self
    }

    /// Add field
    pub fn with_field(mut self, key: &str, value: f64) -> Self {
        self.fields.insert(key.to_string(), value);
        self
    }

    /// Set correlation ID
    pub fn with_correlation_id(mut self, id: &str) -> Self {
        self.correlation_id = Some(id.to_string());
        self
    }

    /// Set timestamp
    pub fn with_timestamp(mut self, timestamp_ns: u64) -> Self {
        self.timestamp_ns = timestamp_ns;
        self
    }

    /// Format as InfluxDB Line Protocol
    pub fn to_influx_line(&self) -> String {
        // measurement,tag1=val1,tag2=val2 field1=val1,field2=val2 timestamp
        let mut line = self.measurement.clone();

        // Add tags (sorted for consistency)
        let mut tag_pairs: Vec<_> = self.tags.iter().collect();
        tag_pairs.sort_by_key(|(k, _)| *k);
        for (key, value) in tag_pairs {
            line.push_str(&format!(",{}={}", escape_influx(key), escape_influx(value)));
        }

        // Add fields
        line.push(' ');
        let mut field_pairs: Vec<_> = self.fields.iter().collect();
        field_pairs.sort_by_key(|(k, _)| *k);
        let field_str: Vec<String> =
            field_pairs.iter().map(|(k, v)| format!("{}={}", k, v)).collect();
        line.push_str(&field_str.join(","));

        // Add timestamp
        line.push_str(&format!(" {}", self.timestamp_ns));

        line
    }

    /// Format as JSON
    pub fn to_json(&self) -> String {
        let tags_json: Vec<String> =
            self.tags.iter().map(|(k, v)| format!("\"{}\":\"{}\"", k, v)).collect();

        let fields_json: Vec<String> =
            self.fields.iter().map(|(k, v)| format!("\"{}\":{}", k, v)).collect();

        let correlation = self
            .correlation_id
            .as_ref()
            .map(|id| format!(",\"correlation_id\":\"{}\"", id))
            .unwrap_or_default();

        format!(
            r#"{{"measurement":"{}","tags":{{{}}},"fields":{{{}}},"timestamp_ns":{},"schema_version":{}{}}}"#,
            self.measurement,
            tags_json.join(","),
            fields_json.join(","),
            self.timestamp_ns,
            self.schema_version,
            correlation
        )
    }
}

/// Escape string for InfluxDB Line Protocol
fn escape_influx(s: &str) -> String {
    s.replace(' ', "\\ ").replace(',', "\\,").replace('=', "\\=")
}

/// Event batch
#[derive(Debug, Clone)]
pub struct EventBatch {
    /// Events in batch
    pub events: Vec<MetricEvent>,
    /// Batch ID
    pub batch_id: u64,
    /// Created timestamp
    pub created_ns: u64,
}

impl EventBatch {
    /// Create new batch
    pub fn new(batch_id: u64) -> Self {
        Self { events: Vec::new(), batch_id, created_ns: now_nanos() }
    }

    /// Add event
    pub fn add(&mut self, event: MetricEvent) {
        self.events.push(event);
    }

    /// Get batch size
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// Format as InfluxDB batch
    pub fn to_influx_batch(&self) -> String {
        format_events(&self.events, MetricEvent::to_influx_line)
    }

    /// Format as JSON Lines
    pub fn to_json_lines(&self) -> String {
        format_events(&self.events, MetricEvent::to_json)
    }
}

/// Sink health status
#[derive(Debug, Clone)]
pub struct SinkHealth {
    /// Is connected
    pub connected: bool,
    /// Last successful write timestamp
    pub last_write_ns: Option<u64>,
    /// Events written
    pub events_written: u64,
    /// Write errors
    pub write_errors: u64,
}

impl Default for SinkHealth {
    fn default() -> Self {
        Self { connected: true, last_write_ns: None, events_written: 0, write_errors: 0 }
    }
}

/// Retry configuration
#[derive(Debug, Clone)]
pub struct RetryConfig {
    /// Max retries
    pub max_retries: u32,
    /// Initial delay (ms)
    pub initial_delay_ms: u64,
    /// Max delay (ms)
    pub max_delay_ms: u64,
    /// Backoff multiplier
    pub multiplier: f64,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self { max_retries: 3, initial_delay_ms: 100, max_delay_ms: 10000, multiplier: 2.0 }
    }
}

impl RetryConfig {
    /// Calculate delay for attempt
    pub fn delay_for_attempt(&self, attempt: u32) -> u64 {
        let delay = self.initial_delay_ms as f64 * self.multiplier.powi(attempt as i32);
        (delay as u64).min(self.max_delay_ms)
    }
}

/// Event streamer
#[derive(Debug)]
pub struct EventStreamer {
    /// Sink type
    sink_type: SinkType,
    /// Batch size
    batch_size: usize,
    /// Current batch
    current_batch: EventBatch,
    /// Batch counter
    batch_counter: u64,
    /// Retry config
    retry_config: RetryConfig,
    /// Health status
    health: SinkHealth,
    /// Enable compression
    compression: bool,
    /// Correlation ID generator counter
    correlation_counter: u64,
    /// Output buffer (for file sink)
    output_buffer: Vec<String>,
}

impl Default for EventStreamer {
    fn default() -> Self {
        Self::new(SinkType::Console)
    }
}

impl EventStreamer {
    /// Create new streamer
    pub fn new(sink_type: SinkType) -> Self {
        Self {
            sink_type,
            batch_size: DEFAULT_BATCH_SIZE,
            current_batch: EventBatch::new(0),
            batch_counter: 0,
            retry_config: RetryConfig::default(),
            health: SinkHealth::default(),
            compression: false,
            correlation_counter: 0,
            output_buffer: Vec::new(),
        }
    }

    /// Set batch size
    pub fn with_batch_size(mut self, size: usize) -> Self {
        self.batch_size = size.max(1);
        self
    }

    /// Enable compression
    pub fn with_compression(mut self, enabled: bool) -> Self {
        self.compression = enabled;
        self
    }

    /// Set retry config
    pub fn with_retry(mut self, config: RetryConfig) -> Self {
        self.retry_config = config;
        self
    }

    /// Generate correlation ID
    pub fn generate_correlation_id(&mut self) -> String {
        self.correlation_counter += 1;
        format!("cbtop-{}-{}", std::process::id(), self.correlation_counter)
    }

    /// Send event
    pub fn send(&mut self, event: MetricEvent) -> bool {
        self.current_batch.add(event);

        if self.current_batch.len() >= self.batch_size {
            self.flush()
        } else {
            true
        }
    }

    /// Flush current batch
    pub fn flush(&mut self) -> bool {
        if self.current_batch.is_empty() {
            return true;
        }

        let result = self.write_batch(&self.current_batch.clone());

        if result {
            self.health.events_written += self.current_batch.len() as u64;
            self.health.last_write_ns = Some(now_nanos());

            // Start new batch
            self.batch_counter += 1;
            self.current_batch = EventBatch::new(self.batch_counter);
            true
        } else {
            self.health.write_errors += 1;
            false
        }
    }

    /// Write batch to sink
    fn write_batch(&mut self, batch: &EventBatch) -> bool {
        match self.sink_type {
            SinkType::Console => {
                for event in &batch.events {
                    println!("{}", event.to_json());
                }
                true
            }
            SinkType::InfluxDb => {
                // In production, this would make HTTP request
                // For now, store in buffer
                self.output_buffer.push(batch.to_influx_batch());
                true
            }
            SinkType::JsonLines => {
                self.output_buffer.push(batch.to_json_lines());
                true
            }
            SinkType::Kafka => {
                // Simulated Kafka produce
                self.output_buffer.push(batch.to_json_lines());
                true
            }
        }
    }

    /// Get health status
    pub fn health(&self) -> &SinkHealth {
        &self.health
    }

    /// Check if healthy
    pub fn is_healthy(&self) -> bool {
        self.health.connected && self.health.write_errors == 0
    }

    /// Get output buffer (for testing)
    pub fn output_buffer(&self) -> &[String] {
        &self.output_buffer
    }

    /// Clear output buffer
    pub fn clear_buffer(&mut self) {
        self.output_buffer.clear();
    }

    /// Get events written count
    pub fn events_written(&self) -> u64 {
        self.health.events_written
    }

    /// Get pending events count
    pub fn pending_count(&self) -> usize {
        self.current_batch.len()
    }

    /// Graceful shutdown - flush remaining events
    pub fn shutdown(&mut self) -> bool {
        self.flush()
    }
}

/// Compress data using simple run-length encoding (placeholder)
pub fn compress_data(data: &[u8]) -> Vec<u8> {
    // In production, use flate2 or lz4
    // For now, just return as-is
    data.to_vec()
}

/// Create event from performance sample
pub fn event_from_sample(metric: &str, value: f64, tags: &[(&str, &str)]) -> MetricEvent {
    let mut event = MetricEvent::new(metric).with_field("value", value);

    for (key, val) in tags {
        event = event.with_tag(key, val);
    }

    event
}

#[cfg(test)]
mod tests;
