//! OTLP span receiver for trace waterfall visualization
//!
//! Receives spans from OpenTelemetry export and organizes them
//! for Gantt chart display with critical path highlighting.

use super::{Collector, MetricValue, Metrics};
use anyhow::Result;
use std::collections::{HashMap, VecDeque};
use std::time::Instant;

/// Span kind (per OpenTelemetry spec)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum SpanKind {
    /// Internal operation
    #[default]
    Internal,
    /// Server handling request
    Server,
    /// Client making request
    Client,
    /// Producer sending message
    Producer,
    /// Consumer receiving message
    Consumer,
}

impl SpanKind {
    /// Get display name
    pub fn name(&self) -> &'static str {
        match self {
            Self::Internal => "internal",
            Self::Server => "server",
            Self::Client => "client",
            Self::Producer => "producer",
            Self::Consumer => "consumer",
        }
    }
}

/// Span record for display
#[derive(Debug, Clone)]
pub struct SpanRecord {
    /// Trace ID (128-bit hex string)
    pub trace_id: String,
    /// Span ID (64-bit hex string)
    pub span_id: String,
    /// Parent span ID (if any)
    pub parent_span_id: Option<String>,
    /// Operation name
    pub name: String,
    /// Span kind
    pub kind: SpanKind,
    /// Start time (nanoseconds since epoch)
    pub start_time_ns: u64,
    /// End time (nanoseconds since epoch)
    pub end_time_ns: u64,
    /// Span attributes
    pub attributes: HashMap<String, String>,
    /// Status code (0 = OK, 1 = Error)
    pub status_code: u8,
    /// Status message (for errors)
    pub status_message: Option<String>,
    /// Whether span is on critical path
    pub is_critical_path: bool,
    /// Nesting depth (for display indentation)
    pub depth: usize,
}

impl SpanRecord {
    /// Create new span record
    pub fn new(
        trace_id: &str,
        span_id: &str,
        name: &str,
        start_time_ns: u64,
        end_time_ns: u64,
    ) -> Self {
        Self {
            trace_id: trace_id.to_string(),
            span_id: span_id.to_string(),
            parent_span_id: None,
            name: name.to_string(),
            kind: SpanKind::Internal,
            start_time_ns,
            end_time_ns,
            attributes: HashMap::new(),
            status_code: 0,
            status_message: None,
            is_critical_path: false,
            depth: 0,
        }
    }

    /// Set parent span ID
    pub fn with_parent(mut self, parent_id: &str) -> Self {
        self.parent_span_id = Some(parent_id.to_string());
        self
    }

    /// Set span kind
    pub fn with_kind(mut self, kind: SpanKind) -> Self {
        self.kind = kind;
        self
    }

    /// Add attribute
    pub fn with_attribute(mut self, key: &str, value: &str) -> Self {
        self.attributes.insert(key.to_string(), value.to_string());
        self
    }

    /// Set error status
    pub fn with_error(mut self, message: &str) -> Self {
        self.status_code = 1;
        self.status_message = Some(message.to_string());
        self
    }

    /// Get duration in nanoseconds
    pub fn duration_ns(&self) -> u64 {
        self.end_time_ns.saturating_sub(self.start_time_ns)
    }

    /// Get duration in microseconds
    pub fn duration_us(&self) -> u64 {
        self.duration_ns() / 1000
    }

    /// Get duration in milliseconds
    pub fn duration_ms(&self) -> f64 {
        self.duration_ns() as f64 / 1_000_000.0
    }

    /// Check if span has error
    pub fn is_error(&self) -> bool {
        self.status_code != 0
    }
}

/// OTLP span receiver and organizer
pub struct SpanReceiver {
    /// Recent spans (ring buffer)
    spans: VecDeque<SpanRecord>,
    /// Maximum spans to keep
    max_spans: usize,
    /// Spans by trace ID for tree organization
    traces: HashMap<String, Vec<usize>>,
    /// Total span count
    total_spans: u64,
    /// Error span count
    error_spans: u64,
    /// Average span duration (ms)
    avg_duration_ms: f64,
    /// Duration sum for averaging
    duration_sum: f64,
    /// Whether receiver is available
    available: bool,
    /// Last span receive time
    last_receive: Option<Instant>,
}

impl SpanReceiver {
    /// Create new span receiver
    pub fn new() -> Self {
        Self::with_capacity(100)
    }

    /// Create span receiver with custom capacity
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            spans: VecDeque::with_capacity(capacity),
            max_spans: capacity,
            traces: HashMap::new(),
            total_spans: 0,
            error_spans: 0,
            avg_duration_ms: 0.0,
            duration_sum: 0.0,
            available: true,
            last_receive: None,
        }
    }

    /// Receive a new span
    pub fn receive(&mut self, mut span: SpanRecord) {
        // Update statistics
        self.total_spans += 1;
        if span.is_error() {
            self.error_spans += 1;
        }

        self.duration_sum += span.duration_ms();
        self.avg_duration_ms = self.duration_sum / self.total_spans as f64;
        self.last_receive = Some(Instant::now());

        // Calculate depth based on parent chain
        span.depth = self.calculate_depth(&span);

        // Evict oldest if at capacity
        if self.spans.len() >= self.max_spans {
            if let Some(old) = self.spans.pop_front() {
                // Remove from trace index
                if let Some(indices) = self.traces.get_mut(&old.trace_id) {
                    indices.retain(|&i| i != 0);
                    // Adjust indices
                    for idx in indices.iter_mut() {
                        *idx = idx.saturating_sub(1);
                    }
                }
            }
        }

        // Add to spans and index
        let idx = self.spans.len();
        self.traces.entry(span.trace_id.clone()).or_default().push(idx);
        self.spans.push_back(span);

        // Recalculate critical path for affected trace
        self.update_critical_path();
    }

    /// Calculate nesting depth for a span
    fn calculate_depth(&self, span: &SpanRecord) -> usize {
        let mut depth = 0;
        let mut parent_id = span.parent_span_id.clone();

        while let Some(ref pid) = parent_id {
            if let Some(parent) = self.spans.iter().find(|s| &s.span_id == pid) {
                depth += 1;
                parent_id.clone_from(&parent.parent_span_id);
            } else {
                break;
            }
        }

        depth
    }

    /// Update critical path highlighting
    fn update_critical_path(&mut self) {
        // Simple critical path: longest span at each depth level
        // For each trace, find the span with longest duration
        for indices in self.traces.values() {
            if indices.is_empty() {
                continue;
            }

            // Find max duration span
            let mut max_idx = indices[0];
            let mut max_duration = 0;

            for &idx in indices {
                if idx < self.spans.len() {
                    let duration = self.spans[idx].duration_ns();
                    if duration > max_duration {
                        max_duration = duration;
                        max_idx = idx;
                    }
                }
            }

            // Mark as critical path
            if max_idx < self.spans.len() {
                self.spans[max_idx].is_critical_path = true;
            }
        }
    }

    /// Get all spans
    pub fn spans(&self) -> &VecDeque<SpanRecord> {
        &self.spans
    }

    /// Get spans for a specific trace
    pub fn trace_spans(&self, trace_id: &str) -> Vec<&SpanRecord> {
        self.traces
            .get(trace_id)
            .map(|indices| indices.iter().filter_map(|&i| self.spans.get(i)).collect())
            .unwrap_or_default()
    }

    /// Get total span count
    pub fn total_count(&self) -> u64 {
        self.total_spans
    }

    /// Get error span count
    pub fn error_count(&self) -> u64 {
        self.error_spans
    }

    /// Get average duration in milliseconds
    pub fn avg_duration_ms(&self) -> f64 {
        self.avg_duration_ms
    }

    /// Get time range of current spans (start, end in ns)
    pub fn time_range(&self) -> Option<(u64, u64)> {
        if self.spans.is_empty() {
            return None;
        }

        let min = self.spans.iter().map(|s| s.start_time_ns).min()?;
        let max = self.spans.iter().map(|s| s.end_time_ns).max()?;
        Some((min, max))
    }

    /// Get unique trace IDs
    pub fn trace_ids(&self) -> Vec<&str> {
        self.traces.keys().map(String::as_str).collect()
    }
}

impl Default for SpanReceiver {
    fn default() -> Self {
        Self::new()
    }
}

impl Collector for SpanReceiver {
    fn collect(&mut self) -> Result<Metrics> {
        let mut values = HashMap::new();

        values.insert("span.total.count".to_string(), MetricValue::Counter(self.total_spans));
        values.insert("span.error.count".to_string(), MetricValue::Counter(self.error_spans));
        values.insert("span.avg_duration_ms".to_string(), MetricValue::Gauge(self.avg_duration_ms));
        values.insert("span.active.count".to_string(), MetricValue::Gauge(self.spans.len() as f64));
        values.insert("span.trace.count".to_string(), MetricValue::Gauge(self.traces.len() as f64));

        // Calculate spans by kind
        for kind in [
            SpanKind::Internal,
            SpanKind::Server,
            SpanKind::Client,
            SpanKind::Producer,
            SpanKind::Consumer,
        ] {
            let count = self.spans.iter().filter(|s| s.kind == kind).count();
            values.insert(
                format!("span.kind.{}.count", kind.name()),
                MetricValue::Gauge(count as f64),
            );
        }

        Ok(Metrics::new(values))
    }

    fn is_available(&self) -> bool {
        self.available
    }

    fn name(&self) -> &'static str {
        "span"
    }

    fn reset(&mut self) {
        self.spans.clear();
        self.traces.clear();
        self.total_spans = 0;
        self.error_spans = 0;
        self.avg_duration_ms = 0.0;
        self.duration_sum = 0.0;
        self.last_receive = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_span_kind_name() {
        assert_eq!(SpanKind::Internal.name(), "internal");
        assert_eq!(SpanKind::Server.name(), "server");
        assert_eq!(SpanKind::Client.name(), "client");
    }

    #[test]
    fn test_span_record_new() {
        let span = SpanRecord::new("trace123", "span456", "http.request", 1000, 2000);

        assert_eq!(span.trace_id, "trace123");
        assert_eq!(span.span_id, "span456");
        assert_eq!(span.name, "http.request");
        assert_eq!(span.start_time_ns, 1000);
        assert_eq!(span.end_time_ns, 2000);
        assert_eq!(span.duration_ns(), 1000);
        assert!(!span.is_error());
    }

    #[test]
    fn test_span_record_with_parent() {
        let span = SpanRecord::new("t", "s1", "op", 0, 1000).with_parent("parent_span");
        assert_eq!(span.parent_span_id, Some("parent_span".to_string()));
    }

    #[test]
    fn test_span_record_with_error() {
        let span = SpanRecord::new("t", "s1", "op", 0, 1000).with_error("connection timeout");
        assert!(span.is_error());
        assert_eq!(span.status_message, Some("connection timeout".to_string()));
    }

    #[test]
    fn test_span_record_duration() {
        let span = SpanRecord::new("t", "s1", "op", 1_000_000, 3_000_000);
        assert_eq!(span.duration_ns(), 2_000_000);
        assert_eq!(span.duration_us(), 2_000);
        assert!((span.duration_ms() - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_span_receiver_new() {
        let receiver = SpanReceiver::new();
        assert!(receiver.is_available());
        assert_eq!(receiver.name(), "span");
        assert_eq!(receiver.total_count(), 0);
    }

    #[test]
    fn test_span_receiver_receive() {
        let mut receiver = SpanReceiver::new();

        let span = SpanRecord::new("trace1", "span1", "op", 0, 1_000_000);
        receiver.receive(span);

        assert_eq!(receiver.total_count(), 1);
        assert_eq!(receiver.spans().len(), 1);
        assert!((receiver.avg_duration_ms() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_span_receiver_error_tracking() {
        let mut receiver = SpanReceiver::new();

        receiver.receive(SpanRecord::new("t", "s1", "op", 0, 1000));
        receiver.receive(SpanRecord::new("t", "s2", "op", 0, 1000).with_error("fail"));
        receiver.receive(SpanRecord::new("t", "s3", "op", 0, 1000).with_error("fail"));

        assert_eq!(receiver.total_count(), 3);
        assert_eq!(receiver.error_count(), 2);
    }

    #[test]
    fn test_span_receiver_capacity() {
        let mut receiver = SpanReceiver::with_capacity(5);

        for i in 0..10 {
            let span =
                SpanRecord::new("trace", &format!("span{}", i), "op", i * 1000, (i + 1) * 1000);
            receiver.receive(span);
        }

        assert_eq!(receiver.spans().len(), 5);
        assert_eq!(receiver.total_count(), 10);

        // Should have the last 5 spans
        let first = receiver.spans().front().unwrap();
        assert_eq!(first.span_id, "span5");
    }

    #[test]
    fn test_span_receiver_trace_spans() {
        let mut receiver = SpanReceiver::new();

        receiver.receive(SpanRecord::new("trace1", "s1", "op1", 0, 1000));
        receiver.receive(SpanRecord::new("trace2", "s2", "op2", 0, 1000));
        receiver.receive(SpanRecord::new("trace1", "s3", "op3", 0, 1000));

        let trace1_spans = receiver.trace_spans("trace1");
        assert_eq!(trace1_spans.len(), 2);

        let trace2_spans = receiver.trace_spans("trace2");
        assert_eq!(trace2_spans.len(), 1);
    }

    #[test]
    fn test_span_receiver_time_range() {
        let mut receiver = SpanReceiver::new();

        receiver.receive(SpanRecord::new("t", "s1", "op", 100, 200));
        receiver.receive(SpanRecord::new("t", "s2", "op", 50, 300));
        receiver.receive(SpanRecord::new("t", "s3", "op", 150, 250));

        let (start, end) = receiver.time_range().unwrap();
        assert_eq!(start, 50);
        assert_eq!(end, 300);
    }

    #[test]
    fn test_span_receiver_collect() {
        let mut receiver = SpanReceiver::new();
        receiver.receive(SpanRecord::new("t", "s1", "op", 0, 1000).with_kind(SpanKind::Server));

        let metrics = receiver.collect().unwrap();
        assert!(metrics.values.contains_key("span.total.count"));
        assert!(metrics.values.contains_key("span.kind.server.count"));
    }

    #[test]
    fn test_span_receiver_reset() {
        let mut receiver = SpanReceiver::new();
        receiver.receive(SpanRecord::new("t", "s1", "op", 0, 1000));

        receiver.reset();
        assert!(receiver.spans().is_empty());
        assert_eq!(receiver.total_count(), 0);
        assert!(receiver.traces.is_empty());
    }

    #[test]
    fn test_span_receiver_depth_calculation() {
        let mut receiver = SpanReceiver::new();

        // Root span
        receiver.receive(SpanRecord::new("t", "root", "op", 0, 1000));

        // Child span
        receiver.receive(SpanRecord::new("t", "child", "op", 100, 900).with_parent("root"));

        // Grandchild span
        receiver.receive(SpanRecord::new("t", "grandchild", "op", 200, 800).with_parent("child"));

        assert_eq!(receiver.spans()[0].depth, 0);
        assert_eq!(receiver.spans()[1].depth, 1);
        assert_eq!(receiver.spans()[2].depth, 2);
    }
}
