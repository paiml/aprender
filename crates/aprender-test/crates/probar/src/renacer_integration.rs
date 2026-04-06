//! Renacer Integration for Probar (Issue #9)
//!
//! Deep WASM tracing integration with renacer for full E2E observability.
//! Enables trace context propagation across JS→WASM boundaries.
//!
//! ## Features
//!
//! - W3C Trace Context propagation to browser
//! - Unified trace output (browser + WASM spans)
//! - Chrome trace format export (chrome://tracing compatible)
//! - Correlation of console messages with trace spans
//!
//! ## Usage
//!
//! ```ignore
//! use probar::{Browser, BrowserConfig, TracingConfig};
//!
//! let config = BrowserConfig::default()
//!     .with_tracing(TracingConfig::default());
//!
//! let browser = Browser::launch(config).await?;
//! let mut page = browser.new_page().await?;
//!
//! // Trace context is automatically injected
//! page.goto("http://localhost:8080").await?;
//!
//! // Export trace in Chrome format
//! let trace = page.export_chrome_trace().await?;
//! std::fs::write("trace.json", trace)?;
//! ```

use std::time::{SystemTime, UNIX_EPOCH};

/// Tracing configuration for renacer integration
#[derive(Debug, Clone)]
pub struct TracingConfig {
    /// Enable tracing (default: true)
    pub enabled: bool,
    /// Service name for traces
    pub service_name: String,
    /// Generate trace ID (default: random)
    pub trace_id: Option<String>,
    /// Sample rate (0.0 - 1.0, default: 1.0)
    pub sample_rate: f64,
    /// Include console messages in trace
    pub capture_console: bool,
    /// Include network requests in trace
    pub capture_network: bool,
    /// Include user interactions in trace
    pub capture_interactions: bool,
}

impl Default for TracingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            service_name: "probar-browser-test".to_string(),
            trace_id: None,
            sample_rate: 1.0,
            capture_console: true,
            capture_network: true,
            capture_interactions: true,
        }
    }
}

impl TracingConfig {
    /// Create a new tracing configuration
    #[must_use]
    pub fn new(service_name: impl Into<String>) -> Self {
        Self {
            service_name: service_name.into(),
            ..Default::default()
        }
    }

    /// Disable tracing
    #[must_use]
    pub const fn disabled() -> Self {
        Self {
            enabled: false,
            service_name: String::new(),
            trace_id: None,
            sample_rate: 0.0,
            capture_console: false,
            capture_network: false,
            capture_interactions: false,
        }
    }

    /// Set service name
    #[must_use]
    pub fn with_service_name(mut self, name: impl Into<String>) -> Self {
        self.service_name = name.into();
        self
    }

    /// Set trace ID
    #[must_use]
    pub fn with_trace_id(mut self, id: impl Into<String>) -> Self {
        self.trace_id = Some(id.into());
        self
    }

    /// Set sample rate
    #[must_use]
    pub fn with_sample_rate(mut self, rate: f64) -> Self {
        self.sample_rate = rate.clamp(0.0, 1.0);
        self
    }

    /// Enable/disable console capture
    #[must_use]
    pub const fn with_console_capture(mut self, enabled: bool) -> Self {
        self.capture_console = enabled;
        self
    }

    /// Enable/disable network capture
    #[must_use]
    pub const fn with_network_capture(mut self, enabled: bool) -> Self {
        self.capture_network = enabled;
        self
    }
}

/// W3C Trace Context for distributed tracing
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TraceContext {
    /// 128-bit trace ID (32 hex chars)
    pub trace_id: String,
    /// 64-bit parent span ID (16 hex chars)
    pub parent_id: String,
    /// Trace flags (sampled = 01)
    pub flags: u8,
}

impl TraceContext {
    /// Generate a new trace context with random IDs
    #[must_use]
    pub fn new() -> Self {
        let trace_id = generate_trace_id();
        let parent_id = generate_span_id();
        Self {
            trace_id,
            parent_id,
            flags: 0x01, // sampled
        }
    }

    /// Create from existing trace ID
    #[must_use]
    pub fn with_trace_id(trace_id: impl Into<String>) -> Self {
        Self {
            trace_id: trace_id.into(),
            parent_id: generate_span_id(),
            flags: 0x01,
        }
    }

    /// Parse from W3C traceparent header
    ///
    /// Format: "00-{trace_id}-{parent_id}-{flags}"
    pub fn parse(traceparent: &str) -> Option<Self> {
        let parts: Vec<&str> = traceparent.split('-').collect();
        if parts.len() != 4 {
            return None;
        }

        // Validate version
        if parts[0] != "00" {
            return None;
        }

        // Validate trace_id length
        if parts[1].len() != 32 {
            return None;
        }

        // Validate parent_id length
        if parts[2].len() != 16 {
            return None;
        }

        // Parse flags
        let flags = u8::from_str_radix(parts[3], 16).ok()?;

        Some(Self {
            trace_id: parts[1].to_string(),
            parent_id: parts[2].to_string(),
            flags,
        })
    }

    /// Format as W3C traceparent header
    #[must_use]
    pub fn to_traceparent(&self) -> String {
        format!("00-{}-{}-{:02x}", self.trace_id, self.parent_id, self.flags)
    }

    /// Generate a new child span ID
    #[must_use]
    pub fn child(&self) -> Self {
        Self {
            trace_id: self.trace_id.clone(),
            parent_id: generate_span_id(),
            flags: self.flags,
        }
    }

    /// Check if sampled
    #[must_use]
    pub const fn is_sampled(&self) -> bool {
        self.flags & 0x01 != 0
    }
}

impl Default for TraceContext {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for TraceContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_traceparent())
    }
}

/// A trace span representing a unit of work
#[derive(Debug, Clone)]
pub struct TraceSpan {
    /// Span name
    pub name: String,
    /// Trace context
    pub context: TraceContext,
    /// Start timestamp (microseconds since epoch)
    pub start_us: u64,
    /// End timestamp (microseconds since epoch)
    pub end_us: Option<u64>,
    /// Span category (e.g., "browser", "wasm", "network")
    pub category: String,
    /// Span attributes
    pub attributes: Vec<(String, String)>,
    /// Child spans
    pub children: Vec<TraceSpan>,
}

impl TraceSpan {
    /// Create a new trace span
    #[must_use]
    pub fn new(name: impl Into<String>, category: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            context: TraceContext::new(),
            start_us: now_micros(),
            end_us: None,
            category: category.into(),
            attributes: Vec::new(),
            children: Vec::new(),
        }
    }

    /// Create with existing context
    #[must_use]
    pub fn with_context(
        name: impl Into<String>,
        category: impl Into<String>,
        context: TraceContext,
    ) -> Self {
        Self {
            name: name.into(),
            context,
            start_us: now_micros(),
            end_us: None,
            category: category.into(),
            attributes: Vec::new(),
            children: Vec::new(),
        }
    }

    /// Add an attribute
    pub fn add_attribute(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.attributes.push((key.into(), value.into()));
    }

    /// End the span
    pub fn end(&mut self) {
        if self.end_us.is_none() {
            self.end_us = Some(now_micros());
        }
    }

    /// Get duration in microseconds
    #[must_use]
    pub fn duration_us(&self) -> Option<u64> {
        self.end_us.map(|end| end.saturating_sub(self.start_us))
    }

    /// Add a child span
    pub fn add_child(&mut self, child: TraceSpan) {
        self.children.push(child);
    }
}

/// Chrome trace event format (for chrome://tracing)
#[derive(Debug, Clone, serde::Serialize)]
pub struct ChromeTraceEvent {
    /// Event name
    pub name: String,
    /// Category
    pub cat: String,
    /// Phase: B (begin), E (end), X (complete), I (instant)
    pub ph: String,
    /// Timestamp in microseconds
    pub ts: u64,
    /// Duration in microseconds (for X phase)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dur: Option<u64>,
    /// Process ID
    pub pid: u32,
    /// Thread ID
    pub tid: u32,
    /// Arguments
    #[serde(skip_serializing_if = "Option::is_none")]
    pub args: Option<serde_json::Value>,
}

impl ChromeTraceEvent {
    /// Create a complete (X) event from a span
    pub fn from_span(span: &TraceSpan, pid: u32, tid: u32) -> Self {
        let args = if span.attributes.is_empty() {
            None
        } else {
            let map: serde_json::Map<String, serde_json::Value> = span
                .attributes
                .iter()
                .map(|(k, v)| (k.clone(), serde_json::Value::String(v.clone())))
                .collect();
            Some(serde_json::Value::Object(map))
        };

        Self {
            name: span.name.clone(),
            cat: span.category.clone(),
            ph: "X".to_string(),
            ts: span.start_us,
            dur: span.duration_us(),
            pid,
            tid,
            args,
        }
    }

    /// Create an instant (I) event
    pub fn instant(
        name: impl Into<String>,
        category: impl Into<String>,
        ts: u64,
        pid: u32,
        tid: u32,
    ) -> Self {
        Self {
            name: name.into(),
            cat: category.into(),
            ph: "I".to_string(),
            ts,
            dur: None,
            pid,
            tid,
            args: None,
        }
    }
}

/// Chrome trace format output
#[derive(Debug, serde::Serialize)]
pub struct ChromeTrace {
    /// Trace events
    #[serde(rename = "traceEvents")]
    pub trace_events: Vec<ChromeTraceEvent>,
    /// Metadata
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Value>,
}

impl ChromeTrace {
    /// Create an empty trace
    #[must_use]
    pub fn new() -> Self {
        Self {
            trace_events: Vec::new(),
            metadata: None,
        }
    }

    /// Add a span and its children recursively
    pub fn add_span(&mut self, span: &TraceSpan, pid: u32, tid: u32) {
        self.trace_events
            .push(ChromeTraceEvent::from_span(span, pid, tid));

        for child in &span.children {
            self.add_span(child, pid, tid + 1);
        }
    }

    /// Add an instant event
    pub fn add_instant(
        &mut self,
        name: impl Into<String>,
        category: impl Into<String>,
        ts: u64,
        pid: u32,
        tid: u32,
    ) {
        self.trace_events
            .push(ChromeTraceEvent::instant(name, category, ts, pid, tid));
    }

    /// Set metadata
    pub fn set_metadata(&mut self, metadata: serde_json::Value) {
        self.metadata = Some(metadata);
    }

    /// Export as JSON string
    ///
    /// # Errors
    ///
    /// Returns error if serialization fails
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Export as JSON bytes
    ///
    /// # Errors
    ///
    /// Returns error if serialization fails
    pub fn to_json_bytes(&self) -> Result<Vec<u8>, serde_json::Error> {
        serde_json::to_vec_pretty(self)
    }
}

impl Default for ChromeTrace {
    fn default() -> Self {
        Self::new()
    }
}

/// Unified trace collector for browser tests
#[derive(Debug, Default)]
pub struct TraceCollector {
    /// Root trace context
    pub root_context: Option<TraceContext>,
    /// Collected spans
    pub spans: Vec<TraceSpan>,
    /// Console message timestamps (for correlation)
    pub console_timestamps: Vec<(u64, String)>,
    /// Service name
    pub service_name: String,
}

impl TraceCollector {
    /// Create a new trace collector
    #[must_use]
    pub fn new(service_name: impl Into<String>) -> Self {
        let root_context = TraceContext::new();
        Self {
            root_context: Some(root_context),
            spans: Vec::new(),
            console_timestamps: Vec::new(),
            service_name: service_name.into(),
        }
    }

    /// Start a new span
    pub fn start_span(
        &mut self,
        name: impl Into<String>,
        category: impl Into<String>,
    ) -> TraceSpan {
        let context = self
            .root_context
            .as_ref()
            .map(TraceContext::child)
            .unwrap_or_default();

        TraceSpan::with_context(name, category, context)
    }

    /// Record a completed span
    pub fn record_span(&mut self, span: TraceSpan) {
        self.spans.push(span);
    }

    /// Record a console message
    pub fn record_console(&mut self, message: impl Into<String>) {
        self.console_timestamps.push((now_micros(), message.into()));
    }

    /// Get the traceparent header value
    #[must_use]
    pub fn traceparent(&self) -> Option<String> {
        self.root_context.as_ref().map(TraceContext::to_traceparent)
    }

    /// Export as Chrome trace format
    #[must_use]
    pub fn to_chrome_trace(&self) -> ChromeTrace {
        let mut trace = ChromeTrace::new();
        let pid = 1;

        // Add all spans
        for (tid, span) in self.spans.iter().enumerate() {
            trace.add_span(span, pid, tid as u32);
        }

        // Add console messages as instant events
        for (ts, msg) in &self.console_timestamps {
            trace.add_instant(msg.clone(), "console", *ts, pid, 0);
        }

        // Add metadata
        let metadata = serde_json::json!({
            "service_name": self.service_name,
            "trace_id": self.root_context.as_ref().map(|c| &c.trace_id),
        });
        trace.set_metadata(metadata);

        trace
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Generate a random 128-bit trace ID (32 hex chars)
fn generate_trace_id() -> String {
    use std::collections::hash_map::RandomState;
    use std::hash::{BuildHasher, Hasher};

    let hasher = RandomState::new();
    let mut h1 = hasher.build_hasher();
    let mut h2 = hasher.build_hasher();

    h1.write_u64(now_micros());
    h2.write_u64(std::process::id() as u64);

    format!("{:016x}{:016x}", h1.finish(), h2.finish())
}

/// Generate a random 64-bit span ID (16 hex chars)
fn generate_span_id() -> String {
    use std::collections::hash_map::RandomState;
    use std::hash::{BuildHasher, Hasher};

    let hasher = RandomState::new();
    let mut h = hasher.build_hasher();
    h.write_u64(now_micros());
    h.write_u32(std::process::id());

    format!("{:016x}", h.finish())
}

/// Get current time in microseconds since epoch
fn now_micros() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_micros() as u64)
        .unwrap_or(0)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::float_cmp)]
mod tests {
    use super::*;

    mod trace_context_tests {
        use super::*;

        #[test]
        fn test_new() {
            let ctx = TraceContext::new();
            assert_eq!(ctx.trace_id.len(), 32);
            assert_eq!(ctx.parent_id.len(), 16);
            assert!(ctx.is_sampled());
        }

        #[test]
        fn test_parse_valid() {
            let traceparent = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01";
            let ctx = TraceContext::parse(traceparent).unwrap();
            assert_eq!(ctx.trace_id, "0af7651916cd43dd8448eb211c80319c");
            assert_eq!(ctx.parent_id, "b7ad6b7169203331");
            assert_eq!(ctx.flags, 1);
        }

        #[test]
        fn test_parse_invalid_parts() {
            assert!(TraceContext::parse("00-abc").is_none());
        }

        #[test]
        fn test_parse_invalid_version() {
            let traceparent = "01-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01";
            assert!(TraceContext::parse(traceparent).is_none());
        }

        #[test]
        fn test_parse_invalid_trace_id_length() {
            let traceparent = "00-abc-b7ad6b7169203331-01";
            assert!(TraceContext::parse(traceparent).is_none());
        }

        #[test]
        fn test_to_traceparent() {
            let ctx = TraceContext {
                trace_id: "0af7651916cd43dd8448eb211c80319c".to_string(),
                parent_id: "b7ad6b7169203331".to_string(),
                flags: 1,
            };
            assert_eq!(
                ctx.to_traceparent(),
                "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"
            );
        }

        #[test]
        fn test_child() {
            let ctx = TraceContext::new();
            let child = ctx.child();
            assert_eq!(child.trace_id, ctx.trace_id);
            assert_ne!(child.parent_id, ctx.parent_id);
        }

        #[test]
        fn test_display() {
            let ctx = TraceContext::new();
            let s = format!("{ctx}");
            assert!(s.starts_with("00-"));
        }
    }

    mod tracing_config_tests {
        use super::*;

        #[test]
        fn test_default() {
            let config = TracingConfig::default();
            assert!(config.enabled);
            assert_eq!(config.sample_rate, 1.0);
            assert!(config.capture_console);
        }

        #[test]
        fn test_disabled() {
            let config = TracingConfig::disabled();
            assert!(!config.enabled);
            assert_eq!(config.sample_rate, 0.0);
        }

        #[test]
        fn test_builder() {
            let config = TracingConfig::new("my-service")
                .with_sample_rate(0.5)
                .with_console_capture(false);
            assert_eq!(config.service_name, "my-service");
            assert!((config.sample_rate - 0.5).abs() < f64::EPSILON);
            assert!(!config.capture_console);
        }

        #[test]
        fn test_sample_rate_clamped() {
            let config = TracingConfig::default().with_sample_rate(2.0);
            assert!((config.sample_rate - 1.0).abs() < f64::EPSILON);

            let config = TracingConfig::default().with_sample_rate(-1.0);
            assert!((config.sample_rate - 0.0).abs() < f64::EPSILON);
        }
    }

    mod trace_span_tests {
        use super::*;

        #[test]
        fn test_new() {
            let span = TraceSpan::new("test-span", "test");
            assert_eq!(span.name, "test-span");
            assert_eq!(span.category, "test");
            assert!(span.end_us.is_none());
        }

        #[test]
        fn test_add_attribute() {
            let mut span = TraceSpan::new("test", "test");
            span.add_attribute("key", "value");
            assert_eq!(span.attributes.len(), 1);
            assert_eq!(span.attributes[0], ("key".to_string(), "value".to_string()));
        }

        #[test]
        fn test_end() {
            let mut span = TraceSpan::new("test", "test");
            std::thread::sleep(std::time::Duration::from_millis(1));
            span.end();
            assert!(span.end_us.is_some());
            assert!(span.duration_us().unwrap() > 0);
        }

        #[test]
        fn test_duration_before_end() {
            let span = TraceSpan::new("test", "test");
            assert!(span.duration_us().is_none());
        }
    }

    mod chrome_trace_tests {
        use super::*;

        #[test]
        fn test_new() {
            let trace = ChromeTrace::new();
            assert!(trace.trace_events.is_empty());
            assert!(trace.metadata.is_none());
        }

        #[test]
        fn test_add_span() {
            let mut trace = ChromeTrace::new();
            let mut span = TraceSpan::new("test-span", "test");
            span.end();
            trace.add_span(&span, 1, 1);
            assert_eq!(trace.trace_events.len(), 1);
            assert_eq!(trace.trace_events[0].name, "test-span");
        }

        #[test]
        fn test_add_instant() {
            let mut trace = ChromeTrace::new();
            trace.add_instant("instant-event", "test", 1000, 1, 1);
            assert_eq!(trace.trace_events.len(), 1);
            assert_eq!(trace.trace_events[0].ph, "I");
        }

        #[test]
        fn test_to_json() {
            let mut trace = ChromeTrace::new();
            trace.add_instant("test", "test", 0, 1, 1);
            let json = trace.to_json().unwrap();
            assert!(json.contains("traceEvents"));
            assert!(json.contains("test"));
        }

        #[test]
        fn test_nested_spans() {
            let mut trace = ChromeTrace::new();
            let mut parent = TraceSpan::new("parent", "test");
            let mut child = TraceSpan::new("child", "test");
            child.end();
            parent.add_child(child);
            parent.end();
            trace.add_span(&parent, 1, 1);
            assert_eq!(trace.trace_events.len(), 2);
        }
    }

    mod trace_collector_tests {
        use super::*;

        #[test]
        fn test_new() {
            let collector = TraceCollector::new("test-service");
            assert!(collector.root_context.is_some());
            assert_eq!(collector.service_name, "test-service");
        }

        #[test]
        fn test_start_span() {
            let mut collector = TraceCollector::new("test");
            let span = collector.start_span("test-span", "browser");
            assert_eq!(span.name, "test-span");
            assert_eq!(span.category, "browser");
        }

        #[test]
        fn test_record_span() {
            let mut collector = TraceCollector::new("test");
            let mut span = collector.start_span("test", "test");
            span.end();
            collector.record_span(span);
            assert_eq!(collector.spans.len(), 1);
        }

        #[test]
        fn test_record_console() {
            let mut collector = TraceCollector::new("test");
            collector.record_console("test message");
            assert_eq!(collector.console_timestamps.len(), 1);
        }

        #[test]
        fn test_traceparent() {
            let collector = TraceCollector::new("test");
            let traceparent = collector.traceparent().unwrap();
            assert!(traceparent.starts_with("00-"));
        }

        #[test]
        fn test_to_chrome_trace() {
            let mut collector = TraceCollector::new("test-service");
            let mut span = collector.start_span("test-span", "browser");
            span.end();
            collector.record_span(span);
            collector.record_console("console message");

            let chrome_trace = collector.to_chrome_trace();
            assert_eq!(chrome_trace.trace_events.len(), 2); // 1 span + 1 console
        }
    }
}
