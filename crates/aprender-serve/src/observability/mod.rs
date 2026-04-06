//! Observability Module for Production Monitoring
//!
//! Provides comprehensive observability features per spec §7:
//! - Trueno-DB metrics integration for time-series storage
//! - Renacer tracing for distributed request tracking
//! - A/B testing support for model comparison
//!
//! ## Example
//!
//! ```rust,ignore
//! use realizar::observability::{ObservabilityConfig, Observer, ABTest};
//!
//! let config = ObservabilityConfig::new()
//!     .with_trueno_db("trueno-db://metrics")
//!     .with_tracing(true);
//!
//! let observer = Observer::new(config);
//!
//! // Record inference metrics
//! observer.record_inference("model-v1", 150, 32, Duration::from_millis(45));
//!
//! // A/B testing
//! let ab_test = ABTest::new("model-comparison")
//!     .with_variant("control", "model-v1", 0.5)
//!     .with_variant("treatment", "model-v2", 0.5);
//! let variant = ab_test.select("user-123");
//! ```

use std::{
    collections::HashMap,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, RwLock,
    },
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use serde::{Deserialize, Serialize};

// ============================================================================
// W3C Trace Context (per OpenTelemetry specification)
// ============================================================================

/// W3C Trace Context for distributed tracing
/// Implements <https://www.w3.org/TR/trace-context/>
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceContext {
    /// Trace ID (32 hex chars, 16 bytes)
    pub trace_id: String,
    /// Parent Span ID (16 hex chars, 8 bytes)
    pub parent_span_id: Option<String>,
    /// Trace flags (sampled, etc.)
    pub trace_flags: u8,
    /// Trace state (vendor-specific data)
    pub trace_state: Option<String>,
}

impl TraceContext {
    /// Create a new trace context with a fresh trace ID
    #[must_use]
    pub fn new() -> Self {
        Self {
            trace_id: generate_trace_id(),
            parent_span_id: None,
            trace_flags: 0x01, // Sampled
            trace_state: None,
        }
    }

    /// Create child context with new span ID
    #[must_use]
    pub fn child(&self, parent_span_id: &str) -> Self {
        Self {
            trace_id: self.trace_id.clone(),
            parent_span_id: Some(parent_span_id.to_string()),
            trace_flags: self.trace_flags,
            trace_state: self.trace_state.clone(),
        }
    }

    /// Parse from W3C traceparent header
    /// Format: {version}-{trace_id}-{parent_id}-{flags}
    /// Example: 00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01
    #[must_use]
    pub fn from_traceparent(header: &str) -> Option<Self> {
        let parts: Vec<&str> = header.split('-').collect();
        if parts.len() != 4 {
            return None;
        }

        let version = parts[0];
        if version != "00" {
            return None; // Only version 00 supported
        }

        let trace_id = parts[1];
        let parent_span_id = parts[2];
        let flags = u8::from_str_radix(parts[3], 16).ok()?;

        // Validate lengths
        if trace_id.len() != 32 || parent_span_id.len() != 16 {
            return None;
        }

        Some(Self {
            trace_id: trace_id.to_string(),
            parent_span_id: Some(parent_span_id.to_string()),
            trace_flags: flags,
            trace_state: None,
        })
    }

    /// Generate W3C traceparent header
    #[must_use]
    pub fn to_traceparent(&self, span_id: &str) -> String {
        format!("00-{}-{}-{:02x}", self.trace_id, span_id, self.trace_flags)
    }

    /// Set tracestate header value
    #[must_use]
    pub fn with_tracestate(mut self, state: impl Into<String>) -> Self {
        self.trace_state = Some(state.into());
        self
    }

    /// Check if trace is sampled
    #[must_use]
    pub fn is_sampled(&self) -> bool {
        self.trace_flags & 0x01 != 0
    }

    /// Set sampled flag
    pub fn set_sampled(&mut self, sampled: bool) {
        if sampled {
            self.trace_flags |= 0x01;
        } else {
            self.trace_flags &= !0x01;
        }
    }
}

impl Default for TraceContext {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Latency Histogram for Percentile Calculations
// ============================================================================

/// Histogram for tracking latency distributions and calculating percentiles
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LatencyHistogram {
    /// Bucket boundaries in microseconds
    buckets: Vec<u64>,
    /// Count per bucket
    counts: Vec<u64>,
    /// Total count
    total: u64,
    /// Sum of all values (for mean calculation)
    sum: u64,
    /// Min value seen
    min: Option<u64>,
    /// Max value seen
    max: Option<u64>,
}

impl LatencyHistogram {
    /// Create histogram with default buckets (exponential: 1ms to 60s)
    #[must_use]
    pub fn new() -> Self {
        // Buckets: 1ms, 2ms, 5ms, 10ms, 25ms, 50ms, 100ms, 250ms, 500ms, 1s, 2.5s, 5s, 10s, 30s, 60s
        let buckets = vec![
            1_000,      // 1ms
            2_000,      // 2ms
            5_000,      // 5ms
            10_000,     // 10ms
            25_000,     // 25ms
            50_000,     // 50ms
            100_000,    // 100ms
            250_000,    // 250ms
            500_000,    // 500ms
            1_000_000,  // 1s
            2_500_000,  // 2.5s
            5_000_000,  // 5s
            10_000_000, // 10s
            30_000_000, // 30s
            60_000_000, // 60s
        ];
        let counts = vec![0; buckets.len() + 1]; // +1 for overflow bucket
        Self {
            buckets,
            counts,
            total: 0,
            sum: 0,
            min: None,
            max: None,
        }
    }

    /// Create histogram with custom buckets (in microseconds)
    #[must_use]
    pub fn with_buckets(mut buckets: Vec<u64>) -> Self {
        buckets.sort_unstable();
        let counts = vec![0; buckets.len() + 1];
        Self {
            buckets,
            counts,
            total: 0,
            sum: 0,
            min: None,
            max: None,
        }
    }

    /// Record a latency value in microseconds
    pub fn observe(&mut self, value_us: u64) {
        self.total += 1;
        self.sum += value_us;

        // Update min/max
        self.min = Some(self.min.map_or(value_us, |m| m.min(value_us)));
        self.max = Some(self.max.map_or(value_us, |m| m.max(value_us)));

        // Find bucket
        let bucket_idx = self.buckets.iter().position(|&b| value_us <= b);
        match bucket_idx {
            Some(idx) => self.counts[idx] += 1,
            None => *self.counts.last_mut().unwrap_or(&mut 0) += 1,
        }
    }

    /// Record latency from Duration
    pub fn observe_duration(&mut self, duration: Duration) {
        self.observe(duration.as_micros() as u64);
    }

    /// Get percentile value (0-100)
    #[must_use]
    pub fn percentile(&self, p: f64) -> Option<u64> {
        if self.total == 0 || !(0.0..=100.0).contains(&p) {
            return None;
        }

        let target = ((p / 100.0) * self.total as f64).ceil() as u64;
        let mut cumulative = 0u64;

        for (i, &count) in self.counts.iter().enumerate() {
            cumulative += count;
            if cumulative >= target {
                return if i < self.buckets.len() {
                    Some(self.buckets[i])
                } else {
                    self.max // Overflow bucket
                };
            }
        }

        self.max
    }

    /// Get p50 (median)
    #[must_use]
    pub fn p50(&self) -> Option<u64> {
        self.percentile(50.0)
    }

    /// Get p95
    #[must_use]
    pub fn p95(&self) -> Option<u64> {
        self.percentile(95.0)
    }

    /// Get p99
    #[must_use]
    pub fn p99(&self) -> Option<u64> {
        self.percentile(99.0)
    }

    /// Get mean latency
    #[must_use]
    pub fn mean(&self) -> Option<f64> {
        if self.total == 0 {
            None
        } else {
            Some(self.sum as f64 / self.total as f64)
        }
    }

    /// Get total count
    #[must_use]
    pub fn count(&self) -> u64 {
        self.total
    }

    /// Get min value
    #[must_use]
    pub fn min(&self) -> Option<u64> {
        self.min
    }

    /// Get max value
    #[must_use]
    pub fn max_val(&self) -> Option<u64> {
        self.max
    }

    /// Export as Prometheus histogram format
    #[must_use]
    pub fn to_prometheus(&self, name: &str, labels: &str) -> String {
        use std::fmt::Write;
        let mut output = String::new();

        // Bucket counters (cumulative)
        let mut cumulative = 0u64;
        for (i, &boundary) in self.buckets.iter().enumerate() {
            cumulative += self.counts[i];
            let le = boundary as f64 / 1_000_000.0; // Convert to seconds
            writeln!(
                output,
                "{name}_bucket{{le=\"{le:.6}\",{labels}}} {cumulative}"
            )
            .expect("fmt::Write for String is infallible");
        }
        // +Inf bucket
        cumulative += self.counts.last().copied().unwrap_or(0);
        writeln!(output, "{name}_bucket{{le=\"+Inf\",{labels}}} {cumulative}")
            .expect("fmt::Write for String is infallible");

        // Sum and count
        let sum_secs = self.sum as f64 / 1_000_000.0;
        writeln!(output, "{name}_sum{{{labels}}} {sum_secs:.6}")
            .expect("fmt::Write for String is infallible");
        writeln!(output, "{name}_count{{{labels}}} {}", self.total)
            .expect("fmt::Write for String is infallible");

        output
    }
}

// ============================================================================
// OpenTelemetry-Compatible Span Export
// ============================================================================

/// OpenTelemetry-compatible span for export
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OtelSpan {
    /// Trace ID (hex string)
    #[serde(rename = "traceId")]
    pub trace_id: String,
    /// Span ID (hex string)
    #[serde(rename = "spanId")]
    pub span_id: String,
    /// Parent Span ID (optional)
    #[serde(rename = "parentSpanId", skip_serializing_if = "Option::is_none")]
    pub parent_span_id: Option<String>,
    /// Operation name
    #[serde(rename = "operationName")]
    pub operation_name: String,
    /// Service name
    #[serde(rename = "serviceName")]
    pub service_name: String,
    /// Start time (Unix epoch microseconds)
    #[serde(rename = "startTimeUnixNano")]
    pub start_time: u64,
    /// End time (Unix epoch microseconds)
    #[serde(rename = "endTimeUnixNano")]
    pub end_time: u64,
    /// Span kind
    #[serde(rename = "kind")]
    pub kind: SpanKind,
    /// Status
    pub status: OtelStatus,
    /// Attributes
    pub attributes: Vec<OtelAttribute>,
}

/// OpenTelemetry span kind
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum SpanKind {
    /// Internal operation
    #[default]
    Internal,
    /// Server-side of RPC
    Server,
    /// Client-side of RPC
    Client,
    /// Message producer
    Producer,
    /// Message consumer
    Consumer,
}

/// OpenTelemetry status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OtelStatus {
    /// Status code
    pub code: OtelStatusCode,
    /// Optional message
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
}

/// OpenTelemetry status code
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum OtelStatusCode {
    /// Unset
    #[default]
    Unset,
    /// OK
    Ok,
    /// Error
    Error,
}

/// OpenTelemetry attribute
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OtelAttribute {
    /// Attribute key
    pub key: String,
    /// Attribute value
    pub value: OtelValue,
}

include!("mod_otel_value.rs");
include!("mod_variant.rs");
