
/// OpenTelemetry attribute value
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum OtelValue {
    /// String value
    String {
        /// The string value
        string_value: String,
    },
    /// Integer value
    Int {
        /// The integer value
        int_value: i64,
    },
    /// Float value
    Float {
        /// The double/float value
        double_value: f64,
    },
    /// Boolean value
    Bool {
        /// The boolean value
        bool_value: bool,
    },
}

impl From<&str> for OtelValue {
    fn from(s: &str) -> Self {
        OtelValue::String {
            string_value: s.to_string(),
        }
    }
}

impl From<String> for OtelValue {
    fn from(s: String) -> Self {
        OtelValue::String { string_value: s }
    }
}

impl From<i64> for OtelValue {
    fn from(v: i64) -> Self {
        OtelValue::Int { int_value: v }
    }
}

impl From<f64> for OtelValue {
    fn from(v: f64) -> Self {
        OtelValue::Float { double_value: v }
    }
}

impl From<bool> for OtelValue {
    fn from(v: bool) -> Self {
        OtelValue::Bool { bool_value: v }
    }
}

// ============================================================================
// OBS-001: Configuration
// ============================================================================

/// Observability configuration
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ObservabilityConfig {
    /// Trueno-DB connection URI for metrics storage
    pub trueno_db_uri: Option<String>,
    /// Enable distributed tracing
    pub tracing_enabled: bool,
    /// Sampling rate for traces (0.0 - 1.0)
    pub trace_sample_rate: f64,
    /// Metrics flush interval in seconds
    pub flush_interval_secs: u64,
    /// Enable A/B testing
    pub ab_testing_enabled: bool,
}

impl ObservabilityConfig {
    /// Create new configuration
    #[must_use]
    pub fn new() -> Self {
        Self {
            trueno_db_uri: None,
            tracing_enabled: true,
            trace_sample_rate: 1.0,
            flush_interval_secs: 60,
            ab_testing_enabled: true,
        }
    }

    /// Set Trueno-DB URI
    #[must_use]
    pub fn with_trueno_db(mut self, uri: impl Into<String>) -> Self {
        self.trueno_db_uri = Some(uri.into());
        self
    }

    /// Enable/disable tracing
    #[must_use]
    pub fn with_tracing(mut self, enabled: bool) -> Self {
        self.tracing_enabled = enabled;
        self
    }

    /// Set trace sampling rate
    #[must_use]
    pub fn with_sample_rate(mut self, rate: f64) -> Self {
        self.trace_sample_rate = rate.clamp(0.0, 1.0);
        self
    }

    /// Set flush interval
    #[must_use]
    pub fn with_flush_interval(mut self, secs: u64) -> Self {
        self.flush_interval_secs = secs;
        self
    }
}

// ============================================================================
// OBS-002: Metrics Point
// ============================================================================

/// A single metrics data point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricPoint {
    /// Metric name
    pub name: String,
    /// Metric value
    pub value: f64,
    /// Timestamp (Unix epoch millis)
    pub timestamp: u64,
    /// Labels/tags
    pub labels: HashMap<String, String>,
}

impl MetricPoint {
    /// Create a new metric point
    #[must_use]
    pub fn new(name: impl Into<String>, value: f64) -> Self {
        Self {
            name: name.into(),
            value,
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_millis() as u64)
                .unwrap_or(0),
            labels: HashMap::new(),
        }
    }

    /// Add a label
    #[must_use]
    pub fn with_label(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.labels.insert(key.into(), value.into());
        self
    }

    /// Format for Trueno-DB line protocol
    #[must_use]
    pub fn to_line_protocol(&self) -> String {
        let labels_str = if self.labels.is_empty() {
            String::new()
        } else {
            let pairs: Vec<String> = self
                .labels
                .iter()
                .map(|(k, v)| format!("{k}={v}"))
                .collect();
            format!(",{}", pairs.join(","))
        };

        format!(
            "{}{} value={} {}",
            self.name, labels_str, self.value, self.timestamp
        )
    }
}

// ============================================================================
// OBS-003: Tracing Span
// ============================================================================

/// A tracing span for distributed request tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Span {
    /// Unique span ID
    pub span_id: String,
    /// Trace ID (shared across spans in same request)
    pub trace_id: String,
    /// Parent span ID (if any)
    pub parent_id: Option<String>,
    /// Operation name
    pub operation: String,
    /// Service name
    pub service: String,
    /// Start timestamp (Unix epoch micros)
    pub start_time: u64,
    /// Duration in microseconds
    pub duration_us: Option<u64>,
    /// Span status
    pub status: SpanStatus,
    /// Span attributes
    pub attributes: HashMap<String, String>,
}

/// Span status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum SpanStatus {
    /// Span is in progress
    #[default]
    InProgress,
    /// Span completed successfully
    Ok,
    /// Span completed with error
    Error,
}

impl Span {
    /// Create a new span
    #[must_use]
    pub fn new(operation: impl Into<String>, trace_id: impl Into<String>) -> Self {
        Self {
            span_id: generate_id(),
            trace_id: trace_id.into(),
            parent_id: None,
            operation: operation.into(),
            service: "realizar".to_string(),
            start_time: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_micros() as u64)
                .unwrap_or(0),
            duration_us: None,
            status: SpanStatus::InProgress,
            attributes: HashMap::new(),
        }
    }

    /// Create a child span
    #[must_use]
    pub fn child(&self, operation: impl Into<String>) -> Self {
        let mut span = Self::new(operation, self.trace_id.clone());
        span.parent_id = Some(self.span_id.clone());
        span
    }

    /// Set parent span
    #[must_use]
    pub fn with_parent(mut self, parent_id: impl Into<String>) -> Self {
        self.parent_id = Some(parent_id.into());
        self
    }

    /// Add attribute
    #[must_use]
    pub fn with_attribute(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.attributes.insert(key.into(), value.into());
        self
    }

    /// End the span with success
    pub fn end_ok(&mut self) {
        self.status = SpanStatus::Ok;
        self.duration_us = Some(
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_micros() as u64)
                .unwrap_or(0)
                .saturating_sub(self.start_time),
        );
    }

    /// End the span with error
    pub fn end_error(&mut self, error: impl Into<String>) {
        self.status = SpanStatus::Error;
        self.attributes.insert("error".to_string(), error.into());
        self.duration_us = Some(
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_micros() as u64)
                .unwrap_or(0)
                .saturating_sub(self.start_time),
        );
    }

    /// Get duration as Duration
    #[must_use]
    pub fn duration(&self) -> Option<Duration> {
        self.duration_us.map(Duration::from_micros)
    }

    /// Set span kind
    #[must_use]
    pub fn with_kind(mut self, kind: SpanKind) -> Self {
        self.attributes
            .insert("span.kind".to_string(), format!("{kind:?}"));
        self
    }

    /// Convert to OpenTelemetry-compatible span format
    #[must_use]
    pub fn to_otel(&self) -> OtelSpan {
        let end_time = self.start_time + self.duration_us.unwrap_or(0);

        let status = match self.status {
            SpanStatus::Ok => OtelStatus {
                code: OtelStatusCode::Ok,
                message: None,
            },
            SpanStatus::Error => OtelStatus {
                code: OtelStatusCode::Error,
                message: self.attributes.get("error").cloned(),
            },
            SpanStatus::InProgress => OtelStatus {
                code: OtelStatusCode::Unset,
                message: None,
            },
        };

        let attributes: Vec<OtelAttribute> = self
            .attributes
            .iter()
            .map(|(k, v)| OtelAttribute {
                key: k.clone(),
                value: OtelValue::from(v.as_str()),
            })
            .collect();

        let kind = self
            .attributes
            .get("span.kind")
            .map_or(SpanKind::Internal, |k| match k.as_str() {
                "Server" => SpanKind::Server,
                "Client" => SpanKind::Client,
                "Producer" => SpanKind::Producer,
                "Consumer" => SpanKind::Consumer,
                _ => SpanKind::Internal,
            });

        OtelSpan {
            trace_id: self.trace_id.clone(),
            span_id: self.span_id.clone(),
            parent_span_id: self.parent_id.clone(),
            operation_name: self.operation.clone(),
            service_name: self.service.clone(),
            start_time: self.start_time * 1000, // Convert to nanoseconds
            end_time: end_time * 1000,
            kind,
            status,
            attributes,
        }
    }

    /// Get trace context for propagation
    #[must_use]
    pub fn trace_context(&self) -> TraceContext {
        TraceContext {
            trace_id: self.trace_id.clone(),
            parent_span_id: Some(self.span_id.clone()),
            trace_flags: 0x01, // Sampled
            trace_state: None,
        }
    }

    /// Generate traceparent header for this span
    #[must_use]
    pub fn traceparent(&self) -> String {
        format!("00-{}-{}-01", self.trace_id, self.span_id)
    }
}

// ============================================================================
// OBS-004: A/B Testing
// ============================================================================

/// A/B test configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ABTest {
    /// Test name
    pub name: String,
    /// Test variants
    pub variants: Vec<ABVariant>,
    /// Whether test is active
    pub active: bool,
    /// Test start timestamp
    pub start_time: u64,
}

/// A/B test variant
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ABVariant {
    /// Variant name
    pub name: String,
    /// Model to use for this variant
    pub model: String,
    /// Traffic weight (0.0 - 1.0)
    pub weight: f64,
}
