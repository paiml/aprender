//! Types, enums, configs, and metric definitions for observability backends.

use std::collections::HashMap;
use std::time::Instant;

/// Result type for observability operations
pub type ObservabilityResult<T> = Result<T, ObservabilityError>;

/// Errors in observability operations
#[derive(Debug, Clone, PartialEq)]
pub enum ObservabilityError {
    /// Backend connection failed
    ConnectionFailed { backend: String, reason: String },
    /// Authentication failed
    AuthenticationFailed { backend: String },
    /// Invalid configuration
    InvalidConfig { reason: String },
    /// Rate limited by backend
    RateLimited {
        backend: String,
        retry_after_sec: u64,
    },
    /// Export failed
    ExportFailed { backend: String, reason: String },
    /// Backend not configured
    BackendNotConfigured { backend: String },
    /// Serialization error
    SerializationError { reason: String },
}

impl std::fmt::Display for ObservabilityError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ConnectionFailed { backend, reason } => {
                write!(f, "Connection to {} failed: {}", backend, reason)
            }
            Self::AuthenticationFailed { backend } => {
                write!(f, "Authentication failed for {}", backend)
            }
            Self::InvalidConfig { reason } => {
                write!(f, "Invalid configuration: {}", reason)
            }
            Self::RateLimited {
                backend,
                retry_after_sec,
            } => {
                write!(
                    f,
                    "{} rate limited, retry after {}s",
                    backend, retry_after_sec
                )
            }
            Self::ExportFailed { backend, reason } => {
                write!(f, "Export to {} failed: {}", backend, reason)
            }
            Self::BackendNotConfigured { backend } => {
                write!(f, "Backend {} not configured", backend)
            }
            Self::SerializationError { reason } => {
                write!(f, "Serialization error: {}", reason)
            }
        }
    }
}

impl std::error::Error for ObservabilityError {}

/// Supported observability backends
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ObservabilityBackend {
    /// Datadog (DogStatsD)
    Datadog,
    /// New Relic
    NewRelic,
    /// Honeycomb
    Honeycomb,
    /// OpenTelemetry Collector
    Otlp,
    /// Generic webhook
    Webhook,
}

impl ObservabilityBackend {
    /// Get backend name
    pub fn name(&self) -> &'static str {
        match self {
            Self::Datadog => "Datadog",
            Self::NewRelic => "NewRelic",
            Self::Honeycomb => "Honeycomb",
            Self::Otlp => "OTLP",
            Self::Webhook => "Webhook",
        }
    }
}

/// Datadog configuration
#[derive(Debug, Clone)]
pub struct DatadogConfig {
    /// DogStatsD host (default: localhost)
    pub host: String,
    /// DogStatsD port (default: 8125)
    pub port: u16,
    /// API key for Datadog API
    pub api_key: Option<String>,
    /// Default tags to add to all metrics
    pub default_tags: Vec<String>,
    /// Metric prefix
    pub prefix: String,
}

impl Default for DatadogConfig {
    fn default() -> Self {
        Self {
            host: "localhost".to_string(),
            port: 8125,
            api_key: None,
            default_tags: Vec::new(),
            prefix: "cbtop".to_string(),
        }
    }
}

/// New Relic configuration
#[derive(Debug, Clone)]
pub struct NewRelicConfig {
    /// API endpoint (default: US)
    pub endpoint: String,
    /// API key (required)
    pub api_key: String,
    /// Account ID
    pub account_id: String,
    /// Default attributes
    pub default_attributes: HashMap<String, String>,
}

impl NewRelicConfig {
    /// Create New Relic config with required fields
    pub fn new(api_key: impl Into<String>, account_id: impl Into<String>) -> Self {
        Self {
            endpoint: "https://metric-api.newrelic.com/metric/v1".to_string(),
            api_key: api_key.into(),
            account_id: account_id.into(),
            default_attributes: HashMap::new(),
        }
    }

    /// Set custom endpoint (e.g., EU region)
    pub fn with_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.endpoint = endpoint.into();
        self
    }
}

/// Honeycomb configuration
#[derive(Debug, Clone)]
pub struct HoneycombConfig {
    /// API endpoint
    pub endpoint: String,
    /// API key (required)
    pub api_key: String,
    /// Dataset name
    pub dataset: String,
    /// Service name
    pub service_name: String,
}

impl HoneycombConfig {
    /// Create Honeycomb config with required fields
    pub fn new(api_key: impl Into<String>, dataset: impl Into<String>) -> Self {
        Self {
            endpoint: "https://api.honeycomb.io/1/events".to_string(),
            api_key: api_key.into(),
            dataset: dataset.into(),
            service_name: "cbtop".to_string(),
        }
    }
}

/// OpenTelemetry Collector configuration
#[derive(Debug, Clone)]
pub struct OtlpConfig {
    /// OTLP endpoint (default: localhost:4317)
    pub endpoint: String,
    /// Use HTTP instead of gRPC
    pub use_http: bool,
    /// Headers to send with requests
    pub headers: HashMap<String, String>,
    /// Resource attributes
    pub resource_attributes: HashMap<String, String>,
}

impl Default for OtlpConfig {
    fn default() -> Self {
        let mut resource_attributes = HashMap::new();
        resource_attributes.insert("service.name".to_string(), "cbtop".to_string());

        Self {
            endpoint: "http://localhost:4317".to_string(),
            use_http: false,
            headers: HashMap::new(),
            resource_attributes,
        }
    }
}

/// Webhook configuration
#[derive(Debug, Clone)]
pub struct WebhookConfig {
    /// Webhook URL
    pub url: String,
    /// HTTP method (default: POST)
    pub method: String,
    /// Headers to send
    pub headers: HashMap<String, String>,
    /// Authentication token
    pub auth_token: Option<String>,
}

impl WebhookConfig {
    /// Create webhook config with URL
    pub fn new(url: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            method: "POST".to_string(),
            headers: HashMap::new(),
            auth_token: None,
        }
    }

    /// Set authentication token
    pub fn with_auth(mut self, token: impl Into<String>) -> Self {
        self.auth_token = Some(token.into());
        self
    }
}

/// A metric to export
#[derive(Debug, Clone)]
pub struct ExportMetric {
    /// Metric name
    pub name: String,
    /// Metric value
    pub value: f64,
    /// Metric type (gauge, counter, histogram)
    pub metric_type: MetricExportType,
    /// Tags/labels
    pub tags: HashMap<String, String>,
    /// Timestamp in nanoseconds
    pub timestamp_ns: u64,
    /// Optional unit
    pub unit: Option<String>,
}

impl ExportMetric {
    /// Create a gauge metric
    pub fn gauge(name: impl Into<String>, value: f64) -> Self {
        Self {
            name: name.into(),
            value,
            metric_type: MetricExportType::Gauge,
            tags: HashMap::new(),
            timestamp_ns: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(0),
            unit: None,
        }
    }

    /// Create a counter metric
    pub fn counter(name: impl Into<String>, value: f64) -> Self {
        Self {
            name: name.into(),
            value,
            metric_type: MetricExportType::Counter,
            tags: HashMap::new(),
            timestamp_ns: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(0),
            unit: None,
        }
    }

    /// Add a tag
    pub fn with_tag(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.tags.insert(key.into(), value.into());
        self
    }

    /// Set unit
    pub fn with_unit(mut self, unit: impl Into<String>) -> Self {
        self.unit = Some(unit.into());
        self
    }
}

/// Metric type for export
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetricExportType {
    /// Gauge (point-in-time value)
    Gauge,
    /// Counter (monotonically increasing)
    Counter,
    /// Histogram (distribution)
    Histogram,
}

/// Export result from a backend
#[derive(Debug, Clone)]
pub struct ExportResult {
    /// Backend that was exported to
    pub backend: ObservabilityBackend,
    /// Whether export succeeded
    pub success: bool,
    /// Number of metrics exported
    pub metrics_exported: usize,
    /// Export duration
    pub duration_ms: u64,
    /// Error message if failed
    pub error: Option<String>,
}

/// Health status of a backend
#[derive(Debug, Clone)]
pub struct BackendHealth {
    /// Backend type
    pub backend: ObservabilityBackend,
    /// Whether backend is healthy
    pub healthy: bool,
    /// Last successful export time
    pub last_success: Option<Instant>,
    /// Consecutive failures
    pub consecutive_failures: u32,
    /// Average export latency
    pub avg_latency_ms: f64,
}
