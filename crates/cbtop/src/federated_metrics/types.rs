//! Core types for federated metrics aggregation.

use std::collections::HashMap;

/// Result type for federated operations
pub type FederatedResult<T> = Result<T, FederatedError>;

/// Errors in federated operations
#[derive(Debug, Clone, PartialEq)]
pub enum FederatedError {
    /// Host not found in federation
    HostNotFound { host_id: String },
    /// Network partition detected
    PartitionDetected { affected_hosts: Vec<String> },
    /// Merge conflict that couldn't be resolved
    MergeConflict { reason: String },
    /// Clock drift too large
    ClockDriftExceeded { drift_ms: i64, max_ms: i64 },
    /// Memory limit exceeded
    MemoryLimitExceeded { used_bytes: usize, limit_bytes: usize },
    /// Invalid configuration
    InvalidConfig { reason: String },
}

impl std::fmt::Display for FederatedError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::HostNotFound { host_id } => write!(f, "Host not found: {}", host_id),
            Self::PartitionDetected { affected_hosts } => {
                write!(f, "Partition detected affecting: {:?}", affected_hosts)
            }
            Self::MergeConflict { reason } => write!(f, "Merge conflict: {}", reason),
            Self::ClockDriftExceeded { drift_ms, max_ms } => {
                write!(f, "Clock drift {}ms exceeds max {}ms", drift_ms, max_ms)
            }
            Self::MemoryLimitExceeded { used_bytes, limit_bytes } => {
                write!(f, "Memory {} exceeds limit {}", used_bytes, limit_bytes)
            }
            Self::InvalidConfig { reason } => write!(f, "Invalid config: {}", reason),
        }
    }
}

impl std::error::Error for FederatedError {}

/// Unique identifier for a metric sample
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SampleId {
    /// Host that generated the sample
    pub host_id: String,
    /// Logical timestamp (Lamport clock)
    pub logical_time: u64,
    /// Unique sequence number within host
    pub sequence: u64,
}

impl SampleId {
    /// Create a new sample ID
    pub fn new(host_id: impl Into<String>, logical_time: u64, sequence: u64) -> Self {
        Self { host_id: host_id.into(), logical_time, sequence }
    }
}

/// A single metric sample with vector clock
#[derive(Debug, Clone)]
pub struct MetricSample {
    /// Unique identifier
    pub id: SampleId,
    /// Metric name
    pub metric_name: String,
    /// Metric value
    pub value: f64,
    /// Wall clock timestamp (nanoseconds)
    pub timestamp_ns: u64,
    /// Vector clock for causal ordering
    pub vector_clock: HashMap<String, u64>,
}

impl MetricSample {
    /// Create a new metric sample
    pub fn new(
        host_id: impl Into<String>,
        logical_time: u64,
        sequence: u64,
        metric_name: impl Into<String>,
        value: f64,
    ) -> Self {
        let host = host_id.into();
        let mut vector_clock = HashMap::new();
        vector_clock.insert(host.clone(), logical_time);

        Self {
            id: SampleId::new(host, logical_time, sequence),
            metric_name: metric_name.into(),
            value,
            timestamp_ns: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(0),
            vector_clock,
        }
    }
}
