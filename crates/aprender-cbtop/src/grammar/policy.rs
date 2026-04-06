//! Execution policy (Theme equivalent in Grammar of Graphics).

use std::time::Duration;

use super::resources::ByteSize;

/// Quality of Service level
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QosLevel {
    /// Best effort (no guarantees)
    BestEffort,
    /// Background (lowest priority)
    Background,
    /// Interactive (balanced)
    Interactive,
    /// Realtime (highest priority)
    Realtime,
}

/// Retry policy
#[derive(Debug, Clone, PartialEq)]
pub struct RetryPolicy {
    /// Maximum retry attempts
    pub max_retries: usize,
    /// Initial backoff duration
    pub initial_backoff: Duration,
    /// Backoff multiplier
    pub backoff_multiplier: f64,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_retries: 3,
            initial_backoff: Duration::from_millis(100),
            backoff_multiplier: 2.0,
        }
    }
}

/// Resource limits
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ResourceLimits {
    /// Maximum memory usage
    pub max_memory: Option<ByteSize>,
    /// Maximum CPU cores
    pub max_cores: Option<usize>,
    /// Maximum GPU memory
    pub max_gpu_memory: Option<ByteSize>,
}

/// Observability configuration
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ObservabilityConfig {
    /// Enable tracing
    pub tracing: bool,
    /// Enable metrics
    pub metrics: bool,
    /// Sampling rate (0.0-1.0)
    pub sampling_rate: f64,
}

/// Execution policy (analogous to Theme)
#[derive(Debug, Clone, PartialEq)]
pub struct ExecutionPolicy {
    /// Quality of Service level
    pub qos: QosLevel,
    /// Preemption allowed
    pub preemptible: bool,
    /// Timeout constraint
    pub timeout: Option<Duration>,
    /// Retry policy
    pub retry: RetryPolicy,
    /// Resource limits
    pub limits: ResourceLimits,
    /// Observability config
    pub observability: ObservabilityConfig,
}

impl Default for ExecutionPolicy {
    fn default() -> Self {
        Self {
            qos: QosLevel::Interactive,
            preemptible: true,
            timeout: None,
            retry: RetryPolicy::default(),
            limits: ResourceLimits::default(),
            observability: ObservabilityConfig::default(),
        }
    }
}

impl ExecutionPolicy {
    /// Create realtime policy (low latency, non-preemptible)
    pub fn realtime() -> Self {
        Self {
            qos: QosLevel::Realtime,
            preemptible: false,
            timeout: Some(Duration::from_millis(100)),
            ..Default::default()
        }
    }

    /// Create batch policy (high throughput, preemptible)
    pub fn batch() -> Self {
        Self {
            qos: QosLevel::BestEffort,
            preemptible: true,
            timeout: None,
            ..Default::default()
        }
    }

    /// Create interactive policy (balanced)
    pub fn interactive() -> Self {
        Self::default()
    }

    /// Create debug policy (full tracing, relaxed limits)
    pub fn debug() -> Self {
        Self {
            qos: QosLevel::BestEffort,
            preemptible: true,
            timeout: None,
            observability: ObservabilityConfig {
                tracing: true,
                metrics: true,
                sampling_rate: 1.0,
            },
            ..Default::default()
        }
    }
}
