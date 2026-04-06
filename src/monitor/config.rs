//! Monitor configuration and error types (TRUENO-SPEC-010 Section 8.2)

use std::time::Duration;

// ============================================================================
// Monitor Configuration (TRUENO-SPEC-010 Section 8.2)
// ============================================================================

/// Configuration for GPU monitoring
#[derive(Debug, Clone)]
pub struct MonitorConfig {
    /// Polling interval for metrics collection
    pub poll_interval: Duration,
    /// History buffer size (number of samples to retain)
    pub history_size: usize,
    /// Enable background collection thread
    pub background_collection: bool,
}

/// 60 seconds of history at 100ms polling interval
const DEFAULT_HISTORY_SIZE: usize = 600;
/// 60 seconds of history at 50ms high-frequency polling
const HIGH_FREQ_HISTORY_SIZE: usize = 1200;
/// 60 seconds of history at 500ms low-overhead polling
const LOW_OVERHEAD_HISTORY_SIZE: usize = 120;

impl Default for MonitorConfig {
    fn default() -> Self {
        Self {
            poll_interval: Duration::from_millis(100),
            history_size: DEFAULT_HISTORY_SIZE,
            background_collection: false,
        }
    }
}

impl MonitorConfig {
    /// Create config for high-frequency monitoring
    #[must_use]
    pub fn high_frequency() -> Self {
        Self {
            poll_interval: Duration::from_millis(50),
            history_size: HIGH_FREQ_HISTORY_SIZE,
            background_collection: true,
        }
    }

    /// Create config for low-overhead monitoring
    #[must_use]
    pub fn low_overhead() -> Self {
        Self {
            poll_interval: Duration::from_millis(500),
            history_size: LOW_OVERHEAD_HISTORY_SIZE,
            background_collection: false,
        }
    }
}

// ============================================================================
// Error Types
// ============================================================================

/// Errors from GPU monitoring operations
#[derive(Debug, Clone)]
pub enum MonitorError {
    /// No GPU device available
    NoDevice,
    /// Invalid device index
    InvalidDevice(u32),
    /// GPU backend initialization failed
    BackendInit(String),
    /// Metrics query failed
    QueryFailed(String),
    /// Feature not available on this GPU/platform
    NotAvailable(String),
}

impl std::fmt::Display for MonitorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoDevice => write!(f, "No GPU device available"),
            Self::InvalidDevice(idx) => write!(f, "Invalid device index: {idx}"),
            Self::BackendInit(msg) => write!(f, "Backend initialization failed: {msg}"),
            Self::QueryFailed(msg) => write!(f, "Metrics query failed: {msg}"),
            Self::NotAvailable(msg) => write!(f, "Feature not available: {msg}"),
        }
    }
}

impl std::error::Error for MonitorError {}
