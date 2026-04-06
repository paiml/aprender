//! Error types for cbtop
//!
//! # Yuan Gate Error Handling
//!
//! Reference: Yuan, D., et al. (2014). "Simple Testing Can Prevent Most Critical Failures"
//! - 92% of catastrophic failures caused by `_ => {}` catch-all patterns
//! - All errors are explicit, no catch-all patterns allowed

use thiserror::Error;

/// Main error type for cbtop
#[derive(Error, Debug)]
pub enum CbtopError {
    /// Terminal I/O error
    #[error("Terminal error: {0}")]
    Terminal(#[from] std::io::Error),

    /// Configuration error
    #[error("Configuration error: {0}")]
    Config(String),

    /// File I/O error (string message)
    #[error("I/O error: {0}")]
    Io(String),

    /// Brick verification failed
    #[error("Brick verification failed: {brick_name} - {reason}")]
    BrickVerification { brick_name: String, reason: String },

    /// Collector error
    #[error("Collector error: {0}")]
    Collector(#[from] CollectorError),

    /// Load generator error
    #[error("Load generator error: {0}")]
    LoadGenerator(#[from] LoadError),

    /// No GPU available
    #[error("No GPU available")]
    NoGpu,

    /// Invalid device index
    #[error("Invalid device index: {0}")]
    InvalidDevice(u32),

    /// Budget exceeded
    #[error("Budget exceeded: {phase} took {elapsed_ms}ms (budget: {budget_ms}ms)")]
    BudgetExceeded { phase: String, elapsed_ms: u64, budget_ms: u32 },

    /// Render error
    #[error("Render error: {0}")]
    Render(String),
}

/// Collector-specific errors
#[derive(Error, Debug)]
pub enum CollectorError {
    /// Failed to read /proc filesystem
    #[error("Failed to read /proc: {0}")]
    ProcFs(String),

    /// Failed to read sysfs
    #[error("Failed to read sysfs: {0}")]
    SysFs(String),

    /// NVML error
    #[error("NVML error: {0}")]
    Nvml(String),

    /// wgpu error
    #[error("wgpu error: {0}")]
    Wgpu(String),

    /// Data source not available
    #[error("Data source not available: {0}")]
    NotAvailable(String),

    /// Parse error
    #[error("Parse error: {0}")]
    Parse(String),
}

/// Load generator errors
#[derive(Error, Debug)]
pub enum LoadError {
    /// Backend not available
    #[error("Backend not available: {0}")]
    BackendNotAvailable(String),

    /// Allocation failed
    #[error("Allocation failed: {0}")]
    AllocationFailed(String),

    /// Kernel launch failed
    #[error("Kernel launch failed: {0}")]
    KernelLaunchFailed(String),

    /// Synchronization failed
    #[error("Synchronization failed: {0}")]
    SyncFailed(String),

    /// Invalid configuration
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),
}

/// Result type alias for cbtop
pub type Result<T> = std::result::Result<T, CbtopError>;
