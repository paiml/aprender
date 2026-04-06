//! Error types for validation operations
//!
//! Error codes per specification Section 9.1

use std::path::PathBuf;
use thiserror::Error;

/// Errors that can occur during validation operations
#[derive(Error, Debug)]
pub enum ValidateError {
    /// V001: Baseline directory not found
    #[error("V001: Baseline directory not found: {path}")]
    BaselineNotFound { path: PathBuf },

    /// V002: Invalid baseline manifest
    #[error("V002: Invalid baseline manifest: {reason}")]
    InvalidManifest { reason: String },

    /// V003: Baseline version mismatch
    #[error("V003: Baseline version mismatch: expected {expected}, found {found}")]
    VersionMismatch { expected: String, found: String },

    /// V004: Syscall count mismatch
    #[error("V004: Syscall count mismatch: baseline={baseline}, current={current}")]
    SyscallCountMismatch { baseline: usize, current: usize },

    /// V005: Syscall sequence mismatch
    #[error(
        "V005: Syscall sequence mismatch at index {index}: expected {expected}, found {found}"
    )]
    SequenceMismatch { index: usize, expected: String, found: String },

    /// V006: Timing regression detected
    #[error("V006: Timing regression: {syscall} ({baseline_ms:.2}ms -> {actual_ms:.2}ms, {delta_percent:+.1}%)")]
    TimingRegression { syscall: String, baseline_ms: f64, actual_ms: f64, delta_percent: f64 },

    /// V007: Canary output mismatch
    #[error("V007: Canary output mismatch: expected {expected:?}, found {found:?}")]
    CanaryMismatch { expected: String, found: String },

    /// V008: APR model not found
    #[error("V008: APR model not found: {path}")]
    AprModelNotFound { path: PathBuf },

    /// V009: Tensor statistics violation
    #[error("V009: Tensor statistics violation: {tensor} - {reason}")]
    TensorStatsViolation { tensor: String, reason: String },

    /// V010: Configuration error
    #[error("V010: Configuration error: {reason}")]
    ConfigError { reason: String },

    /// IO error wrapper
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    /// JSON serialization error
    #[error("JSON error: {0}")]
    JsonError(#[from] serde_json::Error),

    /// Trace capture error
    #[error("Trace capture failed: {0}")]
    TraceCaptureError(String),
}

/// Result type for validation operations
pub type Result<T> = std::result::Result<T, ValidateError>;
