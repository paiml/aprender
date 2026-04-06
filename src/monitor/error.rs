//! Error types for the TUI monitoring system.
//!
//! This module provides error types for all monitoring operations including
//! metric collection, configuration parsing, and rendering.

use std::io;
use thiserror::Error;

/// Error type for monitoring operations.
///
/// Follows the Popperian falsification criterion #52: Invalid config produces
/// clear error message with line number.
#[derive(Debug, Error)]
pub enum MonitorError {
    /// A metric collector is not available on this system.
    #[error("collector '{0}' is not available on this system")]
    CollectorUnavailable(&'static str),

    /// Failed to collect metrics from a collector.
    #[error("failed to collect metrics from '{collector}': {message}")]
    CollectionFailed {
        /// The collector that failed.
        collector: &'static str,
        /// Error message describing the failure.
        message: String,
    },

    /// Configuration parsing error with line number.
    #[error("configuration error at line {line}: {message}")]
    ConfigParse {
        /// Line number where the error occurred (1-indexed).
        line: usize,
        /// Error message describing the issue.
        message: String,
    },

    /// Configuration file not found.
    #[error("configuration file not found: {0}")]
    ConfigNotFound(String),

    /// Invalid configuration value.
    #[error("invalid configuration value for '{key}': {message}")]
    ConfigInvalid {
        /// The configuration key with invalid value.
        key: String,
        /// Error message describing why the value is invalid.
        message: String,
    },

    /// Theme file not found or invalid.
    #[error("theme error: {0}")]
    ThemeError(String),

    /// Terminal initialization or rendering error.
    #[error("terminal error: {0}")]
    TerminalError(#[from] io::Error),

    /// Ring buffer capacity exceeded (should never happen with bounded buffers).
    #[error("ring buffer capacity exceeded: requested {requested}, capacity {capacity}")]
    BufferOverflow {
        /// Requested operation size.
        requested: usize,
        /// Buffer capacity.
        capacity: usize,
    },

    /// Remote agent connection error.
    #[cfg(feature = "monitor-remote")]
    #[error("remote agent error: {0}")]
    RemoteError(String),

    /// GPU collector error.
    #[cfg(feature = "monitor-nvidia")]
    #[error("NVIDIA GPU error: {0}")]
    NvidiaError(String),

    /// Process not found (for kill/signal operations).
    #[error("process {0} not found")]
    ProcessNotFound(u32),

    /// Permission denied for operation.
    #[error("permission denied: {0}")]
    PermissionDenied(String),
}

/// Result type alias for monitoring operations.
pub type Result<T> = std::result::Result<T, MonitorError>;

// ============================================================================
// Tests - Written FIRST per EXTREME TDD
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Falsification criterion #52: Error includes line number for config errors.
    #[test]
    fn test_config_parse_error_includes_line_number() {
        let err = MonitorError::ConfigParse { line: 42, message: "invalid value".to_string() };
        let display = err.to_string();

        assert!(display.contains("42"), "Error should include line number: {display}");
        assert!(display.contains("invalid value"), "Error should include message: {display}");
    }

    #[test]
    fn test_collector_unavailable_includes_collector_name() {
        let err = MonitorError::CollectorUnavailable("nvidia_gpu");
        let display = err.to_string();

        assert!(display.contains("nvidia_gpu"), "Error should include collector name: {display}");
    }

    #[test]
    fn test_collection_failed_includes_details() {
        let err = MonitorError::CollectionFailed {
            collector: "cpu",
            message: "/proc/stat not readable".to_string(),
        };
        let display = err.to_string();

        assert!(display.contains("cpu"), "Error should include collector: {display}");
        assert!(display.contains("/proc/stat"), "Error should include message: {display}");
    }

    #[test]
    fn test_config_invalid_includes_key() {
        let err = MonitorError::ConfigInvalid {
            key: "update_ms".to_string(),
            message: "must be positive".to_string(),
        };
        let display = err.to_string();

        assert!(display.contains("update_ms"), "Error should include key: {display}");
    }

    #[test]
    fn test_buffer_overflow_includes_sizes() {
        let err = MonitorError::BufferOverflow { requested: 1000, capacity: 500 };
        let display = err.to_string();

        assert!(display.contains("1000"), "Error should include requested size: {display}");
        assert!(display.contains("500"), "Error should include capacity: {display}");
    }

    #[test]
    fn test_io_error_conversion() {
        let io_err = io::Error::new(io::ErrorKind::NotFound, "file not found");
        let monitor_err: MonitorError = io_err.into();

        assert!(
            matches!(monitor_err, MonitorError::TerminalError(_)),
            "Should convert to TerminalError"
        );
    }

    #[test]
    fn test_error_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<MonitorError>();
    }

    #[test]
    fn test_config_not_found() {
        let err = MonitorError::ConfigNotFound("/etc/trueno.toml".to_string());
        let display = err.to_string();

        assert!(display.contains("/etc/trueno.toml"), "Error should include path: {display}");
    }

    #[test]
    fn test_theme_error() {
        let err = MonitorError::ThemeError("invalid color format".to_string());
        let display = err.to_string();

        assert!(
            display.contains("invalid color format"),
            "Error should include message: {display}"
        );
    }

    #[test]
    fn test_process_not_found() {
        let err = MonitorError::ProcessNotFound(12345);
        let display = err.to_string();

        assert!(display.contains("12345"), "Error should include PID: {display}");
    }

    #[test]
    fn test_permission_denied() {
        let err = MonitorError::PermissionDenied("cannot send signal to init".to_string());
        let display = err.to_string();

        assert!(display.contains("cannot send signal"), "Error should include reason: {display}");
    }

    #[test]
    fn test_terminal_error_display() {
        let io_err = io::Error::new(io::ErrorKind::PermissionDenied, "access denied");
        let err: MonitorError = io_err.into();
        let display = err.to_string();

        assert!(display.contains("access denied"), "Error should include IO error: {display}");
    }

    #[test]
    fn test_error_debug_format() {
        let err = MonitorError::CollectorUnavailable("test");
        let debug = format!("{err:?}");
        assert!(debug.contains("CollectorUnavailable"));
    }
}
