//! Error types for the CLI

use thiserror::Error;

/// Result type for CLI operations
pub type CliResult<T> = Result<T, CliError>;

/// Errors that can occur in the CLI
#[derive(Debug, Error)]
pub enum CliError {
    /// Configuration error
    #[error("Configuration error: {message}")]
    Config {
        /// Error message
        message: String,
    },

    /// Test execution error
    #[error("Test execution failed: {message}")]
    TestExecution {
        /// Error message
        message: String,
    },

    /// IO error
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Probar library error
    #[error("Probar error: {0}")]
    Probar(#[from] jugar_probar::ProbarError),

    /// Invalid argument
    #[error("Invalid argument: {message}")]
    InvalidArgument {
        /// Error message
        message: String,
    },

    /// Report generation error
    #[error("Report generation failed: {message}")]
    ReportGeneration {
        /// Error message
        message: String,
    },

    /// Recording error
    #[error("Recording failed: {message}")]
    Recording {
        /// Error message
        message: String,
    },

    /// Generic error for extensible error handling
    #[error("{0}")]
    Generic(String),
}

impl CliError {
    /// Create a configuration error
    #[must_use]
    pub fn config(message: impl Into<String>) -> Self {
        Self::Config {
            message: message.into(),
        }
    }

    /// Create a test execution error
    #[must_use]
    pub fn test_execution(message: impl Into<String>) -> Self {
        Self::TestExecution {
            message: message.into(),
        }
    }

    /// Create an invalid argument error
    #[must_use]
    pub fn invalid_argument(message: impl Into<String>) -> Self {
        Self::InvalidArgument {
            message: message.into(),
        }
    }

    /// Create a report generation error
    #[must_use]
    pub fn report_generation(message: impl Into<String>) -> Self {
        Self::ReportGeneration {
            message: message.into(),
        }
    }

    /// Create a recording error
    #[must_use]
    pub fn recording(message: impl Into<String>) -> Self {
        Self::Recording {
            message: message.into(),
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn test_config_error() {
        let err = CliError::config("bad config");
        assert!(err.to_string().contains("Configuration"));
        assert!(err.to_string().contains("bad config"));
    }

    #[test]
    fn test_test_execution_error() {
        let err = CliError::test_execution("test failed");
        assert!(err.to_string().contains("Test execution"));
    }

    #[test]
    fn test_invalid_argument_error() {
        let err = CliError::invalid_argument("bad arg");
        assert!(err.to_string().contains("Invalid argument"));
    }

    #[test]
    fn test_report_generation_error() {
        let err = CliError::report_generation("report failed");
        assert!(err.to_string().contains("Report"));
    }

    #[test]
    fn test_recording_error() {
        let err = CliError::recording("recording failed");
        assert!(err.to_string().contains("Recording"));
    }

    #[test]
    fn test_io_error_from() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let cli_err: CliError = io_err.into();
        assert!(cli_err.to_string().contains("I/O"));
    }
}
