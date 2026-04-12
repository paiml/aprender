//! Error types for apr-qa-runner

use thiserror::Error;

/// Result type alias
pub type Result<T> = std::result::Result<T, Error>;

/// Errors that can occur during playbook execution
#[derive(Debug, Error)]
pub enum Error {
    /// Playbook parsing error
    #[error("Playbook parse error: {0}")]
    PlaybookParseError(String),

    /// Command execution failed
    #[error("Command failed: {command} (exit code: {exit_code})")]
    CommandFailed {
        /// The command that failed
        command: String,
        /// Exit code
        exit_code: i32,
        /// Stderr output
        stderr: String,
    },

    /// Command timed out
    #[error("Command timed out after {timeout_ms}ms: {command}")]
    Timeout {
        /// The command that timed out
        command: String,
        /// Timeout in milliseconds
        timeout_ms: u64,
    },

    /// Gateway check failed
    #[error("Gateway check failed: {gate}")]
    GatewayFailed {
        /// The gate that failed
        gate: String,
        /// Reason for failure
        reason: String,
    },

    /// IO error
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    /// IO error (non-from variant for conversion module)
    #[error("IO error: {0}")]
    Io(std::io::Error),

    /// Execution error
    #[error("Execution error: {0}")]
    Execution(String),

    /// Execution failed with details
    #[error("Execution failed: {command} - {reason}")]
    ExecutionFailed {
        /// The command that failed
        command: String,
        /// Reason for failure
        reason: String,
    },

    /// Serialization error
    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),

    /// YAML error
    #[error("YAML error: {0}")]
    YamlError(#[from] serde_yaml::Error),

    /// Generator error
    #[error("Generator error: {0}")]
    GeneratorError(#[from] apr_qa_gen::Error),

    /// Provenance validation error (PMAT-PROV-001)
    #[error("Provenance error: {0}")]
    Provenance(#[from] crate::provenance::ProvenanceError),

    /// Validation error (PMAT-266: naming convention, schema validation)
    #[error("Validation error: {0}")]
    Validation(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = Error::CommandFailed {
            command: "apr run".to_string(),
            exit_code: 1,
            stderr: "error".to_string(),
        };
        assert!(err.to_string().contains("apr run"));
        assert!(err.to_string().contains("exit code: 1"));
    }

    #[test]
    fn test_timeout_error() {
        let err = Error::Timeout {
            command: "apr serve".to_string(),
            timeout_ms: 30000,
        };
        assert!(err.to_string().contains("30000ms"));
    }

    #[test]
    fn test_validation_error() {
        let err = Error::Validation("Invalid playbook name".to_string());
        assert!(err.to_string().contains("Validation error"));
        assert!(err.to_string().contains("Invalid playbook name"));
    }
}
