//! Error types for apr-qa-gen

use thiserror::Error;

/// Result type alias for apr-qa-gen operations
pub type Result<T> = std::result::Result<T, Error>;

/// Errors that can occur during scenario generation
#[derive(Debug, Error)]
pub enum Error {
    /// Model not found in registry
    #[error("Model not found: {0}")]
    ModelNotFound(String),

    /// Invalid scenario configuration
    #[error("Invalid scenario: {0}")]
    InvalidScenario(String),

    /// Oracle evaluation failed
    #[error("Oracle error: {0}")]
    OracleError(String),

    /// Serialization error
    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),

    /// YAML serialization error
    #[error("YAML error: {0}")]
    YamlError(#[from] serde_yaml::Error),

    /// IO error
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = Error::ModelNotFound("test-model".to_string());
        assert_eq!(err.to_string(), "Model not found: test-model");
    }

    #[test]
    fn test_error_from_io() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let err: Error = io_err.into();
        assert!(matches!(err, Error::IoError(_)));
    }

    #[test]
    fn test_error_invalid_scenario() {
        let err = Error::InvalidScenario("bad config".to_string());
        assert_eq!(err.to_string(), "Invalid scenario: bad config");
    }

    #[test]
    fn test_error_oracle_error() {
        let err = Error::OracleError("oracle failed".to_string());
        assert_eq!(err.to_string(), "Oracle error: oracle failed");
    }

    #[test]
    fn test_error_from_serde_json() {
        let json_err: serde_json::Error = serde_json::from_str::<i32>("not json").unwrap_err();
        let err: Error = json_err.into();
        assert!(matches!(err, Error::SerializationError(_)));
        assert!(err.to_string().contains("Serialization error"));
    }

    #[test]
    fn test_error_from_serde_yaml() {
        let yaml_err: serde_yaml::Error = serde_yaml::from_str::<i32>("not: [yaml").unwrap_err();
        let err: Error = yaml_err.into();
        assert!(matches!(err, Error::YamlError(_)));
        assert!(err.to_string().contains("YAML error"));
    }

    #[test]
    fn test_error_debug() {
        let err = Error::ModelNotFound("test".to_string());
        let debug_str = format!("{err:?}");
        assert!(debug_str.contains("ModelNotFound"));
    }
}
