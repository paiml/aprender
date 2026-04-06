//! Cloud storage error types

use thiserror::Error;

/// Cloud storage errors
#[derive(Debug, Error)]
pub enum CloudError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Artifact not found: {0}")]
    NotFound(String),

    #[error("Backend error: {0}")]
    Backend(String),

    #[error("Configuration error: {0}")]
    Config(String),

    #[error("Permission denied: {0}")]
    PermissionDenied(String),

    #[error("Network error: {0}")]
    Network(String),
}

/// Result type for cloud operations
pub type Result<T> = std::result::Result<T, CloudError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cloud_error_display() {
        let io_err =
            CloudError::Io(std::io::Error::new(std::io::ErrorKind::NotFound, "file not found"));
        assert!(io_err.to_string().contains("IO error"));

        let not_found = CloudError::NotFound("abc123".to_string());
        assert!(not_found.to_string().contains("abc123"));

        let backend = CloudError::Backend("connection failed".to_string());
        assert!(backend.to_string().contains("connection failed"));

        let config = CloudError::Config("invalid config".to_string());
        assert!(config.to_string().contains("invalid config"));

        let permission = CloudError::PermissionDenied("access denied".to_string());
        assert!(permission.to_string().contains("access denied"));

        let network = CloudError::Network("timeout".to_string());
        assert!(network.to_string().contains("timeout"));
    }
}
