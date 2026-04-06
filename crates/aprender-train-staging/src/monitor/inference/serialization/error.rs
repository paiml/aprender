//! Serialization error types.

/// Serialization error
#[derive(Debug)]
pub enum SerializationError {
    /// Invalid format
    InvalidFormat(String),
    /// Version mismatch
    VersionMismatch { expected: u8, actual: u8 },
    /// JSON error
    Json(serde_json::Error),
    /// IO error
    Io(std::io::Error),
}

impl std::fmt::Display for SerializationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SerializationError::InvalidFormat(msg) => write!(f, "Invalid format: {msg}"),
            SerializationError::VersionMismatch { expected, actual } => {
                write!(f, "Version mismatch: expected {expected}, got {actual}")
            }
            SerializationError::Json(e) => write!(f, "JSON error: {e}"),
            SerializationError::Io(e) => write!(f, "IO error: {e}"),
        }
    }
}

impl std::error::Error for SerializationError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            SerializationError::Json(e) => Some(e),
            SerializationError::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<serde_json::Error> for SerializationError {
    fn from(e: serde_json::Error) -> Self {
        SerializationError::Json(e)
    }
}

impl From<std::io::Error> for SerializationError {
    fn from(e: std::io::Error) -> Self {
        SerializationError::Io(e)
    }
}
