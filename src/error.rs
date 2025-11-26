//! Error types for Aprender operations.
//!
//! Provides rich error context for library consumers.

use std::fmt;

/// Main error type for Aprender operations.
///
/// Provides detailed context about failures including dimension mismatches,
/// singular matrices, convergence issues, and invalid hyperparameters.
///
/// # Examples
///
/// ```
/// use aprender::error::AprenderError;
///
/// let err = AprenderError::DimensionMismatch {
///     expected: "100x10".to_string(),
///     actual: "100x5".to_string(),
/// };
/// assert!(err.to_string().contains("dimension mismatch"));
/// ```
#[derive(Debug)]
pub enum AprenderError {
    /// Matrix/vector dimensions don't match for the operation.
    DimensionMismatch {
        /// Expected dimensions description
        expected: String,
        /// Actual dimensions found
        actual: String,
    },

    /// Matrix is singular (non-invertible).
    SingularMatrix {
        /// Determinant value (close to zero)
        det: f64,
    },

    /// Optimization failed to converge within iteration limit.
    ConvergenceFailure {
        /// Number of iterations attempted
        iterations: usize,
        /// Final loss value
        final_loss: f64,
    },

    /// Invalid hyperparameter value provided.
    InvalidHyperparameter {
        /// Parameter name
        param: String,
        /// Provided value
        value: String,
        /// Constraint description
        constraint: String,
    },

    /// Requested compute backend is not available.
    BackendUnavailable {
        /// Backend name (e.g., "GPU", "AVX-512")
        backend: String,
    },

    /// I/O error (file not found, permission denied, etc.).
    Io(std::io::Error),

    /// Serialization/deserialization error.
    Serialization(String),

    /// Generic error with string message.
    Other(String),

    /// Invalid or corrupt model format.
    FormatError {
        /// Error description
        message: String,
    },

    /// Unsupported format version.
    UnsupportedVersion {
        /// Version found
        found: (u8, u8),
        /// Maximum supported version
        supported: (u8, u8),
    },

    /// Checksum verification failed.
    ChecksumMismatch {
        /// Expected checksum
        expected: u32,
        /// Actual checksum
        actual: u32,
    },

    /// Signature verification failed.
    SignatureInvalid {
        /// Reason for failure
        reason: String,
    },

    /// Decryption failed (wrong password or corrupt data).
    DecryptionFailed {
        /// Error details
        message: String,
    },
}

impl fmt::Display for AprenderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AprenderError::DimensionMismatch { expected, actual } => {
                write!(
                    f,
                    "Matrix dimension mismatch: expected {expected}, got {actual}"
                )
            }
            AprenderError::SingularMatrix { det } => {
                write!(
                    f,
                    "Singular matrix detected: determinant = {det}, cannot invert"
                )
            }
            AprenderError::ConvergenceFailure {
                iterations,
                final_loss,
            } => {
                write!(
                    f,
                    "Convergence failure after {iterations} iterations, loss = {final_loss}"
                )
            }
            AprenderError::InvalidHyperparameter {
                param,
                value,
                constraint,
            } => {
                write!(
                    f,
                    "Invalid hyperparameter: {param} = {value}, expected {constraint}"
                )
            }
            AprenderError::BackendUnavailable { backend } => {
                write!(f, "Backend not available: {backend}")
            }
            AprenderError::Io(e) => write!(f, "I/O error: {e}"),
            AprenderError::Serialization(msg) => write!(f, "Serialization error: {msg}"),
            AprenderError::Other(msg) => write!(f, "{msg}"),
            AprenderError::FormatError { message } => {
                write!(f, "Invalid model format: {message}")
            }
            AprenderError::UnsupportedVersion { found, supported } => {
                write!(
                    f,
                    "Unsupported format version: found {}.{}, max supported {}.{}",
                    found.0, found.1, supported.0, supported.1
                )
            }
            AprenderError::ChecksumMismatch { expected, actual } => {
                write!(
                    f,
                    "Checksum mismatch: expected 0x{expected:08X}, got 0x{actual:08X}"
                )
            }
            AprenderError::SignatureInvalid { reason } => {
                write!(f, "Invalid signature: {reason}")
            }
            AprenderError::DecryptionFailed { message } => {
                write!(f, "Decryption failed: {message}")
            }
        }
    }
}

impl std::error::Error for AprenderError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            AprenderError::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for AprenderError {
    fn from(err: std::io::Error) -> Self {
        AprenderError::Io(err)
    }
}

impl From<&str> for AprenderError {
    fn from(msg: &str) -> Self {
        AprenderError::Other(msg.to_string())
    }
}

impl From<String> for AprenderError {
    fn from(msg: String) -> Self {
        AprenderError::Other(msg)
    }
}

#[allow(clippy::cmp_owned)]
impl PartialEq<&str> for AprenderError {
    fn eq(&self, other: &&str) -> bool {
        self.to_string() == *other
    }
}

#[allow(clippy::cmp_owned)]
impl PartialEq<AprenderError> for &str {
    fn eq(&self, other: &AprenderError) -> bool {
        *self == other.to_string()
    }
}

/// Convenience type alias for Results.
pub type Result<T> = std::result::Result<T, AprenderError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dimension_mismatch_display() {
        let err = AprenderError::DimensionMismatch {
            expected: "100x10".to_string(),
            actual: "100x5".to_string(),
        };
        assert!(err.to_string().contains("dimension mismatch"));
        assert!(err.to_string().contains("100x10"));
        assert!(err.to_string().contains("100x5"));
    }

    #[test]
    fn test_singular_matrix_display() {
        let err = AprenderError::SingularMatrix { det: 1e-15 };
        let msg = err.to_string();
        assert!(msg.contains("Singular matrix"));
        assert!(msg.contains("0.000000000000001") || msg.contains("1e-15"));
    }

    #[test]
    fn test_convergence_failure_display() {
        let err = AprenderError::ConvergenceFailure {
            iterations: 100,
            final_loss: 0.42,
        };
        assert!(err.to_string().contains("Convergence failure"));
        assert!(err.to_string().contains("100"));
        assert!(err.to_string().contains("0.42"));
    }

    #[test]
    fn test_invalid_hyperparameter_display() {
        let err = AprenderError::InvalidHyperparameter {
            param: "learning_rate".to_string(),
            value: "-0.1".to_string(),
            constraint: ">0".to_string(),
        };
        assert!(err.to_string().contains("Invalid hyperparameter"));
        assert!(err.to_string().contains("learning_rate"));
        assert!(err.to_string().contains("-0.1"));
        assert!(err.to_string().contains(">0"));
    }

    #[test]
    fn test_backend_unavailable_display() {
        let err = AprenderError::BackendUnavailable {
            backend: "AVX-512".to_string(),
        };
        assert!(err.to_string().contains("Backend not available"));
        assert!(err.to_string().contains("AVX-512"));
    }

    #[test]
    fn test_from_str() {
        let err: AprenderError = "test error".into();
        assert!(matches!(err, AprenderError::Other(_)));
        assert_eq!(err.to_string(), "test error");
    }

    #[test]
    fn test_from_string() {
        let err: AprenderError = "test error".to_string().into();
        assert!(matches!(err, AprenderError::Other(_)));
        assert_eq!(err.to_string(), "test error");
    }
}
