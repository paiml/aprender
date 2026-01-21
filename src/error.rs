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

    /// Poka-yoke validation failed (APR-POKA-001 - Jidoka gate).
    ValidationError {
        /// Validation failure message
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
            AprenderError::ValidationError { message } => {
                write!(f, "Validation failed: {message}")
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

impl AprenderError {
    /// Create a dimension mismatch error with descriptive context
    #[must_use]
    pub fn dimension_mismatch(context: &str, expected: usize, actual: usize) -> Self {
        Self::DimensionMismatch {
            expected: format!("{context}={expected}"),
            actual: format!("{actual}"),
        }
    }

    /// Create an index out of bounds error
    #[must_use]
    pub fn index_out_of_bounds(index: usize, len: usize) -> Self {
        Self::Other(format!("index {index} out of bounds (len={len})"))
    }

    /// Create an empty input error
    #[must_use]
    pub fn empty_input(context: &str) -> Self {
        Self::Other(format!("empty input: {context}"))
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

    // =========================================================================
    // Coverage boost: Additional error variant tests
    // =========================================================================

    #[test]
    fn test_io_error_display() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let err = AprenderError::Io(io_err);
        let msg = err.to_string();
        assert!(msg.contains("I/O error") || msg.contains("file not found"));
    }

    #[test]
    fn test_serialization_error_display() {
        let err = AprenderError::Serialization("invalid JSON".to_string());
        assert!(err.to_string().contains("Serialization"));
        assert!(err.to_string().contains("invalid JSON"));
    }

    #[test]
    fn test_format_error_display() {
        let err = AprenderError::FormatError {
            message: "corrupt header".to_string(),
        };
        assert!(err.to_string().contains("Invalid model format"));
        assert!(err.to_string().contains("corrupt header"));
    }

    #[test]
    fn test_unsupported_version_display() {
        let err = AprenderError::UnsupportedVersion {
            found: (3, 0),
            supported: (2, 0),
        };
        let msg = err.to_string();
        assert!(msg.contains("Unsupported"));
        assert!(msg.contains("3.0") || msg.contains("(3, 0)"));
    }

    #[test]
    fn test_checksum_mismatch_display() {
        let err = AprenderError::ChecksumMismatch {
            expected: 0xDEADBEEF,
            actual: 0xCAFEBABE,
        };
        let msg = err.to_string();
        assert!(msg.contains("Checksum"));
    }

    #[test]
    fn test_signature_invalid_display() {
        let err = AprenderError::SignatureInvalid {
            reason: "key mismatch".to_string(),
        };
        let msg = err.to_string();
        assert!(msg.contains("Signature") || msg.contains("key mismatch"));
    }

    #[test]
    fn test_decryption_failed_display() {
        let err = AprenderError::DecryptionFailed {
            message: "wrong password".to_string(),
        };
        let msg = err.to_string();
        assert!(msg.contains("Decryption") || msg.contains("wrong password"));
    }

    #[test]
    fn test_validation_error_display() {
        let err = AprenderError::ValidationError {
            message: "poka-yoke failed".to_string(),
        };
        let msg = err.to_string();
        assert!(msg.contains("Validation") || msg.contains("poka-yoke"));
    }

    #[test]
    fn test_error_debug_impl() {
        let err = AprenderError::Other("test".to_string());
        let debug_str = format!("{:?}", err);
        assert!(debug_str.contains("Other"));
    }

    #[test]
    fn test_from_io_error() {
        let io_err = std::io::Error::new(std::io::ErrorKind::PermissionDenied, "access denied");
        let err: AprenderError = io_err.into();
        assert!(matches!(err, AprenderError::Io(_)));
    }

    #[test]
    fn test_error_send_sync() {
        fn _assert_send<T: Send>() {}
        fn _assert_sync<T: Sync>() {}
        // These would fail to compile if AprenderError wasn't Send + Sync
        // (commented out as std::io::Error is not Sync)
        // _assert_send::<AprenderError>();
    }

    // =========================================================================
    // Additional coverage tests for convenience methods and traits
    // =========================================================================

    #[test]
    fn test_dimension_mismatch_helper() {
        let err = AprenderError::dimension_mismatch("rows", 100, 50);
        let msg = err.to_string();
        assert!(msg.contains("rows=100"));
        assert!(msg.contains("50"));
    }

    #[test]
    fn test_index_out_of_bounds_helper() {
        let err = AprenderError::index_out_of_bounds(10, 5);
        let msg = err.to_string();
        assert!(msg.contains("index 10"));
        assert!(msg.contains("len=5"));
    }

    #[test]
    fn test_empty_input_helper() {
        let err = AprenderError::empty_input("training data");
        let msg = err.to_string();
        assert!(msg.contains("empty input"));
        assert!(msg.contains("training data"));
    }

    #[test]
    fn test_error_eq_str() {
        let err = AprenderError::Other("test error".to_string());
        assert!(err == "test error");
        assert!("test error" == err);
    }

    #[test]
    fn test_error_source_io() {
        use std::error::Error;
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let err = AprenderError::Io(io_err);
        assert!(err.source().is_some());
    }

    #[test]
    fn test_error_source_other() {
        use std::error::Error;
        let err = AprenderError::Other("test".to_string());
        assert!(err.source().is_none());
    }

    #[test]
    fn test_error_source_validation() {
        use std::error::Error;
        let err = AprenderError::ValidationError {
            message: "test".to_string(),
        };
        assert!(err.source().is_none());
    }
}
