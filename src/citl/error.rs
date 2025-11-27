//! CITL-specific error types.
//!
//! Provides rich error context for CITL operations including compiler
//! timeouts, parse failures, and fix generation errors.

use std::fmt;
use std::path::PathBuf;
use std::time::Duration;

/// CITL-specific error type.
///
/// Designed for NASA-level fault tolerance with detailed context
/// for debugging and recovery.
#[derive(Debug)]
pub enum CITLError {
    /// Compiler process timed out.
    CompilerTimeout {
        /// Timeout duration
        timeout: Duration,
        /// Partial output if available
        partial_output: Option<String>,
    },

    /// Compiler binary not found.
    CompilerNotFound {
        /// Path searched
        path: PathBuf,
        /// Compiler name
        compiler: String,
    },

    /// Failed to parse compiler output.
    ParseError {
        /// Raw output that couldn't be parsed
        raw: String,
        /// Parse error details
        details: String,
    },

    /// No fix found for the given error.
    NoFixFound {
        /// Error code we tried to fix
        error_code: String,
        /// Number of candidates tried
        candidates_tried: usize,
    },

    /// Maximum iterations exceeded without success.
    MaxIterationsExceeded {
        /// Iterations attempted
        iterations: usize,
        /// Remaining error count
        remaining_errors: usize,
    },

    /// Configuration error.
    ConfigurationError {
        /// Error description
        message: String,
    },

    /// I/O error.
    Io(std::io::Error),

    /// Pattern library error.
    PatternLibraryError {
        /// Error description
        message: String,
    },

    /// Unsupported source language.
    UnsupportedLanguage {
        /// Language name
        language: String,
    },

    /// Compilation failed (wraps the failure result).
    CompilationFailed {
        /// Number of errors
        error_count: usize,
        /// First error message
        first_error: String,
    },
}

impl fmt::Display for CITLError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CITLError::CompilerTimeout {
                timeout,
                partial_output,
            } => {
                write!(f, "Compiler timed out after {timeout:?}")?;
                if let Some(output) = partial_output {
                    write!(
                        f,
                        " (partial output: {}...)",
                        &output[..output.len().min(100)]
                    )?;
                }
                Ok(())
            }
            CITLError::CompilerNotFound { path, compiler } => {
                write!(f, "Compiler '{compiler}' not found at {}", path.display())
            }
            CITLError::ParseError { raw, details } => {
                write!(
                    f,
                    "Failed to parse compiler output: {} (raw: {}...)",
                    details,
                    &raw[..raw.len().min(50)]
                )
            }
            CITLError::NoFixFound {
                error_code,
                candidates_tried,
            } => {
                write!(
                    f,
                    "No fix found for {error_code} after trying {candidates_tried} candidates"
                )
            }
            CITLError::MaxIterationsExceeded {
                iterations,
                remaining_errors,
            } => {
                write!(
                    f,
                    "Max iterations ({iterations}) exceeded with {remaining_errors} errors remaining"
                )
            }
            CITLError::ConfigurationError { message } => {
                write!(f, "Configuration error: {message}")
            }
            CITLError::Io(e) => write!(f, "I/O error: {e}"),
            CITLError::PatternLibraryError { message } => {
                write!(f, "Pattern library error: {message}")
            }
            CITLError::UnsupportedLanguage { language } => {
                write!(f, "Unsupported language: {language}")
            }
            CITLError::CompilationFailed {
                error_count,
                first_error,
            } => {
                write!(
                    f,
                    "Compilation failed with {error_count} errors: {first_error}"
                )
            }
        }
    }
}

impl std::error::Error for CITLError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            CITLError::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for CITLError {
    fn from(err: std::io::Error) -> Self {
        CITLError::Io(err)
    }
}

/// Convenience type alias for CITL Results.
pub type CITLResult<T> = Result<T, CITLError>;

#[cfg(test)]
mod tests {
    use super::*;
    use std::error::Error;

    #[test]
    fn test_compiler_timeout_display() {
        let err = CITLError::CompilerTimeout {
            timeout: Duration::from_secs(30),
            partial_output: None,
        };
        assert!(err.to_string().contains("timed out"));
        assert!(err.to_string().contains("30"));
    }

    #[test]
    fn test_compiler_timeout_with_partial_output() {
        let err = CITLError::CompilerTimeout {
            timeout: Duration::from_secs(30),
            partial_output: Some("error[E0308]: mismatched types".to_string()),
        };
        let msg = err.to_string();
        assert!(msg.contains("timed out"));
        assert!(msg.contains("partial output"));
    }

    #[test]
    fn test_compiler_not_found_display() {
        let err = CITLError::CompilerNotFound {
            path: PathBuf::from("/usr/bin/rustc"),
            compiler: "rustc".to_string(),
        };
        assert!(err.to_string().contains("rustc"));
        assert!(err.to_string().contains("not found"));
    }

    #[test]
    fn test_parse_error_display() {
        let err = CITLError::ParseError {
            raw: "garbage output from compiler".to_string(),
            details: "expected JSON".to_string(),
        };
        assert!(err.to_string().contains("expected JSON"));
        assert!(err.to_string().contains("garbage"));
    }

    #[test]
    fn test_no_fix_found_display() {
        let err = CITLError::NoFixFound {
            error_code: "E0308".to_string(),
            candidates_tried: 10,
        };
        assert!(err.to_string().contains("E0308"));
        assert!(err.to_string().contains("10 candidates"));
    }

    #[test]
    fn test_max_iterations_display() {
        let err = CITLError::MaxIterationsExceeded {
            iterations: 20,
            remaining_errors: 5,
        };
        assert!(err.to_string().contains("20"));
        assert!(err.to_string().contains("5 errors"));
    }

    #[test]
    fn test_configuration_error_display() {
        let err = CITLError::ConfigurationError {
            message: "missing compiler".to_string(),
        };
        assert!(err.to_string().contains("missing compiler"));
    }

    #[test]
    fn test_io_error_conversion() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let citl_err: CITLError = io_err.into();
        assert!(matches!(citl_err, CITLError::Io(_)));
        assert!(citl_err.to_string().contains("file not found"));
    }

    #[test]
    fn test_pattern_library_error_display() {
        let err = CITLError::PatternLibraryError {
            message: "corrupted index".to_string(),
        };
        assert!(err.to_string().contains("corrupted index"));
    }

    #[test]
    fn test_unsupported_language_display() {
        let err = CITLError::UnsupportedLanguage {
            language: "Cobol".to_string(),
        };
        assert!(err.to_string().contains("Cobol"));
        assert!(err.to_string().contains("Unsupported"));
    }

    #[test]
    fn test_compilation_failed_display() {
        let err = CITLError::CompilationFailed {
            error_count: 3,
            first_error: "mismatched types".to_string(),
        };
        assert!(err.to_string().contains("3 errors"));
        assert!(err.to_string().contains("mismatched types"));
    }

    #[test]
    fn test_error_source() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "test");
        let citl_err = CITLError::Io(io_err);
        assert!(citl_err.source().is_some());

        let other_err = CITLError::ConfigurationError {
            message: "test".to_string(),
        };
        assert!(other_err.source().is_none());
    }
}
