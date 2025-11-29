//! TSP-specific error types with actionable hints.
//!
//! Toyota Way Principle: *Jidoka* - Stop immediately on errors, provide clear diagnostics.

use std::path::PathBuf;

/// TSP-specific errors with actionable hints
#[derive(Debug)]
pub enum TspError {
    /// Invalid .apr file format
    InvalidFormat { message: String, hint: String },
    /// Checksum verification failed
    ChecksumMismatch { expected: u32, computed: u32 },
    /// Instance parsing failed
    ParseError {
        file: PathBuf,
        line: Option<usize>,
        cause: String,
    },
    /// Invalid instance data
    InvalidInstance { message: String },
    /// Solver failed to find solution
    SolverFailed { algorithm: String, reason: String },
    /// Budget exhausted without convergence
    BudgetExhausted { evaluations: usize, best_found: f64 },
    /// I/O error
    Io(std::io::Error),
}

impl std::fmt::Display for TspError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidFormat { message, hint } => {
                write!(f, "Invalid .apr format: {message}\nHint: {hint}")
            }
            Self::ChecksumMismatch { expected, computed } => {
                write!(
                    f,
                    "Model file corrupted: checksum mismatch\n\
                     Expected: 0x{expected:08X}, Computed: 0x{computed:08X}\n\
                     Hint: Re-train the model or restore from backup"
                )
            }
            Self::ParseError { file, line, cause } => {
                if let Some(line_num) = line {
                    write!(
                        f,
                        "Parse error in {} at line {}: {}",
                        file.display(),
                        line_num,
                        cause
                    )
                } else {
                    write!(f, "Parse error in {}: {}", file.display(), cause)
                }
            }
            Self::InvalidInstance { message } => {
                write!(f, "Invalid TSP instance: {message}")
            }
            Self::SolverFailed { algorithm, reason } => {
                write!(f, "{algorithm} solver failed: {reason}")
            }
            Self::BudgetExhausted {
                evaluations,
                best_found,
            } => {
                write!(
                    f,
                    "Budget exhausted after {evaluations} evaluations\n\
                     Best solution found: {best_found:.2}\n\
                     Hint: Increase --iterations or --timeout"
                )
            }
            Self::Io(e) => write!(f, "I/O error: {e}"),
        }
    }
}

impl std::error::Error for TspError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for TspError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

/// Result type alias for TSP operations
pub type TspResult<T> = Result<T, TspError>;

#[cfg(test)]
mod tests {
    use super::*;
    use std::error::Error;

    #[test]
    fn test_invalid_format_display() {
        let err = TspError::InvalidFormat {
            message: "Missing magic number".into(),
            hint: "Ensure file starts with APR\\x00".into(),
        };
        let msg = err.to_string();
        assert!(msg.contains("Invalid .apr format"));
        assert!(msg.contains("Missing magic number"));
        assert!(msg.contains("Hint:"));
    }

    #[test]
    fn test_checksum_mismatch_display() {
        let err = TspError::ChecksumMismatch {
            expected: 0xDEAD_BEEF,
            computed: 0xCAFE_BABE,
        };
        let msg = err.to_string();
        assert!(msg.contains("checksum mismatch"));
        assert!(msg.contains("DEADBEEF"));
        assert!(msg.contains("CAFEBABE"));
        assert!(msg.contains("Re-train"));
    }

    #[test]
    fn test_parse_error_with_line() {
        let err = TspError::ParseError {
            file: PathBuf::from("test.tsp"),
            line: Some(42),
            cause: "Invalid coordinate".into(),
        };
        let msg = err.to_string();
        assert!(msg.contains("test.tsp"));
        assert!(msg.contains("line 42"));
        assert!(msg.contains("Invalid coordinate"));
    }

    #[test]
    fn test_parse_error_without_line() {
        let err = TspError::ParseError {
            file: PathBuf::from("test.tsp"),
            line: None,
            cause: "File not found".into(),
        };
        let msg = err.to_string();
        assert!(msg.contains("test.tsp"));
        assert!(!msg.contains("line"));
    }

    #[test]
    fn test_invalid_instance_display() {
        let err = TspError::InvalidInstance {
            message: "Dimension must be positive".into(),
        };
        let msg = err.to_string();
        assert!(msg.contains("Invalid TSP instance"));
        assert!(msg.contains("Dimension must be positive"));
    }

    #[test]
    fn test_solver_failed_display() {
        let err = TspError::SolverFailed {
            algorithm: "ACO".into(),
            reason: "No feasible tour found".into(),
        };
        let msg = err.to_string();
        assert!(msg.contains("ACO solver failed"));
        assert!(msg.contains("No feasible tour"));
    }

    #[test]
    fn test_budget_exhausted_display() {
        let err = TspError::BudgetExhausted {
            evaluations: 10000,
            best_found: 12345.67,
        };
        let msg = err.to_string();
        assert!(msg.contains("10000 evaluations"));
        assert!(msg.contains("12345.67"));
        assert!(msg.contains("--iterations"));
    }

    #[test]
    fn test_io_error_conversion() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file missing");
        let tsp_err: TspError = io_err.into();
        assert!(matches!(tsp_err, TspError::Io(_)));
        let msg = tsp_err.to_string();
        assert!(msg.contains("I/O error"));
    }

    #[test]
    fn test_error_source() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "test");
        let tsp_err = TspError::Io(io_err);
        assert!(tsp_err.source().is_some());

        let other_err = TspError::InvalidInstance {
            message: "test".into(),
        };
        assert!(other_err.source().is_none());
    }
}
