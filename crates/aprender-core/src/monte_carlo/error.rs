//! Error types for Monte Carlo simulations
//!
//! Implements Jidoka (自働化) principle: errors provide actionable context
//! and fail fast with clear diagnostic messages.

use std::fmt;

/// Result type for Monte Carlo operations
pub type Result<T> = std::result::Result<T, MonteCarloError>;

/// Errors that can occur during Monte Carlo simulation
#[derive(Debug, Clone)]
pub enum MonteCarloError {
    /// CSV parsing error
    CsvParse {
        /// Line number where error occurred
        line: usize,
        /// Column name or index
        column: String,
        /// Description of the parse error
        message: String,
    },

    /// Required field missing from data
    MissingField {
        /// Name of the missing field
        field: String,
        /// Hint for resolution
        hint: String,
    },

    /// Invalid value in data
    InvalidValue {
        /// Field containing invalid value
        field: String,
        /// The invalid value as string
        value: String,
        /// Expected format or range
        expected: String,
    },

    /// IO error during file operations
    IoError {
        /// Path to file
        path: String,
        /// Error message
        message: String,
    },

    /// Insufficient data for analysis
    InsufficientData {
        /// Number of samples provided
        provided: usize,
        /// Minimum required
        required: usize,
        /// Context for requirement
        context: String,
    },

    /// Simulation did not converge
    ConvergenceFailure {
        /// Number of iterations run
        iterations: usize,
        /// Current precision achieved
        achieved_precision: f64,
        /// Target precision
        target_precision: f64,
    },

    /// Invalid model parameters
    InvalidModelParams {
        /// Parameter name
        param: String,
        /// Value provided
        value: f64,
        /// Constraint violated
        constraint: String,
    },

    /// Correlation matrix is not valid
    InvalidCorrelationMatrix {
        /// Reason for invalidity
        reason: String,
    },

    /// Invalid date format or range
    InvalidDate {
        /// The invalid date string
        value: String,
        /// Expected format
        expected_format: String,
    },

    /// Confidence level out of valid range
    InvalidConfidenceLevel {
        /// The invalid value
        value: f64,
    },

    /// Numerical computation error
    NumericalError {
        /// Operation that failed
        operation: String,
        /// Error description
        message: String,
    },

    /// Empty data provided
    EmptyData {
        /// Context describing what was empty
        context: String,
    },
}

impl fmt::Display for MonteCarloError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CsvParse {
                line,
                column,
                message,
            } => {
                write!(
                    f,
                    "CSV parse error at line {line}, column '{column}': {message}"
                )
            }

            Self::MissingField { field, hint } => {
                write!(f, "Missing required field '{field}'. Hint: {hint}")
            }

            Self::InvalidValue {
                field,
                value,
                expected,
            } => {
                write!(
                    f,
                    "Invalid value '{value}' for field '{field}'. Expected: {expected}"
                )
            }

            Self::IoError { path, message } => {
                write!(f, "IO error for '{path}': {message}")
            }

            Self::InsufficientData {
                provided,
                required,
                context,
            } => {
                write!(
                    f,
                    "Insufficient data: {provided} samples provided, {required} required. Context: {context}"
                )
            }

            Self::ConvergenceFailure {
                iterations,
                achieved_precision,
                target_precision,
            } => {
                write!(
                    f,
                    "Simulation did not converge after {iterations} iterations. \
                     Achieved precision: {achieved_precision:.4}, target: {target_precision:.4}"
                )
            }

            Self::InvalidModelParams {
                param,
                value,
                constraint,
            } => {
                write!(
                    f,
                    "Invalid parameter '{param}' = {value}. Constraint: {constraint}"
                )
            }

            Self::InvalidCorrelationMatrix { reason } => {
                write!(f, "Invalid correlation matrix: {reason}")
            }

            Self::InvalidDate {
                value,
                expected_format,
            } => {
                write!(
                    f,
                    "Invalid date '{value}'. Expected format: {expected_format}"
                )
            }

            Self::InvalidConfidenceLevel { value } => {
                write!(
                    f,
                    "Invalid confidence level {value}. Must be in range (0, 1)"
                )
            }

            Self::NumericalError { operation, message } => {
                write!(f, "Numerical error in {operation}: {message}")
            }

            Self::EmptyData { context } => {
                write!(f, "Empty data provided: {context}")
            }
        }
    }
}

impl std::error::Error for MonteCarloError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = MonteCarloError::MissingField {
            field: "revenue".to_string(),
            hint: "Ensure CSV has 'revenue' column".to_string(),
        };
        let msg = format!("{err}");
        assert!(msg.contains("revenue"));
        assert!(msg.contains("Hint"));
    }

    #[test]
    fn test_csv_parse_error() {
        let err = MonteCarloError::CsvParse {
            line: 5,
            column: "price".to_string(),
            message: "expected number".to_string(),
        };
        let msg = format!("{err}");
        assert!(msg.contains("line 5"));
        assert!(msg.contains("price"));
    }

    #[test]
    fn test_convergence_failure() {
        let err = MonteCarloError::ConvergenceFailure {
            iterations: 10000,
            achieved_precision: 0.05,
            target_precision: 0.01,
        };
        let msg = format!("{err}");
        assert!(msg.contains("10000"));
        assert!(msg.contains("0.05"));
    }

    #[test]
    fn test_invalid_value_error() {
        let err = MonteCarloError::InvalidValue {
            field: "rate".to_string(),
            value: "-0.5".to_string(),
            expected: "positive number".to_string(),
        };
        let msg = format!("{err}");
        assert!(msg.contains("rate"));
        assert!(msg.contains("-0.5"));
        assert!(msg.contains("positive"));
    }

    #[test]
    fn test_io_error() {
        let err = MonteCarloError::IoError {
            path: "/tmp/data.csv".to_string(),
            message: "file not found".to_string(),
        };
        let msg = format!("{err}");
        assert!(msg.contains("/tmp/data.csv"));
        assert!(msg.contains("file not found"));
    }

    #[test]
    fn test_insufficient_data_error() {
        let err = MonteCarloError::InsufficientData {
            provided: 5,
            required: 100,
            context: "VaR calculation".to_string(),
        };
        let msg = format!("{err}");
        assert!(msg.contains("5"));
        assert!(msg.contains("100"));
        assert!(msg.contains("VaR"));
    }

    #[test]
    fn test_invalid_model_params_error() {
        let err = MonteCarloError::InvalidModelParams {
            param: "volatility".to_string(),
            value: -0.2,
            constraint: "must be non-negative".to_string(),
        };
        let msg = format!("{err}");
        assert!(msg.contains("volatility"));
        assert!(msg.contains("-0.2"));
        assert!(msg.contains("non-negative"));
    }

    #[test]
    fn test_invalid_correlation_matrix_error() {
        let err = MonteCarloError::InvalidCorrelationMatrix {
            reason: "not positive semi-definite".to_string(),
        };
        let msg = format!("{err}");
        assert!(msg.contains("positive semi-definite"));
    }

    #[test]
    fn test_invalid_date_error() {
        let err = MonteCarloError::InvalidDate {
            value: "2024-13-45".to_string(),
            expected_format: "YYYY-MM-DD".to_string(),
        };
        let msg = format!("{err}");
        assert!(msg.contains("2024-13-45"));
        assert!(msg.contains("YYYY-MM-DD"));
    }

    #[test]
    fn test_invalid_confidence_level_error() {
        let err = MonteCarloError::InvalidConfidenceLevel { value: 1.5 };
        let msg = format!("{err}");
        assert!(msg.contains("1.5"));
        assert!(msg.contains("(0, 1)"));
    }

    #[test]
    fn test_numerical_error() {
        let err = MonteCarloError::NumericalError {
            operation: "matrix inversion".to_string(),
            message: "singular matrix".to_string(),
        };
        let msg = format!("{err}");
        assert!(msg.contains("matrix inversion"));
        assert!(msg.contains("singular"));
    }

    #[test]
    fn test_empty_data_error() {
        let err = MonteCarloError::EmptyData {
            context: "returns vector".to_string(),
        };
        let msg = format!("{err}");
        assert!(msg.contains("Empty data"));
        assert!(msg.contains("returns vector"));
    }

    #[test]
    fn test_error_debug() {
        let err = MonteCarloError::EmptyData {
            context: "test".to_string(),
        };
        let debug = format!("{err:?}");
        assert!(debug.contains("EmptyData"));
    }

    #[test]
    fn test_error_clone() {
        let err = MonteCarloError::InvalidConfidenceLevel { value: 0.99 };
        let cloned = err.clone();
        assert_eq!(format!("{err}"), format!("{cloned}"));
    }

    #[test]
    fn test_error_is_std_error() {
        let err: Box<dyn std::error::Error> = Box::new(MonteCarloError::EmptyData {
            context: "test".to_string(),
        });
        assert!(err.to_string().contains("Empty data"));
    }
}
