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
}
