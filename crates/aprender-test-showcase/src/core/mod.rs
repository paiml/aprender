//! Core calculator module with 100% test coverage
//!
//! Probar Principles:
//! - Error prevention: Type-safe operations prevent invalid states
//! - Anomaly: Automatic anomaly detection

pub mod evaluator;
pub mod history;
mod operations;
pub mod parser;

pub use operations::{Calculator, Operation};

use std::collections::VecDeque;

/// Result type for calculator operations
pub type CalcResult<T> = Result<T, CalcError>;

/// Calculator error types - exhaustive enum ensures all cases handled
#[derive(Debug, Clone, PartialEq)]
pub enum CalcError {
    /// Division by zero attempted
    DivisionByZero,
    /// Result overflowed (infinity)
    Overflow,
    /// Invalid expression syntax
    ParseError(String),
    /// Empty expression provided
    EmptyExpression,
    /// Invalid result (NaN or other)
    InvalidResult(String),
    /// Anomaly violation detected
    AnomalyViolation(AnomalyViolation),
}

impl std::fmt::Display for CalcError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DivisionByZero => write!(f, "Division by zero"),
            Self::Overflow => write!(f, "Overflow: result exceeds maximum value"),
            Self::ParseError(msg) => write!(f, "Invalid expression: {msg}"),
            Self::EmptyExpression => write!(f, "Empty expression"),
            Self::InvalidResult(msg) => write!(f, "Invalid result: {msg}"),
            Self::AnomalyViolation(v) => write!(f, "Anomaly violation: {v}"),
        }
    }
}

impl std::error::Error for CalcError {}

/// Anomaly violation types - anomalies detected during calculation
#[derive(Debug, Clone, PartialEq)]
pub enum AnomalyViolation {
    /// NaN detected in result
    NaN,
    /// Infinity detected in result
    Infinite,
    /// Result exceeds maximum magnitude
    Overflow(f64),
}

impl std::fmt::Display for AnomalyViolation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NaN => write!(f, "NaN detected"),
            Self::Infinite => write!(f, "Infinite value detected"),
            Self::Overflow(v) => write!(f, "Overflow: {v} exceeds maximum magnitude"),
        }
    }
}

/// Anomaly validator - automatic anomaly detection by Probar
///
/// Implements the Anomaly principle: "Automation with human touch"
/// Detects anomalies and stops the line before defects propagate.
#[derive(Debug, Clone)]
pub struct AnomalyValidator {
    /// Maximum allowed result magnitude
    pub max_magnitude: f64,
    /// Detect NaN/Infinity
    pub check_special_values: bool,
    /// History of recent results for drift detection
    history: VecDeque<f64>,
    /// Maximum history size
    max_history: usize,
}

impl Default for AnomalyValidator {
    fn default() -> Self {
        Self::new()
    }
}

impl AnomalyValidator {
    /// Default maximum magnitude (f64::MAX / 2 for safety margin)
    pub const DEFAULT_MAX_MAGNITUDE: f64 = 1e100;

    /// Creates a new validator with default settings
    #[must_use]
    pub fn new() -> Self {
        Self {
            max_magnitude: Self::DEFAULT_MAX_MAGNITUDE,
            check_special_values: true,
            history: VecDeque::new(),
            max_history: 100,
        }
    }

    /// Creates a validator with custom maximum magnitude
    #[must_use]
    pub fn with_max_magnitude(max_magnitude: f64) -> Self {
        Self {
            max_magnitude,
            check_special_values: true,
            history: VecDeque::new(),
            max_history: 100,
        }
    }

    /// Validates a calculation result (Anomaly check)
    ///
    /// Returns the result if valid, or a violation describing the anomaly.
    pub fn validate(&mut self, result: f64) -> Result<f64, AnomalyViolation> {
        // Check for NaN
        if self.check_special_values && result.is_nan() {
            return Err(AnomalyViolation::NaN);
        }

        // Check for Infinity
        if self.check_special_values && result.is_infinite() {
            return Err(AnomalyViolation::Infinite);
        }

        // Check magnitude bounds
        if result.abs() > self.max_magnitude {
            return Err(AnomalyViolation::Overflow(result));
        }

        // Record in history for drift detection
        self.record_result(result);

        Ok(result)
    }

    /// Records a result in history
    fn record_result(&mut self, result: f64) {
        if self.history.len() >= self.max_history {
            self.history.pop_front();
        }
        self.history.push_back(result);
    }

    /// Returns the history of recent results
    #[must_use]
    pub fn history(&self) -> &VecDeque<f64> {
        &self.history
    }

    /// Clears the history
    pub fn clear_history(&mut self) {
        self.history.clear();
    }

    /// Returns the number of results in history
    #[must_use]
    pub fn history_len(&self) -> usize {
        self.history.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ===== CalcError tests =====

    #[test]
    fn test_calc_error_display_division_by_zero() {
        let err = CalcError::DivisionByZero;
        assert_eq!(format!("{err}"), "Division by zero");
    }

    #[test]
    fn test_calc_error_display_overflow() {
        let err = CalcError::Overflow;
        assert_eq!(format!("{err}"), "Overflow: result exceeds maximum value");
    }

    #[test]
    fn test_calc_error_display_parse_error() {
        let err = CalcError::ParseError("unexpected token".into());
        assert_eq!(format!("{err}"), "Invalid expression: unexpected token");
    }

    #[test]
    fn test_calc_error_display_empty_expression() {
        let err = CalcError::EmptyExpression;
        assert_eq!(format!("{err}"), "Empty expression");
    }

    #[test]
    fn test_calc_error_display_invalid_result() {
        let err = CalcError::InvalidResult("NaN".into());
        assert_eq!(format!("{err}"), "Invalid result: NaN");
    }

    #[test]
    fn test_calc_error_display_jidoka_violation() {
        let err = CalcError::AnomalyViolation(AnomalyViolation::NaN);
        assert_eq!(format!("{err}"), "Anomaly violation: NaN detected");
    }

    #[test]
    fn test_calc_error_is_error_trait() {
        let err: Box<dyn std::error::Error> = Box::new(CalcError::DivisionByZero);
        assert!(err.to_string().contains("Division"));
    }

    // ===== AnomalyViolation tests =====

    #[test]
    fn test_jidoka_violation_display_nan() {
        let v = AnomalyViolation::NaN;
        assert_eq!(format!("{v}"), "NaN detected");
    }

    #[test]
    fn test_jidoka_violation_display_infinite() {
        let v = AnomalyViolation::Infinite;
        assert_eq!(format!("{v}"), "Infinite value detected");
    }

    #[test]
    fn test_jidoka_violation_display_overflow() {
        let v = AnomalyViolation::Overflow(1e200);
        assert!(format!("{v}").contains("exceeds maximum magnitude"));
    }

    // ===== AnomalyValidator tests =====

    #[test]
    fn test_jidoka_validator_new() {
        let v = AnomalyValidator::new();
        assert_eq!(v.max_magnitude, AnomalyValidator::DEFAULT_MAX_MAGNITUDE);
        assert!(v.check_special_values);
        assert!(v.history.is_empty());
    }

    #[test]
    fn test_jidoka_validator_default() {
        let v = AnomalyValidator::default();
        assert_eq!(v.max_magnitude, AnomalyValidator::DEFAULT_MAX_MAGNITUDE);
    }

    #[test]
    fn test_jidoka_validator_with_max_magnitude() {
        let v = AnomalyValidator::with_max_magnitude(100.0);
        assert_eq!(v.max_magnitude, 100.0);
    }

    #[test]
    fn test_jidoka_validate_valid_result() {
        let mut v = AnomalyValidator::new();
        assert_eq!(v.validate(42.0), Ok(42.0));
    }

    #[test]
    fn test_jidoka_validate_nan() {
        let mut v = AnomalyValidator::new();
        assert_eq!(v.validate(f64::NAN), Err(AnomalyViolation::NaN));
    }

    #[test]
    fn test_jidoka_validate_positive_infinity() {
        let mut v = AnomalyValidator::new();
        assert_eq!(v.validate(f64::INFINITY), Err(AnomalyViolation::Infinite));
    }

    #[test]
    fn test_jidoka_validate_negative_infinity() {
        let mut v = AnomalyValidator::new();
        assert_eq!(
            v.validate(f64::NEG_INFINITY),
            Err(AnomalyViolation::Infinite)
        );
    }

    #[test]
    fn test_jidoka_validate_overflow_positive() {
        let mut v = AnomalyValidator::with_max_magnitude(100.0);
        let result = v.validate(150.0);
        assert!(matches!(result, Err(AnomalyViolation::Overflow(_))));
    }

    #[test]
    fn test_jidoka_validate_overflow_negative() {
        let mut v = AnomalyValidator::with_max_magnitude(100.0);
        let result = v.validate(-150.0);
        assert!(matches!(result, Err(AnomalyViolation::Overflow(_))));
    }

    #[test]
    fn test_jidoka_validate_at_boundary() {
        let mut v = AnomalyValidator::with_max_magnitude(100.0);
        assert_eq!(v.validate(100.0), Ok(100.0));
        assert_eq!(v.validate(-100.0), Ok(-100.0));
    }

    #[test]
    fn test_jidoka_history_recording() {
        let mut v = AnomalyValidator::new();
        v.validate(1.0).unwrap();
        v.validate(2.0).unwrap();
        v.validate(3.0).unwrap();
        assert_eq!(v.history_len(), 3);
        assert_eq!(
            v.history().iter().copied().collect::<Vec<_>>(),
            vec![1.0, 2.0, 3.0]
        );
    }

    #[test]
    fn test_jidoka_history_clear() {
        let mut v = AnomalyValidator::new();
        v.validate(1.0).unwrap();
        v.validate(2.0).unwrap();
        v.clear_history();
        assert_eq!(v.history_len(), 0);
    }

    #[test]
    fn test_jidoka_history_max_size() {
        let mut v = AnomalyValidator::new();
        v.max_history = 3;
        for i in 0..5 {
            v.validate(i as f64).unwrap();
        }
        assert_eq!(v.history_len(), 3);
        assert_eq!(
            v.history().iter().copied().collect::<Vec<_>>(),
            vec![2.0, 3.0, 4.0]
        );
    }

    #[test]
    fn test_jidoka_special_values_disabled() {
        let mut v = AnomalyValidator::new();
        v.check_special_values = false;
        // NaN still recorded but not rejected
        assert!(v.validate(f64::NAN).is_ok());
    }
}
