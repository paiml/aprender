//! Showcase Calculator - 100% Test Coverage Demo
//!
//! This crate demonstrates achieving 100% test coverage across both
//! TUI and WASM platforms using Probar's unified testing framework.
//!
//! # Probar Testing Principles
//!
//! - **Error prevention**: Type-safe operations prevent invalid states at compile time
//! - **Anomaly**: Automatic anomaly detection during calculations
//! - **Balanced testing**: Balanced test distribution across components
//! - **Visual feedback**: History tracking for visibility into operations
//! - **Kaizen**: Mutation testing for continuous improvement
//!
//! # Example
//!
//! ```rust
//! use showcase_calculator::prelude::*;
//!
//! // Create an evaluator
//! let mut eval = Evaluator::new();
//!
//! // Evaluate expressions
//! let result = eval.evaluate_str("42 * (3 + 7)").unwrap();
//! assert_eq!(result, 420.0);
//!
//! // With Anomaly validation
//! let validator = AnomalyValidator::with_max_magnitude(100.0);
//! let mut safe_eval = Evaluator::with_validator(validator);
//! assert!(safe_eval.evaluate_str("50 * 3").is_err()); // Exceeds max
//! ```

// Allow common test patterns in this showcase crate
#![cfg_attr(
    test,
    allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::float_cmp
    )
)]
#![deny(missing_docs)]
#![deny(missing_debug_implementations)]

pub mod core;
pub mod driver;

#[cfg(feature = "tui")]
pub mod tui;

/// WASM module - always available for testing
/// (Mock DOM allows testing without actual browser bindings)
pub mod wasm;

/// Probar Advanced Testing Module
/// Demonstrates all Probar features: Page Objects, Accessibility,
/// Visual Regression, Device Emulation, Fixtures, Replay, UX Coverage
#[cfg(test)]
mod probar_tests;

/// Prelude for convenient imports
pub mod prelude {
    pub use crate::core::evaluator::Evaluator;
    pub use crate::core::history::{History, HistoryEntry};
    pub use crate::core::parser::{AstNode, Parser, Token, Tokenizer};
    pub use crate::core::{
        AnomalyValidator, AnomalyViolation, CalcError, CalcResult, Calculator, Operation,
    };
    pub use crate::driver::{CalculatorDriver, HistoryItem};

    #[cfg(feature = "tui")]
    pub use crate::driver::TuiDriver;

    pub use crate::wasm::{DomElement, DomEvent, MockDom, WasmCalculator, WasmDriver};
}

#[cfg(test)]
mod tests {
    use super::prelude::*;

    #[test]
    fn test_prelude_imports() {
        // Verify all prelude exports work
        let mut eval = Evaluator::new();
        let result = eval.evaluate_str("2 + 3").unwrap();
        assert_eq!(result, 5.0);
    }

    #[test]
    fn test_calculator_direct() {
        let mut calc = Calculator::new();
        let result = calc.calculate(6.0, 7.0, Operation::Multiply).unwrap();
        assert_eq!(result, 42.0);
    }

    #[test]
    fn test_parser_direct() {
        let ast = Parser::parse_str("1 + 2 * 3").unwrap();
        let mut eval = Evaluator::new();
        assert_eq!(eval.evaluate(&ast).unwrap(), 7.0);
    }

    #[test]
    fn test_history_tracking() {
        let mut history = History::new();
        history.record("10 / 2", 5.0);
        assert_eq!(history.len(), 1);
        assert_eq!(history.last().unwrap().display(), "10 / 2 = 5");
    }

    #[test]
    fn test_jidoka_validation() {
        let validator = AnomalyValidator::with_max_magnitude(50.0);
        let mut eval = Evaluator::with_validator(validator);

        // Within bounds
        assert!(eval.evaluate_str("5 * 5").is_ok());

        // Exceeds bounds
        assert!(matches!(
            eval.evaluate_str("10 * 10"),
            Err(CalcError::AnomalyViolation(_))
        ));
    }

    #[test]
    fn test_error_handling() {
        let mut eval = Evaluator::new();

        // Division by zero
        assert!(matches!(
            eval.evaluate_str("1 / 0"),
            Err(CalcError::DivisionByZero)
        ));

        // Empty expression
        assert!(matches!(
            eval.evaluate_str(""),
            Err(CalcError::EmptyExpression)
        ));

        // Parse error
        assert!(matches!(
            eval.evaluate_str("1 + + 2"),
            Err(CalcError::ParseError(_))
        ));
    }

    #[test]
    fn test_all_operations() {
        let mut eval = Evaluator::new();

        assert_eq!(eval.evaluate_str("10 + 5").unwrap(), 15.0);
        assert_eq!(eval.evaluate_str("10 - 3").unwrap(), 7.0);
        assert_eq!(eval.evaluate_str("6 * 7").unwrap(), 42.0);
        assert_eq!(eval.evaluate_str("20 / 4").unwrap(), 5.0);
        assert_eq!(eval.evaluate_str("17 % 5").unwrap(), 2.0);
        assert_eq!(eval.evaluate_str("2 ^ 10").unwrap(), 1024.0);
    }

    #[test]
    fn test_complex_expressions() {
        let mut eval = Evaluator::new();

        // PEMDAS: 2 + 3 * 4 = 2 + 12 = 14
        assert_eq!(eval.evaluate_str("2 + 3 * 4").unwrap(), 14.0);

        // Parentheses: (2 + 3) * 4 = 5 * 4 = 20
        assert_eq!(eval.evaluate_str("(2 + 3) * 4").unwrap(), 20.0);

        // Power right associative: 2^3^2 = 2^9 = 512
        assert_eq!(eval.evaluate_str("2 ^ 3 ^ 2").unwrap(), 512.0);

        // Complex: 42 * (3 + 7) = 42 * 10 = 420
        assert_eq!(eval.evaluate_str("42 * (3 + 7)").unwrap(), 420.0);

        // Unary minus: -5 + 10 = 5
        assert_eq!(eval.evaluate_str("-5 + 10").unwrap(), 5.0);
    }
}
