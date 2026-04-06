//! Assertions for test validation.
//!
//! This module provides assertion helpers for testing including:
//! - Basic assertions (equality, boolean, etc.)
//! - Soft assertions (collect multiple failures)
//! - Retry assertions (poll until success or timeout)
//! - Equation verification (physics, game invariants - EDD compliance)

mod equation;
mod retry;
mod soft;

use std::fmt::Debug;

// Re-export submodules
pub use equation::{
    EnergyVerifier, EquationContext, EquationResult, EquationVerifier, InvariantVerifier,
    KinematicVerifier, MomentumVerifier, Variable,
};
pub use retry::{
    retry_contains, retry_eq, retry_none, retry_some, retry_true, AssertionCheckResult,
    RetryAssertion, RetryConfig, RetryError, RetryResult,
};
pub use soft::{
    AssertionFailure, AssertionMode, AssertionSummary, SoftAssertionError, SoftAssertions,
};

/// Result of an assertion
#[derive(Debug, Clone)]
pub struct AssertionResult {
    /// Whether the assertion passed
    pub passed: bool,
    /// Human-readable message
    pub message: String,
}

impl AssertionResult {
    /// Create a passing assertion result
    #[must_use]
    pub const fn pass() -> Self {
        Self {
            passed: true,
            message: String::new(),
        }
    }

    /// Create a failing assertion result
    #[must_use]
    pub fn fail(message: impl Into<String>) -> Self {
        Self {
            passed: false,
            message: message.into(),
        }
    }
}

/// Assertion helpers for testing
#[derive(Debug)]
pub struct Assertion;

impl Assertion {
    /// Assert two values are equal
    #[must_use]
    pub fn equals<T: PartialEq + Debug>(expected: &T, actual: &T) -> AssertionResult {
        contract_pre_assertion_evaluation!();
        if expected == actual {
            AssertionResult::pass()
        } else {
            AssertionResult::fail(format!("expected {expected:?}, got {actual:?}"))
        }
    }

    /// Assert a string contains a substring
    #[must_use]
    pub fn contains(haystack: &str, needle: &str) -> AssertionResult {
        if haystack.contains(needle) {
            AssertionResult::pass()
        } else {
            AssertionResult::fail(format!("expected '{haystack}' to contain '{needle}'"))
        }
    }

    /// Assert a value is in a range
    #[must_use]
    pub fn in_range(value: f64, min: f64, max: f64) -> AssertionResult {
        contract_pre_assertion_evaluation!();
        if value >= min && value <= max {
            AssertionResult::pass()
        } else {
            AssertionResult::fail(format!("expected {value} to be in range [{min}, {max}]"))
        }
    }

    /// Assert a condition is true
    #[must_use]
    pub fn is_true(condition: bool, message: &str) -> AssertionResult {
        if condition {
            AssertionResult::pass()
        } else {
            AssertionResult::fail(message)
        }
    }

    /// Assert a condition is false
    #[must_use]
    pub fn is_false(condition: bool, message: &str) -> AssertionResult {
        if condition {
            AssertionResult::fail(message)
        } else {
            AssertionResult::pass()
        }
    }

    /// Assert an Option is Some
    #[must_use]
    pub fn is_some<T>(opt: &Option<T>) -> AssertionResult {
        if opt.is_some() {
            AssertionResult::pass()
        } else {
            AssertionResult::fail("expected Some, got None")
        }
    }

    /// Assert an Option is None
    #[must_use]
    pub fn is_none<T>(opt: &Option<T>) -> AssertionResult {
        if opt.is_none() {
            AssertionResult::pass()
        } else {
            AssertionResult::fail("expected None, got Some")
        }
    }

    /// Assert a Result is Ok
    #[must_use]
    pub fn is_ok<T, E>(result: &Result<T, E>) -> AssertionResult {
        if result.is_ok() {
            AssertionResult::pass()
        } else {
            AssertionResult::fail("expected Ok, got Err")
        }
    }

    /// Assert a Result is Err
    #[must_use]
    pub fn is_err<T, E>(result: &Result<T, E>) -> AssertionResult {
        if result.is_err() {
            AssertionResult::pass()
        } else {
            AssertionResult::fail("expected Err, got Ok")
        }
    }

    /// Assert two floats are approximately equal
    #[must_use]
    pub fn approx_eq(a: f64, b: f64, epsilon: f64) -> AssertionResult {
        contract_pre_assertion_evaluation!();
        if (a - b).abs() < epsilon {
            AssertionResult::pass()
        } else {
            AssertionResult::fail(format!("expected {a} ≈ {b} (epsilon: {epsilon})"))
        }
    }

    /// Assert a collection has expected length
    #[must_use]
    pub fn has_length<T>(collection: &[T], expected: usize) -> AssertionResult {
        if collection.len() == expected {
            AssertionResult::pass()
        } else {
            AssertionResult::fail(format!(
                "expected length {expected}, got {}",
                collection.len()
            ))
        }
    }
}
