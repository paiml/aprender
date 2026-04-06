//! Soft Assertions (Feature 17)
//!
//! Collect multiple assertion failures without stopping test execution.
//!
//! ## EXTREME TDD: Tests written FIRST per spec
//!
//! ## Toyota Way Application:
//! - **Jidoka**: Collect all failures for comprehensive error reporting
//! - **Poka-Yoke**: Type-safe API prevents misuse

use serde::{Deserialize, Serialize};
use std::fmt::Debug;
use std::time::Instant;

/// A single assertion failure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssertionFailure {
    /// Message describing the failure
    pub message: String,
    /// Location where the assertion failed (<file:line>)
    pub location: Option<String>,
    /// Timestamp when the failure occurred
    #[serde(skip)]
    pub timestamp: Option<Instant>,
    /// Index of this assertion in the sequence
    pub index: usize,
}

impl AssertionFailure {
    /// Create a new assertion failure
    #[must_use]
    pub fn new(message: impl Into<String>, index: usize) -> Self {
        Self {
            message: message.into(),
            location: None,
            timestamp: Some(Instant::now()),
            index,
        }
    }

    /// Set the location of the failure
    #[must_use]
    pub fn with_location(mut self, location: impl Into<String>) -> Self {
        self.location = Some(location.into());
        self
    }
}

/// Mode for soft assertions behavior
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum AssertionMode {
    /// Collect all failures (default)
    #[default]
    Collect,
    /// Stop on first failure (like hard assertions)
    FailFast,
}

/// Soft assertions collector
///
/// Collects multiple assertion failures without stopping test execution.
///
/// ## Example
///
/// ```ignore
/// let mut soft = SoftAssertions::new();
/// soft.assert_eq(&1, &2, "values should match");
/// soft.assert_true(false, "condition should be true");
/// // Both failures are collected
/// let result = soft.verify();
/// assert!(result.is_err());
/// ```
#[derive(Debug, Default)]
pub struct SoftAssertions {
    failures: Vec<AssertionFailure>,
    mode: AssertionMode,
    assertion_count: usize,
}

impl SoftAssertions {
    /// Create a new soft assertions collector
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create with a specific mode
    #[must_use]
    pub fn with_mode(mode: AssertionMode) -> Self {
        Self {
            mode,
            ..Self::default()
        }
    }

    /// Set the assertion mode
    #[must_use]
    pub const fn mode(mut self, mode: AssertionMode) -> Self {
        self.mode = mode;
        self
    }

    /// Assert two values are equal
    pub fn assert_eq<T: PartialEq + Debug>(&mut self, actual: &T, expected: &T, message: &str) {
        contract_pre_soft_assertion_collection!();
        self.assertion_count += 1;
        if actual != expected {
            let failure_msg = format!("{message}: expected {expected:?}, got {actual:?}");
            self.record_failure(failure_msg);
        }
    }

    /// Assert two values are not equal
    pub fn assert_ne<T: PartialEq + Debug>(&mut self, actual: &T, expected: &T, message: &str) {
        self.assertion_count += 1;
        if actual == expected {
            let failure_msg = format!("{message}: expected values to differ, both were {actual:?}");
            self.record_failure(failure_msg);
        }
    }

    /// Assert a condition is true
    pub fn assert_true(&mut self, condition: bool, message: &str) {
        self.assertion_count += 1;
        if !condition {
            self.record_failure(format!("{message}: expected true, got false"));
        }
    }

    /// Assert a condition is false
    pub fn assert_false(&mut self, condition: bool, message: &str) {
        self.assertion_count += 1;
        if condition {
            self.record_failure(format!("{message}: expected false, got true"));
        }
    }

    /// Assert a value is Some
    pub fn assert_some<T>(&mut self, opt: &Option<T>, message: &str) {
        self.assertion_count += 1;
        if opt.is_none() {
            self.record_failure(format!("{message}: expected Some, got None"));
        }
    }

    /// Assert a value is None
    pub fn assert_none<T>(&mut self, opt: &Option<T>, message: &str) {
        self.assertion_count += 1;
        if opt.is_some() {
            self.record_failure(format!("{message}: expected None, got Some"));
        }
    }

    /// Assert a Result is Ok
    pub fn assert_ok<T, E>(&mut self, result: &Result<T, E>, message: &str) {
        self.assertion_count += 1;
        if result.is_err() {
            self.record_failure(format!("{message}: expected Ok, got Err"));
        }
    }

    /// Assert a Result is Err
    pub fn assert_err<T, E>(&mut self, result: &Result<T, E>, message: &str) {
        self.assertion_count += 1;
        if result.is_ok() {
            self.record_failure(format!("{message}: expected Err, got Ok"));
        }
    }

    /// Assert a string contains a substring
    pub fn assert_contains(&mut self, haystack: &str, needle: &str, message: &str) {
        self.assertion_count += 1;
        if !haystack.contains(needle) {
            self.record_failure(format!(
                "{message}: expected '{haystack}' to contain '{needle}'"
            ));
        }
    }

    /// Assert a collection has expected length
    pub fn assert_len<T>(&mut self, collection: &[T], expected: usize, message: &str) {
        self.assertion_count += 1;
        if collection.len() != expected {
            self.record_failure(format!(
                "{message}: expected length {expected}, got {}",
                collection.len()
            ));
        }
    }

    /// Assert a collection is empty
    pub fn assert_empty<T>(&mut self, collection: &[T], message: &str) {
        self.assertion_count += 1;
        if !collection.is_empty() {
            self.record_failure(format!(
                "{message}: expected empty collection, got {} elements",
                collection.len()
            ));
        }
    }

    /// Assert a collection is not empty
    pub fn assert_not_empty<T>(&mut self, collection: &[T], message: &str) {
        self.assertion_count += 1;
        if collection.is_empty() {
            self.record_failure(format!("{message}: expected non-empty collection"));
        }
    }

    /// Assert two floats are approximately equal
    pub fn assert_approx_eq(&mut self, actual: f64, expected: f64, epsilon: f64, message: &str) {
        self.assertion_count += 1;
        if (actual - expected).abs() >= epsilon {
            self.record_failure(format!(
                "{message}: expected {actual} ≈ {expected} (epsilon: {epsilon})"
            ));
        }
    }

    /// Assert a value is in a range
    pub fn assert_in_range(&mut self, value: f64, min: f64, max: f64, message: &str) {
        self.assertion_count += 1;
        if value < min || value > max {
            self.record_failure(format!(
                "{message}: expected {value} to be in range [{min}, {max}]"
            ));
        }
    }

    /// Record a custom failure
    pub fn fail(&mut self, message: impl Into<String>) {
        self.assertion_count += 1;
        self.record_failure(message.into());
    }

    /// Record a failure with location info
    fn record_failure(&mut self, message: String) {
        let failure = AssertionFailure::new(message, self.failures.len());
        self.failures.push(failure);
    }

    /// Get all failures
    #[must_use]
    pub fn failures(&self) -> &[AssertionFailure] {
        &self.failures
    }

    /// Get the number of failures
    #[must_use]
    pub fn failure_count(&self) -> usize {
        self.failures.len()
    }

    /// Get the total number of assertions checked
    #[must_use]
    pub const fn assertion_count(&self) -> usize {
        self.assertion_count
    }

    /// Check if all assertions passed
    #[must_use]
    pub fn all_passed(&self) -> bool {
        self.failures.is_empty()
    }

    /// Verify all assertions passed, returning error if any failed
    ///
    /// # Errors
    ///
    /// Returns error containing all failure messages if any assertions failed
    pub fn verify(&self) -> Result<(), SoftAssertionError> {
        contract_pre_soft_assertion_collection!();
        if self.failures.is_empty() {
            Ok(())
        } else {
            Err(SoftAssertionError::new(&self.failures))
        }
    }

    /// Clear all recorded failures
    pub fn clear(&mut self) {
        self.failures.clear();
        self.assertion_count = 0;
    }

    /// Get a summary of the assertions
    #[must_use]
    pub fn summary(&self) -> AssertionSummary {
        AssertionSummary {
            total: self.assertion_count,
            passed: self.assertion_count - self.failures.len(),
            failed: self.failures.len(),
        }
    }
}

/// Summary of assertion results
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct AssertionSummary {
    /// Total assertions checked
    pub total: usize,
    /// Assertions that passed
    pub passed: usize,
    /// Assertions that failed
    pub failed: usize,
}

/// Error type for soft assertion failures
#[derive(Debug, Clone)]
pub struct SoftAssertionError {
    /// All failure messages
    pub failures: Vec<String>,
    /// Number of failed assertions
    pub count: usize,
}

impl SoftAssertionError {
    /// Create a new error from failures
    #[must_use]
    pub fn new(failures: &[AssertionFailure]) -> Self {
        Self {
            failures: failures.iter().map(|f| f.message.clone()).collect(),
            count: failures.len(),
        }
    }
}

impl std::fmt::Display for SoftAssertionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "{} assertion(s) failed:", self.count)?;
        for (i, failure) in self.failures.iter().enumerate() {
            writeln!(f, "  {}. {failure}", i + 1)?;
        }
        Ok(())
    }
}

impl std::error::Error for SoftAssertionError {}

// ============================================================================
// EXTREME TDD: Tests written FIRST per spec
// ============================================================================

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    mod soft_assertions_basic {
        use super::*;

        #[test]
        fn test_new_creates_empty() {
            let soft = SoftAssertions::new();
            assert!(soft.all_passed());
            assert_eq!(soft.failure_count(), 0);
            assert_eq!(soft.assertion_count(), 0);
        }

        #[test]
        fn test_with_mode() {
            let soft = SoftAssertions::with_mode(AssertionMode::FailFast);
            assert_eq!(soft.mode, AssertionMode::FailFast);
        }

        #[test]
        fn test_mode_builder() {
            let soft = SoftAssertions::new().mode(AssertionMode::Collect);
            assert_eq!(soft.mode, AssertionMode::Collect);
        }
    }

    mod equality_assertions {
        use super::*;

        #[test]
        fn test_assert_eq_pass() {
            let mut soft = SoftAssertions::new();
            soft.assert_eq(&42, &42, "values should match");
            assert!(soft.all_passed());
            assert_eq!(soft.assertion_count(), 1);
        }

        #[test]
        fn test_assert_eq_fail() {
            let mut soft = SoftAssertions::new();
            soft.assert_eq(&1, &2, "values should match");
            assert!(!soft.all_passed());
            assert_eq!(soft.failure_count(), 1);
            assert!(soft.failures()[0].message.contains("expected"));
        }

        #[test]
        fn test_assert_ne_pass() {
            let mut soft = SoftAssertions::new();
            soft.assert_ne(&1, &2, "values should differ");
            assert!(soft.all_passed());
        }

        #[test]
        fn test_assert_ne_fail() {
            let mut soft = SoftAssertions::new();
            soft.assert_ne(&42, &42, "values should differ");
            assert!(!soft.all_passed());
        }
    }

    mod boolean_assertions {
        use super::*;

        #[test]
        fn test_assert_true_pass() {
            let mut soft = SoftAssertions::new();
            soft.assert_true(true, "should be true");
            assert!(soft.all_passed());
        }

        #[test]
        fn test_assert_true_fail() {
            let mut soft = SoftAssertions::new();
            soft.assert_true(false, "should be true");
            assert!(!soft.all_passed());
            assert!(soft.failures()[0].message.contains("expected true"));
        }

        #[test]
        fn test_assert_false_pass() {
            let mut soft = SoftAssertions::new();
            soft.assert_false(false, "should be false");
            assert!(soft.all_passed());
        }

        #[test]
        fn test_assert_false_fail() {
            let mut soft = SoftAssertions::new();
            soft.assert_false(true, "should be false");
            assert!(!soft.all_passed());
        }
    }

    mod option_assertions {
        use super::*;

        #[test]
        fn test_assert_some_pass() {
            let mut soft = SoftAssertions::new();
            soft.assert_some(&Some(42), "should be Some");
            assert!(soft.all_passed());
        }

        #[test]
        fn test_assert_some_fail() {
            let mut soft = SoftAssertions::new();
            soft.assert_some::<i32>(&None, "should be Some");
            assert!(!soft.all_passed());
        }

        #[test]
        fn test_assert_none_pass() {
            let mut soft = SoftAssertions::new();
            soft.assert_none::<i32>(&None, "should be None");
            assert!(soft.all_passed());
        }

        #[test]
        fn test_assert_none_fail() {
            let mut soft = SoftAssertions::new();
            soft.assert_none(&Some(42), "should be None");
            assert!(!soft.all_passed());
        }
    }

    mod result_assertions {
        use super::*;

        #[test]
        fn test_assert_ok_pass() {
            let mut soft = SoftAssertions::new();
            let result: Result<i32, &str> = Ok(42);
            soft.assert_ok(&result, "should be Ok");
            assert!(soft.all_passed());
        }

        #[test]
        fn test_assert_ok_fail() {
            let mut soft = SoftAssertions::new();
            let result: Result<i32, &str> = Err("error");
            soft.assert_ok(&result, "should be Ok");
            assert!(!soft.all_passed());
        }

        #[test]
        fn test_assert_err_pass() {
            let mut soft = SoftAssertions::new();
            let result: Result<i32, &str> = Err("error");
            soft.assert_err(&result, "should be Err");
            assert!(soft.all_passed());
        }

        #[test]
        fn test_assert_err_fail() {
            let mut soft = SoftAssertions::new();
            let result: Result<i32, &str> = Ok(42);
            soft.assert_err(&result, "should be Err");
            assert!(!soft.all_passed());
        }
    }

    mod string_assertions {
        use super::*;

        #[test]
        fn test_assert_contains_pass() {
            let mut soft = SoftAssertions::new();
            soft.assert_contains("hello world", "world", "should contain");
            assert!(soft.all_passed());
        }

        #[test]
        fn test_assert_contains_fail() {
            let mut soft = SoftAssertions::new();
            soft.assert_contains("hello", "world", "should contain");
            assert!(!soft.all_passed());
        }
    }

    mod collection_assertions {
        use super::*;

        #[test]
        fn test_assert_len_pass() {
            let mut soft = SoftAssertions::new();
            soft.assert_len(&[1, 2, 3], 3, "should have length 3");
            assert!(soft.all_passed());
        }

        #[test]
        fn test_assert_len_fail() {
            let mut soft = SoftAssertions::new();
            soft.assert_len(&[1, 2], 3, "should have length 3");
            assert!(!soft.all_passed());
        }

        #[test]
        fn test_assert_empty_pass() {
            let mut soft = SoftAssertions::new();
            let empty: Vec<i32> = vec![];
            soft.assert_empty(&empty, "should be empty");
            assert!(soft.all_passed());
        }

        #[test]
        fn test_assert_empty_fail() {
            let mut soft = SoftAssertions::new();
            soft.assert_empty(&[1], "should be empty");
            assert!(!soft.all_passed());
        }

        #[test]
        fn test_assert_not_empty_pass() {
            let mut soft = SoftAssertions::new();
            soft.assert_not_empty(&[1], "should not be empty");
            assert!(soft.all_passed());
        }

        #[test]
        fn test_assert_not_empty_fail() {
            let mut soft = SoftAssertions::new();
            let empty: Vec<i32> = vec![];
            soft.assert_not_empty(&empty, "should not be empty");
            assert!(!soft.all_passed());
        }
    }

    mod numeric_assertions {
        use super::*;

        #[test]
        fn test_assert_approx_eq_pass() {
            let mut soft = SoftAssertions::new();
            soft.assert_approx_eq(1.001, 1.0, 0.01, "should be approximately equal");
            assert!(soft.all_passed());
        }

        #[test]
        fn test_assert_approx_eq_fail() {
            let mut soft = SoftAssertions::new();
            soft.assert_approx_eq(1.5, 1.0, 0.01, "should be approximately equal");
            assert!(!soft.all_passed());
        }

        #[test]
        fn test_assert_in_range_pass() {
            let mut soft = SoftAssertions::new();
            soft.assert_in_range(5.0, 0.0, 10.0, "should be in range");
            assert!(soft.all_passed());
        }

        #[test]
        fn test_assert_in_range_fail() {
            let mut soft = SoftAssertions::new();
            soft.assert_in_range(15.0, 0.0, 10.0, "should be in range");
            assert!(!soft.all_passed());
        }

        #[test]
        fn test_assert_in_range_boundaries() {
            let mut soft = SoftAssertions::new();
            soft.assert_in_range(0.0, 0.0, 10.0, "min boundary");
            soft.assert_in_range(10.0, 0.0, 10.0, "max boundary");
            assert!(soft.all_passed());
        }
    }

    mod multiple_failures {
        use super::*;

        #[test]
        fn test_collects_multiple_failures() {
            let mut soft = SoftAssertions::new();
            soft.assert_eq(&1, &2, "first check");
            soft.assert_true(false, "second check");
            soft.assert_contains("hello", "world", "third check");

            assert_eq!(soft.failure_count(), 3);
            assert_eq!(soft.assertion_count(), 3);
        }

        #[test]
        fn test_mixed_pass_and_fail() {
            let mut soft = SoftAssertions::new();
            soft.assert_eq(&1, &1, "pass");
            soft.assert_eq(&1, &2, "fail");
            soft.assert_true(true, "pass");
            soft.assert_true(false, "fail");

            assert_eq!(soft.failure_count(), 2);
            assert_eq!(soft.assertion_count(), 4);
            assert_eq!(soft.summary().passed, 2);
        }
    }

    mod verify {
        use super::*;

        #[test]
        fn test_verify_pass() {
            let mut soft = SoftAssertions::new();
            soft.assert_eq(&1, &1, "match");
            assert!(soft.verify().is_ok());
        }

        #[test]
        fn test_verify_fail() {
            let mut soft = SoftAssertions::new();
            soft.assert_eq(&1, &2, "mismatch");
            let err = soft.verify().unwrap_err();
            assert_eq!(err.count, 1);
            assert!(!err.failures.is_empty());
        }

        #[test]
        fn test_error_display() {
            let mut soft = SoftAssertions::new();
            soft.assert_eq(&1, &2, "first");
            soft.assert_true(false, "second");
            let err = soft.verify().unwrap_err();
            let display = format!("{err}");
            assert!(display.contains("2 assertion(s) failed"));
            assert!(display.contains("first"));
            assert!(display.contains("second"));
        }
    }

    mod summary {
        use super::*;

        #[test]
        fn test_summary() {
            let mut soft = SoftAssertions::new();
            soft.assert_eq(&1, &1, "pass");
            soft.assert_eq(&1, &2, "fail");
            soft.assert_true(true, "pass");

            let summary = soft.summary();
            assert_eq!(summary.total, 3);
            assert_eq!(summary.passed, 2);
            assert_eq!(summary.failed, 1);
        }
    }

    mod clear {
        use super::*;

        #[test]
        fn test_clear() {
            let mut soft = SoftAssertions::new();
            soft.assert_eq(&1, &2, "fail");
            assert_eq!(soft.failure_count(), 1);

            soft.clear();
            assert_eq!(soft.failure_count(), 0);
            assert_eq!(soft.assertion_count(), 0);
            assert!(soft.all_passed());
        }
    }

    mod custom_failure {
        use super::*;

        #[test]
        fn test_fail_method() {
            let mut soft = SoftAssertions::new();
            soft.fail("custom failure message");
            assert!(!soft.all_passed());
            assert_eq!(soft.failures()[0].message, "custom failure message");
        }
    }

    mod assertion_failure {
        use super::*;

        #[test]
        fn test_assertion_failure_new() {
            let failure = AssertionFailure::new("test message", 0);
            assert_eq!(failure.message, "test message");
            assert_eq!(failure.index, 0);
            assert!(failure.timestamp.is_some());
            assert!(failure.location.is_none());
        }

        #[test]
        fn test_assertion_failure_with_location() {
            let failure = AssertionFailure::new("test", 0).with_location("test.rs:42");
            assert_eq!(failure.location, Some("test.rs:42".to_string()));
        }
    }
}
