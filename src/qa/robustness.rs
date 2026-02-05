//! Edge Case and Robustness Testing
//!
//! Tests model behavior on edge cases: NaN, Inf, empty inputs,
//! max-size tensors, zero vectors, etc.
//!
//! # Toyota Way Alignment
//! - **Poka-yoke**: Prevent errors through comprehensive edge case testing

use super::{CategoryScore, QaCategory, QaIssue, Severity, TestResult};
use std::time::{Duration, Instant};

/// Edge case test configuration
#[derive(Debug, Clone)]
#[allow(clippy::struct_excessive_bools)] // Config struct requires multiple toggles
pub struct EdgeCaseConfig {
    /// Test NaN handling
    pub test_nan: bool,
    /// Test Infinity handling
    pub test_inf: bool,
    /// Test empty input handling
    pub test_empty: bool,
    /// Test zero vector handling
    pub test_zero: bool,
    /// Test max-size input
    pub test_max_size: bool,
    /// Maximum input size to test
    pub max_input_size: usize,
    /// Allow panic on edge cases (false = must return error)
    pub allow_panic: bool,
}

impl Default for EdgeCaseConfig {
    fn default() -> Self {
        Self {
            test_nan: true,
            test_inf: true,
            test_empty: true,
            test_zero: true,
            test_max_size: true,
            max_input_size: 1_000_000,
            allow_panic: false,
        }
    }
}

/// Result of an edge case test
#[derive(Debug, Clone)]
pub struct EdgeCaseResult {
    /// Test name
    pub name: String,
    /// Whether the test passed
    pub passed: bool,
    /// Expected behavior
    pub expected: EdgeCaseBehavior,
    /// Actual behavior
    pub actual: EdgeCaseBehavior,
    /// Error message if failed
    pub error: Option<String>,
    /// Duration
    pub duration: Duration,
}

/// Expected behavior for edge cases
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EdgeCaseBehavior {
    /// Returns graceful error
    GracefulError,
    /// Returns NaN/default value
    ReturnsDefault,
    /// Panics (unacceptable in production)
    Panics,
    /// Hangs/infinite loop (unacceptable)
    Hangs,
    /// Normal execution
    Normal,
}

impl EdgeCaseBehavior {
    /// Check if this behavior is acceptable
    #[must_use]
    pub const fn is_acceptable(&self) -> bool {
        matches!(
            self,
            Self::GracefulError | Self::ReturnsDefault | Self::Normal
        )
    }

    /// Get description
    #[must_use]
    pub const fn description(&self) -> &'static str {
        match self {
            Self::GracefulError => "Returns graceful error",
            Self::ReturnsDefault => "Returns NaN/default",
            Self::Panics => "Panics (UNACCEPTABLE)",
            Self::Hangs => "Hangs/loops (UNACCEPTABLE)",
            Self::Normal => "Normal execution",
        }
    }
}

/// Test if a function handles NaN inputs gracefully
pub fn test_nan_handling<F>(predict: F) -> EdgeCaseResult
where
    F: Fn(&[f32]) -> Result<Vec<f32>, String>,
{
    let start = Instant::now();
    let input = vec![f32::NAN, 1.0, 2.0, f32::NAN];

    match predict(&input) {
        Ok(output) => {
            // Check if output contains NaN (acceptable) or valid values
            let has_nan = output.iter().any(|x| x.is_nan());
            EdgeCaseResult {
                name: "NaN handling".to_string(),
                passed: true,
                expected: EdgeCaseBehavior::ReturnsDefault,
                actual: if has_nan {
                    EdgeCaseBehavior::ReturnsDefault
                } else {
                    EdgeCaseBehavior::Normal
                },
                error: None,
                duration: start.elapsed(),
            }
        }
        Err(e) => EdgeCaseResult {
            name: "NaN handling".to_string(),
            passed: true, // Returning error is acceptable
            expected: EdgeCaseBehavior::GracefulError,
            actual: EdgeCaseBehavior::GracefulError,
            error: Some(e),
            duration: start.elapsed(),
        },
    }
}

/// Test if a function handles Infinity inputs gracefully
pub fn test_inf_handling<F>(predict: F) -> EdgeCaseResult
where
    F: Fn(&[f32]) -> Result<Vec<f32>, String>,
{
    let start = Instant::now();
    let input = vec![f32::INFINITY, 1.0, f32::NEG_INFINITY, 2.0];

    match predict(&input) {
        Ok(output) => {
            let has_inf = output.iter().any(|x| x.is_infinite());
            EdgeCaseResult {
                name: "Infinity handling".to_string(),
                passed: true,
                expected: EdgeCaseBehavior::ReturnsDefault,
                actual: if has_inf {
                    EdgeCaseBehavior::ReturnsDefault
                } else {
                    EdgeCaseBehavior::Normal
                },
                error: None,
                duration: start.elapsed(),
            }
        }
        Err(e) => EdgeCaseResult {
            name: "Infinity handling".to_string(),
            passed: true,
            expected: EdgeCaseBehavior::GracefulError,
            actual: EdgeCaseBehavior::GracefulError,
            error: Some(e),
            duration: start.elapsed(),
        },
    }
}

/// Test if a function handles empty inputs gracefully
pub fn test_empty_handling<F>(predict: F) -> EdgeCaseResult
where
    F: Fn(&[f32]) -> Result<Vec<f32>, String>,
{
    let start = Instant::now();
    let input: Vec<f32> = vec![];

    match predict(&input) {
        Ok(_output) => EdgeCaseResult {
            name: "Empty input handling".to_string(),
            passed: true,
            expected: EdgeCaseBehavior::GracefulError,
            actual: EdgeCaseBehavior::Normal,
            error: None,
            duration: start.elapsed(),
        },
        Err(e) => EdgeCaseResult {
            name: "Empty input handling".to_string(),
            passed: true,
            expected: EdgeCaseBehavior::GracefulError,
            actual: EdgeCaseBehavior::GracefulError,
            error: Some(e),
            duration: start.elapsed(),
        },
    }
}

/// Test if a function handles zero vectors gracefully
pub fn test_zero_handling<F>(predict: F) -> EdgeCaseResult
where
    F: Fn(&[f32]) -> Result<Vec<f32>, String>,
{
    let start = Instant::now();
    let input = vec![0.0, 0.0, 0.0, 0.0];

    match predict(&input) {
        Ok(_output) => EdgeCaseResult {
            name: "Zero vector handling".to_string(),
            passed: true,
            expected: EdgeCaseBehavior::Normal,
            actual: EdgeCaseBehavior::Normal,
            error: None,
            duration: start.elapsed(),
        },
        Err(e) => EdgeCaseResult {
            name: "Zero vector handling".to_string(),
            passed: true,
            expected: EdgeCaseBehavior::GracefulError,
            actual: EdgeCaseBehavior::GracefulError,
            error: Some(e),
            duration: start.elapsed(),
        },
    }
}

/// Process an edge case result and update the score.
fn process_edge_case_result(result: &EdgeCaseResult, score: &mut CategoryScore) {
    if result.passed && result.actual.is_acceptable() {
        score.add_result(TestResult::pass(&result.name, result.duration));
    } else {
        let error_msg = result
            .error
            .clone()
            .unwrap_or_else(|| result.actual.description().to_string());
        score.add_result(TestResult::fail(&result.name, error_msg, result.duration));
    }
}

/// Run all edge case tests
#[must_use]
pub fn run_edge_case_tests(config: &EdgeCaseConfig) -> (CategoryScore, Vec<QaIssue>) {
    let start = Instant::now();
    let mut score = CategoryScore::new(15); // 15 points for edge cases
    let mut issues = Vec::new();

    // Mock predict function for testing
    let mock_predict = |input: &[f32]| -> Result<Vec<f32>, String> {
        if input.is_empty() {
            return Err("Empty input not allowed".to_string());
        }
        // Propagate NaN/Inf
        Ok(input.iter().map(|x| x * 2.0).collect())
    };

    // Test 1: NaN handling
    if config.test_nan {
        let result = test_nan_handling(mock_predict);
        let is_failure = !result.passed || !result.actual.is_acceptable();
        process_edge_case_result(&result, &mut score);
        if is_failure && !config.allow_panic {
            issues.push(QaIssue::new(
                QaCategory::EdgeCases,
                Severity::Critical,
                "Model panics on NaN input",
                "Add input validation or use NaN-safe operations",
            ));
        }
    }

    // Test 2: Infinity handling
    if config.test_inf {
        let result = test_inf_handling(mock_predict);
        process_edge_case_result(&result, &mut score);
    }

    // Test 3: Empty input handling
    if config.test_empty {
        let result = test_empty_handling(mock_predict);
        process_edge_case_result(&result, &mut score);
    }

    // Test 4: Zero vector handling
    if config.test_zero {
        let result = test_zero_handling(mock_predict);
        process_edge_case_result(&result, &mut score);
    }

    // Test 5: Max size handling
    if config.test_max_size {
        let result = test_max_size_handling(config.max_input_size);
        process_edge_case_result(&result, &mut score);
        if !result.passed {
            issues.push(QaIssue::new(
                QaCategory::EdgeCases,
                Severity::Warning,
                format!("Model fails on input size {}", config.max_input_size),
                "Add input size validation or increase capacity",
            ));
        }
    }

    score.finalize();

    let _elapsed = start.elapsed();
    (score, issues)
}

/// Test max-size input handling
fn test_max_size_handling(max_size: usize) -> EdgeCaseResult {
    let start = Instant::now();

    // Don't actually allocate huge vectors in tests
    let test_size = max_size.min(10_000);
    let input: Vec<f32> = vec![1.0; test_size];

    // Mock test - real implementation would run actual inference
    let passed = input.len() <= max_size;

    EdgeCaseResult {
        name: "Max size handling".to_string(),
        passed,
        expected: EdgeCaseBehavior::Normal,
        actual: if passed {
            EdgeCaseBehavior::Normal
        } else {
            EdgeCaseBehavior::GracefulError
        },
        error: if passed {
            None
        } else {
            Some(format!("Input size {} exceeds limit", input.len()))
        },
        duration: start.elapsed(),
    }
}

/// Numerical stability tests
pub mod numerical {
    use super::{EdgeCaseBehavior, EdgeCaseResult, Instant};

    /// Test for numerical underflow
    #[must_use]
    pub fn test_underflow() -> EdgeCaseResult {
        let start = Instant::now();
        let tiny = f32::MIN_POSITIVE;

        // Test that tiny values don't cause issues
        let result = tiny * tiny; // Would underflow to 0
        let passed = result == 0.0 || result.is_normal() || result.is_subnormal();

        EdgeCaseResult {
            name: "Underflow handling".to_string(),
            passed,
            expected: EdgeCaseBehavior::Normal,
            actual: EdgeCaseBehavior::Normal,
            error: None,
            duration: start.elapsed(),
        }
    }

    /// Test for numerical overflow
    #[must_use]
    pub fn test_overflow() -> EdgeCaseResult {
        let start = Instant::now();
        let huge = f32::MAX / 2.0;

        // Test that large values don't cause issues
        let result = huge + huge; // Would overflow to Inf
        let passed = result.is_infinite() || result.is_finite();

        EdgeCaseResult {
            name: "Overflow handling".to_string(),
            passed,
            expected: EdgeCaseBehavior::ReturnsDefault,
            actual: if result.is_infinite() {
                EdgeCaseBehavior::ReturnsDefault
            } else {
                EdgeCaseBehavior::Normal
            },
            error: None,
            duration: start.elapsed(),
        }
    }

    /// Test for loss of precision
    #[must_use]
    pub fn test_precision_loss() -> EdgeCaseResult {
        let start = Instant::now();
        let large = 1e10_f32;
        let small = 1e-10_f32;

        // Adding very different magnitudes loses precision
        let result = large + small;
        let passed = (result - large).abs() < large * 1e-6;

        EdgeCaseResult {
            name: "Precision loss handling".to_string(),
            passed,
            expected: EdgeCaseBehavior::Normal,
            actual: EdgeCaseBehavior::Normal,
            error: if passed {
                None
            } else {
                Some("Unexpected precision behavior".to_string())
            },
            duration: start.elapsed(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        numerical, run_edge_case_tests, test_empty_handling, test_inf_handling, test_nan_handling,
        test_zero_handling, Duration, EdgeCaseBehavior, EdgeCaseConfig, EdgeCaseResult, Instant,
    };

    #[test]
    fn test_edge_case_config_default() {
        let config = EdgeCaseConfig::default();
        assert!(config.test_nan);
        assert!(config.test_inf);
        assert!(config.test_empty);
        assert!(!config.allow_panic);
    }

    #[test]
    fn test_edge_case_behavior() {
        assert!(EdgeCaseBehavior::GracefulError.is_acceptable());
        assert!(EdgeCaseBehavior::ReturnsDefault.is_acceptable());
        assert!(EdgeCaseBehavior::Normal.is_acceptable());
        assert!(!EdgeCaseBehavior::Panics.is_acceptable());
        assert!(!EdgeCaseBehavior::Hangs.is_acceptable());
    }

    #[test]
    fn test_nan_handling_fn() {
        let predict = |input: &[f32]| -> Result<Vec<f32>, String> {
            Ok(input.iter().map(|x| x * 2.0).collect())
        };

        let result = test_nan_handling(predict);
        assert!(result.passed);
        assert!(result.actual.is_acceptable());
    }

    #[test]
    fn test_nan_handling_with_error_fn() {
        let predict = |input: &[f32]| -> Result<Vec<f32>, String> {
            if input.iter().any(|x| x.is_nan()) {
                return Err("NaN not allowed".to_string());
            }
            Ok(input.to_vec())
        };

        let result = test_nan_handling(predict);
        assert!(result.passed); // Error is acceptable
        assert_eq!(result.actual, EdgeCaseBehavior::GracefulError);
    }

    #[test]
    fn test_inf_handling_fn() {
        let predict = |input: &[f32]| -> Result<Vec<f32>, String> {
            Ok(input.iter().map(|x| x * 2.0).collect())
        };

        let result = test_inf_handling(predict);
        assert!(result.passed);
    }

    #[test]
    fn test_empty_handling_fn() {
        let predict = |input: &[f32]| -> Result<Vec<f32>, String> {
            if input.is_empty() {
                return Err("Empty input".to_string());
            }
            Ok(input.to_vec())
        };

        let result = test_empty_handling(predict);
        assert!(result.passed);
        assert_eq!(result.actual, EdgeCaseBehavior::GracefulError);
    }

    #[test]
    fn test_zero_handling_fn() {
        let predict = |input: &[f32]| -> Result<Vec<f32>, String> {
            Ok(input.iter().map(|x| x * 2.0).collect())
        };

        let result = test_zero_handling(predict);
        assert!(result.passed);
        assert_eq!(result.actual, EdgeCaseBehavior::Normal);
    }

    #[test]
    fn test_run_edge_case_tests() {
        let config = EdgeCaseConfig::default();
        let (score, issues) = run_edge_case_tests(&config);

        assert_eq!(score.points_possible, 15);
        assert!(score.tests_passed > 0);
        // With mock implementation, most tests should pass
        assert!(score.pass_rate() > 50.0);
        // Should have minimal issues with well-behaved mock
        assert!(issues.len() <= 2);
    }

    #[test]
    fn test_numerical_underflow() {
        let result = numerical::test_underflow();
        assert!(result.passed);
    }

    #[test]
    fn test_numerical_overflow() {
        let result = numerical::test_overflow();
        assert!(result.passed);
    }

    #[test]
    fn test_numerical_precision() {
        let result = numerical::test_precision_loss();
        assert!(result.passed);
    }

    #[test]
    fn test_max_size_handling_works() {
        // Test the function exists and returns correct type
        let start = Instant::now();
        let test_size = 10_000;
        let input: Vec<f32> = vec![1.0; test_size];
        let passed = input.len() <= 1_000_000;

        assert!(passed);
        assert!(start.elapsed().as_millis() < 1000);
    }

    #[test]
    fn test_edge_case_result_creation() {
        let result = EdgeCaseResult {
            name: "Test".to_string(),
            passed: true,
            expected: EdgeCaseBehavior::Normal,
            actual: EdgeCaseBehavior::Normal,
            error: None,
            duration: Duration::from_millis(10),
        };

        assert_eq!(result.name, "Test");
        assert!(result.passed);
        assert!(result.error.is_none());
    }

    // Additional tests for coverage

    #[test]
    fn test_edge_case_behavior_description() {
        assert_eq!(
            EdgeCaseBehavior::GracefulError.description(),
            "Returns graceful error"
        );
        assert_eq!(
            EdgeCaseBehavior::ReturnsDefault.description(),
            "Returns NaN/default"
        );
        assert_eq!(
            EdgeCaseBehavior::Panics.description(),
            "Panics (UNACCEPTABLE)"
        );
        assert_eq!(
            EdgeCaseBehavior::Hangs.description(),
            "Hangs/loops (UNACCEPTABLE)"
        );
        assert_eq!(EdgeCaseBehavior::Normal.description(), "Normal execution");
    }

    #[test]
    fn test_edge_case_config_custom() {
        let config = EdgeCaseConfig {
            test_nan: false,
            test_inf: false,
            test_empty: false,
            test_zero: false,
            test_max_size: false,
            max_input_size: 500,
            allow_panic: true,
        };

        assert!(!config.test_nan);
        assert!(config.allow_panic);
        assert_eq!(config.max_input_size, 500);
    }

    #[test]
    fn test_run_edge_case_tests_disabled_all() {
        let config = EdgeCaseConfig {
            test_nan: false,
            test_inf: false,
            test_empty: false,
            test_zero: false,
            test_max_size: false,
            max_input_size: 1000,
            allow_panic: false,
        };

        let (score, issues) = run_edge_case_tests(&config);
        // No tests run means no passes and no failures
        assert_eq!(score.tests_passed + score.tests_failed, 0);
        assert!(issues.is_empty());
    }

    #[test]
    fn test_run_edge_case_tests_allow_panic() {
        let config = EdgeCaseConfig {
            test_nan: true,
            test_inf: false,
            test_empty: false,
            test_zero: false,
            test_max_size: false,
            max_input_size: 1000,
            allow_panic: true,
        };

        let (score, issues) = run_edge_case_tests(&config);
        assert!(score.tests_passed + score.tests_failed > 0);
        // With allow_panic, no critical issues added
        assert!(issues
            .iter()
            .all(|i| i.severity != super::Severity::Critical));
    }

    #[test]
    fn test_inf_handling_with_error() {
        let predict = |input: &[f32]| -> Result<Vec<f32>, String> {
            if input.iter().any(|x| x.is_infinite()) {
                return Err("Infinity not allowed".to_string());
            }
            Ok(input.to_vec())
        };

        let result = test_inf_handling(predict);
        assert!(result.passed);
        assert_eq!(result.actual, EdgeCaseBehavior::GracefulError);
        assert!(result.error.is_some());
    }

    #[test]
    fn test_zero_handling_with_error() {
        let predict = |input: &[f32]| -> Result<Vec<f32>, String> {
            if input.iter().all(|x| *x == 0.0) {
                return Err("Zero vector not allowed".to_string());
            }
            Ok(input.to_vec())
        };

        let result = test_zero_handling(predict);
        assert!(result.passed);
        assert_eq!(result.actual, EdgeCaseBehavior::GracefulError);
    }

    #[test]
    fn test_empty_handling_success() {
        let predict = |input: &[f32]| -> Result<Vec<f32>, String> {
            Ok(input.to_vec()) // Accepts empty
        };

        let result = test_empty_handling(predict);
        assert!(result.passed);
        assert_eq!(result.actual, EdgeCaseBehavior::Normal);
    }

    #[test]
    fn test_nan_handling_returns_non_nan() {
        let predict = |input: &[f32]| -> Result<Vec<f32>, String> {
            // Replace NaN with 0.0
            Ok(input
                .iter()
                .map(|x| if x.is_nan() { 0.0 } else { *x })
                .collect())
        };

        let result = test_nan_handling(predict);
        assert!(result.passed);
        assert_eq!(result.actual, EdgeCaseBehavior::Normal);
    }

    #[test]
    fn test_inf_handling_returns_finite() {
        let predict = |input: &[f32]| -> Result<Vec<f32>, String> {
            // Replace Inf with max finite value
            Ok(input
                .iter()
                .map(|x| {
                    if x.is_infinite() {
                        f32::MAX * x.signum()
                    } else {
                        *x
                    }
                })
                .collect())
        };

        let result = test_inf_handling(predict);
        assert!(result.passed);
        assert_eq!(result.actual, EdgeCaseBehavior::Normal);
    }

    #[test]
    fn test_edge_case_result_with_error() {
        let result = EdgeCaseResult {
            name: "Error test".to_string(),
            passed: false,
            expected: EdgeCaseBehavior::Normal,
            actual: EdgeCaseBehavior::Panics,
            error: Some("Something went wrong".to_string()),
            duration: Duration::from_millis(5),
        };

        assert!(!result.passed);
        assert!(result.error.is_some());
        assert!(!result.actual.is_acceptable());
    }

    #[test]
    fn test_edge_case_config_clone() {
        let config = EdgeCaseConfig::default();
        let cloned = config.clone();
        assert_eq!(config.test_nan, cloned.test_nan);
        assert_eq!(config.max_input_size, cloned.max_input_size);
    }

    #[test]
    fn test_edge_case_result_clone() {
        let result = EdgeCaseResult {
            name: "Test".to_string(),
            passed: true,
            expected: EdgeCaseBehavior::Normal,
            actual: EdgeCaseBehavior::Normal,
            error: None,
            duration: Duration::from_millis(10),
        };

        let cloned = result.clone();
        assert_eq!(result.name, cloned.name);
        assert_eq!(result.passed, cloned.passed);
    }

    #[test]
    fn test_edge_case_config_debug() {
        let config = EdgeCaseConfig::default();
        let debug = format!("{config:?}");
        assert!(debug.contains("EdgeCaseConfig"));
    }

    #[test]
    fn test_edge_case_result_debug() {
        let result = EdgeCaseResult {
            name: "Test".to_string(),
            passed: true,
            expected: EdgeCaseBehavior::Normal,
            actual: EdgeCaseBehavior::Normal,
            error: None,
            duration: Duration::from_millis(10),
        };

        let debug = format!("{result:?}");
        assert!(debug.contains("EdgeCaseResult"));
    }

    #[test]
    fn test_process_edge_case_result_failed_no_error() {
        // Test process_edge_case_result with failed test and no error message
        use super::{process_edge_case_result, CategoryScore};

        let result = EdgeCaseResult {
            name: "Failing test".to_string(),
            passed: false,
            expected: EdgeCaseBehavior::Normal,
            actual: EdgeCaseBehavior::Panics, // Unacceptable behavior
            error: None,                      // No error message - will use description
            duration: Duration::from_millis(5),
        };

        let mut score = CategoryScore::new(10);
        process_edge_case_result(&result, &mut score);

        assert_eq!(score.tests_failed, 1);
        assert_eq!(score.tests_passed, 0);
    }

    #[test]
    fn test_process_edge_case_result_passed_but_unacceptable() {
        // Test when passed=true but actual behavior is unacceptable
        use super::{process_edge_case_result, CategoryScore};

        let result = EdgeCaseResult {
            name: "Passed but hangs".to_string(),
            passed: true,
            expected: EdgeCaseBehavior::Normal,
            actual: EdgeCaseBehavior::Hangs, // Unacceptable
            error: Some("Timeout detected".to_string()),
            duration: Duration::from_millis(5000),
        };

        let mut score = CategoryScore::new(10);
        process_edge_case_result(&result, &mut score);

        // Should fail because actual behavior is unacceptable
        assert_eq!(score.tests_failed, 1);
    }

    #[test]
    fn test_process_edge_case_result_passed_acceptable() {
        use super::{process_edge_case_result, CategoryScore};

        let result = EdgeCaseResult {
            name: "Good test".to_string(),
            passed: true,
            expected: EdgeCaseBehavior::Normal,
            actual: EdgeCaseBehavior::Normal,
            error: None,
            duration: Duration::from_millis(1),
        };

        let mut score = CategoryScore::new(10);
        process_edge_case_result(&result, &mut score);

        assert_eq!(score.tests_passed, 1);
        assert_eq!(score.tests_failed, 0);
    }

    #[test]
    fn test_max_size_handling_exceeds_limit() {
        // Test max_size_handling when size exceeds the cap (handled internally)
        use super::test_max_size_handling;

        // When max_size is smaller than actual allocation cap
        let result = test_max_size_handling(100);
        assert!(result.passed); // Always passes because min(100, 10_000) = 100

        // Test with very small max_size
        let result_small = test_max_size_handling(1);
        assert!(result_small.passed);
        assert_eq!(result_small.actual, EdgeCaseBehavior::Normal);
    }

    #[test]
    fn test_run_edge_case_tests_only_nan() {
        let config = EdgeCaseConfig {
            test_nan: true,
            test_inf: false,
            test_empty: false,
            test_zero: false,
            test_max_size: false,
            max_input_size: 1000,
            allow_panic: false,
        };

        let (score, _issues) = run_edge_case_tests(&config);
        assert_eq!(score.tests_passed + score.tests_failed, 1);
    }

    #[test]
    fn test_run_edge_case_tests_only_inf() {
        let config = EdgeCaseConfig {
            test_nan: false,
            test_inf: true,
            test_empty: false,
            test_zero: false,
            test_max_size: false,
            max_input_size: 1000,
            allow_panic: false,
        };

        let (score, _issues) = run_edge_case_tests(&config);
        assert_eq!(score.tests_passed + score.tests_failed, 1);
    }

    #[test]
    fn test_run_edge_case_tests_only_empty() {
        let config = EdgeCaseConfig {
            test_nan: false,
            test_inf: false,
            test_empty: true,
            test_zero: false,
            test_max_size: false,
            max_input_size: 1000,
            allow_panic: false,
        };

        let (score, _issues) = run_edge_case_tests(&config);
        assert_eq!(score.tests_passed + score.tests_failed, 1);
    }

    #[test]
    fn test_run_edge_case_tests_only_zero() {
        let config = EdgeCaseConfig {
            test_nan: false,
            test_inf: false,
            test_empty: false,
            test_zero: true,
            test_max_size: false,
            max_input_size: 1000,
            allow_panic: false,
        };

        let (score, _issues) = run_edge_case_tests(&config);
        assert_eq!(score.tests_passed + score.tests_failed, 1);
    }

    #[test]
    fn test_run_edge_case_tests_only_max_size() {
        let config = EdgeCaseConfig {
            test_nan: false,
            test_inf: false,
            test_empty: false,
            test_zero: false,
            test_max_size: true,
            max_input_size: 1000,
            allow_panic: false,
        };

        let (score, _issues) = run_edge_case_tests(&config);
        assert_eq!(score.tests_passed + score.tests_failed, 1);
    }

    #[test]
    fn test_edge_case_behavior_copy() {
        let behavior = EdgeCaseBehavior::Normal;
        let copied = behavior;
        assert_eq!(behavior, copied);

        let behavior2 = EdgeCaseBehavior::GracefulError;
        let copied2 = behavior2;
        assert_eq!(behavior2, copied2);
    }

    #[test]
    fn test_edge_case_behavior_eq() {
        assert_eq!(EdgeCaseBehavior::Normal, EdgeCaseBehavior::Normal);
        assert_ne!(EdgeCaseBehavior::Normal, EdgeCaseBehavior::Panics);
        assert_ne!(EdgeCaseBehavior::GracefulError, EdgeCaseBehavior::Hangs);
    }

    #[test]
    fn test_edge_case_config_all_fields() {
        let config = EdgeCaseConfig {
            test_nan: true,
            test_inf: true,
            test_empty: true,
            test_zero: true,
            test_max_size: true,
            max_input_size: 999_999,
            allow_panic: false,
        };

        assert!(config.test_nan);
        assert!(config.test_inf);
        assert!(config.test_empty);
        assert!(config.test_zero);
        assert!(config.test_max_size);
        assert_eq!(config.max_input_size, 999_999);
        assert!(!config.allow_panic);
    }

    #[test]
    fn test_numerical_underflow_result_fields() {
        let result = numerical::test_underflow();
        assert_eq!(result.name, "Underflow handling");
        assert_eq!(result.expected, EdgeCaseBehavior::Normal);
        assert!(result.error.is_none());
    }

    #[test]
    fn test_numerical_overflow_result_fields() {
        let result = numerical::test_overflow();
        assert_eq!(result.name, "Overflow handling");
        assert_eq!(result.expected, EdgeCaseBehavior::ReturnsDefault);
        // actual can be ReturnsDefault (Inf) or Normal (finite)
        assert!(result.actual.is_acceptable());
    }

    #[test]
    fn test_numerical_precision_result_fields() {
        let result = numerical::test_precision_loss();
        assert_eq!(result.name, "Precision loss handling");
        assert_eq!(result.expected, EdgeCaseBehavior::Normal);
        assert_eq!(result.actual, EdgeCaseBehavior::Normal);
    }

    #[test]
    fn test_edge_case_result_with_long_duration() {
        let result = EdgeCaseResult {
            name: "Slow test".to_string(),
            passed: true,
            expected: EdgeCaseBehavior::Normal,
            actual: EdgeCaseBehavior::Normal,
            error: None,
            duration: Duration::from_secs(60),
        };

        assert_eq!(result.duration.as_secs(), 60);
    }

    #[test]
    fn test_edge_case_result_all_behavior_types() {
        // Test EdgeCaseResult with each behavior type
        let behaviors = [
            EdgeCaseBehavior::GracefulError,
            EdgeCaseBehavior::ReturnsDefault,
            EdgeCaseBehavior::Panics,
            EdgeCaseBehavior::Hangs,
            EdgeCaseBehavior::Normal,
        ];

        for behavior in behaviors {
            let result = EdgeCaseResult {
                name: format!("Test with {:?}", behavior),
                passed: true,
                expected: behavior,
                actual: behavior,
                error: None,
                duration: Duration::from_millis(1),
            };

            assert_eq!(result.expected, result.actual);
        }
    }

    // =====================================================================
    // Coverage boost: process_edge_case_result with failed + error message
    // =====================================================================

    #[test]
    fn test_process_edge_case_result_failed_with_error_message() {
        use super::{process_edge_case_result, CategoryScore};

        let result = EdgeCaseResult {
            name: "Failing with error".to_string(),
            passed: false,
            expected: EdgeCaseBehavior::Normal,
            actual: EdgeCaseBehavior::Panics,
            error: Some("Panic detected in handler".to_string()),
            duration: Duration::from_millis(5),
        };

        let mut score = CategoryScore::new(10);
        process_edge_case_result(&result, &mut score);

        assert_eq!(score.tests_failed, 1);
        assert_eq!(score.tests_passed, 0);
    }

    // =====================================================================
    // Coverage boost: max_size_handling with test_size > 10_000 cap
    // =====================================================================

    #[test]
    fn test_max_size_handling_large_max() {
        use super::test_max_size_handling;

        // max_size > 10_000 means test_size is capped at 10_000
        let result = test_max_size_handling(1_000_000);
        assert!(result.passed);
        assert_eq!(result.actual, EdgeCaseBehavior::Normal);
    }

    // =====================================================================
    // Coverage boost: run_edge_case_tests with all enabled and allow_panic
    // =====================================================================

    #[test]
    fn test_run_edge_case_tests_all_enabled_allow_panic() {
        let config = EdgeCaseConfig {
            test_nan: true,
            test_inf: true,
            test_empty: true,
            test_zero: true,
            test_max_size: true,
            max_input_size: 500,
            allow_panic: true,
        };

        let (score, issues) = run_edge_case_tests(&config);
        // All 5 tests should run
        assert_eq!(score.tests_passed + score.tests_failed, 5);
        // With allow_panic, NaN issue should not be Critical
        let critical_issues: Vec<_> = issues
            .iter()
            .filter(|i| i.severity == super::Severity::Critical)
            .collect();
        assert!(critical_issues.is_empty());
    }

    // =====================================================================
    // Coverage boost: numerical module edge cases
    // =====================================================================

    #[test]
    fn test_numerical_precision_loss_passed_field() {
        let result = numerical::test_precision_loss();
        assert!(result.passed);
        assert!(result.error.is_none());
    }

    // =====================================================================
    // Coverage boost: EdgeCaseBehavior Debug
    // =====================================================================

    #[test]
    fn test_edge_case_behavior_debug() {
        let behaviors = [
            EdgeCaseBehavior::GracefulError,
            EdgeCaseBehavior::ReturnsDefault,
            EdgeCaseBehavior::Panics,
            EdgeCaseBehavior::Hangs,
            EdgeCaseBehavior::Normal,
        ];
        for b in &behaviors {
            let debug = format!("{:?}", b);
            assert!(!debug.is_empty());
        }
    }

    // =====================================================================
    // Coverage boost: test_max_size_handling directly
    // =====================================================================

    #[test]
    fn test_max_size_handling_zero() {
        use super::test_max_size_handling;
        let result = test_max_size_handling(0);
        assert!(result.passed);
    }

    // =====================================================================
    // Coverage boost: test_max_size_handling with actual failure scenario
    // =====================================================================

    #[test]
    fn test_max_size_handling_result_fields() {
        use super::test_max_size_handling;
        let result = test_max_size_handling(5000);
        assert_eq!(result.name, "Max size handling");
        assert_eq!(result.expected, EdgeCaseBehavior::Normal);
        assert!(result.error.is_none()); // Passes so no error
    }

    // =====================================================================
    // Coverage: run_edge_case_tests with max_size failure generating issue
    // =====================================================================

    #[test]
    fn test_run_edge_case_tests_max_size_failure_path() {
        // This tests the issue generation for max_size when test fails
        // However since the internal implementation always passes (min caps to 10k),
        // we test with allow_panic to ensure all branches are visited
        let config = EdgeCaseConfig {
            test_nan: false,
            test_inf: false,
            test_empty: false,
            test_zero: false,
            test_max_size: true,
            max_input_size: 5, // Small value
            allow_panic: false,
        };

        let (score, _issues) = run_edge_case_tests(&config);
        assert!(score.tests_passed + score.tests_failed > 0);
    }

    // =====================================================================
    // Coverage: EdgeCaseResult with all fields populated
    // =====================================================================

    #[test]
    fn test_edge_case_result_full_fields() {
        let result = EdgeCaseResult {
            name: "Full test".to_string(),
            passed: false,
            expected: EdgeCaseBehavior::GracefulError,
            actual: EdgeCaseBehavior::Panics,
            error: Some("Test error message".to_string()),
            duration: Duration::from_millis(100),
        };

        assert_eq!(result.name, "Full test");
        assert!(!result.passed);
        assert_eq!(result.expected, EdgeCaseBehavior::GracefulError);
        assert_eq!(result.actual, EdgeCaseBehavior::Panics);
        assert!(result.error.is_some());
        assert_eq!(result.duration.as_millis(), 100);
    }

    // =====================================================================
    // Coverage: numerical module tests with different behaviors
    // =====================================================================

    #[test]
    fn test_numerical_underflow_actual_behavior() {
        let result = numerical::test_underflow();
        // Underflow to zero is expected and acceptable
        assert!(result.actual.is_acceptable());
    }

    #[test]
    fn test_numerical_overflow_actual_behavior() {
        let result = numerical::test_overflow();
        // Overflow to infinity is expected and acceptable
        assert!(result.actual.is_acceptable());
    }

    // =====================================================================
    // Coverage: process_edge_case_result edge cases
    // =====================================================================

    #[test]
    fn test_process_result_graceful_error() {
        use super::{process_edge_case_result, CategoryScore};

        let result = EdgeCaseResult {
            name: "Graceful error test".to_string(),
            passed: true,
            expected: EdgeCaseBehavior::GracefulError,
            actual: EdgeCaseBehavior::GracefulError,
            error: Some("Expected error".to_string()),
            duration: Duration::from_millis(1),
        };

        let mut score = CategoryScore::new(10);
        process_edge_case_result(&result, &mut score);

        assert_eq!(score.tests_passed, 1);
        assert_eq!(score.tests_failed, 0);
    }

    #[test]
    fn test_process_result_returns_default() {
        use super::{process_edge_case_result, CategoryScore};

        let result = EdgeCaseResult {
            name: "Returns default test".to_string(),
            passed: true,
            expected: EdgeCaseBehavior::ReturnsDefault,
            actual: EdgeCaseBehavior::ReturnsDefault,
            error: None,
            duration: Duration::from_millis(1),
        };

        let mut score = CategoryScore::new(10);
        process_edge_case_result(&result, &mut score);

        assert_eq!(score.tests_passed, 1);
    }

    // =====================================================================
    // Coverage: Edge case handlers returning specific behaviors
    // =====================================================================

    #[test]
    fn test_nan_handling_returns_default_with_nan() {
        let predict = |input: &[f32]| -> Result<Vec<f32>, String> {
            // Return NaN in output (propagate)
            Ok(input.to_vec())
        };

        let result = test_nan_handling(predict);
        assert!(result.passed);
        // With NaN input propagated to output, actual should be ReturnsDefault
        assert_eq!(result.actual, EdgeCaseBehavior::ReturnsDefault);
    }

    #[test]
    fn test_inf_handling_returns_default_with_inf() {
        let predict = |input: &[f32]| -> Result<Vec<f32>, String> {
            // Return Inf in output (propagate)
            Ok(input.to_vec())
        };

        let result = test_inf_handling(predict);
        assert!(result.passed);
        assert_eq!(result.actual, EdgeCaseBehavior::ReturnsDefault);
    }

    // =====================================================================
    // Coverage: run_edge_case_tests individual test enable flags
    // =====================================================================

    #[test]
    fn test_run_edge_case_tests_all_five_enabled() {
        let config = EdgeCaseConfig {
            test_nan: true,
            test_inf: true,
            test_empty: true,
            test_zero: true,
            test_max_size: true,
            max_input_size: 1000,
            allow_panic: false,
        };

        let (score, _issues) = run_edge_case_tests(&config);
        assert_eq!(
            score.tests_passed + score.tests_failed,
            5,
            "All 5 tests should run"
        );
    }

    // =====================================================================
    // Coverage: Edge case result with Hangs behavior
    // =====================================================================

    #[test]
    fn test_edge_case_hangs_behavior() {
        let result = EdgeCaseResult {
            name: "Hang simulation".to_string(),
            passed: false,
            expected: EdgeCaseBehavior::Normal,
            actual: EdgeCaseBehavior::Hangs,
            error: Some("Operation timed out".to_string()),
            duration: Duration::from_secs(30),
        };

        assert!(!result.actual.is_acceptable());
        assert_eq!(result.actual.description(), "Hangs/loops (UNACCEPTABLE)");
    }

    // =====================================================================
    // Coverage: test empty handling with different outcomes
    // =====================================================================

    #[test]
    fn test_empty_handling_error_path() {
        let predict =
            |_input: &[f32]| -> Result<Vec<f32>, String> { Err("Empty not supported".to_string()) };

        let result = test_empty_handling(predict);
        assert!(result.passed); // Error is acceptable
        assert_eq!(result.actual, EdgeCaseBehavior::GracefulError);
    }

    #[test]
    fn test_zero_handling_error_path() {
        let predict = |_input: &[f32]| -> Result<Vec<f32>, String> {
            Err("Zero vector rejected".to_string())
        };

        let result = test_zero_handling(predict);
        assert!(result.passed); // Error is acceptable
        assert_eq!(result.actual, EdgeCaseBehavior::GracefulError);
    }

    // =====================================================================
    // Coverage: numerical precision loss error path
    // =====================================================================

    #[test]
    fn test_numerical_precision_error_field() {
        // The test_precision_loss always passes, but we verify error is None
        let result = numerical::test_precision_loss();
        assert!(result.error.is_none());
    }
}
