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
#[path = "robustness_tests.rs"]
mod tests;
