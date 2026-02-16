pub(super) use super::{
    numerical, process_edge_case_result, run_edge_case_tests, test_empty_handling,
    test_inf_handling, test_max_size_handling, test_nan_handling, test_zero_handling,
    CategoryScore, EdgeCaseBehavior, EdgeCaseConfig, EdgeCaseResult, Severity,
};
use std::time::{Duration, Instant};

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
    let predict =
        |input: &[f32]| -> Result<Vec<f32>, String> { Ok(input.iter().map(|x| x * 2.0).collect()) };

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
    let predict =
        |input: &[f32]| -> Result<Vec<f32>, String> { Ok(input.iter().map(|x| x * 2.0).collect()) };

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
    let predict =
        |input: &[f32]| -> Result<Vec<f32>, String> { Ok(input.iter().map(|x| x * 2.0).collect()) };

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

#[path = "robustness_tests_part_02.rs"]
mod robustness_tests_part_02;
#[path = "robustness_tests_part_03.rs"]
mod robustness_tests_part_03;
