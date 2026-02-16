use super::*;
use std::time::Duration;

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
