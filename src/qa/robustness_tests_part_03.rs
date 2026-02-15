
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
    let predict =
        |_input: &[f32]| -> Result<Vec<f32>, String> { Err("Zero vector rejected".to_string()) };

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
