use super::*;

#[test]
fn test_validator_positive_float() {
    let v = FuzzInputValidator::new();
    assert!(v.validate_float(1.0).is_ok());
    assert!(v.validate_float(100.0).is_ok());
    assert!(v.validate_float(-1.0).is_ok());
}

#[test]
fn test_validator_rejects_nan() {
    let v = FuzzInputValidator::new();
    assert_eq!(v.validate_float(f64::NAN), Err(FuzzValidationError::NaN));
}

#[test]
fn test_validator_rejects_infinity() {
    let v = FuzzInputValidator::new();
    assert_eq!(
        v.validate_float(f64::INFINITY),
        Err(FuzzValidationError::Infinity)
    );
    assert_eq!(
        v.validate_float(f64::NEG_INFINITY),
        Err(FuzzValidationError::Infinity)
    );
}

#[test]
fn test_validator_positive_only() {
    let v = FuzzInputValidator::positive_only();
    assert!(v.validate_float(1.0).is_ok());
    assert!(matches!(
        v.validate_float(-1.0),
        Err(FuzzValidationError::NegativeValue(_))
    ));
    assert_eq!(v.validate_float(0.0), Err(FuzzValidationError::ZeroValue));
}

#[test]
fn test_safe_div() {
    assert_eq!(safe_div(10.0, 2.0), Some(5.0));
    assert_eq!(safe_div(1.0, 0.0), None);
    assert_eq!(safe_div(0.0, 0.0), None);
    assert_eq!(safe_div(f64::NAN, 1.0), None);
}

#[test]
fn test_checked_add() {
    assert_eq!(checked_add_u64(1, 2), Some(3));
    assert_eq!(checked_add_u64(u64::MAX, 1), None);
}

#[test]
fn test_checked_mul() {
    assert_eq!(checked_mul_u64(2, 3), Some(6));
    assert_eq!(checked_mul_u64(u64::MAX, 2), None);
}

#[test]
fn test_bound_value() {
    assert_eq!(bound_value(5.0, 0.0, 10.0), 5.0);
    assert_eq!(bound_value(-5.0, 0.0, 10.0), 0.0);
    assert_eq!(bound_value(15.0, 0.0, 10.0), 10.0);
    assert_eq!(bound_value(f64::NAN, 0.0, 10.0), 5.0);
}

#[test]
fn test_sanitize_float() {
    assert_eq!(sanitize_float(5.0), 5.0);
    assert_eq!(sanitize_float(f64::NAN), 0.0);
    assert_eq!(sanitize_float(f64::INFINITY), f64::MAX);
    assert_eq!(sanitize_float(f64::NEG_INFINITY), f64::MIN);
}

#[test]
fn test_fuzz_result() {
    let mut result = FuzzResult::new("test");
    result.record_success();
    result.record_success();
    result.record_failure("bad input".to_string(), "failed".to_string());

    assert_eq!(result.test_cases, 3);
    assert_eq!(result.failures, 1);
    assert!(!result.passed());
    assert!((result.failure_rate() - 33.33).abs() < 1.0);
}

#[test]
fn test_fuzz_suite() {
    let mut suite = FuzzSuite::new();

    let mut r1 = FuzzResult::new("target1");
    r1.record_success();
    r1.coverage_percent = 80.0;
    suite.add_result(r1);

    let mut r2 = FuzzResult::new("target2");
    r2.record_success();
    r2.coverage_percent = 90.0;
    suite.add_result(r2);

    let summary = suite.summary();
    assert_eq!(summary.total_targets, 2);
    assert_eq!(summary.targets_passed, 2);
    assert!(summary.overall_passed);
    assert!((summary.avg_coverage - 85.0).abs() < 0.1);
}

#[test]
fn test_string_validation() {
    let v = FuzzInputValidator::new();
    assert!(v.validate_string("hello").is_ok());
    assert!(v.validate_string("").is_ok());

    let long_string = "a".repeat(2000);
    assert!(matches!(
        v.validate_string(&long_string),
        Err(FuzzValidationError::StringTooLong(_))
    ));
}

#[test]
fn test_float_edge_cases_fn() {
    let results = test_float_edge_cases(|x| x * 2.0);
    assert!(!results.is_empty());
    // All multiplication by 2 should succeed (no panics)
    assert!(results.iter().all(|(_, r)| r.is_ok()));
}

#[test]
fn test_u64_edge_cases_fn() {
    let results = test_u64_edge_cases(|x| x.saturating_add(1));
    assert!(!results.is_empty());
    // saturating_add should never panic
    assert!(results.iter().all(|(_, r)| r.is_ok()));
}

#[test]
fn test_fuzz_target_config() {
    let config = FuzzTargetConfig::new("test")
        .with_iterations(1000)
        .with_timeout(30)
        .with_seed(42);

    assert_eq!(config.name, "test");
    assert_eq!(config.iterations, 1000);
    assert_eq!(config.timeout_secs, 30);
    assert_eq!(config.seed, Some(42));
}
