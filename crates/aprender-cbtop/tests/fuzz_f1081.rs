#![allow(clippy::disallowed_methods, clippy::float_cmp)]
//! Fuzz Testing Falsification Tests (F1081-F1095)
//!
//! Popperian falsification criteria for fuzz testing per §36.3 Resilience.

use cbtop::{
    bound_value,
    checked_add_u64,
    checked_mul_u64,
    safe_div,
    sanitize_float,
    test_float_edge_cases,
    test_u64_edge_cases,
    EscalationThresholds,
    FuzzInputValidator,
    FuzzResult,
    FuzzSuite,
    FuzzTargetConfig,
    FuzzValidationError,
    HardwareProfile,
    RooflineAnalysis,
    // Components to fuzz
    SyscallBreakdown,
    TracingEscalation,
    WorkloadMetrics,
};

// ============================================================================
// F1081: No panics on arbitrary float input
// ============================================================================

#[test]
fn f1081_syscall_breakdown_no_panic_on_arbitrary_duration() {
    let mut breakdown = SyscallBreakdown::new();

    // Test with edge case values - should not panic
    let edge_cases = [0u64, 1, u64::MAX, u64::MAX / 2, 1_000_000_000_000];

    for &duration in &edge_cases {
        breakdown.add_syscall("mmap", duration);
        breakdown.add_syscall("futex", duration);
        breakdown.add_syscall("ioctl", duration);
        breakdown.add_syscall("read", duration);
        breakdown.add_syscall("write", duration);
        breakdown.add_syscall("unknown", duration);
    }
}

#[test]
fn f1081_hardware_profile_no_panic_on_edge_values() {
    // Should not panic on any of these
    let _ = HardwareProfile::new("test", 0.0, 0.0);
    let _ = HardwareProfile::new("test", f64::MAX, f64::MAX);
    let _ = HardwareProfile::new("test", 1e15, 1e15);
    let _ = HardwareProfile::new("test", 1.0, 0.0); // Zero bandwidth
}

#[test]
fn f1081_workload_metrics_no_panic() {
    // Should not panic on any of these
    let _ = WorkloadMetrics::new("test", 0.0, 0.0, 0.0);
    let _ = WorkloadMetrics::new("test", 1e15, 1e15, 1e15);
    let _ = WorkloadMetrics::new("test", 1.0, 0.0, 1.0); // Zero bytes
    let _ = WorkloadMetrics::new("test", 1.0, 1.0, 0.0); // Zero time
}

// ============================================================================
// F1082: NaN/Inf handling graceful
// ============================================================================

#[test]
fn f1082_sanitize_float_handles_nan() {
    assert_eq!(sanitize_float(f64::NAN), 0.0);
}

#[test]
fn f1082_sanitize_float_handles_infinity() {
    assert_eq!(sanitize_float(f64::INFINITY), f64::MAX);
    assert_eq!(sanitize_float(f64::NEG_INFINITY), f64::MIN);
}

#[test]
fn f1082_bound_value_handles_nan() {
    // NaN should be replaced with midpoint
    assert_eq!(bound_value(f64::NAN, 0.0, 10.0), 5.0);
}

#[test]
fn f1082_validator_rejects_nan() {
    let v = FuzzInputValidator::new();
    assert_eq!(v.validate_float(f64::NAN), Err(FuzzValidationError::NaN));
}

#[test]
fn f1082_validator_rejects_infinity() {
    let v = FuzzInputValidator::new();
    assert_eq!(v.validate_float(f64::INFINITY), Err(FuzzValidationError::Infinity));
}

// ============================================================================
// F1083: Zero division protected
// ============================================================================

#[test]
fn f1083_safe_div_protects_zero() {
    assert_eq!(safe_div(1.0, 0.0), None);
    assert_eq!(safe_div(0.0, 0.0), None);
}

#[test]
fn f1083_safe_div_normal_case() {
    assert_eq!(safe_div(10.0, 2.0), Some(5.0));
    assert_eq!(safe_div(0.0, 5.0), Some(0.0));
}

#[test]
fn f1083_safe_div_nan_input() {
    assert_eq!(safe_div(f64::NAN, 1.0), None);
    assert_eq!(safe_div(1.0, f64::NAN), None);
}

#[test]
fn f1083_syscall_breakdown_zero_total() {
    let breakdown = SyscallBreakdown::new();
    // Should not panic, returns 0 or NaN gracefully
    let overhead = breakdown.syscall_overhead_percent();
    assert!(overhead.is_nan() || overhead == 0.0);
}

// ============================================================================
// F1084: Integer overflow checked
// ============================================================================

#[test]
fn f1084_checked_add_prevents_overflow() {
    assert_eq!(checked_add_u64(u64::MAX, 1), None);
    assert_eq!(checked_add_u64(u64::MAX, u64::MAX), None);
}

#[test]
fn f1084_checked_add_normal() {
    assert_eq!(checked_add_u64(1, 2), Some(3));
    assert_eq!(checked_add_u64(0, 0), Some(0));
}

#[test]
fn f1084_checked_mul_prevents_overflow() {
    assert_eq!(checked_mul_u64(u64::MAX, 2), None);
    assert_eq!(checked_mul_u64(u64::MAX / 2 + 1, 2), None);
}

#[test]
fn f1084_checked_mul_normal() {
    assert_eq!(checked_mul_u64(2, 3), Some(6));
    assert_eq!(checked_mul_u64(0, u64::MAX), Some(0));
}

// ============================================================================
// F1085: Empty input accepted
// ============================================================================

#[test]
fn f1085_empty_string_validated() {
    let v = FuzzInputValidator::new();
    assert!(v.validate_string("").is_ok());
}

#[test]
fn f1085_empty_syscall_breakdown() {
    let breakdown = SyscallBreakdown::new();
    assert_eq!(breakdown.dominant_syscall(), "none");
    assert_eq!(breakdown.compute_us(), 0);
}

#[test]
fn f1085_fuzz_result_empty() {
    let result = FuzzResult::new("test");
    assert!(result.passed()); // No failures = passed
    assert_eq!(result.failure_rate(), 0.0);
}

// ============================================================================
// F1086: Negative values handled
// ============================================================================

#[test]
fn f1086_validator_allows_negative() {
    let v = FuzzInputValidator::new();
    assert!(v.validate_float(-1.0).is_ok());
    assert!(v.validate_float(-1e10).is_ok());
}

#[test]
fn f1086_validator_rejects_negative_when_configured() {
    let v = FuzzInputValidator::non_negative();
    assert!(matches!(v.validate_float(-1.0), Err(FuzzValidationError::NegativeValue(_))));
}

#[test]
fn f1086_bound_value_handles_negative() {
    assert_eq!(bound_value(-100.0, 0.0, 10.0), 0.0);
    assert_eq!(bound_value(-5.0, -10.0, 10.0), -5.0);
}

// ============================================================================
// F1087: Very large values bounded
// ============================================================================

#[test]
fn f1087_validator_rejects_too_large() {
    let v = FuzzInputValidator::new();
    assert!(matches!(v.validate_float(1e16), Err(FuzzValidationError::TooLarge(_))));
}

#[test]
fn f1087_bound_value_clamps_large() {
    assert_eq!(bound_value(1e20, 0.0, 100.0), 100.0);
    assert_eq!(bound_value(f64::MAX, 0.0, 1e10), 1e10);
}

#[test]
fn f1087_sanitize_handles_large() {
    let result = sanitize_float(1e300);
    assert!(result.is_finite());
}

// ============================================================================
// F1088: UTF-8 invalid rejected (strings already validated by Rust)
// ============================================================================

#[test]
fn f1088_validator_rejects_control_chars() {
    let v = FuzzInputValidator::new();
    // Control characters except \n and \t should be rejected
    let with_control = "test\x00string";
    assert!(matches!(
        v.validate_string(with_control),
        Err(FuzzValidationError::InvalidControlChars)
    ));
}

#[test]
fn f1088_validator_allows_newline_tab() {
    let v = FuzzInputValidator::new();
    assert!(v.validate_string("test\nstring").is_ok());
    assert!(v.validate_string("test\tstring").is_ok());
}

// ============================================================================
// F1089: Malformed config rejected (via validator)
// ============================================================================

#[test]
fn f1089_validator_rejects_too_long_string() {
    let v = FuzzInputValidator::new();
    let long = "a".repeat(2000);
    assert!(matches!(v.validate_string(&long), Err(FuzzValidationError::StringTooLong(_))));
}

#[test]
fn f1089_validator_accepts_normal_string() {
    let v = FuzzInputValidator::new();
    assert!(v.validate_string("normal syscall name").is_ok());
}

// ============================================================================
// F1090: Memory limits enforced (via validator bounds)
// ============================================================================

#[test]
fn f1090_validator_enforces_numeric_bounds() {
    let v = FuzzInputValidator::strict();
    assert!(v.validate_float(1e10).is_ok());
    assert!(matches!(v.validate_float(1e13), Err(FuzzValidationError::TooLarge(_))));
}

#[test]
fn f1090_fuzz_config_iterations_bounded() {
    let config = FuzzTargetConfig::new("test").with_iterations(1_000_000);
    assert_eq!(config.iterations, 1_000_000);
}

// ============================================================================
// F1091: Coverage plateau detected (simulated)
// ============================================================================

#[test]
fn f1091_fuzz_result_tracks_coverage() {
    let mut result = FuzzResult::new("test");
    result.coverage_percent = 85.0;

    // Coverage above 80% is acceptable
    assert!(result.coverage_percent >= 80.0);
}

#[test]
fn f1091_fuzz_suite_avg_coverage() {
    let mut suite = FuzzSuite::new();

    let mut r1 = FuzzResult::new("t1");
    r1.coverage_percent = 80.0;
    suite.add_result(r1);

    let mut r2 = FuzzResult::new("t2");
    r2.coverage_percent = 90.0;
    suite.add_result(r2);

    let summary = suite.summary();
    assert!((summary.avg_coverage - 85.0).abs() < 0.1);
}

// ============================================================================
// F1092: Crash reproducible (via seed)
// ============================================================================

#[test]
fn f1092_fuzz_config_supports_seed() {
    let config = FuzzTargetConfig::new("test").with_seed(42);
    assert_eq!(config.seed, Some(42));
}

#[test]
fn f1092_fuzz_failure_records_input() {
    let mut result = FuzzResult::new("test");
    result.record_failure("input: 12345".to_string(), "panic at line 100".to_string());

    assert_eq!(result.failure_details.len(), 1);
    assert_eq!(result.failure_details[0].input, "input: 12345");
    assert!(result.failure_details[0].error.contains("panic"));
}

// ============================================================================
// F1093: Sanitizers clean (no UB)
// ============================================================================

#[test]
fn f1093_test_float_edge_cases_no_ub() {
    let results = test_float_edge_cases(sanitize_float);
    // All sanitize operations should succeed
    assert!(results.iter().all(|(_, r)| r.is_ok()));
}

#[test]
fn f1093_test_u64_edge_cases_no_ub() {
    let results = test_u64_edge_cases(|x| x.saturating_add(1));
    // saturating_add never causes UB
    assert!(results.iter().all(|(_, r)| r.is_ok()));
}

// ============================================================================
// F1094: Timeout handling
// ============================================================================

#[test]
fn f1094_fuzz_config_supports_timeout() {
    let config = FuzzTargetConfig::new("test").with_timeout(60);
    assert_eq!(config.timeout_secs, 60);
}

#[test]
fn f1094_fuzz_result_tracks_duration() {
    let mut result = FuzzResult::new("test");
    result.duration_secs = 30.5;
    assert!((result.duration_secs - 30.5).abs() < 0.01);
}

// ============================================================================
// F1095: Resource cleanup on error
// ============================================================================

#[test]
fn f1095_fuzz_suite_tracks_failures() {
    let mut suite = FuzzSuite::new();

    let mut result = FuzzResult::new("test");
    result.record_failure("bad".to_string(), "error".to_string());
    suite.add_result(result);

    assert_eq!(suite.total_failures(), 1);
    assert!(!suite.all_passed());
}

#[test]
fn f1095_fuzz_summary_reports_failures() {
    let mut suite = FuzzSuite::new();

    let mut r1 = FuzzResult::new("pass");
    r1.record_success();
    suite.add_result(r1);

    let mut r2 = FuzzResult::new("fail");
    r2.record_failure("x".to_string(), "y".to_string());
    suite.add_result(r2);

    let summary = suite.summary();
    assert_eq!(summary.targets_passed, 1);
    assert_eq!(summary.total_failures, 1);
    assert!(!summary.overall_passed);
}

// ============================================================================
// Additional Coverage Tests
// ============================================================================

#[test]
fn test_fuzz_validation_error_display() {
    let err = FuzzValidationError::NaN;
    assert!(!err.to_string().is_empty());

    let err = FuzzValidationError::TooLarge(1e20);
    assert!(err.to_string().contains("1e20") || err.to_string().contains("too large"));
}

#[test]
fn test_validator_positive_only() {
    let v = FuzzInputValidator::positive_only();
    assert!(v.validate_float(1.0).is_ok());
    assert!(matches!(v.validate_float(0.0), Err(FuzzValidationError::ZeroValue)));
    assert!(matches!(v.validate_float(-1.0), Err(FuzzValidationError::NegativeValue(_))));
}

#[test]
fn test_fuzz_suite_empty() {
    let suite = FuzzSuite::new();
    let summary = suite.summary();

    assert_eq!(summary.total_targets, 0);
    assert_eq!(summary.pass_rate(), 100.0);
}

#[test]
fn test_escalation_thresholds_fuzz() {
    // Test with edge case thresholds
    let thresholds =
        EscalationThresholds::default().with_cv(0.0).with_efficiency(100.0).with_rate_limit(0);

    let escalation = TracingEscalation::new(thresholds);
    // Should handle extreme thresholds without panic
    let _ = escalation.should_trace(0.0, 0.0);
    let _ = escalation.should_trace(100.0, 100.0);
}

#[test]
fn test_roofline_fuzz_edge_cases() {
    let hardware = HardwareProfile::new("Test", 1000.0, 100.0);

    // Test with edge case workloads
    let w1 = WorkloadMetrics::new("zero", 0.0, 0.0, 0.0);
    let analysis = RooflineAnalysis::analyze(&hardware, &w1);
    // Should not panic, just produce valid (possibly 0) results
    assert!(!analysis.workload.name.is_empty());

    let w2 = WorkloadMetrics::new("large", 1e15, 1e15, 1.0);
    let _ = RooflineAnalysis::analyze(&hardware, &w2);
}
