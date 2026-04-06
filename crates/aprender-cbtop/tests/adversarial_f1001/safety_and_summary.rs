//! F1014-F1020: Safety, resource, and summary tests

use cbtop::{
    AdversarialError, AdversarialTactic, AdversarialTestSummary, CancellationToken,
    ConfigValidator, InputValidator, RecoveryHandler, ResourceLimiter,
};
use std::time::Duration;

// ============================================================================
// F1014: NaN propagation controlled
// ============================================================================

#[test]
fn f1014_nan_detected_in_floats() {
    let validator = InputValidator::new();
    let data = vec![1.0, 2.0, f32::NAN, 4.0];
    let result = validator.validate_floats(&data);
    assert!(matches!(
        result,
        Err(AdversarialError::NaNDetected { index: 2 })
    ));
}

#[test]
fn f1014_config_nan_rejected() {
    let validator = ConfigValidator::new();
    let result = validator.validate_numeric("value", f64::NAN);
    assert!(matches!(
        result,
        Err(AdversarialError::ConfigParseError { .. })
    ));
}

// ============================================================================
// F1015: Inf propagation controlled
// ============================================================================

#[test]
fn f1015_positive_inf_detected() {
    let validator = InputValidator::new();
    let data = vec![1.0, f32::INFINITY, 3.0];
    let result = validator.validate_floats(&data);
    assert!(matches!(
        result,
        Err(AdversarialError::InfinityDetected {
            index: 1,
            positive: true
        })
    ));
}

#[test]
fn f1015_negative_inf_detected() {
    let validator = InputValidator::new();
    let data = vec![1.0, f32::NEG_INFINITY, 3.0];
    let result = validator.validate_floats(&data);
    assert!(matches!(
        result,
        Err(AdversarialError::InfinityDetected {
            index: 1,
            positive: false
        })
    ));
}

// ============================================================================
// F1016: Stack overflow prevented
// ============================================================================

#[test]
fn f1016_stack_depth_limit_enforced() {
    let mut limiter = ResourceLimiter::new().with_max_depth(10);

    // Should succeed 10 times
    for _ in 0..10 {
        assert!(limiter.enter_recursion().is_ok());
    }

    // 11th should fail
    let result = limiter.enter_recursion();
    assert!(matches!(
        result,
        Err(AdversarialError::StackOverflow { .. })
    ));
}

#[test]
fn f1016_recursion_exit_decrements_depth() {
    let mut limiter = ResourceLimiter::new().with_max_depth(3);

    limiter.enter_recursion().unwrap();
    limiter.enter_recursion().unwrap();
    limiter.enter_recursion().unwrap();
    // At max

    limiter.exit_recursion();
    // Should be able to enter again
    assert!(limiter.enter_recursion().is_ok());
}

// ============================================================================
// F1017: Resource exhaustion graceful
// ============================================================================

#[test]
fn f1017_cumulative_memory_tracked() {
    let mut limiter = ResourceLimiter::new().with_max_memory(1000);

    limiter.request_memory(400).unwrap();
    limiter.request_memory(400).unwrap();
    // Third request would exceed
    let result = limiter.request_memory(400);
    assert!(matches!(
        result,
        Err(AdversarialError::ResourceExhausted { .. })
    ));
}

// ============================================================================
// F1018: Timeout enforcement correct
// ============================================================================

#[test]
fn f1018_timeout_not_reached() {
    let mut limiter = ResourceLimiter::new().with_timeout(Duration::from_secs(60));
    limiter.start_operation();

    // Immediately check - should not timeout
    assert!(limiter.check_timeout("test_op").is_ok());
}

#[test]
fn f1018_timeout_enforced() {
    let mut limiter = ResourceLimiter::new().with_timeout(Duration::from_millis(10));
    limiter.start_operation();

    // Wait past timeout
    std::thread::sleep(Duration::from_millis(20));

    let result = limiter.check_timeout("test_op");
    assert!(matches!(result, Err(AdversarialError::Timeout { .. })));
}

// ============================================================================
// F1019: Cancellation safe
// ============================================================================

#[test]
fn f1019_uncancelled_check_succeeds() {
    let token = CancellationToken::new();
    assert!(token.check("operation").is_ok());
}

#[test]
fn f1019_cancelled_check_fails() {
    let token = CancellationToken::new();
    token.cancel();
    let result = token.check("operation");
    assert!(matches!(result, Err(AdversarialError::Cancelled { .. })));
}

// ============================================================================
// F1020: Recovery after failure
// ============================================================================

#[test]
fn f1020_recovery_without_checkpoint_fails() {
    let handler: RecoveryHandler<i32> = RecoveryHandler::new();
    let result = handler.recover();
    assert!(matches!(
        result,
        Err(AdversarialError::RecoveryFailed { .. })
    ));
}

#[test]
fn f1020_recovery_with_checkpoint_succeeds() {
    let mut handler: RecoveryHandler<String> = RecoveryHandler::new();
    handler.checkpoint("saved_state".to_string());

    let recovered = handler.recover().unwrap();
    assert_eq!(recovered, "saved_state");
}

#[test]
fn f1020_has_checkpoint_accurate() {
    let mut handler: RecoveryHandler<u32> = RecoveryHandler::new();
    assert!(!handler.has_checkpoint());

    handler.checkpoint(42);
    assert!(handler.has_checkpoint());
}

// ============================================================================
// Tactic Coverage Tests
// ============================================================================

#[test]
fn tactic_all_tactics_defined() {
    let tactics = AdversarialTactic::all();
    assert_eq!(tactics.len(), 5);
}

#[test]
fn tactic_names_non_empty() {
    for tactic in AdversarialTactic::all() {
        assert!(!tactic.name().is_empty());
        assert!(!tactic.tool().is_empty());
    }
}

// ============================================================================
// Summary Tests
// ============================================================================

#[test]
fn summary_tracks_passes() {
    let mut summary = AdversarialTestSummary::new();
    summary.record_pass(AdversarialTactic::BitFlipInjection);
    summary.record_pass(AdversarialTactic::ConfigFuzzing);

    assert_eq!(summary.total_tests, 2);
    assert_eq!(summary.passed, 2);
    assert_eq!(summary.failed, 0);
    assert!(summary.all_passed());
    assert!((summary.pass_rate() - 100.0).abs() < 1e-10);
}

#[test]
fn summary_tracks_failures() {
    let mut summary = AdversarialTestSummary::new();
    summary.record_pass(AdversarialTactic::BitFlipInjection);
    summary.record_fail(AdversarialTactic::ConfigFuzzing, "config failed");

    assert_eq!(summary.total_tests, 2);
    assert_eq!(summary.passed, 1);
    assert_eq!(summary.failed, 1);
    assert!(!summary.all_passed());
    assert!((summary.pass_rate() - 50.0).abs() < 1e-10);
}

#[test]
fn summary_empty_pass_rate_zero() {
    let summary = AdversarialTestSummary::new();
    assert!((summary.pass_rate() - 0.0).abs() < 1e-10);
}
