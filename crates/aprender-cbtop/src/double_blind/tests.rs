use super::*;

#[test]
fn test_role_permissions() {
    assert!(Role::Dev.can_claim());
    assert!(!Role::Dev.can_verify());
    assert!(!Role::Dev.can_approve());

    assert!(!Role::Qa.can_claim());
    assert!(Role::Qa.can_verify());
    assert!(!Role::Qa.can_approve());

    assert!(!Role::System.can_claim());
    assert!(!Role::System.can_verify());
    assert!(Role::System.can_approve());
}

#[test]
fn test_criterion_hash() {
    let c1 = FalsificationCriterion::new("F001", "Test", "Pass");
    let c2 = FalsificationCriterion::new("F001", "Test", "Pass");
    let c3 = FalsificationCriterion::new("F002", "Test", "Pass");

    assert_eq!(c1.hash(), c2.hash());
    assert_ne!(c1.hash(), c3.hash());
}

#[test]
fn test_claim_validation() {
    let mut claim = FalsificationClaim::new("C001", "Feature X", "dev@example.com", "1.0.0");

    // Invalid without criteria
    assert!(!claim.is_valid());

    // Valid with criteria
    claim.add_criterion(FalsificationCriterion::new("F001", "Test", "Pass"));
    assert!(claim.is_valid());
}

#[test]
fn test_claim_hash_verification() {
    let mut claim = FalsificationClaim::new("C001", "Feature", "dev", "1.0");
    claim.add_criterion(FalsificationCriterion::new("F001", "Test", "Pass"));

    assert!(claim.verify_hash());

    // Tamper with hash
    claim.criteria_hash = 0;
    assert!(!claim.verify_hash());
}

#[test]
fn test_scorecard_calculation() {
    let mut scorecard = ScorecardV2::new();
    scorecard.set_score("Core Correctness", 85);
    scorecard.set_score("Performance", 92);
    scorecard.set_score("Resilience", 80);
    scorecard.set_score("Usability", 95);

    // 0.30*85 + 0.30*92 + 0.20*80 + 0.20*95 = 25.5 + 27.6 + 16 + 19 = 88.1
    assert!((scorecard.total_score() - 88.1).abs() < 0.1);
    assert!(scorecard.passes());
    assert_eq!(scorecard.grade(), "B");
}

#[test]
fn test_scorecard_weights_valid() {
    let scorecard = ScorecardV2::new();
    assert!(scorecard.weights_valid());
}

#[test]
fn test_verification_attempt_tracking() {
    let mut attempt = VerificationAttempt::new("V001", "BB-C001", "qa@example.com");
    attempt.record_criterion("F001", true);
    attempt.record_criterion("F002", false);
    attempt.record_criterion("F003", true);

    assert_eq!(attempt.passed_count(), 2);
    assert_eq!(attempt.failed_count(), 1);
    assert!(attempt.has_falsification());
}
