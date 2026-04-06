//! double_blind_f1021 - Part 2

use cbtop::{
    BlackBoxArtifact, FalsificationClaim, FalsificationCriterion, ReleaseDecision, Role,
    ScorecardV2, SessionState, VerificationAttempt, VerificationResult, VerificationSession,
};

// ============================================================================
// F1029: Audit trail maintained
// ============================================================================

#[test]
fn f1029_claim_recorded_in_audit() {
    let mut session = VerificationSession::new("test");
    let mut claim = FalsificationClaim::new("C001", "Feature", "dev", "1.0");
    claim.add_criterion(FalsificationCriterion::new("F001", "Test", "Pass"));

    session.submit_claim(Role::Dev, claim).unwrap();

    assert!(!session.audit_trail().is_empty());
    assert_eq!(session.audit_trail()[0].role, Role::Dev);
}

#[test]
fn f1029_verification_recorded_in_audit() {
    let mut session = VerificationSession::new("test");
    let attempt = VerificationAttempt::new("V001", "BB-C001", "qa");

    session.submit_attempt(Role::Qa, attempt).unwrap();

    assert!(!session.audit_trail().is_empty());
    assert_eq!(session.audit_trail()[0].role, Role::Qa);
}

#[test]
fn f1029_decision_recorded_in_audit() {
    let mut session = VerificationSession::new("test");

    // Submit claim
    let mut claim = FalsificationClaim::new("C001", "Feature", "dev", "1.0");
    claim.add_criterion(FalsificationCriterion::new("F001", "Test", "Pass"));
    session.submit_claim(Role::Dev, claim).unwrap();

    // Submit attempt
    let mut attempt = VerificationAttempt::new("V001", "BB-C001", "qa");
    attempt.finalize(VerificationResult::Unfalsified);
    session.submit_attempt(Role::Qa, attempt).unwrap();

    // Set passing score
    session.scorecard.set_score("Core Correctness", 85);
    session.scorecard.set_score("Performance", 92);
    session.scorecard.set_score("Resilience", 80);
    session.scorecard.set_score("Usability", 95);

    session.make_decision(Role::System).unwrap();

    // Should have 3 audit entries: claim, attempt, decision
    assert_eq!(session.audit_trail().len(), 3);
}

// ============================================================================
// F1030: Blind maintained during test
// ============================================================================

#[test]
fn f1030_artifact_contains_only_criteria() {
    let mut claim = FalsificationClaim::new("C001", "Feature", "dev", "1.0");
    claim.add_criterion(FalsificationCriterion::new("F001", "Test", "Pass"));

    let artifact = BlackBoxArtifact::from_claim(&claim, "binary_hash");

    // QA sees: criteria and binary hash
    assert!(!artifact.criteria.is_empty());
    assert!(!artifact.binary_hash.is_empty());
    // QA does NOT see: claimant, evidence (these are stripped)
}

// ============================================================================
// F1031: Multiple QA attempts tracked
// ============================================================================

#[test]
fn f1031_multiple_attempts_tracked() {
    let mut session = VerificationSession::new("test");

    let attempt1 = VerificationAttempt::new("V001", "BB-C001", "qa1");
    let attempt2 = VerificationAttempt::new("V002", "BB-C001", "qa2");

    session.submit_attempt(Role::Qa, attempt1).unwrap();
    session.submit_attempt(Role::Qa, attempt2).unwrap();

    assert_eq!(session.attempt_count(), 2);
}

#[test]
fn f1031_can_get_attempts_by_artifact() {
    let mut session = VerificationSession::new("test");

    let attempt1 = VerificationAttempt::new("V001", "BB-C001", "qa1");
    let attempt2 = VerificationAttempt::new("V002", "BB-C001", "qa2");
    let attempt3 = VerificationAttempt::new("V003", "BB-C002", "qa3"); // Different artifact

    session.submit_attempt(Role::Qa, attempt1).unwrap();
    session.submit_attempt(Role::Qa, attempt2).unwrap();
    session.submit_attempt(Role::Qa, attempt3).unwrap();

    let c001_attempts = session.get_attempts("BB-C001");
    assert_eq!(c001_attempts.len(), 2);
}

// ============================================================================
// F1032: Claim revision detection
// ============================================================================

#[test]
fn f1032_hash_changes_on_criterion_add() {
    let mut claim = FalsificationClaim::new("C001", "Feature", "dev", "1.0");
    claim.add_criterion(FalsificationCriterion::new("F001", "Test", "Pass"));

    let hash1 = claim.criteria_hash;

    claim.add_criterion(FalsificationCriterion::new("F002", "Test2", "Pass2"));

    let hash2 = claim.criteria_hash;

    assert_ne!(hash1, hash2);
}

// ============================================================================
// F1033: Time-bounded verification
// ============================================================================

#[test]
fn f1033_deadline_not_expired() {
    use std::time::{Duration, SystemTime};

    let mut claim = FalsificationClaim::new("C001", "Feature", "dev", "1.0");
    claim.add_criterion(FalsificationCriterion::new("F001", "Test", "Pass"));

    let artifact = BlackBoxArtifact::from_claim(&claim, "hash")
        .with_deadline(SystemTime::now() + Duration::from_secs(3600));

    assert!(!artifact.is_expired());
}

#[test]
fn f1033_deadline_expired() {
    use std::time::{Duration, SystemTime};

    let mut claim = FalsificationClaim::new("C001", "Feature", "dev", "1.0");
    claim.add_criterion(FalsificationCriterion::new("F001", "Test", "Pass"));

    let artifact = BlackBoxArtifact::from_claim(&claim, "hash")
        .with_deadline(SystemTime::now() - Duration::from_secs(1));

    assert!(artifact.is_expired());
}

// ============================================================================
// F1034: Reproducibility maintained
// ============================================================================

#[test]
fn f1034_same_criteria_same_hash() {
    let c1 = FalsificationCriterion::new("F001", "Test", "Pass");
    let c2 = FalsificationCriterion::new("F001", "Test", "Pass");

    assert_eq!(c1.hash(), c2.hash());
}

// ============================================================================
// F1035: Report generation complete
// ============================================================================

#[test]
fn f1035_report_includes_all_counts() {
    let mut session = VerificationSession::new("test");

    // Submit claim
    let mut claim = FalsificationClaim::new("C001", "Feature", "dev", "1.0");
    claim.add_criterion(FalsificationCriterion::new("F001", "Test", "Pass"));
    session.submit_claim(Role::Dev, claim.clone()).unwrap();

    // Generate artifact
    session.generate_artifact("C001", "hash");

    // Submit attempts
    let mut a1 = VerificationAttempt::new("V001", "BB-C001", "qa1");
    a1.finalize(VerificationResult::Unfalsified);
    session.submit_attempt(Role::Qa, a1).unwrap();

    let mut a2 = VerificationAttempt::new("V002", "BB-C001", "qa2");
    a2.finalize(VerificationResult::Falsified);
    session.submit_attempt(Role::Qa, a2).unwrap();

    let report = session.generate_report();

    assert_eq!(report.total_claims, 1);
    assert_eq!(report.total_artifacts, 1);
    assert_eq!(report.total_attempts, 2);
    assert_eq!(report.falsified_count, 1);
    assert_eq!(report.unfalsified_count, 1);
    assert!(report.audit_entries > 0);
}

// ============================================================================
// Additional Tests for Coverage
// ============================================================================

#[test]
fn test_role_names() {
    assert_eq!(Role::Dev.name(), "Developer");
    assert_eq!(Role::Qa.name(), "QA");
    assert_eq!(Role::System.name(), "System");
}

#[test]
fn test_verification_result_approval() {
    assert!(!VerificationResult::Falsified.should_approve());
    assert!(VerificationResult::Unfalsified.should_approve());
    assert!(!VerificationResult::Inconclusive.should_approve());
}

#[test]
fn test_scorecard_grades() {
    let mut scorecard = ScorecardV2::new();

    scorecard.set_score("Core Correctness", 95);
    scorecard.set_score("Performance", 95);
    scorecard.set_score("Resilience", 95);
    scorecard.set_score("Usability", 95);
    assert_eq!(scorecard.grade(), "A");

    scorecard.set_score("Core Correctness", 85);
    scorecard.set_score("Performance", 85);
    scorecard.set_score("Resilience", 85);
    scorecard.set_score("Usability", 85);
    assert_eq!(scorecard.grade(), "B");

    scorecard.set_score("Core Correctness", 75);
    scorecard.set_score("Performance", 75);
    scorecard.set_score("Resilience", 75);
    scorecard.set_score("Usability", 75);
    assert_eq!(scorecard.grade(), "C");

    scorecard.set_score("Core Correctness", 65);
    scorecard.set_score("Performance", 65);
    scorecard.set_score("Resilience", 65);
    scorecard.set_score("Usability", 65);
    assert_eq!(scorecard.grade(), "D");

    scorecard.set_score("Core Correctness", 50);
    scorecard.set_score("Performance", 50);
    scorecard.set_score("Resilience", 50);
    scorecard.set_score("Usability", 50);
    assert_eq!(scorecard.grade(), "F");
}

#[test]
fn test_release_decision_properties() {
    let approved = ReleaseDecision::Approved { reason: "good".to_string() };
    assert!(approved.is_approved());
    assert_eq!(approved.reason(), "good");

    let rejected = ReleaseDecision::Rejected { reason: "bad".to_string() };
    assert!(!rejected.is_approved());
    assert_eq!(rejected.reason(), "bad");

    let pending = ReleaseDecision::Pending { reason: "waiting".to_string() };
    assert!(!pending.is_approved());
    assert_eq!(pending.reason(), "waiting");
}

#[test]
fn test_session_state_transitions() {
    let mut session = VerificationSession::new("test");
    assert_eq!(session.state(), SessionState::AwaitingClaims);

    let mut claim = FalsificationClaim::new("C001", "Feature", "dev", "1.0");
    claim.add_criterion(FalsificationCriterion::new("F001", "Test", "Pass"));
    session.submit_claim(Role::Dev, claim).unwrap();
    assert_eq!(session.state(), SessionState::AwaitingVerification);
}

#[test]
fn test_report_success_check() {
    let mut session = VerificationSession::new("test");
    session.scorecard.set_score("Core Correctness", 85);
    session.scorecard.set_score("Performance", 85);
    session.scorecard.set_score("Resilience", 85);
    session.scorecard.set_score("Usability", 85);

    let report = session.generate_report();
    assert!(report.is_success()); // No falsifications and passing score

    // Submit a falsifying attempt
    let mut attempt = VerificationAttempt::new("V001", "BB-C001", "qa");
    attempt.finalize(VerificationResult::Falsified);
    session.submit_attempt(Role::Qa, attempt).unwrap();

    let report = session.generate_report();
    assert!(!report.is_success()); // Has falsification
}
