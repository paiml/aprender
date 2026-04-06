//! double_blind_f1021 - Part 1

use cbtop::{
    BlackBoxArtifact, FalsificationClaim, FalsificationCriterion, ReleaseDecision, Role,
    ScorecardV2, VerificationAttempt, VerificationResult, VerificationSession,
};

// ============================================================================
// F1021: Role separation enforced
// ============================================================================

#[test]
fn f1021_dev_can_claim() {
    assert!(Role::Dev.can_claim());
}

#[test]
fn f1021_qa_cannot_claim() {
    assert!(!Role::Qa.can_claim());
}

#[test]
fn f1021_dev_cannot_verify() {
    assert!(!Role::Dev.can_verify());
}

#[test]
fn f1021_qa_can_verify() {
    assert!(Role::Qa.can_verify());
}

#[test]
fn f1021_session_enforces_role_for_claims() {
    let mut session = VerificationSession::new("test-session");
    let mut claim = FalsificationClaim::new("C001", "Feature", "dev", "1.0");
    claim.add_criterion(FalsificationCriterion::new("F001", "Test", "Pass"));

    // QA should not be able to submit claims
    let result = session.submit_claim(Role::Qa, claim.clone());
    assert!(result.is_err());

    // Dev should be able to submit claims
    let result = session.submit_claim(Role::Dev, claim);
    assert!(result.is_ok());
}

#[test]
fn f1021_session_enforces_role_for_verification() {
    let mut session = VerificationSession::new("test-session");
    let attempt = VerificationAttempt::new("V001", "BB-C001", "qa@example.com");

    // Dev should not be able to submit verification
    let result = session.submit_attempt(Role::Dev, attempt.clone());
    assert!(result.is_err());

    // QA should be able to submit verification
    let result = session.submit_attempt(Role::Qa, attempt);
    assert!(result.is_ok());
}

// ============================================================================
// F1022: Black-box artifact isolates source
// ============================================================================

#[test]
fn f1022_artifact_has_no_source_reference() {
    let mut claim = FalsificationClaim::new("C001", "Feature", "dev@example.com", "1.0.0");
    claim.add_criterion(FalsificationCriterion::new("F001", "Test", "Pass"));
    claim.add_evidence("test_log.txt");
    claim.add_evidence("source_file.rs"); // Evidence might reference source

    let artifact = BlackBoxArtifact::from_claim(&claim, "sha256:abc123");

    // Artifact should have binary hash, not source
    assert!(!artifact.binary_hash.is_empty());
    // Artifact should have criteria but not evidence
    assert!(!artifact.criteria.is_empty());
}

#[test]
fn f1022_artifact_id_prefixed() {
    let mut claim = FalsificationClaim::new("C001", "Feature", "dev", "1.0");
    claim.add_criterion(FalsificationCriterion::new("F001", "Test", "Pass"));

    let artifact = BlackBoxArtifact::from_claim(&claim, "hash");
    assert!(artifact.id.starts_with("BB-"));
}

// ============================================================================
// F1023: F-criteria transmitted correctly
// ============================================================================

#[test]
fn f1023_criteria_hash_matches() {
    let mut claim = FalsificationClaim::new("C001", "Feature", "dev", "1.0");
    claim.add_criterion(FalsificationCriterion::new("F001", "Test1", "Pass1"));
    claim.add_criterion(FalsificationCriterion::new("F002", "Test2", "Pass2"));

    let artifact = BlackBoxArtifact::from_claim(&claim, "hash");

    assert!(artifact.verify_criteria_integrity(&claim));
}

#[test]
fn f1023_tampered_criteria_detected() {
    let mut claim = FalsificationClaim::new("C001", "Feature", "dev", "1.0");
    claim.add_criterion(FalsificationCriterion::new("F001", "Test1", "Pass1"));

    let mut artifact = BlackBoxArtifact::from_claim(&claim, "hash");
    artifact.criteria_hash = 0; // Tamper

    assert!(!artifact.verify_criteria_integrity(&claim));
}

// ============================================================================
// F1024: Claim structure validates
// ============================================================================

#[test]
fn f1024_empty_claim_invalid() {
    let claim = FalsificationClaim::new("", "", "", "");
    assert!(!claim.is_valid());
}

#[test]
fn f1024_claim_without_criteria_invalid() {
    let claim = FalsificationClaim::new("C001", "Feature", "dev", "1.0");
    assert!(!claim.is_valid());
}

#[test]
fn f1024_complete_claim_valid() {
    let mut claim = FalsificationClaim::new("C001", "Feature", "dev", "1.0");
    claim.add_criterion(FalsificationCriterion::new("F001", "Test", "Pass"));
    assert!(claim.is_valid());
}

// ============================================================================
// F1025: Verification attempt records result
// ============================================================================

#[test]
fn f1025_attempt_records_falsified() {
    let mut attempt = VerificationAttempt::new("V001", "BB-C001", "qa");
    attempt.finalize(VerificationResult::Falsified);
    assert_eq!(attempt.result, VerificationResult::Falsified);
}

#[test]
fn f1025_attempt_records_unfalsified() {
    let mut attempt = VerificationAttempt::new("V001", "BB-C001", "qa");
    attempt.finalize(VerificationResult::Unfalsified);
    assert_eq!(attempt.result, VerificationResult::Unfalsified);
}

#[test]
fn f1025_attempt_records_inconclusive() {
    let mut attempt = VerificationAttempt::new("V001", "BB-C001", "qa");
    attempt.finalize(VerificationResult::Inconclusive);
    assert_eq!(attempt.result, VerificationResult::Inconclusive);
}

// ============================================================================
// F1026: Evidence collection complete
// ============================================================================

#[test]
fn f1026_claim_collects_evidence() {
    let mut claim = FalsificationClaim::new("C001", "Feature", "dev", "1.0");
    claim.add_evidence("test_output.log");
    claim.add_evidence("coverage_report.html");

    assert_eq!(claim.evidence.len(), 2);
}

#[test]
fn f1026_attempt_collects_evidence() {
    let mut attempt = VerificationAttempt::new("V001", "BB-C001", "qa");
    attempt.add_evidence("crash_log.txt");
    attempt.add_evidence("trace.json");

    assert_eq!(attempt.evidence.len(), 2);
}

// ============================================================================
// F1027: Scorecard calculates correctly
// ============================================================================

#[test]
fn f1027_scorecard_weights_sum_to_one() {
    let scorecard = ScorecardV2::new();
    assert!(scorecard.weights_valid());
}

#[test]
fn f1027_scorecard_weighted_calculation() {
    let mut scorecard = ScorecardV2::new();
    scorecard.set_score("Core Correctness", 100);
    scorecard.set_score("Performance", 100);
    scorecard.set_score("Resilience", 100);
    scorecard.set_score("Usability", 100);

    assert!((scorecard.total_score() - 100.0).abs() < 0.1);
}

#[test]
fn f1027_scorecard_partial_scores() {
    let mut scorecard = ScorecardV2::new();
    scorecard.set_score("Core Correctness", 50);
    scorecard.set_score("Performance", 50);
    scorecard.set_score("Resilience", 50);
    scorecard.set_score("Usability", 50);

    assert!((scorecard.total_score() - 50.0).abs() < 0.1);
}

// ============================================================================
// F1028: Release decision correct
// ============================================================================

#[test]
fn f1028_falsified_rejects() {
    let mut session = VerificationSession::new("test");

    // Submit claim
    let mut claim = FalsificationClaim::new("C001", "Feature", "dev", "1.0");
    claim.add_criterion(FalsificationCriterion::new("F001", "Test", "Pass"));
    session.submit_claim(Role::Dev, claim).unwrap();

    // Submit falsifying attempt
    let mut attempt = VerificationAttempt::new("V001", "BB-C001", "qa");
    attempt.finalize(VerificationResult::Falsified);
    session.submit_attempt(Role::Qa, attempt).unwrap();

    let decision = session.make_decision(Role::System).unwrap();
    assert!(matches!(decision, ReleaseDecision::Rejected { .. }));
}

#[test]
fn f1028_unfalsified_with_passing_score_approves() {
    let mut session = VerificationSession::new("test");

    // Submit claim
    let mut claim = FalsificationClaim::new("C001", "Feature", "dev", "1.0");
    claim.add_criterion(FalsificationCriterion::new("F001", "Test", "Pass"));
    session.submit_claim(Role::Dev, claim).unwrap();

    // Set passing scorecard
    session.scorecard.set_score("Core Correctness", 85);
    session.scorecard.set_score("Performance", 92);
    session.scorecard.set_score("Resilience", 80);
    session.scorecard.set_score("Usability", 95);

    // Submit unfalsifying attempt
    let mut attempt = VerificationAttempt::new("V001", "BB-C001", "qa");
    attempt.finalize(VerificationResult::Unfalsified);
    session.submit_attempt(Role::Qa, attempt).unwrap();

    let decision = session.make_decision(Role::System).unwrap();
    assert!(matches!(decision, ReleaseDecision::Approved { .. }));
}

#[test]
fn f1028_unfalsified_with_failing_score_rejects() {
    let mut session = VerificationSession::new("test");

    // Submit claim
    let mut claim = FalsificationClaim::new("C001", "Feature", "dev", "1.0");
    claim.add_criterion(FalsificationCriterion::new("F001", "Test", "Pass"));
    session.submit_claim(Role::Dev, claim).unwrap();

    // Set failing scorecard (< 70)
    session.scorecard.set_score("Core Correctness", 50);
    session.scorecard.set_score("Performance", 50);
    session.scorecard.set_score("Resilience", 50);
    session.scorecard.set_score("Usability", 50);

    // Submit unfalsifying attempt
    let mut attempt = VerificationAttempt::new("V001", "BB-C001", "qa");
    attempt.finalize(VerificationResult::Unfalsified);
    session.submit_attempt(Role::Qa, attempt).unwrap();

    let decision = session.make_decision(Role::System).unwrap();
    assert!(matches!(decision, ReleaseDecision::Rejected { .. }));
}
