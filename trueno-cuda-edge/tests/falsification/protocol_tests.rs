//! Falsification protocol integration tests.
//!
//! These tests verify the falsification framework itself — that all 50 claims
//! are tracked, that the report correctly computes coverage, and that the
//! claim statuses transition correctly.

#![allow(clippy::unwrap_used)]

use trueno_cuda_edge::falsification::{
    all_claims, claims_for_framework, ClaimStatus, FalsificationReport, Framework,
};

/// The falsification checklist must have exactly 50 claims.
#[test]
fn protocol_has_50_claims() {
    let claims = all_claims();
    assert_eq!(claims.len(), 50, "Specification requires exactly 50 falsification claims");
}

/// All claim IDs must be unique.
#[test]
fn claim_ids_unique() {
    let claims = all_claims();
    let mut ids: Vec<_> = claims.iter().map(|c| c.id).collect();
    ids.sort_unstable();
    let original_len = ids.len();
    ids.dedup();
    assert_eq!(ids.len(), original_len, "All claim IDs must be unique");
}

/// Framework claim counts must sum to 50.
#[test]
fn framework_claims_sum_to_50() {
    let null_fuzzer = claims_for_framework(Framework::NullFuzzer).len();
    let shmem_prober = claims_for_framework(Framework::ShmemProber).len();
    let lifecycle = claims_for_framework(Framework::LifecycleChaos).len();
    let quant = claims_for_framework(Framework::QuantOracle).len();
    let ptx = claims_for_framework(Framework::PtxPoison).len();
    let supervisor = claims_for_framework(Framework::Supervisor).len();

    let total = null_fuzzer + shmem_prober + lifecycle + quant + ptx + supervisor;
    assert_eq!(total, 50, "Framework claims must sum to 50");
}

/// New report initializes all claims as Pending.
#[test]
fn new_report_all_pending() {
    let report = FalsificationReport::new();
    assert_eq!(
        report.count_by_status(ClaimStatus::Pending),
        50,
        "New report must have 50 pending claims"
    );
}

/// Coverage starts at 0% for new report.
#[test]
fn initial_coverage_is_zero() {
    let report = FalsificationReport::new();
    assert!((report.coverage() - 0.0).abs() < f64::EPSILON, "Initial coverage must be 0%");
}

/// Coverage reaches 100% when all claims are verified.
#[test]
fn full_coverage_when_all_verified() {
    let mut report = FalsificationReport::new();
    for claim in all_claims() {
        report.mark_verified(claim.id);
    }
    assert!(
        (report.coverage() - 1.0).abs() < f64::EPSILON,
        "Coverage must be 100% when all verified"
    );
    assert!(report.is_complete(), "Report must be complete");
}

/// Violated claims count toward coverage.
#[test]
fn violated_counts_toward_coverage() {
    let mut report = FalsificationReport::new();

    // Mark half as verified, half as violated
    let claims = all_claims();
    for (i, claim) in claims.iter().enumerate() {
        if i % 2 == 0 {
            report.mark_verified(claim.id);
        } else {
            report.mark_violated(claim.id);
        }
    }

    assert!(
        (report.coverage() - 1.0).abs() < f64::EPSILON,
        "Violated + Verified must equal 100% coverage"
    );
}

/// Skipped claims are excluded from coverage denominator.
#[test]
fn skipped_excluded_from_coverage() {
    let mut report = FalsificationReport::new();
    let claims = all_claims();

    // Verify 10, skip 40
    for (i, claim) in claims.iter().enumerate() {
        if i < 10 {
            report.mark_verified(claim.id);
        } else {
            report.mark_skipped(claim.id);
        }
    }

    // 10 verified / (50 - 40 skipped) = 10/10 = 100%
    assert!(
        (report.coverage() - 1.0).abs() < f64::EPSILON,
        "Coverage must be 100% when all non-skipped are verified"
    );
}

/// In-progress claims prevent completion.
#[test]
fn in_progress_prevents_completion() {
    let mut report = FalsificationReport::new();
    let claims = all_claims();

    // Verify all but one, mark one as in-progress
    for (i, claim) in claims.iter().enumerate() {
        if i == 0 {
            report.mark_in_progress(claim.id);
        } else {
            report.mark_verified(claim.id);
        }
    }

    assert!(!report.is_complete(), "In-progress claims must prevent completion");
}

/// Violated claims can be retrieved.
#[test]
fn violated_claims_retrievable() {
    let mut report = FalsificationReport::new();
    report.mark_violated("NF-001");
    report.mark_violated("SP-002");

    let violated = report.violated_claims();
    assert_eq!(violated.len(), 2);
    assert!(violated.contains(&"NF-001".to_string()));
    assert!(violated.contains(&"SP-002".to_string()));
}

/// Framework grouping works correctly.
#[test]
fn by_framework_groups_correctly() {
    let report = FalsificationReport::new();
    let grouped = report.by_framework();

    assert!(grouped.contains_key(&Framework::NullFuzzer));
    assert!(grouped.contains_key(&Framework::ShmemProber));
    assert!(grouped.contains_key(&Framework::LifecycleChaos));
    assert!(grouped.contains_key(&Framework::QuantOracle));
    assert!(grouped.contains_key(&Framework::PtxPoison));
    assert!(grouped.contains_key(&Framework::Supervisor));

    // Verify counts
    assert_eq!(grouped[&Framework::NullFuzzer].len(), 10);
    assert_eq!(grouped[&Framework::ShmemProber].len(), 10);
    assert_eq!(grouped[&Framework::LifecycleChaos].len(), 8);
    assert_eq!(grouped[&Framework::QuantOracle].len(), 8);
    assert_eq!(grouped[&Framework::PtxPoison].len(), 8);
    assert_eq!(grouped[&Framework::Supervisor].len(), 6);
}

/// Status retrieval returns correct values.
#[test]
fn status_retrieval() {
    let mut report = FalsificationReport::new();

    assert_eq!(report.status("NF-001"), Some(ClaimStatus::Pending));

    report.mark_verified("NF-001");
    assert_eq!(report.status("NF-001"), Some(ClaimStatus::Verified));

    report.mark_violated("NF-001");
    assert_eq!(report.status("NF-001"), Some(ClaimStatus::Violated));

    // Non-existent claim
    assert_eq!(report.status("NONEXISTENT"), None);
}

/// Framework display strings are correct.
#[test]
fn framework_display() {
    assert_eq!(Framework::NullFuzzer.to_string(), "null_fuzzer");
    assert_eq!(Framework::ShmemProber.to_string(), "shmem_prober");
    assert_eq!(Framework::LifecycleChaos.to_string(), "lifecycle_chaos");
    assert_eq!(Framework::QuantOracle.to_string(), "quant_oracle");
    assert_eq!(Framework::PtxPoison.to_string(), "ptx_poison");
    assert_eq!(Framework::Supervisor.to_string(), "supervisor");
}

/// All claims have non-empty descriptions.
#[test]
fn all_claims_have_descriptions() {
    for claim in all_claims() {
        assert!(!claim.description.is_empty(), "Claim {} must have a description", claim.id);
    }
}

/// All claims have valid priorities (1-3).
#[test]
fn all_claims_have_valid_priorities() {
    for claim in all_claims() {
        assert!(
            (1..=3).contains(&claim.priority),
            "Claim {} priority {} must be in [1, 3]",
            claim.id,
            claim.priority
        );
    }
}
