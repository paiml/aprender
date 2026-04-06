// ============================================================================
// Falsification Protocol -- Coverage Tracking
// ============================================================================

use trueno_cuda_edge::falsification::{all_claims, ClaimStatus, FalsificationReport, Framework};

/// Verify 50-point protocol completeness.
#[test]
fn protocol_completeness() {
    let claims = all_claims();
    assert_eq!(claims.len(), 50);
}

/// Test claim framework distribution.
#[test]
fn framework_distribution() {
    let claims = all_claims();

    let null_fuzzer = claims.iter().filter(|c| c.framework == Framework::NullFuzzer).count();
    let shmem = claims.iter().filter(|c| c.framework == Framework::ShmemProber).count();
    let lifecycle = claims.iter().filter(|c| c.framework == Framework::LifecycleChaos).count();
    let quant = claims.iter().filter(|c| c.framework == Framework::QuantOracle).count();
    let ptx = claims.iter().filter(|c| c.framework == Framework::PtxPoison).count();
    let supervisor = claims.iter().filter(|c| c.framework == Framework::Supervisor).count();

    assert_eq!(null_fuzzer, 10);
    assert_eq!(shmem, 10);
    assert_eq!(lifecycle, 8);
    assert_eq!(quant, 8);
    assert_eq!(ptx, 8);
    assert_eq!(supervisor, 6);
}

/// Test report status tracking.
#[test]
fn report_status_tracking() {
    let mut report = FalsificationReport::new();

    // All start pending
    assert_eq!(report.status("NF-001"), Some(ClaimStatus::Pending));

    // Mark verified
    report.mark_verified("NF-001");
    assert_eq!(report.status("NF-001"), Some(ClaimStatus::Verified));

    // Mark violated
    report.mark_violated("NF-002");
    assert_eq!(report.status("NF-002"), Some(ClaimStatus::Violated));

    // Coverage increases
    assert!(report.coverage() > 0.0);
}

/// Test framework grouping.
#[test]
fn framework_grouping() {
    let report = FalsificationReport::new();
    let grouped = report.by_framework();

    assert!(grouped.contains_key(&Framework::NullFuzzer));
    assert!(grouped.contains_key(&Framework::ShmemProber));
    assert!(grouped.contains_key(&Framework::LifecycleChaos));
    assert!(grouped.contains_key(&Framework::QuantOracle));
    assert!(grouped.contains_key(&Framework::PtxPoison));
    assert!(grouped.contains_key(&Framework::Supervisor));
}
