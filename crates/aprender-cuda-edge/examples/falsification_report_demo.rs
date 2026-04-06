//! Falsification Protocol Demo
//!
//! Demonstrates the 50-point falsification checklist and coverage tracking.
//!
//! Run with: cargo run --example falsification_report_demo

use trueno_cuda_edge::falsification::{
    all_claims, claims_for_framework, ClaimStatus, FalsificationReport, Framework,
};

const FRAMEWORKS: [Framework; 6] = [
    Framework::NullFuzzer,
    Framework::ShmemProber,
    Framework::LifecycleChaos,
    Framework::QuantOracle,
    Framework::PtxPoison,
    Framework::Supervisor,
];

fn main() {
    println!("=== Falsification Protocol Demo ===\n");

    demo_protocol_overview();
    demo_sample_claims();
    let report = demo_report_tracking();
    demo_violated_claims(&report);
    demo_framework_summary(&report);

    println!("\n=== Demo Complete ===");
}

fn demo_protocol_overview() {
    println!("1. Protocol Overview");
    println!("   ──────────────────");

    let claims = all_claims();
    println!("   Total claims: {}", claims.len());

    println!("\n   Claims by framework:");
    for fw in &FRAMEWORKS {
        let count = claims_for_framework(*fw).len();
        println!("   │ {:<20} {:>2} claims", format!("{}", fw), count);
    }

    println!();
}

fn demo_sample_claims() {
    println!("2. Sample Claims (first 3 per framework)");
    println!("   ──────────────────────────────────────");

    for fw in &FRAMEWORKS {
        println!("\n   {}:", fw);
        for claim in claims_for_framework(*fw).iter().take(3) {
            println!("   │ {} [P{}]: {}", claim.id, claim.priority, claim.description);
        }
    }

    println!();
}

fn print_report_status(report: &FalsificationReport, label: &str, statuses: &[ClaimStatus]) {
    println!("\n   {}:", label);
    for status in statuses {
        println!("   │ {:?}: {}", status, report.count_by_status(*status));
    }
    println!("   │ Coverage: {:.1}%", report.coverage() * 100.0);
}

fn demo_report_tracking() -> FalsificationReport {
    println!("3. Report Tracking");
    println!("   ─────────────────");

    let mut report = FalsificationReport::new();

    print_report_status(&report, "Initial state", &[ClaimStatus::Pending]);

    for claim in claims_for_framework(Framework::NullFuzzer) {
        report.mark_verified(claim.id);
    }
    print_report_status(
        &report,
        "After verifying NullFuzzer (10 claims)",
        &[ClaimStatus::Verified, ClaimStatus::Pending],
    );

    report.mark_violated("SP-001");
    print_report_status(&report, "After finding bug in SP-001", &[ClaimStatus::Violated]);

    for claim in ["LC-005", "LC-006", "LC-007", "LC-008"] {
        report.mark_skipped(claim);
    }
    print_report_status(
        &report,
        "After skipping hardware-dependent claims",
        &[ClaimStatus::Skipped],
    );

    println!();

    // 4. Completion Check
    println!("4. Completion Check");
    println!("   ──────────────────");

    println!("   Is complete: {}", report.is_complete());
    println!("   Remaining pending: {}", report.count_by_status(ClaimStatus::Pending));

    for claim in all_claims() {
        if report.status(claim.id) == Some(ClaimStatus::Pending) {
            report.mark_verified(claim.id);
        }
    }

    println!("\n   After verifying all remaining:");
    println!("   │ Is complete: {}", report.is_complete());
    println!("   │ Final coverage: {:.1}%", report.coverage() * 100.0);

    println!();

    report
}

fn demo_violated_claims(report: &FalsificationReport) {
    println!("5. Violated Claims Report");
    println!("   ────────────────────────");

    let violated = report.violated_claims();
    if violated.is_empty() {
        println!("   No violations found (all claims verified).");
    } else {
        println!("   Violations found:");
        for id in &violated {
            println!("   │ {} - Bug detected!", id);
        }
    }

    println!();
}

fn demo_framework_summary(report: &FalsificationReport) {
    println!("6. Framework Summary");
    println!("   ───────────────────");

    let grouped = report.by_framework();
    for fw in &FRAMEWORKS {
        if let Some(claims) = grouped.get(fw) {
            let verified = claims.iter().filter(|(_, s)| *s == ClaimStatus::Verified).count();
            let violated = claims.iter().filter(|(_, s)| *s == ClaimStatus::Violated).count();
            let skipped = claims.iter().filter(|(_, s)| *s == ClaimStatus::Skipped).count();

            println!(
                "   │ {:<20} ✓{:>2} verified  ✗{:>2} violated  ○{:>2} skipped",
                format!("{}", fw),
                verified,
                violated,
                skipped
            );
        }
    }
}
