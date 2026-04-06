//! Severity classification, bug report formatting, and comprehensive bug report tests

use super::*;

// ============================================================================
// COMPREHENSIVE BUG REPORT TEST
// ============================================================================

#[test]
fn test_generate_ptx_bug_report() {
    println!("\n╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║                         PTX BUG HUNTING REPORT                                ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════╝\n");

    let mut bugs_found = Vec::new();

    let edge_cases: Vec<(&str, &str, bool, Option<PtxBugClass>)> = vec![
        // (ptx, description, should_have_bug, expected_bug)
        (
            "st.shared.f32 [%rd0], %f0;",
            "Shared mem 64-bit addressing",
            true,  // Expect a bug
            Some(PtxBugClass::SharedMemU64Addressing),
        ),
        (
            "ld.shared.f32 %f0, [%rd5];",
            "Shared mem load 64-bit",
            true,  // Expect a bug
            Some(PtxBugClass::SharedMemU64Addressing),
        ),
        (
            ".visible .entry test() { .shared .b8 s[1024]; st.shared.f32 [%r0], %f0; ld.shared.f32 %f1, [%r1]; ret; }",
            "Missing barrier (strict mode)",
            true,  // Expect a bug
            Some(PtxBugClass::MissingBarrierSync),
        ),
        (
            ".visible .entry test() { .local .b8 l[32]; ret; }",
            "Register spills",
            true,  // Expect a bug
            Some(PtxBugClass::RegisterSpills),
        ),
        (
            ".version 8.0\n.target sm_70",
            "Missing entry point",
            true,  // Expect a bug
            Some(PtxBugClass::MissingEntryPoint),
        ),
        (
            ".visible .entry valid() { .reg .f32 %f<4>; ret; }",
            "Valid kernel",
            false, // Expect no bugs
            None,
        ),
    ];

    for (ptx, desc, should_have_bug, expected) in &edge_cases {
        let result = if desc.contains("strict") {
            PtxBugAnalyzer::strict().analyze(ptx)
        } else {
            PtxBugAnalyzer::new().analyze(ptx)
        };

        // Check if we found the expected bug (or no bug if expected is None)
        let found_expected = match expected {
            Some(bug_class) => result.has_bug(bug_class),
            None => !result.has_bugs(),
        };

        // Test passes if: (should_have_bug AND bug found) OR (!should_have_bug AND no bugs)
        let test_passed = if *should_have_bug {
            found_expected // Expected a bug, found it
        } else {
            !result.has_bugs() // Expected no bugs, found none
        };

        if !test_passed {
            bugs_found.push((desc.to_string(), ptx.to_string(), format!("{:?}", result.bugs)));
        }
    }

    if bugs_found.is_empty() {
        println!("All {} test cases passed!", edge_cases.len());
    } else {
        println!("Found {} issues:\n", bugs_found.len());
        for (i, (desc, input, err)) in bugs_found.iter().enumerate() {
            println!("ISSUE #{}: {}", i + 1, desc);
            println!("  Input: {}", input.replace('\n', "\\n"));
            println!("  Result: {}", err);
            println!();
        }
    }

    // All edge cases should work as expected
    assert!(bugs_found.is_empty(), "Edge case tests should all pass");
}

// ============================================================================
// SEVERITY CLASSIFICATION TESTS
// ============================================================================

#[test]
fn test_bug_severity_correct() {
    // P0 Critical
    assert_eq!(PtxBugClass::MissingBarrierSync.severity(), BugSeverity::Critical);
    assert_eq!(PtxBugClass::SharedMemU64Addressing.severity(), BugSeverity::Critical);
    assert_eq!(PtxBugClass::LoopBranchToEnd.severity(), BugSeverity::Critical);

    // P1 High
    assert_eq!(PtxBugClass::RegisterSpills.severity(), BugSeverity::High);
    assert_eq!(PtxBugClass::NonInPlaceLoopAccumulator.severity(), BugSeverity::High);

    // P2 Medium
    assert_eq!(PtxBugClass::RedundantMoves.severity(), BugSeverity::Medium);
    assert_eq!(PtxBugClass::UnoptimizedMemoryPattern.severity(), BugSeverity::Medium);

    // False Positive
    assert_eq!(PtxBugClass::MissingEntryPoint.severity(), BugSeverity::FalsePositive);
}

#[test]
fn test_count_by_severity() {
    let ptx = r"
.visible .entry test() {
    .local .b8 __local[32];
    .shared .b8 smem[1024];
    st.shared.f32 [%rd0], %f0;
    ret;
}
";
    let result = PtxBugAnalyzer::new().analyze(ptx);

    // Should have: SharedMemU64Addressing (P0) and RegisterSpills (P1)
    assert!(result.count_by_severity(BugSeverity::Critical) >= 1);
    assert!(result.count_by_severity(BugSeverity::High) >= 1);
}

// ============================================================================
// BUG REPORT FORMATTING
// ============================================================================

#[test]
fn test_bug_report_formatting() {
    let ptx = r"
.visible .entry test() {
    .local .b8 __local[32];
    .shared .b8 smem[1024];
    st.shared.f32 [%rd0], %f0;
    ret;
}
";
    let result = PtxBugAnalyzer::new().analyze(ptx);
    let report = result.format_report();

    assert!(report.contains("PTX BUG HUNTING REPORT"));
    assert!(report.contains("P0 CRITICAL BUGS:"));
    assert!(report.contains("P1 HIGH BUGS:"));
    assert!(report.contains("SUMMARY"));
    assert!(report.contains("Kernel: test"));
}
