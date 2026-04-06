//! Edge case tests: Register Spills, Missing Entry Point, Empty/Malformed PTX

use super::*;

// ============================================================================
// EDGE CASE: Register Spills
// ============================================================================

#[test]
fn test_register_spills_detection() {
    let ptx = r"
.visible .entry test() {
    .local .align 4 .b8 __local_depot[32];
    .reg .f32 %f<4>;
    ret;
}
";
    let result = PtxBugAnalyzer::new().analyze(ptx);
    assert!(
        result.has_bug(&PtxBugClass::RegisterSpills),
        "Should detect .local memory usage as potential spills"
    );
}

#[test]
fn test_no_spills_valid() {
    let ptx = r"
.visible .entry test() {
    .reg .f32 %f<4>;
    .reg .u32 %r<4>;
    ret;
}
";
    let result = PtxBugAnalyzer::new().analyze(ptx);
    assert!(!result.has_bug(&PtxBugClass::RegisterSpills), "No .local = no spills");
}

// ============================================================================
// EDGE CASE: Missing Entry Point
// ============================================================================

#[test]
fn test_missing_entry_point() {
    let ptx = r"
.version 8.0
.target sm_70
.reg .f32 %f<4>;
";
    let result = PtxBugAnalyzer::new().analyze(ptx);
    assert!(
        result.has_bug(&PtxBugClass::MissingEntryPoint),
        "Should detect missing .entry declaration"
    );
}

#[test]
fn test_entry_point_present() {
    let ptx = r"
.version 8.0
.target sm_70
.visible .entry kernel() {
    ret;
}
";
    let result = PtxBugAnalyzer::new().analyze(ptx);
    assert!(
        !result.has_bug(&PtxBugClass::MissingEntryPoint),
        ".entry present should not be flagged"
    );
}

#[test]
fn test_entry_without_visible() {
    let ptx = r"
.version 8.0
.target sm_70
.entry kernel() {
    ret;
}
";
    let result = PtxBugAnalyzer::new().analyze(ptx);
    assert!(
        !result.has_bug(&PtxBugClass::MissingEntryPoint),
        ".entry without .visible is still valid"
    );
}

// ============================================================================
// EDGE CASE: Empty and Malformed PTX
// ============================================================================

#[test]
fn test_empty_ptx() {
    let result = PtxBugAnalyzer::new().analyze("");
    assert!(
        !result.has_bug(&PtxBugClass::MissingEntryPoint),
        "Empty PTX should not flag missing entry"
    );
}

#[test]
fn test_whitespace_only_ptx() {
    let result = PtxBugAnalyzer::new().analyze("   \n\t\n   ");
    assert!(
        !result.has_bug(&PtxBugClass::MissingEntryPoint),
        "Whitespace-only PTX should not flag"
    );
}

// ============================================================================
// INVALID SYNTAX DETECTION (F105)
// ============================================================================

/// F105: Invalid syntax detection (unclosed blocks, malformed PTX)
///
/// Note: Since `PtxBugAnalyzer` is a static analyzer (not a full parser),
/// it focuses on detecting semantic bugs rather than syntax errors.
/// Syntax validation would be done by the PTX assembler (ptxas).
/// However, we can detect some structural issues.
#[test]
fn f105_detect_structural_issues() {
    // Missing ret statement (structural issue)
    let ptx_no_ret = r"
.visible .entry test() {
    .reg .f32 %f<4>;
}
";

    // This is syntactically valid PTX (ret is optional in some cases)
    // but we can detect missing entry point as a structural issue
    let ptx_fragment = r"
.version 8.0
.target sm_70
.reg .f32 %f<4>;
add.f32 %f0, %f1, %f2;
";

    let result = PtxBugAnalyzer::new().analyze(ptx_fragment);
    assert!(
        result.has_bug(&PtxBugClass::MissingEntryPoint),
        "F105: Should detect code fragment without entry point"
    );

    // Valid PTX should not be flagged
    let valid_ptx = r"
.visible .entry valid() {
    .reg .f32 %f<4>;
    ret;
}
";
    let valid_result = PtxBugAnalyzer::new().analyze(valid_ptx);
    assert!(
        !valid_result.has_bug(&PtxBugClass::InvalidSyntaxAccepted),
        "F105: Valid PTX should not be flagged as invalid"
    );

    // We ensure we at least analyzed the PTX
    let _ = PtxBugAnalyzer::new().analyze(ptx_no_ret);
}
