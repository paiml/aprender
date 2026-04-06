use super::super::analyzer::*;
use super::super::types::*;

#[test]
fn test_loop_branch_to_end_detection() {
    let ptx = r"
.visible .entry test() {
main_loop:
    // loop body
    bra main_loop_end;
main_loop_end:
    ret;
}
";
    let result = PtxBugAnalyzer::strict().analyze(ptx);
    assert!(result.has_bug(&PtxBugClass::LoopBranchToEnd));
}

#[test]
fn test_conditional_branch_not_flagged() {
    let ptx = r"
.visible .entry test() {
loop_start:
    @%p0 bra loop_end;
    bra loop_start;
loop_end:
    ret;
}
";
    let result = PtxBugAnalyzer::strict().analyze(ptx);
    // Conditional branch should NOT be flagged
    assert!(!result.has_bug(&PtxBugClass::LoopBranchToEnd));
}

#[test]
fn test_register_spills_detection() {
    let ptx = r"
.visible .entry test() {
    .local .align 4 .b8 __local_depot[32];
    ret;
}
";
    let result = PtxBugAnalyzer::new().analyze(ptx);
    assert!(result.has_bug(&PtxBugClass::RegisterSpills));
}

#[test]
fn test_missing_entry_point_detection() {
    let ptx = r"
.version 8.0
.target sm_70
.reg .f32 %f<4>;
";
    let result = PtxBugAnalyzer::new().analyze(ptx);
    assert!(result.has_bug(&PtxBugClass::MissingEntryPoint));
}

#[test]
fn test_valid_kernel_no_bugs() {
    let ptx = r"
.version 8.0
.target sm_70
.visible .entry valid_kernel() {
    .reg .f32 %f<4>;
    .reg .u32 %r<4>;
    ret;
}
";
    let result = PtxBugAnalyzer::new().analyze(ptx);
    assert!(result.is_valid());
    assert!(!result.has_bugs());
}

#[test]
fn test_bug_severity_classification() {
    assert_eq!(PtxBugClass::MissingBarrierSync.severity(), BugSeverity::Critical);
    assert_eq!(PtxBugClass::SharedMemU64Addressing.severity(), BugSeverity::Critical);
    assert_eq!(PtxBugClass::RegisterSpills.severity(), BugSeverity::High);
    assert_eq!(PtxBugClass::MissingEntryPoint.severity(), BugSeverity::FalsePositive);
}

#[test]
fn test_bug_report_format() {
    let ptx = r"
.visible .entry test() {
    .shared .b8 smem[1024];
    st.shared.f32 [%rd0], %f0;
    ret;
}
";
    let result = PtxBugAnalyzer::new().analyze(ptx);
    let report = result.format_report();

    assert!(report.contains("PTX BUG HUNTING REPORT"));
    assert!(report.contains("P0 CRITICAL BUGS:"));
    assert!(report.contains("SUMMARY"));
}

#[test]
fn test_kernel_name_extraction() {
    let ptx = r"
.visible .entry gemm_tiled() {
    ret;
}
";
    let result = PtxBugAnalyzer::new().analyze(ptx);
    assert_eq!(result.kernel_name, Some("gemm_tiled".to_string()));
}

#[test]
fn test_count_by_severity() {
    let report = PtxBugReport {
        kernel_name: Some("test".to_string()),
        bugs: vec![
            PtxBug {
                class: PtxBugClass::MissingBarrierSync,
                line: 1,
                instruction: "test".to_string(),
                message: "test".to_string(),
                fix: None,
            },
            PtxBug {
                class: PtxBugClass::RegisterSpills,
                line: 2,
                instruction: "test".to_string(),
                message: "test".to_string(),
                fix: None,
            },
        ],
        lines_analyzed: 10,
        strict_mode: true,
    };

    assert_eq!(report.count_by_severity(BugSeverity::Critical), 1);
    assert_eq!(report.count_by_severity(BugSeverity::High), 1);
    assert_eq!(report.count_by_severity(BugSeverity::Medium), 0);
}

/// F101: Detect `st.shared [%rd0]`
#[test]
fn f101_detect_shared_u64_addressing() {
    let ptx = "st.shared.f32 [%rd5], %f0;";
    let result = PtxBugAnalyzer::new().analyze(ptx);
    assert!(result.has_bug(&PtxBugClass::SharedMemU64Addressing));
}

/// F102: Detect missing `bar.sync`
#[test]
fn f102_detect_missing_barrier() {
    let ptx = r"
.visible .entry test() {
    .shared .b8 smem[1024];
    st.shared.f32 [%r0], %f0;
    ld.shared.f32 %f1, [%r1];
    ret;
}
";
    let result = PtxBugAnalyzer::strict().analyze(ptx);
    assert!(result.has_bug(&PtxBugClass::MissingBarrierSync));
}

/// F103: Detect `bra loop_end` in loop
#[test]
fn f103_detect_loop_branch_end() {
    let ptx = r"
.entry test() {
test_loop:
    bra test_loop_end;
test_loop_end:
    ret;
}
";
    let result = PtxBugAnalyzer::strict().analyze(ptx);
    assert!(result.has_bug(&PtxBugClass::LoopBranchToEnd));
}

/// F104: Valid PTX passes
#[test]
fn f104_valid_ptx_passes() {
    let ptx = r"
.version 8.0
.target sm_70
.visible .entry valid() {
    .reg .f32 %f<4>;
    ret;
}
";
    let result = PtxBugAnalyzer::new().analyze(ptx);
    assert!(result.is_valid());
}

/// F106: Missing `.entry` detected
#[test]
fn f106_missing_entry_detected() {
    let ptx = ".version 8.0
.target sm_70
.reg .f32 %f<4>;";
    let result = PtxBugAnalyzer::new().analyze(ptx);
    assert!(result.has_bug(&PtxBugClass::MissingEntryPoint));
}
