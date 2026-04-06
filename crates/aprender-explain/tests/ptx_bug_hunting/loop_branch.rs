//! Edge case tests: Loop Branch Direction

use super::*;

#[test]
fn test_loop_branch_to_end_unconditional() {
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
    assert!(
        result.has_bug(&PtxBugClass::LoopBranchToEnd),
        "Should detect unconditional branch to loop end"
    );
}

#[test]
fn test_loop_branch_conditional_valid() {
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
    assert!(
        !result.has_bug(&PtxBugClass::LoopBranchToEnd),
        "Conditional branch should not be flagged"
    );
}

#[test]
fn test_loop_branch_to_start_valid() {
    let ptx = r"
.visible .entry test() {
loop_start:
    // loop body
    bra loop_start;
    ret;
}
";
    let result = PtxBugAnalyzer::strict().analyze(ptx);
    assert!(
        !result.has_bug(&PtxBugClass::LoopBranchToEnd),
        "Branch to loop start should not be flagged"
    );
}
