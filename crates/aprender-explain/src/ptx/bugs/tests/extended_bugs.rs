use super::super::analyzer::*;
use super::super::types::*;

// ========================================================================
// EMPTY LOOP BODY TESTS
// ========================================================================

/// Test empty loop body detection
#[test]
fn test_empty_loop_body_detected() {
    let ptx = r"
.visible .entry test() {
empty_loop:
    // Just comments here
    bra empty_loop;
    ret;
}
";
    let result = PtxBugAnalyzer::new().analyze(ptx);
    assert!(result.has_bug(&PtxBugClass::EmptyLoopBody));
}

/// Test valid loop body not flagged
#[test]
fn test_valid_loop_body_not_flagged() {
    let ptx = r"
.visible .entry test() {
    .reg .f32 %f<4>;
    .reg .u32 %r<4>;
compute_loop:
    add.f32 %f0, %f0, %f1;
    add.u32 %r0, %r0, 1;
    setp.lt.u32 %p0, %r0, %r1;
    @%p0 bra compute_loop;
    ret;
}
";
    let result = PtxBugAnalyzer::new().analyze(ptx);
    assert!(!result.has_bug(&PtxBugClass::EmptyLoopBody));
}

/// Test loop with only conditional branch not flagged
#[test]
fn test_loop_with_exit_condition_not_flagged() {
    let ptx = r"
.visible .entry test() {
    .reg .u32 %r<4>;
    .reg .pred %p<2>;
check_loop:
    setp.lt.u32 %p0, %r0, %r1;
    @%p0 bra check_loop;
    ret;
}
";
    let result = PtxBugAnalyzer::new().analyze(ptx);
    // Has setp which is computation
    assert!(!result.has_bug(&PtxBugClass::EmptyLoopBody));
}

// ========================================================================
// MISSING BOUNDS CHECK TESTS
// ========================================================================

/// Test missing bounds check detection
#[test]
fn test_missing_bounds_check() {
    let ptx = r"
.visible .entry test() {
    .reg .u64 %rd<4>;
    .reg .f32 %f<4>;
    mov.u32 %r0, %tid.x;
    ld.global.f32 %f0, [%rd0];
    st.global.f32 [%rd1], %f0;
    ret;
}
";
    let result = PtxBugAnalyzer::new().analyze(ptx);
    assert!(result.has_bug(&PtxBugClass::MissingBoundsCheck));
}

/// Test proper bounds check not flagged
#[test]
fn test_proper_bounds_check_not_flagged() {
    let ptx = r"
.visible .entry test() {
    .reg .u64 %rd<4>;
    .reg .f32 %f<4>;
    .reg .u32 %r<4>;
    .reg .pred %p<2>;
    mov.u32 %r0, %tid.x;
    setp.lt.u32 %p0, %r0, %r1;
    @%p0 bra do_work;
    bra done;
do_work:
    ld.global.f32 %f0, [%rd0];
    st.global.f32 [%rd1], %f0;
done:
    ret;
}
";
    let result = PtxBugAnalyzer::new().analyze(ptx);
    assert!(!result.has_bug(&PtxBugClass::MissingBoundsCheck));
}

/// Test kernel without global memory not flagged
#[test]
fn test_no_global_mem_no_bounds_check_needed() {
    let ptx = r"
.visible .entry test() {
    .reg .u32 %r<4>;
    mov.u32 %r0, %tid.x;
    add.u32 %r1, %r0, 1;
    ret;
}
";
    let result = PtxBugAnalyzer::new().analyze(ptx);
    // No global memory, so no bounds check needed
    assert!(!result.has_bug(&PtxBugClass::MissingBoundsCheck));
}

// ========================================================================
// DEAD CODE TESTS
// ========================================================================

/// Test dead code after ret
#[test]
fn test_dead_code_after_ret() {
    let ptx = r"
.visible .entry test() {
    .reg .f32 %f<4>;
    add.f32 %f0, %f1, %f2;
    ret;
    mul.f32 %f3, %f0, %f1;
}
";
    let result = PtxBugAnalyzer::new().analyze(ptx);
    assert!(result.has_bug(&PtxBugClass::DeadCode));
}

/// Test dead code after unconditional branch
#[test]
fn test_dead_code_after_branch() {
    let ptx = r"
.visible .entry test() {
    .reg .f32 %f<4>;
    bra skip;
    add.f32 %f0, %f1, %f2;
skip:
    ret;
}
";
    let result = PtxBugAnalyzer::new().analyze(ptx);
    assert!(result.has_bug(&PtxBugClass::DeadCode));
}

/// Test reachable code not flagged (label after branch)
#[test]
fn test_reachable_code_not_flagged() {
    let ptx = r"
.visible .entry test() {
    .reg .f32 %f<4>;
    .reg .pred %p<2>;
    @%p0 bra skip;
    add.f32 %f0, %f1, %f2;
skip:
    mul.f32 %f3, %f0, %f1;
    ret;
}
";
    let result = PtxBugAnalyzer::new().analyze(ptx);
    // Conditional branch, code after is reachable
    assert!(!result.has_bug(&PtxBugClass::DeadCode));
}

/// Test code after label is reachable
#[test]
fn test_code_after_label_reachable() {
    let ptx = r"
.visible .entry test() {
    .reg .f32 %f<4>;
    bra middle;
middle:
    add.f32 %f0, %f1, %f2;
    ret;
}
";
    let result = PtxBugAnalyzer::new().analyze(ptx);
    // The add after middle: label is reachable via the branch
    assert!(!result.has_bug(&PtxBugClass::DeadCode));
}

// ========================================================================
// NEW BUG CLASS SEVERITY/CODE TESTS
// ========================================================================

/// Test new extended bug class severities
#[test]
fn test_extended_bug_severities() {
    assert_eq!(PtxBugClass::EmptyLoopBody.severity(), BugSeverity::High);
    assert_eq!(PtxBugClass::MissingBoundsCheck.severity(), BugSeverity::High);
    assert_eq!(PtxBugClass::DeadCode.severity(), BugSeverity::Medium);
}

/// Test new extended bug class codes
#[test]
fn test_extended_bug_codes() {
    assert_eq!(PtxBugClass::EmptyLoopBody.code(), "EMPTY_LOOP");
    assert_eq!(PtxBugClass::MissingBoundsCheck.code(), "NO_BOUNDS_CHECK");
    assert_eq!(PtxBugClass::DeadCode.code(), "DEAD_CODE");
}

// ========================================================================
// PARITY-114: EARLY EXIT BEFORE BARRIER TESTS
// ========================================================================

/// PARITY-114: Detect conditional early exit before barrier
#[test]
fn test_parity114_conditional_exit_before_barrier() {
    let ptx = r"
.visible .entry kernel() {
    mov.u32 %r0, %tid.x;
    setp.lt.u32 %p0, %r0, 32;

loop_start:
    @!%p0 bra exit;
    ld.shared.f32 %f0, [%r0];
    bar.sync 0;
    st.shared.f32 [%r0], %f0;
    bra loop_start;

loop_start_end:
done:
    ret;
}
";
    let result = PtxBugAnalyzer::strict().analyze(ptx);
    assert!(result.has_bug(&PtxBugClass::EarlyExitBeforeBarrier));
    // Verify it's P0 Critical
    assert_eq!(PtxBugClass::EarlyExitBeforeBarrier.severity(), BugSeverity::Critical);
}

/// PARITY-114: Detect unconditional early exit before barrier
#[test]
fn test_parity114_unconditional_exit_before_barrier() {
    let ptx = r"
.visible .entry kernel() {
loop_start:
    bra exit;
    bar.sync 0;
    bra loop_start;

loop_start_end:
done:
    ret;
}
";
    let result = PtxBugAnalyzer::strict().analyze(ptx);
    assert!(result.has_bug(&PtxBugClass::EarlyExitBeforeBarrier));
}

/// PARITY-114: Safe kernel with barrier before any possible exit
#[test]
fn test_parity114_safe_barrier_first() {
    let ptx = r"
.visible .entry kernel() {
    mov.u32 %r0, %tid.x;
    setp.lt.u32 %p0, %r0, 32;

loop_start:
    ld.shared.f32 %f0, [%r0];
    bar.sync 0;
    st.shared.f32 [%r0], %f0;
    bra loop_start;

loop_start_end:
    @!%p0 bra exit;
    st.global.f32 [%r1], %f0;
exit:
    ret;
}
";
    let result = PtxBugAnalyzer::strict().analyze(ptx);
    assert!(!result.has_bug(&PtxBugClass::EarlyExitBeforeBarrier));
}

/// PARITY-114: Exit after loop end is safe
#[test]
fn test_parity114_exit_after_loop_is_safe() {
    let ptx = r"
.visible .entry kernel() {
k_tile_loop:
    bar.sync 0;
    ld.shared.f32 %f0, [%r0];
    bra k_tile_loop;

k_tile_end:
    @!%p0 bra exit;
    st.global.f32 [%r1], %f0;
done:
    ret;
}
";
    let result = PtxBugAnalyzer::strict().analyze(ptx);
    assert!(!result.has_bug(&PtxBugClass::EarlyExitBeforeBarrier));
}

/// PARITY-114: Non-strict mode does not flag barrier issues
#[test]
fn test_parity114_non_strict_mode() {
    let ptx = r"
.visible .entry kernel() {
loop_start:
    @!%p0 bra exit;
    bar.sync 0;
    bra loop_start;

loop_start_end:
done:
    ret;
}
";
    // Non-strict mode should NOT flag this
    let result = PtxBugAnalyzer::new().analyze(ptx);
    assert!(!result.has_bug(&PtxBugClass::EarlyExitBeforeBarrier));

    // Strict mode SHOULD flag this
    let strict_result = PtxBugAnalyzer::strict().analyze(ptx);
    assert!(strict_result.has_bug(&PtxBugClass::EarlyExitBeforeBarrier));
}

/// PARITY-114: Bug class properties
#[test]
fn test_parity114_bug_class_properties() {
    assert_eq!(PtxBugClass::EarlyExitBeforeBarrier.code(), "EARLY_EXIT_BARRIER");
    assert_eq!(PtxBugClass::EarlyExitBeforeBarrier.severity(), BugSeverity::Critical);
}

/// PARITY-114: `kv_loop` pattern (attention kernels) - safe after fix
#[test]
fn test_parity114_attention_kv_loop_safe() {
    let ptx = r"
.visible .entry flash_attention() {
kv_loop:
    bar.sync 0;
    wmma.mma.sync.aligned.row.col.m16n16k16.f32.f16.f16.f32;
    bra kv_loop;

kv_loop_end:
    @!%p_valid bra exit;
    st.global.f32 [%out], %f0;
done:
    ret;
}
";
    let result = PtxBugAnalyzer::strict().analyze(ptx);
    assert!(!result.has_bug(&PtxBugClass::EarlyExitBeforeBarrier));
}
