//! Edge case tests: Barrier Synchronization (PARITY-114 pattern)

use super::*;

#[test]
fn test_missing_barrier_strict_mode() {
    let ptx = r"
.visible .entry test() {
    .shared .b8 smem[1024];
    st.shared.f32 [%r0], %f0;
    ld.shared.f32 %f1, [%r1];
    ret;
}
";
    let result = PtxBugAnalyzer::strict().analyze(ptx);
    assert!(
        result.has_bug(&PtxBugClass::MissingBarrierSync),
        "PARITY-114: Should detect missing barrier between st.shared and ld.shared"
    );
}

#[test]
fn test_barrier_present_valid() {
    let ptx = r"
.visible .entry test() {
    .shared .b8 smem[1024];
    st.shared.f32 [%r0], %f0;
    bar.sync 0;
    ld.shared.f32 %f1, [%r1];
    ret;
}
";
    let result = PtxBugAnalyzer::strict().analyze(ptx);
    // With barrier present, the specific st/ld bug should not trigger
    let missing_barrier_bugs: Vec<_> = result.bugs_of_class(&PtxBugClass::MissingBarrierSync);
    let has_st_ld_bug =
        missing_barrier_bugs.iter().any(|b| b.message.contains("ld.shared follows st.shared"));
    assert!(!has_st_ld_bug, "Should not flag missing barrier when bar.sync is present");
}

#[test]
fn test_multiple_barriers() {
    let ptx = r"
.visible .entry test() {
    .shared .b8 smem[1024];
    st.shared.f32 [%r0], %f0;
    bar.sync 0;
    ld.shared.f32 %f1, [%r1];
    st.shared.f32 [%r2], %f2;
    bar.sync 0;
    ld.shared.f32 %f3, [%r3];
    ret;
}
";
    let result = PtxBugAnalyzer::strict().analyze(ptx);
    // Check that we don't have false positives for properly synchronized code
    let missing_barrier_bugs: Vec<_> = result.bugs_of_class(&PtxBugClass::MissingBarrierSync);
    // There should be no st/ld pattern bugs since barriers are correctly placed
    assert!(
        missing_barrier_bugs.iter().all(|b| !b.message.contains("ld.shared follows st.shared")),
        "Should not flag when barriers are correctly placed"
    );
}
