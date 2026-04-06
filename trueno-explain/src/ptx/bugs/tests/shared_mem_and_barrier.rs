use super::super::analyzer::*;
use super::super::types::*;

#[test]
fn test_shared_mem_u64_detection() {
    let ptx = r"
.visible .entry test() {
    .reg .u64 %rd<5>;
    .reg .f32 %f<2>;
    .shared .b8 smem[4096];
    st.shared.f32 [%rd0], %f0;
    ret;
}
";
    let result = PtxBugAnalyzer::new().analyze(ptx);
    assert!(result.has_bug(&PtxBugClass::SharedMemU64Addressing));
}

#[test]
fn test_shared_mem_u32_valid() {
    let ptx = r"
.visible .entry test() {
    .reg .u32 %r<5>;
    .reg .f32 %f<2>;
    .shared .b8 smem[4096];
    st.shared.f32 [%r0], %f0;
    ret;
}
";
    let result = PtxBugAnalyzer::new().analyze(ptx);
    assert!(!result.has_bug(&PtxBugClass::SharedMemU64Addressing));
}

#[test]
fn test_missing_barrier_sync_strict() {
    let ptx = r"
.visible .entry test() {
    .shared .b8 smem[1024];
    st.shared.f32 [%r0], %f0;
    ld.shared.f32 %f1, [%r1];
    ret;
}
";
    // Non-strict mode: no warning
    let normal_result = PtxBugAnalyzer::new().analyze(ptx);
    assert!(!normal_result.has_bug(&PtxBugClass::MissingBarrierSync));

    // Strict mode: warning
    let strict_result = PtxBugAnalyzer::strict().analyze(ptx);
    assert!(strict_result.has_bug(&PtxBugClass::MissingBarrierSync));
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
    // Should not have the broad "no bar.sync" warning
    let missing_barrier_bugs: Vec<_> = result.bugs_of_class(&PtxBugClass::MissingBarrierSync);
    // The specific st/ld pattern should not trigger since bar.sync is present
    assert!(missing_barrier_bugs
        .iter()
        .all(|b| !b.message.contains("ld.shared follows st.shared")));
}
