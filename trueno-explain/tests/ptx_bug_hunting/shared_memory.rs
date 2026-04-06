//! Edge case tests: Shared Memory Addressing

use super::*;

#[test]
fn test_shared_mem_u64_addressing_bug() {
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
    assert!(
        result.has_bug(&PtxBugClass::SharedMemU64Addressing),
        "BUG FOUND: Should detect 64-bit addressing for shared memory"
    );
}

#[test]
fn test_shared_mem_ld_u64_addressing() {
    let ptx = r"
.visible .entry test() {
    .reg .u64 %rd<5>;
    .reg .f32 %f<2>;
    .shared .b8 smem[4096];
    ld.shared.f32 %f0, [%rd0];
    ret;
}
";
    let result = PtxBugAnalyzer::new().analyze(ptx);
    assert!(
        result.has_bug(&PtxBugClass::SharedMemU64Addressing),
        "BUG FOUND: Should detect 64-bit addressing in ld.shared"
    );
}

#[test]
fn test_shared_mem_u32_addressing_valid() {
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
    assert!(
        !result.has_bug(&PtxBugClass::SharedMemU64Addressing),
        "32-bit addressing should be valid"
    );
}
