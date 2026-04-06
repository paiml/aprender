use super::super::analyzer::*;
use super::super::types::*;

/// Test `RedundantMoves` detection - consecutive mov chain
#[test]
fn test_redundant_moves_chain() {
    let ptx = r"
.visible .entry test() {
    mov.u32 %r1, %r0;
    mov.u32 %r2, %r1;
    ret;
}
";
    let result = PtxBugAnalyzer::new().analyze(ptx);
    assert!(result.has_bug(&PtxBugClass::RedundantMoves));
}

/// Test `RedundantMoves` - no chain (valid)
#[test]
fn test_redundant_moves_no_chain() {
    let ptx = r"
.visible .entry test() {
    mov.u32 %r1, %r0;
    add.u32 %r2, %r1, 1;
    mov.u32 %r3, %r2;
    ret;
}
";
    let result = PtxBugAnalyzer::new().analyze(ptx);
    assert!(!result.has_bug(&PtxBugClass::RedundantMoves));
}

/// Test `UnoptimizedMemoryPattern` - multiple single loads
#[test]
fn test_unoptimized_memory_single_loads() {
    let ptx = r"
.visible .entry test() {
    ld.global.f32 %f0, [%rd0];
    ld.global.f32 %f1, [%rd1];
    ld.global.f32 %f2, [%rd2];
    ld.global.f32 %f3, [%rd3];
    ret;
}
";
    let result = PtxBugAnalyzer::new().analyze(ptx);
    assert!(result.has_bug(&PtxBugClass::UnoptimizedMemoryPattern));
}

/// Test `UnoptimizedMemoryPattern` - vector loads (valid)
#[test]
fn test_unoptimized_memory_vector_loads() {
    let ptx = r"
.visible .entry test() {
    ld.global.v4.f32 {%f0, %f1, %f2, %f3}, [%rd0];
    ret;
}
";
    let result = PtxBugAnalyzer::new().analyze(ptx);
    assert!(!result.has_bug(&PtxBugClass::UnoptimizedMemoryPattern));
}

/// Test `UnoptimizedMemoryPattern` - few single loads (acceptable)
#[test]
fn test_unoptimized_memory_few_loads() {
    let ptx = r"
.visible .entry test() {
    ld.global.f32 %f0, [%rd0];
    ld.global.f32 %f1, [%rd1];
    ret;
}
";
    let result = PtxBugAnalyzer::new().analyze(ptx);
    // Only 2 single loads - below threshold of 4, should not flag
    assert!(!result.has_bug(&PtxBugClass::UnoptimizedMemoryPattern));
}

/// Test suspicious stride detection in strict mode
#[test]
fn test_unoptimized_memory_suspicious_stride() {
    let ptx = r"
.visible .entry test() {
    mul.wide.u32 %rd0, %r0, 17;
    ld.global.f32 %f0, [%rd0];
    ret;
}
";
    let result = PtxBugAnalyzer::strict().analyze(ptx);
    assert!(result.has_bug(&PtxBugClass::UnoptimizedMemoryPattern));
}

/// Test normal strides are not flagged
#[test]
fn test_unoptimized_memory_normal_stride() {
    let ptx = r"
.visible .entry test() {
    mul.wide.u32 %rd0, %r0, 4;
    ld.global.f32 %f0, [%rd0];
    ret;
}
";
    let result = PtxBugAnalyzer::strict().analyze(ptx);
    // Stride 4 is normal for f32
    assert!(!result.has_bug(&PtxBugClass::UnoptimizedMemoryPattern));
}

/// Test high register pressure detection
#[test]
fn test_high_register_pressure() {
    let ptx = r"
.visible .entry test() {
    .reg .b32 %r<64>;
    .reg .b64 %rd<16>;
    .reg .f32 %f<32>;
    .reg .pred %p<4>;
    ret;
}
";
    // 64 + 16 + 32 + 4 = 116 registers > 64 threshold
    let result = PtxBugAnalyzer::new().analyze(ptx);
    assert!(result.has_bug(&PtxBugClass::HighRegisterPressure));
}

/// Test acceptable register pressure (no bug)
#[test]
fn test_normal_register_pressure() {
    let ptx = r"
.visible .entry test() {
    .reg .b32 %r<16>;
    .reg .b64 %rd<8>;
    .reg .f32 %f<8>;
    .reg .pred %p<4>;
    ret;
}
";
    // 16 + 8 + 8 + 4 = 36 registers < 64 threshold
    let result = PtxBugAnalyzer::new().analyze(ptx);
    assert!(!result.has_bug(&PtxBugClass::HighRegisterPressure));
}

/// Test predicate overflow detection
#[test]
fn test_predicate_overflow() {
    let ptx = r"
.visible .entry test() {
    .reg .pred %p<12>;
    .reg .b32 %r<4>;
    ret;
}
";
    // 12 predicates > 8 limit
    let result = PtxBugAnalyzer::new().analyze(ptx);
    assert!(result.has_bug(&PtxBugClass::PredicateOverflow));
}

/// Test acceptable predicate count (no bug)
#[test]
fn test_normal_predicate_count() {
    let ptx = r"
.visible .entry test() {
    .reg .pred %p<8>;
    .reg .b32 %r<4>;
    ret;
}
";
    // 8 predicates = limit, should not flag
    let result = PtxBugAnalyzer::new().analyze(ptx);
    assert!(!result.has_bug(&PtxBugClass::PredicateOverflow));
}
