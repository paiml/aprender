use super::super::analyzer::*;
use super::super::types::*;

/// Test placeholder code detection - "omitted"
#[test]
fn test_placeholder_code_omitted() {
    let ptx = r"
.visible .entry test() {
    // ... loading logic omitted for brevity
    ret;
}
";
    let result = PtxBugAnalyzer::new().analyze(ptx);
    assert!(result.has_bug(&PtxBugClass::PlaceholderCode));
}

/// Test placeholder code detection - "simplified"
#[test]
fn test_placeholder_code_simplified() {
    let ptx = r"
.visible .entry test() {
    // Simplified: only first element
    st.global.f32 [%rd0], %f0;
    ret;
}
";
    let result = PtxBugAnalyzer::new().analyze(ptx);
    assert!(result.has_bug(&PtxBugClass::PlaceholderCode));
}

/// Test placeholder code detection - "placeholder"
#[test]
fn test_placeholder_code_explicit() {
    let ptx = r"
.visible .entry test() {
    // This is placeholder code for now
    ret;
}
";
    let result = PtxBugAnalyzer::new().analyze(ptx);
    assert!(result.has_bug(&PtxBugClass::PlaceholderCode));
}

/// Test no placeholder code (clean kernel)
#[test]
fn test_no_placeholder_code() {
    let ptx = r"
.visible .entry test() {
    // Load input
    ld.global.f32 %f0, [%rd0];
    // Compute result
    mul.f32 %f1, %f0, %f0;
    // Store output
    st.global.f32 [%rd1], %f1;
    ret;
}
";
    let result = PtxBugAnalyzer::new().analyze(ptx);
    assert!(!result.has_bug(&PtxBugClass::PlaceholderCode));
}

/// Test new bug class severities
#[test]
fn test_new_bug_severities() {
    assert_eq!(PtxBugClass::HighRegisterPressure.severity(), BugSeverity::High);
    assert_eq!(PtxBugClass::PredicateOverflow.severity(), BugSeverity::High);
    assert_eq!(PtxBugClass::PlaceholderCode.severity(), BugSeverity::High);
}

/// Test new bug class codes
#[test]
fn test_new_bug_codes() {
    assert_eq!(PtxBugClass::HighRegisterPressure.code(), "HIGH_REG_PRESSURE");
    assert_eq!(PtxBugClass::PredicateOverflow.code(), "PRED_OVERFLOW");
    assert_eq!(PtxBugClass::PlaceholderCode.code(), "PLACEHOLDER_CODE");
}

// ========================================================================
// WHITELIST TESTS
// ========================================================================

/// Test whitelist suppresses matching bug
#[test]
fn test_whitelist_suppresses_bug() {
    let ptx = r"
.visible .entry q4k_gemm_ggml() {
    .reg .b32 %r<64>;
    .reg .b64 %rd<16>;
    .reg .f32 %f<32>;
    ret;
}
";
    // Without whitelist: should flag high register pressure
    let result_no_whitelist = PtxBugAnalyzer::new().analyze(ptx);
    assert!(result_no_whitelist.has_bug(&PtxBugClass::HighRegisterPressure));

    // With quantized whitelist: q4k* should be suppressed
    let result_with_whitelist = PtxBugAnalyzer::with_quantized_whitelist().analyze(ptx);
    assert!(!result_with_whitelist.has_bug(&PtxBugClass::HighRegisterPressure));
}

/// Test whitelist with exact kernel name match
#[test]
fn test_whitelist_exact_match() {
    let ptx = r"
.visible .entry special_kernel() {
    .reg .b32 %r<64>;
    .reg .b64 %rd<16>;
    .reg .f32 %f<32>;
    ret;
}
";
    // With exact match whitelist
    let analyzer = PtxBugAnalyzer::new().with_whitelist(
        "special_kernel",
        PtxBugClass::HighRegisterPressure,
        "Expected high regs",
    );
    let result = analyzer.analyze(ptx);
    assert!(!result.has_bug(&PtxBugClass::HighRegisterPressure));
}

/// Test whitelist doesn't suppress non-matching kernels
#[test]
fn test_whitelist_no_match() {
    let ptx = r"
.visible .entry other_kernel() {
    .reg .b32 %r<64>;
    .reg .b64 %rd<16>;
    .reg .f32 %f<32>;
    ret;
}
";
    // q4k* whitelist should not match "other_kernel"
    let result = PtxBugAnalyzer::with_quantized_whitelist().analyze(ptx);
    assert!(result.has_bug(&PtxBugClass::HighRegisterPressure));
}

/// Test performance whitelist covers Tensor Core kernels
#[test]
fn test_performance_whitelist_tensor_core() {
    let ptx = r"
.visible .entry gemm_tensor_core() {
    .reg .b32 %r<64>;
    .reg .b64 %rd<32>;
    .reg .f32 %f<64>;
    .reg .pred %p<12>;
    ret;
}
";
    // Without whitelist: should flag both issues
    let result_no_whitelist = PtxBugAnalyzer::new().analyze(ptx);
    assert!(result_no_whitelist.has_bug(&PtxBugClass::HighRegisterPressure));
    assert!(result_no_whitelist.has_bug(&PtxBugClass::PredicateOverflow));

    // With performance whitelist: both should be suppressed
    let result_with_whitelist = PtxBugAnalyzer::with_performance_whitelist().analyze(ptx);
    assert!(!result_with_whitelist.has_bug(&PtxBugClass::HighRegisterPressure));
    assert!(!result_with_whitelist.has_bug(&PtxBugClass::PredicateOverflow));
}

/// Test performance whitelist covers attention kernels
#[test]
fn test_performance_whitelist_attention() {
    let ptx = r"
.visible .entry flash_attention_tensor_core() {
    .reg .b32 %r<64>;
    .reg .b64 %rd<32>;
    .reg .f32 %f<48>;
    ret;
}
";
    // With performance whitelist: register pressure should be suppressed
    let result = PtxBugAnalyzer::with_performance_whitelist().analyze(ptx);
    assert!(!result.has_bug(&PtxBugClass::HighRegisterPressure));
}
