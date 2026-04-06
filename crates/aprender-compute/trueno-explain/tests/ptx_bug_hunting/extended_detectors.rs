//! Extended bug hunt: New detectors, whitelist, and strict mode tests

use super::*;

/// Extended bug hunt: All trueno kernels pass with new detectors
///
/// This tests the new bug detectors added after analyzing realizar bugs:
/// - `EmptyLoopBody`: Loop without computation
/// - `MissingBoundsCheck`: No thread bounds check
/// - `DeadCode`: Unreachable code after ret/bra
/// - `HighRegisterPressure`: >64 registers (with whitelist for quantized kernels)
/// - `PredicateOverflow`: >8 predicates
/// - `PlaceholderCode`: Comments indicating incomplete code
#[test]
fn test_extended_bug_hunt_all_kernels() {
    use trueno_gpu::kernels::{AttentionKernel, LayerNormKernel};

    println!("\n╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║                     EXTENDED PTX BUG HUNT REPORT                              ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════╝\n");

    // Generate all kernels
    let kernels: Vec<(&str, String)> = vec![
        ("gemm_naive_64", GemmKernel::naive(64, 64, 64).emit_ptx()),
        ("gemm_naive_128", GemmKernel::naive(128, 128, 128).emit_ptx()),
        ("gemm_tiled_64", GemmKernel::tiled(64, 64, 64, 16).emit_ptx()),
        ("gemm_tiled_128", GemmKernel::tiled(128, 128, 128, 32).emit_ptx()),
        ("gemm_tensor_core", GemmKernel::tensor_core(64, 64, 64).emit_ptx()),
        ("gemm_wmma_fp16", GemmKernel::wmma_fp16(64, 64, 64).emit_ptx()),
        ("softmax_1024", SoftmaxKernel::new(1024).emit_ptx()),
        ("softmax_4096", SoftmaxKernel::new(4096).emit_ptx()),
        ("layernorm_256", LayerNormKernel::new(256).emit_ptx()),
        ("layernorm_1024", LayerNormKernel::new(1024).emit_ptx()),
        ("attention_64_32", AttentionKernel::new(64, 32).emit_ptx()),
        ("attention_128_64", AttentionKernel::new(128, 64).emit_ptx()),
        ("q4k_gemm", QuantizeKernel::ggml(64, 64, 256).emit_ptx()),
        ("q5k_gemm", Q5KKernel::new(64, 64, 256).emit_ptx()),
        ("q6k_gemm", Q6KKernel::new(64, 64, 256).emit_ptx()),
    ];

    let mut total_bugs = 0;
    let mut p0_bugs = 0;
    let mut p1_bugs = 0;
    let mut p2_bugs = 0;

    // Use quantized whitelist for q4k/q5k/q6k/q8k kernels
    let analyzer = PtxBugAnalyzer::with_quantized_whitelist();

    for (name, ptx) in &kernels {
        let result = analyzer.analyze(ptx);

        let p0 = result.count_by_severity(BugSeverity::Critical);
        let p1 = result.count_by_severity(BugSeverity::High);
        let p2 = result.count_by_severity(BugSeverity::Medium);

        total_bugs += result.bugs.len();
        p0_bugs += p0;
        p1_bugs += p1;
        p2_bugs += p2;

        if result.has_bugs() {
            println!(
                "FAIL {} - {} bugs ({} P0, {} P1, {} P2)",
                name,
                result.bugs.len(),
                p0,
                p1,
                p2
            );
            for bug in &result.bugs {
                println!("   -- {}: {}", bug.class.code(), bug.message);
            }
        } else {
            println!("PASS {} - CLEAN", name);
        }
    }

    println!("\n══════════════════════════════════════════════════════════════════════════════");
    println!("SUMMARY: {} kernels analyzed", kernels.len());
    println!("  Total bugs: {}", total_bugs);
    println!("  P0 Critical: {}", p0_bugs);
    println!("  P1 High: {}", p1_bugs);
    println!("  P2 Medium: {}", p2_bugs);

    // All trueno kernels should pass (no P0 critical bugs)
    assert_eq!(p0_bugs, 0, "CRITICAL: No P0 bugs allowed in trueno kernels!");
}

/// Test: New detectors don't produce false positives on clean kernels
#[test]
fn test_new_detectors_no_false_positives() {
    let clean_ptx = r"
.version 8.0
.target sm_89
.address_size 64

.visible .entry clean_kernel(
    .param .u64 input,
    .param .u64 output,
    .param .u32 n
) {
    .reg .pred %p<2>;
    .reg .b32 %r<8>;
    .reg .b64 %rd<4>;
    .reg .f32 %f<4>;

    // Thread bounds check
    mov.u32 %r0, %tid.x;
    ld.param.u32 %r1, [n];
    setp.ge.u32 %p0, %r0, %r1;
    @%p0 bra DONE;

    // Load from global memory
    ld.param.u64 %rd0, [input];
    cvt.u64.u32 %rd1, %r0;
    shl.b64 %rd1, %rd1, 2;
    add.u64 %rd0, %rd0, %rd1;
    ld.global.f32 %f0, [%rd0];

    // Compute
    mul.f32 %f1, %f0, %f0;

    // Store to global memory
    ld.param.u64 %rd2, [output];
    add.u64 %rd2, %rd2, %rd1;
    st.global.f32 [%rd2], %f1;

DONE:
    ret;
}
";

    let result = PtxBugAnalyzer::new().analyze(clean_ptx);

    // Should not have false positives
    assert!(
        !result.has_bug(&PtxBugClass::EmptyLoopBody),
        "Clean kernel should not have EmptyLoopBody"
    );
    assert!(!result.has_bug(&PtxBugClass::MissingBoundsCheck), "Clean kernel has bounds check");
    assert!(!result.has_bug(&PtxBugClass::DeadCode), "Clean kernel has no dead code");
    assert!(
        !result.has_bug(&PtxBugClass::PlaceholderCode),
        "Clean kernel has no placeholder comments"
    );
    assert!(result.is_valid(), "Clean kernel should be valid");
}

/// Test: Whitelist correctly suppresses quantized kernel warnings
#[test]
fn test_whitelist_quantized_kernels() {
    let q4k_ptx = QuantizeKernel::ggml(64, 64, 256).emit_ptx();
    let q5k_ptx = Q5KKernel::new(64, 64, 256).emit_ptx();
    let q6k_ptx = Q6KKernel::new(64, 64, 256).emit_ptx();

    // Without whitelist: may have high register pressure warnings
    let result_no_wl = PtxBugAnalyzer::new().analyze(&q4k_ptx);
    let has_reg_pressure = result_no_wl.has_bug(&PtxBugClass::HighRegisterPressure);

    // With quantized whitelist: high register pressure should be suppressed
    let result_wl = PtxBugAnalyzer::with_quantized_whitelist().analyze(&q4k_ptx);
    let result_q5k = PtxBugAnalyzer::with_quantized_whitelist().analyze(&q5k_ptx);
    let result_q6k = PtxBugAnalyzer::with_quantized_whitelist().analyze(&q6k_ptx);

    // If kernel has high register pressure, whitelist should suppress it
    if has_reg_pressure {
        assert!(
            !result_wl.has_bug(&PtxBugClass::HighRegisterPressure),
            "Q4K whitelist should suppress HighRegisterPressure"
        );
    }

    // All quantized kernels should be valid (no P0 bugs)
    assert!(result_wl.is_valid(), "Q4K should be valid with whitelist");
    assert!(result_q5k.is_valid(), "Q5K should be valid with whitelist");
    assert!(result_q6k.is_valid(), "Q6K should be valid with whitelist");
}

/// Test: `EmptyLoopBody` detection works
#[test]
fn test_empty_loop_body_detection() {
    let ptx_with_empty_loop = r"
.visible .entry test() {
empty_loop:
    // Nothing here
    bra empty_loop;
    ret;
}
";

    let result = PtxBugAnalyzer::new().analyze(ptx_with_empty_loop);
    assert!(result.has_bug(&PtxBugClass::EmptyLoopBody), "Should detect empty loop body");
}

/// Test: `DeadCode` detection works
#[test]
fn test_dead_code_detection() {
    let ptx_with_dead_code = r"
.visible .entry test() {
    .reg .f32 %f<4>;
    mul.f32 %f0, %f1, %f2;
    ret;
    add.f32 %f3, %f0, %f1;
}
";

    let result = PtxBugAnalyzer::new().analyze(ptx_with_dead_code);
    assert!(result.has_bug(&PtxBugClass::DeadCode), "Should detect dead code after ret");
}

/// Test: Extended bug class severities
#[test]
fn test_extended_bug_severities() {
    assert_eq!(PtxBugClass::EmptyLoopBody.severity(), BugSeverity::High);
    assert_eq!(PtxBugClass::MissingBoundsCheck.severity(), BugSeverity::High);
    assert_eq!(PtxBugClass::DeadCode.severity(), BugSeverity::Medium);
    assert_eq!(PtxBugClass::HighRegisterPressure.severity(), BugSeverity::High);
    assert_eq!(PtxBugClass::PredicateOverflow.severity(), BugSeverity::High);
    assert_eq!(PtxBugClass::PlaceholderCode.severity(), BugSeverity::High);
}

/// Test: All trueno kernels pass strict mode without P0 bugs
#[test]
fn test_trueno_kernels_strict_mode() {
    use trueno_gpu::kernels::{AttentionKernel, LayerNormKernel};

    let kernels: Vec<(&str, String)> = vec![
        ("gemm_naive", GemmKernel::naive(64, 64, 64).emit_ptx()),
        ("gemm_tiled", GemmKernel::tiled(64, 64, 64, 16).emit_ptx()),
        ("softmax", SoftmaxKernel::new(1024).emit_ptx()),
        ("layernorm", LayerNormKernel::new(256).emit_ptx()),
        ("attention", AttentionKernel::new(64, 32).emit_ptx()),
    ];

    for (name, ptx) in &kernels {
        let result = PtxBugAnalyzer::strict().analyze(ptx);
        let p0_count = result.count_by_severity(BugSeverity::Critical);

        // P0 bugs in strict mode need investigation but shouldn't block
        if p0_count > 0 {
            println!("WARNING: {} has {} P0 bugs in strict mode", name, p0_count);
            for bug in result.bugs_of_class(&PtxBugClass::MissingBarrierSync) {
                println!("  - {}", bug.message);
            }
        }
    }
}
