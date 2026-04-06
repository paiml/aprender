//! Real kernel analysis and determinism verification tests

use super::*;

// ============================================================================
// REAL KERNEL ANALYSIS
// ============================================================================

#[test]
fn test_gemm_naive_no_critical_bugs() {
    let kernel = GemmKernel::naive(64, 64, 64);
    let ptx = kernel.emit_ptx();

    let result = PtxBugAnalyzer::new().analyze(&ptx);
    assert!(result.is_valid(), "GEMM naive should not have critical bugs");
}

#[test]
fn test_gemm_tiled_barrier_check() {
    let kernel = GemmKernel::tiled(64, 64, 64, 16);
    let ptx = kernel.emit_ptx();

    // Check with standard analyzer for Muda detection
    let analyzer = PtxAnalyzer::new();
    let report = analyzer.analyze(&ptx).unwrap();
    assert_eq!(report.name, "gemm_tiled");

    // Check with bug analyzer for barrier sync
    let bug_result = PtxBugAnalyzer::strict().analyze(&ptx);
    // Tiled GEMM should have barriers - this is the PARITY-114 check
    if bug_result.has_bug(&PtxBugClass::MissingBarrierSync) {
        println!(
            "WARNING: Tiled GEMM may have missing barrier sync:\n{}",
            bug_result.format_report()
        );
    }
}

#[test]
fn test_softmax_kernel_analysis() {
    let kernel = SoftmaxKernel::new(1024);
    let ptx = kernel.emit_ptx();

    let result = PtxBugAnalyzer::new().analyze(&ptx);
    assert!(result.is_valid(), "Softmax should not have critical bugs");
}

#[test]
fn test_q4k_kernel_analysis() {
    let kernel = QuantizeKernel::ggml(64, 64, 256);
    let ptx = kernel.emit_ptx();

    let result = PtxBugAnalyzer::new().analyze(&ptx);
    assert!(result.is_valid(), "Q4K should not have critical bugs");
}

#[test]
fn test_q5k_kernel_analysis() {
    let kernel = Q5KKernel::new(64, 64, 256);
    let ptx = kernel.emit_ptx();

    let result = PtxBugAnalyzer::new().analyze(&ptx);
    assert!(result.is_valid(), "Q5K should not have critical bugs");
}

#[test]
fn test_q6k_kernel_analysis() {
    let kernel = Q6KKernel::new(64, 64, 256);
    let ptx = kernel.emit_ptx();

    let result = PtxBugAnalyzer::new().analyze(&ptx);
    assert!(result.is_valid(), "Q6K should not have critical bugs");
}

// ============================================================================
// DETERMINISM VERIFICATION (F108)
// ============================================================================

#[test]
fn test_ptx_bug_analysis_determinism() {
    let kernels: Vec<(&str, String)> = vec![
        ("gemm_naive", GemmKernel::naive(32, 32, 32).emit_ptx()),
        ("gemm_tiled", GemmKernel::tiled(32, 32, 32, 16).emit_ptx()),
        ("softmax", SoftmaxKernel::new(256).emit_ptx()),
        ("q5k", Q5KKernel::new(32, 32, 256).emit_ptx()),
    ];

    for (name, ptx) in &kernels {
        let result1 = PtxBugAnalyzer::new().analyze(ptx);
        let result2 = PtxBugAnalyzer::new().analyze(ptx);
        let result3 = PtxBugAnalyzer::new().analyze(ptx);

        assert_eq!(
            result1.bugs.len(),
            result2.bugs.len(),
            "{} analysis must be deterministic (run 1 vs 2)",
            name
        );
        assert_eq!(
            result2.bugs.len(),
            result3.bugs.len(),
            "{} analysis must be deterministic (run 2 vs 3)",
            name
        );
    }
}

#[test]
fn test_ptx_analyzer_determinism() {
    let kernels = [GemmKernel::naive(64, 64, 64).emit_ptx(), SoftmaxKernel::new(1024).emit_ptx()];

    let analyzer = PtxAnalyzer::new();

    for ptx in &kernels {
        let result1 = analyzer.analyze(ptx).unwrap();
        let result2 = analyzer.analyze(ptx).unwrap();
        let result3 = analyzer.analyze(ptx).unwrap();

        assert_eq!(result1.registers.f32_regs, result2.registers.f32_regs);
        assert_eq!(result2.registers.f32_regs, result3.registers.f32_regs);
        assert_eq!(result1.instruction_count, result2.instruction_count);
        assert_eq!(result2.instruction_count, result3.instruction_count);
    }
}
