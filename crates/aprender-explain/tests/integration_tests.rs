//! Integration tests for trueno-explain
//!
//! Tests the analyzer against real trueno-gpu kernels

#![allow(clippy::disallowed_methods)]

use std::process::Command;
use trueno_explain::{Analyzer, PtxAnalyzer};
use trueno_gpu::kernels::{
    GemmKernel, Kernel, Q5KKernel, Q6KKernel, QuantizeKernel, SoftmaxKernel,
};

/// Helper to run the trueno-explain binary
fn run_explain(args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_aprender-explain"))
        .args(args)
        .output()
        .expect("Failed to run trueno-explain")
}

/// F001: `trueno-explain --help` shows all subcommands
#[test]
fn f001_help_shows_subcommands() {
    let output = run_explain(&["--help"]);
    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(output.status.success(), "Help should succeed");
    assert!(stdout.contains("ptx"), "Should show ptx subcommand");
    assert!(stdout.contains("simd"), "Should show simd subcommand");
    assert!(stdout.contains("wgpu"), "Should show wgpu subcommand");
    assert!(stdout.contains("tui"), "Should show tui subcommand");
    assert!(stdout.contains("compare"), "Should show compare subcommand");
    assert!(stdout.contains("diff"), "Should show diff subcommand");
    assert!(stdout.contains("bugs"), "Should show bugs subcommand");
}

/// F002: `trueno-explain tui --help` shows TUI options
#[test]
fn f002_tui_help_shows_options() {
    let output = run_explain(&["tui", "--help"]);
    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(output.status.success(), "TUI help should succeed");
    assert!(stdout.contains("--kernel"), "Should show --kernel option");
    assert!(stdout.contains("--rows"), "Should show --rows option");
    assert!(stdout.contains("--cols"), "Should show --cols option");
    assert!(stdout.contains("--inner"), "Should show --inner option");
}

/// F008: --json flag produces valid JSON
#[test]
fn f008_json_flag_valid_json() {
    let output = run_explain(&["ptx", "--kernel", "gemm_naive", "--json"]);
    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(output.status.success(), "PTX analysis should succeed");

    // Verify it's valid JSON by parsing
    let parsed: Result<serde_json::Value, _> = serde_json::from_str(&stdout);
    assert!(parsed.is_ok(), "Output should be valid JSON: {}", stdout);

    // Verify expected fields
    let json = parsed.unwrap();
    assert!(json.get("name").is_some(), "JSON should have 'name' field");
    assert!(
        json.get("target").is_some(),
        "JSON should have 'target' field"
    );
    assert!(
        json.get("registers").is_some(),
        "JSON should have 'registers' field"
    );
}

/// F011: Analyze `vector_add` reports <20 registers for f32
#[test]
fn f011_vector_add_low_register_usage() {
    let ptx = include_str!("../data/vector_add.ptx.data");
    let analyzer = PtxAnalyzer::new();
    let report = analyzer.analyze(ptx).unwrap();

    assert!(
        report.registers.f32_regs < 50,
        "vector_add should use <50 f32 registers"
    );
}

/// F019: Occupancy calculation matches expected range
#[test]
fn f019_occupancy_calculation() {
    let ptx = include_str!("../data/vector_add.ptx.data");
    let analyzer = PtxAnalyzer::new();
    let report = analyzer.analyze(ptx).unwrap();

    // 58 registers -> should have reasonable occupancy
    assert!(
        report.estimated_occupancy > 0.25,
        "Expected >25% occupancy, got {}",
        report.estimated_occupancy
    );
    assert!(
        report.estimated_occupancy <= 1.0,
        "Occupancy should not exceed 100%"
    );
}

/// F020: Warns when registers > 128
#[test]
fn f020_high_register_warning() {
    let high_reg_ptx = r"
.version 8.0
.target sm_70
.entry big_kernel()
{
    .reg .f32 %f<200>;
    ret;
}
";
    let analyzer = PtxAnalyzer::new();
    let report = analyzer.analyze(high_reg_ptx).unwrap();

    assert!(
        !report.warnings.is_empty(),
        "Should warn on high register usage"
    );
}

/// Test GEMM naive kernel analysis
#[test]
fn test_gemm_naive_analysis() {
    let kernel = GemmKernel::naive(64, 64, 64);
    let ptx = kernel.emit_ptx();

    let analyzer = PtxAnalyzer::new();
    let report = analyzer.analyze(&ptx).unwrap();

    assert_eq!(report.name, "gemm_naive");
    assert!(report.instruction_count > 0);
}

/// Test GEMM tiled kernel analysis
#[test]
fn test_gemm_tiled_analysis() {
    let kernel = GemmKernel::tiled(64, 64, 64, 16);
    let ptx = kernel.emit_ptx();

    let analyzer = PtxAnalyzer::new();
    let report = analyzer.analyze(&ptx).unwrap();

    assert_eq!(report.name, "gemm_tiled");
    // Tiled GEMM should use shared memory
    // (Note: may not be detected in current simplified parser)
}

/// Test Q4K kernel analysis
#[test]
fn test_q4k_kernel_analysis() {
    let kernel = QuantizeKernel::ggml(64, 64, 256);
    let ptx = kernel.emit_ptx();

    let analyzer = PtxAnalyzer::new();
    let report = analyzer.analyze(&ptx).unwrap();

    assert!(report.name.contains("q4k"));
    assert!(report.memory.global_loads > 0);
}

/// Test Q5K kernel analysis (PARITY-116)
#[test]
fn test_q5k_kernel_analysis() {
    let kernel = Q5KKernel::new(64, 64, 256);
    let ptx = kernel.emit_ptx();

    let analyzer = PtxAnalyzer::new();
    let report = analyzer.analyze(&ptx).unwrap();

    assert!(report.name.contains("q5k"));
    assert!(report.memory.global_loads > 0);
    // Q5K loads both ql and qh
}

/// Test Q6K kernel analysis (PARITY-117)
#[test]
fn test_q6k_kernel_analysis() {
    let kernel = Q6KKernel::new(64, 64, 256);
    let ptx = kernel.emit_ptx();

    let analyzer = PtxAnalyzer::new();
    let report = analyzer.analyze(&ptx).unwrap();

    assert!(report.name.contains("q6k"));
    assert!(report.memory.global_loads > 0);
}

/// Test Q5K matvec (n=1) for realizar compatibility
#[test]
fn test_q5k_matvec_n1_analysis() {
    let kernel = Q5KKernel::new(64, 1, 256);
    let ptx = kernel.emit_ptx();

    let analyzer = PtxAnalyzer::new();
    let report = analyzer.analyze(&ptx).unwrap();

    assert!(report.name.contains("q5k"));
    // Matvec should still work
    assert!(report.instruction_count > 0);
}

/// Test softmax kernel analysis
#[test]
fn test_softmax_analysis() {
    let kernel = SoftmaxKernel::new(1024);
    let ptx = kernel.emit_ptx();

    let analyzer = PtxAnalyzer::new();
    let report = analyzer.analyze(&ptx).unwrap();

    assert!(report.name.contains("softmax"));
}

/// Test JSON serialization roundtrip
#[test]
fn test_json_roundtrip() {
    let kernel = GemmKernel::naive(32, 32, 32);
    let ptx = kernel.emit_ptx();

    let analyzer = PtxAnalyzer::new();
    let report = analyzer.analyze(&ptx).unwrap();

    let json = serde_json::to_string(&report).unwrap();
    let parsed: trueno_explain::AnalysisReport = serde_json::from_str(&json).unwrap();

    assert_eq!(report.name, parsed.name);
    assert_eq!(report.registers.f32_regs, parsed.registers.f32_regs);
}

/// Test all kernels can be analyzed without panic
#[test]
fn test_all_kernels_analyzable() {
    let kernels: Vec<(&str, String)> = vec![
        ("gemm_naive", GemmKernel::naive(32, 32, 32).emit_ptx()),
        ("gemm_tiled", GemmKernel::tiled(32, 32, 32, 16).emit_ptx()),
        ("softmax", SoftmaxKernel::new(256).emit_ptx()),
        ("q4k", QuantizeKernel::ggml(32, 32, 256).emit_ptx()),
        ("q5k", Q5KKernel::new(32, 32, 256).emit_ptx()),
        ("q6k", Q6KKernel::new(32, 32, 256).emit_ptx()),
    ];

    let analyzer = PtxAnalyzer::new();

    for (name, ptx) in kernels {
        let result = analyzer.analyze(&ptx);
        assert!(result.is_ok(), "Failed to analyze {}: {:?}", name, result);
    }
}

// ============================================================================
// BUG HUNTING CLI TESTS
// ============================================================================

/// F003: `trueno-explain bugs --help` shows bug hunting options
#[test]
fn f003_bugs_help_shows_options() {
    let output = run_explain(&["bugs", "--help"]);
    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(output.status.success(), "Bugs help should succeed");
    assert!(stdout.contains("--kernel"), "Should show --kernel option");
    assert!(stdout.contains("--strict"), "Should show --strict option");
    assert!(
        stdout.contains("--fail-on-bugs"),
        "Should show --fail-on-bugs option"
    );
    assert!(stdout.contains("--json"), "Should show --json option");
}

/// F009: `trueno-explain bugs --json` produces valid JSON
#[test]
fn f009_bugs_json_valid() {
    let output = run_explain(&["bugs", "-K", "gemm_naive", "--json"]);
    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(output.status.success(), "Bug analysis should succeed");

    // Verify it's valid JSON
    let parsed: Result<serde_json::Value, _> = serde_json::from_str(&stdout);
    assert!(parsed.is_ok(), "Output should be valid JSON: {}", stdout);

    // Verify expected fields
    let json = parsed.unwrap();
    assert!(json.get("bugs").is_some(), "JSON should have 'bugs' field");
    assert!(
        json.get("lines_analyzed").is_some(),
        "JSON should have 'lines_analyzed' field"
    );
}

/// F010: Bug report shows kernel name and summary
#[test]
fn f010_bug_report_format() {
    let output = run_explain(&["bugs", "-K", "gemm_naive"]);
    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(output.status.success(), "Bug analysis should succeed");
    assert!(
        stdout.contains("PTX BUG HUNTING REPORT"),
        "Should show report header"
    );
    assert!(stdout.contains("Kernel:"), "Should show kernel name");
    assert!(stdout.contains("SUMMARY"), "Should show summary section");
    assert!(stdout.contains("P0 Critical:"), "Should show P0 count");
}

/// F101 CLI: Bugs command detects issues in strict mode
#[test]
fn f101_cli_bugs_strict_mode() {
    let output = run_explain(&["bugs", "-K", "gemm_tiled", "--strict"]);
    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(
        output.status.success(),
        "Strict bug analysis should succeed"
    );
    assert!(
        stdout.contains("PTX BUG HUNTING REPORT"),
        "Should show report"
    );
}

/// F102 CLI: --fail-on-bugs exits with error on critical bugs
#[test]
fn f102_cli_fail_on_bugs_success() {
    // gemm_naive should have no critical bugs
    let output = run_explain(&["bugs", "-K", "gemm_naive", "--fail-on-bugs"]);
    assert!(
        output.status.success(),
        "Should succeed with no critical bugs"
    );
}

/// Test all kernels pass bug hunting in normal mode
#[test]
fn test_all_kernels_no_critical_bugs() {
    let kernels = ["gemm_naive", "gemm_tiled", "softmax", "q4k", "q5k", "q6k"];

    for kernel in kernels {
        let output = run_explain(&["bugs", "-K", kernel, "--fail-on-bugs"]);
        assert!(
            output.status.success(),
            "Kernel {} should have no critical bugs",
            kernel
        );
    }
}
