//! Coverage tracking (F107) tests

use super::*;

/// F107: Coverage tracking reports >=90% for real kernels
#[test]
fn f107_ptx_comprehensive_coverage() {
    let mut coverage = PtxCoverageTrackerBuilder::new()
        .feature("barrier_sync")
        .feature("shared_memory")
        .feature("global_memory")
        .feature("register_allocation")
        .feature("loop_patterns")
        .feature("control_flow")
        .feature("entry_point")
        .feature("predicates")
        .feature("fma_ops")
        .feature("local_memory")
        .build();

    // Run all PTX test cases from trueno-gpu kernels
    let kernels: Vec<String> = vec![
        GemmKernel::naive(64, 64, 64).emit_ptx(),
        GemmKernel::tiled(64, 64, 64, 16).emit_ptx(),
        SoftmaxKernel::new(1024).emit_ptx(),
        QuantizeKernel::ggml(64, 64, 256).emit_ptx(),
        Q5KKernel::new(64, 64, 256).emit_ptx(),
        Q6KKernel::new(64, 64, 256).emit_ptx(),
    ];

    for ptx in &kernels {
        coverage.analyze(ptx);
    }

    let report = coverage.generate_report();

    println!("\n╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║                       PTX FEATURE COVERAGE REPORT                             ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════╝\n");
    println!("Total Features: {}", report.total_features);
    println!("Covered Features: {}", report.covered_features);
    println!("Coverage: {:.1}%\n", report.coverage * 100.0);
    println!("Feature Details:");
    for feature in &report.features {
        let status = if feature.covered { "V" } else { "X" };
        println!("  {} {}: {} hits", status, feature.name, feature.hit_count);
    }

    assert!(
        report.coverage >= 0.90,
        "F107: PTX coverage must be >=90%, got {:.1}%",
        report.coverage * 100.0
    );
}

/// Test coverage tracker with minimal features
#[test]
fn test_coverage_tracker_basic() {
    let mut coverage = PtxCoverageTrackerBuilder::new()
        .feature("register_allocation")
        .feature("entry_point")
        .build();

    let ptx = r"
.visible .entry test() {
    .reg .f32 %f<4>;
    ret;
}
";

    coverage.analyze(ptx);
    let report = coverage.generate_report();

    assert_eq!(report.total_features, 2);
    assert!(report.coverage >= 0.5, "Should cover at least entry_point and register_allocation");
}

/// Test default coverage tracker
#[test]
fn test_coverage_tracker_default() {
    let mut coverage = PtxCoverageTracker::default();

    let ptx = GemmKernel::naive(32, 32, 32).emit_ptx();
    coverage.analyze(&ptx);

    let report = coverage.generate_report();
    assert!(report.total_features >= 6, "Default tracker should have 6+ features");
}
