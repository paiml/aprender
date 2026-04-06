#![allow(clippy::disallowed_methods, clippy::float_cmp)]
//! Roofline Model Falsification Tests (F1061-F1070)
//!
//! Popperian falsification criteria for Williams Roofline Model per §35.3.

use cbtop::roofline::profiles;
use cbtop::{
    BatchRooflineAnalysis, BottleneckType, HardwareProfile, RooflineAnalysis, RooflinePlot,
    RooflinePlotPoint, WorkloadMetrics,
};

// ============================================================================
// F1061: Ridge point calculation
// ============================================================================

#[test]
fn f1061_ridge_point_formula() {
    // Ridge = peak_gflops / peak_bandwidth_gbps
    let profile = HardwareProfile::new("Test", 1000.0, 100.0);
    assert!((profile.ridge_point() - 10.0).abs() < 0.01);
}

#[test]
fn f1061_ridge_point_zero_bandwidth() {
    let profile = HardwareProfile::new("Test", 1000.0, 0.0);
    assert!((profile.ridge_point() - 0.0).abs() < 0.01);
}

#[test]
fn f1061_a100_ridge_point() {
    let a100 = profiles::a100_sxm();
    // A100: 19500 GFLOPS / 2039 GB/s ≈ 9.56 FLOP/Byte
    assert!((a100.ridge_point() - 9.56).abs() < 0.1);
}

#[test]
fn f1061_h100_ridge_point() {
    let h100 = profiles::h100_sxm();
    // H100: 51200 GFLOPS / 3350 GB/s ≈ 15.28 FLOP/Byte
    assert!((h100.ridge_point() - 15.28).abs() < 0.1);
}

// ============================================================================
// F1062: Bottleneck classification
// ============================================================================

#[test]
fn f1062_memory_bound_classification() {
    let profile = HardwareProfile::new("Test", 1000.0, 100.0);
    // OI = 5 < Ridge = 10 → memory-bound
    assert_eq!(profile.classify_bottleneck(5.0), BottleneckType::MemoryBound);
}

#[test]
fn f1062_compute_bound_classification() {
    let profile = HardwareProfile::new("Test", 1000.0, 100.0);
    // OI = 20 > Ridge = 10 → compute-bound
    assert_eq!(profile.classify_bottleneck(20.0), BottleneckType::ComputeBound);
}

#[test]
fn f1062_balanced_classification() {
    let profile = HardwareProfile::new("Test", 1000.0, 100.0);
    // OI = 10 ≈ Ridge = 10 → balanced (within 10%)
    assert_eq!(profile.classify_bottleneck(10.0), BottleneckType::Balanced);
}

#[test]
fn f1062_balanced_boundary_low() {
    let profile = HardwareProfile::new("Test", 1000.0, 100.0);
    // OI = 9.0 (0.9 * ridge) → balanced
    assert_eq!(profile.classify_bottleneck(9.0), BottleneckType::Balanced);
}

#[test]
fn f1062_balanced_boundary_high() {
    let profile = HardwareProfile::new("Test", 1000.0, 100.0);
    // OI = 11.0 (1.1 * ridge) → balanced
    assert_eq!(profile.classify_bottleneck(11.0), BottleneckType::Balanced);
}

// ============================================================================
// F1063: Operational intensity calculation
// ============================================================================

#[test]
fn f1063_operational_intensity_formula() {
    let workload = WorkloadMetrics::new("test", 1000.0, 100.0, 1.0);
    // OI = FLOP / Bytes = 1000 / 100 = 10
    assert!((workload.operational_intensity() - 10.0).abs() < 0.01);
}

#[test]
fn f1063_operational_intensity_zero_bytes() {
    let workload = WorkloadMetrics::new("test", 1000.0, 0.0, 1.0);
    assert!((workload.operational_intensity() - 0.0).abs() < 0.01);
}

#[test]
fn f1063_measured_gflops_calculation() {
    // 1e9 FLOP / 0.01s = 100 GFLOPS
    let workload = WorkloadMetrics::new("test", 1e9, 1e8, 0.01);
    assert!((workload.measured_gflops - 100.0).abs() < 0.1);
}

// ============================================================================
// F1064: Theoretical peak calculation
// ============================================================================

#[test]
fn f1064_theoretical_peak_memory_bound() {
    let profile = HardwareProfile::new("Test", 1000.0, 100.0);
    // OI = 5 → peak = bandwidth * OI = 100 * 5 = 500 GFLOPS
    assert!((profile.theoretical_peak_at_oi(5.0) - 500.0).abs() < 0.1);
}

#[test]
fn f1064_theoretical_peak_compute_bound() {
    let profile = HardwareProfile::new("Test", 1000.0, 100.0);
    // OI = 20 → peak = min(1000, 100 * 20) = 1000 GFLOPS
    assert!((profile.theoretical_peak_at_oi(20.0) - 1000.0).abs() < 0.1);
}

#[test]
fn f1064_theoretical_peak_at_ridge() {
    let profile = HardwareProfile::new("Test", 1000.0, 100.0);
    // OI = 10 (ridge) → peak = 1000 GFLOPS
    assert!((profile.theoretical_peak_at_oi(10.0) - 1000.0).abs() < 0.1);
}

// ============================================================================
// F1065: Roofline analysis integration
// ============================================================================

#[test]
fn f1065_analysis_attained_efficiency() {
    let hardware = HardwareProfile::new("Test", 1000.0, 100.0);
    // OI = 10, measured = 100 GFLOPS, theoretical = 1000 → 10% efficiency
    let workload = WorkloadMetrics::new("matmul", 1e9, 1e8, 0.01);
    let analysis = RooflineAnalysis::analyze(&hardware, &workload);

    assert!((analysis.attained_efficiency - 10.0).abs() < 0.1);
}

#[test]
fn f1065_analysis_bottleneck_type() {
    let hardware = HardwareProfile::new("Test", 1000.0, 100.0);
    let workload = WorkloadMetrics::new("matmul", 1e9, 1e8, 0.01);
    let analysis = RooflineAnalysis::analyze(&hardware, &workload);

    assert_eq!(analysis.bottleneck, BottleneckType::Balanced);
}

#[test]
fn f1065_analysis_has_recommendation() {
    let hardware = HardwareProfile::new("Test", 1000.0, 100.0);
    let workload = WorkloadMetrics::new("matmul", 1e9, 1e8, 0.01);
    let analysis = RooflineAnalysis::analyze(&hardware, &workload);

    let rec = analysis.recommendation();
    assert!(!rec.is_empty());
    assert!(rec.contains("OI="));
}

// ============================================================================
// F1066: Hardware profiles
// ============================================================================

#[test]
fn f1066_all_profiles_available() {
    let all = profiles::all();
    assert!(all.len() >= 7);
}

#[test]
fn f1066_rtx_4090_profile() {
    let rtx4090 = profiles::rtx_4090();
    assert!((rtx4090.peak_gflops - 82580.0).abs() < 100.0);
    assert!((rtx4090.peak_bandwidth_gbps - 1008.0).abs() < 10.0);
}

#[test]
fn f1066_mi250x_profile() {
    let mi250x = profiles::mi250x();
    assert!((mi250x.peak_gflops - 47872.0).abs() < 100.0);
}

// ============================================================================
// F1067: Bottleneck recommendations
// ============================================================================

#[test]
fn f1067_memory_bound_recommendation() {
    let rec = BottleneckType::MemoryBound.recommendation();
    assert!(rec.contains("memory"));
}

#[test]
fn f1067_compute_bound_recommendation() {
    let rec = BottleneckType::ComputeBound.recommendation();
    assert!(rec.contains("compute"));
}

#[test]
fn f1067_balanced_recommendation() {
    let rec = BottleneckType::Balanced.recommendation();
    assert!(rec.contains("both") || rec.contains("Both"));
}

#[test]
fn f1067_bottleneck_names() {
    assert_eq!(BottleneckType::MemoryBound.name(), "memory-bound");
    assert_eq!(BottleneckType::ComputeBound.name(), "compute-bound");
    assert_eq!(BottleneckType::Balanced.name(), "balanced");
}

// ============================================================================
// F1068: Plot point generation
// ============================================================================

#[test]
fn f1068_plot_point_log_scale() {
    let point = RooflinePlotPoint::new("test", 8.0, 64.0);
    // log2(8) = 3, log2(64) = 6
    assert!((point.log_oi - 3.0).abs() < 0.01);
    assert!((point.log_perf - 6.0).abs() < 0.01);
}

#[test]
fn f1068_plot_has_memory_bound_line() {
    let hardware = profiles::a100_sxm();
    let plot = RooflinePlot::generate(&hardware, &[]);

    assert!(!plot.memory_bound_line.is_empty());
}

#[test]
fn f1068_plot_has_compute_bound_line() {
    let hardware = profiles::a100_sxm();
    let plot = RooflinePlot::generate(&hardware, &[]);

    assert!(!plot.compute_bound_line.is_empty());
}

#[test]
fn f1068_plot_has_ridge_point() {
    let hardware = profiles::a100_sxm();
    let plot = RooflinePlot::generate(&hardware, &[]);

    assert!((plot.ridge_point.oi - hardware.ridge_point()).abs() < 0.1);
}

// ============================================================================
// F1069: Batch analysis
// ============================================================================

#[test]
fn f1069_batch_analysis_multiple() {
    let hardware = HardwareProfile::new("Test", 1000.0, 100.0);
    let workloads = vec![
        WorkloadMetrics::new("w1", 1e9, 2e8, 0.01), // OI=5 memory-bound
        WorkloadMetrics::new("w2", 1e9, 5e7, 0.01), // OI=20 compute-bound
        WorkloadMetrics::new("w3", 1e9, 1e8, 0.01), // OI=10 balanced
    ];

    let batch = BatchRooflineAnalysis::analyze(&hardware, &workloads);
    assert_eq!(batch.analyses.len(), 3);
}

#[test]
fn f1069_batch_summary() {
    let hardware = HardwareProfile::new("Test", 1000.0, 100.0);
    let workloads = vec![
        WorkloadMetrics::new("w1", 1e9, 2e8, 0.01), // OI=5 memory-bound
        WorkloadMetrics::new("w2", 1e9, 5e7, 0.01), // OI=20 compute-bound
        WorkloadMetrics::new("w3", 1e9, 1e8, 0.01), // OI=10 balanced
    ];

    let batch = BatchRooflineAnalysis::analyze(&hardware, &workloads);
    let summary = batch.summary();

    assert_eq!(summary.total, 3);
    assert_eq!(summary.memory_bound, 1);
    assert_eq!(summary.compute_bound, 1);
    assert_eq!(summary.balanced, 1);
}

#[test]
fn f1069_empty_batch() {
    let hardware = HardwareProfile::new("Test", 1000.0, 100.0);
    let batch = BatchRooflineAnalysis::analyze(&hardware, &[]);
    let summary = batch.summary();

    assert_eq!(summary.total, 0);
    assert!((summary.avg_efficiency - 0.0).abs() < 0.01);
}

// ============================================================================
// F1070: Integration with workload metrics
// ============================================================================

#[test]
fn f1070_workload_name_preserved() {
    let workload = WorkloadMetrics::new("my_kernel", 1e9, 1e8, 0.01);
    assert_eq!(workload.name, "my_kernel");
}

#[test]
fn f1070_execution_time_stored() {
    let workload = WorkloadMetrics::new("test", 1e9, 1e8, 0.5);
    assert!((workload.execution_time_s - 0.5).abs() < 0.01);
}

#[test]
fn f1070_zero_execution_time() {
    let workload = WorkloadMetrics::new("test", 1e9, 1e8, 0.0);
    assert!((workload.measured_gflops - 0.0).abs() < 0.01);
}
