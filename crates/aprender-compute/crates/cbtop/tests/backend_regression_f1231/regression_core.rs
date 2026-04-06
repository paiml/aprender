//! F1231-F1235: Core backend regression tests

use cbtop::{Backend, BackendMeasurement, BackendRegressionDetector, BackendWorkload};

// =============================================================================
// F1231: Backend Comparison Tests
// =============================================================================

/// F1231.1: Backend comparison works for all backends
#[test]
fn f1231_backend_comparison() {
    let mut detector = BackendRegressionDetector::new();

    // Add measurements for different backends
    detector.add(Backend::Scalar, BackendWorkload::Gemm, 1024, 1000.0, 1000.0, 50.0);
    detector.add(Backend::Sse2, BackendWorkload::Gemm, 1024, 500.0, 2000.0, 60.0);
    detector.add(Backend::Avx2, BackendWorkload::Gemm, 1024, 250.0, 4000.0, 80.0);

    let cmp = detector
        .compare_backends(Backend::Scalar, Backend::Avx2, BackendWorkload::Gemm, 1024)
        .unwrap();

    assert_eq!(cmp.baseline, Backend::Scalar);
    assert_eq!(cmp.comparison, Backend::Avx2);
    assert!(cmp.speedup > 1.0);
}

/// F1231.2: Missing backend returns None
#[test]
fn f1231_missing_backend() {
    let mut detector = BackendRegressionDetector::new();

    detector.add(Backend::Scalar, BackendWorkload::Gemm, 1024, 1000.0, 1000.0, 50.0);

    let result = detector.compare_backends(
        Backend::Scalar,
        Backend::Cuda, // Not added
        BackendWorkload::Gemm,
        1024,
    );

    assert!(result.is_none());
}

// =============================================================================
// F1232: Efficiency Ratio Tests
// =============================================================================

/// F1232.1: Efficiency ratio calculated correctly
#[test]
fn f1232_efficiency_ratio() {
    let mut detector = BackendRegressionDetector::new();

    detector.add(Backend::Scalar, BackendWorkload::Gemm, 1024, 1000.0, 1000.0, 50.0);
    detector.add(Backend::Avx2, BackendWorkload::Gemm, 1024, 250.0, 4000.0, 80.0);

    let cmp = detector
        .compare_backends(Backend::Scalar, Backend::Avx2, BackendWorkload::Gemm, 1024)
        .unwrap();

    // Efficiency ratio = 80 / 50 = 1.6
    assert!((cmp.efficiency_ratio - 1.6).abs() < 0.1);
}

/// F1232.2: Zero baseline efficiency handled
#[test]
fn f1232_zero_baseline() {
    let mut detector = BackendRegressionDetector::new();

    detector.add(Backend::Scalar, BackendWorkload::Gemm, 1024, 1000.0, 1000.0, 0.0);
    detector.add(Backend::Avx2, BackendWorkload::Gemm, 1024, 250.0, 4000.0, 80.0);

    let cmp = detector
        .compare_backends(Backend::Scalar, Backend::Avx2, BackendWorkload::Gemm, 1024)
        .unwrap();

    // Should not divide by zero
    assert_eq!(cmp.efficiency_ratio, 0.0);
}

// =============================================================================
// F1233: Size Cliff Detection Tests
// =============================================================================

/// F1233.1: Size cliff detected when >10% drop
#[test]
fn f1233_cliff_detection() {
    let mut detector = BackendRegressionDetector::new().with_cliff_threshold(10.0);

    // Efficiency drops from 90% to 60% at 4M elements
    detector.add(Backend::Avx2, BackendWorkload::Gemm, 1_000_000, 100.0, 10000.0, 90.0);
    detector.add(Backend::Avx2, BackendWorkload::Gemm, 2_000_000, 200.0, 10000.0, 88.0);
    detector.add(Backend::Avx2, BackendWorkload::Gemm, 4_000_000, 500.0, 8000.0, 60.0);

    let cliffs = detector.detect_size_cliffs(Backend::Avx2, BackendWorkload::Gemm);

    assert!(!cliffs.is_empty());
    let cliff = &cliffs[0];
    assert!(cliff.drop_percent > 10.0);
    assert_eq!(cliff.size_after, 4_000_000);
}

/// F1233.2: No cliff for gradual changes
#[test]
fn f1233_no_cliff_gradual() {
    let mut detector = BackendRegressionDetector::new().with_cliff_threshold(10.0);

    // Gradual efficiency changes
    detector.add(Backend::Avx2, BackendWorkload::Gemm, 1024, 100.0, 10000.0, 90.0);
    detector.add(Backend::Avx2, BackendWorkload::Gemm, 2048, 200.0, 10000.0, 88.0);
    detector.add(Backend::Avx2, BackendWorkload::Gemm, 4096, 400.0, 10000.0, 85.0);

    let cliffs = detector.detect_size_cliffs(Backend::Avx2, BackendWorkload::Gemm);

    assert!(cliffs.is_empty());
}

// =============================================================================
// F1234: GPU Transfer Overhead Tests
// =============================================================================

/// F1234.1: GPU transfer overhead measured
#[test]
fn f1234_transfer_overhead() {
    let mut detector = BackendRegressionDetector::new();

    let m = BackendMeasurement::new(Backend::Cuda, BackendWorkload::Gemm, 1024, 100.0, 10000.0)
        .with_gpu_timing(30.0, 70.0);

    detector.add_measurement(m);

    let analysis =
        detector.analyze_transfer_overhead(Backend::Cuda, BackendWorkload::Gemm).unwrap();

    assert!((analysis.average_overhead - 0.3).abs() < 0.01);
}

/// F1234.2: Non-GPU returns None
#[test]
fn f1234_non_gpu_no_analysis() {
    let mut detector = BackendRegressionDetector::new();

    detector.add(Backend::Avx2, BackendWorkload::Gemm, 1024, 100.0, 10000.0, 80.0);

    let analysis = detector.analyze_transfer_overhead(Backend::Avx2, BackendWorkload::Gemm);

    assert!(analysis.is_none());
}

// =============================================================================
// F1235: Best Backend Selection Tests
// =============================================================================

/// F1235.1: Best backend selected by throughput
#[test]
fn f1235_best_backend() {
    let mut detector = BackendRegressionDetector::new();

    detector.add(Backend::Scalar, BackendWorkload::Gemm, 1024, 1000.0, 1000.0, 50.0);
    detector.add(Backend::Avx2, BackendWorkload::Gemm, 1024, 250.0, 4000.0, 80.0);
    detector.add(Backend::Cuda, BackendWorkload::Gemm, 1024, 100.0, 10000.0, 95.0);

    let rec = detector.recommend_backend(BackendWorkload::Gemm, 1024).unwrap();

    assert_eq!(rec.backend, Backend::Cuda);
    assert!(rec.expected_efficiency >= 90.0);
}

/// F1235.2: Returns None for unknown workload/size
#[test]
fn f1235_unknown_returns_none() {
    let detector = BackendRegressionDetector::new();

    let rec = detector.recommend_backend(BackendWorkload::Gemm, 1024);

    assert!(rec.is_none());
}
