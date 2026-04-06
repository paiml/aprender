use super::*;

#[test]
fn test_backend_names() {
    assert_eq!(Backend::Scalar.name(), "Scalar");
    assert_eq!(Backend::Avx2.name(), "AVX2");
    assert_eq!(Backend::Cuda.name(), "CUDA");
}

#[test]
fn test_backend_is_gpu() {
    assert!(!Backend::Scalar.is_gpu());
    assert!(!Backend::Avx2.is_gpu());
    assert!(Backend::Cuda.is_gpu());
    assert!(Backend::Metal.is_gpu());
}

#[test]
fn test_backend_is_simd() {
    assert!(!Backend::Scalar.is_simd());
    assert!(Backend::Sse2.is_simd());
    assert!(Backend::Avx2.is_simd());
    assert!(!Backend::Cuda.is_simd());
}

#[test]
fn test_measurement_creation() {
    let m = BackendMeasurement::new(Backend::Avx2, WorkloadType::Gemm, 1024, 100.0, 10000.0)
        .with_efficiency(85.0);

    assert_eq!(m.backend, Backend::Avx2);
    assert_eq!(m.size, 1024);
    assert_eq!(m.efficiency_percent, 85.0);
}

#[test]
fn test_detector_add_measurement() {
    let mut detector = BackendRegressionDetector::new();

    detector.add(
        Backend::Scalar,
        WorkloadType::Gemm,
        1024,
        1000.0,
        1000.0,
        50.0,
    );
    detector.add(Backend::Avx2, WorkloadType::Gemm, 1024, 250.0, 4000.0, 80.0);

    assert_eq!(detector.measurement_count(), 2);
}

#[test]
fn test_compare_backends() {
    let mut detector = BackendRegressionDetector::new();

    detector.add(
        Backend::Scalar,
        WorkloadType::Gemm,
        1024,
        1000.0,
        1000.0,
        50.0,
    );
    detector.add(Backend::Avx2, WorkloadType::Gemm, 1024, 250.0, 4000.0, 80.0);

    let cmp = detector
        .compare_backends(Backend::Scalar, Backend::Avx2, WorkloadType::Gemm, 1024)
        .unwrap();

    assert!(cmp.speedup > 3.0);
    assert!(!cmp.is_regression);
}

#[test]
fn test_detect_cliff() {
    let mut detector = BackendRegressionDetector::new().with_cliff_threshold(10.0);

    // Normal efficiency at small sizes
    detector.add(
        Backend::Avx2,
        WorkloadType::Gemm,
        1024,
        100.0,
        10000.0,
        90.0,
    );
    detector.add(
        Backend::Avx2,
        WorkloadType::Gemm,
        2048,
        200.0,
        10000.0,
        88.0,
    );
    // Cliff: efficiency drops significantly
    detector.add(Backend::Avx2, WorkloadType::Gemm, 4096, 500.0, 8000.0, 60.0);

    let cliffs = detector.detect_size_cliffs(Backend::Avx2, WorkloadType::Gemm);

    assert!(!cliffs.is_empty());
    assert!(cliffs[0].drop_percent > 10.0);
}

#[test]
fn test_recommend_backend() {
    let mut detector = BackendRegressionDetector::new();

    detector.add(
        Backend::Scalar,
        WorkloadType::Gemm,
        1024,
        1000.0,
        1000.0,
        50.0,
    );
    detector.add(Backend::Avx2, WorkloadType::Gemm, 1024, 250.0, 4000.0, 80.0);
    detector.add(
        Backend::Cuda,
        WorkloadType::Gemm,
        1024,
        100.0,
        10000.0,
        95.0,
    );

    let rec = detector
        .recommend_backend(WorkloadType::Gemm, 1024)
        .unwrap();

    assert_eq!(rec.backend, Backend::Cuda);
}

#[test]
fn test_transfer_overhead() {
    let m = BackendMeasurement::new(Backend::Cuda, WorkloadType::Gemm, 1024, 100.0, 10000.0)
        .with_gpu_timing(30.0, 70.0);

    let overhead = m.transfer_overhead().unwrap();
    assert!((overhead - 0.3).abs() < 0.01);
}

#[test]
fn test_summary() {
    let mut detector = BackendRegressionDetector::new();

    detector.add(
        Backend::Scalar,
        WorkloadType::Gemm,
        1024,
        1000.0,
        1000.0,
        50.0,
    );
    detector.add(Backend::Avx2, WorkloadType::Gemm, 1024, 250.0, 4000.0, 80.0);

    let summary = detector.summary();

    assert_eq!(summary.measurement_count, 2);
    assert_eq!(summary.backend_count, 2);
}
