use super::*;

#[test]
fn test_backend_selector_default_thresholds() {
    // Falsifiable claim A-005, A-006
    let selector = BackendSelector::default();

    assert_eq!(selector.gpu_threshold(), 100_000);
    assert_eq!(selector.parallel_threshold(), 1_000);
}

#[test]
fn test_backend_selector_simd_only() {
    // N < 1,000 should use SIMD only
    let selector = BackendSelector::default();

    assert_eq!(selector.select_for_size(100, false), BackendCategory::SimdOnly);
    assert_eq!(selector.select_for_size(999, false), BackendCategory::SimdOnly);
    assert_eq!(selector.select_for_size(999, true), BackendCategory::SimdOnly);
}

#[test]
fn test_backend_selector_simd_parallel() {
    // 1,000 <= N < 100,000 should use SIMD + Parallel
    let selector = BackendSelector::default();

    assert_eq!(selector.select_for_size(1_000, false), BackendCategory::SimdParallel);
    assert_eq!(selector.select_for_size(50_000, false), BackendCategory::SimdParallel);
    assert_eq!(selector.select_for_size(99_999, false), BackendCategory::SimdParallel);
}

#[test]
fn test_backend_selector_gpu() {
    // N >= 100,000 should use GPU (if available)
    let selector = BackendSelector::default();

    assert_eq!(selector.select_for_size(100_000, true), BackendCategory::Gpu);
    assert_eq!(selector.select_for_size(1_000_000, true), BackendCategory::Gpu);
}

#[test]
fn test_backend_selector_gpu_fallback() {
    // N >= 100,000 without GPU should fallback to SIMD + Parallel
    let selector = BackendSelector::default();

    assert_eq!(selector.select_for_size(100_000, false), BackendCategory::SimdParallel);
}

#[test]
fn test_backend_selector_boundary() {
    // Falsifiable claim A-005
    let selector = BackendSelector::default();

    // At GPU threshold boundary
    assert!(selector.is_at_gpu_boundary(100_000));
    assert!(selector.is_at_gpu_boundary(99_999));
    assert!(!selector.is_at_gpu_boundary(99_998));

    // At parallel threshold boundary
    assert!(selector.is_at_parallel_boundary(1_000));
    assert!(selector.is_at_parallel_boundary(999));
    assert!(!selector.is_at_parallel_boundary(998));
}

#[test]
fn test_backend_tolerance_default() {
    let tolerance = BackendTolerance::default();

    assert_eq!(tolerance.scalar_vs_simd, 0.0);
    assert!((tolerance.simd_vs_gpu - 1e-5).abs() < 1e-10);
    assert!((tolerance.gpu_vs_gpu - 1e-6).abs() < 1e-10);
}

#[test]
fn test_backend_tolerance_strict() {
    let tolerance = BackendTolerance::strict();

    assert_eq!(tolerance.scalar_vs_simd, 0.0);
    assert_eq!(tolerance.simd_vs_gpu, 0.0);
    assert_eq!(tolerance.gpu_vs_gpu, 0.0);
}

#[test]
fn test_backend_tolerance_for_backends() {
    // Falsifiable claim A-002, A-003, A-004
    let tolerance = BackendTolerance::default();

    // Scalar vs Scalar
    assert_eq!(tolerance.for_backends(Backend::Scalar, Backend::Scalar), 0.0);

    // Scalar vs SIMD (should be exact)
    assert_eq!(tolerance.for_backends(Backend::Scalar, Backend::AVX2), 0.0);

    // GPU vs GPU
    assert_eq!(tolerance.for_backends(Backend::GPU, Backend::GPU), tolerance.gpu_vs_gpu);

    // SIMD vs GPU
    assert_eq!(tolerance.for_backends(Backend::AVX2, Backend::GPU), tolerance.simd_vs_gpu);
}

#[test]
fn test_heijunka_schedule_creation() {
    let backends = vec![Backend::Scalar, Backend::AVX2];
    let sizes = vec![100, 1000];
    let cycles = 2;

    let scheduler = HeijunkaScheduler::new(backends.clone(), sizes, cycles, 42);

    // Should have backends.len() * sizes.len() * cycles tests
    // 2 backends * 2 sizes * 2 cycles = 8 tests
    assert_eq!(scheduler.remaining(), 8);
}

#[test]
fn test_heijunka_deterministic_seeds() {
    // Falsifiable claim B-017
    let backends = vec![Backend::Scalar, Backend::AVX2];
    let sizes = vec![100, 1000];
    let cycles = 2;

    let mut scheduler1 = HeijunkaScheduler::new(backends.clone(), sizes.clone(), cycles, 42);
    let mut scheduler2 = HeijunkaScheduler::new(backends, sizes, cycles, 42);

    // Seeds should be identical for same configuration
    while let (Some(t1), Some(t2)) = (scheduler1.next_test(), scheduler2.next_test()) {
        assert_eq!(t1.seed, t2.seed, "Seeds must be deterministic");
    }
}

#[test]
fn test_heijunka_consumes_all_tests() {
    let backends = vec![Backend::Scalar];
    let sizes = vec![100];
    let cycles = 5;

    let mut scheduler = HeijunkaScheduler::new(backends, sizes, cycles, 42);

    let mut count = 0;
    while scheduler.next_test().is_some() {
        count += 1;
    }

    assert_eq!(count, 5);
    assert!(scheduler.is_empty());
}

#[test]
fn test_heijunka_different_master_seeds() {
    // Falsifiable claim B-018
    let backends = vec![Backend::Scalar];
    let sizes = vec![100];
    let cycles = 1;

    let mut scheduler1 = HeijunkaScheduler::new(backends.clone(), sizes.clone(), cycles, 42);
    let mut scheduler2 = HeijunkaScheduler::new(backends, sizes, cycles, 43);

    let t1 = scheduler1.next_test().unwrap();
    let t2 = scheduler2.next_test().unwrap();

    assert_ne!(t1.seed, t2.seed, "Different master seeds must produce different test seeds");
}

#[test]
fn test_sim_test_config_builder() {
    let config = SimTestConfig::builder()
        .seed(42)
        .tolerance(BackendTolerance::strict())
        .backends(vec![Backend::Scalar, Backend::AVX2, Backend::GPU])
        .input_sizes(vec![100, 1000, 10000])
        .cycles(5)
        .build();

    assert_eq!(config.seed, 42);
    assert_eq!(config.backends.len(), 3);
    assert_eq!(config.input_sizes.len(), 3);
    assert_eq!(config.cycles, 5);
}

#[test]
fn test_sim_test_config_creates_scheduler() {
    let config = SimTestConfig::builder()
        .seed(42)
        .backends(vec![Backend::Scalar, Backend::AVX2])
        .input_sizes(vec![100, 1000])
        .cycles(3)
        .build();

    let scheduler = config.create_scheduler();

    // 2 backends * 2 sizes * 3 cycles = 12 tests
    assert_eq!(scheduler.remaining(), 12);
}

#[test]
fn test_backend_category_debug() {
    let category = BackendCategory::Gpu;
    let debug = format!("{category:?}");
    assert!(debug.contains("Gpu"));
}
