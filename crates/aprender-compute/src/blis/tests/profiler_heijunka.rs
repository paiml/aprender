use super::super::*;

// ========================================================================
// Profiler Tests
// ========================================================================

#[test]
fn test_profiler_records_timing() {
    let mut profiler = BlisProfiler::enabled();

    let n = 128;
    let a: Vec<f32> = vec![1.0; n * n];
    let b: Vec<f32> = vec![1.0; n * n];
    let mut c = vec![0.0; n * n];

    gemm_blis(n, n, n, &a, &b, &mut c, Some(&mut profiler)).unwrap();

    assert!(profiler.macro_stats.count > 0);
    assert!(profiler.macro_stats.flops > 0);
    // NOTE: micro_stats may be 0 when AVX-512 path is taken (records macro-level only).
    // The AVX-512 8×16 microkernel (C-AVX512-PROF-001) records at macro granularity.
    // micro_stats are populated only by the generic BLIS 5-loop path.
}

#[test]
fn test_kaizen_metrics() {
    let mut metrics = KaizenMetrics::default();

    metrics.record(100, 100, 100, std::time::Duration::from_micros(100));

    assert_eq!(metrics.flops, 2_000_000); // 2 * 100^3
    assert!(metrics.gflops() > 0.0);
}

// ========================================================================
// Heijunka Tests
// ========================================================================

#[test]
fn test_heijunka_balanced_partition() {
    let scheduler = HeijunkaScheduler { num_threads: 4, variance_threshold: 0.05 };

    // Use m = 4*MC so it divides evenly into 4 blocks
    let m = 4 * MC;
    let partitions = scheduler.partition_m(m, MC);

    // Should have 4 partitions
    assert_eq!(partitions.len(), 4);

    // Each partition should be exactly equal (MC rows each)
    let sizes: Vec<usize> = partitions.iter().map(|r| r.len()).collect();
    let avg = sizes.iter().sum::<usize>() as f32 / sizes.len() as f32;

    for size in &sizes {
        let variance = ((*size as f32 - avg) / avg).abs();
        assert!(variance < 0.01, "Partition variance too high: {}", variance);
    }

    // Also test uneven case - should still work
    let m_uneven = 3 * MC + MC / 2;
    let partitions_uneven = scheduler.partition_m(m_uneven, MC);
    assert!(partitions_uneven.len() <= 4);
    let total: usize = partitions_uneven.iter().map(|r| r.len()).sum();
    assert_eq!(total, m_uneven); // All rows covered
}

#[test]
fn test_heijunka_single_thread() {
    let scheduler = HeijunkaScheduler { num_threads: 1, variance_threshold: 0.05 };
    let partitions = scheduler.partition_m(100, MC);
    assert_eq!(partitions.len(), 1);
    assert_eq!(partitions[0], 0..100);
}

#[test]
fn test_heijunka_more_threads_than_blocks() {
    let scheduler = HeijunkaScheduler { num_threads: 8, variance_threshold: 0.05 };
    // m=MC means 1 block but 8 threads — some threads get nothing
    let partitions = scheduler.partition_m(MC, MC);
    assert!(!partitions.is_empty());
    let total: usize = partitions.iter().map(|r| r.len()).sum();
    assert_eq!(total, MC);
}

#[test]
fn test_heijunka_default() {
    let scheduler = HeijunkaScheduler::default();
    assert!(scheduler.num_threads >= 1);
    assert!((scheduler.variance_threshold - 0.05).abs() < 1e-6);
}

#[test]
fn test_gemm_blis_parallel_small_matrix() {
    // Small matrix falls through to sequential gemm_blis
    let m = 10;
    let n = 10;
    let k = 10;
    let a: Vec<f32> = (0..m * k).map(|i| i as f32).collect();
    let b: Vec<f32> = vec![0.0; k * n];
    // Identity-like: b[i*n+i] = 1.0
    let mut b_mut = b;
    for i in 0..k.min(n) {
        b_mut[i * n + i] = 1.0;
    }
    let mut c = vec![0.0; m * n];

    gemm_blis_parallel(m, n, k, &a, &b_mut, &mut c).unwrap();
    // First row of C should match first row of A (for k<=n)
    for j in 0..k.min(n) {
        assert!((c[j] - a[j]).abs() < 1e-3, "c[{}] = {}, a[{}] = {}", j, c[j], j, a[j]);
    }
}

#[test]
fn test_gemm_blis_parallel_dimension_mismatch() {
    let mut c = vec![0.0; 4];
    // a.len() = 3 != m*k = 2*2 = 4
    let result = gemm_blis_parallel(2, 2, 2, &[1.0, 2.0, 3.0], &[1.0; 4], &mut c);
    assert!(result.is_err());
}
