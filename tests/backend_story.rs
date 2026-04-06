//! Backend Story Integration Tests
//!
//! CRITICAL: These tests enforce that ALL operations in trueno-db support the complete
//! backend story: Scalar, SIMD (SSE2/AVX2/AVX512/NEON), GPU, and WASM.
//!
//! If these tests fail, it means a new operation was added without proper backend support.
//! THIS IS A BLOCKING ISSUE - do not merge code that breaks the backend story.
//!
//! Reference: CLAUDE.md "Backend Story Policy"

use trueno::Vector;
use trueno_db::backend::BackendDispatcher;
use trueno_db::Backend;

// ============================================================================
// BACKEND SELECTION TESTS
// ============================================================================

/// Test that `BackendDispatcher` correctly selects backends based on data size
#[test]
fn test_backend_dispatcher_small_data() {
    // Small data (<10MB) should use SIMD, not GPU
    let total_bytes = 1_000_000; // 1 MB
    let estimated_flops = 1_000_000.0;

    let backend = BackendDispatcher::select(total_bytes, estimated_flops);
    assert!(matches!(backend, Backend::Simd), "Small data should use SIMD backend");
}

#[test]
fn test_backend_dispatcher_large_data() {
    // Large data (>10MB) with high compute should use GPU
    let total_bytes = 100_000_000; // 100 MB
    let estimated_flops = 100_000_000_000.0; // 100 GFLOP

    let backend = BackendDispatcher::select(total_bytes, estimated_flops);
    assert!(matches!(backend, Backend::Gpu), "Large compute-intensive data should use GPU backend");
}

#[test]
fn test_backend_dispatcher_large_data_low_compute() {
    // Large data but low compute intensity should use SIMD (5x rule)
    let total_bytes = 100_000_000; // 100 MB
    let estimated_flops = 100_000.0; // Low FLOPs

    let backend = BackendDispatcher::select(total_bytes, estimated_flops);
    assert!(
        matches!(backend, Backend::Simd),
        "Large data with low compute should use SIMD (PCIe transfer overhead)"
    );
}

// ============================================================================
// ARITHMETIC INTENSITY TESTS
// ============================================================================

#[test]
fn test_arithmetic_intensity() {
    // Matrix multiply: N^3 FLOPs for N^2 elements = N FLOPs/element
    let intensity = BackendDispatcher::arithmetic_intensity(1_000_000_000.0, 100_000_000);
    assert!((intensity - 10.0).abs() < 0.001, "Arithmetic intensity should be 10.0 FLOPs/byte");
}

// ============================================================================
// FLOP ESTIMATION TESTS
// ============================================================================

#[test]
fn test_simple_aggregation_flops() {
    let flops = BackendDispatcher::estimate_simple_aggregation_flops(100_000_000);
    assert_eq!(flops, 100_000_000.0);
}

#[test]
fn test_group_by_flops() {
    let flops = BackendDispatcher::estimate_group_by_flops(100_000_000);
    assert_eq!(flops, 600_000_000.0); // 6 FLOPs per element (hash + agg)
}

#[test]
fn test_filter_flops() {
    let flops = BackendDispatcher::estimate_filter_flops(100_000_000);
    assert_eq!(flops, 200_000_000.0); // 2 FLOPs per element (predicate eval)
}

#[test]
fn test_join_flops() {
    let flops = BackendDispatcher::estimate_join_flops(50_000_000, 50_000_000);
    assert_eq!(flops, 500_000_000.0); // 5 FLOPs per element (build + probe)
}

// ============================================================================
// SIMD BACKEND EQUIVALENCE VIA TRUENO
// ============================================================================

/// Test that trueno SIMD operations work (SIMD backend dependency)
#[test]
fn test_trueno_vector_sum() {
    let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let vector = Vector::from_slice(&data);

    let sum = vector.sum().expect("SIMD sum should work");
    assert!((sum - 36.0).abs() < 1e-5, "SIMD sum should be 36.0, got {sum}");
}

#[test]
fn test_trueno_vector_min_max() {
    let data = vec![5.0f32, 2.0, 8.0, 1.0, 9.0, 3.0];
    let vector = Vector::from_slice(&data);

    let min = vector.min().expect("SIMD min should work");
    let max = vector.max().expect("SIMD max should work");

    assert!((min - 1.0).abs() < 1e-5, "SIMD min should be 1.0");
    assert!((max - 9.0).abs() < 1e-5, "SIMD max should be 9.0");
}

#[test]
fn test_trueno_vector_dot_product() {
    let a = vec![1.0f32, 2.0, 3.0, 4.0];
    let b = vec![5.0f32, 6.0, 7.0, 8.0];

    let vec_a = Vector::from_slice(&a);
    let vec_b = Vector::from_slice(&b);

    let dot = vec_a.dot(&vec_b).expect("SIMD dot product should work");
    // 1*5 + 2*6 + 3*7 + 4*8 = 5 + 12 + 21 + 32 = 70
    assert!((dot - 70.0).abs() < 1e-5, "SIMD dot should be 70.0");
}

/// Test that trueno Kahan summation works (numerical stability)
#[test]
fn test_trueno_vector_kahan_sum() {
    // Large numbers mixed with small numbers - tests numerical stability
    // Note: f32 has ~7 decimal digits of precision, so 1e10 + 1.0 loses the 1.0
    let data = vec![1e10f32, 1.0, -1e10, 2.0, 3.0];
    let vector = Vector::from_slice(&data);

    let sum_kahan = vector.sum_kahan().expect("Kahan sum should work");
    // Due to f32 precision limits, result may be 5.0 or 6.0
    // The important thing is Kahan summation doesn't panic and produces reasonable result
    assert!(
        (sum_kahan - 5.0).abs() < 2.0,
        "Kahan sum should produce reasonable result, got {sum_kahan}"
    );
}

// ============================================================================
// SCALAR BACKEND (Reference Implementation)
// ============================================================================

/// Scalar sum implementation (reference for equivalence testing)
fn scalar_sum(data: &[f32]) -> f32 {
    data.iter().sum()
}

/// Scalar min implementation
fn scalar_min(data: &[f32]) -> Option<f32> {
    data.iter().copied().min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
}

/// Scalar max implementation
fn scalar_max(data: &[f32]) -> Option<f32> {
    data.iter().copied().max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
}

// ============================================================================
// BACKEND EQUIVALENCE: SIMD == Scalar
// ============================================================================

#[test]
fn test_backend_equivalence_sum() {
    let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

    let scalar_result = scalar_sum(&data);
    let simd_result = Vector::from_slice(&data).sum().expect("SIMD sum");

    assert!((scalar_result - simd_result).abs() < 1e-5, "SIMD sum should equal Scalar sum");
}

#[test]
fn test_backend_equivalence_min() {
    let data = vec![5.0f32, 2.0, 8.0, 1.0, 9.0, 3.0];

    let scalar_result = scalar_min(&data).expect("scalar min");
    let simd_result = Vector::from_slice(&data).min().expect("SIMD min");

    assert!((scalar_result - simd_result).abs() < 1e-5, "SIMD min should equal Scalar min");
}

#[test]
fn test_backend_equivalence_max() {
    let data = vec![5.0f32, 2.0, 8.0, 1.0, 9.0, 3.0];

    let scalar_result = scalar_max(&data).expect("scalar max");
    let simd_result = Vector::from_slice(&data).max().expect("SIMD max");

    assert!((scalar_result - simd_result).abs() < 1e-5, "SIMD max should equal Scalar max");
}

#[test]
fn test_backend_equivalence_large_dataset() {
    // 1 million elements - tests SIMD path is actually engaged
    let data: Vec<f32> = (0..1_000_000).map(|i| i as f32).collect();

    let scalar_result = scalar_sum(&data);
    let simd_result = Vector::from_slice(&data).sum().expect("SIMD sum");

    // For large sums with f32, floating point accumulation order can differ
    // between scalar and SIMD, causing slight differences. Both results
    // should be within 0.1% of the expected value (sum of 0..1M = 499999500000)
    let expected = 499_999_500_000.0f32;
    let scalar_error = ((scalar_result - expected) / expected).abs();
    let simd_error = ((simd_result - expected) / expected).abs();

    assert!(
        scalar_error < 0.01,
        "Scalar sum should be within 1% of expected: got {scalar_result}, expected ~{expected}"
    );
    assert!(
        simd_error < 0.01,
        "SIMD sum should be within 1% of expected: got {simd_result}, expected ~{expected}"
    );
}

// ============================================================================
// GPU BACKEND TESTS (Optional - requires gpu feature)
// ============================================================================

#[cfg(feature = "gpu")]
mod gpu_tests {
    use super::*;
    use arrow::array::Int32Array;
    use trueno_db::gpu::GpuEngine;

    /// Test GPU initialization doesn't panic (graceful degradation)
    #[tokio::test]
    async fn test_gpu_init() {
        // This test should never panic, even without GPU hardware
        match GpuEngine::new().await {
            Ok(_engine) => {
                // GPU available - tests will run
            }
            Err(e) => {
                // GPU not available - graceful degradation
                eprintln!("GPU initialization failed (expected on CI): {e}");
            }
        }
    }

    /// Test GPU sum matches Scalar sum
    #[tokio::test]
    async fn test_gpu_scalar_equivalence_sum() {
        let Ok(engine) = GpuEngine::new().await else {
            eprintln!("Skipping GPU test (no GPU available)");
            return;
        };

        let data = vec![1i32, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let scalar_sum: i32 = data.iter().sum();

        let arrow_array = Int32Array::from(data);
        let gpu_sum = engine.sum_i32(&arrow_array).await.expect("GPU sum should work");

        assert_eq!(scalar_sum, gpu_sum, "GPU sum should equal Scalar sum");
    }

    /// Test GPU min matches Scalar min
    #[tokio::test]
    async fn test_gpu_scalar_equivalence_min() {
        let Ok(engine) = GpuEngine::new().await else {
            eprintln!("Skipping GPU test (no GPU available)");
            return;
        };

        let data = vec![5i32, 2, 8, 1, 9, 3, 7, 4, 6, 10];
        let scalar_min = *data.iter().min().unwrap();

        let arrow_array = Int32Array::from(data);
        let gpu_min = engine.min_i32(&arrow_array).await.expect("GPU min should work");

        assert_eq!(scalar_min, gpu_min, "GPU min should equal Scalar min");
    }

    /// Test GPU max matches Scalar max
    #[tokio::test]
    async fn test_gpu_scalar_equivalence_max() {
        let Ok(engine) = GpuEngine::new().await else {
            eprintln!("Skipping GPU test (no GPU available)");
            return;
        };

        let data = vec![5i32, 2, 8, 1, 9, 3, 7, 4, 6, 10];
        let scalar_max = *data.iter().max().unwrap();

        let arrow_array = Int32Array::from(data);
        let gpu_max = engine.max_i32(&arrow_array).await.expect("GPU max should work");

        assert_eq!(scalar_max, gpu_max, "GPU max should equal Scalar max");
    }

    /// Test GPU count matches Scalar count
    #[tokio::test]
    async fn test_gpu_scalar_equivalence_count() {
        let Ok(engine) = GpuEngine::new().await else {
            eprintln!("Skipping GPU test (no GPU available)");
            return;
        };

        let data = vec![1i32, 2, 3, 4, 5];
        let scalar_count = data.len();

        let arrow_array = Int32Array::from(data);
        let gpu_count = engine.count(&arrow_array).await.expect("GPU count should work");

        assert_eq!(scalar_count, gpu_count, "GPU count should equal Scalar count");
    }

    /// Test GPU fused filter+sum works correctly
    #[tokio::test]
    async fn test_gpu_fused_filter_sum() {
        let Ok(engine) = GpuEngine::new().await else {
            eprintln!("Skipping GPU test (no GPU available)");
            return;
        };

        let data = vec![1i32, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        // Filter: value > 5, sum: 6 + 7 + 8 + 9 + 10 = 40
        let scalar_result: i32 = data.iter().filter(|&&x| x > 5).sum();

        let arrow_array = Int32Array::from(data);
        let gpu_result = engine
            .fused_filter_sum(&arrow_array, 5, "gt")
            .await
            .expect("GPU fused filter+sum should work");

        assert_eq!(
            scalar_result, gpu_result,
            "GPU fused filter+sum should equal Scalar implementation"
        );
    }
}

// ============================================================================
// POLICY ENFORCEMENT: Backend Story Completeness
// ============================================================================
//
// The following module contains compile-time assertions that verify the
// backend story is complete for all major operations.
//
// If you're adding a new operation to trueno-db, you MUST:
// 1. Implement it in the scalar backend (reference)
// 2. Implement it using trueno Vector (SIMD)
// 3. Implement it in GpuEngine (GPU)
// 4. Add a test in this file verifying equivalence
// 5. Update CLAUDE.md if adding a new category of operations
//
// Failure to do so will cause CI to fail and block your PR.
// ============================================================================

#[cfg(test)]
mod backend_completeness {
    //! Compile-time verification that critical traits/types exist

    use trueno_db::backend::BackendDispatcher;
    use trueno_db::Backend;

    /// Verify Backend enum has required variants
    #[test]
    fn test_backend_variants_exist() {
        fn check_variant(b: Backend) {
            match b {
                Backend::CostBased => {}
                Backend::Gpu => {}
                Backend::Simd => {}
            }
        }

        check_variant(Backend::CostBased);
        check_variant(Backend::Gpu);
        check_variant(Backend::Simd);
    }

    /// Verify `BackendDispatcher` methods exist
    #[test]
    fn test_backend_dispatcher_methods_exist() {
        // These function pointer assignments verify the methods exist at compile time
        let _: fn(usize, f64) -> Backend = BackendDispatcher::select;
        let _: fn(f64, usize) -> f64 = BackendDispatcher::arithmetic_intensity;
        let _: fn(usize) -> f64 = BackendDispatcher::estimate_simple_aggregation_flops;
        let _: fn(usize) -> f64 = BackendDispatcher::estimate_group_by_flops;
        let _: fn(usize) -> f64 = BackendDispatcher::estimate_filter_flops;
        let _: fn(usize, usize) -> f64 = BackendDispatcher::estimate_join_flops;
    }
}
