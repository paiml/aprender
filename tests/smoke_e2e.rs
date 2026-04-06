#![allow(clippy::disallowed_methods, clippy::float_cmp)]
//! TRUENO-SPEC-013: E2E Smoke Tests for Backend Equivalence
//!
//! This module implements the smoke test requirements from TRUENO-SPEC-013 Section 3.2.
//! Tests verify that all backends (Scalar, SIMD, WGPU, CUDA) produce equivalent results.
//!
//! # Running
//! ```bash
//! # All smoke tests
//! make smoke
//!
//! # Individual backends
//! cargo test --test smoke_e2e smoke_simd -- --nocapture
//! cargo test --test smoke_e2e smoke_wgpu --features gpu -- --nocapture
//! ```
//!
//! # Toyota Way Alignment
//! - **Genchi Genbutsu**: Actually execute code on real hardware, don't simulate
//! - **Jidoka**: Stop the line when backend results don't match
//! - **Poka-Yoke**: Smoke tests catch bugs before they propagate

use trueno::Vector;

// Tolerance for floating-point comparison (SPEC Section 2.2)
const FP_TOLERANCE: f32 = 1e-5;

// ============================================================================
// SIMD BACKEND SMOKE TESTS
// ============================================================================

/// Smoke test: Vector addition across SIMD backends
#[test]
fn smoke_simd_vector_add() {
    let size = 10_000;
    let a: Vec<f32> = (0..size).map(|i| i as f32 * 0.1).collect();
    let b: Vec<f32> = (0..size).map(|i| (size - i) as f32 * 0.1).collect();

    let va = Vector::from_slice(&a);
    let vb = Vector::from_slice(&b);

    let result = va.add(&vb).expect("Vector add failed");

    // Verify against expected (scalar computation)
    let expected: Vec<f32> = a.iter().zip(b.iter()).map(|(x, y)| x + y).collect();
    for (i, (got, want)) in result.as_slice().iter().zip(expected.iter()).enumerate() {
        assert!((got - want).abs() < FP_TOLERANCE, "Mismatch at index {i}: got {got}, want {want}");
    }
}

/// Smoke test: Dot product across SIMD backends
#[test]
fn smoke_simd_dot_product() {
    let size = 10_000;
    let a: Vec<f32> = (0..size).map(|i| (i as f32).sin()).collect();
    let b: Vec<f32> = (0..size).map(|i| (i as f32).cos()).collect();

    let va = Vector::from_slice(&a);
    let vb = Vector::from_slice(&b);

    let result = va.dot(&vb).expect("Dot product failed");

    // Verify against scalar computation
    let expected: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    assert!(
        (result - expected).abs() < FP_TOLERANCE * size as f32,
        "Dot product mismatch: got {result}, want {expected}"
    );
}

/// Smoke test: Vector norm
#[test]
fn smoke_simd_vector_norm() {
    let size = 10_000;
    let data: Vec<f32> = (0..size).map(|i| (i as f32 * 0.01).sin()).collect();

    let v = Vector::from_slice(&data);
    let result = v.norm_l2().expect("norm_l2 failed");

    // Verify against scalar computation
    // Norm accumulates errors across all elements, so use larger tolerance
    let expected = data.iter().map(|x| x * x).sum::<f32>().sqrt();
    let tolerance = FP_TOLERANCE * (size as f32).sqrt(); // Scale tolerance with sqrt(n)
    assert!(
        (result - expected).abs() < tolerance,
        "Norm mismatch: got {result}, want {expected}, diff {}, tolerance {}",
        (result - expected).abs(),
        tolerance
    );
}

/// Smoke test: Element-wise multiply
#[test]
fn smoke_simd_vector_mul() {
    let size = 10_000;
    let a: Vec<f32> = (0..size).map(|i| i as f32 * 0.01).collect();
    let b: Vec<f32> = (0..size).map(|i| (size - i) as f32 * 0.01).collect();

    let va = Vector::from_slice(&a);
    let vb = Vector::from_slice(&b);

    let result = va.mul(&vb).expect("Vector mul failed");

    // Verify against expected
    let expected: Vec<f32> = a.iter().zip(b.iter()).map(|(x, y)| x * y).collect();
    for (i, (got, want)) in result.as_slice().iter().zip(expected.iter()).enumerate() {
        assert!((got - want).abs() < FP_TOLERANCE, "Mismatch at index {i}: got {got}, want {want}");
    }
}

/// Smoke test: ReLU activation (common PTX bug area)
#[test]
fn smoke_simd_relu() {
    let size = 10_000;
    let data: Vec<f32> =
        (0..size).map(|i| if i % 2 == 0 { i as f32 } else { -(i as f32) }).collect();

    let v = Vector::from_slice(&data);
    let result = v.relu().expect("ReLU failed");

    // Verify ReLU: max(0, x)
    for (i, (got, orig)) in result.as_slice().iter().zip(data.iter()).enumerate() {
        let want = orig.max(0.0);
        assert!(
            (got - want).abs() < FP_TOLERANCE,
            "ReLU mismatch at index {i}: got {got}, want {want}"
        );
    }
}

/// Smoke test: Softmax (numerical stability test)
#[test]
fn smoke_simd_softmax() {
    let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
    let v = Vector::from_slice(&data);

    let result = v.softmax().expect("Softmax failed");

    // Verify softmax sums to 1
    let sum: f32 = result.as_slice().iter().sum();
    assert!((sum - 1.0).abs() < FP_TOLERANCE, "Softmax sum should be 1.0, got {sum}");

    // Verify all values are positive
    for (i, val) in result.as_slice().iter().enumerate() {
        assert!(*val > 0.0, "Softmax value at {i} should be positive: {val}");
    }
}

// ============================================================================
// WGPU BACKEND SMOKE TESTS
// ============================================================================

#[cfg(feature = "gpu")]
mod wgpu_tests {
    use super::*;
    use trueno::backends::gpu::GpuBackend;

    /// Smoke test: WGPU vector operations
    #[test]
    fn smoke_wgpu_vector_add() {
        // Skip if no GPU available
        if !GpuBackend::is_available() {
            eprintln!("Skipping WGPU test: no GPU available");
            return;
        }

        let size = 100_000; // Larger for GPU threshold
        let a: Vec<f32> = (0..size).map(|i| i as f32 * 0.1).collect();
        let b: Vec<f32> = (0..size).map(|i| (size - i) as f32 * 0.1).collect();

        let va = Vector::from_slice(&a);
        let vb = Vector::from_slice(&b);

        let result = va.add(&vb).expect("WGPU vector add failed");

        // Verify against scalar baseline
        let expected: Vec<f32> = a.iter().zip(b.iter()).map(|(x, y)| x + y).collect();

        let mut max_diff = 0.0f32;
        for (got, want) in result.as_slice().iter().zip(expected.iter()) {
            let diff = (got - want).abs();
            if diff > max_diff {
                max_diff = diff;
            }
        }

        // WGPU may have slightly higher tolerance (2 ULP as per spec)
        let wgpu_tolerance = FP_TOLERANCE * 2.0;
        assert!(
            max_diff < wgpu_tolerance,
            "WGPU max diff {max_diff} exceeds tolerance {wgpu_tolerance}"
        );
    }

    /// Smoke test: WGPU matrix multiply
    #[test]
    fn smoke_wgpu_matmul() {
        if !GpuBackend::is_available() {
            eprintln!("Skipping WGPU matmul test: no GPU available");
            return;
        }

        use trueno::Matrix;

        // 256x256 matrix multiply (above GPU threshold)
        let n = 256;
        let a_data: Vec<f32> = (0..n * n).map(|i| (i as f32 * 0.001) % 1.0).collect();
        let b_data: Vec<f32> = (0..n * n).map(|i| ((n * n - i) as f32 * 0.001) % 1.0).collect();

        let a = Matrix::from_vec(n, n, a_data.clone()).expect("Matrix A creation failed");
        let b = Matrix::from_vec(n, n, b_data.clone()).expect("Matrix B creation failed");
        let result = a.matmul(&b).expect("WGPU matmul failed");

        // Verify dimensions
        assert_eq!(result.as_slice().len(), n * n, "Result dimensions wrong");

        // Spot check a few values against scalar baseline
        for i in 0..3 {
            for j in 0..3 {
                let mut expected = 0.0f32;
                for k in 0..n {
                    expected += a_data[i * n + k] * b_data[k * n + j];
                }
                let got = result.as_slice()[i * n + j];
                let diff = (got - expected).abs();
                assert!(diff < FP_TOLERANCE * n as f32, "Matmul[{i}][{j}] diff {diff} too large");
            }
        }
    }
}

// ============================================================================
// BACKEND EQUIVALENCE TESTS
// ============================================================================

/// Test that all backends produce equivalent results for the same operation
#[test]
fn smoke_backend_equivalence() {
    let size = 1000;
    let a: Vec<f32> = (0..size).map(|i| (i as f32 * 0.1).sin()).collect();
    let b: Vec<f32> = (0..size).map(|i| (i as f32 * 0.1).cos()).collect();

    // Force scalar backend
    let scalar_a = Vector::from_slice_with_backend(&a, trueno::Backend::Scalar);
    let scalar_b = Vector::from_slice_with_backend(&b, trueno::Backend::Scalar);
    let scalar_result = scalar_a.add(&scalar_b).expect("Scalar add failed");

    // Auto backend (will select best available)
    let auto_a = Vector::from_slice(&a);
    let auto_b = Vector::from_slice(&b);
    let auto_result = auto_a.add(&auto_b).expect("Auto add failed");

    // Compare results
    for (i, (scalar, auto)) in
        scalar_result.as_slice().iter().zip(auto_result.as_slice().iter()).enumerate()
    {
        assert!(
            (scalar - auto).abs() < FP_TOLERANCE,
            "Backend equivalence failed at index {i}: scalar={scalar}, auto={auto}"
        );
    }
}

// ============================================================================
// EDGE CASE SMOKE TESTS (Poka-Yoke)
// ============================================================================

/// Empty input handling
#[test]
fn smoke_empty_input() {
    let empty: Vec<f32> = vec![];
    let v = Vector::from_slice(&empty);
    assert_eq!(v.len(), 0);
}

/// Single element
#[test]
fn smoke_single_element() {
    let single = vec![42.0f32];
    let v = Vector::from_slice(&single);
    let norm = v.norm_l2().expect("norm_l2 failed");
    assert!((norm - 42.0).abs() < FP_TOLERANCE, "Single element norm: {norm}");
}

/// Non-aligned sizes (test remainder handling)
#[test]
fn smoke_non_aligned_17() {
    let size = 17; // Not divisible by 4, 8, or 16
    let a: Vec<f32> = (0..size).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..size).map(|i| (size - i) as f32).collect();

    let va = Vector::from_slice(&a);
    let vb = Vector::from_slice(&b);

    let result = va.add(&vb).expect("Non-aligned add failed");

    // All values should be 17.0
    for (i, val) in result.as_slice().iter().enumerate() {
        assert!((val - 17.0).abs() < FP_TOLERANCE, "Non-aligned result at {i}: {val}");
    }
}

/// NaN propagation
#[test]
fn smoke_nan_propagation() {
    let data = vec![1.0f32, f32::NAN, 3.0];
    let v = Vector::from_slice(&data);

    let result = v.add(&v).expect("NaN add failed");

    // NaN should propagate
    assert!(result.as_slice()[1].is_nan(), "NaN should propagate");
}

/// Infinity handling
#[test]
fn smoke_infinity_handling() {
    let data = vec![1.0f32, f32::INFINITY, -f32::INFINITY];
    let v = Vector::from_slice(&data);

    let result = v.mul(&v).expect("Infinity mul failed");

    assert!(result.as_slice()[1].is_infinite(), "Infinity should persist");
}

// ============================================================================
// PERFORMANCE SMOKE TESTS
// ============================================================================

/// Smoke test should complete in < 2 minutes (SPEC Section 2.3)
#[test]
fn smoke_performance_baseline() {
    let start = std::time::Instant::now();

    // Run a representative workload
    let size = 100_000;
    let iterations = 10;

    for _ in 0..iterations {
        let a: Vec<f32> = (0..size).map(|i| i as f32 * 0.01).collect();
        let b: Vec<f32> = (0..size).map(|i| (size - i) as f32 * 0.01).collect();

        let va = Vector::from_slice(&a);
        let vb = Vector::from_slice(&b);

        let _ = va.add(&vb).expect("Add failed");
        let _ = va.mul(&vb).expect("Mul failed");
        let _ = va.dot(&vb).expect("Dot failed");
    }

    let elapsed = start.elapsed();
    let max_duration = std::time::Duration::from_secs(120); // 2 minutes

    assert!(elapsed < max_duration, "Smoke test took {:?}, exceeds 2 minute limit", elapsed);

    println!("Smoke test completed in {:?}", elapsed);
}
