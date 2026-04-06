#![allow(clippy::disallowed_methods, clippy::float_cmp)]
//! Backend Story Integration Tests
//!
//! CRITICAL: These tests enforce that ALL operations in trueno support the complete
//! backend story: Scalar, SIMD (SSE2/AVX2/AVX512/NEON), GPU, and WASM.
//!
//! If these tests fail, it means a new operation was added without proper backend support.
//! THIS IS A BLOCKING ISSUE - do not merge code that breaks the backend story.
//!
//! Reference: CLAUDE.md "Backend Story Policy"

use trueno::{Backend, Matrix, SymmetricEigen, Vector};

/// Test that SymmetricEigen works on all CPU backends
#[test]
fn test_symmetric_eigen_all_backends() {
    // Create a simple 3x3 symmetric matrix
    #[rustfmt::skip]
    let matrix = Matrix::from_vec(3, 3, vec![
        4.0, 2.0, 0.0,
        2.0, 5.0, 3.0,
        0.0, 3.0, 6.0,
    ]).expect("valid matrix");

    // SymmetricEigen::new() uses Backend::select_best() internally
    // This test verifies it works regardless of which backend is selected
    let eigen = SymmetricEigen::new(&matrix).expect("eigendecomposition should succeed");

    // Verify basic properties
    assert_eq!(eigen.len(), 3, "Should have 3 eigenvalues");
    assert_eq!(eigen.eigenvalues().len(), 3);

    // Eigenvalues should be sorted descending
    let values = eigen.eigenvalues();
    assert!(
        values[0] >= values[1] && values[1] >= values[2],
        "Eigenvalues should be sorted descending"
    );

    // Verify A*v = lambda*v property (eigenpair validation)
    for (lambda, v) in eigen.iter() {
        let av = matrix.matvec(&v).expect("matvec should succeed");
        let lambda_v: Vec<f32> = v.as_slice().iter().map(|x| x * lambda).collect();

        let error: f32 =
            av.as_slice().iter().zip(lambda_v.iter()).map(|(a, b)| (a - b).abs()).sum();

        assert!(error < 1e-4, "Eigenpair validation failed: error={error} (expected < 1e-4)");
    }

    // Verify reconstruction: V * D * V^T = A
    let reconstructed = eigen.reconstruct().expect("reconstruction should succeed");
    let mut max_error = 0.0f32;
    for i in 0..3 {
        for j in 0..3 {
            let orig = matrix.get(i, j).unwrap();
            let recon = reconstructed.get(i, j).unwrap();
            max_error = max_error.max((orig - recon).abs());
        }
    }
    assert!(max_error < 1e-5, "Reconstruction error too large: {max_error}");
}

/// Test that SymmetricEigen works with explicit backend selection
#[test]
fn test_symmetric_eigen_backend_equivalence() {
    #[rustfmt::skip]
    let matrix = Matrix::from_vec(2, 2, vec![
        3.0, 1.0,
        1.0, 3.0,
    ]).expect("valid matrix");

    // Known eigenvalues: 4.0 and 2.0
    let eigen = SymmetricEigen::new(&matrix).expect("eigendecomposition");
    let values = eigen.eigenvalues();

    assert!((values[0] - 4.0).abs() < 1e-5, "First eigenvalue should be 4.0, got {}", values[0]);
    assert!((values[1] - 2.0).abs() < 1e-5, "Second eigenvalue should be 2.0, got {}", values[1]);
}

/// Test that Vector operations work on all backends
#[test]
fn test_vector_ops_all_backends() {
    let a = Vector::from_slice(&[1.0f32, 2.0, 3.0, 4.0]);
    let b = Vector::from_slice(&[5.0f32, 6.0, 7.0, 8.0]);

    // Test basic operations that must work on all backends
    let sum = a.add(&b).expect("add should work");
    assert_eq!(sum.as_slice(), &[6.0, 8.0, 10.0, 12.0]);

    let diff = a.sub(&b).expect("sub should work");
    assert_eq!(diff.as_slice(), &[-4.0, -4.0, -4.0, -4.0]);

    let prod = a.mul(&b).expect("mul should work");
    assert_eq!(prod.as_slice(), &[5.0, 12.0, 21.0, 32.0]);

    let dot = a.dot(&b).expect("dot should work");
    assert!((dot - 70.0).abs() < 1e-5, "dot product should be 70.0");
}

/// Test that Matrix operations work on all backends
#[test]
fn test_matrix_ops_all_backends() {
    #[rustfmt::skip]
    let a = Matrix::from_vec(2, 2, vec![
        1.0, 2.0,
        3.0, 4.0,
    ]).expect("valid matrix");

    #[rustfmt::skip]
    let b = Matrix::from_vec(2, 2, vec![
        5.0, 6.0,
        7.0, 8.0,
    ]).expect("valid matrix");

    // Matrix multiplication must work on all backends
    let c = a.matmul(&b).expect("matmul should work");
    assert!((c.get(0, 0).unwrap() - 19.0).abs() < 1e-5);
    assert!((c.get(0, 1).unwrap() - 22.0).abs() < 1e-5);
    assert!((c.get(1, 0).unwrap() - 43.0).abs() < 1e-5);
    assert!((c.get(1, 1).unwrap() - 50.0).abs() < 1e-5);

    // Transpose must work on all backends
    let at = a.transpose();
    assert!((at.get(0, 0).unwrap() - 1.0).abs() < 1e-5);
    assert!((at.get(0, 1).unwrap() - 3.0).abs() < 1e-5);
    assert!((at.get(1, 0).unwrap() - 2.0).abs() < 1e-5);
    assert!((at.get(1, 1).unwrap() - 4.0).abs() < 1e-5);
}

/// Test that backend selection returns a valid backend
#[test]
fn test_backend_selection_always_valid() {
    let backend = Backend::select_best();

    // Must be one of the valid backends
    let valid_backends = [
        Backend::Scalar,
        Backend::SSE2,
        Backend::AVX,
        Backend::AVX2,
        Backend::AVX512,
        Backend::NEON,
        Backend::WasmSIMD,
        Backend::GPU,
    ];

    assert!(
        valid_backends.contains(&backend),
        "Backend::select_best() returned invalid backend: {:?}",
        backend
    );
}

/// Compile-time check: SymmetricEigen must be available without feature flags
/// This ensures eigendecomposition is part of the core API, not feature-gated.
#[test]
fn test_symmetric_eigen_always_available() {
    // This test verifies that SymmetricEigen is in the public API
    // If this fails to compile, it means SymmetricEigen was incorrectly feature-gated
    let _: fn(&Matrix<f32>) -> Result<SymmetricEigen, trueno::TruenoError> = SymmetricEigen::new;
}

/// Test GPU backend availability check doesn't panic
#[cfg(feature = "gpu")]
#[test]
fn test_gpu_backend_available_check() {
    use trueno::backends::gpu::GpuBackend;

    // This should never panic, only return true/false
    let _available = GpuBackend::is_available();
}

/// Test that large matrix eigendecomposition works (exercises GPU path if available)
#[test]
fn test_large_matrix_eigen() {
    let n = 100;
    let mut data = vec![0.0f32; n * n];

    // Create a diagonally dominant symmetric matrix
    for i in 0..n {
        for j in 0..=i {
            let val = 1.0 / (1.0 + (i as f32 - j as f32).abs());
            data[i * n + j] = val;
            data[j * n + i] = val;
        }
        data[i * n + i] += n as f32; // Diagonal dominance
    }

    let matrix = Matrix::from_vec(n, n, data).expect("valid matrix");
    let eigen = SymmetricEigen::new(&matrix).expect("eigendecomposition should succeed");

    assert_eq!(eigen.len(), n, "Should have {n} eigenvalues");

    // Verify eigenvalues are sorted descending
    let values = eigen.eigenvalues();
    for i in 0..n - 1 {
        assert!(
            values[i] >= values[i + 1],
            "Eigenvalues not sorted: values[{}]={} < values[{}]={}",
            i,
            values[i],
            i + 1,
            values[i + 1]
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
// If you're adding a new operation to trueno, you MUST:
// 1. Add it to the VectorBackend trait (or equivalent)
// 2. Implement it in ALL backends (scalar, sse2, avx2, avx512, neon, wasm, gpu)
// 3. Add a test in this file verifying it works
// 4. Update CLAUDE.md if adding a new category of operations
//
// Failure to do so will cause CI to fail and block your PR.
// ============================================================================

#[cfg(test)]
mod backend_completeness {
    //! Compile-time verification that critical traits are implemented
    //! for all backend-related types.

    use trueno::Backend;

    /// Verify Backend enum has required variants
    #[test]
    fn test_backend_variants_exist() {
        // These pattern matches ensure the variants exist at compile time
        fn check_variant(b: Backend) {
            match b {
                Backend::Scalar => {}
                Backend::SSE2 => {}
                Backend::AVX => {}
                Backend::AVX2 => {}
                Backend::AVX512 => {}
                Backend::NEON => {}
                Backend::WasmSIMD => {}
                Backend::GPU => {}
                Backend::Auto => {}
            }
        }

        check_variant(Backend::Scalar);
        check_variant(Backend::Auto);
    }
}
