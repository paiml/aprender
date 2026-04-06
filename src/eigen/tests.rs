use super::*;

// =========================================================================
// RED PHASE: Tests that define expected behavior
// =========================================================================

#[test]
fn test_symmetric_eigen_2x2_simple() {
    // Simple 2x2 symmetric matrix: [[2, 1], [1, 2]]
    // Eigenvalues: 3, 1
    // Eigenvectors: [1/√2, 1/√2], [1/√2, -1/√2]
    let m = Matrix::from_vec(2, 2, vec![2.0, 1.0, 1.0, 2.0]).expect("valid matrix");

    let eigen = SymmetricEigen::new(&m).expect("eigendecomposition should succeed");

    let values = eigen.eigenvalues();
    assert_eq!(values.len(), 2);

    // Eigenvalues should be in descending order
    assert!(values[0] >= values[1], "eigenvalues must be descending");

    // Check eigenvalue values (with tolerance)
    assert!((values[0] - 3.0).abs() < 1e-5, "first eigenvalue should be 3, got {}", values[0]);
    assert!((values[1] - 1.0).abs() < 1e-5, "second eigenvalue should be 1, got {}", values[1]);
}

#[test]
fn test_symmetric_eigen_identity() {
    // Identity matrix has all eigenvalues = 1
    let m = Matrix::identity(3);

    let eigen = SymmetricEigen::new(&m).expect("eigendecomposition should succeed");

    let values = eigen.eigenvalues();
    assert_eq!(values.len(), 3);

    for (i, &val) in values.iter().enumerate() {
        assert!((val - 1.0).abs() < 1e-5, "eigenvalue {} should be 1, got {}", i, val);
    }
}

#[test]
fn test_symmetric_eigen_diagonal() {
    // Diagonal matrix: eigenvalues are the diagonal elements
    let m = Matrix::from_vec(3, 3, vec![5.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 1.0])
        .expect("valid matrix");

    let eigen = SymmetricEigen::new(&m).expect("eigendecomposition should succeed");

    let values = eigen.eigenvalues();

    // Should be sorted descending: 5, 3, 1
    assert!((values[0] - 5.0).abs() < 1e-5, "got {}", values[0]);
    assert!((values[1] - 3.0).abs() < 1e-5, "got {}", values[1]);
    assert!((values[2] - 1.0).abs() < 1e-5, "got {}", values[2]);
}

#[test]
fn test_symmetric_eigen_eigenvectors_orthogonal() {
    let m = Matrix::from_vec(3, 3, vec![4.0, 2.0, 0.0, 2.0, 5.0, 3.0, 0.0, 3.0, 6.0])
        .expect("valid matrix");

    let eigen = SymmetricEigen::new(&m).expect("eigendecomposition should succeed");

    // Eigenvectors should be orthonormal: V^T × V = I
    let v = eigen.eigenvectors();
    let vt = v.transpose();
    let product = vt.matmul(v).expect("matmul should succeed");

    // Check if product is approximately identity
    for i in 0..3 {
        for j in 0..3 {
            let expected = if i == j { 1.0 } else { 0.0 };
            let actual = product.get(i, j).unwrap();
            assert!(
                (actual - expected).abs() < 1e-4,
                "V^T×V[{},{}] = {}, expected {}",
                i,
                j,
                actual,
                expected
            );
        }
    }
}

#[test]
fn test_symmetric_eigen_reconstruction() {
    let m = Matrix::from_vec(2, 2, vec![4.0, 2.0, 2.0, 4.0]).expect("valid matrix");

    let eigen = SymmetricEigen::new(&m).expect("eigendecomposition should succeed");
    let reconstructed = eigen.reconstruct().expect("reconstruction should succeed");

    // Reconstructed matrix should match original
    for i in 0..2 {
        for j in 0..2 {
            let original = m.get(i, j).unwrap();
            let recon = reconstructed.get(i, j).unwrap();
            assert!(
                (original - recon).abs() < 1e-4,
                "A[{},{}] = {}, reconstructed = {}",
                i,
                j,
                original,
                recon
            );
        }
    }
}

#[test]
fn test_symmetric_eigen_av_equals_lambda_v() {
    // For each eigenpair (λ, v): A×v = λ×v
    let m = Matrix::from_vec(2, 2, vec![3.0, 1.0, 1.0, 3.0]).expect("valid matrix");

    let eigen = SymmetricEigen::new(&m).expect("eigendecomposition should succeed");

    for (lambda, v) in eigen.iter() {
        // Compute A×v
        let av = m.matvec(&v).expect("matvec should succeed");

        // Compute λ×v
        let lambda_v: Vec<f32> = v.as_slice().iter().map(|&x| x * lambda).collect();

        // Check equality
        for (i, (&av_i, &lv_i)) in av.as_slice().iter().zip(lambda_v.iter()).enumerate() {
            assert!((av_i - lv_i).abs() < 1e-4, "A×v[{}] = {}, λv[{}] = {}", i, av_i, i, lv_i);
        }
    }
}

#[test]
fn test_symmetric_eigen_error_non_square() {
    let m = Matrix::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).expect("valid matrix");

    let result = SymmetricEigen::new(&m);
    assert!(result.is_err());

    let err = result.unwrap_err();
    assert!(matches!(err, TruenoError::InvalidInput(_)), "expected InvalidInput error");
}

#[test]
fn test_symmetric_eigen_error_empty() {
    let m = Matrix::zeros(0, 0);

    let result = SymmetricEigen::new(&m);
    assert!(result.is_err());
}

#[test]
fn test_symmetric_eigen_1x1() {
    let m = Matrix::from_vec(1, 1, vec![7.0]).expect("valid matrix");

    let eigen = SymmetricEigen::new(&m).expect("eigendecomposition should succeed");

    assert_eq!(eigen.eigenvalues().len(), 1);
    assert!((eigen.eigenvalues()[0] - 7.0).abs() < 1e-6);
}

#[test]
fn test_symmetric_eigen_iterator() {
    let m = Matrix::from_vec(2, 2, vec![2.0, 0.0, 0.0, 1.0]).expect("valid matrix");

    let eigen = SymmetricEigen::new(&m).expect("eigendecomposition should succeed");

    let pairs: Vec<_> = eigen.iter().collect();
    assert_eq!(pairs.len(), 2);

    // First eigenvalue is larger
    assert!(pairs[0].0 >= pairs[1].0);
}

#[test]
fn test_symmetric_eigen_len() {
    let m = Matrix::identity(5);
    let eigen = SymmetricEigen::new(&m).expect("eigendecomposition should succeed");

    assert_eq!(eigen.len(), 5);
    assert!(!eigen.is_empty());
}

#[test]
fn test_symmetric_eigen_eigenvector_accessor() {
    let m = Matrix::identity(3);
    let eigen = SymmetricEigen::new(&m).expect("eigendecomposition should succeed");

    let v0 = eigen.eigenvector(0);
    assert!(v0.is_some());
    assert_eq!(v0.unwrap().len(), 3);

    let v_invalid = eigen.eigenvector(10);
    assert!(v_invalid.is_none());
}

#[test]
fn test_symmetric_eigen_covariance_matrix() {
    // Typical covariance matrix from PCA
    // Points: [(1,2), (3,4), (5,6)] centered → [(-2,-2), (0,0), (2,2)]
    // Cov = [[8/3, 8/3], [8/3, 8/3]] ≈ [[2.67, 2.67], [2.67, 2.67]]
    let m = Matrix::from_vec(2, 2, vec![2.67, 2.67, 2.67, 2.67]).expect("valid matrix");

    let eigen = SymmetricEigen::new(&m).expect("eigendecomposition should succeed");

    // Eigenvalues: 5.34, 0 (approximately)
    let values = eigen.eigenvalues();
    assert!(values[0] > 5.0, "first eigenvalue should be ~5.34");
    assert!(values[1].abs() < 0.1, "second eigenvalue should be ~0");
}

#[test]
fn test_symmetric_eigen_negative_eigenvalues() {
    // Matrix with negative eigenvalues
    // [[0, 1], [1, 0]] has eigenvalues 1, -1
    let m = Matrix::from_vec(2, 2, vec![0.0, 1.0, 1.0, 0.0]).expect("valid matrix");

    let eigen = SymmetricEigen::new(&m).expect("eigendecomposition should succeed");

    let values = eigen.eigenvalues();
    assert!((values[0] - 1.0).abs() < 1e-5, "first eigenvalue should be 1");
    assert!((values[1] - (-1.0)).abs() < 1e-5, "second eigenvalue should be -1");
}

#[test]
fn test_symmetric_eigen_backend() {
    // Test that the backend() method returns the expected value
    let m = Matrix::from_vec(2, 2, vec![4.0, 1.0, 1.0, 3.0]).expect("valid matrix");
    let eigen = SymmetricEigen::new(&m).expect("eigendecomposition should succeed");

    // backend() should return the current backend
    let backend = eigen.backend();
    // On this machine, it should be AVX2
    #[cfg(target_arch = "x86_64")]
    {
        use crate::Backend;
        assert!(
            matches!(backend, Backend::AVX2 | Backend::Scalar | Backend::SSE2),
            "expected valid backend, got {:?}",
            backend
        );
    }
}

// =========================================================================
// Property-based tests (proptest)
// =========================================================================

#[cfg(test)]
mod proptest_tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(50))]

        #[test]
        fn prop_eigenvalues_descending(n in 2usize..6) {
            // Generate random symmetric matrix
            let mut data = vec![0.0f32; n * n];
            for i in 0..n {
                for j in i..n {
                    let val = (i + j) as f32 / (n as f32);
                    data[i * n + j] = val;
                    data[j * n + i] = val;
                }
            }

            let m = Matrix::from_vec(n, n, data).expect("valid matrix");
            let eigen = SymmetricEigen::new(&m).expect("eigen should succeed");

            let values = eigen.eigenvalues();
            for i in 1..values.len() {
                prop_assert!(
                    values[i - 1] >= values[i],
                    "eigenvalues not descending: {} < {}",
                    values[i - 1],
                    values[i]
                );
            }
        }

        #[test]
        fn prop_eigenvector_count_matches_dimension(n in 1usize..8) {
            let m = Matrix::identity(n);
            let eigen = SymmetricEigen::new(&m).expect("eigen should succeed");

            prop_assert_eq!(eigen.len(), n);
            prop_assert_eq!(eigen.eigenvalues().len(), n);
            prop_assert_eq!(eigen.eigenvectors().rows(), n);
            prop_assert_eq!(eigen.eigenvectors().cols(), n);
        }

        #[test]
        fn prop_reconstruction_accuracy(
            a in 1.0f32..10.0,  // Ensure positive diagonal for conditioning
            b in -5.0f32..5.0,  // Off-diagonal smaller than diagonal
            c in 1.0f32..10.0   // Ensure positive diagonal for conditioning
        ) {
            // Create symmetric 2x2 matrix [[a+|b|, b], [b, c+|b|]]
            // Add |b| to diagonal for better conditioning
            let diag_a = a + b.abs();
            let diag_c = c + b.abs();
            let m = Matrix::from_vec(2, 2, vec![diag_a, b, b, diag_c]).expect("valid matrix");

            if let Ok(eigen) = SymmetricEigen::new(&m) {
                if let Ok(recon) = eigen.reconstruct() {
                    // Use relative error for numerical stability
                    let frobenius_orig: f32 = [diag_a, b, b, diag_c].iter()
                        .map(|x| x * x).sum::<f32>().sqrt();
                    let max_allowed_error = 0.01 * frobenius_orig.max(1.0);

                    for i in 0..2 {
                        for j in 0..2 {
                            let orig = m.get(i, j).unwrap();
                            let rec = recon.get(i, j).unwrap();
                            prop_assert!(
                                (orig - rec).abs() < max_allowed_error,
                                "reconstruction error: {} vs {}, allowed: {}",
                                orig,
                                rec,
                                max_allowed_error
                            );
                        }
                    }
                }
            }
        }
    }
}

// =========================================================================
// GPU Tests - Exercise compute_gpu path for large symmetric matrices
// =========================================================================

#[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
mod gpu_tests {
    use super::*;

    /// Test compute_gpu via public API with a diagonal matrix large enough to
    /// exceed GPU_THRESHOLD (n >= 1000).
    /// Diagonal matrices have known eigenvalues (the diagonal elements).
    #[test]
    fn test_symmetric_eigen_gpu_diagonal() {
        use crate::backends::gpu::GpuBackend;

        if !GpuBackend::is_available() {
            eprintln!("GPU not available, skipping test_symmetric_eigen_gpu_diagonal");
            return;
        }

        // n=1000 triggers the GPU path (GPU_THRESHOLD = 1000)
        let n = 1000;
        let mut data = vec![0.0f32; n * n];
        // Set diagonal: eigenvalues are n, n-1, ..., 1
        for i in 0..n {
            data[i * n + i] = (n - i) as f32;
        }

        let m = Matrix::from_vec(n, n, data).expect("valid matrix");
        let eigen = SymmetricEigen::new(&m).expect("GPU eigendecomposition should succeed");

        let values = eigen.eigenvalues();
        assert_eq!(values.len(), n);

        // Eigenvalues should be in descending order
        for i in 1..values.len() {
            assert!(
                values[i - 1] >= values[i] - 1e-3,
                "eigenvalues not descending at index {}: {} < {}",
                i,
                values[i - 1],
                values[i]
            );
        }

        // For a diagonal matrix, eigenvalues should be the diagonal elements
        // The largest should be n=1000, the smallest should be 1
        assert!(
            (values[0] - n as f32).abs() < 1.0,
            "largest eigenvalue should be ~{}, got {}",
            n,
            values[0]
        );
        assert!(
            (values[n - 1] - 1.0).abs() < 1.0,
            "smallest eigenvalue should be ~1.0, got {}",
            values[n - 1]
        );
    }

    /// Test compute_gpu with scaled identity plus small perturbation.
    /// This converges very quickly (1 sweep) since the matrix is nearly diagonal.
    #[test]
    fn test_symmetric_eigen_gpu_near_diagonal() {
        use crate::backends::gpu::GpuBackend;

        if !GpuBackend::is_available() {
            eprintln!("GPU not available, skipping test_symmetric_eigen_gpu_near_diagonal");
            return;
        }

        // n=1000 triggers GPU path
        let n = 1000;
        let mut data = vec![0.0f32; n * n];

        // Nearly diagonal: strong diagonal dominance ensures fast convergence.
        // Diagonal = i+1, tiny off-diagonal perturbation every 100 elements.
        for i in 0..n {
            data[i * n + i] = (i + 1) as f32;
        }
        // Add small off-diagonal blocks every 100 elements for non-trivial structure
        for block in 0..10 {
            let base = block * 100;
            if base + 1 < n {
                data[base * n + (base + 1)] = 0.001;
                data[(base + 1) * n + base] = 0.001;
            }
        }

        let m = Matrix::from_vec(n, n, data).expect("valid matrix");
        let eigen = SymmetricEigen::new(&m).expect("GPU eigendecomposition should succeed");

        let values = eigen.eigenvalues();
        assert_eq!(values.len(), n);

        // Eigenvalues should be in descending order
        for i in 1..values.len() {
            assert!(
                values[i - 1] >= values[i] - 1e-2,
                "eigenvalues not descending at index {}: {} < {}",
                i,
                values[i - 1],
                values[i]
            );
        }

        // Trace should be preserved: sum of eigenvalues = sum of diagonal = n*(n+1)/2
        let trace_expected = (n * (n + 1)) as f32 / 2.0;
        let trace_actual: f32 = values.iter().sum();
        assert!(
            (trace_actual - trace_expected).abs() / trace_expected < 0.01,
            "trace mismatch: expected {}, got {}",
            trace_expected,
            trace_actual
        );

        // Largest eigenvalue should be ~1000
        assert!(
            (values[0] - 1000.0).abs() < 1.0,
            "largest eigenvalue should be ~1000, got {}",
            values[0]
        );
    }

    /// Test compute_gpu directly by calling the private method.
    #[test]
    fn test_compute_gpu_direct() {
        use crate::backends::gpu::GpuBackend;

        if !GpuBackend::is_available() {
            eprintln!("GPU not available, skipping test_compute_gpu_direct");
            return;
        }

        // Use a small symmetric matrix to test the direct compute_gpu method
        // (it internally falls back to CPU for n < 64, but still exercises compute_gpu dispatch)
        let n = 3;
        let data = vec![4.0, 2.0, 0.0, 2.0, 5.0, 3.0, 0.0, 3.0, 6.0];

        let m = Matrix::from_vec(n, n, data).expect("valid matrix");

        // Call compute_gpu directly (it will use the CPU fallback path inside
        // GpuDevice::symmetric_eigen since n < 64, but this still exercises the
        // compute_gpu wrapper in eigen/mod.rs)
        let result = SymmetricEigen::compute_gpu(&m);
        assert!(result.is_ok(), "compute_gpu failed: {:?}", result.err());

        let eigen = result.unwrap();
        let values = eigen.eigenvalues();
        assert_eq!(values.len(), 3);

        // Eigenvalues should be in descending order
        assert!(values[0] >= values[1]);
        assert!(values[1] >= values[2]);

        // The matrix [[4,2,0],[2,5,3],[0,3,6]] has eigenvalues approximately:
        // 8.86, 4.38, 1.76 (trace = 15)
        let trace: f32 = values.iter().sum();
        assert!(
            (trace - 15.0).abs() < 0.1,
            "eigenvalue sum should equal trace (15), got {}",
            trace
        );

        // Backend should be GPU
        assert!(
            matches!(eigen.backend(), crate::Backend::GPU),
            "expected GPU backend, got {:?}",
            eigen.backend()
        );
    }
}
