//! Contract tests for linear algebra: sparse matrices, dense solvers, BLAS Level 3.
//!
//! These tests exercise real mathematical properties against the implementations
//! in trueno-sparse and trueno-solve crates.

#[cfg(test)]
mod tests {
    // =========================================================================
    // Helper functions
    // =========================================================================

    /// Infinity norm of a vector.
    fn vec_inf_norm(v: &[f32]) -> f32 {
        v.iter().map(|x| x.abs()).fold(0.0_f32, f32::max)
    }

    /// Dense matrix-vector multiply: y = A * x (row-major A of shape rows x cols).
    fn dense_matvec(a: &[f32], x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
        let mut y = vec![0.0_f32; rows];
        for i in 0..rows {
            for j in 0..cols {
                y[i] += a[i * cols + j] * x[j];
            }
        }
        y
    }

    /// Dense matrix-matrix multiply: C = A * B (row-major).
    fn dense_matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        let mut c = vec![0.0_f32; m * n];
        for i in 0..m {
            for j in 0..n {
                for p in 0..k {
                    c[i * n + j] += a[i * k + p] * b[p * n + j];
                }
            }
        }
        c
    }

    /// Check if a matrix is approximately symmetric (row-major, n x n).
    fn is_symmetric(m: &[f32], n: usize, tol: f32) -> bool {
        for i in 0..n {
            for j in (i + 1)..n {
                if (m[i * n + j] - m[j * n + i]).abs() > tol {
                    return false;
                }
            }
        }
        true
    }

    /// Infinity norm of a row-major matrix.
    fn mat_inf_norm(a: &[f32], rows: usize, cols: usize) -> f32 {
        let mut max = 0.0_f32;
        for i in 0..rows {
            let row_sum: f32 = (0..cols).map(|j| a[i * cols + j].abs()).sum();
            if row_sum > max {
                max = row_sum;
            }
        }
        max
    }

    // =========================================================================
    // SPARSE MATRIX TESTS
    // =========================================================================

    use trueno_sparse::{BsrMatrix, CooMatrix, CsrMatrix, SellMatrix, SparseOps};

    // ---- BSR round-trip ----

    #[test]
    fn test_bsr_to_csr_roundtrip() {
        // Create a known 4x4 dense matrix, convert to BSR with block_size=2,
        // then BSR -> CSR -> dense, and verify it matches the original.
        #[rustfmt::skip]
        let dense = vec![
            1.0, 2.0, 0.0, 0.0,
            3.0, 4.0, 0.0, 0.0,
            0.0, 0.0, 5.0, 6.0,
            0.0, 0.0, 7.0, 8.0,
        ];

        let bsr = BsrMatrix::from_dense(&dense, 4, 4, 2);
        assert_eq!(bsr.rows(), 4);
        assert_eq!(bsr.cols(), 4);

        let csr = bsr.to_csr().expect("BSR->CSR should succeed");
        assert_eq!(csr.rows(), 4);
        assert_eq!(csr.cols(), 4);

        // Convert CSR back to dense and compare
        let recovered = csr.to_dense();
        for i in 0..16 {
            assert!(
                (dense[i] - recovered[i]).abs() < 1e-6,
                "Mismatch at index {}: expected {}, got {}",
                i,
                dense[i],
                recovered[i]
            );
        }

        // Now go back to BSR from the recovered dense
        let bsr2 = BsrMatrix::from_dense(&recovered, 4, 4, 2);
        let csr2 = bsr2.to_csr().expect("second BSR->CSR");
        let recovered2 = csr2.to_dense();
        for i in 0..16 {
            assert!(
                (recovered[i] - recovered2[i]).abs() < 1e-6,
                "Round-trip mismatch at index {}",
                i
            );
        }
    }

    // ---- BSR alpha/beta SpMV ----

    #[test]
    fn test_bsr_alpha_beta() {
        // y = alpha * A * x + beta * y
        #[rustfmt::skip]
        let dense = vec![
            2.0, 1.0,
            1.0, 3.0,
        ];
        let bsr = BsrMatrix::from_dense(&dense, 2, 2, 2);

        let x = vec![1.0, 2.0];
        let mut y = vec![10.0, 20.0];
        let alpha = 2.0;
        let beta = 0.5;

        // Expected: y = 2.0 * A * [1,2] + 0.5 * [10,20]
        // A*x = [2*1+1*2, 1*1+3*2] = [4, 7]
        // y = 2*[4,7] + 0.5*[10,20] = [8,14] + [5,10] = [13,24]
        bsr.spmv(alpha, &x, beta, &mut y).unwrap();

        assert!((y[0] - 13.0).abs() < 1e-5, "y[0] = {}, expected 13.0", y[0]);
        assert!((y[1] - 24.0).abs() < 1e-5, "y[1] = {}, expected 24.0", y[1]);
    }

    // ---- SELL matches CSR ----

    #[test]
    fn test_sell_matches_csr() {
        // Build a 4x4 sparse matrix via COO, convert to CSR, then SELL.
        // Verify SpMV results match.
        let coo = CooMatrix::new(
            4,
            4,
            vec![0, 0, 1, 2, 2, 3],
            vec![0, 2, 1, 0, 2, 3],
            vec![1.0_f32, 3.0, 5.0, 2.0, 4.0, 6.0],
        )
        .unwrap();
        let csr = CsrMatrix::from_coo(&coo);
        let sell = SellMatrix::from_csr(&csr, 2);

        let x = vec![1.0_f32, 2.0, 3.0, 4.0];

        // CSR SpMV
        let mut y_csr = vec![0.0_f32; 4];
        csr.spmv(1.0, &x, 0.0, &mut y_csr).unwrap();

        // SELL SpMV
        let mut y_sell = vec![0.0_f32; 4];
        sell.spmv(1.0, &x, 0.0, &mut y_sell).unwrap();

        for i in 0..4 {
            assert!(
                (y_csr[i] - y_sell[i]).abs() < 1e-5,
                "Mismatch at {}: CSR={}, SELL={}",
                i,
                y_csr[i],
                y_sell[i]
            );
        }
    }

    // ---- SELL dimension mismatch ----

    #[test]
    fn test_sell_spmv_dimension_mismatch() {
        let csr = CsrMatrix::identity(3);
        let sell = SellMatrix::from_csr(&csr, 2);

        let x = vec![1.0_f32; 5]; // Wrong: should be 3
        let mut y = vec![0.0_f32; 3];
        let result = sell.spmv(1.0, &x, 0.0, &mut y);
        assert!(result.is_err(), "Should fail on dimension mismatch");
    }

    // ---- SELL alpha/beta ----

    #[test]
    fn test_sell_spmv_alpha_beta() {
        let csr = CsrMatrix::identity(3);
        let sell = SellMatrix::from_csr(&csr, 2);

        let x = vec![1.0_f32, 2.0, 3.0];
        let mut y = vec![10.0_f32, 20.0, 30.0];

        // y = 2.0 * I * x + 0.5 * y = [2,4,6] + [5,10,15] = [7,14,21]
        sell.spmv(2.0, &x, 0.5, &mut y).unwrap();

        assert!((y[0] - 7.0).abs() < 1e-5);
        assert!((y[1] - 14.0).abs() < 1e-5);
        assert!((y[2] - 21.0).abs() < 1e-5);
    }

    // ---- SpGEMM tests ----

    #[test]
    fn test_spgemm_identity() {
        // A * I = A
        let coo =
            CooMatrix::new(3, 3, vec![0, 1, 2, 0], vec![0, 1, 2, 2], vec![2.0_f32, 3.0, 4.0, 5.0])
                .unwrap();
        let a = CsrMatrix::from_coo(&coo);
        let identity = CsrMatrix::<f32>::identity(3);

        let c = trueno_sparse::spgemm(&a, &identity).unwrap();
        let a_dense = a.to_dense();
        let c_dense = c.to_dense();

        for i in 0..9 {
            assert!(
                (a_dense[i] - c_dense[i]).abs() < 1e-5,
                "A*I mismatch at {}: A={}, C={}",
                i,
                a_dense[i],
                c_dense[i]
            );
        }
    }

    #[test]
    fn test_spgemm_identity_left() {
        // I * A = A
        let coo =
            CooMatrix::new(3, 3, vec![0, 1, 2, 0], vec![0, 1, 2, 2], vec![2.0_f32, 3.0, 4.0, 5.0])
                .unwrap();
        let a = CsrMatrix::from_coo(&coo);
        let identity = CsrMatrix::<f32>::identity(3);

        let c = trueno_sparse::spgemm(&identity, &a).unwrap();
        let a_dense = a.to_dense();
        let c_dense = c.to_dense();

        for i in 0..9 {
            assert!(
                (a_dense[i] - c_dense[i]).abs() < 1e-5,
                "I*A mismatch at {}: A={}, C={}",
                i,
                a_dense[i],
                c_dense[i]
            );
        }
    }

    #[test]
    fn test_spgemm_known_product() {
        // A = [[1, 2], [0, 3]]
        // B = [[4, 0], [1, 2]]
        // A*B = [[6, 4], [3, 6]]
        let coo_a =
            CooMatrix::new(2, 2, vec![0, 0, 1], vec![0, 1, 1], vec![1.0_f32, 2.0, 3.0]).unwrap();
        let coo_b =
            CooMatrix::new(2, 2, vec![0, 1, 1], vec![0, 0, 1], vec![4.0_f32, 1.0, 2.0]).unwrap();
        let a = CsrMatrix::from_coo(&coo_a);
        let b = CsrMatrix::from_coo(&coo_b);

        let c = trueno_sparse::spgemm(&a, &b).unwrap();
        let c_dense = c.to_dense();

        let expected = [6.0, 4.0, 3.0, 6.0];
        for i in 0..4 {
            assert!(
                (c_dense[i] - expected[i]).abs() < 1e-5,
                "Product mismatch at {}: got {}, expected {}",
                i,
                c_dense[i],
                expected[i]
            );
        }
    }

    #[test]
    fn test_spgemm_dimension_mismatch() {
        // A is 2x3, B is 2x2 => dimension mismatch (A.cols=3 != B.rows=2)
        let a = CsrMatrix::new(2, 3, vec![0, 1, 2], vec![0, 1], vec![1.0_f32, 2.0]).unwrap();
        let b = CsrMatrix::<f32>::identity(2);
        let result = trueno_sparse::spgemm(&a, &b);
        assert!(result.is_err(), "Should fail: A.cols != B.rows");
    }

    #[test]
    fn test_spgemm_sparse_result() {
        // Multiply two sparse matrices and verify the result is also sparse.
        // A = diag(1, 2, 3), B = diag(4, 5, 6) => C = diag(4, 10, 18)
        let a =
            CsrMatrix::new(3, 3, vec![0, 1, 2, 3], vec![0, 1, 2], vec![1.0_f32, 2.0, 3.0]).unwrap();
        let b =
            CsrMatrix::new(3, 3, vec![0, 1, 2, 3], vec![0, 1, 2], vec![4.0_f32, 5.0, 6.0]).unwrap();

        let c = trueno_sparse::spgemm(&a, &b).unwrap();
        // Result should have exactly 3 nonzeros (diagonal)
        assert_eq!(c.nnz(), 3, "Product of diagonal matrices should have 3 nnz");

        let c_dense = c.to_dense();
        assert!((c_dense[0] - 4.0).abs() < 1e-5);
        assert!((c_dense[4] - 10.0).abs() < 1e-5);
        assert!((c_dense[8] - 18.0).abs() < 1e-5);
        // Off-diagonals should be zero
        assert!(c_dense[1].abs() < 1e-5);
        assert!(c_dense[3].abs() < 1e-5);
    }

    // ---- SpMM (sparse * dense matrix) ----

    #[test]
    fn test_spmm_matches_dense() {
        // Sparse A (3x3) times dense B (3x2), compare with dense A * B.
        let coo = CooMatrix::new(
            3,
            3,
            vec![0, 0, 1, 2, 2],
            vec![0, 1, 1, 0, 2],
            vec![1.0_f32, 2.0, 3.0, 4.0, 5.0],
        )
        .unwrap();
        let csr = CsrMatrix::from_coo(&coo);
        let a_dense = csr.to_dense();

        // B is 3x2, row-major
        let b = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b_cols = 2;

        // Sparse SpMM
        let mut c_sparse = vec![0.0_f32; 3 * b_cols];
        csr.spmm(1.0, &b, b_cols, 0.0, &mut c_sparse).unwrap();

        // Dense matmul
        let c_dense = dense_matmul(&a_dense, &b, 3, 3, b_cols);

        for i in 0..6 {
            assert!(
                (c_sparse[i] - c_dense[i]).abs() < 1e-4,
                "SpMM mismatch at {}: sparse={}, dense={}",
                i,
                c_sparse[i],
                c_dense[i]
            );
        }
    }

    #[test]
    fn test_spmm_dimension_mismatch() {
        let csr = CsrMatrix::<f32>::identity(3);
        // B has wrong row count (4 instead of 3)
        let b = vec![1.0_f32; 4 * 2];
        let mut c = vec![0.0_f32; 3 * 2];
        let result = csr.spmm(1.0, &b, 2, 0.0, &mut c);
        assert!(result.is_err(), "Should fail on dimension mismatch");
    }

    // ---- SpMV tests ----

    #[test]
    fn test_spmv_known_result() {
        // A = [[1, 0, 2],
        //      [0, 3, 0],
        //      [4, 0, 5]]
        // x = [1, 2, 3]
        // y = A*x = [7, 6, 19]
        let coo = CooMatrix::new(
            3,
            3,
            vec![0, 0, 1, 2, 2],
            vec![0, 2, 1, 0, 2],
            vec![1.0_f32, 2.0, 3.0, 4.0, 5.0],
        )
        .unwrap();
        let csr = CsrMatrix::from_coo(&coo);

        let x = vec![1.0_f32, 2.0, 3.0];
        let mut y = vec![0.0_f32; 3];
        csr.spmv(1.0, &x, 0.0, &mut y).unwrap();

        assert!((y[0] - 7.0).abs() < 1e-5, "y[0]={}, expected 7", y[0]);
        assert!((y[1] - 6.0).abs() < 1e-5, "y[1]={}, expected 6", y[1]);
        assert!((y[2] - 19.0).abs() < 1e-5, "y[2]={}, expected 19", y[2]);
    }

    #[test]
    fn test_spmv_dimension_mismatch() {
        let csr = CsrMatrix::<f32>::identity(3);
        let x = vec![1.0_f32; 5]; // Wrong size
        let mut y = vec![0.0_f32; 3];
        let result = csr.spmv(1.0, &x, 0.0, &mut y);
        assert!(result.is_err());
    }

    #[test]
    fn test_spmv_alpha_beta() {
        // y = alpha * A * x + beta * y
        // A = I(3), x = [1,2,3], y_init = [10,20,30]
        // y = 2 * [1,2,3] + 0.5 * [10,20,30] = [2,4,6] + [5,10,15] = [7,14,21]
        let csr = CsrMatrix::<f32>::identity(3);
        let x = vec![1.0_f32, 2.0, 3.0];
        let mut y = vec![10.0_f32, 20.0, 30.0];

        csr.spmv(2.0, &x, 0.5, &mut y).unwrap();

        assert!((y[0] - 7.0).abs() < 1e-5);
        assert!((y[1] - 14.0).abs() < 1e-5);
        assert!((y[2] - 21.0).abs() < 1e-5);
    }

    #[test]
    fn prop_spmv_backward_error() {
        // For a known sparse matrix, verify ||Ax - y|| / (||A|| * ||x||) < eps
        // where eps is proportional to machine epsilon * nnz_per_row.
        let coo = CooMatrix::new(
            3,
            3,
            vec![0, 0, 0, 1, 1, 1, 2, 2, 2],
            vec![0, 1, 2, 0, 1, 2, 0, 1, 2],
            vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        )
        .unwrap();
        let csr = CsrMatrix::from_coo(&coo);
        let a_dense = csr.to_dense();

        let x = vec![0.1_f32, 0.2, 0.3];
        let mut y = vec![0.0_f32; 3];
        csr.spmv(1.0, &x, 0.0, &mut y).unwrap();

        // Compute reference dense result
        let y_ref = dense_matvec(&a_dense, &x, 3, 3);

        // Compute residual
        let residual: f32 =
            y.iter().zip(y_ref.iter()).map(|(a, b)| (a - b).abs()).fold(0.0_f32, f32::max);

        let a_norm = mat_inf_norm(&a_dense, 3, 3);
        let x_norm = vec_inf_norm(&x);

        // Backward error bound: proportional to nnz_per_row * machine epsilon
        let eps = 3.0 * f32::EPSILON * a_norm * x_norm;
        assert!(residual < eps, "Backward error too large: residual={}, bound={}", residual, eps);
    }

    // =========================================================================
    // SOLVER TESTS (dense factorizations)
    // =========================================================================

    use trueno_solve::{
        cholesky, lu_factorize, qr_factorize, svd, symm, syrk, trmm, trsm, DiagonalType,
        TriangularSide,
    };

    // ---- Cholesky tests ----

    #[test]
    fn test_cholesky_spd_solve() {
        // A = [[4, 2], [2, 3]] (symmetric positive definite)
        // b = [1, 2]
        // Solve Ax = b
        let a = [4.0_f32, 2.0, 2.0, 3.0];
        let b = [1.0_f32, 2.0];

        let chol = cholesky(&a, 2).unwrap();
        let x = chol.solve(&b).unwrap();

        // Verify: Ax should equal b
        let ax = dense_matvec(&a, &x, 2, 2);
        for i in 0..2 {
            assert!(
                (ax[i] - b[i]).abs() < 1e-4,
                "Cholesky solve: Ax[{}]={}, b[{}]={}",
                i,
                ax[i],
                i,
                b[i]
            );
        }
    }

    #[test]
    fn test_cholesky_non_spd() {
        // A = [[1, 0], [0, -1]] - not positive definite
        let a = [1.0_f32, 0.0, 0.0, -1.0];
        let result = cholesky(&a, 2);
        assert!(result.is_err(), "Should fail on non-SPD matrix");
    }

    #[test]
    fn test_cholesky_residual() {
        // A = [[9, 3, 3], [3, 5, 2], [3, 2, 4]] (SPD)
        // b = [1, 2, 3]
        // Verify ||Ax - b|| / ||b|| < eps
        let a = [9.0_f32, 3.0, 3.0, 3.0, 5.0, 2.0, 3.0, 2.0, 4.0];
        let b = [1.0_f32, 2.0, 3.0];

        let chol = cholesky(&a, 3).unwrap();
        let x = chol.solve(&b).unwrap();

        let ax = dense_matvec(&a, &x, 3, 3);
        let residual: Vec<f32> = ax.iter().zip(b.iter()).map(|(a, b)| a - b).collect();
        let res_norm = vec_inf_norm(&residual);
        let b_norm = vec_inf_norm(&b);

        let relative_residual = res_norm / b_norm;
        assert!(
            relative_residual < 1e-4,
            "Cholesky residual too large: ||Ax-b||/||b|| = {}",
            relative_residual
        );
    }

    // ---- LU tests ----

    #[test]
    fn test_lu_backward_error() {
        // A = [[2, 1, 1], [4, 3, 3], [8, 7, 9]]
        // b = [4, 10, 24]
        // Expected solution: x = [1, 1, 1]
        let a = [2.0_f32, 1.0, 1.0, 4.0, 3.0, 3.0, 8.0, 7.0, 9.0];
        let b = [4.0_f32, 10.0, 24.0];

        let lu = lu_factorize(&a, 3).unwrap();
        let x = lu.solve(&b).unwrap();

        // Verify backward error: ||Ax - b|| / (||A|| * ||x||)
        let ax = dense_matvec(&a, &x, 3, 3);
        let residual: Vec<f32> = ax.iter().zip(b.iter()).map(|(a, b)| a - b).collect();
        let res_norm = vec_inf_norm(&residual);
        let a_norm = mat_inf_norm(&a, 3, 3);
        let x_norm = vec_inf_norm(&x);

        let backward_error = res_norm / (a_norm * x_norm);
        assert!(backward_error < 1e-5, "LU backward error too large: {}", backward_error);
    }

    #[test]
    fn test_lu_singular_detected() {
        // Singular matrix: [[1, 2], [2, 4]]
        let a = [1.0_f32, 2.0, 2.0, 4.0];
        let result = lu_factorize(&a, 2);
        assert!(result.is_err(), "Should detect singular matrix");
    }

    #[test]
    fn test_lu_solution_residual() {
        // A = [[3, 2], [1, 4]], b = [5, 5]
        let a = [3.0_f32, 2.0, 1.0, 4.0];
        let b = [5.0_f32, 5.0];

        let lu = lu_factorize(&a, 2).unwrap();
        let x = lu.solve(&b).unwrap();

        // Check Ax = b
        let ax = dense_matvec(&a, &x, 2, 2);
        for i in 0..2 {
            assert!(
                (ax[i] - b[i]).abs() < 1e-4,
                "LU residual: Ax[{}]={}, b[{}]={}",
                i,
                ax[i],
                i,
                b[i]
            );
        }
    }

    // ---- QR tests ----

    #[test]
    fn test_qr_orthogonality() {
        // A = [[1, 2], [3, 4], [5, 6]] (3x2, tall-skinny)
        let a = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let qr = qr_factorize(&a, 3, 2).unwrap();
        let q = qr.extract_q(); // 3x3

        // Verify Q^T * Q = I (within tolerance)
        // Compute Q^T * Q properly:
        let mut qt = vec![0.0_f32; 9];
        for i in 0..3 {
            for j in 0..3 {
                qt[i * 3 + j] = q[j * 3 + i]; // transpose
            }
        }
        let qtq = dense_matmul(&qt, &q, 3, 3, 3);

        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (qtq[i * 3 + j] - expected).abs() < 1e-4,
                    "Q^T*Q[{},{}] = {}, expected {}",
                    i,
                    j,
                    qtq[i * 3 + j],
                    expected
                );
            }
        }
    }

    #[test]
    fn test_qr_reconstruction() {
        // Verify Q * R = A (within the first m rows, n cols)
        let a = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // 3x2
        let qr = qr_factorize(&a, 3, 2).unwrap();
        let q = qr.extract_q(); // 3x3
        let r = qr.extract_r(); // 2x2

        // Q (3x3) * [R (2x2); 0] => we need Q * R_full where R_full is 3x2
        let mut r_full = vec![0.0_f32; 3 * 2];
        for i in 0..2 {
            for j in 0..2 {
                r_full[i * 2 + j] = r[i * 2 + j];
            }
        }
        // r_full row 2 is zeros

        let qr_product = dense_matmul(&q, &r_full, 3, 3, 2);

        for i in 0..6 {
            assert!(
                (qr_product[i] - a[i]).abs() < 1e-4,
                "QR reconstruction: (QR)[{}] = {}, A[{}] = {}",
                i,
                qr_product[i],
                i,
                a[i]
            );
        }
    }

    // ---- SVD tests ----

    #[test]
    fn test_svd_singular_values_nonneg_decreasing() {
        let a = [3.0_f32, 2.0, 2.0, 2.0, 3.0, -2.0]; // 2x3
        let result = svd(&a, 2, 3).unwrap();

        for s in &result.sigma {
            assert!(*s >= 0.0, "Singular value should be non-negative: {}", s);
        }

        for i in 1..result.sigma.len() {
            assert!(
                result.sigma[i - 1] >= result.sigma[i] - 1e-6,
                "Singular values not decreasing: s[{}]={} < s[{}]={}",
                i - 1,
                result.sigma[i - 1],
                i,
                result.sigma[i]
            );
        }
    }

    #[test]
    fn test_svd_reconstruction() {
        // A = U * diag(sigma) * V^T
        let a = [1.0_f32, 2.0, 3.0, 4.0]; // 2x2
        let result = svd(&a, 2, 2).unwrap();

        let m = result.m;
        let n = result.n;
        let min_mn = m.min(n);

        // Reconstruct: A_approx = U * diag(sigma) * V^T
        // U is m x m, sigma is min_mn, V^T is n x n
        // First compute U * diag(sigma): m x min_mn
        let mut u_sigma = vec![0.0_f32; m * n];
        for i in 0..m {
            for j in 0..min_mn {
                u_sigma[i * n + j] = result.u[i * m + j] * result.sigma[j];
            }
        }

        // Then u_sigma * V^T: m x n
        let reconstructed = dense_matmul(&u_sigma, &result.vt, m, n, n);

        for i in 0..4 {
            assert!(
                (reconstructed[i] - a[i]).abs() < 1e-3,
                "SVD reconstruction: A_approx[{}]={}, A[{}]={}",
                i,
                reconstructed[i],
                i,
                a[i]
            );
        }
    }

    #[test]
    fn test_svd_orthogonality_u() {
        // For a square matrix, U should be fully orthogonal.
        let a = [1.0_f32, 2.0, 3.0, 4.0]; // 2x2
        let result = svd(&a, 2, 2).unwrap();
        let m = result.m;

        // Compute U^T * U
        let mut ut = vec![0.0_f32; m * m];
        for i in 0..m {
            for j in 0..m {
                ut[i * m + j] = result.u[j * m + i];
            }
        }
        let utu = dense_matmul(&ut, &result.u, m, m, m);

        for i in 0..m {
            for j in 0..m {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (utu[i * m + j] - expected).abs() < 1e-3,
                    "U^T*U[{},{}] = {}, expected {}",
                    i,
                    j,
                    utu[i * m + j],
                    expected
                );
            }
        }

        // For a tall matrix, verify orthogonality of the first min(m,n) columns.
        let a_tall = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // 3x2
        let result_tall = svd(&a_tall, 3, 2).unwrap();
        let m2 = result_tall.m;
        let min_mn = m2.min(result_tall.n);

        // Check that the first min_mn columns of U are orthonormal
        for i in 0..min_mn {
            for j in 0..min_mn {
                let mut dot = 0.0_f32;
                for k in 0..m2 {
                    dot += result_tall.u[k * m2 + i] * result_tall.u[k * m2 + j];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (dot - expected).abs() < 1e-3,
                    "U columns ({},{}) dot = {}, expected {}",
                    i,
                    j,
                    dot,
                    expected
                );
            }
        }
    }

    #[test]
    fn test_svd_singular_values_nonneg() {
        let a = [1.0_f32, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0]; // diagonal 3x3
        let result = svd(&a, 3, 3).unwrap();

        for (i, s) in result.sigma.iter().enumerate() {
            assert!(*s >= -1e-7, "Singular value {} should be non-negative: {}", i, s);
        }

        // For a diagonal matrix, singular values should be |diag| sorted descending
        assert!((result.sigma[0] - 3.0).abs() < 1e-4, "s[0]={}", result.sigma[0]);
        assert!((result.sigma[1] - 2.0).abs() < 1e-4, "s[1]={}", result.sigma[1]);
        assert!((result.sigma[2] - 1.0).abs() < 1e-4, "s[2]={}", result.sigma[2]);
    }

    // =========================================================================
    // BLAS LEVEL 3 TESTS
    // =========================================================================

    // ---- SYRK tests ----

    #[test]
    fn test_syrk_identity() {
        // C = alpha * A * A^T + beta * C with A = I(2), alpha=1, beta=0
        // Result should be I(2) since I * I^T = I
        let a = [1.0_f32, 0.0, 0.0, 1.0]; // 2x2 identity
        let mut c = vec![0.0_f32; 4];
        syrk(&a, &mut c, 2, 2, 1.0, 0.0).unwrap();

        // C should be I
        assert!((c[0] - 1.0).abs() < 1e-5);
        assert!((c[1] - 0.0).abs() < 1e-5);
        assert!((c[2] - 0.0).abs() < 1e-5);
        assert!((c[3] - 1.0).abs() < 1e-5);

        // Must be symmetric
        assert!(is_symmetric(&c, 2, 1e-6));
    }

    #[test]
    fn test_syrk_known_value() {
        // A = [[1, 2], [3, 4]] (2x2, n=2, k=2)
        // C = A * A^T = [[1*1+2*2, 1*3+2*4], [3*1+4*2, 3*3+4*4]]
        //             = [[5, 11], [11, 25]]
        let a = [1.0_f32, 2.0, 3.0, 4.0];
        let mut c = vec![0.0_f32; 4];
        syrk(&a, &mut c, 2, 2, 1.0, 0.0).unwrap();

        assert!((c[0] - 5.0).abs() < 1e-5, "C[0,0]={}, expected 5", c[0]);
        assert!((c[1] - 11.0).abs() < 1e-5, "C[0,1]={}, expected 11", c[1]);
        assert!((c[2] - 11.0).abs() < 1e-5, "C[1,0]={}, expected 11", c[2]);
        assert!((c[3] - 25.0).abs() < 1e-5, "C[1,1]={}, expected 25", c[3]);
    }

    #[test]
    fn test_syrk_symmetry() {
        // A = [[1, 2, 3], [4, 5, 6], [7, 8, 9]] (3x3, n=3, k=3)
        // C = A * A^T must be symmetric
        let a = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let mut c = vec![0.0_f32; 9];
        syrk(&a, &mut c, 3, 3, 1.0, 0.0).unwrap();

        assert!(is_symmetric(&c, 3, 1e-5), "SYRK result must be symmetric: {:?}", c);
    }

    // ---- TRMM tests ----

    #[test]
    fn test_trmm_identity() {
        // B = alpha * I * B = alpha * B
        // With alpha=1.0: result should be B unchanged
        let a = [1.0_f32, 0.0, 0.0, 1.0]; // 2x2 identity (lower triangular)
        let mut b = vec![5.0_f32, 6.0, 7.0, 8.0]; // 2x2

        trmm(&a, &mut b, 2, 2, 1.0).unwrap();

        assert!((b[0] - 5.0).abs() < 1e-5);
        assert!((b[1] - 6.0).abs() < 1e-5);
        assert!((b[2] - 7.0).abs() < 1e-5);
        assert!((b[3] - 8.0).abs() < 1e-5);
    }

    #[test]
    fn test_trmm_lower_triangular() {
        // A = [[2, 0], [3, 4]] (lower triangular)
        // B = [[1, 0], [0, 1]] (identity)
        // Result = alpha * A * B = A (when alpha=1)
        let a = [2.0_f32, 0.0, 3.0, 4.0];
        let mut b = vec![1.0_f32, 0.0, 0.0, 1.0];

        trmm(&a, &mut b, 2, 2, 1.0).unwrap();

        // Row 0: only A[0,0]*B[0,:] contributes (lower triangular)
        // B[0,:] = [2*1 + 0*0, 2*0 + 0*1] = [2, 0]
        // Row 1: A[1,0]*B[0,:] + A[1,1]*B[1,:]
        // B[1,:] = [3*1 + 4*0, 3*0 + 4*1] = [3, 4]
        assert!((b[0] - 2.0).abs() < 1e-5, "b[0,0]={}", b[0]);
        assert!((b[1] - 0.0).abs() < 1e-5, "b[0,1]={}", b[1]);
        assert!((b[2] - 3.0).abs() < 1e-5, "b[1,0]={}", b[2]);
        assert!((b[3] - 4.0).abs() < 1e-5, "b[1,1]={}", b[3]);
    }

    // ---- SYMM tests ----

    #[test]
    fn test_symm_known_product() {
        // A = [[2, 1], [1, 3]] (symmetric)
        // B = [[1, 0], [0, 2]]
        // C = alpha * A * B + beta * C
        // With alpha=1, beta=0:
        // C = A * B = [[2*1+1*0, 2*0+1*2], [1*1+3*0, 1*0+3*2]] = [[2, 2], [1, 6]]
        let a = [2.0_f32, 1.0, 1.0, 3.0];
        let b = [1.0_f32, 0.0, 0.0, 2.0];
        let mut c = vec![0.0_f32; 4];

        symm(&a, &b, &mut c, 2, 2, 1.0, 0.0).unwrap();

        assert!((c[0] - 2.0).abs() < 1e-5, "C[0,0]={}, expected 2", c[0]);
        assert!((c[1] - 2.0).abs() < 1e-5, "C[0,1]={}, expected 2", c[1]);
        assert!((c[2] - 1.0).abs() < 1e-5, "C[1,0]={}, expected 1", c[2]);
        assert!((c[3] - 6.0).abs() < 1e-5, "C[1,1]={}, expected 6", c[3]);
    }

    // ---- TRSM tests ----

    #[test]
    fn test_trsm_backward_error() {
        // Solve A * X = B where A is lower triangular
        // A = [[2, 0], [3, 4]]
        // B = [[4, 6], [11, 16]]
        // Expected: X = [[2, 3], [1.25, 1.75]]
        let a = [2.0_f32, 0.0, 3.0, 4.0];
        let b = [4.0_f32, 6.0, 11.0, 16.0];
        let n = 2;
        let nrhs = 2;

        let result = trsm(&a, &b, n, nrhs, TriangularSide::Lower, DiagonalType::NonUnit).unwrap();

        // Verify A * X = B (backward error check)
        // Compute A * X
        let x = &result.x;
        for col in 0..nrhs {
            for i in 0..n {
                let mut ax_ij = 0.0_f32;
                for j in 0..n {
                    ax_ij += a[i * n + j] * x[j * nrhs + col];
                }
                assert!(
                    (ax_ij - b[i * nrhs + col]).abs() < 1e-4,
                    "TRSM backward error: AX[{},{}]={}, B[{},{}]={}",
                    i,
                    col,
                    ax_ij,
                    i,
                    col,
                    b[i * nrhs + col]
                );
            }
        }
    }

    #[test]
    fn test_trsm_singular_detected() {
        // A = [[1, 0], [2, 0]] - singular (zero on diagonal)
        let a = [1.0_f32, 0.0, 2.0, 0.0];
        let b = [1.0_f32, 2.0];
        let result = trsm(&a, &b, 2, 1, TriangularSide::Lower, DiagonalType::NonUnit);
        assert!(result.is_err(), "Should detect singular triangular matrix");
    }
}
