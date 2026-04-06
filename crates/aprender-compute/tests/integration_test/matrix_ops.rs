//! Matrix operations integration tests

use proptest::prelude::*;
use trueno::{Matrix, Vector};

const PROPTEST_CASES: u32 = 50;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(PROPTEST_CASES))]

    /// Integration test: Matrix multiplication
    #[test]
    fn integration_matrix_matmul(
        m in 2usize..10,
        n in 2usize..10,
        p in 2usize..10
    ) {
        let a = Matrix::zeros(m, n);
        let b = Matrix::zeros(n, p);
        let c = a.matmul(&b)?;

        prop_assert_eq!(c.rows(), m);
        prop_assert_eq!(c.cols(), p);

        // Test with identity
        let identity = Matrix::identity(n);
        let a_i = a.matmul(&identity)?;
        prop_assert_eq!(a_i.rows(), a.rows());
        prop_assert_eq!(a_i.cols(), a.cols());
    }

    /// Integration test: Matrix transpose
    #[test]
    fn integration_matrix_transpose(
        m in 2usize..10,
        n in 2usize..10
    ) {
        let a = Matrix::zeros(m, n);
        let t = a.transpose();

        prop_assert_eq!(t.rows(), n);
        prop_assert_eq!(t.cols(), m);

        // Double transpose returns original dimensions
        let tt = t.transpose();
        prop_assert_eq!(tt.rows(), m);
        prop_assert_eq!(tt.cols(), n);
    }

    /// Integration test: Matrix constructors
    #[test]
    fn integration_matrix_constructors(
        rows in 1usize..20,
        cols in 1usize..20
    ) {
        // zeros
        let zeros = Matrix::zeros(rows, cols);
        prop_assert_eq!(zeros.rows(), rows);
        prop_assert_eq!(zeros.cols(), cols);

        // identity (square only)
        let n = rows.min(cols);
        let identity = Matrix::identity(n);
        prop_assert_eq!(identity.rows(), n);
        prop_assert_eq!(identity.cols(), n);

        // from_vec
        let data = vec![1.0f32; rows * cols];
        let from_vec = Matrix::from_vec(rows, cols, data)?;
        prop_assert_eq!(from_vec.rows(), rows);
        prop_assert_eq!(from_vec.cols(), cols);
    }

    /// Integration test: Matrix-vector multiplication (matvec)
    #[test]
    fn integration_matrix_matvec(
        m in 2usize..20,
        n in 2usize..20
    ) {
        // Create m x n matrix and n-dimensional vector
        let matrix_data = (0..m * n).map(|i| (i % 10) as f32).collect();
        let matrix = Matrix::from_vec(m, n, matrix_data)?;
        let vector_data: Vec<f32> = (0..n).map(|i| (i % 5) as f32).collect();
        let vector = Vector::from_slice(&vector_data);

        // Compute matrix-vector product
        let result = matrix.matvec(&vector)?;

        // Result should be m-dimensional
        prop_assert_eq!(result.len(), m);

        // Test with identity matrix: I x v = v
        let identity = Matrix::identity(n);
        let iv = identity.matvec(&vector)?;
        prop_assert_eq!(iv.len(), n);
    }

    /// Integration test: Vector-matrix multiplication (vecmat)
    #[test]
    fn integration_matrix_vecmat(
        m in 2usize..20,
        n in 2usize..20
    ) {
        // Create m-dimensional vector and m x n matrix
        let vector_data: Vec<f32> = (0..m).map(|i| (i % 5) as f32).collect();
        let vector = Vector::from_slice(&vector_data);
        let matrix_data = (0..m * n).map(|i| (i % 10) as f32).collect();
        let matrix = Matrix::from_vec(m, n, matrix_data)?;

        // Compute vector-matrix product
        let result = Matrix::vecmat(&vector, &matrix)?;

        // Result should be n-dimensional
        prop_assert_eq!(result.len(), n);

        // Test with identity matrix: v^T x I = v^T
        let identity = Matrix::identity(m);
        let vi = Matrix::vecmat(&vector, &identity)?;
        prop_assert_eq!(vi.len(), m);
    }
}
