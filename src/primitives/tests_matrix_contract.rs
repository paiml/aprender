// =========================================================================
// FALSIFY-MX: Matrix primitives contract (aprender primitives)
//
// Five-Whys (PMAT-354):
//   Why 1: aprender had no inline FALSIFY-MX-* tests for Matrix
//   Why 2: matrix tests exist but lack contract-mapped FALSIFY naming
//   Why 3: no YAML contract for matrix primitives yet
//   Why 4: aprender predates the inline FALSIFY convention
//   Why 5: Matrix algebra was "obviously correct" (basic linear algebra)
//
// References:
//   - Golub & Van Loan (2013) "Matrix Computations"
// =========================================================================

use super::*;

/// FALSIFY-MX-001: Transpose involution: (A^T)^T = A
#[test]
fn falsify_mx_001_transpose_involution() {
    let a = Matrix::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).expect("valid");
    let att = a.transpose().transpose();

    assert_eq!(att.shape(), a.shape(), "FALSIFIED MX-001: shape mismatch");
    for i in 0..2 {
        for j in 0..3 {
            assert!(
                (att.get(i, j) - a.get(i, j)).abs() < 1e-6,
                "FALSIFIED MX-001: (A^T)^T[{i},{j}] != A[{i},{j}]"
            );
        }
    }
}

/// FALSIFY-MX-002: Transpose swaps shape: (m×n)^T = (n×m)
#[test]
fn falsify_mx_002_transpose_swaps_shape() {
    let a = Matrix::from_vec(3, 5, vec![0.0; 15]).expect("valid");
    let at = a.transpose();

    assert_eq!(
        at.shape(),
        (5, 3),
        "FALSIFIED MX-002: transpose shape={:?}, expected (5,3)",
        at.shape()
    );
}

/// FALSIFY-MX-003: Matmul shape: (m×k) * (k×n) = (m×n)
#[test]
fn falsify_mx_003_matmul_shape() {
    let a = Matrix::from_vec(2, 3, vec![1.0; 6]).expect("valid");
    let b = Matrix::from_vec(3, 4, vec![1.0; 12]).expect("valid");
    let c = a.matmul(&b).expect("compatible dims");

    assert_eq!(
        c.shape(),
        (2, 4),
        "FALSIFIED MX-003: (2x3)*(3x4) shape={:?}, expected (2,4)",
        c.shape()
    );
}

/// FALSIFY-MX-004: Identity matmul: A * I = A
#[test]
fn falsify_mx_004_identity_matmul() {
    let a =
        Matrix::from_vec(3, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]).expect("valid");
    let eye = Matrix::eye(3);
    let result = a.matmul(&eye).expect("compatible dims");

    for i in 0..3 {
        for j in 0..3 {
            assert!(
                (result.get(i, j) - a.get(i, j)).abs() < 1e-5,
                "FALSIFIED MX-004: (A*I)[{i},{j}]={} != A[{i},{j}]={}",
                result.get(i, j),
                a.get(i, j)
            );
        }
    }
}

mod matrix_proptest_falsify {
    use super::*;
    use proptest::prelude::*;

    /// FALSIFY-MX-001-prop: Transpose involution for random matrices
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(30))]

        #[test]
        fn falsify_mx_001_prop_transpose_involution(
            rows in 1..=8usize,
            cols in 1..=8usize,
            seed in 0..500u32,
        ) {
            let data: Vec<f32> = (0..rows * cols)
                .map(|i| ((i as f32 + seed as f32) * 0.37).sin() * 10.0)
                .collect();
            let a = Matrix::from_vec(rows, cols, data).expect("valid");
            let att = a.transpose().transpose();

            prop_assert_eq!(att.shape(), a.shape(), "FALSIFIED MX-001-prop: shape mismatch");
            for i in 0..rows {
                for j in 0..cols {
                    prop_assert!(
                        (att.get(i, j) - a.get(i, j)).abs() < 1e-5,
                        "FALSIFIED MX-001-prop: (A^T)^T[{},{}] != A[{},{}]",
                        i, j, i, j
                    );
                }
            }
        }
    }

    /// FALSIFY-MX-004-prop: Identity matmul for random square matrices
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(20))]

        #[test]
        fn falsify_mx_004_prop_identity_matmul(
            n in 1..=6usize,
            seed in 0..500u32,
        ) {
            let data: Vec<f32> = (0..n * n)
                .map(|i| ((i as f32 + seed as f32) * 0.37).sin() * 10.0)
                .collect();
            let a = Matrix::from_vec(n, n, data).expect("valid");
            let eye = Matrix::eye(n);
            let result = a.matmul(&eye).expect("compatible");

            for i in 0..n {
                for j in 0..n {
                    prop_assert!(
                        (result.get(i, j) - a.get(i, j)).abs() < 1e-3,
                        "FALSIFIED MX-004-prop: (A*I)[{},{}] != A[{},{}]",
                        i, j, i, j
                    );
                }
            }
        }
    }
}
