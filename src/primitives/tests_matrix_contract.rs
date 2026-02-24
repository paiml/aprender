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
