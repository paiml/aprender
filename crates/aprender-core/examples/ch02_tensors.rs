#![allow(clippy::disallowed_methods)]
//! Chapter 2: Tensor Computation with aprender-compute
//!
//! Demonstrates Vector/Matrix primitives and SIMD-backed operations.
//! Citation: Khudia et al., "FBGEMM," arXiv:2101.05615
//! Contract: contracts/apr-book-ch02-v1.yaml

use aprender::primitives::{Matrix, Vector};

fn main() {
    // Vector dot product (f32 — SIMD-accelerated)
    let a = Vector::from_slice(&[1.0_f32, 2.0, 3.0, 4.0]);
    let b = Vector::from_slice(&[5.0_f32, 6.0, 7.0, 8.0]);
    let dot = a.dot(&b);
    println!("dot(a, b) = {dot}");
    assert!((dot - 70.0).abs() < 1e-4, "dot product contract");

    // Matrix-vector multiplication
    let m = Matrix::from_vec(2, 2, vec![1.0_f32, 2.0, 3.0, 4.0]).expect("valid 2x2 matrix");
    let v = Vector::from_slice(&[1.0_f32, 1.0]);
    let result = m.matvec(&v).expect("matvec should succeed");
    println!("M @ v = [{:.1}, {:.1}]", result[0], result[1]);
    assert!((result[0] - 3.0).abs() < 1e-4, "matvec row 0 contract");
    assert!((result[1] - 7.0).abs() < 1e-4, "matvec row 1 contract");

    // Matrix dimensions contract
    assert_eq!(m.n_rows(), 2, "rows contract");
    assert_eq!(m.n_cols(), 2, "cols contract");

    println!("Chapter 2 contracts: PASSED");
}
