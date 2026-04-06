#![cfg_attr(
    test,
    allow(
        clippy::expect_used,
        clippy::unwrap_used,
        clippy::disallowed_methods,
        clippy::float_cmp,
        clippy::panic,
        clippy::useless_vec
    )
)]
//! Dense linear algebra solvers with provable contracts.
//!
//! # Solvers
//!
//! - **LU**: Partial pivoting, PA = LU
//! - **QR**: Householder reflections, A = QR
//! - **SVD**: One-sided Jacobi, A = UΣV^T
//! - **Cholesky**: A = LL^T for positive definite matrices
//!
//! # Example
//!
//! ```
//! use trueno_solve::lu_factorize;
//!
//! // Solve Ax = b where A = [[2, 1], [1, 3]]
//! let a = [2.0, 1.0, 1.0, 3.0_f32];
//! let lu = lu_factorize(&a, 2).unwrap();
//! let x = lu.solve(&[5.0, 7.0]).unwrap();
//! // x ≈ [1.6, 1.8]
//! assert!((x[0] - 1.6).abs() < 1e-5);
//! assert!((x[1] - 1.8).abs() < 1e-5);
//! ```

mod blas3;
mod cholesky;
mod error;
mod lu;
mod qr;
mod svd;
mod trsm;

#[cfg(test)]
mod tests;

pub use blas3::{
    f32_to_f16, gemm_ex, gemm_ex_epilogue, gemm_strided_batched, symm, syr2k, syrk, trmm, Epilogue,
};
pub use cholesky::{cholesky, CholeskyFactorization};
pub use error::SolverError;
pub use lu::{lu_factorize, LuFactorization};
pub use qr::{qr_factorize, QrFactorization};
pub use svd::{svd, SvdResult};
pub use trsm::{trsm, DiagonalType, TriangularSide, TrsmResult};

/// Unified solver trait for factorization-based linear system solving.
///
/// Implemented by `LuFactorization`, `QrFactorization`, and `CholeskyFactorization`.
pub trait Solver {
    /// Solve `Ax = b` using this factorization.
    ///
    /// # Errors
    ///
    /// Returns error on dimension mismatch or singular matrix.
    fn solve(&self, b: &[f32]) -> Result<Vec<f32>, SolverError>;

    /// Matrix dimension (n for n×n systems, or min(m,n) for rectangular).
    fn dimension(&self) -> usize;
}

impl Solver for LuFactorization {
    fn solve(&self, b: &[f32]) -> Result<Vec<f32>, SolverError> {
        self.solve(b)
    }
    fn dimension(&self) -> usize {
        self.n
    }
}

impl Solver for QrFactorization {
    fn solve(&self, b: &[f32]) -> Result<Vec<f32>, SolverError> {
        self.solve(b)
    }
    fn dimension(&self) -> usize {
        self.n
    }
}

impl Solver for CholeskyFactorization {
    fn solve(&self, b: &[f32]) -> Result<Vec<f32>, SolverError> {
        self.solve(b)
    }
    fn dimension(&self) -> usize {
        self.n
    }
}
