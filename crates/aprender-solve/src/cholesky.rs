//! Cholesky factorization for symmetric positive definite matrices.
//!
//! # Contract
//!
//! A = L L^T where L is lower triangular with positive diagonal.
//!
//! ## Proof obligations
//! - L is lower triangular
//! - ||A - L L^T||_F / ||A||_F < n · u
//! - L diagonal entries are positive

use crate::error::SolverError;

/// Cholesky factorization result.
#[derive(Debug)]
pub struct CholeskyFactorization {
    /// Dimension.
    pub n: usize,
    /// Lower triangular factor (row-major, only lower triangle is valid).
    pub l: Vec<f32>,
}

/// Cholesky factorization: A = L L^T.
///
/// # Errors
///
/// Returns `NotPositiveDefinite` if a non-positive pivot is encountered.
pub fn cholesky(a: &[f32], n: usize) -> Result<CholeskyFactorization, SolverError> {
    if a.len() != n * n {
        return Err(SolverError::NotSquare {
            rows: n,
            cols: a.len() / n.max(1),
        });
    }

    let mut l = vec![0.0f32; n * n];

    for j in 0..n {
        // Diagonal entry
        let mut sum = f64::from(a[j * n + j]);
        for k in 0..j {
            let ljk = f64::from(l[j * n + k]);
            sum -= ljk * ljk;
        }

        if sum <= 0.0 {
            return Err(SolverError::NotPositiveDefinite(j));
        }
        l[j * n + j] = sum.sqrt() as f32;

        let ljj_inv = 1.0 / f64::from(l[j * n + j]);

        // Below-diagonal entries
        for i in (j + 1)..n {
            let mut sum = f64::from(a[i * n + j]);
            for k in 0..j {
                sum -= f64::from(l[i * n + k]) * f64::from(l[j * n + k]);
            }
            l[i * n + j] = (sum * ljj_inv) as f32;
        }
    }

    Ok(CholeskyFactorization { n, l })
}

impl CholeskyFactorization {
    /// Solve Ax = b using Cholesky factors: L L^T x = b.
    ///
    /// # Errors
    ///
    /// Returns error on dimension mismatch.
    pub fn solve(&self, b: &[f32]) -> Result<Vec<f32>, SolverError> {
        if b.len() != self.n {
            return Err(SolverError::DimensionMismatch {
                matrix_n: self.n,
                rhs_len: b.len(),
            });
        }

        let n = self.n;

        // Forward substitution: L y = b
        let mut y = b.to_vec();
        for i in 0..n {
            let mut sum = f64::from(y[i]);
            for j in 0..i {
                sum -= f64::from(self.l[i * n + j]) * f64::from(y[j]);
            }
            y[i] = (sum / f64::from(self.l[i * n + i])) as f32;
        }

        // Back substitution: L^T x = y
        let mut x = y;
        for i in (0..n).rev() {
            let mut sum = f64::from(x[i]);
            for j in (i + 1)..n {
                sum -= f64::from(self.l[j * n + i]) * f64::from(x[j]);
            }
            x[i] = (sum / f64::from(self.l[i * n + i])) as f32;
        }

        Ok(x)
    }
}
