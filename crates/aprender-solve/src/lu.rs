//! LU factorization with partial pivoting.
//!
//! # Contract: solve-lu-v1.yaml
//!
//! PA = LU where L is unit lower triangular, U is upper triangular,
//! P is a permutation matrix (stored as pivot vector).
//!
//! ## Proof obligations
//! - ||PA - LU||_F / ||A||_F < n · u
//! - L is unit lower triangular, U is upper triangular
//! - Solution accuracy: ||Ax - b|| / (||A|| · ||x||) < κ(A) · n · u

use crate::error::SolverError;

/// LU factorization result (in-place: L and U stored in the same matrix).
///
/// After factorization, the matrix `a` contains:
/// - Upper triangle (including diagonal): U
/// - Strict lower triangle: L (unit diagonal implicit)
#[derive(Debug)]
pub struct LuFactorization {
    /// Row dimension.
    pub n: usize,
    /// Factored matrix (L\U packed).
    pub lu: Vec<f32>,
    /// Pivot indices: row i was swapped with row pivot[i].
    pub pivot: Vec<usize>,
}

/// LU factorization with partial pivoting.
///
/// # Contract: solve-lu-v1.yaml / lu_factorization
///
/// Stores L and U in-place: U in upper triangle, L (unit diagonal) in strict lower.
///
/// # Errors
///
/// Returns `SingularMatrix` if a zero pivot is encountered.
#[allow(clippy::cast_precision_loss)]
pub fn lu_factorize(a: &[f32], n: usize) -> Result<LuFactorization, SolverError> {
    if a.len() != n * n {
        return Err(SolverError::NotSquare {
            rows: n,
            cols: a.len() / n.max(1),
        });
    }

    let mut lu = a.to_vec();
    let mut pivot: Vec<usize> = (0..n).collect();

    for k in 0..n {
        // Partial pivoting: find max |a[i][k]| for i >= k
        let mut max_val = lu[k * n + k].abs();
        let mut max_row = k;
        for i in (k + 1)..n {
            let val = lu[i * n + k].abs();
            if val > max_val {
                max_val = val;
                max_row = i;
            }
        }

        if max_val < f32::EPSILON * 1e3 {
            return Err(SolverError::SingularMatrix(k));
        }

        // Swap rows k and max_row
        if max_row != k {
            pivot.swap(k, max_row);
            for j in 0..n {
                lu.swap(k * n + j, max_row * n + j);
            }
        }

        // Eliminate below diagonal
        let pivot_val = lu[k * n + k];
        for i in (k + 1)..n {
            let factor = lu[i * n + k] / pivot_val;
            lu[i * n + k] = factor; // Store L factor

            for j in (k + 1)..n {
                lu[i * n + j] -= factor * lu[k * n + j];
            }
        }
    }

    Ok(LuFactorization { n, lu, pivot })
}

impl LuFactorization {
    /// Solve Ax = b using the LU factorization.
    ///
    /// # Contract: solve-lu-v1.yaml / solution_accuracy
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
        let mut x = vec![0.0f32; n];

        // Apply permutation to b
        for i in 0..n {
            x[i] = b[self.pivot[i]];
        }

        // Forward substitution (L * y = Pb)
        for i in 1..n {
            let mut sum = x[i];
            for j in 0..i {
                sum -= self.lu[i * n + j] * x[j];
            }
            x[i] = sum;
        }

        // Back substitution (U * x = y)
        for i in (0..n).rev() {
            let mut sum = x[i];
            for j in (i + 1)..n {
                sum -= self.lu[i * n + j] * x[j];
            }
            x[i] = sum / self.lu[i * n + i];
        }

        Ok(x)
    }

    /// Extract L matrix (unit lower triangular).
    pub fn extract_l(&self) -> Vec<f32> {
        let n = self.n;
        let mut l = vec![0.0f32; n * n];
        for i in 0..n {
            l[i * n + i] = 1.0; // Unit diagonal
            for j in 0..i {
                l[i * n + j] = self.lu[i * n + j];
            }
        }
        l
    }

    /// Extract U matrix (upper triangular).
    pub fn extract_u(&self) -> Vec<f32> {
        let n = self.n;
        let mut u = vec![0.0f32; n * n];
        for i in 0..n {
            for j in i..n {
                u[i * n + j] = self.lu[i * n + j];
            }
        }
        u
    }

    /// Extract permutation matrix.
    pub fn extract_p(&self) -> Vec<f32> {
        let n = self.n;
        let mut p = vec![0.0f32; n * n];
        for (i, &pi) in self.pivot.iter().enumerate() {
            p[i * n + pi] = 1.0;
        }
        p
    }
}
