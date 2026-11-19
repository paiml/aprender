//! f64 Cholesky factorization in a fixed operation order (SIM3D P3, #7248).
//!
//! Contract: `contracts/simular/f64-dense-solve-v1.yaml`. A = L L^T; every sum
//! runs in ascending `k` with plain `f64` ops (no FMA, no reordering), so the
//! bits are the same on every host and in WASM.

use crate::error::SolverError;

/// f64 Cholesky factor (row-major, only the lower triangle is valid).
#[derive(Debug)]
pub struct CholeskyF64 {
    /// Dimension.
    pub n: usize,
    /// Lower triangular factor, row-major, upper triangle zero.
    pub l: Vec<f64>,
}

/// Factor the symmetric positive definite `a` (row-major `n*n`, lower triangle read).
///
/// # Errors
///
/// `NotSquare` on a bad length, `NotPositiveDefinite(j)` on a non-positive or
/// NaN pivot at column `j`.
pub fn cholesky_f64(a: &[f64], n: usize) -> Result<CholeskyF64, SolverError> {
    if a.len() != n * n {
        return Err(SolverError::NotSquare {
            rows: n,
            cols: a.len() / n.max(1),
        });
    }
    let mut l = vec![0.0f64; n * n];
    for j in 0..n {
        let mut d = a[j * n + j];
        for k in 0..j {
            d -= l[j * n + k] * l[j * n + k];
        }
        if d.is_nan() || d <= 0.0 {
            return Err(SolverError::NotPositiveDefinite(j));
        }
        let ljj = d.sqrt();
        l[j * n + j] = ljj;
        for i in (j + 1)..n {
            let mut s = a[i * n + j];
            for k in 0..j {
                s -= l[i * n + k] * l[j * n + k];
            }
            l[i * n + j] = s / ljj;
        }
    }
    Ok(CholeskyF64 { n, l })
}

impl CholeskyF64 {
    /// Solve `A x = b` (forward, then back substitution, fixed index order).
    ///
    /// # Errors
    ///
    /// `DimensionMismatch` when `b.len() != n`.
    pub fn solve(&self, b: &[f64]) -> Result<Vec<f64>, SolverError> {
        let n = self.n;
        if b.len() != n {
            return Err(SolverError::DimensionMismatch {
                matrix_n: n,
                rhs_len: b.len(),
            });
        }
        let mut x = b.to_vec();
        for i in 0..n {
            let mut s = x[i];
            for j in 0..i {
                s -= self.l[i * n + j] * x[j];
            }
            x[i] = s / self.l[i * n + i];
        }
        for i in (0..n).rev() {
            let mut s = x[i];
            for j in (i + 1)..n {
                s -= self.l[j * n + i] * x[j];
            }
            x[i] = s / self.l[i * n + i];
        }
        Ok(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // L = [[2,0,0],[1,2,0],[1,1,2]] so A = L L^T is exact in f64.
    const A: [f64; 9] = [4.0, 2.0, 2.0, 2.0, 5.0, 3.0, 2.0, 3.0, 6.0];

    #[test]
    fn falsify_sim3d_chol_exact_factor() {
        let f = cholesky_f64(&A, 3).unwrap();
        assert_eq!(f.l, vec![2.0, 0.0, 0.0, 1.0, 2.0, 0.0, 1.0, 1.0, 2.0]);
    }

    #[test]
    fn falsify_sim3d_chol_exact_solve() {
        // b = A * [1,2,3]
        let x = cholesky_f64(&A, 3)
            .unwrap()
            .solve(&[14.0, 21.0, 26.0])
            .unwrap();
        assert_eq!(x, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn falsify_sim3d_chol_rejects_indefinite_and_bad_shape() {
        let r = cholesky_f64(&[1.0, 2.0, 2.0, 1.0], 2);
        assert!(matches!(r, Err(SolverError::NotPositiveDefinite(1))));
        assert!(matches!(
            cholesky_f64(&[f64::NAN], 1),
            Err(SolverError::NotPositiveDefinite(0))
        ));
        assert!(matches!(
            cholesky_f64(&A, 2),
            Err(SolverError::NotSquare { .. })
        ));
        let f = cholesky_f64(&A, 3).unwrap();
        assert!(matches!(
            f.solve(&[1.0]),
            Err(SolverError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn falsify_sim3d_chol_f64_precision_and_repeatable() {
        // Hilbert(5): f32 storage cannot reach this residual.
        let n = 5;
        let h: Vec<f64> = (0..n * n)
            .map(|p| 1.0 / ((p / n + p % n + 1) as f64))
            .collect();
        let b: Vec<f64> = (0..n).map(|i| (i + 1) as f64).collect();
        let x1 = cholesky_f64(&h, n).unwrap().solve(&b).unwrap();
        let x2 = cholesky_f64(&h, n).unwrap().solve(&b).unwrap();
        assert!(x1.iter().zip(&x2).all(|(a, b)| a.to_bits() == b.to_bits()));
        for i in 0..n {
            let r: f64 = (0..n).map(|j| h[i * n + j] * x1[j]).sum::<f64>() - b[i];
            assert!(r.abs() < 1e-9, "residual {r}");
        }
    }
}
