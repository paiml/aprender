//! QR factorization via Householder reflections.
//!
//! # Contract: solve-qr-v1.yaml
//!
//! A = QR where Q is orthogonal (Q^T Q = I), R is upper triangular.
//!
//! ## Proof obligations
//! - ||Q^T Q - I||_F < √n · u
//! - ||A - QR||_F / ||A||_F < √n · u
//! - R is upper triangular

use crate::error::SolverError;

/// QR factorization result.
#[derive(Debug)]
pub struct QrFactorization {
    /// Rows.
    pub m: usize,
    /// Columns.
    pub n: usize,
    /// Householder vectors stored in lower triangle of QR, R in upper.
    pub qr: Vec<f32>,
    /// Householder scalar factors (tau).
    pub tau: Vec<f32>,
}

/// QR factorization via Householder reflections.
///
/// # Contract: solve-qr-v1.yaml / qr_factorization
///
/// # Errors
///
/// Returns error if m < n (not tall-skinny).
#[allow(clippy::cast_precision_loss)]
pub fn qr_factorize(a: &[f32], m: usize, n: usize) -> Result<QrFactorization, SolverError> {
    if m < n {
        return Err(SolverError::QrNotTallSkinny { m, n });
    }
    if a.len() != m * n {
        return Err(SolverError::NotSquare { rows: m, cols: n });
    }

    let mut qr = a.to_vec();
    let min_mn = m.min(n);
    let mut tau = vec![0.0f32; min_mn];

    for k in 0..min_mn {
        // Compute Householder vector for column k, rows k..m
        let mut norm_sq = 0.0f64;
        for i in k..m {
            let v = f64::from(qr[i * n + k]);
            norm_sq += v * v;
        }
        let norm = norm_sq.sqrt();

        if norm < f64::from(f32::EPSILON) {
            tau[k] = 0.0;
            continue;
        }

        // Choose sign to avoid cancellation
        let alpha = f64::from(qr[k * n + k]);
        let beta = if alpha >= 0.0 { -norm } else { norm };

        tau[k] = ((beta - alpha) / beta) as f32;
        let scale = 1.0 / (alpha - beta);

        // Scale the Householder vector
        for i in (k + 1)..m {
            qr[i * n + k] = (f64::from(qr[i * n + k]) * scale) as f32;
        }
        qr[k * n + k] = beta as f32;

        // Apply Householder reflection to remaining columns
        for j in (k + 1)..n {
            let mut dot = f64::from(qr[k * n + j]);
            for i in (k + 1)..m {
                dot += f64::from(qr[i * n + k]) * f64::from(qr[i * n + j]);
            }
            dot *= f64::from(tau[k]);

            qr[k * n + j] -= dot as f32;
            for i in (k + 1)..m {
                qr[i * n + j] -= (f64::from(qr[i * n + k]) * dot) as f32;
            }
        }
    }

    Ok(QrFactorization { m, n, qr, tau })
}

impl QrFactorization {
    /// Extract R (upper triangular, n×n).
    pub fn extract_r(&self) -> Vec<f32> {
        let n = self.n;
        let mut r = vec![0.0f32; n * n];
        for i in 0..n {
            for j in i..n {
                r[i * n + j] = self.qr[i * self.n + j];
            }
        }
        r
    }

    /// Extract Q (m×m orthogonal matrix).
    ///
    /// Builds Q by applying Householder reflectors in reverse.
    pub fn extract_q(&self) -> Vec<f32> {
        let m = self.m;
        let n = self.n;

        // Start with identity
        let mut q = vec![0.0f32; m * m];
        for i in 0..m {
            q[i * m + i] = 1.0;
        }

        let min_mn = m.min(n);
        for k in (0..min_mn).rev() {
            if self.tau[k].abs() < f32::EPSILON {
                continue;
            }

            // Build Householder vector v
            // v[k] = 1.0, v[k+1..m] from qr storage
            for j in k..m {
                let mut dot = f64::from(q[k * m + j]);
                // v[k] = 1.0 implicitly
                for i in (k + 1)..m {
                    let vi = f64::from(self.qr[i * n + k]);
                    dot += vi * f64::from(q[i * m + j]);
                }
                dot *= f64::from(self.tau[k]);

                q[k * m + j] -= dot as f32;
                for i in (k + 1)..m {
                    let vi = f64::from(self.qr[i * n + k]);
                    q[i * m + j] -= (vi * dot) as f32;
                }
            }
        }

        q
    }

    /// Solve Ax = b using QR: x = R^{-1} Q^T b (least-squares for overdetermined).
    ///
    /// # Errors
    ///
    /// Returns error on dimension mismatch.
    pub fn solve(&self, b: &[f32]) -> Result<Vec<f32>, SolverError> {
        if b.len() != self.m {
            return Err(SolverError::DimensionMismatch { matrix_n: self.m, rhs_len: b.len() });
        }

        let m = self.m;
        let n = self.n;

        // Apply Q^T to b
        let mut qtb = b.to_vec();
        let min_mn = m.min(n);
        for k in 0..min_mn {
            if self.tau[k].abs() < f32::EPSILON {
                continue;
            }
            let mut dot = f64::from(qtb[k]);
            for i in (k + 1)..m {
                dot += f64::from(self.qr[i * n + k]) * f64::from(qtb[i]);
            }
            dot *= f64::from(self.tau[k]);
            qtb[k] -= dot as f32;
            for i in (k + 1)..m {
                qtb[i] -= (f64::from(self.qr[i * n + k]) * dot) as f32;
            }
        }

        // Back-substitution with R (upper n×n block)
        let mut x = vec![0.0f32; n];
        for i in (0..n).rev() {
            let mut sum = f64::from(qtb[i]);
            for j in (i + 1)..n {
                sum -= f64::from(self.qr[i * n + j]) * f64::from(x[j]);
            }
            let diag = f64::from(self.qr[i * n + i]);
            if diag.abs() < f64::from(f32::EPSILON) {
                return Err(SolverError::SingularMatrix(i));
            }
            x[i] = (sum / diag) as f32;
        }

        Ok(x)
    }
}
