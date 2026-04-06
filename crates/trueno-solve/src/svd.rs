//! Singular Value Decomposition via one-sided Jacobi rotations.
//!
//! # Contract: solve-svd-v1.yaml
//!
//! A = U Σ V^T where U, V orthogonal, Σ diagonal non-negative decreasing.
//!
//! ## Proof obligations
//! - ||U^T U - I||_F < √m · u
//! - ||V^T V - I||_F < √n · u
//! - σ_i ≥ σ_{i+1} for all i
//! - ||A - U Σ V^T||_F / ||A||_F < √min(m,n) · u

use crate::error::SolverError;

/// SVD result: A = U * diag(sigma) * V^T.
#[derive(Debug)]
pub struct SvdResult {
    /// Left singular vectors (m × m row-major).
    pub u: Vec<f32>,
    /// Singular values (min(m,n) elements, sorted descending).
    pub sigma: Vec<f32>,
    /// Right singular vectors transposed (n × n row-major).
    pub vt: Vec<f32>,
    /// Rows of original matrix.
    pub m: usize,
    /// Columns of original matrix.
    pub n: usize,
}

/// Compute SVD via one-sided Jacobi rotations.
///
/// # Contract: solve-svd-v1.yaml / svd
///
/// Uses iterative Jacobi rotations on A^T A to compute V, then
/// U = A V Σ^{-1}.
///
/// # Errors
///
/// Returns error on dimension mismatch.
#[allow(clippy::cast_precision_loss)]
pub fn svd(a: &[f32], m: usize, n: usize) -> Result<SvdResult, SolverError> {
    if a.len() != m * n {
        return Err(SolverError::SvdDimensionMismatch { m, n });
    }

    let min_mn = m.min(n);
    let mut work = a.to_vec();

    // Initialize V = I(n×n)
    let mut v = vec![0.0f32; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }

    // One-sided Jacobi sweeps
    let tol = f32::EPSILON * (m as f32).sqrt();
    jacobi_sweeps(&mut work, &mut v, m, n, tol);

    // Extract singular values and compute U
    let sigma = extract_singular_values(&work, m, n, min_mn);
    let u = compute_u_matrix(&work, &sigma, m, n, min_mn);

    // Sort and assemble final result
    Ok(assemble_sorted_result(&u, &v, &sigma, m, n, min_mn))
}

/// Run one-sided Jacobi rotation sweeps until convergence.
fn jacobi_sweeps(work: &mut [f32], v: &mut [f32], m: usize, n: usize, tol: f32) {
    let max_sweeps = 100;
    for _sweep in 0..max_sweeps {
        let mut converged = true;
        for p in 0..n {
            for q in (p + 1)..n {
                if apply_jacobi_rotation(work, v, m, n, p, q, tol) {
                    converged = false;
                }
            }
        }
        if converged {
            break;
        }
    }
}

/// Apply a single Jacobi rotation to column pair (p, q).
/// Returns true if a rotation was applied (columns were not orthogonal).
fn apply_jacobi_rotation(
    work: &mut [f32],
    v: &mut [f32],
    m: usize,
    n: usize,
    p: usize,
    q: usize,
    tol: f32,
) -> bool {
    let (app, apq, aqq) = gram_elements(work, m, n, p, q);

    let scale = (app * aqq).sqrt();
    if apq.abs() < f64::from(tol) * scale || scale < f64::EPSILON {
        return false;
    }

    let (c, s) = jacobi_rotation_angle(app, apq, aqq);
    rotate_columns(work, m, n, p, q, c, s);
    rotate_columns(v, n, n, p, q, c, s);
    true
}

/// Compute 2×2 Gram matrix elements for columns p and q.
fn gram_elements(work: &[f32], m: usize, n: usize, p: usize, q: usize) -> (f64, f64, f64) {
    let mut app = 0.0f64;
    let mut apq = 0.0f64;
    let mut aqq = 0.0f64;
    for i in 0..m {
        let wp = f64::from(work[i * n + p]);
        let wq = f64::from(work[i * n + q]);
        app += wp * wp;
        apq += wp * wq;
        aqq += wq * wq;
    }
    (app, apq, aqq)
}

/// Compute Jacobi rotation cosine and sine from Gram matrix elements.
fn jacobi_rotation_angle(app: f64, apq: f64, aqq: f64) -> (f64, f64) {
    let tau = (aqq - app) / (2.0 * apq);
    let t = if tau >= 0.0 {
        1.0 / (tau + (1.0 + tau * tau).sqrt())
    } else {
        -1.0 / (-tau + (1.0 + tau * tau).sqrt())
    };
    let c = 1.0 / (1.0 + t * t).sqrt();
    let s = t * c;
    (c, s)
}

/// Apply Givens rotation to columns p and q of a row-major matrix.
fn rotate_columns(mat: &mut [f32], rows: usize, cols: usize, p: usize, q: usize, c: f64, s: f64) {
    for i in 0..rows {
        let mp = f64::from(mat[i * cols + p]);
        let mq = f64::from(mat[i * cols + q]);
        mat[i * cols + p] = (c * mp - s * mq) as f32;
        mat[i * cols + q] = (s * mp + c * mq) as f32;
    }
}

/// Extract singular values as column norms of the work matrix.
fn extract_singular_values(work: &[f32], m: usize, n: usize, min_mn: usize) -> Vec<f32> {
    let mut sigma = vec![0.0f32; min_mn];
    for j in 0..min_mn {
        let mut norm_sq = 0.0f64;
        for i in 0..m {
            let val = f64::from(work[i * n + j]);
            norm_sq += val * val;
        }
        sigma[j] = norm_sq.sqrt() as f32;
    }
    sigma
}

/// Compute U = AV * Σ^{-1} (normalize columns of work).
fn compute_u_matrix(work: &[f32], sigma: &[f32], m: usize, n: usize, min_mn: usize) -> Vec<f32> {
    let mut u = vec![0.0f32; m * m];
    for j in 0..min_mn {
        if sigma[j] > f32::EPSILON {
            let inv_sigma = 1.0 / sigma[j];
            for i in 0..m {
                u[i * m + j] = work[i * n + j] * inv_sigma;
            }
        }
    }
    for j in min_mn..m {
        u[j * m + j] = 1.0;
    }
    u
}

/// Sort singular values descending and assemble the final SvdResult.
#[allow(clippy::cast_precision_loss)]
fn assemble_sorted_result(
    u: &[f32],
    v: &[f32],
    sigma: &[f32],
    m: usize,
    n: usize,
    min_mn: usize,
) -> SvdResult {
    let mut indices: Vec<usize> = (0..min_mn).collect();
    indices.sort_by(|&a, &b| sigma[b].partial_cmp(&sigma[a]).unwrap_or(std::cmp::Ordering::Equal));

    let mut sigma_sorted = vec![0.0f32; min_mn];
    let mut u_sorted = vec![0.0f32; m * m];
    let mut v_sorted = vec![0.0f32; n * n];

    for (new_j, &old_j) in indices.iter().enumerate() {
        sigma_sorted[new_j] = sigma[old_j];
        for i in 0..m {
            u_sorted[i * m + new_j] = u[i * m + old_j];
        }
        for i in 0..n {
            v_sorted[i * n + new_j] = v[i * n + old_j];
        }
    }

    for j in min_mn..m {
        u_sorted[j * m + j] = 1.0;
    }
    for j in min_mn..n {
        v_sorted[j * n + j] = 1.0;
    }

    // Transpose V to get V^T
    let mut vt = vec![0.0f32; n * n];
    for i in 0..n {
        for j in 0..n {
            vt[i * n + j] = v_sorted[j * n + i];
        }
    }

    SvdResult { u: u_sorted, sigma: sigma_sorted, vt, m, n }
}
