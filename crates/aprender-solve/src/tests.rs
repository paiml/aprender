//! Solver tests — provable contracts and falsification.

use crate::blas3::{gemm_ex_epilogue, Epilogue};
use crate::cholesky::cholesky;
use crate::lu::lu_factorize;
use crate::qr::qr_factorize;
use crate::svd::svd;
use crate::trsm::{trsm, DiagonalType, TriangularSide};

// ============================================================================
// LU factorization tests
// ============================================================================

#[test]
fn test_lu_identity() {
    let a = [1.0, 0.0, 0.0, 1.0_f32];
    let lu = lu_factorize(&a, 2).expect("lu ok");
    let x = lu.solve(&[3.0, 7.0]).expect("solve ok");
    assert!((x[0] - 3.0).abs() < 1e-5);
    assert!((x[1] - 7.0).abs() < 1e-5);
}

#[test]
fn test_lu_2x2() {
    // A = [[2, 1], [1, 3]], b = [5, 7]
    // x = [1.6, 1.8]
    let a = [2.0, 1.0, 1.0, 3.0_f32];
    let lu = lu_factorize(&a, 2).expect("lu ok");
    let x = lu.solve(&[5.0, 7.0]).expect("solve ok");
    assert!((x[0] - 1.6).abs() < 1e-5, "x[0]={}", x[0]);
    assert!((x[1] - 1.8).abs() < 1e-5, "x[1]={}", x[1]);
}

#[test]
fn test_lu_3x3() {
    // A = [[2, -1, 0], [-1, 2, -1], [0, -1, 2]] (tridiagonal)
    let a = [2.0, -1.0, 0.0, -1.0, 2.0, -1.0, 0.0, -1.0, 2.0_f32];
    let lu = lu_factorize(&a, 3).expect("lu ok");
    let b = [1.0, 0.0, 1.0_f32];
    let x = lu.solve(&b).expect("solve ok");

    // Verify Ax ≈ b
    for i in 0..3 {
        let mut ax_i = 0.0f32;
        for j in 0..3 {
            ax_i += a[i * 3 + j] * x[j];
        }
        assert!(
            (ax_i - b[i]).abs() < 1e-4,
            "Ax[{i}]={ax_i}, b[{i}]={}",
            b[i]
        );
    }
}

#[test]
fn test_lu_backward_error() {
    let a = [4.0, 2.0, 1.0, 2.0, 5.0, 3.0, 1.0, 3.0, 6.0_f32];
    let lu = lu_factorize(&a, 3).expect("lu ok");

    let l = lu.extract_l();
    let u = lu.extract_u();
    let p = lu.extract_p();

    // Verify PA ≈ LU
    let n = 3;
    let mut pa = vec![0.0f32; n * n];
    let mut lu_product = vec![0.0f32; n * n];

    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                pa[i * n + j] += p[i * n + k] * a[k * n + j];
                lu_product[i * n + j] += l[i * n + k] * u[k * n + j];
            }
        }
    }

    let mut max_err = 0.0f32;
    for i in 0..n * n {
        max_err = max_err.max((pa[i] - lu_product[i]).abs());
    }
    assert!(max_err < 1e-4, "PA ≠ LU, max_err={max_err}");
}

#[test]
fn test_lu_singular_detected() {
    let a = [1.0, 2.0, 2.0, 4.0_f32]; // Singular: row2 = 2*row1
    assert!(lu_factorize(&a, 2).is_err());
}

// ============================================================================
// QR factorization tests
// ============================================================================

#[test]
fn test_qr_identity() {
    let a = [1.0, 0.0, 0.0, 1.0_f32];
    let qr = qr_factorize(&a, 2, 2).expect("qr ok");
    let r = qr.extract_r();

    // R should be ±I (signs may flip)
    assert!(r[0].abs() > 0.9);
    assert!(r[3].abs() > 0.9);
}

#[test]
fn test_qr_orthogonality() {
    let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0_f32]; // 3×2
    let qr = qr_factorize(&a, 3, 2).expect("qr ok");
    let q = qr.extract_q();

    // Check Q^T Q ≈ I
    let m = 3;
    let mut qtq = vec![0.0f32; m * m];
    for i in 0..m {
        for j in 0..m {
            for k in 0..m {
                qtq[i * m + j] += q[k * m + i] * q[k * m + j];
            }
        }
    }

    for i in 0..m {
        for j in 0..m {
            let expected = if i == j { 1.0 } else { 0.0 };
            assert!(
                (qtq[i * m + j] - expected).abs() < 1e-4,
                "Q^TQ[{i},{j}] = {}, expected {expected}",
                qtq[i * m + j]
            );
        }
    }
}

#[test]
fn test_qr_reconstruction() {
    let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0_f32]; // 3×2
    let m = 3;
    let n = 2;
    let qr = qr_factorize(&a, m, n).expect("qr ok");
    let q = qr.extract_q();
    let r = qr.extract_r();

    // QR should reconstruct A
    for i in 0..m {
        for j in 0..n {
            let mut qr_ij = 0.0f32;
            for k in 0..n {
                qr_ij += q[i * m + k] * r[k * n + j];
            }
            assert!(
                (qr_ij - a[i * n + j]).abs() < 1e-3,
                "QR[{i},{j}]={qr_ij}, A[{i},{j}]={}",
                a[i * n + j]
            );
        }
    }
}

#[test]
fn test_qr_solve_least_squares() {
    // Overdetermined system: 3×2
    let a = [1.0, 1.0, 1.0, 2.0, 1.0, 3.0_f32];
    let b = [1.0, 2.0, 3.0_f32]; // y ≈ x (least squares fit)

    let qr = qr_factorize(&a, 3, 2).expect("qr ok");
    let x = qr.solve(&b).expect("solve ok");

    // Verify the residual is reasonable
    let mut residual = 0.0f32;
    for i in 0..3 {
        let mut ax = 0.0f32;
        for j in 0..2 {
            ax += a[i * 2 + j] * x[j];
        }
        residual += (ax - b[i]).powi(2);
    }
    assert!(
        residual.sqrt() < 1.0,
        "Residual too large: {}",
        residual.sqrt()
    );
}

// ============================================================================
// SVD tests
// ============================================================================

#[test]
fn test_svd_2x2_identity() {
    let a = [1.0, 0.0, 0.0, 1.0_f32];
    let result = svd(&a, 2, 2).expect("svd ok");

    // Singular values should be [1, 1]
    assert!(
        (result.sigma[0] - 1.0).abs() < 1e-4,
        "σ[0]={}",
        result.sigma[0]
    );
    assert!(
        (result.sigma[1] - 1.0).abs() < 1e-4,
        "σ[1]={}",
        result.sigma[1]
    );
}

#[test]
fn test_svd_singular_values_nonneg_decreasing() {
    let a = [3.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0_f32];
    let result = svd(&a, 3, 3).expect("svd ok");

    for s in &result.sigma {
        assert!(*s >= -1e-6, "Negative singular value: {s}");
    }
    for i in 0..result.sigma.len() - 1 {
        assert!(
            result.sigma[i] >= result.sigma[i + 1] - 1e-6,
            "σ[{}]={} < σ[{}]={}",
            i,
            result.sigma[i],
            i + 1,
            result.sigma[i + 1]
        );
    }
}

#[test]
fn test_svd_reconstruction() {
    let a = [1.0, 2.0, 3.0, 4.0_f32]; // 2×2
    let m = 2;
    let n = 2;
    let result = svd(&a, m, n).expect("svd ok");

    // Reconstruct: A ≈ U * diag(σ) * V^T
    let min_mn = m.min(n);
    let mut recon = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f64;
            for k in 0..min_mn {
                sum += f64::from(result.u[i * m + k])
                    * f64::from(result.sigma[k])
                    * f64::from(result.vt[k * n + j]);
            }
            recon[i * n + j] = sum as f32;
        }
    }

    for i in 0..m * n {
        assert!(
            (recon[i] - a[i]).abs() < 1e-3,
            "Reconstruction error at {i}: recon={}, orig={}",
            recon[i],
            a[i]
        );
    }
}

#[test]
fn test_svd_orthogonality_u() {
    let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0_f32]; // 3×2
    let result = svd(&a, 3, 2).expect("svd ok");

    let m = result.m;
    let min_mn = result.m.min(result.n);
    // Check first min_mn columns of U are orthonormal
    for i in 0..min_mn {
        for j in 0..min_mn {
            let mut dot = 0.0f64;
            for k in 0..m {
                dot += f64::from(result.u[k * m + i]) * f64::from(result.u[k * m + j]);
            }
            let expected = if i == j { 1.0 } else { 0.0 };
            assert!(
                (dot - expected).abs() < 1e-3,
                "U orthogonality: U^T U[{i},{j}]={dot}, expected {expected}"
            );
        }
    }
}

// ============================================================================
// Cholesky tests
// ============================================================================

#[test]
fn test_cholesky_2x2() {
    // A = [[4, 2], [2, 3]] (positive definite)
    let a = [4.0, 2.0, 2.0, 3.0_f32];
    let chol = cholesky(&a, 2).expect("cholesky ok");

    // L L^T should reconstruct A
    let n = 2;
    let mut recon = vec![0.0f32; 4];
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                recon[i * n + j] += chol.l[i * n + k] * chol.l[j * n + k];
            }
        }
    }

    for i in 0..4 {
        assert!(
            (recon[i] - a[i]).abs() < 1e-5,
            "LL^T[{i}]={}, A[{i}]={}",
            recon[i],
            a[i]
        );
    }
}

#[test]
fn test_cholesky_solve() {
    let a = [4.0, 2.0, 2.0, 3.0_f32];
    let chol = cholesky(&a, 2).expect("cholesky ok");
    let x = chol.solve(&[8.0, 7.0]).expect("solve ok");

    // Verify Ax ≈ b
    let ax0 = 4.0 * x[0] + 2.0 * x[1];
    let ax1 = 2.0 * x[0] + 3.0 * x[1];
    assert!((ax0 - 8.0).abs() < 1e-4);
    assert!((ax1 - 7.0).abs() < 1e-4);
}

#[test]
fn test_cholesky_not_positive_definite() {
    // Not positive definite: eigenvalues include negative
    let a = [1.0, 3.0, 3.0, 1.0_f32];
    assert!(cholesky(&a, 2).is_err());
}

#[test]
fn test_cholesky_3x3() {
    // A = [[4, 12, -16], [12, 37, -43], [-16, -43, 98]]
    let a = [4.0, 12.0, -16.0, 12.0, 37.0, -43.0, -16.0, -43.0, 98.0_f32];
    let chol = cholesky(&a, 3).expect("cholesky ok");

    // Reconstruct
    let n = 3;
    let mut recon = vec![0.0f32; 9];
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                recon[i * n + j] += chol.l[i * n + k] * chol.l[j * n + k];
            }
        }
    }

    for i in 0..9 {
        assert!(
            (recon[i] - a[i]).abs() < 1e-3,
            "LL^T[{i}]={}, A[{i}]={}",
            recon[i],
            a[i]
        );
    }
}

// ============================================================================
// Property-based tests
// ============================================================================

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn prop_lu_solve_residual(
            a00 in -10.0f32..10.0,
            a01 in -10.0f32..10.0,
            a10 in -10.0f32..10.0,
            a11 in -10.0f32..10.0,
            b0 in -10.0f32..10.0,
            b1 in -10.0f32..10.0,
        ) {
            let a = [a00, a01, a10, a11];
            // Skip ill-conditioned matrices: f32 loses ~log10(cond) digits of precision,
            // so a 1e-3 relative residual bound requires cond(A) < ~1e4. For a 2x2 matrix
            // with entries in [-10,10], cond(A) ≈ ||A||_F^2 / |det(A)|; require the ratio
            // |det(A)| / ||A||_F^2 to exceed 1e-4 (equivalent cond < 1e4).
            let det = a00 * a11 - a01 * a10;
            let fro_sq = a00*a00 + a01*a01 + a10*a10 + a11*a11;
            prop_assume!(fro_sq > 1e-6 && det.abs() / fro_sq > 1e-4);
            if let Ok(lu) = lu_factorize(&a, 2) {
                if let Ok(x) = lu.solve(&[b0, b1]) {
                    // Verify ||Ax - b|| is small
                    let r0 = a00 * x[0] + a01 * x[1] - b0;
                    let r1 = a10 * x[0] + a11 * x[1] - b1;
                    let residual = (r0 * r0 + r1 * r1).sqrt();
                    let b_norm = (b0 * b0 + b1 * b1).sqrt() + 1e-10;
                    prop_assert!(residual / b_norm < 1e-3, "Residual too large: {residual}");
                }
            }
        }

        #[test]
        fn prop_svd_values_nonneg(
            a00 in -5.0f32..5.0,
            a01 in -5.0f32..5.0,
            a10 in -5.0f32..5.0,
            a11 in -5.0f32..5.0,
        ) {
            let a = [a00, a01, a10, a11];
            if let Ok(result) = svd(&a, 2, 2) {
                for s in &result.sigma {
                    prop_assert!(*s >= -1e-5, "Negative singular value: {s}");
                }
                if result.sigma.len() >= 2 {
                    prop_assert!(result.sigma[0] >= result.sigma[1] - 1e-5);
                }
            }
        }
    }
}

// ============================================================================
// TRSM (Triangular Solve AX = B)
// ============================================================================

#[test]
fn test_trsm_lower_2x2() -> Result<(), Box<dyn std::error::Error>> {
    // L = [[2, 0], [3, 4]], B = [[1], [2]]
    // Lx = b => x0 = 1/2, x1 = (2 - 3*0.5)/4 = 0.125
    let l = [2.0, 0.0, 3.0, 4.0_f32];
    let b = [1.0, 2.0_f32];
    let result = trsm(&l, &b, 2, 1, TriangularSide::Lower, DiagonalType::NonUnit)?;
    assert!((result.x[0] - 0.5).abs() < 1e-5);
    assert!((result.x[1] - 0.125).abs() < 1e-5);
    Ok(())
}

#[test]
fn test_trsm_upper_2x2() -> Result<(), Box<dyn std::error::Error>> {
    // U = [[3, 1], [0, 2]], B = [[5], [4]]
    // x1 = 4/2 = 2, x0 = (5 - 1*2)/3 = 1
    let u = [3.0, 1.0, 0.0, 2.0_f32];
    let b = [5.0, 4.0_f32];
    let result = trsm(&u, &b, 2, 1, TriangularSide::Upper, DiagonalType::NonUnit)?;
    assert!((result.x[0] - 1.0).abs() < 1e-5);
    assert!((result.x[1] - 2.0).abs() < 1e-5);
    Ok(())
}

#[test]
fn test_trsm_unit_diagonal() -> Result<(), Box<dyn std::error::Error>> {
    // L = [[1, 0], [3, 1]], B = [[1], [5]]  (unit diagonal)
    // x0 = 1, x1 = 5 - 3*1 = 2
    let l = [1.0, 0.0, 3.0, 1.0_f32];
    let b = [1.0, 5.0_f32];
    let result = trsm(&l, &b, 2, 1, TriangularSide::Lower, DiagonalType::Unit)?;
    assert!((result.x[0] - 1.0).abs() < 1e-5);
    assert!((result.x[1] - 2.0).abs() < 1e-5);
    Ok(())
}

#[test]
fn test_trsm_multiple_rhs() -> Result<(), Box<dyn std::error::Error>> {
    // L = [[2, 0], [1, 3]], B = [[2, 4], [5, 7]]  (n=2, nrhs=2)
    let l = [2.0, 0.0, 1.0, 3.0_f32];
    let b = [2.0, 4.0, 5.0, 7.0_f32];
    let result = trsm(&l, &b, 2, 2, TriangularSide::Lower, DiagonalType::NonUnit)?;
    // Column 0: x0=2/2=1, x1=(5-1*1)/3=4/3
    assert!((result.x[0] - 1.0).abs() < 1e-5);
    assert!((result.x[2] - 4.0 / 3.0).abs() < 1e-4);
    // Column 1: x0=4/2=2, x1=(7-1*2)/3=5/3
    assert!((result.x[1] - 2.0).abs() < 1e-5);
    assert!((result.x[3] - 5.0 / 3.0).abs() < 1e-4);
    Ok(())
}

#[test]
fn test_trsm_backward_error() -> Result<(), Box<dyn std::error::Error>> {
    // Verify ||AX - B||/||B|| is small
    let l = [3.0, 0.0, 0.0, 2.0, 5.0, 0.0, 1.0, 4.0, 6.0_f32];
    let b = [1.0, 2.0, 3.0_f32];
    let result = trsm(&l, &b, 3, 1, TriangularSide::Lower, DiagonalType::NonUnit)?;

    // Compute AX
    let n = 3;
    let mut ax = vec![0.0f32; n];
    for i in 0..n {
        for j in 0..n {
            ax[i] += l[i * n + j] * result.x[j];
        }
    }
    // ||AX - B|| / ||B||
    let mut num = 0.0f32;
    let mut den = 0.0f32;
    for i in 0..n {
        num += (ax[i] - b[i]).powi(2);
        den += b[i].powi(2);
    }
    let rel_err = (num / den).sqrt();
    assert!(rel_err < 1e-5, "TRSM backward error {rel_err} too large");
    Ok(())
}

#[test]
fn test_trsm_singular_detected() {
    let a = [1.0, 0.0, 0.0, 0.0_f32]; // zero diagonal at (1,1)
    let b = [1.0, 2.0_f32];
    let result = trsm(&a, &b, 2, 1, TriangularSide::Lower, DiagonalType::NonUnit);
    assert!(result.is_err());
}

// ============================================================================
// BLAS Level-3 tests (Contract: blas-level3-v1.yaml)
// ============================================================================

use crate::blas3::{symm, syrk, trmm};

#[test]
fn test_syrk_identity() -> Result<(), Box<dyn std::error::Error>> {
    // A = I (2×2), C = α·I·Iᵀ + β·0 = α·I
    let a = [1.0, 0.0, 0.0, 1.0_f32];
    let mut c = [0.0_f32; 4];
    syrk(&a, &mut c, 2, 2, 1.0, 0.0)?;
    assert!((c[0] - 1.0).abs() < 1e-6);
    assert!((c[3] - 1.0).abs() < 1e-6);
    assert!(c[1].abs() < 1e-6);
    assert!(c[2].abs() < 1e-6);
    Ok(())
}

#[test]
fn test_syrk_known_value() -> Result<(), Box<dyn std::error::Error>> {
    // A = [[1, 2], [3, 4]], C = A·Aᵀ = [[5, 11], [11, 25]]
    let a = [1.0, 2.0, 3.0, 4.0_f32];
    let mut c = [0.0_f32; 4];
    syrk(&a, &mut c, 2, 2, 1.0, 0.0)?;
    assert!((c[0] - 5.0).abs() < 1e-4);
    assert!((c[1] - 11.0).abs() < 1e-4);
    assert!((c[2] - 11.0).abs() < 1e-4); // symmetric
    assert!((c[3] - 25.0).abs() < 1e-4);
    Ok(())
}

#[test]
fn test_syrk_symmetry() -> Result<(), Box<dyn std::error::Error>> {
    // Result should always be symmetric
    let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0_f32]; // 2×3
    let mut c = [0.0_f32; 4];
    syrk(&a, &mut c, 2, 3, 2.0, 0.0)?;
    assert!(
        (c[1] - c[2]).abs() < 1e-6,
        "Not symmetric: c[1]={}, c[2]={}",
        c[1],
        c[2]
    );
    Ok(())
}

#[test]
fn test_syrk_alpha_beta() -> Result<(), Box<dyn std::error::Error>> {
    let a = [1.0, 0.0, 0.0, 1.0_f32];
    let mut c = [10.0, 0.0, 0.0, 10.0_f32];
    // C = 2·I·Iᵀ + 0.5·C = [[2+5, 0], [0, 2+5]] = [[7, 0], [0, 7]]
    syrk(&a, &mut c, 2, 2, 2.0, 0.5)?;
    assert!((c[0] - 7.0).abs() < 1e-5);
    assert!((c[3] - 7.0).abs() < 1e-5);
    Ok(())
}

#[test]
fn test_trmm_identity() -> Result<(), Box<dyn std::error::Error>> {
    // A = I (lower triangular), B should be unchanged (scaled by α)
    let a = [1.0, 0.0, 0.0, 1.0_f32];
    let mut b = [3.0, 4.0, 5.0, 6.0_f32]; // 2×2
    trmm(&a, &mut b, 2, 2, 1.0)?;
    assert!((b[0] - 3.0).abs() < 1e-6);
    assert!((b[1] - 4.0).abs() < 1e-6);
    assert!((b[2] - 5.0).abs() < 1e-6);
    assert!((b[3] - 6.0).abs() < 1e-6);
    Ok(())
}

#[test]
fn test_trmm_lower_triangular() -> Result<(), Box<dyn std::error::Error>> {
    // A = [[2, 0], [3, 4]], B = [[1], [1]]
    // A·B = [[2], [7]]
    let a = [2.0, 0.0, 3.0, 4.0_f32];
    let mut b = [1.0, 1.0_f32]; // 2×1
    trmm(&a, &mut b, 2, 1, 1.0)?;
    assert!((b[0] - 2.0).abs() < 1e-5);
    assert!((b[1] - 7.0).abs() < 1e-5);
    Ok(())
}

#[test]
fn test_trmm_alpha_scaling() -> Result<(), Box<dyn std::error::Error>> {
    let a = [1.0, 0.0, 0.0, 1.0_f32];
    let mut b = [5.0, 10.0_f32];
    trmm(&a, &mut b, 2, 1, 3.0)?;
    assert!((b[0] - 15.0).abs() < 1e-5);
    assert!((b[1] - 30.0).abs() < 1e-5);
    Ok(())
}

#[test]
fn test_symm_known_product() -> Result<(), Box<dyn std::error::Error>> {
    // A = [[1, 2], [2, 3]] (symmetric), B = [[1, 0], [0, 1]]
    // C = A·B = A
    let a = [1.0, 2.0, 2.0, 3.0_f32];
    let b = [1.0, 0.0, 0.0, 1.0_f32];
    let mut c = [0.0_f32; 4];
    symm(&a, &b, &mut c, 2, 2, 1.0, 0.0)?;
    assert!((c[0] - 1.0).abs() < 1e-5);
    assert!((c[1] - 2.0).abs() < 1e-5);
    assert!((c[2] - 2.0).abs() < 1e-5);
    assert!((c[3] - 3.0).abs() < 1e-5);
    Ok(())
}

#[test]
fn test_symm_alpha_beta() -> Result<(), Box<dyn std::error::Error>> {
    let a = [1.0, 0.0, 0.0, 1.0_f32]; // I
    let b = [2.0, 3.0_f32]; // 2×1
    let mut c = [10.0, 20.0_f32]; // 2×1
                                  // C = 2·I·B + 0.5·C = [4+5, 6+10] = [9, 16]
    symm(&a, &b, &mut c, 2, 1, 2.0, 0.5)?;
    assert!((c[0] - 9.0).abs() < 1e-5);
    assert!((c[1] - 16.0).abs() < 1e-5);
    Ok(())
}

#[test]
fn test_blas3_dimension_mismatch() {
    let a = [1.0_f32; 4]; // 2×2
    let mut c = [0.0_f32; 3]; // wrong size
    assert!(syrk(&a, &mut c, 2, 2, 1.0, 0.0).is_err());
}

use crate::blas3::syr2k;

#[test]
fn test_syr2k_identity() -> Result<(), Box<dyn std::error::Error>> {
    // A = B = I, C = α·(I·Iᵀ + I·Iᵀ) + β·0 = 2α·I
    let a = [1.0, 0.0, 0.0, 1.0_f32];
    let b = [1.0, 0.0, 0.0, 1.0_f32];
    let mut c = [0.0_f32; 4];
    syr2k(&a, &b, &mut c, 2, 2, 1.0, 0.0)?;
    assert!((c[0] - 2.0).abs() < 1e-5);
    assert!((c[3] - 2.0).abs() < 1e-5);
    assert!(c[1].abs() < 1e-5);
    assert!(c[2].abs() < 1e-5);
    Ok(())
}

#[test]
fn test_syr2k_known_value() -> Result<(), Box<dyn std::error::Error>> {
    // A = [[1, 2]], B = [[3, 4]] (1×2 each)
    // C = A·Bᵀ + B·Aᵀ = [1*3+2*4] + [3*1+4*2] = [11] + [11] = [22]
    let a = [1.0, 2.0_f32];
    let b = [3.0, 4.0_f32];
    let mut c = [0.0_f32; 1];
    syr2k(&a, &b, &mut c, 1, 2, 1.0, 0.0)?;
    assert!((c[0] - 22.0).abs() < 1e-4, "got {}", c[0]);
    Ok(())
}

#[test]
fn test_syr2k_symmetry() -> Result<(), Box<dyn std::error::Error>> {
    let a = [1.0, 2.0, 3.0, 4.0_f32]; // 2×2
    let b = [5.0, 6.0, 7.0, 8.0_f32]; // 2×2
    let mut c = [0.0_f32; 4];
    syr2k(&a, &b, &mut c, 2, 2, 1.0, 0.0)?;
    assert!(
        (c[1] - c[2]).abs() < 1e-5,
        "Not symmetric: c[1]={}, c[2]={}",
        c[1],
        c[2]
    );
    Ok(())
}

#[test]
fn test_syr2k_alpha_beta() -> Result<(), Box<dyn std::error::Error>> {
    let a = [1.0, 0.0, 0.0, 1.0_f32];
    let b = [1.0, 0.0, 0.0, 1.0_f32];
    let mut c = [10.0, 0.0, 0.0, 10.0_f32];
    // C = 2·(I·Iᵀ + I·Iᵀ) + 0.5·C = [[4+5, 0], [0, 4+5]] = [[9, 0], [0, 9]]
    syr2k(&a, &b, &mut c, 2, 2, 2.0, 0.5)?;
    assert!((c[0] - 9.0).abs() < 1e-5);
    assert!((c[3] - 9.0).abs() < 1e-5);
    Ok(())
}

// ============================================================================
// gemmEx (mixed-precision) tests
// ============================================================================

use crate::blas3::{f32_to_f16, gemm_ex, gemm_strided_batched};

#[test]
fn test_gemm_ex_identity() -> Result<(), Box<dyn std::error::Error>> {
    // A = 2×2 identity in f16, B = [[1,2],[3,4]] in f16
    let a = [
        f32_to_f16(1.0),
        f32_to_f16(0.0),
        f32_to_f16(0.0),
        f32_to_f16(1.0),
    ];
    let b = [
        f32_to_f16(1.0),
        f32_to_f16(2.0),
        f32_to_f16(3.0),
        f32_to_f16(4.0),
    ];
    let mut c = [0.0_f32; 4];
    gemm_ex(&a, &b, &mut c, 2, 2, 2, 1.0, 0.0)?;
    assert!((c[0] - 1.0).abs() < 1e-3);
    assert!((c[1] - 2.0).abs() < 1e-3);
    assert!((c[2] - 3.0).abs() < 1e-3);
    assert!((c[3] - 4.0).abs() < 1e-3);
    Ok(())
}

#[test]
fn test_gemm_ex_alpha_beta() -> Result<(), Box<dyn std::error::Error>> {
    // C = 2·A·B + 0.5·C where A=B=I, C=[10,0,0,10]
    let i16 = [
        f32_to_f16(1.0),
        f32_to_f16(0.0),
        f32_to_f16(0.0),
        f32_to_f16(1.0),
    ];
    let mut c = [10.0, 0.0, 0.0, 10.0_f32];
    gemm_ex(&i16, &i16, &mut c, 2, 2, 2, 2.0, 0.5)?;
    // C = 2·I + 0.5·[10,0,0,10] = [[2+5, 0], [0, 2+5]] = [[7, 0], [0, 7]]
    assert!((c[0] - 7.0).abs() < 1e-3);
    assert!((c[3] - 7.0).abs() < 1e-3);
    Ok(())
}

#[test]
fn test_gemm_ex_matmul_2x3_3x2() -> Result<(), Box<dyn std::error::Error>> {
    // A = [[1,2,3],[4,5,6]], B = [[7,8],[9,10],[11,12]]
    let a: Vec<u16> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0_f32]
        .iter()
        .map(|&v| f32_to_f16(v))
        .collect();
    let b: Vec<u16> = [7.0, 8.0, 9.0, 10.0, 11.0, 12.0_f32]
        .iter()
        .map(|&v| f32_to_f16(v))
        .collect();
    let mut c = [0.0_f32; 4];
    gemm_ex(&a, &b, &mut c, 2, 2, 3, 1.0, 0.0)?;
    // C = [[58, 64], [139, 154]]
    assert!((c[0] - 58.0).abs() < 0.5);
    assert!((c[1] - 64.0).abs() < 0.5);
    assert!((c[2] - 139.0).abs() < 0.5);
    assert!((c[3] - 154.0).abs() < 0.5);
    Ok(())
}

#[test]
fn test_f16_roundtrip() {
    // Test f32 → f16 → f32 roundtrip for representable values
    let vals = [0.0_f32, 1.0, -1.0, 0.5, 65504.0];
    for &v in &vals {
        let h = f32_to_f16(v);
        // Use gemm_ex with 1×1 identity to convert back (α=1, β=0)
        let mut c = [0.0_f32];
        gemm_ex(&[h], &[f32_to_f16(1.0)], &mut c, 1, 1, 1, 1.0, 0.0).expect("gemm_ex roundtrip ok");
        assert!(
            (c[0] - v).abs() < 0.01,
            "f16 roundtrip failed for {v}: got {}",
            c[0]
        );
    }
}

// ============================================================================
// gemmStridedBatched tests
// ============================================================================

#[test]
fn test_gemm_strided_batched_single() -> Result<(), Box<dyn std::error::Error>> {
    // Single batch: same as regular GEMM
    let a = [1.0, 2.0, 3.0, 4.0_f32]; // 2×2
    let b = [5.0, 6.0, 7.0, 8.0_f32]; // 2×2
    let mut c = [0.0_f32; 4];
    gemm_strided_batched(&a, 4, &b, 4, &mut c, 4, 1, 2, 2, 2, 1.0, 0.0)?;
    // C = [[1·5+2·7, 1·6+2·8], [3·5+4·7, 3·6+4·8]] = [[19,22],[43,50]]
    assert!((c[0] - 19.0).abs() < 1e-5);
    assert!((c[1] - 22.0).abs() < 1e-5);
    assert!((c[2] - 43.0).abs() < 1e-5);
    assert!((c[3] - 50.0).abs() < 1e-5);
    Ok(())
}

#[test]
fn test_gemm_strided_batched_two_batches() -> Result<(), Box<dyn std::error::Error>> {
    // Two batches of 2×2 matmuls
    // Batch 0: A=I, B=[[1,2],[3,4]] → C=[[1,2],[3,4]]
    // Batch 1: A=[[2,0],[0,2]], B=[[1,0],[0,1]] → C=[[2,0],[0,2]]
    let a = [1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0_f32];
    let b = [1.0, 2.0, 3.0, 4.0, 1.0, 0.0, 0.0, 1.0_f32];
    let mut c = [0.0_f32; 8];
    gemm_strided_batched(&a, 4, &b, 4, &mut c, 4, 2, 2, 2, 2, 1.0, 0.0)?;
    // Batch 0
    assert!((c[0] - 1.0).abs() < 1e-5);
    assert!((c[1] - 2.0).abs() < 1e-5);
    assert!((c[2] - 3.0).abs() < 1e-5);
    assert!((c[3] - 4.0).abs() < 1e-5);
    // Batch 1
    assert!((c[4] - 2.0).abs() < 1e-5);
    assert!((c[5] - 0.0).abs() < 1e-5);
    assert!((c[6] - 0.0).abs() < 1e-5);
    assert!((c[7] - 2.0).abs() < 1e-5);
    Ok(())
}

#[test]
fn test_gemm_strided_batched_alpha_beta() -> Result<(), Box<dyn std::error::Error>> {
    let a = [1.0, 0.0, 0.0, 1.0_f32]; // I
    let b = [1.0, 0.0, 0.0, 1.0_f32]; // I
    let mut c = [10.0, 0.0, 0.0, 10.0_f32];
    // C = 3·I·I + 0.5·C = [[3+5, 0], [0, 3+5]] = [[8, 0], [0, 8]]
    gemm_strided_batched(&a, 4, &b, 4, &mut c, 4, 1, 2, 2, 2, 3.0, 0.5)?;
    assert!((c[0] - 8.0).abs() < 1e-5);
    assert!((c[3] - 8.0).abs() < 1e-5);
    Ok(())
}

#[test]
fn test_gemm_strided_batched_zero_batch() -> Result<(), Box<dyn std::error::Error>> {
    let a = [1.0_f32; 4];
    let b = [1.0_f32; 4];
    let mut c = [999.0_f32; 4];
    gemm_strided_batched(&a, 4, &b, 4, &mut c, 4, 0, 2, 2, 2, 1.0, 0.0)?;
    // Zero batches: C unchanged
    assert!((c[0] - 999.0).abs() < 1e-5);
    Ok(())
}

#[test]
fn test_gemm_ex_buffer_mismatch() {
    let a = [f32_to_f16(1.0); 4]; // 2×2
    let b = [f32_to_f16(1.0); 2]; // only 2 elements, need 4 for 2×2
    let mut c = [0.0_f32; 4];
    let result = gemm_ex(&a, &b, &mut c, 2, 2, 2, 1.0, 0.0);
    assert!(result.is_err());
}

// ============================================================================
// Solver trait tests
// ============================================================================

use crate::Solver;

#[test]
fn test_solver_trait_lu() -> Result<(), Box<dyn std::error::Error>> {
    let lu = lu_factorize(&[2.0, 1.0, 1.0, 3.0_f32], 2)?;
    let solver: &dyn Solver = &lu;
    assert_eq!(solver.dimension(), 2);
    let x = solver.solve(&[5.0, 7.0])?;
    assert!((x[0] - 1.6).abs() < 1e-5);
    assert!((x[1] - 1.8).abs() < 1e-5);
    Ok(())
}

#[test]
fn test_solver_trait_cholesky() -> Result<(), Box<dyn std::error::Error>> {
    let chol = cholesky(&[4.0, 2.0, 2.0, 3.0_f32], 2)?;
    let solver: &dyn Solver = &chol;
    assert_eq!(solver.dimension(), 2);
    let x = solver.solve(&[8.0, 7.0])?;
    assert!((x[0] - 1.25).abs() < 1e-4);
    assert!((x[1] - 1.5).abs() < 1e-4);
    Ok(())
}

#[test]
fn test_solver_trait_qr() -> Result<(), Box<dyn std::error::Error>> {
    let qr = qr_factorize(&[1.0, 0.0, 0.0, 1.0_f32], 2, 2)?;
    let solver: &dyn Solver = &qr;
    assert_eq!(solver.dimension(), 2);
    let x = solver.solve(&[3.0, 7.0])?;
    assert!((x[0] - 3.0).abs() < 1e-5);
    assert!((x[1] - 7.0).abs() < 1e-5);
    Ok(())
}

// ── Epilogue fusion tests (§8 parity) ──────────────────────────

#[test]
fn test_gemm_ex_epilogue_none() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<u16> = [1.0, 0.0, 0.0, 1.0_f32]
        .iter()
        .map(|&v| f32_to_f16(v))
        .collect();
    let b: Vec<u16> = [3.0, 4.0, 5.0, 6.0_f32]
        .iter()
        .map(|&v| f32_to_f16(v))
        .collect();
    let mut c = [0.0_f32; 4];
    gemm_ex_epilogue(&a, &b, &mut c, 2, 2, 2, 1.0, 0.0, Epilogue::None, None)?;
    assert!((c[0] - 3.0).abs() < 0.1);
    assert!((c[3] - 6.0).abs() < 0.1);
    Ok(())
}

#[test]
fn test_gemm_ex_epilogue_relu() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<u16> = [1.0, 0.0, 0.0, 1.0_f32]
        .iter()
        .map(|&v| f32_to_f16(v))
        .collect();
    let b: Vec<u16> = [-3.0, 4.0, 5.0, -6.0_f32]
        .iter()
        .map(|&v| f32_to_f16(v))
        .collect();
    let mut c = [0.0_f32; 4];
    gemm_ex_epilogue(&a, &b, &mut c, 2, 2, 2, 1.0, 0.0, Epilogue::Relu, None)?;
    assert!((c[0] - 0.0).abs() < 0.1); // max(0, -3) = 0
    assert!((c[1] - 4.0).abs() < 0.1); // max(0, 4) = 4
    assert!((c[3] - 0.0).abs() < 0.1); // max(0, -6) = 0
    Ok(())
}

#[test]
fn test_gemm_ex_epilogue_bias() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<u16> = [1.0, 0.0, 0.0, 1.0_f32]
        .iter()
        .map(|&v| f32_to_f16(v))
        .collect();
    let b: Vec<u16> = [1.0, 2.0, 3.0, 4.0_f32]
        .iter()
        .map(|&v| f32_to_f16(v))
        .collect();
    let mut c = [0.0_f32; 4];
    let bias = [10.0, 20.0_f32];
    gemm_ex_epilogue(
        &a,
        &b,
        &mut c,
        2,
        2,
        2,
        1.0,
        0.0,
        Epilogue::Bias,
        Some(&bias),
    )?;
    assert!((c[0] - 11.0).abs() < 0.1); // 1 + 10
    assert!((c[1] - 22.0).abs() < 0.1); // 2 + 20
    Ok(())
}

#[test]
fn test_gemm_ex_epilogue_gelu() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<u16> = [1.0, 0.0, 0.0, 1.0_f32]
        .iter()
        .map(|&v| f32_to_f16(v))
        .collect();
    let b: Vec<u16> = [2.0, 0.0, 0.0, -2.0_f32]
        .iter()
        .map(|&v| f32_to_f16(v))
        .collect();
    let mut c = [0.0_f32; 4];
    gemm_ex_epilogue(&a, &b, &mut c, 2, 2, 2, 1.0, 0.0, Epilogue::Gelu, None)?;
    // GELU(2.0) ≈ 1.9545, GELU(-2.0) ≈ -0.0455
    assert!(c[0] > 1.9);
    assert!(c[3] < 0.0 && c[3] > -0.1);
    Ok(())
}

#[test]
fn test_gemm_ex_epilogue_bias_required() {
    let a: Vec<u16> = [1.0_f32].iter().map(|&v| f32_to_f16(v)).collect();
    let b: Vec<u16> = [1.0_f32].iter().map(|&v| f32_to_f16(v)).collect();
    let mut c = [0.0_f32];
    // Bias epilogue without bias → error
    let result = gemm_ex_epilogue(&a, &b, &mut c, 1, 1, 1, 1.0, 0.0, Epilogue::Bias, None);
    assert!(result.is_err());
}

// ============================================================================
// POPPERIAN FALSIFICATION: Adversarial edge cases (GH #152)
// ============================================================================

#[test]
fn test_falsify_lu_1x1() {
    let a = [5.0_f32];
    let lu = lu_factorize(&a, 1).expect("1x1 LU");
    let x = lu.solve(&[15.0]).expect("1x1 solve");
    assert!((x[0] - 3.0).abs() < 1e-5);
}

#[test]
fn test_falsify_cholesky_1x1() {
    let a = [9.0_f32];
    let chol = cholesky(&a, 1).expect("1x1 cholesky");
    assert!((chol.l[0] - 3.0).abs() < 1e-5); // sqrt(9)=3
    let x = chol.solve(&[18.0]).expect("1x1 solve");
    assert!((x[0] - 2.0).abs() < 1e-5); // 9*x=18
}

#[test]
fn test_falsify_cholesky_negative_diagonal() {
    // Diagonal matrix with negative entry → not SPD
    let a = [-1.0, 0.0, 0.0, 4.0_f32];
    assert!(cholesky(&a, 2).is_err());
}

#[test]
fn test_falsify_svd_zero_matrix() {
    let a = [0.0_f32; 4]; // 2×2 zero
    let result = svd(&a, 2, 2).expect("svd of zero");
    for s in &result.sigma {
        assert!(
            s.abs() < 1e-5,
            "Zero matrix should have zero singular values: {s}"
        );
    }
}

#[test]
fn test_falsify_svd_1x1() {
    let a = [7.0_f32];
    let result = svd(&a, 1, 1).expect("svd 1x1");
    assert!((result.sigma[0] - 7.0).abs() < 1e-4);
}

#[test]
fn test_falsify_svd_rectangular_tall() {
    // 3×1 matrix
    let a = [3.0, 4.0, 0.0_f32];
    let result = svd(&a, 3, 1).expect("svd 3x1");
    // Singular value = ||[3,4,0]|| = 5
    assert!((result.sigma[0] - 5.0).abs() < 0.1, "σ={}", result.sigma[0]);
}

#[test]
fn test_falsify_trsm_1x1() -> Result<(), Box<dyn std::error::Error>> {
    let a = [4.0_f32];
    let b = [12.0_f32];
    let result = trsm(&a, &b, 1, 1, TriangularSide::Lower, DiagonalType::NonUnit)?;
    assert!((result.x[0] - 3.0).abs() < 1e-5);
    Ok(())
}

#[test]
fn test_falsify_syrk_zero_k() -> Result<(), Box<dyn std::error::Error>> {
    // A is n×0 (zero columns) → C = α·0 + β·C
    let a: &[f32] = &[];
    let mut c = [10.0, 0.0, 0.0, 10.0_f32];
    syrk(a, &mut c, 2, 0, 1.0, 0.5)?;
    assert!((c[0] - 5.0).abs() < 1e-5); // 0 + 0.5*10
    assert!((c[3] - 5.0).abs() < 1e-5);
    Ok(())
}

#[test]
fn test_falsify_gemm_ex_1x1() -> Result<(), Box<dyn std::error::Error>> {
    let a = [f32_to_f16(3.0)];
    let b = [f32_to_f16(7.0)];
    let mut c = [0.0_f32];
    gemm_ex(&a, &b, &mut c, 1, 1, 1, 1.0, 0.0)?;
    assert!((c[0] - 21.0).abs() < 0.5);
    Ok(())
}

#[test]
fn test_falsify_epilogue_bias_relu() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<u16> = [1.0, 0.0, 0.0, 1.0_f32]
        .iter()
        .map(|&v| f32_to_f16(v))
        .collect();
    let b: Vec<u16> = [-10.0, 5.0, 3.0, -7.0_f32]
        .iter()
        .map(|&v| f32_to_f16(v))
        .collect();
    let mut c = [0.0_f32; 4];
    let bias = [100.0, 200.0_f32];
    gemm_ex_epilogue(
        &a,
        &b,
        &mut c,
        2,
        2,
        2,
        1.0,
        0.0,
        Epilogue::BiasRelu,
        Some(&bias),
    )?;
    // C = matmul + bias, then ReLU
    // [(-10+100), (5+200)] = [90, 205] → relu: [90, 205]
    // [(3+100), (-7+200)] = [103, 193] → relu: [103, 193]
    assert!(c[0] > 80.0, "BiasRelu c[0]={}", c[0]);
    assert!(c[1] > 190.0, "BiasRelu c[1]={}", c[1]);
    Ok(())
}

#[test]
fn test_falsify_epilogue_bias_gelu() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<u16> = [1.0, 0.0, 0.0, 1.0_f32]
        .iter()
        .map(|&v| f32_to_f16(v))
        .collect();
    let b: Vec<u16> = [2.0, -2.0, 0.0, 3.0_f32]
        .iter()
        .map(|&v| f32_to_f16(v))
        .collect();
    let mut c = [0.0_f32; 4];
    let bias = [1.0, 1.0_f32];
    gemm_ex_epilogue(
        &a,
        &b,
        &mut c,
        2,
        2,
        2,
        1.0,
        0.0,
        Epilogue::BiasGelu,
        Some(&bias),
    )?;
    // C[0,0] = gelu(2+1) = gelu(3) ≈ 2.996
    assert!(c[0] > 2.9, "BiasGelu c[0]={}", c[0]);
    // C[0,1] = gelu(-2+1) = gelu(-1) ≈ -0.159
    assert!(c[1] < 0.0 && c[1] > -0.2, "BiasGelu c[1]={}", c[1]);
    Ok(())
}

#[test]
fn test_falsify_solver_trait_dimension_check() -> Result<(), Box<dyn std::error::Error>> {
    let lu = lu_factorize(&[4.0, 1.0, 1.0, 3.0_f32], 2)?;
    let solver: &dyn crate::Solver = &lu;
    // Wrong dimension RHS
    let result = solver.solve(&[1.0, 2.0, 3.0]); // 3 elements for 2×2 system
    assert!(result.is_err(), "Solver should reject wrong-dimension RHS");
    Ok(())
}

#[test]
fn test_falsify_batched_independence() -> Result<(), Box<dyn std::error::Error>> {
    // Verify batch 0 doesn't pollute batch 1
    let a = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0_f32]; // batch0=I, batch1=zero
    let b = [5.0, 6.0, 7.0, 8.0, 1.0, 1.0, 1.0, 1.0_f32];
    let mut c = [0.0_f32; 8];
    gemm_strided_batched(&a, 4, &b, 4, &mut c, 4, 2, 2, 2, 2, 1.0, 0.0)?;
    // Batch 0: I * B = B
    assert!((c[0] - 5.0).abs() < 1e-5);
    // Batch 1: zero * B = 0
    for i in 4..8 {
        assert!(
            c[i].abs() < 1e-5,
            "Batch independence violated at c[{i}]={}",
            c[i]
        );
    }
    Ok(())
}

// ============================================================================
// OBLIG-SOLVE-F32-F16-RNE: f32_to_f16 IEEE round-to-nearest-even
//
// The old hand-rolled f32_to_f16 truncated the mantissa (round-toward-zero)
// and truncated the subnormal/overflow boundaries. These falsifiers are RED on
// that implementation and GREEN on the RNE fix (bit-identical to half).
// ============================================================================

#[test]
fn falsify_f32_to_f16_known_rne_divergences() {
    // Each case is a value where round-toward-zero (old) and round-to-nearest-even
    // (correct) disagree. The asserted bits are `half::f16::from_f32`'s output.
    // 255.99: old truncated to 0x5BFF; RNE rounds the carry up to 0x5C00.
    assert_eq!(
        f32_to_f16(255.99),
        0x5C00,
        "255.99 must round up (mantissa carry)"
    );
    // 65520.0: old truncated to max-finite 0x7BFF; RNE rounds the tie up to +Inf.
    assert_eq!(f32_to_f16(65520.0), 0x7C00, "65520.0 must round to +Inf");
    // Smallest f32 above the f16 min-subnormal half-way point: old gave 0x0000.
    assert_eq!(
        f32_to_f16(5.9604645e-8),
        0x0001,
        "subnormal half-way must round to nearest even (0x0001)"
    );
    // 65504.0 is exactly the max finite f16 — must NOT overflow.
    assert_eq!(f32_to_f16(65504.0), 0x7BFF, "max finite f16 preserved");
}

#[test]
fn falsify_f32_to_f16_ties_to_even() {
    // Exact ties round to the even mantissa. 2049.0 sits on a tie boundary in f16;
    // half encodes it as 0x6800 (mantissa rounds to even).
    assert_eq!(f32_to_f16(2049.0), half::f16::from_f32(2049.0).to_bits());
    assert_eq!(f32_to_f16(2048.5), half::f16::from_f32(2048.5).to_bits());
    assert_eq!(f32_to_f16(1000.5), half::f16::from_f32(1000.5).to_bits());
}

#[test]
fn falsify_f32_to_f16_bit_identical_to_half_on_grid() {
    // Strided exhaustive sweep over the full 2^32 f32 space (step keeps it fast).
    // Asserts bit-identity to half::f16::from_f32 (the IEEE RNE reference),
    // treating any-NaN == any-NaN as equal (NaN payload is canonicalized).
    let step = 0x29u32; // odd, coprime-ish stride to hit every exponent/mantissa region
    let mut u: u32 = 0;
    loop {
        let v = f32::from_bits(u);
        let got = f32_to_f16(v);
        let want = half::f16::from_f32(v).to_bits();
        if got != want {
            let both_nan =
                half::f16::from_bits(got).is_nan() && half::f16::from_bits(want).is_nan();
            assert!(
                both_nan,
                "f32_to_f16 diverges from half at bits={u:#010x} (v={v:e}): got={got:#06x} want={want:#06x}"
            );
        }
        let (next, overflow) = u.overflowing_add(step);
        if overflow {
            break;
        }
        u = next;
    }
}

#[test]
fn falsify_f32_to_f16_edge_values_match_half() {
    for &v in &[
        0.0f32,
        -0.0,
        1.0,
        -1.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
        6.1035156e-5, // smallest normal f16
        5.9604645e-8, // smallest subnormal f16
        3.0e-8,
        255.99,
        65504.0,
        65520.0,
        70000.0, // overflow
        1e-10,   // underflow to zero
    ] {
        assert_eq!(
            f32_to_f16(v),
            half::f16::from_f32(v).to_bits(),
            "f32_to_f16({v:e}) must equal half"
        );
    }
}
