//! Triangular solve: AX = B (TRSM — cuBLAS parity).
//!
//! Solves for X where A is an n×n triangular matrix and B is n×nrhs.
//! Supports lower/upper triangular, unit/non-unit diagonal.

use crate::error::SolverError;

/// Whether the triangular matrix is lower or upper.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TriangularSide {
    /// Lower triangular (L[i][j] = 0 for j > i).
    Lower,
    /// Upper triangular (U[i][j] = 0 for j < i).
    Upper,
}

/// Whether the diagonal is unit (all ones) or general.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiagonalType {
    /// Diagonal elements are 1 (not stored/used).
    Unit,
    /// Diagonal elements are general (read from matrix).
    NonUnit,
}

/// Triangular solve result.
#[derive(Debug)]
pub struct TrsmResult {
    /// Solution matrix X, stored row-major n×nrhs.
    pub x: Vec<f32>,
    /// Number of rows/columns of A.
    pub n: usize,
    /// Number of right-hand sides (columns of B).
    pub nrhs: usize,
}

/// Solve AX = B where A is triangular.
///
/// # Errors
///
/// Returns error if dimensions don't match or A has zero diagonal.
pub fn trsm(
    a: &[f32],
    b: &[f32],
    n: usize,
    nrhs: usize,
    side: TriangularSide,
    diag: DiagonalType,
) -> Result<TrsmResult, SolverError> {
    if a.len() != n * n {
        return Err(SolverError::DimensionMismatch { matrix_n: n, rhs_len: a.len() });
    }
    if b.len() != n * nrhs {
        return Err(SolverError::DimensionMismatch { matrix_n: n, rhs_len: b.len() });
    }

    let mut x = b.to_vec();

    match side {
        TriangularSide::Lower => forward_substitution(a, &mut x, n, nrhs, diag)?,
        TriangularSide::Upper => back_substitution(a, &mut x, n, nrhs, diag)?,
    }

    Ok(TrsmResult { x, n, nrhs })
}

/// Apply diagonal scaling (unit or non-unit) to a substitution result.
fn apply_diagonal(
    a: &[f32],
    n: usize,
    i: usize,
    sum: f32,
    diag: DiagonalType,
) -> Result<f32, SolverError> {
    match diag {
        DiagonalType::Unit => Ok(sum),
        DiagonalType::NonUnit => {
            let d = a[i * n + i];
            if d.abs() < f32::EPSILON {
                return Err(SolverError::SingularMatrix(i));
            }
            Ok(sum / d)
        }
    }
}

/// Forward substitution for lower triangular systems.
fn forward_substitution(
    a: &[f32],
    x: &mut [f32],
    n: usize,
    nrhs: usize,
    diag: DiagonalType,
) -> Result<(), SolverError> {
    for col in 0..nrhs {
        for i in 0..n {
            let mut sum = x[i * nrhs + col];
            for j in 0..i {
                sum -= a[i * n + j] * x[j * nrhs + col];
            }
            x[i * nrhs + col] = apply_diagonal(a, n, i, sum, diag)?;
        }
    }
    Ok(())
}

/// Back substitution for upper triangular systems.
fn back_substitution(
    a: &[f32],
    x: &mut [f32],
    n: usize,
    nrhs: usize,
    diag: DiagonalType,
) -> Result<(), SolverError> {
    for col in 0..nrhs {
        for i in (0..n).rev() {
            let mut sum = x[i * nrhs + col];
            for j in (i + 1)..n {
                sum -= a[i * n + j] * x[j * nrhs + col];
            }
            x[i * nrhs + col] = apply_diagonal(a, n, i, sum, diag)?;
        }
    }
    Ok(())
}
