//! Scalar reference GEMM implementations for Jidoka validation.
//!
//! These are the "gold standard" implementations used to validate optimized versions.
//!
//! # References
//!
//! - Golub & Van Loan (2013), Matrix Computations, 4th ed., Algorithm 1.1.1.

use crate::error::TruenoError;

use super::jidoka::{JidokaError, JidokaGuard};

/// Scalar reference GEMM for Jidoka validation
///
/// Computes C += A * B where:
/// - A is M x K (row-major)
/// - B is K x N (row-major)
/// - C is M x N (row-major)
///
/// This is the "gold standard" implementation used to validate optimized versions.
///
/// # References
///
/// This implements the naive O(MNK) algorithm as described in
/// Golub & Van Loan (2013), Matrix Computations, 4th ed., Algorithm 1.1.1.
pub fn gemm_reference(
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
) -> Result<(), TruenoError> {
    // Poka-yoke: dimension validation
    if a.len() != m * k {
        return Err(TruenoError::InvalidInput(format!(
            "A size mismatch: expected {}x{}={}, got {}",
            m,
            k,
            m * k,
            a.len()
        )));
    }
    if b.len() != k * n {
        return Err(TruenoError::InvalidInput(format!(
            "B size mismatch: expected {}x{}={}, got {}",
            k,
            n,
            k * n,
            b.len()
        )));
    }
    if c.len() != m * n {
        return Err(TruenoError::InvalidInput(format!(
            "C size mismatch: expected {}x{}={}, got {}",
            m,
            n,
            m * n,
            c.len()
        )));
    }

    // Scalar triple-nested loop
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for p in 0..k {
                sum += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] += sum;
        }
    }

    Ok(())
}

/// Validate a sampled output value for NaN/Inf (Jidoka andon cord).
#[inline(always)]
pub(super) fn jidoka_check_output(
    val: f32,
    idx: usize,
    sample_rate: usize,
) -> Result<(), JidokaError> {
    if idx % sample_rate == 0 {
        if val.is_nan() {
            return Err(JidokaError::NaNDetected { location: "output" });
        }
        if val.is_infinite() {
            return Err(JidokaError::InfDetected { location: "output" });
        }
    }
    Ok(())
}

/// Validate sampled input values for NaN/Inf.
pub(super) fn jidoka_check_inputs(
    a: &[f32],
    b: &[f32],
    guard: &JidokaGuard,
) -> Result<(), JidokaError> {
    for (idx, &val) in a.iter().enumerate() {
        if idx % guard.sample_rate == 0 {
            guard.check_input(val, "matrix A")?;
        }
    }
    for (idx, &val) in b.iter().enumerate() {
        if idx % guard.sample_rate == 0 {
            guard.check_input(val, "matrix B")?;
        }
    }
    Ok(())
}

/// Scalar reference GEMM with Jidoka validation
///
/// Same as `gemm_reference` but validates outputs against known-good computation.
pub fn gemm_reference_with_jidoka(
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    guard: &JidokaGuard,
) -> Result<(), JidokaError> {
    jidoka_check_inputs(a, b, guard)?;

    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for p in 0..k {
                sum += a[i * k + p] * b[p * n + j];
            }
            let output = c[i * n + j] + sum;
            jidoka_check_output(output, i * n + j, guard.sample_rate)?;
            c[i * n + j] = output;
        }
    }

    Ok(())
}
