//! 3D FFT and batched transforms.
//!
//! # Contract: fft-3d-v1.yaml
//!
//! 3D FFT via row-column-depth decomposition using 1D Stockham FFT.
//! Batched 1D FFT for processing multiple signals.
//!
//! ## Proof obligations
//! - 3D impulse → all ones
//! - Roundtrip: IFFT3D(FFT3D(x)) = x
//! - Parseval energy conservation

use crate::complex::Complex;
use crate::error::FftError;
use crate::stockham::FftPlan;

/// 3D FFT: forward transform of nx × ny × nz data (row-major).
///
/// Each dimension must be a power of two.
///
/// # Errors
///
/// Returns error if dimensions are invalid or input/output size mismatch.
pub fn fft_3d(
    input: &[Complex],
    output: &mut [Complex],
    nx: usize,
    ny: usize,
    nz: usize,
) -> Result<(), FftError> {
    fft_3d_impl(input, output, nx, ny, nz, false)
}

/// 3D inverse FFT.
///
/// # Errors
///
/// Returns error if dimensions are invalid or input/output size mismatch.
pub fn ifft_3d(
    input: &[Complex],
    output: &mut [Complex],
    nx: usize,
    ny: usize,
    nz: usize,
) -> Result<(), FftError> {
    fft_3d_impl(input, output, nx, ny, nz, true)
}

/// Implementation for both forward and inverse 3D FFT.
fn fft_3d_impl(
    input: &[Complex],
    output: &mut [Complex],
    nx: usize,
    ny: usize,
    nz: usize,
    inverse: bool,
) -> Result<(), FftError> {
    let total = nx * ny * nz;
    validate_3d(input.len(), output.len(), nx, ny, nz, total)?;

    let plan_x = FftPlan::new(nx)?;
    let plan_y = FftPlan::new(ny)?;
    let plan_z = FftPlan::new(nz)?;

    // Copy input to output as working buffer
    output.copy_from_slice(input);

    // Phase 1: FFT along z (innermost dimension)
    transform_along_z(output, nx, ny, nz, &plan_z, inverse)?;

    // Phase 2: FFT along y (middle dimension)
    transform_along_y(output, nx, ny, nz, &plan_y, inverse)?;

    // Phase 3: FFT along x (outermost dimension)
    transform_along_x(output, nx, ny, nz, &plan_x, inverse)?;

    Ok(())
}

/// Validate 3D FFT dimensions.
fn validate_3d(
    in_len: usize,
    out_len: usize,
    nx: usize,
    ny: usize,
    nz: usize,
    total: usize,
) -> Result<(), FftError> {
    if nx == 0 || ny == 0 || nz == 0 {
        return Err(FftError::ZeroLength);
    }
    if in_len != total {
        return Err(FftError::DimensionMismatch3d { len: in_len, nx, ny, nz });
    }
    if out_len != total {
        return Err(FftError::DimensionMismatch3d { len: out_len, nx, ny, nz });
    }
    Ok(())
}

/// FFT along z-axis (contiguous in memory).
fn transform_along_z(
    data: &mut [Complex],
    nx: usize,
    ny: usize,
    nz: usize,
    plan: &FftPlan,
    inverse: bool,
) -> Result<(), FftError> {
    let mut row = vec![Complex::ZERO; nz];
    let mut tmp = vec![Complex::ZERO; nz];

    for ix in 0..nx {
        for iy in 0..ny {
            let base = ix * ny * nz + iy * nz;
            row.copy_from_slice(&data[base..base + nz]);
            if inverse {
                plan.inverse(&row, &mut tmp)?;
            } else {
                plan.forward(&row, &mut tmp)?;
            }
            data[base..base + nz].copy_from_slice(&tmp);
        }
    }
    Ok(())
}

/// FFT along y-axis (stride = nz).
fn transform_along_y(
    data: &mut [Complex],
    nx: usize,
    ny: usize,
    nz: usize,
    plan: &FftPlan,
    inverse: bool,
) -> Result<(), FftError> {
    let mut col = vec![Complex::ZERO; ny];
    let mut tmp = vec![Complex::ZERO; ny];

    for ix in 0..nx {
        for iz in 0..nz {
            // Gather y-stride
            for iy in 0..ny {
                col[iy] = data[ix * ny * nz + iy * nz + iz];
            }
            if inverse {
                plan.inverse(&col, &mut tmp)?;
            } else {
                plan.forward(&col, &mut tmp)?;
            }
            // Scatter back
            for iy in 0..ny {
                data[ix * ny * nz + iy * nz + iz] = tmp[iy];
            }
        }
    }
    Ok(())
}

/// FFT along x-axis (stride = ny * nz).
fn transform_along_x(
    data: &mut [Complex],
    nx: usize,
    ny: usize,
    nz: usize,
    plan: &FftPlan,
    inverse: bool,
) -> Result<(), FftError> {
    let mut col = vec![Complex::ZERO; nx];
    let mut tmp = vec![Complex::ZERO; nx];
    let yz = ny * nz;

    for iy in 0..ny {
        for iz in 0..nz {
            // Gather x-stride
            for ix in 0..nx {
                col[ix] = data[ix * yz + iy * nz + iz];
            }
            if inverse {
                plan.inverse(&col, &mut tmp)?;
            } else {
                plan.forward(&col, &mut tmp)?;
            }
            // Scatter back
            for ix in 0..nx {
                data[ix * yz + iy * nz + iz] = tmp[ix];
            }
        }
    }
    Ok(())
}

/// Batched 1D FFT: apply FFT to `batch_count` signals of length `n`.
///
/// Input layout: signal 0 at `input[0..n]`, signal 1 at `input[n..2n]`, etc.
///
/// # Errors
///
/// Returns error if n is not a power of two or buffer sizes don't match.
pub fn fft_batched(
    input: &[Complex],
    output: &mut [Complex],
    n: usize,
    batch_count: usize,
    inverse: bool,
) -> Result<(), FftError> {
    let total = n * batch_count;
    if input.len() != total {
        return Err(FftError::OutputLengthMismatch { expected: total, got: input.len() });
    }
    if output.len() != total {
        return Err(FftError::OutputLengthMismatch { expected: total, got: output.len() });
    }

    let plan = FftPlan::new(n)?;

    for b in 0..batch_count {
        let offset = b * n;
        let src = &input[offset..offset + n];
        let dst = &mut output[offset..offset + n];
        if inverse {
            plan.inverse(src, dst)?;
        } else {
            plan.forward(src, dst)?;
        }
    }

    Ok(())
}
