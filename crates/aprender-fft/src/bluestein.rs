//! Bluestein chirp-z transform for arbitrary-length FFT.
//!
//! # Contract: fft-bluestein-v1.yaml
//!
//! Converts an N-point DFT (any N) to a circular convolution of length M
//! where M is the next power of two >= 2N-1. Uses the identity:
//!
//!   X[k] = chirp[k] · Σ_n (x[n] · chirp[n]) · chirp_conj[k-n]
//!
//! where chirp[n] = exp(-iπn²/N).

use crate::complex::Complex;
use crate::error::FftError;
use crate::stockham::FftPlan;
use std::f64::consts::PI;

/// Arbitrary-length FFT using Bluestein's algorithm.
///
/// Works for any positive length N (not just powers of two).
///
/// # Errors
///
/// Returns error if input is empty or output length doesn't match.
pub fn bluestein_fft(
    input: &[Complex],
    output: &mut [Complex],
    inverse: bool,
) -> Result<(), FftError> {
    let n = input.len();
    if n == 0 {
        return Err(FftError::ZeroLength);
    }
    if output.len() != n {
        return Err(FftError::OutputLengthMismatch { expected: n, got: output.len() });
    }

    // For power-of-two sizes, delegate directly
    if n.is_power_of_two() {
        let plan = FftPlan::new(n)?;
        if inverse {
            return plan.inverse(input, output);
        }
        return plan.forward(input, output);
    }

    // Convolution length: next power of two >= 2N-1
    let m = (2 * n - 1).next_power_of_two();

    // Precompute chirp: chirp[n] = exp(-i·π·n²/N) for forward, conjugate for inverse
    let sign = if inverse { 1.0 } else { -1.0 };
    let chirp = precompute_chirp(n, sign);

    // Build the sequences for convolution
    let (a_padded, b_padded) = build_convolution_sequences(input, &chirp, n, m);

    // Convolve via FFT
    let conv = fft_convolve(&a_padded, &b_padded, m)?;

    // Extract result: X[k] = chirp[k] * conv[k]
    let scale = if inverse { 1.0 / n as f32 } else { 1.0 };
    for k in 0..n {
        output[k] = (chirp[k] * conv[k]).scale(scale);
    }

    Ok(())
}

/// Precompute chirp factors: chirp[n] = exp(sign · i·π·n²/N).
fn precompute_chirp(n: usize, sign: f64) -> Vec<Complex> {
    let nf = n as f64;
    (0..n)
        .map(|k| {
            let kf = k as f64;
            let angle = sign * PI * kf * kf / nf;
            Complex::new(angle.cos() as f32, angle.sin() as f32)
        })
        .collect()
}

/// Build zero-padded sequences for convolution.
fn build_convolution_sequences(
    input: &[Complex],
    chirp: &[Complex],
    n: usize,
    m: usize,
) -> (Vec<Complex>, Vec<Complex>) {
    // a[n] = x[n] * chirp[n], zero-padded to M
    let mut a = vec![Complex::ZERO; m];
    for i in 0..n {
        a[i] = input[i] * chirp[i];
    }

    // b[k] = conj(chirp[k]) for circular convolution kernel
    // b[0..N] = conj(chirp[0..N]), b[M-N+1..M] = conj(chirp[N-1..1]) (wrap-around)
    let mut b = vec![Complex::ZERO; m];
    b[0] = chirp[0].conj();
    for i in 1..n {
        let c = chirp[i].conj();
        b[i] = c;
        b[m - i] = c;
    }

    (a, b)
}

/// Circular convolution via FFT: conv = IFFT(FFT(a) * FFT(b)).
fn fft_convolve(a: &[Complex], b: &[Complex], m: usize) -> Result<Vec<Complex>, FftError> {
    let plan = FftPlan::new(m)?;

    let mut fa = vec![Complex::ZERO; m];
    let mut fb = vec![Complex::ZERO; m];
    plan.forward(a, &mut fa)?;
    plan.forward(b, &mut fb)?;

    // Pointwise multiply
    let mut product = vec![Complex::ZERO; m];
    for i in 0..m {
        product[i] = fa[i] * fb[i];
    }

    let mut conv = vec![Complex::ZERO; m];
    plan.inverse(&product, &mut conv)?;

    Ok(conv)
}
