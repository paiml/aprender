//! Stockham auto-sort FFT algorithm.
//!
//! # Contract: fft-stockham-v1.yaml
//!
//! Uses iterative decimation-in-time with bit-reversal permutation.
//! Twiddle factors precomputed in f64 for VkFFT-class accuracy.
//!
//! ## Proof obligations
//! - Parseval energy conservation: Σ|x_n|² = (1/N)·Σ|X_k|²
//! - Inverse roundtrip: IFFT(FFT(x)) = x within tolerance
//! - Linearity: FFT(αx + βy) = α·FFT(x) + β·FFT(y)

use crate::complex::Complex;
use crate::error::FftError;
use std::f64::consts::PI;

/// Precomputed FFT plan for a specific size.
///
/// Twiddle factors are computed in f64 precision and stored as f32
/// (VkFFT approach for superior accuracy vs cuFFT).
#[derive(Debug, Clone)]
pub struct FftPlan {
    n: usize,
    log2_n: u32,
    /// Twiddle factors: w[k] = exp(-2πi·k/N) for k in 0..N/2.
    twiddles: Vec<Complex>,
}

impl FftPlan {
    /// Create a new FFT plan for the given size.
    ///
    /// # Errors
    ///
    /// Returns error if `n` is zero or not a power of two.
    pub fn new(n: usize) -> Result<Self, FftError> {
        if n == 0 {
            return Err(FftError::ZeroLength);
        }
        if !n.is_power_of_two() {
            return Err(FftError::NotPowerOfTwo(n));
        }
        let log2_n = n.trailing_zeros();

        // Precompute twiddle factors in f64
        let half = n / 2;
        let mut twiddles = Vec::with_capacity(half.max(1));
        if half > 0 {
            for k in 0..half {
                let angle = -2.0 * PI * (k as f64) / (n as f64);
                twiddles.push(Complex::new(angle.cos() as f32, angle.sin() as f32));
            }
        }

        Ok(Self { n, log2_n, twiddles })
    }

    /// Transform size.
    pub fn len(&self) -> usize {
        self.n
    }

    /// Whether the plan is empty.
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// Forward FFT: X_k = Σ x_n · exp(-2πi·k·n/N).
    ///
    /// # Errors
    ///
    /// Returns error if input/output dimensions don't match plan size.
    pub fn forward(&self, input: &[Complex], output: &mut [Complex]) -> Result<(), FftError> {
        self.validate_buffers(input.len(), output.len())?;
        self.transform(input, output, false);
        Ok(())
    }

    /// Inverse FFT: x_n = (1/N) · Σ X_k · exp(+2πi·k·n/N).
    ///
    /// # Errors
    ///
    /// Returns error if input/output dimensions don't match plan size.
    pub fn inverse(&self, input: &[Complex], output: &mut [Complex]) -> Result<(), FftError> {
        self.validate_buffers(input.len(), output.len())?;
        self.transform(input, output, true);

        let scale = 1.0 / self.n as f32;
        for x in output.iter_mut() {
            *x = x.scale(scale);
        }
        Ok(())
    }

    /// Real-to-complex FFT. Output has N/2+1 elements (Hermitian symmetry).
    ///
    /// # Errors
    ///
    /// Returns error on dimension mismatch.
    pub fn forward_r2c(&self, input: &[f32], output: &mut [Complex]) -> Result<(), FftError> {
        if input.len() != self.n {
            return Err(FftError::OutputLengthMismatch { expected: self.n, got: input.len() });
        }
        let expected_out = self.n / 2 + 1;
        if output.len() != expected_out {
            return Err(FftError::R2cOutputLengthMismatch {
                expected: expected_out,
                got: output.len(),
            });
        }

        let complex_input: Vec<Complex> = input.iter().map(|&r| Complex::new(r, 0.0)).collect();
        let mut full_output = vec![Complex::ZERO; self.n];
        self.transform(&complex_input, &mut full_output, false);
        output.copy_from_slice(&full_output[..expected_out]);
        Ok(())
    }

    /// Complex-to-real inverse FFT (C2R). Input has N/2+1 elements (Hermitian),
    /// output has N real values.
    ///
    /// Reconstructs the full spectrum from Hermitian symmetry, applies inverse FFT,
    /// and returns the real part.
    ///
    /// # Errors
    ///
    /// Returns error on dimension mismatch.
    pub fn inverse_c2r(&self, input: &[Complex], output: &mut [f32]) -> Result<(), FftError> {
        let expected_in = self.n / 2 + 1;
        if input.len() != expected_in {
            return Err(FftError::R2cOutputLengthMismatch {
                expected: expected_in,
                got: input.len(),
            });
        }
        if output.len() != self.n {
            return Err(FftError::OutputLengthMismatch { expected: self.n, got: output.len() });
        }

        // Reconstruct full N-point spectrum from Hermitian symmetry: X[N-k] = conj(X[k])
        let mut full_input = vec![Complex::ZERO; self.n];
        for (i, &val) in input.iter().enumerate() {
            full_input[i] = val;
        }
        for k in 1..self.n / 2 {
            full_input[self.n - k] = input[k].conj();
        }

        let mut full_output = vec![Complex::ZERO; self.n];
        self.transform(&full_input, &mut full_output, true);

        let scale = 1.0 / self.n as f32;
        for (i, x) in full_output.iter().enumerate() {
            output[i] = x.re * scale;
        }
        Ok(())
    }

    fn validate_buffers(&self, in_len: usize, out_len: usize) -> Result<(), FftError> {
        if in_len != self.n {
            return Err(FftError::OutputLengthMismatch { expected: self.n, got: in_len });
        }
        if out_len != self.n {
            return Err(FftError::OutputLengthMismatch { expected: self.n, got: out_len });
        }
        Ok(())
    }

    /// Iterative Cooley-Tukey decimation-in-time FFT.
    ///
    /// 1. Bit-reversal permutation
    /// 2. log2(N) butterfly stages with precomputed twiddle factors
    fn transform(&self, input: &[Complex], output: &mut [Complex], inverse: bool) {
        let n = self.n;
        if n == 1 {
            output[0] = input[0];
            return;
        }

        // Bit-reversal permutation
        for i in 0..n {
            output[bit_reverse(i, self.log2_n)] = input[i];
        }

        // Butterfly stages
        let mut half_size = 1;
        for _stage in 0..self.log2_n {
            let full_size = half_size << 1;
            let tw_step = n / full_size;

            let mut group_start = 0;
            while group_start < n {
                for k in 0..half_size {
                    let tw_idx = k * tw_step;
                    let tw =
                        if inverse { self.twiddles[tw_idx].conj() } else { self.twiddles[tw_idx] };

                    let even_idx = group_start + k;
                    let odd_idx = even_idx + half_size;

                    let even = output[even_idx];
                    let odd = tw * output[odd_idx];

                    output[even_idx] = even + odd;
                    output[odd_idx] = even - odd;
                }
                group_start += full_size;
            }

            half_size = full_size;
        }
    }
}

/// Bit-reversal of `i` with `bits` significant bits.
#[inline]
fn bit_reverse(mut i: usize, bits: u32) -> usize {
    let mut result = 0;
    for _ in 0..bits {
        result = (result << 1) | (i & 1);
        i >>= 1;
    }
    result
}

/// Free-function R2C forward FFT. Output has N/2+1 complex elements.
///
/// # Errors
///
/// Returns error if N is not power of two or dimensions mismatch.
pub fn fft_r2c(input: &[f32], output: &mut [Complex]) -> Result<(), FftError> {
    let plan = FftPlan::new(input.len())?;
    plan.forward_r2c(input, output)
}

/// Free-function C2R inverse FFT. Input has N/2+1 complex elements, output has N reals.
///
/// # Errors
///
/// Returns error if N is not power of two or dimensions mismatch.
pub fn fft_c2r(input: &[Complex], output: &mut [f32], n: usize) -> Result<(), FftError> {
    let plan = FftPlan::new(n)?;
    plan.inverse_c2r(input, output)
}

/// 2D FFT via row-column decomposition.
///
/// # Errors
///
/// Returns error if dimensions don't match or aren't powers of two.
pub fn fft_2d(
    input: &[Complex],
    output: &mut [Complex],
    nx: usize,
    ny: usize,
) -> Result<(), FftError> {
    if input.len() != nx * ny {
        return Err(FftError::DimensionMismatch2d { len: input.len(), nx, ny });
    }
    if output.len() != nx * ny {
        return Err(FftError::DimensionMismatch2d { len: output.len(), nx, ny });
    }

    let plan_x = FftPlan::new(nx)?;
    let plan_y = FftPlan::new(ny)?;

    // FFT along rows
    let mut row_buf = vec![Complex::ZERO; nx];
    let mut temp = input.to_vec();
    for row in 0..ny {
        let start = row * nx;
        plan_x.forward(&temp[start..start + nx], &mut row_buf)?;
        temp[start..start + nx].copy_from_slice(&row_buf);
    }

    // FFT along columns
    let mut col_in = vec![Complex::ZERO; ny];
    let mut col_out = vec![Complex::ZERO; ny];
    for col in 0..nx {
        for row in 0..ny {
            col_in[row] = temp[row * nx + col];
        }
        plan_y.forward(&col_in, &mut col_out)?;
        for row in 0..ny {
            output[row * nx + col] = col_out[row];
        }
    }

    Ok(())
}
