#![cfg_attr(
    test,
    allow(
        clippy::expect_used,
        clippy::unwrap_used,
        clippy::disallowed_methods,
        clippy::float_cmp,
        clippy::panic
    )
)]
//! Fast Fourier Transform library with Stockham auto-sort algorithm.
//!
//! # Contract: fft-stockham-v1.yaml
//!
//! Provides 1D and 2D FFT with provable properties:
//! - Parseval energy conservation
//! - Inverse roundtrip accuracy
//! - Linearity
//!
//! # Example
//!
//! ```
//! use trueno_fft::{Complex, FftPlan};
//!
//! let plan = FftPlan::new(4).unwrap();
//! let input = [
//!     Complex::new(1.0, 0.0),
//!     Complex::new(2.0, 0.0),
//!     Complex::new(3.0, 0.0),
//!     Complex::new(4.0, 0.0),
//! ];
//! let mut output = [Complex::ZERO; 4];
//! plan.forward(&input, &mut output).unwrap();
//!
//! // Verify Parseval: Σ|x|² = (1/N)·Σ|X|²
//! let energy_time: f32 = input.iter().map(|x| x.norm_sq()).sum();
//! let energy_freq: f32 = output.iter().map(|x| x.norm_sq()).sum::<f32>() / 4.0;
//! assert!((energy_time - energy_freq).abs() < 1e-4);
//! ```

mod bluestein;
mod complex;
mod error;
mod fft3d;
mod stockham;

#[cfg(test)]
mod tests;

pub use bluestein::bluestein_fft;
pub use complex::Complex;
pub use error::FftError;
pub use fft3d::{fft_3d, fft_batched, ifft_3d};
pub use stockham::{fft_2d, fft_c2r, fft_r2c, FftPlan};

/// Abstract FFT trait for pluggable implementations.
///
/// Provides a unified interface for 1D forward/inverse FFT,
/// R2C/C2R transforms, and 2D FFT. The default implementation
/// is `FftPlan` (Stockham auto-sort algorithm).
pub trait Fft {
    /// Forward 1D FFT: time domain → frequency domain.
    fn fft_1d(&self, input: &[Complex], output: &mut [Complex]) -> Result<(), FftError>;

    /// Inverse 1D FFT: frequency domain → time domain.
    fn ifft_1d(&self, input: &[Complex], output: &mut [Complex]) -> Result<(), FftError>;

    /// Real-to-complex 1D FFT (exploits Hermitian symmetry).
    fn fft_r2c(&self, input: &[f32], output: &mut [Complex]) -> Result<(), FftError>;

    /// Complex-to-real 1D inverse FFT.
    fn fft_c2r(&self, input: &[Complex], output: &mut [f32]) -> Result<(), FftError>;

    /// 2D FFT via row-column decomposition.
    fn fft_2d(
        &self,
        input: &[Complex],
        output: &mut [Complex],
        nx: usize,
        ny: usize,
    ) -> Result<(), FftError>;

    /// Transform length.
    fn len(&self) -> usize;

    /// Returns true if length is zero.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Fft for FftPlan {
    fn fft_1d(&self, input: &[Complex], output: &mut [Complex]) -> Result<(), FftError> {
        self.forward(input, output)
    }

    fn ifft_1d(&self, input: &[Complex], output: &mut [Complex]) -> Result<(), FftError> {
        self.inverse(input, output)
    }

    fn fft_r2c(&self, input: &[f32], output: &mut [Complex]) -> Result<(), FftError> {
        self.forward_r2c(input, output)
    }

    fn fft_c2r(&self, input: &[Complex], output: &mut [f32]) -> Result<(), FftError> {
        self.inverse_c2r(input, output)
    }

    fn fft_2d(
        &self,
        input: &[Complex],
        output: &mut [Complex],
        nx: usize,
        ny: usize,
    ) -> Result<(), FftError> {
        fft_2d(input, output, nx, ny)
    }

    fn len(&self) -> usize {
        self.len()
    }

    fn is_empty(&self) -> bool {
        self.is_empty()
    }
}
