//! FFT error types.

/// Errors from FFT operations.
#[derive(Debug, thiserror::Error)]
pub enum FftError {
    /// Input length is zero.
    #[error("FFT input length must be > 0")]
    ZeroLength,

    /// Input length is not a power of two.
    #[error("FFT input length {0} is not a power of two")]
    NotPowerOfTwo(usize),

    /// Output buffer has wrong length.
    #[error("output length {got} does not match expected {expected}")]
    OutputLengthMismatch {
        /// Expected length.
        expected: usize,
        /// Actual length.
        got: usize,
    },

    /// R2C output buffer has wrong length.
    #[error("R2C output length {got} does not match expected {expected} (N/2+1)")]
    R2cOutputLengthMismatch {
        /// Expected length.
        expected: usize,
        /// Actual length.
        got: usize,
    },

    /// 2D dimension mismatch.
    #[error("2D FFT: input length {len} does not match {nx}x{ny}")]
    DimensionMismatch2d {
        /// Input length.
        len: usize,
        /// X dimension.
        nx: usize,
        /// Y dimension.
        ny: usize,
    },

    /// 3D dimension mismatch.
    #[error("3D FFT: input length {len} does not match {nx}x{ny}x{nz}")]
    DimensionMismatch3d {
        /// Input length.
        len: usize,
        /// X dimension.
        nx: usize,
        /// Y dimension.
        ny: usize,
        /// Z dimension.
        nz: usize,
    },
}
