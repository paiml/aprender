//! Image processing error types.

/// Errors from image operations.
#[derive(Debug, thiserror::Error)]
pub enum ImageError {
    /// Image dimensions are zero.
    #[error("image dimensions must be > 0: got {width}x{height}")]
    ZeroDimension {
        /// Image width.
        width: usize,
        /// Image height.
        height: usize,
    },

    /// Kernel dimensions are zero or even.
    #[error("kernel dimensions must be odd and > 0: got {kw}x{kh}")]
    InvalidKernelSize {
        /// Kernel width.
        kw: usize,
        /// Kernel height.
        kh: usize,
    },

    /// Buffer length mismatch.
    #[error("buffer length {got} does not match {width}x{height} = {expected}")]
    BufferLengthMismatch {
        /// Expected length.
        expected: usize,
        /// Actual length.
        got: usize,
        /// Image width.
        width: usize,
        /// Image height.
        height: usize,
    },

    /// Invalid threshold values.
    #[error("thresholds must be 0 ≤ low ≤ high ≤ 1: got low={low}, high={high}")]
    InvalidThresholds {
        /// Low threshold.
        low: f32,
        /// High threshold.
        high: f32,
    },

    /// Data length doesn't match dimensions.
    #[error("data length {got} does not match expected {expected}")]
    DimensionMismatch {
        /// Expected length.
        expected: usize,
        /// Actual length.
        got: usize,
    },

    /// Invalid channel index.
    #[error("channel {channel} out of range (max {max})")]
    InvalidChannel {
        /// Requested channel.
        channel: usize,
        /// Number of channels.
        max: usize,
    },
}
