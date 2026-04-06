//! Error types for trueno-viz operations.

use std::io;
use thiserror::Error;

/// Result type alias using [`Error`].
pub type Result<T> = std::result::Result<T, Error>;

/// Errors that can occur in trueno-viz operations.
#[derive(Error, Debug)]
pub enum Error {
    /// I/O error (file operations, etc.).
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),

    /// PNG encoding error.
    #[error("PNG encoding error: {0}")]
    PngEncoding(#[from] png::EncodingError),

    /// Invalid dimensions for framebuffer or plot.
    #[error("Invalid dimensions: {width}x{height}")]
    InvalidDimensions {
        /// Width value.
        width: u32,
        /// Height value.
        height: u32,
    },

    /// Data length mismatch between x and y arrays.
    #[error("Data length mismatch: x has {x_len} elements, y has {y_len} elements")]
    DataLengthMismatch {
        /// Length of x data.
        x_len: usize,
        /// Length of y data.
        y_len: usize,
    },

    /// Empty data provided where non-empty is required.
    #[error("Empty data provided")]
    EmptyData,

    /// Scale domain error (e.g., log of non-positive value).
    #[error("Scale domain error: {0}")]
    ScaleDomain(String),

    /// Color parsing error.
    #[error("Invalid color: {0}")]
    InvalidColor(String),

    /// Rendering error.
    #[error("Rendering error: {0}")]
    Rendering(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = Error::InvalidDimensions { width: 0, height: 100 };
        assert!(err.to_string().contains("Invalid dimensions"));
    }

    #[test]
    fn test_data_length_mismatch() {
        let err = Error::DataLengthMismatch { x_len: 10, y_len: 20 };
        assert!(err.to_string().contains("10"));
        assert!(err.to_string().contains("20"));
    }
}
