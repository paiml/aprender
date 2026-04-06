//! Tensor error types.

/// Errors from tensor operations.
#[derive(Debug, thiserror::Error)]
pub enum TensorError {
    /// Shape mismatch for an operation.
    #[error("shape mismatch: expected {expected:?}, got {got:?}")]
    ShapeMismatch {
        /// Expected shape.
        expected: Vec<usize>,
        /// Actual shape.
        got: Vec<usize>,
    },

    /// Data length doesn't match shape.
    #[error("data length {len} does not match shape {shape:?} (product = {product})")]
    DataLengthMismatch {
        /// Data length.
        len: usize,
        /// Expected shape.
        shape: Vec<usize>,
        /// Product of shape dims.
        product: usize,
    },

    /// Invalid einsum subscript string.
    #[error("invalid einsum subscript: {0}")]
    InvalidSubscript(String),

    /// Contracted dimensions don't match.
    #[error(
        "contracted dimension mismatch: index '{index}' has size {size_a} in A but {size_b} in B"
    )]
    ContractionDimensionMismatch {
        /// Index label.
        index: char,
        /// Size in tensor A.
        size_a: usize,
        /// Size in tensor B.
        size_b: usize,
    },
}
