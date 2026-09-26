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

    /// A rank-typed conversion met a tensor of another rank.
    #[error("rank mismatch: expected rank {expected}, got rank {got}")]
    RankMismatch {
        /// Rank the type requires.
        expected: usize,
        /// Rank of the tensor.
        got: usize,
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
