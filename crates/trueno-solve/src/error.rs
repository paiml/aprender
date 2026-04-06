//! Solver error types.

/// Errors from solver operations.
#[derive(Debug, thiserror::Error)]
pub enum SolverError {
    /// Matrix is not square.
    #[error("matrix must be square: got {rows}x{cols}")]
    NotSquare {
        /// Number of rows.
        rows: usize,
        /// Number of columns.
        cols: usize,
    },

    /// Matrix is singular (zero pivot encountered).
    #[error("singular matrix: zero pivot at position {0}")]
    SingularMatrix(usize),

    /// Matrix is not positive definite.
    #[error("matrix is not positive definite: non-positive diagonal at position {0}")]
    NotPositiveDefinite(usize),

    /// Dimension mismatch in solve operation.
    #[error("dimension mismatch: matrix is {matrix_n}x{matrix_n}, rhs has {rhs_len} elements")]
    DimensionMismatch {
        /// Matrix dimension.
        matrix_n: usize,
        /// RHS vector length.
        rhs_len: usize,
    },

    /// SVD dimension mismatch.
    #[error("SVD: matrix is {m}x{n}, but output buffers have wrong dimensions")]
    SvdDimensionMismatch {
        /// Rows.
        m: usize,
        /// Columns.
        n: usize,
    },

    /// QR dimension mismatch.
    #[error("QR: matrix is {m}x{n} (requires m >= n)")]
    QrNotTallSkinny {
        /// Rows.
        m: usize,
        /// Columns.
        n: usize,
    },

    /// Invalid input parameter.
    #[error("invalid input: {reason}")]
    InvalidInput {
        /// Reason for invalidity.
        reason: &'static str,
    },

    /// Buffer length mismatch for BLAS operations.
    #[error("buffer length {got} does not match expected {expected} for {rows}x{cols} matrix")]
    BufferLengthMismatch {
        /// Expected length.
        expected: usize,
        /// Actual length.
        got: usize,
        /// Matrix rows.
        rows: usize,
        /// Matrix cols.
        cols: usize,
    },
}
