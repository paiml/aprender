//! Error types for sparse matrix operations.

/// Errors that can occur during sparse matrix construction or operations.
#[derive(Debug, Clone, thiserror::Error)]
pub enum SparseError {
    /// CSR offsets array has wrong length (expected rows + 1).
    #[error("CSR offsets length {actual} != rows + 1 ({expected})")]
    InvalidOffsetsLength {
        /// Actual length of offsets array.
        actual: usize,
        /// Expected length (rows + 1).
        expected: usize,
    },

    /// CSR offsets are not monotonically non-decreasing.
    #[error("CSR offsets not monotonic at index {index}: offsets[{index}]={value} > offsets[{next_index}]={next_value}")]
    NonMonotonicOffsets {
        /// Index where violation occurs.
        index: usize,
        /// Value at the violating index.
        value: u32,
        /// Next index.
        next_index: usize,
        /// Value at next index.
        next_value: u32,
    },

    /// CSR offsets[0] must be 0.
    #[error("CSR offsets[0] = {value}, expected 0")]
    NonZeroFirstOffset {
        /// Actual value of offsets[0].
        value: u32,
    },

    /// CSR offsets[rows] must equal nnz.
    #[error("CSR offsets[rows] = {offset_last} != nnz ({nnz})")]
    OffsetNnzMismatch {
        /// Value of offsets[rows].
        offset_last: u32,
        /// Expected nnz (length of values/col_indices).
        nnz: usize,
    },

    /// Column index out of bounds.
    #[error("Column index {col} >= cols ({cols}) at position {position}")]
    ColumnOutOfBounds {
        /// The out-of-bounds column index.
        col: u32,
        /// Number of columns.
        cols: usize,
        /// Position in col_indices array.
        position: usize,
    },

    /// col_indices and values length mismatch.
    #[error("col_indices length {col_len} != values length {val_len}")]
    LengthMismatch {
        /// Length of col_indices.
        col_len: usize,
        /// Length of values.
        val_len: usize,
    },

    /// COO arrays have mismatched lengths.
    #[error("COO arrays have mismatched lengths: row_indices={row_len}, col_indices={col_len}, values={val_len}")]
    CooLengthMismatch {
        /// Length of row_indices.
        row_len: usize,
        /// Length of col_indices.
        col_len: usize,
        /// Length of values.
        val_len: usize,
    },

    /// COO row index out of bounds.
    #[error("Row index {row} >= rows ({rows}) at position {position}")]
    RowOutOfBounds {
        /// The out-of-bounds row index.
        row: u32,
        /// Number of rows.
        rows: usize,
        /// Position in row_indices array.
        position: usize,
    },

    /// Dimension mismatch in SpMV.
    #[error("SpMV dimension mismatch: matrix cols={matrix_cols}, x length={x_len}")]
    SpMVDimensionMismatch {
        /// Number of columns in the matrix.
        matrix_cols: usize,
        /// Length of x vector.
        x_len: usize,
    },

    /// Output dimension mismatch in SpMV.
    #[error("SpMV output dimension mismatch: matrix rows={matrix_rows}, y length={y_len}")]
    SpMVOutputDimensionMismatch {
        /// Number of rows in the matrix.
        matrix_rows: usize,
        /// Length of y vector.
        y_len: usize,
    },
}
