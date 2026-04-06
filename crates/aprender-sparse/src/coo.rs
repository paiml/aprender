//! Coordinate (COO) sparse matrix format.
//!
//! COO is the natural construction format: store (row, col, value) triplets.
//! Convert to CSR for efficient arithmetic operations.

use crate::SparseError;

/// Coordinate sparse matrix format.
///
/// Stores explicit (row, col, value) triplets. Ideal for incremental
/// construction; convert to [`CsrMatrix`](crate::CsrMatrix) for arithmetic.
///
/// # Invariants (provable contract)
///
/// - All arrays have equal length
/// - `row_indices[i] < rows` for all i
/// - `col_indices[i] < cols` for all i
#[derive(Debug, Clone)]
pub struct CooMatrix<T> {
    /// Number of rows.
    pub rows: usize,
    /// Number of columns.
    pub cols: usize,
    /// Row index for each nonzero.
    pub row_indices: Vec<u32>,
    /// Column index for each nonzero.
    pub col_indices: Vec<u32>,
    /// Value for each nonzero.
    pub values: Vec<T>,
}

impl<T: Clone> CooMatrix<T> {
    /// Create a new COO matrix with validation.
    ///
    /// # Errors
    ///
    /// Returns error if arrays have mismatched lengths or indices are out of bounds.
    pub fn new(
        rows: usize,
        cols: usize,
        row_indices: Vec<u32>,
        col_indices: Vec<u32>,
        values: Vec<T>,
    ) -> Result<Self, SparseError> {
        if row_indices.len() != col_indices.len() || col_indices.len() != values.len() {
            return Err(SparseError::CooLengthMismatch {
                row_len: row_indices.len(),
                col_len: col_indices.len(),
                val_len: values.len(),
            });
        }

        for (i, &row) in row_indices.iter().enumerate() {
            if row as usize >= rows {
                return Err(SparseError::RowOutOfBounds { row, rows, position: i });
            }
        }

        for (i, &col) in col_indices.iter().enumerate() {
            if col as usize >= cols {
                return Err(SparseError::ColumnOutOfBounds { col, cols, position: i });
            }
        }

        Ok(Self { rows, cols, row_indices, col_indices, values })
    }

    /// Number of stored nonzero entries.
    #[must_use]
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// Create an empty COO matrix.
    #[must_use]
    pub fn empty(rows: usize, cols: usize) -> Self {
        Self { rows, cols, row_indices: Vec::new(), col_indices: Vec::new(), values: Vec::new() }
    }
}
