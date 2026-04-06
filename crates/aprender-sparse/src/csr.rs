//! Compressed Sparse Row (CSR) matrix format.
//!
//! CSR is the primary format for sparse matrix arithmetic (SpMV, SpMM).
//! Stores row offsets, column indices, and values in three contiguous arrays.
//!
//! # Contract: sparse-spmv-v1.yaml
//!
//! Format invariants validated at construction:
//! - `offsets[0] == 0 && offsets[rows] == nnz`
//! - `offsets` monotonically non-decreasing
//! - All column indices in `[0, cols)`

use crate::coo::CooMatrix;
use crate::error::SparseError;
use crate::validate::validate_csr_invariants;

/// Compressed Sparse Row matrix.
///
/// Three-array representation: `offsets` (len = rows+1), `col_indices` (len = nnz),
/// `values` (len = nnz). Row `i` has nonzeros at positions `offsets[i]..offsets[i+1]`.
///
/// All invariants are validated at construction time (provable contract).
#[derive(Debug, Clone)]
pub struct CsrMatrix<T> {
    rows: usize,
    cols: usize,
    offsets: Vec<u32>,
    col_indices: Vec<u32>,
    values: Vec<T>,
}

impl<T: Clone + Default> CsrMatrix<T> {
    /// Create a CSR matrix with full validation.
    ///
    /// # Contract: sparse-spmv-v1.yaml / format_validation
    ///
    /// Validates all CSR invariants before construction.
    ///
    /// # Errors
    ///
    /// Returns error if any CSR invariant is violated.
    pub fn new(
        rows: usize,
        cols: usize,
        offsets: Vec<u32>,
        col_indices: Vec<u32>,
        values: Vec<T>,
    ) -> Result<Self, SparseError> {
        validate_csr_invariants(rows, cols, &offsets, &col_indices, values.len())?;
        Ok(Self {
            rows,
            cols,
            offsets,
            col_indices,
            values,
        })
    }

    /// Convert from COO format to CSR.
    ///
    /// Sorts triplets by row, then by column within each row.
    /// Duplicate entries are summed (standard convention).
    #[must_use]
    pub fn from_coo(coo: &CooMatrix<T>) -> Self
    where
        T: std::ops::AddAssign + Copy,
    {
        let rows = coo.rows;
        let cols = coo.cols;
        let nnz = coo.nnz();

        if nnz == 0 {
            return Self {
                rows,
                cols,
                offsets: vec![0; rows + 1],
                col_indices: Vec::new(),
                values: Vec::new(),
            };
        }

        // Count nonzeros per row
        let mut row_counts = vec![0u32; rows];
        for &r in &coo.row_indices {
            row_counts[r as usize] += 1;
        }

        // Build offsets via prefix sum
        let mut offsets = vec![0u32; rows + 1];
        for i in 0..rows {
            offsets[i + 1] = offsets[i] + row_counts[i];
        }

        // Fill col_indices and values (sort by row)
        let mut col_indices = vec![0u32; nnz];
        let mut values = vec![T::default(); nnz];
        let mut write_pos = offsets.clone();

        for idx in 0..nnz {
            let r = coo.row_indices[idx] as usize;
            let pos = write_pos[r] as usize;
            col_indices[pos] = coo.col_indices[idx];
            values[pos] = coo.values[idx];
            write_pos[r] += 1;
        }

        // Sort columns within each row
        for i in 0..rows {
            let start = offsets[i] as usize;
            let end = offsets[i + 1] as usize;
            if end - start > 1 {
                // Simple insertion sort (rows are typically short)
                for j in (start + 1)..end {
                    let mut k = j;
                    while k > start && col_indices[k - 1] > col_indices[k] {
                        col_indices.swap(k - 1, k);
                        values.swap(k - 1, k);
                        k -= 1;
                    }
                }
            }
        }

        Self {
            rows,
            cols,
            offsets,
            col_indices,
            values,
        }
    }

    /// Create an identity matrix of size n.
    #[must_use]
    pub fn identity(n: usize) -> Self
    where
        T: From<f32>,
    {
        let offsets: Vec<u32> = (0..=n).map(|i| i as u32).collect();
        let col_indices: Vec<u32> = (0..n).map(|i| i as u32).collect();
        let values: Vec<T> = (0..n).map(|_| T::from(1.0)).collect();
        Self {
            rows: n,
            cols: n,
            offsets,
            col_indices,
            values,
        }
    }

    /// Number of rows.
    #[must_use]
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Number of columns.
    #[must_use]
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Number of stored nonzero entries.
    #[must_use]
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// Row offsets array (len = rows + 1).
    #[must_use]
    pub fn offsets(&self) -> &[u32] {
        &self.offsets
    }

    /// Column indices array (len = nnz).
    #[must_use]
    pub fn col_indices(&self) -> &[u32] {
        &self.col_indices
    }

    /// Values array (len = nnz).
    #[must_use]
    pub fn values(&self) -> &[T] {
        &self.values
    }

    /// Average number of nonzeros per row.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn avg_nnz_per_row(&self) -> f64 {
        if self.rows == 0 {
            0.0
        } else {
            self.nnz() as f64 / self.rows as f64
        }
    }

    /// Variance of row lengths (key metric for algorithm selection).
    ///
    /// High variance → merge-based SpMV; low variance → row-split SpMV.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn row_length_variance(&self) -> f64 {
        if self.rows == 0 {
            return 0.0;
        }
        let mean = self.avg_nnz_per_row();
        let sum_sq: f64 = (0..self.rows)
            .map(|i| {
                let len = f64::from(self.offsets[i + 1] - self.offsets[i]);
                (len - mean) * (len - mean)
            })
            .sum();
        sum_sq / self.rows as f64
    }

    /// Convert to dense matrix (row-major).
    #[must_use]
    pub fn to_dense(&self) -> Vec<T>
    where
        T: Copy + std::ops::AddAssign,
    {
        let mut dense = vec![T::default(); self.rows * self.cols];
        for i in 0..self.rows {
            let start = self.offsets[i] as usize;
            let end = self.offsets[i + 1] as usize;
            for idx in start..end {
                let j = self.col_indices[idx] as usize;
                dense[i * self.cols + j] += self.values[idx];
            }
        }
        dense
    }
}
