//! Sliced ELLPACK (SELL) sparse matrix format.
//!
//! # Contract: sparse-formats-v1.yaml
//!
//! SELL-C-σ format: rows sorted by length within slices of C rows.
//! Each slice is padded to the max row length in that slice.
//! This gives SIMD-friendly contiguous access patterns.
//!
//! ## References
//! - Kreutzer et al., "A unified sparse matrix data format for modern processors", 2014

use crate::csr::CsrMatrix;
use crate::error::SparseError;

/// Sliced ELLPACK sparse matrix.
///
/// Rows are grouped into slices of `slice_size` rows. Within each slice,
/// columns and values are stored in column-major order, padded to the
/// max row length in that slice.
#[derive(Debug, Clone)]
pub struct SellMatrix {
    rows: usize,
    cols: usize,
    slice_size: usize,
    /// Number of slices = ceil(rows / slice_size).
    num_slices: usize,
    /// Offset into col_indices/values for each slice (len = num_slices + 1).
    slice_offsets: Vec<u32>,
    /// Max row length in each slice (len = num_slices).
    slice_widths: Vec<u32>,
    /// Column indices (padded, column-major within each slice).
    col_indices: Vec<u32>,
    /// Values (padded, column-major within each slice).
    values: Vec<f32>,
}

impl SellMatrix {
    /// Convert a CSR matrix to SELL format with the given slice size.
    ///
    /// Typical slice_size: 32 or 64 (matching SIMD width or warp size).
    #[must_use]
    pub fn from_csr(csr: &CsrMatrix<f32>, slice_size: usize) -> Self {
        let rows = csr.rows();
        let cols = csr.cols();
        let c = if slice_size == 0 { 1 } else { slice_size };
        let num_slices = rows.div_ceil(c);

        let mut slice_offsets = Vec::with_capacity(num_slices + 1);
        let mut slice_widths = Vec::with_capacity(num_slices);
        let mut col_indices = Vec::new();
        let mut values = Vec::new();

        slice_offsets.push(0u32);

        for s in 0..num_slices {
            let row_start = s * c;
            let row_end = (row_start + c).min(rows);
            let actual_rows = row_end - row_start;

            // Find max row length in this slice
            let max_len = compute_slice_width(csr, row_start, row_end);
            slice_widths.push(max_len as u32);

            // Store in column-major order within the slice
            fill_slice_data(csr, row_start, actual_rows, c, max_len, &mut col_indices, &mut values);

            let slice_elements = c * max_len;
            let offset = slice_offsets.last().copied().unwrap_or(0);
            slice_offsets.push(offset + slice_elements as u32);
        }

        Self {
            rows,
            cols,
            slice_size: c,
            num_slices,
            slice_offsets,
            slice_widths,
            col_indices,
            values,
        }
    }

    /// SpMV: y = α·A·x + β·y
    ///
    /// # Errors
    ///
    /// Returns error on dimension mismatch.
    pub fn spmv(&self, alpha: f32, x: &[f32], beta: f32, y: &mut [f32]) -> Result<(), SparseError> {
        if x.len() != self.cols {
            return Err(SparseError::SpMVDimensionMismatch {
                matrix_cols: self.cols,
                x_len: x.len(),
            });
        }
        if y.len() != self.rows {
            return Err(SparseError::SpMVOutputDimensionMismatch {
                matrix_rows: self.rows,
                y_len: y.len(),
            });
        }

        // Scale y by beta
        for val in y.iter_mut() {
            *val *= beta;
        }

        let c = self.slice_size;

        for s in 0..self.num_slices {
            let base = self.slice_offsets[s] as usize;
            let width = self.slice_widths[s] as usize;
            let row_start = s * c;
            let row_end = (row_start + c).min(self.rows);

            spmv_slice(
                &self.col_indices,
                &self.values,
                x,
                y,
                alpha,
                base,
                c,
                width,
                row_start,
                row_end,
            );
        }

        Ok(())
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

    /// Slice size (C parameter).
    #[must_use]
    pub fn slice_size(&self) -> usize {
        self.slice_size
    }

    /// Total stored elements (including padding zeros).
    #[must_use]
    pub fn storage_size(&self) -> usize {
        self.values.len()
    }
}

/// Compute max row length in a slice.
fn compute_slice_width(csr: &CsrMatrix<f32>, row_start: usize, row_end: usize) -> usize {
    let offsets = csr.offsets();
    let mut max_len = 0usize;
    for r in row_start..row_end {
        let len = (offsets[r + 1] - offsets[r]) as usize;
        if len > max_len {
            max_len = len;
        }
    }
    max_len
}

/// Fill column-major data for one slice.
fn fill_slice_data(
    csr: &CsrMatrix<f32>,
    row_start: usize,
    actual_rows: usize,
    c: usize,
    max_len: usize,
    col_indices: &mut Vec<u32>,
    values: &mut Vec<f32>,
) {
    let csr_off = csr.offsets();
    let csr_cols = csr.col_indices();
    let csr_vals = csr.values();

    // Column-major: for each column position j, store all rows
    for j in 0..max_len {
        for local_r in 0..c {
            let global_r = row_start + local_r;
            if local_r < actual_rows {
                let row_start_idx = csr_off[global_r] as usize;
                let row_len = (csr_off[global_r + 1] - csr_off[global_r]) as usize;
                if j < row_len {
                    col_indices.push(csr_cols[row_start_idx + j]);
                    values.push(csr_vals[row_start_idx + j]);
                } else {
                    col_indices.push(0);
                    values.push(0.0);
                }
            } else {
                // Padding rows (beyond actual matrix rows)
                col_indices.push(0);
                values.push(0.0);
            }
        }
    }
}

/// SpMV for one SELL slice.
#[allow(clippy::too_many_arguments)]
fn spmv_slice(
    col_indices: &[u32],
    values: &[f32],
    x: &[f32],
    y: &mut [f32],
    alpha: f32,
    base: usize,
    c: usize,
    width: usize,
    row_start: usize,
    row_end: usize,
) {
    for j in 0..width {
        for local_r in 0..(row_end - row_start) {
            let idx = base + j * c + local_r;
            let col = col_indices[idx] as usize;
            let val = values[idx];
            y[row_start + local_r] += alpha * val * x[col];
        }
    }
}
