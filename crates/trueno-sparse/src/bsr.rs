//! Block Sparse Row (BSR) format.
//!
//! Stores sparse matrices as blocks of dense sub-matrices, aligned on a
//! regular block grid. Efficient for FEM and structured sparsity patterns.

use crate::csr::CsrMatrix;
use crate::error::SparseError;
use crate::ops::SparseOps;

/// Block Sparse Row matrix.
///
/// A matrix of shape `(block_rows * block_size) × (block_cols * block_size)`,
/// where non-zero blocks are stored in CSR-of-blocks layout.
#[derive(Debug, Clone)]
pub struct BsrMatrix {
    /// Number of block rows.
    block_rows: usize,
    /// Number of block columns.
    block_cols: usize,
    /// Block dimension (blocks are block_size × block_size).
    block_size: usize,
    /// Row offsets for block CSR (length = block_rows + 1).
    offsets: Vec<u32>,
    /// Block column indices.
    col_indices: Vec<u32>,
    /// Dense block values, stored row-major per block.
    /// Length = nnz_blocks * block_size * block_size.
    values: Vec<f32>,
}

impl BsrMatrix {
    /// Create a new BSR matrix.
    ///
    /// # Arguments
    ///
    /// - `block_rows`, `block_cols`: number of block rows/columns
    /// - `block_size`: dimension of each square block
    /// - `offsets`: CSR-style row offsets for blocks
    /// - `col_indices`: block column indices
    /// - `values`: dense block data (row-major per block)
    ///
    /// # Errors
    ///
    /// Returns error if structure is invalid.
    pub fn new(
        block_rows: usize,
        block_cols: usize,
        block_size: usize,
        offsets: Vec<u32>,
        col_indices: Vec<u32>,
        values: Vec<f32>,
    ) -> Result<Self, SparseError> {
        if offsets.len() != block_rows + 1 {
            return Err(SparseError::InvalidOffsetsLength {
                actual: offsets.len(),
                expected: block_rows + 1,
            });
        }
        let nnz_blocks = col_indices.len();
        let expected_vals = nnz_blocks * block_size * block_size;
        if values.len() != expected_vals {
            return Err(SparseError::LengthMismatch {
                col_len: expected_vals,
                val_len: values.len(),
            });
        }
        Ok(Self { block_rows, block_cols, block_size, offsets, col_indices, values })
    }

    /// Create BSR from a dense matrix.
    ///
    /// Pads the matrix if dimensions aren't divisible by block_size.
    /// Only stores blocks with at least one non-zero element.
    pub fn from_dense(data: &[f32], rows: usize, cols: usize, block_size: usize) -> Self {
        let br = rows.div_ceil(block_size);
        let bc = cols.div_ceil(block_size);

        let mut offsets = vec![0u32; br + 1];
        let mut col_indices = Vec::new();
        let mut values = Vec::new();
        let bs2 = block_size * block_size;

        for bi in 0..br {
            for bj in 0..bc {
                let mut block = vec![0.0f32; bs2];
                let mut has_nonzero = false;
                for li in 0..block_size {
                    for lj in 0..block_size {
                        let gi = bi * block_size + li;
                        let gj = bj * block_size + lj;
                        if gi < rows && gj < cols {
                            let val = data[gi * cols + gj];
                            block[li * block_size + lj] = val;
                            if val != 0.0 {
                                has_nonzero = true;
                            }
                        }
                    }
                }
                if has_nonzero {
                    col_indices.push(bj as u32);
                    values.extend_from_slice(&block);
                }
            }
            offsets[bi + 1] = col_indices.len() as u32;
        }

        Self { block_rows: br, block_cols: bc, block_size, offsets, col_indices, values }
    }

    /// Convert to CSR format.
    ///
    /// # Errors
    ///
    /// Returns error if the internal conversion produces invalid CSR.
    pub fn to_csr(&self) -> Result<CsrMatrix<f32>, SparseError> {
        let rows = self.block_rows * self.block_size;
        let cols = self.block_cols * self.block_size;
        let bs = self.block_size;
        let bs2 = bs * bs;

        let mut csr_offsets = vec![0u32; rows + 1];
        let mut csr_cols = Vec::new();
        let mut csr_vals = Vec::new();

        for bi in 0..self.block_rows {
            let blk_start = self.offsets[bi] as usize;
            let blk_end = self.offsets[bi + 1] as usize;

            for li in 0..bs {
                let global_row = bi * bs + li;
                if global_row >= rows {
                    break;
                }
                for blk_idx in blk_start..blk_end {
                    let bj = self.col_indices[blk_idx] as usize;
                    for lj in 0..bs {
                        let global_col = bj * bs + lj;
                        if global_col >= cols {
                            continue;
                        }
                        let val = self.values[blk_idx * bs2 + li * bs + lj];
                        if val != 0.0 {
                            csr_cols.push(global_col as u32);
                            csr_vals.push(val);
                        }
                    }
                }
                csr_offsets[global_row + 1] = csr_cols.len() as u32;
            }
        }

        CsrMatrix::new(rows, cols, csr_offsets, csr_cols, csr_vals)
    }

    /// Total matrix rows.
    pub fn rows(&self) -> usize {
        self.block_rows * self.block_size
    }

    /// Total matrix columns.
    pub fn cols(&self) -> usize {
        self.block_cols * self.block_size
    }

    /// Number of non-zero blocks.
    pub fn nnz_blocks(&self) -> usize {
        self.col_indices.len()
    }

    /// Block size.
    pub fn block_size(&self) -> usize {
        self.block_size
    }
}

impl SparseOps for BsrMatrix {
    fn spmv(&self, alpha: f32, x: &[f32], beta: f32, y: &mut [f32]) -> Result<(), SparseError> {
        if x.len() != self.cols() {
            return Err(SparseError::SpMVDimensionMismatch {
                matrix_cols: self.cols(),
                x_len: x.len(),
            });
        }
        if y.len() != self.rows() {
            return Err(SparseError::SpMVOutputDimensionMismatch {
                matrix_rows: self.rows(),
                y_len: y.len(),
            });
        }

        let bs = self.block_size;
        let bs2 = bs * bs;

        // y = beta * y
        for yi in y.iter_mut() {
            *yi *= beta;
        }

        // y += alpha * A * x
        for bi in 0..self.block_rows {
            let blk_start = self.offsets[bi] as usize;
            let blk_end = self.offsets[bi + 1] as usize;

            for blk_idx in blk_start..blk_end {
                let bj = self.col_indices[blk_idx] as usize;
                let block = &self.values[blk_idx * bs2..(blk_idx + 1) * bs2];

                for li in 0..bs {
                    let gi = bi * bs + li;
                    if gi >= y.len() {
                        break;
                    }
                    let mut sum = 0.0f32;
                    for lj in 0..bs {
                        let gj = bj * bs + lj;
                        if gj < x.len() {
                            sum += block[li * bs + lj] * x[gj];
                        }
                    }
                    y[gi] += alpha * sum;
                }
            }
        }

        Ok(())
    }

    fn spmm(
        &self,
        alpha: f32,
        b: &[f32],
        b_cols: usize,
        beta: f32,
        c: &mut [f32],
    ) -> Result<(), SparseError> {
        if b.len() != self.cols() * b_cols {
            return Err(SparseError::SpMVDimensionMismatch {
                matrix_cols: self.cols(),
                x_len: b.len(),
            });
        }
        if c.len() != self.rows() * b_cols {
            return Err(SparseError::SpMVOutputDimensionMismatch {
                matrix_rows: self.rows(),
                y_len: c.len(),
            });
        }

        let bs = self.block_size;
        let bs2 = bs * bs;

        // Scale C by beta
        for ci in c.iter_mut() {
            *ci *= beta;
        }

        // C += alpha * A * B
        for bi in 0..self.block_rows {
            let blk_start = self.offsets[bi] as usize;
            let blk_end = self.offsets[bi + 1] as usize;

            for blk_idx in blk_start..blk_end {
                let bj = self.col_indices[blk_idx] as usize;
                let block = &self.values[blk_idx * bs2..(blk_idx + 1) * bs2];

                for li in 0..bs {
                    let gi = bi * bs + li;
                    if gi >= self.rows() {
                        break;
                    }
                    for lj in 0..bs {
                        let gj = bj * bs + lj;
                        if gj >= self.cols() {
                            continue;
                        }
                        let a_val = alpha * block[li * bs + lj];
                        for k in 0..b_cols {
                            c[gi * b_cols + k] += a_val * b[gj * b_cols + k];
                        }
                    }
                }
            }
        }

        Ok(())
    }
}
