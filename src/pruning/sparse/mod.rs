//! Sparse tensor representations for pruned neural networks.
//!
//! # Toyota Way: Muda (Waste Elimination)
//! Sparse formats store only non-zero values, eliminating memory waste
//! from storing zeros in pruned networks.
//!
//! # Supported Formats
//! - **CSR** (Compressed Sparse Row): Efficient for row-wise operations
//! - **COO** (Coordinate): Simple format, good for construction
//! - **Dense**: Standard dense tensor for comparison/conversion
//!
//! # Hardware Acceleration
//! - 2:4 structured sparsity maps to NVIDIA Ampere sparse tensor cores
//! - Block sparsity enables efficient dense submatrix operations
//!
//! # References
//! - Mishra, A., et al. (2021). Accelerating sparse deep neural networks.

use super::error::PruningError;
use super::mask::SparsityMask;
use crate::autograd::Tensor;

/// Sparse tensor format enumeration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SparseFormat {
    /// Compressed Sparse Row format.
    /// Efficient for row-wise access and matrix-vector multiplication.
    CSR,
    /// Coordinate format.
    /// Simple construction, good for incremental building.
    COO,
    /// Block sparse format with fixed block size.
    /// Efficient when sparsity has block structure.
    Block { height: usize, width: usize },
}

/// Compressed Sparse Row (CSR) representation.
///
/// # Memory Layout
/// For a matrix with `nnz` non-zeros and `nrows` rows:
/// - `values`: \[nnz\] - Non-zero values in row-major order
/// - `col_indices`: \[nnz\] - Column index for each value
/// - `row_ptrs`: \[nrows + 1\] - Start index in `values/col_indices` for each row
///
/// # Example
/// ```text
/// Dense:     [[1, 0, 2],    CSR:
///             [0, 0, 3],    values: [1, 2, 3, 4]
///             [4, 0, 0]]    col_indices: [0, 2, 2, 0]
///                           row_ptrs: [0, 2, 3, 4]
/// ```
#[derive(Debug, Clone)]
pub struct CSRTensor {
    /// Non-zero values.
    pub values: Vec<f32>,
    /// Column indices for each value.
    pub col_indices: Vec<usize>,
    /// Row pointers (start index for each row).
    pub row_ptrs: Vec<usize>,
    /// Number of rows.
    pub nrows: usize,
    /// Number of columns.
    pub ncols: usize,
}

impl CSRTensor {
    /// Create a new CSR tensor from components.
    ///
    /// # Arguments
    /// * `values` - Non-zero values
    /// * `col_indices` - Column index for each value
    /// * `row_ptrs` - Row pointer array
    /// * `nrows` - Number of rows
    /// * `ncols` - Number of columns
    pub fn new(
        values: Vec<f32>,
        col_indices: Vec<usize>,
        row_ptrs: Vec<usize>,
        nrows: usize,
        ncols: usize,
    ) -> Result<Self, PruningError> {
        // Validate lengths
        if values.len() != col_indices.len() {
            return Err(PruningError::InvalidMask {
                reason: format!(
                    "values length ({}) != col_indices length ({})",
                    values.len(),
                    col_indices.len()
                ),
            });
        }

        if row_ptrs.len() != nrows + 1 {
            return Err(PruningError::InvalidMask {
                reason: format!(
                    "row_ptrs length ({}) != nrows + 1 ({})",
                    row_ptrs.len(),
                    nrows + 1
                ),
            });
        }

        // Validate row_ptrs is monotonically increasing
        for i in 1..row_ptrs.len() {
            if row_ptrs[i] < row_ptrs[i - 1] {
                return Err(PruningError::InvalidMask {
                    reason: format!(
                        "row_ptrs not monotonic at index {}: {} < {}",
                        i,
                        row_ptrs[i],
                        row_ptrs[i - 1]
                    ),
                });
            }
        }

        // Validate col_indices are in bounds
        for &col in &col_indices {
            if col >= ncols {
                return Err(PruningError::InvalidMask {
                    reason: format!("col_index {col} >= ncols {ncols}"),
                });
            }
        }

        Ok(Self {
            values,
            col_indices,
            row_ptrs,
            nrows,
            ncols,
        })
    }

    /// Create CSR tensor from dense tensor.
    pub fn from_dense(tensor: &Tensor) -> Result<Self, PruningError> {
        let shape = tensor.shape();
        if shape.len() != 2 {
            return Err(PruningError::ShapeMismatch {
                expected: vec![0, 0],
                got: shape.to_vec(),
            });
        }

        let nrows = shape[0];
        let ncols = shape[1];
        let data = tensor.data();

        let mut values = Vec::new();
        let mut col_indices = Vec::new();
        let mut row_ptrs = vec![0];

        for row in 0..nrows {
            for col in 0..ncols {
                let val = data[row * ncols + col];
                if val != 0.0 {
                    values.push(val);
                    col_indices.push(col);
                }
            }
            row_ptrs.push(values.len());
        }

        Ok(Self {
            values,
            col_indices,
            row_ptrs,
            nrows,
            ncols,
        })
    }

    /// Convert CSR tensor back to dense tensor.
    #[must_use]
    pub fn to_dense(&self) -> Tensor {
        let mut data = vec![0.0f32; self.nrows * self.ncols];

        for row in 0..self.nrows {
            let start = self.row_ptrs[row];
            let end = self.row_ptrs[row + 1];

            for idx in start..end {
                let col = self.col_indices[idx];
                let val = self.values[idx];
                data[row * self.ncols + col] = val;
            }
        }

        Tensor::new(&data, &[self.nrows, self.ncols])
    }

    /// Get number of non-zero elements.
    #[must_use]
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// Get sparsity ratio (fraction of zeros).
    #[must_use]
    pub fn sparsity(&self) -> f32 {
        let total = self.nrows * self.ncols;
        if total == 0 {
            return 0.0;
        }
        1.0 - (self.nnz() as f32 / total as f32)
    }

    /// Get shape as [rows, cols].
    #[must_use]
    pub fn shape(&self) -> [usize; 2] {
        [self.nrows, self.ncols]
    }

    /// Get value at (row, col), or 0 if not present.
    #[must_use]
    pub fn get(&self, row: usize, col: usize) -> f32 {
        if row >= self.nrows || col >= self.ncols {
            return 0.0;
        }

        let start = self.row_ptrs[row];
        let end = self.row_ptrs[row + 1];

        for idx in start..end {
            if self.col_indices[idx] == col {
                return self.values[idx];
            }
        }

        0.0
    }

    /// Sparse matrix-vector multiplication: y = A * x
    ///
    /// # Arguments
    /// * `x` - Input vector of length ncols
    ///
    /// # Returns
    /// Output vector of length nrows
    pub fn matvec(&self, x: &[f32]) -> Result<Vec<f32>, PruningError> {
        if x.len() != self.ncols {
            return Err(PruningError::ShapeMismatch {
                expected: vec![self.ncols],
                got: vec![x.len()],
            });
        }

        let mut y = vec![0.0f32; self.nrows];

        for row in 0..self.nrows {
            let start = self.row_ptrs[row];
            let end = self.row_ptrs[row + 1];

            let mut sum = 0.0f32;
            for idx in start..end {
                let col = self.col_indices[idx];
                sum += self.values[idx] * x[col];
            }
            y[row] = sum;
        }

        Ok(y)
    }

    /// Memory usage in bytes (approximate).
    #[must_use]
    pub fn memory_bytes(&self) -> usize {
        let values_bytes = self.values.len() * size_of::<f32>();
        let col_indices_bytes = self.col_indices.len() * size_of::<usize>();
        let row_ptrs_bytes = self.row_ptrs.len() * size_of::<usize>();
        values_bytes + col_indices_bytes + row_ptrs_bytes
    }

    /// Dense memory usage for comparison.
    #[must_use]
    pub fn dense_memory_bytes(&self) -> usize {
        self.nrows * self.ncols * size_of::<f32>()
    }

    /// Memory savings ratio (dense / sparse).
    #[must_use]
    pub fn memory_savings_ratio(&self) -> f32 {
        let sparse = self.memory_bytes();
        let dense = self.dense_memory_bytes();
        if sparse == 0 {
            return 1.0;
        }
        dense as f32 / sparse as f32
    }
}

/// Coordinate (COO) sparse tensor representation.
///
/// # Memory Layout
/// - `values`: \[nnz\] - Non-zero values
/// - `row_indices`: \[nnz\] - Row index for each value
/// - `col_indices`: \[nnz\] - Column index for each value
///
/// Simple format, good for construction and conversion.
#[derive(Debug, Clone)]
pub struct COOTensor {
    /// Non-zero values.
    pub values: Vec<f32>,
    /// Row indices.
    pub row_indices: Vec<usize>,
    /// Column indices.
    pub col_indices: Vec<usize>,
    /// Number of rows.
    pub nrows: usize,
    /// Number of columns.
    pub ncols: usize,
}

impl COOTensor {
    /// Create a new empty COO tensor.
    #[must_use]
    pub fn new(nrows: usize, ncols: usize) -> Self {
        Self {
            values: Vec::new(),
            row_indices: Vec::new(),
            col_indices: Vec::new(),
            nrows,
            ncols,
        }
    }

    /// Create COO tensor from dense tensor.
    pub fn from_dense(tensor: &Tensor) -> Result<Self, PruningError> {
        let shape = tensor.shape();
        if shape.len() != 2 {
            return Err(PruningError::ShapeMismatch {
                expected: vec![0, 0],
                got: shape.to_vec(),
            });
        }

        let nrows = shape[0];
        let ncols = shape[1];
        let data = tensor.data();

        let mut coo = Self::new(nrows, ncols);

        for row in 0..nrows {
            for col in 0..ncols {
                let val = data[row * ncols + col];
                if val != 0.0 {
                    coo.push(row, col, val);
                }
            }
        }

        Ok(coo)
    }

    /// Add a non-zero element.
    pub fn push(&mut self, row: usize, col: usize, value: f32) {
        self.values.push(value);
        self.row_indices.push(row);
        self.col_indices.push(col);
    }

    /// Convert to dense tensor.
    #[must_use]
    pub fn to_dense(&self) -> Tensor {
        let mut data = vec![0.0f32; self.nrows * self.ncols];

        for i in 0..self.values.len() {
            let row = self.row_indices[i];
            let col = self.col_indices[i];
            if row < self.nrows && col < self.ncols {
                data[row * self.ncols + col] = self.values[i];
            }
        }

        Tensor::new(&data, &[self.nrows, self.ncols])
    }

    /// Convert COO to CSR format.
    pub fn to_csr(&self) -> Result<CSRTensor, PruningError> {
        // Sort by row, then by column
        let mut entries: Vec<(usize, usize, f32)> = self
            .values
            .iter()
            .zip(self.row_indices.iter())
            .zip(self.col_indices.iter())
            .map(|((&v, &r), &c)| (r, c, v))
            .collect();

        entries.sort_by(|a, b| {
            if a.0 == b.0 {
                a.1.cmp(&b.1)
            } else {
                a.0.cmp(&b.0)
            }
        });

        let mut values = Vec::with_capacity(entries.len());
        let mut col_indices = Vec::with_capacity(entries.len());
        let mut row_ptrs = vec![0usize; self.nrows + 1];

        for (row, col, val) in entries {
            values.push(val);
            col_indices.push(col);
            row_ptrs[row + 1] += 1;
        }

        // Convert counts to cumulative pointers
        for i in 1..row_ptrs.len() {
            row_ptrs[i] += row_ptrs[i - 1];
        }

        CSRTensor::new(values, col_indices, row_ptrs, self.nrows, self.ncols)
    }

    /// Get number of non-zero elements.
    #[must_use]
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// Get sparsity ratio.
    #[must_use]
    pub fn sparsity(&self) -> f32 {
        let total = self.nrows * self.ncols;
        if total == 0 {
            return 0.0;
        }
        1.0 - (self.nnz() as f32 / total as f32)
    }

    /// Get shape as [rows, cols].
    #[must_use]
    pub fn shape(&self) -> [usize; 2] {
        [self.nrows, self.ncols]
    }
}

include!("mod_part_02.rs");
