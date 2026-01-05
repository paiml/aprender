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
/// - `values`: [nnz] - Non-zero values in row-major order
/// - `col_indices`: [nnz] - Column index for each value
/// - `row_ptrs`: [nrows + 1] - Start index in `values/col_indices` for each row
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
/// - `values`: [nnz] - Non-zero values
/// - `row_indices`: [nnz] - Row index for each value
/// - `col_indices`: [nnz] - Column index for each value
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

/// Apply a sparsity mask to a tensor and return sparse representation.
///
/// # Arguments
/// * `tensor` - Dense tensor to sparsify
/// * `mask` - Binary sparsity mask (1 = keep, 0 = prune)
/// * `format` - Target sparse format
///
/// # Returns
/// Sparse tensor in requested format
pub fn sparsify(
    tensor: &Tensor,
    mask: &SparsityMask,
    format: SparseFormat,
) -> Result<SparseTensor, PruningError> {
    // Apply mask to get sparse dense tensor
    let mut masked = tensor.clone();
    mask.apply(&mut masked)?;

    match format {
        SparseFormat::CSR => {
            let csr = CSRTensor::from_dense(&masked)?;
            Ok(SparseTensor::CSR(csr))
        }
        SparseFormat::COO => {
            let coo = COOTensor::from_dense(&masked)?;
            Ok(SparseTensor::COO(coo))
        }
        SparseFormat::Block { height, width } => {
            let block = BlockSparseTensor::from_dense(&masked, height, width)?;
            Ok(SparseTensor::Block(block))
        }
    }
}

/// Unified sparse tensor type.
#[derive(Debug, Clone)]
pub enum SparseTensor {
    /// CSR format.
    CSR(CSRTensor),
    /// COO format.
    COO(COOTensor),
    /// Block sparse format.
    Block(BlockSparseTensor),
}

impl SparseTensor {
    /// Convert to dense tensor.
    #[must_use]
    pub fn to_dense(&self) -> Tensor {
        match self {
            SparseTensor::CSR(csr) => csr.to_dense(),
            SparseTensor::COO(coo) => coo.to_dense(),
            SparseTensor::Block(block) => block.to_dense(),
        }
    }

    /// Get number of non-zero elements.
    #[must_use]
    pub fn nnz(&self) -> usize {
        match self {
            SparseTensor::CSR(csr) => csr.nnz(),
            SparseTensor::COO(coo) => coo.nnz(),
            SparseTensor::Block(block) => block.nnz(),
        }
    }

    /// Get sparsity ratio.
    #[must_use]
    pub fn sparsity(&self) -> f32 {
        match self {
            SparseTensor::CSR(csr) => csr.sparsity(),
            SparseTensor::COO(coo) => coo.sparsity(),
            SparseTensor::Block(block) => block.sparsity(),
        }
    }

    /// Get format type.
    #[must_use]
    pub fn format(&self) -> SparseFormat {
        match self {
            SparseTensor::CSR(_) => SparseFormat::CSR,
            SparseTensor::COO(_) => SparseFormat::COO,
            SparseTensor::Block(b) => SparseFormat::Block {
                height: b.block_height,
                width: b.block_width,
            },
        }
    }
}

/// Block sparse tensor representation.
///
/// Stores non-zero blocks as dense submatrices.
/// Efficient when sparsity has natural block structure.
#[derive(Debug, Clone)]
pub struct BlockSparseTensor {
    /// Dense blocks stored as flattened data.
    pub blocks: Vec<Vec<f32>>,
    /// Block row indices.
    pub block_row_indices: Vec<usize>,
    /// Block column indices.
    pub block_col_indices: Vec<usize>,
    /// Block height.
    pub block_height: usize,
    /// Block width.
    pub block_width: usize,
    /// Number of block rows.
    pub nblock_rows: usize,
    /// Number of block columns.
    pub nblock_cols: usize,
}

impl BlockSparseTensor {
    /// Create block sparse tensor from dense tensor.
    ///
    /// # Arguments
    /// * `tensor` - Dense 2D tensor
    /// * `block_height` - Height of each block
    /// * `block_width` - Width of each block
    pub fn from_dense(
        tensor: &Tensor,
        block_height: usize,
        block_width: usize,
    ) -> Result<Self, PruningError> {
        let shape = tensor.shape();
        if shape.len() != 2 {
            return Err(PruningError::ShapeMismatch {
                expected: vec![0, 0],
                got: shape.to_vec(),
            });
        }

        let nrows = shape[0];
        let ncols = shape[1];

        if block_height == 0 || block_width == 0 {
            return Err(PruningError::InvalidPattern {
                message: "Block dimensions must be > 0".to_string(),
            });
        }

        // Pad dimensions if needed
        let nblock_rows = (nrows + block_height - 1) / block_height;
        let nblock_cols = (ncols + block_width - 1) / block_width;

        let data = tensor.data();
        let mut blocks = Vec::new();
        let mut block_row_indices = Vec::new();
        let mut block_col_indices = Vec::new();

        for br in 0..nblock_rows {
            for bc in 0..nblock_cols {
                let mut block = vec![0.0f32; block_height * block_width];
                let mut is_nonzero = false;

                for i in 0..block_height {
                    for j in 0..block_width {
                        let row = br * block_height + i;
                        let col = bc * block_width + j;

                        if row < nrows && col < ncols {
                            let val = data[row * ncols + col];
                            block[i * block_width + j] = val;
                            if val != 0.0 {
                                is_nonzero = true;
                            }
                        }
                    }
                }

                if is_nonzero {
                    blocks.push(block);
                    block_row_indices.push(br);
                    block_col_indices.push(bc);
                }
            }
        }

        Ok(Self {
            blocks,
            block_row_indices,
            block_col_indices,
            block_height,
            block_width,
            nblock_rows,
            nblock_cols,
        })
    }

    /// Convert to dense tensor.
    #[must_use]
    pub fn to_dense(&self) -> Tensor {
        let nrows = self.nblock_rows * self.block_height;
        let ncols = self.nblock_cols * self.block_width;
        let mut data = vec![0.0f32; nrows * ncols];

        for (idx, block) in self.blocks.iter().enumerate() {
            let br = self.block_row_indices[idx];
            let bc = self.block_col_indices[idx];

            for i in 0..self.block_height {
                for j in 0..self.block_width {
                    let row = br * self.block_height + i;
                    let col = bc * self.block_width + j;
                    if row < nrows && col < ncols {
                        data[row * ncols + col] = block[i * self.block_width + j];
                    }
                }
            }
        }

        Tensor::new(&data, &[nrows, ncols])
    }

    /// Get number of non-zero elements.
    #[must_use]
    pub fn nnz(&self) -> usize {
        self.blocks
            .iter()
            .flat_map(|b| b.iter())
            .filter(|&&v| v != 0.0)
            .count()
    }

    /// Get sparsity ratio.
    #[must_use]
    pub fn sparsity(&self) -> f32 {
        let total = self.nblock_rows * self.block_height * self.nblock_cols * self.block_width;
        if total == 0 {
            return 0.0;
        }
        1.0 - (self.nnz() as f32 / total as f32)
    }

    /// Get number of non-zero blocks.
    #[must_use]
    pub fn num_blocks(&self) -> usize {
        self.blocks.len()
    }

    /// Get shape as [rows, cols].
    #[must_use]
    pub fn shape(&self) -> [usize; 2] {
        [
            self.nblock_rows * self.block_height,
            self.nblock_cols * self.block_width,
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ==========================================================================
    // FALSIFICATION: CSR tensor construction
    // ==========================================================================
    #[test]
    fn test_csr_from_dense_basic() {
        let dense = Tensor::new(&[1.0, 0.0, 2.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0], &[3, 3]);

        let csr = CSRTensor::from_dense(&dense).unwrap();

        assert_eq!(csr.nrows, 3, "SPA-01 FALSIFIED: nrows should be 3");
        assert_eq!(csr.ncols, 3, "SPA-01 FALSIFIED: ncols should be 3");
        assert_eq!(csr.nnz(), 4, "SPA-01 FALSIFIED: nnz should be 4");
        assert_eq!(
            csr.values,
            vec![1.0, 2.0, 3.0, 4.0],
            "SPA-01 FALSIFIED: values incorrect"
        );
    }

    #[test]
    fn test_csr_roundtrip() {
        let original = Tensor::new(&[1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 5.0, 0.0], &[3, 3]);

        let csr = CSRTensor::from_dense(&original).unwrap();
        let recovered = csr.to_dense();

        assert_eq!(
            original.data(),
            recovered.data(),
            "SPA-02 FALSIFIED: CSR roundtrip should preserve data"
        );
    }

    #[test]
    fn test_csr_sparsity() {
        let dense = Tensor::new(&[1.0, 0.0, 0.0, 0.0], &[2, 2]);
        let csr = CSRTensor::from_dense(&dense).unwrap();

        assert!(
            (csr.sparsity() - 0.75).abs() < 1e-5,
            "SPA-03 FALSIFIED: Sparsity should be 0.75"
        );
    }

    #[test]
    fn test_csr_get() {
        let dense = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let csr = CSRTensor::from_dense(&dense).unwrap();

        assert_eq!(csr.get(0, 0), 1.0);
        assert_eq!(csr.get(0, 1), 2.0);
        assert_eq!(csr.get(1, 0), 3.0);
        assert_eq!(csr.get(1, 1), 4.0);
        assert_eq!(csr.get(5, 5), 0.0); // Out of bounds returns 0
    }

    #[test]
    fn test_csr_matvec() {
        // [[1, 2], [3, 4]] * [1, 1] = [3, 7]
        let dense = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let csr = CSRTensor::from_dense(&dense).unwrap();

        let x = vec![1.0, 1.0];
        let y = csr.matvec(&x).unwrap();

        assert!(
            (y[0] - 3.0).abs() < 1e-5,
            "SPA-04 FALSIFIED: matvec[0] should be 3"
        );
        assert!(
            (y[1] - 7.0).abs() < 1e-5,
            "SPA-04 FALSIFIED: matvec[1] should be 7"
        );
    }

    #[test]
    fn test_csr_matvec_shape_mismatch() {
        let dense = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let csr = CSRTensor::from_dense(&dense).unwrap();

        let result = csr.matvec(&[1.0]); // Wrong size
        assert!(
            result.is_err(),
            "SPA-05 FALSIFIED: Should error on shape mismatch"
        );
    }

    #[test]
    fn test_csr_memory_savings() {
        // Very sparse matrix
        let mut data = vec![0.0f32; 100];
        data[0] = 1.0;
        data[50] = 2.0;
        let dense = Tensor::new(&data, &[10, 10]);

        let csr = CSRTensor::from_dense(&dense).unwrap();

        assert!(
            csr.memory_savings_ratio() > 1.0,
            "SPA-06 FALSIFIED: Sparse should save memory"
        );
    }

    // ==========================================================================
    // FALSIFICATION: COO tensor construction
    // ==========================================================================
    #[test]
    fn test_coo_from_dense() {
        let dense = Tensor::new(&[1.0, 0.0, 0.0, 2.0], &[2, 2]);
        let coo = COOTensor::from_dense(&dense).unwrap();

        assert_eq!(coo.nnz(), 2, "SPA-07 FALSIFIED: COO nnz should be 2");
        assert_eq!(coo.shape(), [2, 2]);
    }

    #[test]
    fn test_coo_roundtrip() {
        let original = Tensor::new(&[1.0, 2.0, 0.0, 3.0], &[2, 2]);
        let coo = COOTensor::from_dense(&original).unwrap();
        let recovered = coo.to_dense();

        assert_eq!(
            original.data(),
            recovered.data(),
            "SPA-08 FALSIFIED: COO roundtrip should preserve data"
        );
    }

    #[test]
    fn test_coo_to_csr() {
        let dense = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let coo = COOTensor::from_dense(&dense).unwrap();
        let csr = coo.to_csr().unwrap();

        let csr_direct = CSRTensor::from_dense(&dense).unwrap();

        assert_eq!(
            csr.values, csr_direct.values,
            "SPA-09 FALSIFIED: COO->CSR should match direct CSR"
        );
    }

    #[test]
    fn test_coo_push() {
        let mut coo = COOTensor::new(3, 3);
        coo.push(0, 0, 1.0);
        coo.push(1, 2, 2.0);
        coo.push(2, 1, 3.0);

        assert_eq!(coo.nnz(), 3);

        let dense = coo.to_dense();
        assert_eq!(dense.data()[0], 1.0);
        assert_eq!(dense.data()[5], 2.0); // (1, 2) in row-major
        assert_eq!(dense.data()[7], 3.0); // (2, 1) in row-major
    }

    // ==========================================================================
    // FALSIFICATION: Block sparse tensor
    // ==========================================================================
    #[test]
    fn test_block_sparse_from_dense() {
        let dense = Tensor::new(
            &[
                1.0, 2.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
            &[4, 4],
        );

        let block = BlockSparseTensor::from_dense(&dense, 2, 2).unwrap();

        assert_eq!(block.block_height, 2);
        assert_eq!(block.block_width, 2);
        assert_eq!(
            block.num_blocks(),
            1,
            "SPA-10 FALSIFIED: Should have 1 non-zero block"
        );
    }

    #[test]
    fn test_block_sparse_roundtrip() {
        let original = Tensor::new(
            &[
                1.0, 2.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 5.0, 6.0, 0.0, 0.0, 7.0, 8.0,
            ],
            &[4, 4],
        );

        let block = BlockSparseTensor::from_dense(&original, 2, 2).unwrap();
        let recovered = block.to_dense();

        assert_eq!(
            original.data(),
            recovered.data(),
            "SPA-11 FALSIFIED: Block sparse roundtrip should preserve data"
        );
    }

    #[test]
    fn test_block_sparse_invalid_block_size() {
        let dense = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let result = BlockSparseTensor::from_dense(&dense, 0, 2);

        assert!(
            result.is_err(),
            "SPA-12 FALSIFIED: Should error on zero block size"
        );
    }

    // ==========================================================================
    // FALSIFICATION: Sparsify function
    // ==========================================================================
    #[test]
    fn test_sparsify_csr() {
        let tensor = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let mask_tensor = Tensor::new(&[1.0, 0.0, 0.0, 1.0], &[2, 2]);
        let mask = SparsityMask::new(
            mask_tensor,
            super::super::mask::SparsityPattern::Unstructured,
        )
        .unwrap();

        let sparse = sparsify(&tensor, &mask, SparseFormat::CSR).unwrap();

        assert_eq!(sparse.nnz(), 2, "SPA-13 FALSIFIED: Should have 2 non-zeros");
        assert!(
            (sparse.sparsity() - 0.5).abs() < 1e-5,
            "SPA-13 FALSIFIED: Sparsity should be 0.5"
        );
    }

    #[test]
    fn test_sparsify_coo() {
        let tensor = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let mask_tensor = Tensor::new(&[1.0, 1.0, 0.0, 0.0], &[2, 2]);
        let mask = SparsityMask::new(
            mask_tensor,
            super::super::mask::SparsityPattern::Unstructured,
        )
        .unwrap();

        let sparse = sparsify(&tensor, &mask, SparseFormat::COO).unwrap();

        assert_eq!(sparse.nnz(), 2);
        matches!(sparse.format(), SparseFormat::COO);
    }

    // ==========================================================================
    // FALSIFICATION: SparseTensor enum
    // ==========================================================================
    #[test]
    fn test_sparse_tensor_to_dense() {
        let dense = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let csr = CSRTensor::from_dense(&dense).unwrap();
        let sparse = SparseTensor::CSR(csr);

        let recovered = sparse.to_dense();
        assert_eq!(dense.data(), recovered.data());
    }

    #[test]
    fn test_sparse_tensor_format() {
        let dense = Tensor::new(&[1.0], &[1, 1]);

        let csr = SparseTensor::CSR(CSRTensor::from_dense(&dense).unwrap());
        assert_eq!(csr.format(), SparseFormat::CSR);

        let coo = SparseTensor::COO(COOTensor::from_dense(&dense).unwrap());
        assert_eq!(coo.format(), SparseFormat::COO);
    }

    // ==========================================================================
    // FALSIFICATION: Edge cases
    // ==========================================================================
    #[test]
    fn test_csr_empty_matrix() {
        let dense = Tensor::new(&[0.0, 0.0, 0.0, 0.0], &[2, 2]);
        let csr = CSRTensor::from_dense(&dense).unwrap();

        assert_eq!(csr.nnz(), 0);
        assert!((csr.sparsity() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_csr_full_matrix() {
        let dense = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let csr = CSRTensor::from_dense(&dense).unwrap();

        assert_eq!(csr.nnz(), 4);
        assert!(csr.sparsity().abs() < 1e-5);
    }

    #[test]
    fn test_csr_1d_rejected() {
        let dense = Tensor::new(&[1.0, 2.0, 3.0], &[3]);
        let result = CSRTensor::from_dense(&dense);

        assert!(
            result.is_err(),
            "SPA-14 FALSIFIED: 1D tensor should be rejected"
        );
    }

    #[test]
    fn test_coo_sparsity_empty() {
        let coo = COOTensor::new(0, 0);
        assert_eq!(coo.sparsity(), 0.0);
    }

    // ==========================================================================
    // FALSIFICATION: Clone and Debug
    // ==========================================================================
    #[test]
    fn test_csr_clone() {
        let dense = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let csr = CSRTensor::from_dense(&dense).unwrap();
        let cloned = csr.clone();

        assert_eq!(csr.values, cloned.values);
        assert_eq!(csr.nnz(), cloned.nnz());
    }

    #[test]
    fn test_csr_debug() {
        let dense = Tensor::new(&[1.0], &[1, 1]);
        let csr = CSRTensor::from_dense(&dense).unwrap();
        let debug = format!("{:?}", csr);
        assert!(debug.contains("CSRTensor"));
    }

    #[test]
    fn test_coo_debug() {
        let coo = COOTensor::new(2, 2);
        let debug = format!("{:?}", coo);
        assert!(debug.contains("COOTensor"));
    }

    #[test]
    fn test_block_sparse_debug() {
        let dense = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let block = BlockSparseTensor::from_dense(&dense, 1, 1).unwrap();
        let debug = format!("{:?}", block);
        assert!(debug.contains("BlockSparseTensor"));
    }

    #[test]
    fn test_sparse_format_debug() {
        let format = SparseFormat::CSR;
        let debug = format!("{:?}", format);
        assert!(debug.contains("CSR"));
    }

    // ==========================================================================
    // FALSIFICATION: CSR validation
    // ==========================================================================
    #[test]
    fn test_csr_invalid_row_ptrs_length() {
        let result = CSRTensor::new(
            vec![1.0],
            vec![0],
            vec![0], // Should be length 3 for nrows=2
            2,
            2,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_csr_invalid_col_index() {
        let result = CSRTensor::new(
            vec![1.0],
            vec![5], // Out of bounds
            vec![0, 1, 1],
            2,
            2,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_csr_non_monotonic_row_ptrs() {
        let result = CSRTensor::new(
            vec![1.0, 2.0],
            vec![0, 1],
            vec![0, 2, 1], // Not monotonic
            2,
            2,
        );
        assert!(result.is_err());
    }

    // ==========================================================================
    // FALSIFICATION: Sparsify with Block format
    // ==========================================================================
    #[test]
    fn test_sparsify_block() {
        let tensor = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let mask_tensor = Tensor::new(&[1.0, 1.0, 0.0, 0.0], &[2, 2]);
        let mask = SparsityMask::new(
            mask_tensor,
            super::super::mask::SparsityPattern::Unstructured,
        )
        .unwrap();

        let sparse = sparsify(
            &tensor,
            &mask,
            SparseFormat::Block {
                height: 1,
                width: 1,
            },
        )
        .unwrap();

        assert_eq!(sparse.nnz(), 2, "SPA-15 FALSIFIED: Should have 2 non-zeros");
        matches!(sparse.format(), SparseFormat::Block { .. });
    }

    // ==========================================================================
    // FALSIFICATION: SparseTensor Block variant
    // ==========================================================================
    #[test]
    fn test_sparse_tensor_block_to_dense() {
        let dense = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let block = BlockSparseTensor::from_dense(&dense, 1, 1).unwrap();
        let sparse = SparseTensor::Block(block);

        let recovered = sparse.to_dense();
        assert_eq!(dense.data(), recovered.data());
    }

    #[test]
    fn test_sparse_tensor_block_format() {
        let dense = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let block = SparseTensor::Block(BlockSparseTensor::from_dense(&dense, 2, 2).unwrap());

        match block.format() {
            SparseFormat::Block { height, width } => {
                assert_eq!(height, 2);
                assert_eq!(width, 2);
            }
            _ => panic!("SPA-16 FALSIFIED: Should be Block format"),
        }
    }

    #[test]
    fn test_sparse_tensor_block_nnz() {
        let dense = Tensor::new(&[1.0, 0.0, 0.0, 2.0], &[2, 2]);
        let block = SparseTensor::Block(BlockSparseTensor::from_dense(&dense, 1, 1).unwrap());

        assert_eq!(block.nnz(), 2, "SPA-17 FALSIFIED: Should have 2 non-zeros");
    }

    #[test]
    fn test_sparse_tensor_block_sparsity() {
        let dense = Tensor::new(&[1.0, 0.0, 0.0, 0.0], &[2, 2]);
        let block = SparseTensor::Block(BlockSparseTensor::from_dense(&dense, 1, 1).unwrap());

        assert!(
            (block.sparsity() - 0.75).abs() < 1e-5,
            "SPA-18 FALSIFIED: Sparsity should be 0.75"
        );
    }

    // ==========================================================================
    // FALSIFICATION: COO 1D rejected
    // ==========================================================================
    #[test]
    fn test_coo_1d_rejected() {
        let dense = Tensor::new(&[1.0, 2.0, 3.0], &[3]);
        let result = COOTensor::from_dense(&dense);

        assert!(
            result.is_err(),
            "SPA-19 FALSIFIED: 1D tensor should be rejected"
        );
    }

    // ==========================================================================
    // FALSIFICATION: Block sparse 1D rejected
    // ==========================================================================
    #[test]
    fn test_block_sparse_1d_rejected() {
        let dense = Tensor::new(&[1.0, 2.0, 3.0], &[3]);
        let result = BlockSparseTensor::from_dense(&dense, 1, 1);

        assert!(
            result.is_err(),
            "SPA-20 FALSIFIED: 1D tensor should be rejected"
        );
    }

    // ==========================================================================
    // FALSIFICATION: Block sparse shape
    // ==========================================================================
    #[test]
    fn test_block_sparse_shape() {
        let dense = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let block = BlockSparseTensor::from_dense(&dense, 2, 3).unwrap();

        assert_eq!(
            block.shape(),
            [2, 3],
            "SPA-21 FALSIFIED: Shape should be [2, 3]"
        );
    }

    // ==========================================================================
    // FALSIFICATION: Block sparse sparsity edge cases
    // ==========================================================================
    #[test]
    fn test_block_sparse_empty_sparsity() {
        let dense = Tensor::new(&[0.0, 0.0, 0.0, 0.0], &[2, 2]);
        let block = BlockSparseTensor::from_dense(&dense, 1, 1).unwrap();

        assert!(
            (block.sparsity() - 1.0).abs() < 1e-5,
            "SPA-22 FALSIFIED: All zeros should have sparsity 1.0"
        );
    }

    // ==========================================================================
    // FALSIFICATION: CSR values/col_indices length mismatch
    // ==========================================================================
    #[test]
    fn test_csr_values_col_indices_mismatch() {
        let result = CSRTensor::new(
            vec![1.0, 2.0],
            vec![0], // Length mismatch with values
            vec![0, 1, 2],
            2,
            2,
        );
        assert!(
            result.is_err(),
            "SPA-23 FALSIFIED: Should error on length mismatch"
        );
    }

    // ==========================================================================
    // FALSIFICATION: CSR empty sparsity
    // ==========================================================================
    #[test]
    fn test_csr_zero_size_sparsity() {
        let csr = CSRTensor::new(vec![], vec![], vec![0], 0, 0).unwrap();

        assert_eq!(
            csr.sparsity(),
            0.0,
            "SPA-24 FALSIFIED: Zero-size matrix sparsity should be 0"
        );
    }

    // ==========================================================================
    // FALSIFICATION: Memory calculations
    // ==========================================================================
    #[test]
    fn test_csr_dense_memory_bytes() {
        let dense = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let csr = CSRTensor::from_dense(&dense).unwrap();

        // 2 * 2 * 4 bytes = 16 bytes
        assert_eq!(
            csr.dense_memory_bytes(),
            16,
            "SPA-25 FALSIFIED: Dense memory should be 16 bytes"
        );
    }

    #[test]
    fn test_csr_memory_bytes() {
        let dense = Tensor::new(&[1.0, 0.0, 0.0, 2.0], &[2, 2]);
        let csr = CSRTensor::from_dense(&dense).unwrap();

        // 2 values * 4 bytes + 2 col_indices * 8 bytes + 3 row_ptrs * 8 bytes = 8 + 16 + 24 = 48 bytes
        assert!(
            csr.memory_bytes() > 0,
            "SPA-26 FALSIFIED: Memory bytes should be positive"
        );
    }

    #[test]
    fn test_csr_memory_savings_ratio() {
        // Very sparse matrix - should have savings
        let mut data = vec![0.0f32; 100];
        data[0] = 1.0; // Only one non-zero
        let dense = Tensor::new(&data, &[10, 10]);
        let csr = CSRTensor::from_dense(&dense).unwrap();

        // Sparse should save memory for very sparse matrices
        let ratio = csr.memory_savings_ratio();
        assert!(
            ratio > 1.0,
            "SPA-27 FALSIFIED: Very sparse matrix should save memory, got ratio {}",
            ratio
        );
    }

    #[test]
    fn test_csr_memory_savings_dense() {
        // Dense matrix - sparse format has overhead
        let dense = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let csr = CSRTensor::from_dense(&dense).unwrap();

        // For dense matrices, sparse format may not save memory due to index overhead
        let ratio = csr.memory_savings_ratio();
        assert!(ratio > 0.0, "SPA-28 FALSIFIED: Ratio should be positive");
    }

    // ==========================================================================
    // FALSIFICATION: COO Clone
    // ==========================================================================
    #[test]
    fn test_coo_clone() {
        let dense = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let coo = COOTensor::from_dense(&dense).unwrap();
        let cloned = coo.clone();

        assert_eq!(coo.values, cloned.values);
        assert_eq!(coo.nnz(), cloned.nnz());
    }

    // ==========================================================================
    // FALSIFICATION: Block sparse Clone
    // ==========================================================================
    #[test]
    fn test_block_sparse_clone() {
        let dense = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let block = BlockSparseTensor::from_dense(&dense, 1, 1).unwrap();
        let cloned = block.clone();

        assert_eq!(block.num_blocks(), cloned.num_blocks());
        assert_eq!(block.nnz(), cloned.nnz());
    }

    // ==========================================================================
    // FALSIFICATION: SparseTensor Clone
    // ==========================================================================
    #[test]
    fn test_sparse_tensor_clone() {
        let dense = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let csr = SparseTensor::CSR(CSRTensor::from_dense(&dense).unwrap());
        let cloned = csr.clone();

        assert_eq!(csr.nnz(), cloned.nnz());
    }

    // ==========================================================================
    // FALSIFICATION: SparseFormat equality
    // ==========================================================================
    #[test]
    fn test_sparse_format_equality() {
        assert_eq!(SparseFormat::CSR, SparseFormat::CSR);
        assert_eq!(SparseFormat::COO, SparseFormat::COO);
        assert_eq!(
            SparseFormat::Block {
                height: 2,
                width: 2
            },
            SparseFormat::Block {
                height: 2,
                width: 2
            }
        );
        assert_ne!(SparseFormat::CSR, SparseFormat::COO);
        assert_ne!(
            SparseFormat::Block {
                height: 1,
                width: 1
            },
            SparseFormat::Block {
                height: 2,
                width: 2
            }
        );
    }

    // ==========================================================================
    // FALSIFICATION: SparseFormat Copy
    // ==========================================================================
    #[test]
    fn test_sparse_format_copy() {
        let format = SparseFormat::CSR;
        let copied = format; // Copy
        assert_eq!(format, copied);
    }
}
