
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
mod tests;
