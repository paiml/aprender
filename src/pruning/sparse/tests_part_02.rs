use super::*;

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
