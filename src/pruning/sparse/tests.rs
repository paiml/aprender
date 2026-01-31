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
