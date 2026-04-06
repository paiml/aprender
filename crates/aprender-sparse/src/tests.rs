//! Tests for trueno-sparse.
//!
//! Follows provable contract sparse-spmv-v1.yaml falsification tests.

use crate::*;

// ============================================================================
// FALSIFY-SPARSE-001: CSR format validity
// ============================================================================

#[test]
fn test_reject_non_monotonic_offsets() {
    let result = CsrMatrix::<f32>::new(
        3,
        3,
        vec![0, 2, 1, 3], // non-monotonic: offsets[1]=2 > offsets[2]=1
        vec![0, 1, 0],
        vec![1.0, 2.0, 3.0],
    );
    assert!(result.is_err());
    match result.unwrap_err() {
        SparseError::NonMonotonicOffsets { index: 1, .. } => {}
        e => panic!("Expected NonMonotonicOffsets at index 1, got {e:?}"),
    }
}

#[test]
fn test_reject_nonzero_first_offset() {
    let result = CsrMatrix::<f32>::new(
        2,
        2,
        vec![1, 2, 3], // offsets[0] != 0
        vec![0, 1, 0],
        vec![1.0, 2.0, 3.0],
    );
    assert!(result.is_err());
    match result.unwrap_err() {
        SparseError::NonZeroFirstOffset { value: 1 } => {}
        e => panic!("Expected NonZeroFirstOffset, got {e:?}"),
    }
}

#[test]
fn test_reject_offsets_nnz_mismatch() {
    let result = CsrMatrix::<f32>::new(
        2,
        2,
        vec![0, 1, 5], // offsets[2]=5 but only 2 elements
        vec![0, 1],
        vec![1.0, 2.0],
    );
    assert!(result.is_err());
    match result.unwrap_err() {
        SparseError::OffsetNnzMismatch { .. } => {}
        e => panic!("Expected OffsetNnzMismatch, got {e:?}"),
    }
}

#[test]
fn test_reject_wrong_offsets_length() {
    let result = CsrMatrix::<f32>::new(
        3,
        3,
        vec![0, 1], // length 2, expected 4 (rows+1)
        vec![0],
        vec![1.0],
    );
    assert!(result.is_err());
    match result.unwrap_err() {
        SparseError::InvalidOffsetsLength { actual: 2, expected: 4 } => {}
        e => panic!("Expected InvalidOffsetsLength, got {e:?}"),
    }
}

// ============================================================================
// FALSIFY-SPARSE-002: Dimension correctness
// ============================================================================

#[test]
fn test_reject_column_out_of_bounds() {
    let result = CsrMatrix::<f32>::new(
        2,
        3,
        vec![0, 1, 2],
        vec![0, 5], // col=5 >= cols=3
        vec![1.0, 2.0],
    );
    assert!(result.is_err());
    match result.unwrap_err() {
        SparseError::ColumnOutOfBounds { col: 5, cols: 3, .. } => {}
        e => panic!("Expected ColumnOutOfBounds, got {e:?}"),
    }
}

#[test]
fn test_spmv_dimension_mismatch() {
    let a = CsrMatrix::<f32>::new(2, 3, vec![0, 1, 2], vec![0, 1], vec![1.0, 2.0]).unwrap();
    let x = vec![1.0, 2.0]; // length 2, but cols=3
    let mut y = vec![0.0; 2];
    let result = a.spmv(1.0, &x, 0.0, &mut y);
    assert!(result.is_err());
}

#[test]
fn test_spmv_output_dimension_mismatch() {
    let a = CsrMatrix::<f32>::new(2, 3, vec![0, 1, 2], vec![0, 1], vec![1.0, 2.0]).unwrap();
    let x = vec![1.0, 2.0, 3.0];
    let mut y = vec![0.0; 5]; // length 5, but rows=2
    let result = a.spmv(1.0, &x, 0.0, &mut y);
    assert!(result.is_err());
}

// ============================================================================
// FALSIFY-SPARSE-003: Numerical accuracy (backward error bound)
// ============================================================================

#[test]
fn test_spmv_identity() {
    let n = 4;
    let a = CsrMatrix::<f32>::identity(n);
    let x = vec![1.0, 2.0, 3.0, 4.0];
    let mut y = vec![0.0; n];

    a.spmv(1.0, &x, 0.0, &mut y).unwrap();

    for i in 0..n {
        assert!(
            (y[i] - x[i]).abs() < 1e-7,
            "Identity SpMV failed at i={i}: y={}, x={}",
            y[i],
            x[i]
        );
    }
}

#[test]
fn test_spmv_alpha_beta() {
    let a = CsrMatrix::<f32>::identity(3);
    let x = vec![1.0, 2.0, 3.0];
    let mut y = vec![10.0, 20.0, 30.0];

    // y = 2.0 * I * x + 0.5 * y = [2+5, 4+10, 6+15] = [7, 14, 21]
    a.spmv(2.0, &x, 0.5, &mut y).unwrap();

    assert!((y[0] - 7.0).abs() < 1e-5, "y[0]={}", y[0]);
    assert!((y[1] - 14.0).abs() < 1e-5, "y[1]={}", y[1]);
    assert!((y[2] - 21.0).abs() < 1e-5, "y[2]={}", y[2]);
}

#[test]
fn test_spmv_sparse_matrix() {
    // A = [[1, 0, 2],
    //      [0, 3, 0],
    //      [4, 0, 5]]
    let a = CsrMatrix::<f32>::new(
        3,
        3,
        vec![0, 2, 3, 5],
        vec![0, 2, 1, 0, 2],
        vec![1.0, 2.0, 3.0, 4.0, 5.0],
    )
    .unwrap();

    let x = vec![1.0, 2.0, 3.0];
    let mut y = vec![0.0; 3];

    a.spmv(1.0, &x, 0.0, &mut y).unwrap();

    // y[0] = 1*1 + 2*3 = 7
    // y[1] = 3*2 = 6
    // y[2] = 4*1 + 5*3 = 19
    assert!((y[0] - 7.0).abs() < 1e-5);
    assert!((y[1] - 6.0).abs() < 1e-5);
    assert!((y[2] - 19.0).abs() < 1e-5);
}

#[test]
fn test_spmv_empty_rows() {
    // A = [[0, 0],
    //      [1, 0],
    //      [0, 0]]
    let a = CsrMatrix::<f32>::new(3, 2, vec![0, 0, 1, 1], vec![0], vec![1.0]).unwrap();

    let x = vec![5.0, 3.0];
    let mut y = vec![0.0; 3];
    a.spmv(1.0, &x, 0.0, &mut y).unwrap();

    assert!((y[0]).abs() < 1e-7);
    assert!((y[1] - 5.0).abs() < 1e-5);
    assert!((y[2]).abs() < 1e-7);
}

// ============================================================================
// COO construction and conversion
// ============================================================================

#[test]
fn test_coo_to_csr_basic() {
    let coo =
        CooMatrix::new(3, 3, vec![0, 1, 2, 0], vec![0, 1, 2, 2], vec![1.0_f32, 2.0, 3.0, 4.0])
            .unwrap();

    let csr = CsrMatrix::from_coo(&coo);
    assert_eq!(csr.rows(), 3);
    assert_eq!(csr.cols(), 3);
    assert_eq!(csr.nnz(), 4);

    // Row 0 has 2 entries: (0,0)=1.0, (0,2)=4.0
    assert_eq!(csr.offsets()[0], 0);
    assert_eq!(csr.offsets()[1], 2);
    // Row 1 has 1 entry: (1,1)=2.0
    assert_eq!(csr.offsets()[2], 3);
    // Row 2 has 1 entry: (2,2)=3.0
    assert_eq!(csr.offsets()[3], 4);
}

#[test]
fn test_coo_to_csr_empty() {
    let coo = CooMatrix::<f32>::empty(5, 5);
    let csr = CsrMatrix::from_coo(&coo);
    assert_eq!(csr.rows(), 5);
    assert_eq!(csr.nnz(), 0);
}

#[test]
fn test_coo_rejects_bad_row() {
    let result = CooMatrix::new(2, 2, vec![0, 5], vec![0, 1], vec![1.0_f32, 2.0]);
    assert!(result.is_err());
}

#[test]
fn test_coo_rejects_bad_col() {
    let result = CooMatrix::new(2, 2, vec![0, 1], vec![0, 5], vec![1.0_f32, 2.0]);
    assert!(result.is_err());
}

#[test]
fn test_coo_rejects_mismatched_lengths() {
    let result = CooMatrix::new(2, 2, vec![0], vec![0, 1], vec![1.0_f32, 2.0]);
    assert!(result.is_err());
}

// ============================================================================
// SpMM tests
// ============================================================================

#[test]
fn test_spmm_identity() {
    let n = 3;
    let a = CsrMatrix::<f32>::identity(n);
    let b = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]; // 3x3
    let mut c = vec![0.0; 9];

    a.spmm(1.0, &b, 3, 0.0, &mut c).unwrap();

    for i in 0..9 {
        assert!((c[i] - b[i]).abs() < 1e-5, "c[{i}]={}, b={}", c[i], b[i]);
    }
}

#[test]
fn test_spmm_sparse() {
    // A = [[1, 0], [0, 2]]
    let a = CsrMatrix::<f32>::new(2, 2, vec![0, 1, 2], vec![0, 1], vec![1.0, 2.0]).unwrap();

    // B = [[1, 2], [3, 4]]
    let b = vec![1.0, 2.0, 3.0, 4.0];
    let mut c = vec![0.0; 4];

    a.spmm(1.0, &b, 2, 0.0, &mut c).unwrap();

    // C = [[1*1, 1*2], [2*3, 2*4]] = [[1, 2], [6, 8]]
    assert!((c[0] - 1.0).abs() < 1e-5);
    assert!((c[1] - 2.0).abs() < 1e-5);
    assert!((c[2] - 6.0).abs() < 1e-5);
    assert!((c[3] - 8.0).abs() < 1e-5);
}

// ============================================================================
// Dense roundtrip
// ============================================================================

#[test]
fn test_to_dense_roundtrip() {
    let coo = CooMatrix::new(2, 3, vec![0, 0, 1], vec![0, 2, 1], vec![1.0_f32, 2.0, 3.0]).unwrap();

    let csr = CsrMatrix::from_coo(&coo);
    let dense = csr.to_dense();

    // [[1, 0, 2], [0, 3, 0]]
    assert!((dense[0] - 1.0).abs() < 1e-7);
    assert!((dense[1]).abs() < 1e-7);
    assert!((dense[2] - 2.0).abs() < 1e-7);
    assert!((dense[3]).abs() < 1e-7);
    assert!((dense[4] - 3.0).abs() < 1e-7);
    assert!((dense[5]).abs() < 1e-7);
}

// ============================================================================
// Statistics
// ============================================================================

#[test]
fn test_avg_nnz_per_row() {
    // 3 rows, 5 nonzeros
    let a =
        CsrMatrix::<f32>::new(3, 3, vec![0, 2, 3, 5], vec![0, 1, 2, 0, 1], vec![1.0; 5]).unwrap();
    let avg = a.avg_nnz_per_row();
    assert!((avg - 5.0 / 3.0).abs() < 1e-10);
}

#[test]
fn test_row_length_variance() {
    // Row lengths: [2, 1, 2], mean = 5/3
    let a =
        CsrMatrix::<f32>::new(3, 3, vec![0, 2, 3, 5], vec![0, 1, 2, 0, 1], vec![1.0; 5]).unwrap();
    let var = a.row_length_variance();
    // Variance = ((2-5/3)^2 + (1-5/3)^2 + (2-5/3)^2) / 3
    //          = (1/9 + 4/9 + 1/9) / 3 = 6/27 = 2/9
    assert!((var - 2.0 / 9.0).abs() < 1e-10);
}

#[test]
fn test_empty_matrix_stats() {
    let a = CsrMatrix::<f32>::new(0, 0, vec![0], vec![], vec![]).unwrap();
    assert_eq!(a.avg_nnz_per_row(), 0.0);
    assert_eq!(a.row_length_variance(), 0.0);
}

// ============================================================================
// Property-based tests (proptest)
// ============================================================================

mod proptests {
    use super::*;
    use proptest::prelude::*;

    /// Generate a random CSR matrix via COO.
    fn arb_csr(max_dim: usize, max_nnz: usize) -> impl Strategy<Value = CsrMatrix<f32>> {
        (1..=max_dim, 1..=max_dim, 0..=max_nnz)
            .prop_flat_map(|(rows, cols, nnz)| {
                let row_idx = proptest::collection::vec(0..rows as u32, nnz);
                let col_idx = proptest::collection::vec(0..cols as u32, nnz);
                let vals = proptest::collection::vec(-100.0_f32..100.0, nnz);
                (Just(rows), Just(cols), row_idx, col_idx, vals)
            })
            .prop_map(|(rows, cols, ri, ci, vals)| {
                let coo = CooMatrix::new(rows, cols, ri, ci, vals).unwrap();
                CsrMatrix::from_coo(&coo)
            })
    }

    proptest! {
        /// FALSIFY-SPARSE-002: SpMV output has correct dimensions.
        #[test]
        fn prop_spmv_output_dimension(a in arb_csr(20, 50)) {
            let x = vec![1.0_f32; a.cols()];
            let mut y = vec![0.0_f32; a.rows()];
            a.spmv(1.0, &x, 0.0, &mut y).unwrap();
            prop_assert_eq!(y.len(), a.rows());
        }

        /// FALSIFY-SPARSE-003: SpMV matches dense matvec.
        #[test]
        fn prop_spmv_matches_dense(a in arb_csr(15, 40)) {
            let x: Vec<f32> = (0..a.cols()).map(|i| (i as f32 + 1.0) * 0.1).collect();
            let mut y_sparse = vec![0.0_f32; a.rows()];
            a.spmv(1.0, &x, 0.0, &mut y_sparse).unwrap();

            // Dense reference
            let dense = a.to_dense();
            let mut y_dense = vec![0.0_f32; a.rows()];
            for i in 0..a.rows() {
                for j in 0..a.cols() {
                    y_dense[i] += dense[i * a.cols() + j] * x[j];
                }
            }

            for i in 0..a.rows() {
                let err = (y_sparse[i] - y_dense[i]).abs();
                let scale = y_dense[i].abs().max(1.0);
                prop_assert!(
                    err / scale < 1e-4,
                    "SpMV mismatch at row {}: sparse={}, dense={}, err={}",
                    i, y_sparse[i], y_dense[i], err
                );
            }
        }

        /// FALSIFY-SPARSE-005: Linearity: spmv(A, alpha*x) == alpha * spmv(A, x).
        #[test]
        fn prop_spmv_linearity(a in arb_csr(10, 30), alpha in -10.0_f32..10.0) {
            let x: Vec<f32> = (0..a.cols()).map(|i| (i as f32 + 1.0) * 0.1).collect();

            // y1 = A * (alpha * x)
            let scaled_x: Vec<f32> = x.iter().map(|xi| alpha * xi).collect();
            let mut y1 = vec![0.0_f32; a.rows()];
            a.spmv(1.0, &scaled_x, 0.0, &mut y1).unwrap();

            // y2 = alpha * A * x
            let mut y2 = vec![0.0_f32; a.rows()];
            a.spmv(alpha, &x, 0.0, &mut y2).unwrap();

            for i in 0..a.rows() {
                let err = (y1[i] - y2[i]).abs();
                let scale = y1[i].abs().max(y2[i].abs()).max(1.0);
                prop_assert!(
                    err / scale < 1e-3,
                    "Linearity violated at row {}: y1={}, y2={}, alpha={}",
                    i, y1[i], y2[i], alpha
                );
            }
        }

        /// CSR from_coo always produces valid CSR.
        #[test]
        fn prop_coo_to_csr_valid(a in arb_csr(20, 50)) {
            // If we got here, from_coo succeeded. Verify invariants.
            let offsets = a.offsets();
            prop_assert_eq!(offsets[0], 0);
            prop_assert_eq!(offsets[a.rows()] as usize, a.nnz());
            for i in 0..a.rows() {
                prop_assert!(offsets[i] <= offsets[i + 1]);
            }
            for &col in a.col_indices() {
                prop_assert!((col as usize) < a.cols());
            }
        }
    }
}

// ============================================================================
// BSR (Block Sparse Row)
// ============================================================================

#[test]
fn test_bsr_from_dense_identity() {
    // 4x4 identity with block_size=2
    #[rustfmt::skip]
    let dense = [
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0_f32,
    ];
    let bsr = BsrMatrix::from_dense(&dense, 4, 4, 2);
    assert_eq!(bsr.rows(), 4);
    assert_eq!(bsr.cols(), 4);
    assert_eq!(bsr.block_size(), 2);
    assert_eq!(bsr.nnz_blocks(), 2); // 2 non-zero 2x2 blocks on diagonal
}

#[test]
fn test_bsr_spmv_identity() -> Result<(), Box<dyn std::error::Error>> {
    #[rustfmt::skip]
    let dense = [
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0_f32,
    ];
    let bsr = BsrMatrix::from_dense(&dense, 4, 4, 2);
    let x = vec![1.0, 2.0, 3.0, 4.0_f32];
    let mut y = vec![0.0f32; 4];
    bsr.spmv(1.0, &x, 0.0, &mut y)?;
    for i in 0..4 {
        assert!((y[i] - x[i]).abs() < 1e-6);
    }
    Ok(())
}

#[test]
fn test_bsr_spmv_general() -> Result<(), Box<dyn std::error::Error>> {
    // [[1,2,0,0],[3,4,0,0],[0,0,5,6],[0,0,7,8]]
    #[rustfmt::skip]
    let dense = [
        1.0, 2.0, 0.0, 0.0,
        3.0, 4.0, 0.0, 0.0,
        0.0, 0.0, 5.0, 6.0,
        0.0, 0.0, 7.0, 8.0_f32,
    ];
    let bsr = BsrMatrix::from_dense(&dense, 4, 4, 2);
    let x = vec![1.0, 1.0, 1.0, 1.0_f32];
    let mut y = vec![0.0f32; 4];
    bsr.spmv(1.0, &x, 0.0, &mut y)?;
    assert!((y[0] - 3.0).abs() < 1e-6);
    assert!((y[1] - 7.0).abs() < 1e-6);
    assert!((y[2] - 11.0).abs() < 1e-6);
    assert!((y[3] - 15.0).abs() < 1e-6);
    Ok(())
}

#[test]
fn test_bsr_to_csr_roundtrip() -> Result<(), Box<dyn std::error::Error>> {
    #[rustfmt::skip]
    let dense = [
        1.0, 2.0, 0.0, 0.0,
        3.0, 4.0, 0.0, 0.0,
        0.0, 0.0, 5.0, 6.0,
        0.0, 0.0, 7.0, 8.0_f32,
    ];
    let bsr = BsrMatrix::from_dense(&dense, 4, 4, 2);
    let csr = bsr.to_csr()?;
    assert_eq!(csr.rows(), 4);
    assert_eq!(csr.cols(), 4);

    // Verify SpMV gives same result
    let x = vec![1.0, 2.0, 3.0, 4.0_f32];
    let mut y_bsr = vec![0.0f32; 4];
    let mut y_csr = vec![0.0f32; 4];
    bsr.spmv(1.0, &x, 0.0, &mut y_bsr)?;
    csr.spmv(1.0, &x, 0.0, &mut y_csr)?;
    for i in 0..4 {
        assert!((y_bsr[i] - y_csr[i]).abs() < 1e-6);
    }
    Ok(())
}

#[test]
fn test_bsr_alpha_beta() -> Result<(), Box<dyn std::error::Error>> {
    #[rustfmt::skip]
    let dense = [
        2.0, 0.0,
        0.0, 3.0_f32,
    ];
    let bsr = BsrMatrix::from_dense(&dense, 2, 2, 2);
    let x = vec![1.0, 1.0_f32];
    let mut y = vec![10.0, 20.0_f32];
    // y = 2.0 * A * x + 0.5 * y = [2*2, 2*3] + [5, 10] = [9, 16]
    bsr.spmv(2.0, &x, 0.5, &mut y)?;
    assert!((y[0] - 9.0).abs() < 1e-6);
    assert!((y[1] - 16.0).abs() < 1e-6);
    Ok(())
}

// ============================================================================
// SpGEMM tests (Contract: sparse-spgemm-v1.yaml)
// ============================================================================

#[test]
fn test_spgemm_identity() -> Result<(), Box<dyn std::error::Error>> {
    // A * I = A
    let a = CsrMatrix::new(2, 2, vec![0, 1, 2], vec![0, 1], vec![3.0_f32, 5.0])?;
    let eye = CsrMatrix::<f32>::identity(2);
    let c = crate::spgemm::spgemm(&a, &eye)?;
    let dense = c.to_dense();
    assert!((dense[0] - 3.0).abs() < 1e-5);
    assert!((dense[3] - 5.0).abs() < 1e-5);
    assert!(dense[1].abs() < 1e-5);
    assert!(dense[2].abs() < 1e-5);
    Ok(())
}

#[test]
fn test_spgemm_identity_left() -> Result<(), Box<dyn std::error::Error>> {
    // I * A = A
    let a = CsrMatrix::new(3, 3, vec![0, 2, 3, 4], vec![0, 2, 1, 0], vec![1.0_f32, 2.0, 3.0, 4.0])?;
    let eye = CsrMatrix::<f32>::identity(3);
    let c = crate::spgemm::spgemm(&eye, &a)?;
    assert_eq!(c.to_dense(), a.to_dense());
    Ok(())
}

#[test]
fn test_spgemm_known_product() -> Result<(), Box<dyn std::error::Error>> {
    // A = [[1, 2], [3, 4]], B = [[5, 6], [7, 8]]
    // AB = [[19, 22], [43, 50]]
    let a = CsrMatrix::new(2, 2, vec![0, 2, 4], vec![0, 1, 0, 1], vec![1.0_f32, 2.0, 3.0, 4.0])?;
    let b = CsrMatrix::new(2, 2, vec![0, 2, 4], vec![0, 1, 0, 1], vec![5.0_f32, 6.0, 7.0, 8.0])?;
    let c = crate::spgemm::spgemm(&a, &b)?;
    let d = c.to_dense();
    assert!((d[0] - 19.0).abs() < 1e-4, "got {}", d[0]);
    assert!((d[1] - 22.0).abs() < 1e-4);
    assert!((d[2] - 43.0).abs() < 1e-4);
    assert!((d[3] - 50.0).abs() < 1e-4);
    Ok(())
}

#[test]
fn test_spgemm_dimension_mismatch() -> Result<(), Box<dyn std::error::Error>> {
    let a = CsrMatrix::new(2, 3, vec![0, 1, 2], vec![0, 1], vec![1.0_f32, 2.0])?;
    let b = CsrMatrix::new(2, 2, vec![0, 1, 2], vec![0, 1], vec![1.0_f32, 2.0])?;
    // a.cols()=3 != b.rows()=2
    assert!(crate::spgemm::spgemm(&a, &b).is_err());
    Ok(())
}

#[test]
fn test_spgemm_sparse_result() -> Result<(), Box<dyn std::error::Error>> {
    // Diagonal × Diagonal = Diagonal
    let a = CsrMatrix::new(3, 3, vec![0, 1, 2, 3], vec![0, 1, 2], vec![2.0_f32, 3.0, 4.0])?;
    let b = CsrMatrix::new(3, 3, vec![0, 1, 2, 3], vec![0, 1, 2], vec![5.0_f32, 6.0, 7.0])?;
    let c = crate::spgemm::spgemm(&a, &b)?;
    assert_eq!(c.nnz(), 3);
    let d = c.to_dense();
    assert!((d[0] - 10.0).abs() < 1e-4);
    assert!((d[4] - 18.0).abs() < 1e-4);
    assert!((d[8] - 28.0).abs() < 1e-4);
    Ok(())
}

// ============================================================================
// SELL (Sliced ELLPACK) tests
// ============================================================================

#[test]
fn test_sell_identity_spmv() -> Result<(), Box<dyn std::error::Error>> {
    let csr = CsrMatrix::<f32>::identity(4);
    let sell = SellMatrix::from_csr(&csr, 2);
    let x = vec![1.0, 2.0, 3.0, 4.0_f32];
    let mut y = vec![0.0_f32; 4];
    sell.spmv(1.0, &x, 0.0, &mut y)?;
    assert!((y[0] - 1.0).abs() < 1e-6);
    assert!((y[1] - 2.0).abs() < 1e-6);
    assert!((y[2] - 3.0).abs() < 1e-6);
    assert!((y[3] - 4.0).abs() < 1e-6);
    Ok(())
}

#[test]
fn test_sell_matches_csr_spmv() -> Result<(), Box<dyn std::error::Error>> {
    // Sparse matrix with variable row lengths
    let csr = CsrMatrix::new(
        4,
        4,
        vec![0, 2, 3, 5, 6],
        vec![0, 1, 2, 1, 3, 0],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0_f32],
    )?;
    let sell = SellMatrix::from_csr(&csr, 2);
    let x = vec![1.0, 1.0, 1.0, 1.0_f32];

    let mut y_csr = vec![0.0_f32; 4];
    let mut y_sell = vec![0.0_f32; 4];

    csr.spmv(1.0, &x, 0.0, &mut y_csr)?;
    sell.spmv(1.0, &x, 0.0, &mut y_sell)?;

    for i in 0..4 {
        assert!(
            (y_csr[i] - y_sell[i]).abs() < 1e-5,
            "SELL vs CSR mismatch at {i}: csr={}, sell={}",
            y_csr[i],
            y_sell[i]
        );
    }
    Ok(())
}

#[test]
fn test_sell_alpha_beta() -> Result<(), Box<dyn std::error::Error>> {
    let csr = CsrMatrix::<f32>::identity(3);
    let sell = SellMatrix::from_csr(&csr, 4);
    let x = vec![1.0, 2.0, 3.0_f32];
    let mut y = vec![10.0, 20.0, 30.0_f32];
    // y = 2·I·x + 0.5·y = [2+5, 4+10, 6+15] = [7, 14, 21]
    sell.spmv(2.0, &x, 0.5, &mut y)?;
    assert!((y[0] - 7.0).abs() < 1e-5);
    assert!((y[1] - 14.0).abs() < 1e-5);
    assert!((y[2] - 21.0).abs() < 1e-5);
    Ok(())
}

#[test]
fn test_sell_dimension_mismatch() -> Result<(), Box<dyn std::error::Error>> {
    let csr = CsrMatrix::<f32>::identity(3);
    let sell = SellMatrix::from_csr(&csr, 2);
    let x = vec![1.0, 2.0_f32]; // wrong length
    let mut y = vec![0.0_f32; 3];
    assert!(sell.spmv(1.0, &x, 0.0, &mut y).is_err());
    Ok(())
}

#[test]
fn test_sell_properties() -> Result<(), Box<dyn std::error::Error>> {
    let csr = CsrMatrix::new(3, 3, vec![0, 2, 3, 3], vec![0, 1, 2], vec![1.0, 2.0, 3.0_f32])?;
    let sell = SellMatrix::from_csr(&csr, 2);
    assert_eq!(sell.rows(), 3);
    assert_eq!(sell.cols(), 3);
    assert_eq!(sell.slice_size(), 2);
    Ok(())
}

// ── SparseBackend trait tests ──────────────────────────────────

use crate::ops::{ScalarBackend, SparseBackend};

#[test]
fn test_scalar_backend_spmv() -> Result<(), Box<dyn std::error::Error>> {
    // 3×3 identity via scalar backend
    let csr = CsrMatrix::new(3, 3, vec![0, 1, 2, 3], vec![0, 1, 2], vec![1.0, 1.0, 1.0_f32])?;
    let x = [2.0, 3.0, 4.0_f32];
    let mut y = [0.0_f32; 3];
    ScalarBackend::spmv_kernel(&csr, 1.0, &x, 0.0, &mut y);
    assert!((y[0] - 2.0).abs() < 1e-5);
    assert!((y[1] - 3.0).abs() < 1e-5);
    assert!((y[2] - 4.0).abs() < 1e-5);
    Ok(())
}

#[test]
fn test_scalar_backend_alpha_beta() -> Result<(), Box<dyn std::error::Error>> {
    let csr = CsrMatrix::new(2, 2, vec![0, 1, 2], vec![0, 1], vec![3.0, 5.0_f32])?;
    let x = [1.0, 1.0_f32];
    let mut y = [10.0, 20.0_f32];
    ScalarBackend::spmv_kernel(&csr, 2.0, &x, 0.5, &mut y);
    // y[0] = 2.0 * 3.0 + 0.5 * 10.0 = 11.0
    // y[1] = 2.0 * 5.0 + 0.5 * 20.0 = 20.0
    assert!((y[0] - 11.0).abs() < 1e-5);
    assert!((y[1] - 20.0).abs() < 1e-5);
    Ok(())
}

// ── SparseBackend spmm_kernel tests (§2C parity) ───────────────

#[test]
fn test_scalar_backend_spmm_identity() -> Result<(), Box<dyn std::error::Error>> {
    // 2×2 identity CSR × 2×2 dense = identity
    let csr = CsrMatrix::new(2, 2, vec![0, 1, 2], vec![0, 1], vec![1.0, 1.0_f32])?;
    let b = [1.0, 2.0, 3.0, 4.0_f32]; // 2×2 row-major
    let mut c = [0.0_f32; 4];
    ScalarBackend::spmm_kernel(&csr, 1.0, &b, 2, 0.0, &mut c);
    assert!((c[0] - 1.0).abs() < 1e-5);
    assert!((c[1] - 2.0).abs() < 1e-5);
    assert!((c[2] - 3.0).abs() < 1e-5);
    assert!((c[3] - 4.0).abs() < 1e-5);
    Ok(())
}

#[test]
fn test_scalar_backend_spmm_alpha_beta() -> Result<(), Box<dyn std::error::Error>> {
    let csr = CsrMatrix::new(2, 2, vec![0, 1, 2], vec![0, 1], vec![2.0, 3.0_f32])?;
    let b = [1.0, 0.0, 0.0, 1.0_f32]; // identity
    let mut c = [10.0, 20.0, 30.0, 40.0_f32];
    ScalarBackend::spmm_kernel(&csr, 1.0, &b, 2, 0.5, &mut c);
    // C[0,0] = 1.0*2.0*1.0 + 0.5*10.0 = 7.0
    // C[0,1] = 1.0*2.0*0.0 + 0.5*20.0 = 10.0
    // C[1,0] = 1.0*3.0*0.0 + 0.5*30.0 = 15.0
    // C[1,1] = 1.0*3.0*1.0 + 0.5*40.0 = 23.0
    assert!((c[0] - 7.0).abs() < 1e-5);
    assert!((c[1] - 10.0).abs() < 1e-5);
    assert!((c[2] - 15.0).abs() < 1e-5);
    assert!((c[3] - 23.0).abs() < 1e-5);
    Ok(())
}

// ============================================================================
// POPPERIAN FALSIFICATION: Adversarial edge cases (GH #152)
// ============================================================================

#[test]
fn test_falsify_spmv_1x1_matrix() {
    let a = CsrMatrix::new(1, 1, vec![0, 1], vec![0], vec![7.0_f32]).unwrap();
    let mut y = vec![0.0_f32];
    a.spmv(1.0, &[3.0], 0.0, &mut y).unwrap();
    assert!((y[0] - 21.0).abs() < 1e-5);
}

#[test]
fn test_falsify_spmv_zero_nnz_matrix() {
    let a = CsrMatrix::<f32>::new(3, 3, vec![0, 0, 0, 0], vec![], vec![]).unwrap();
    let mut y = vec![99.0_f32; 3];
    // beta=0 should zero out y even with no nonzeros
    a.spmv(1.0, &[1.0, 2.0, 3.0], 0.0, &mut y).unwrap();
    for &v in &y {
        assert!(v.abs() < 1e-7, "Zero-nnz matrix should give zero output, got {v}");
    }
}

#[test]
fn test_falsify_spmv_beta_preserves_with_zero_nnz() {
    let a = CsrMatrix::<f32>::new(2, 2, vec![0, 0, 0], vec![], vec![]).unwrap();
    let mut y = vec![10.0, 20.0_f32];
    // y = 0*A*x + 0.5*y = [5, 10]
    a.spmv(1.0, &[1.0, 1.0], 0.5, &mut y).unwrap();
    assert!((y[0] - 5.0).abs() < 1e-5);
    assert!((y[1] - 10.0).abs() < 1e-5);
}

#[test]
fn test_falsify_coo_duplicate_entries_sum() {
    // Duplicate (0,0) entries: should accumulate in CSR
    let coo = CooMatrix::new(2, 2, vec![0, 0, 1], vec![0, 0, 1], vec![3.0, 4.0, 5.0_f32]).unwrap();
    let csr = CsrMatrix::from_coo(&coo);
    let dense = csr.to_dense();
    // (0,0) should be 3+4=7
    assert!((dense[0] - 7.0).abs() < 1e-5, "Duplicate entries should sum: got {}", dense[0]);
    assert!((dense[3] - 5.0).abs() < 1e-5);
}

#[test]
fn test_falsify_spmm_zero_cols_b() -> Result<(), Box<dyn std::error::Error>> {
    let a = CsrMatrix::<f32>::identity(3);
    let b: Vec<f32> = vec![]; // 3×0 dense matrix
    let mut c: Vec<f32> = vec![];
    a.spmm(1.0, &b, 0, 0.0, &mut c)?;
    assert!(c.is_empty());
    Ok(())
}

#[test]
fn test_falsify_spgemm_empty_matrices() -> Result<(), Box<dyn std::error::Error>> {
    let a = CsrMatrix::<f32>::new(2, 3, vec![0, 0, 0], vec![], vec![])?;
    let b = CsrMatrix::<f32>::new(3, 2, vec![0, 0, 0, 0], vec![], vec![])?;
    let c = crate::spgemm::spgemm(&a, &b)?;
    assert_eq!(c.nnz(), 0);
    assert_eq!(c.rows(), 2);
    assert_eq!(c.cols(), 2);
    Ok(())
}

#[test]
fn test_falsify_sell_single_row() -> Result<(), Box<dyn std::error::Error>> {
    let csr = CsrMatrix::new(1, 3, vec![0, 2], vec![0, 2], vec![1.0, 3.0_f32])?;
    let sell = SellMatrix::from_csr(&csr, 4);
    let mut y = vec![0.0_f32];
    sell.spmv(1.0, &[1.0, 2.0, 3.0], 0.0, &mut y)?;
    // 1*1 + 3*3 = 10
    assert!((y[0] - 10.0).abs() < 1e-5);
    Ok(())
}

#[test]
fn test_falsify_bsr_single_block() -> Result<(), Box<dyn std::error::Error>> {
    let dense = [5.0, 3.0, 3.0, 5.0_f32]; // 2×2 single block
    let bsr = BsrMatrix::from_dense(&dense, 2, 2, 2);
    assert_eq!(bsr.nnz_blocks(), 1);
    let mut y = vec![0.0_f32; 2];
    bsr.spmv(1.0, &[1.0, 1.0], 0.0, &mut y)?;
    assert!((y[0] - 8.0).abs() < 1e-5);
    assert!((y[1] - 8.0).abs() < 1e-5);
    Ok(())
}
