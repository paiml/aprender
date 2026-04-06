use super::*;
use crate::Vector;

// =========================================================================
// Additional coverage tests for untested paths
// =========================================================================

#[test]
fn test_get_mut_valid() {
    let mut m = Matrix::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    if let Some(val) = m.get_mut(0, 1) {
        *val = 99.0;
    }
    assert_eq!(m.get(0, 1), Some(&99.0));
}

#[test]
fn test_get_mut_out_of_bounds() {
    let mut m = Matrix::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    assert!(m.get_mut(5, 0).is_none());
    assert!(m.get_mut(0, 10).is_none());
    assert!(m.get_mut(10, 10).is_none());
}

#[test]
fn test_matrix_zeros_coverage() {
    let m: Matrix<f32> = Matrix::zeros(3, 4);
    assert_eq!(m.rows(), 3);
    assert_eq!(m.cols(), 4);
    for val in m.as_slice() {
        assert_eq!(*val, 0.0);
    }
}

#[test]
fn test_matrix_identity_coverage() {
    let m: Matrix<f32> = Matrix::identity(3);
    assert_eq!(m.get(0, 0), Some(&1.0));
    assert_eq!(m.get(1, 1), Some(&1.0));
    assert_eq!(m.get(2, 2), Some(&1.0));
    assert_eq!(m.get(0, 1), Some(&0.0));
    assert_eq!(m.get(1, 0), Some(&0.0));
}

#[test]
fn test_get_out_of_bounds_coverage() {
    let m = Matrix::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    assert!(m.get(5, 0).is_none());
    assert!(m.get(0, 10).is_none());
}

// =========================================================================
// Kitchen Sink coverage tests (PMAT-018)
// =========================================================================

/// Test pad method with various configurations
#[test]
fn test_pad_kitchen_sink() {
    let m = Matrix::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

    // Symmetric padding
    let padded = m.pad(((1, 1), (1, 1)), 0.0).unwrap();
    assert_eq!(padded.rows(), 4);
    assert_eq!(padded.cols(), 5);
    assert_eq!(padded.get(0, 0), Some(&0.0)); // top-left corner (padding)
    assert_eq!(padded.get(1, 1), Some(&1.0)); // original data starts here

    // Asymmetric padding
    let padded2 = m.pad(((2, 0), (0, 3)), 9.0).unwrap();
    assert_eq!(padded2.rows(), 4);
    assert_eq!(padded2.cols(), 6);
    assert_eq!(padded2.get(0, 0), Some(&9.0)); // padding
    assert_eq!(padded2.get(2, 0), Some(&1.0)); // original data

    // Zero padding
    let padded3 = m.pad(((0, 0), (0, 0)), 0.0).unwrap();
    assert_eq!(padded3.rows(), 2);
    assert_eq!(padded3.cols(), 3);
}

/// Test gather with various axis configurations
#[test]
fn test_gather_kitchen_sink() {
    let m = Matrix::from_vec(3, 4, (0..12).map(|x| x as f32).collect()).unwrap();

    // Gather rows (axis=0)
    let rows = m.gather(&[0, 2], 0).unwrap();
    assert_eq!(rows.rows(), 2);
    assert_eq!(rows.cols(), 4);
    assert_eq!(rows.get(0, 0), Some(&0.0)); // row 0
    assert_eq!(rows.get(1, 0), Some(&8.0)); // row 2

    // Gather columns (axis=1)
    let cols = m.gather(&[1, 3], 1).unwrap();
    assert_eq!(cols.rows(), 3);
    assert_eq!(cols.cols(), 2);
}

/// Test topk with various k values
#[test]
fn test_topk_kitchen_sink() {
    let m = Matrix::from_vec(2, 4, vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0]).unwrap();

    // topk operates on flattened data, not per-row
    // k=1 returns top 1 value from all 8 elements
    let (vals, idxs) = m.topk(1).unwrap();
    assert_eq!(vals.len(), 1);
    assert_eq!(idxs.len(), 1);
    assert_eq!(vals[0], 9.0); // max value in matrix

    // k=3 returns top 3 values from all 8 elements
    let (vals2, idxs2) = m.topk(3).unwrap();
    assert_eq!(vals2.len(), 3);
    assert_eq!(idxs2.len(), 3);

    // k=8 (all elements)
    let (vals3, _) = m.topk(8).unwrap();
    assert_eq!(vals3.len(), 8);

    // k=0 edge case
    let (vals4, idxs4) = m.topk(0).unwrap();
    assert_eq!(vals4.len(), 0);
    assert_eq!(idxs4.len(), 0);
}

/// Test pooling operations edge cases
#[test]
fn test_pooling_kitchen_sink() {
    // Exact divisible size
    let m = Matrix::from_vec(4, 4, (0..16).map(|x| x as f32).collect()).unwrap();

    let max_pooled = m.max_pool2d((2, 2), (2, 2)).unwrap();
    assert_eq!(max_pooled.rows(), 2);
    assert_eq!(max_pooled.cols(), 2);

    let avg_pooled = m.avg_pool2d((2, 2), (2, 2)).unwrap();
    assert_eq!(avg_pooled.rows(), 2);
    assert_eq!(avg_pooled.cols(), 2);

    // Non-exact divisible size
    let m2 = Matrix::from_vec(5, 5, (0..25).map(|x| x as f32).collect()).unwrap();
    let max_pooled2 = m2.max_pool2d((2, 2), (2, 2)).unwrap();
    assert_eq!(max_pooled2.rows(), 2); // floor(5/2)
    assert_eq!(max_pooled2.cols(), 2);
}

/// Test vecmat (v @ M) operation
#[test]
fn test_vecmat_kitchen_sink() {
    let v = Vector::from_slice(&[1.0, 2.0, 3.0]);
    let m =
        Matrix::from_vec(3, 4, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
            .unwrap();

    // v (1x3) @ M (3x4) -> result (1x4)
    let result = Matrix::vecmat(&v, &m).unwrap();
    assert_eq!(result.len(), 4);
    // Manual calculation: [1*1+2*5+3*9, 1*2+2*6+3*10, 1*3+2*7+3*11, 1*4+2*8+3*12]
    // = [1+10+27, 2+12+30, 3+14+33, 4+16+36] = [38, 44, 50, 56]
    assert!((result.as_slice()[0] - 38.0).abs() < 1e-5);
    assert!((result.as_slice()[1] - 44.0).abs() < 1e-5);
    assert!((result.as_slice()[2] - 50.0).abs() < 1e-5);
    assert!((result.as_slice()[3] - 56.0).abs() < 1e-5);
}

/// Test convolve2d edge cases
#[test]
fn test_convolve2d_kitchen_sink() {
    // 3x3 input with 3x3 kernel (produces 1x1)
    let input = Matrix::from_vec(3, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]).unwrap();

    let kernel = Matrix::from_vec(3, 3, vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]).unwrap();

    let result = input.convolve2d(&kernel).unwrap();
    assert_eq!(result.rows(), 1);
    assert_eq!(result.cols(), 1);
    assert!((result.get(0, 0).unwrap() - 5.0).abs() < 1e-5); // Center value

    // 5x5 input with 3x3 kernel (produces 3x3)
    let input5 = Matrix::from_vec(5, 5, (0..25).map(|x| x as f32).collect()).unwrap();
    let result5 = input5.convolve2d(&kernel).unwrap();
    assert_eq!(result5.rows(), 3);
    assert_eq!(result5.cols(), 3);
}

/// Test embedding lookups
#[test]
fn test_embedding_kitchen_sink() {
    // Embedding table: 5 words x 4 dimensions
    let embeddings = Matrix::from_vec(5, 4, (0..20).map(|x| x as f32).collect()).unwrap();

    // Single lookup
    let result = embeddings.embedding_lookup(&[0]).unwrap();
    assert_eq!(result.rows(), 1);
    assert_eq!(result.cols(), 4);

    // Multiple lookups
    let result2 = embeddings.embedding_lookup(&[0, 2, 4]).unwrap();
    assert_eq!(result2.rows(), 3);
    assert_eq!(result2.cols(), 4);

    // Sparse lookup (returns matrix and indices)
    let (result3_matrix, result3_indices) = embeddings.embedding_lookup_sparse(&[0, 1, 2]).unwrap();
    assert_eq!(result3_matrix.rows(), 3);
    assert_eq!(result3_matrix.cols(), 4);
    assert_eq!(result3_indices.len(), 3);
}

/// Test batched_matmul_4d
#[test]
fn test_batched_matmul_4d_kitchen_sink() {
    // batch=2, heads=2, m=3, k=4, n=5
    let batch = 2;
    let heads = 2;
    let m = 3;
    let k = 4;
    let n = 5;

    // A: [batch, heads, m, k] = 2*2*3*4 = 48 elements
    let a_data: Vec<f32> = (0..48).map(|x| x as f32 * 0.1).collect();

    // B: [batch, heads, k, n] = 2*2*4*5 = 80 elements
    let b_data: Vec<f32> = (0..80).map(|x| x as f32 * 0.1).collect();

    // Result should be [batch, heads, m, n] = 2*2*3*5 = 60 elements
    let result = Matrix::batched_matmul_4d(&a_data, &b_data, batch, heads, m, k, n).unwrap();
    assert_eq!(result.len(), batch * heads * m * n);
}

/// Test matmul with remainder handling (non-aligned sizes)
#[test]
fn test_matmul_remainder_kitchen_sink() {
    // Sizes that don't align with SIMD widths
    for m in [1, 3, 5, 7, 9, 13, 17] {
        for k in [1, 3, 5, 7, 9, 15] {
            for n in [1, 3, 5, 7, 11] {
                let a = Matrix::from_vec(m, k, vec![1.0; m * k]).unwrap();
                let b = Matrix::from_vec(k, n, vec![1.0; k * n]).unwrap();
                let c = a.matmul(&b).unwrap();
                assert_eq!(c.rows(), m);
                assert_eq!(c.cols(), n);
                // Each element should equal k (sum of 1.0 x 1.0, k times)
                assert!((c.get(0, 0).unwrap() - k as f32).abs() < 1e-4);
            }
        }
    }
}

/// Test transpose edge cases
#[test]
fn test_transpose_kitchen_sink() {
    // 1xN
    let row = Matrix::from_vec(1, 5, vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
    let col = row.transpose();
    assert_eq!(col.rows(), 5);
    assert_eq!(col.cols(), 1);

    // Nx1
    let col2 = Matrix::from_vec(5, 1, vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
    let row2 = col2.transpose();
    assert_eq!(row2.rows(), 1);
    assert_eq!(row2.cols(), 5);

    // 1x1
    let single = Matrix::from_vec(1, 1, vec![42.0]).unwrap();
    let single_t = single.transpose();
    assert_eq!(single_t.get(0, 0), Some(&42.0));
}
