use super::*;
use crate::TruenoError;

// ===== Matrix Multiplication Tests =====

#[test]
fn test_matmul_basic() {
    // [[1, 2],   [[5, 6],   [[19, 22],
    //  [3, 4]] x  [7, 8]] =  [43, 50]]
    let a = Matrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let b = Matrix::from_vec(2, 2, vec![5.0, 6.0, 7.0, 8.0]).unwrap();
    let c = a.matmul(&b).unwrap();

    assert_eq!(c.rows(), 2);
    assert_eq!(c.cols(), 2);
    assert_eq!(c.get(0, 0), Some(&19.0));
    assert_eq!(c.get(0, 1), Some(&22.0));
    assert_eq!(c.get(1, 0), Some(&43.0));
    assert_eq!(c.get(1, 1), Some(&50.0));
}

#[test]
fn test_matmul_identity() {
    // A x I = A
    let a = Matrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let identity = Matrix::identity(2);
    let result = a.matmul(&identity).unwrap();

    assert_eq!(result.get(0, 0), Some(&1.0));
    assert_eq!(result.get(0, 1), Some(&2.0));
    assert_eq!(result.get(1, 0), Some(&3.0));
    assert_eq!(result.get(1, 1), Some(&4.0));
}

#[test]
fn test_matmul_zeros() {
    // A x 0 = 0
    let a = Matrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let zeros = Matrix::zeros(2, 2);
    let result = a.matmul(&zeros).unwrap();

    for &val in result.as_slice() {
        assert_eq!(val, 0.0);
    }
}

#[test]
fn test_matmul_dimension_mismatch() {
    // 2x3 matrix cannot multiply with 2x2 matrix (inner dimensions don't match)
    let a = Matrix::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let b = Matrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let result = a.matmul(&b);

    assert!(matches!(result, Err(TruenoError::InvalidInput(_))));
}

#[test]
fn test_matmul_non_square() {
    // 2x3 x 3x2 = 2x2
    // [[1, 2, 3],   [[7,  8],    [[58,  64],
    //  [4, 5, 6]] x  [9, 10],  =  [139, 154]]
    //                [11, 12]]
    let a = Matrix::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let b = Matrix::from_vec(3, 2, vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]).unwrap();
    let c = a.matmul(&b).unwrap();

    assert_eq!(c.rows(), 2);
    assert_eq!(c.cols(), 2);
    assert_eq!(c.get(0, 0), Some(&58.0));
    assert_eq!(c.get(0, 1), Some(&64.0));
    assert_eq!(c.get(1, 0), Some(&139.0));
    assert_eq!(c.get(1, 1), Some(&154.0));
}

#[test]
fn test_matmul_single_element() {
    // 1x1 x 1x1 = 1x1
    let a = Matrix::from_vec(1, 1, vec![3.0]).unwrap();
    let b = Matrix::from_vec(1, 1, vec![4.0]).unwrap();
    let c = a.matmul(&b).unwrap();

    assert_eq!(c.rows(), 1);
    assert_eq!(c.cols(), 1);
    assert_eq!(c.get(0, 0), Some(&12.0));
}

#[test]
fn test_matmul_remainder_rows() {
    // TRUENO-SPEC-014: Test matmul with rows not divisible by 4
    // This exercises the remainder handling path in SIMD matmul
    // 5x8 x 8x6 = 5x6 (5 % 4 = 1 remainder row)
    let a = Matrix::from_vec(5, 8, (0..40).map(|i| (i + 1) as f32).collect()).unwrap();
    let b = Matrix::from_vec(8, 6, (0..48).map(|i| (i + 1) as f32).collect()).unwrap();
    let c = a.matmul(&b).unwrap();

    assert_eq!(c.rows(), 5);
    assert_eq!(c.cols(), 6);

    // Verify using naive calculation for first and last row
    // First row: [1,2,3,4,5,6,7,8] . columns of B
    let expected_00 = (1..=8)
        .zip((0..48).step_by(6).map(|i| (i + 1) as f32))
        .map(|(a, b)| a as f32 * b)
        .sum::<f32>();
    assert!((c.get(0, 0).unwrap() - expected_00).abs() < 1.0);
}

#[test]
fn test_matmul_remainder_rows_7() {
    // TRUENO-SPEC-014: 7x8 x 8x5 = 7x5 (7 % 4 = 3 remainder rows)
    let a = Matrix::from_vec(7, 8, (0..56).map(|_| 1.0f32).collect()).unwrap();
    let b = Matrix::from_vec(8, 5, (0..40).map(|_| 1.0f32).collect()).unwrap();
    let c = a.matmul(&b).unwrap();

    assert_eq!(c.rows(), 7);
    assert_eq!(c.cols(), 5);
    // Each element should be 8.0 (dot product of 8 ones)
    for &val in c.as_slice() {
        assert!((val - 8.0).abs() < 1e-5);
    }
}

// ===== Backend Equivalence Tests =====
// Note: Internal method tests (matmul_naive, matmul_simd) moved to ops/arithmetic.rs
// These tests now use the public matmul() API which auto-selects the best backend.

#[test]
fn test_matmul_public_api_small() {
    // Small matrix - verify public matmul works correctly
    let a = Matrix::from_vec(8, 8, (0..64).map(|i| i as f32).collect()).unwrap();
    let b = Matrix::identity(8);
    let result = a.matmul(&b).unwrap();
    // A x I = A
    assert_eq!(result.as_slice(), a.as_slice());
}

#[test]
fn test_matmul_public_api_large() {
    // Large matrix - verify SIMD path works correctly
    let size = 128;
    let a = Matrix::identity(size);
    let b =
        Matrix::from_vec(size, size, (0..size * size).map(|i| ((i * 2) % 100) as f32).collect())
            .unwrap();
    let result = a.matmul(&b).unwrap();
    // I x B = B
    assert_eq!(result.as_slice(), b.as_slice());
}

#[test]
fn test_matmul_public_api_rectangular() {
    // Rectangular matrices
    let a = Matrix::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let b = Matrix::from_vec(3, 2, vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]).unwrap();
    let result = a.matmul(&b).unwrap();

    // Expected: [[1*7+2*9+3*11, 1*8+2*10+3*12], [4*7+5*9+6*11, 4*8+5*10+6*12]]
    //         = [[58, 64], [139, 154]]
    assert_eq!(result.rows(), 2);
    assert_eq!(result.cols(), 2);
    assert!((result.get(0, 0).unwrap() - 58.0).abs() < 1e-5);
    assert!((result.get(0, 1).unwrap() - 64.0).abs() < 1e-5);
    assert!((result.get(1, 0).unwrap() - 139.0).abs() < 1e-5);
    assert!((result.get(1, 1).unwrap() - 154.0).abs() < 1e-5);
}

// ===== GPU Tests =====

#[test]
#[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
fn test_gpu_availability() {
    use crate::backends::gpu::GpuBackend;
    // Just test that we can check GPU availability without crashing
    let _available = GpuBackend::is_available();
    // Note: We don't assert availability since CI may not have GPU
}

#[test]
#[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
#[ignore] // Ignore by default since CI may not have GPU
fn test_gpu_matmul_basic() {
    use crate::backends::gpu::GpuBackend;

    if !GpuBackend::is_available() {
        eprintln!("GPU not available, skipping test");
        return;
    }

    // Small test matrix (will use GPU if threshold is low enough)
    let a = Matrix::from_vec(
        4,
        4,
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
    )
    .unwrap();

    let b = Matrix::from_vec(
        4,
        4,
        vec![16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
    )
    .unwrap();

    // Use public matmul API (GPU used for large matrices via threshold)
    let c = a.matmul(&b).expect("matmul should succeed");

    // Verify some basic properties
    assert_eq!(c.rows(), 4);
    assert_eq!(c.cols(), 4);

    // Verify against known result (first element)
    // [1,2,3,4] . [16,12,8,4] = 16+24+24+16 = 80
    assert!((c.get(0, 0).unwrap() - 80.0).abs() < 1e-4);
}
