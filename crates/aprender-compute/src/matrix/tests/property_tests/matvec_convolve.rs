use super::super::*;

// Unit tests for matrix-vector operations
#[test]
fn test_matvec_basic() {
    let m = Matrix::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let v = Vector::from_slice(&[1.0, 2.0, 3.0]);
    let result = m.matvec(&v).unwrap();

    // [[1, 2, 3]   [1]   [14]
    //  [4, 5, 6]] x [2] = [32]
    //               [3]
    assert_eq!(result.len(), 2);
    assert!((result.as_slice()[0] - 14.0).abs() < 1e-6);
    assert!((result.as_slice()[1] - 32.0).abs() < 1e-6);
}

#[test]
fn test_matvec_identity() {
    let m = Matrix::identity(3);
    let v = Vector::from_slice(&[1.0, 2.0, 3.0]);
    let result = m.matvec(&v).unwrap();

    // Ixv = v
    assert_eq!(result.as_slice(), v.as_slice());
}

#[test]
fn test_matvec_dimension_mismatch() {
    let m = Matrix::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let v = Vector::from_slice(&[1.0, 2.0]); // Wrong size

    assert!(m.matvec(&v).is_err());
}

#[test]
fn test_vecmat_basic() {
    let m = Matrix::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let v = Vector::from_slice(&[1.0, 2.0]);
    let result = Matrix::vecmat(&v, &m).unwrap();

    // [1, 2] x [[1, 2, 3]  = [9, 12, 15]
    //           [4, 5, 6]]
    assert_eq!(result.len(), 3);
    assert!((result.as_slice()[0] - 9.0).abs() < 1e-6);
    assert!((result.as_slice()[1] - 12.0).abs() < 1e-6);
    assert!((result.as_slice()[2] - 15.0).abs() < 1e-6);
}

#[test]
fn test_vecmat_identity() {
    let m = Matrix::identity(3);
    let v = Vector::from_slice(&[1.0, 2.0, 3.0]);
    let result = Matrix::vecmat(&v, &m).unwrap();

    // vxI = v
    assert_eq!(result.as_slice(), v.as_slice());
}

#[test]
fn test_vecmat_dimension_mismatch() {
    let m = Matrix::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let v = Vector::from_slice(&[1.0, 2.0, 3.0]); // Wrong size

    assert!(Matrix::vecmat(&v, &m).is_err());
}

#[test]
fn test_matvec_zero_vector() {
    let m = Matrix::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let v = Vector::from_slice(&[0.0, 0.0, 0.0]);
    let result = m.matvec(&v).unwrap();

    // Ax0 = 0
    assert_eq!(result.as_slice(), &[0.0, 0.0]);
}

#[test]
fn test_vecmat_zero_vector() {
    let m = Matrix::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let v = Vector::from_slice(&[0.0, 0.0]);
    let result = Matrix::vecmat(&v, &m).unwrap();

    // 0xA = 0
    assert_eq!(result.as_slice(), &[0.0, 0.0, 0.0]);
}

#[test]
fn test_matvec_transpose_equivalence() {
    // v^T x A = (A^T x v)^T
    // If A is mxn and v is m-dimensional, then:
    // - v^T x A is n-dimensional
    // - A^T is nxm, so A^T x v needs v to be n-dimensional
    // Actually, this is wrong. Let me use correct equivalence:
    // If A is mxn, v is n-dimensional:
    // - A x v is m-dimensional (matrix-vector)
    // - A^T is nxm, u is m-dimensional:
    // - u^T x A is n-dimensional (vector-matrix)
    // These are equivalent when u = A x v

    let m = Matrix::from_vec(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let v = Vector::from_slice(&[1.0, 2.0]); // 2-dimensional

    // A x v (3x2 times 2D = 3D result)
    let av = m.matvec(&v).unwrap();

    // v^T x A^T (2D times 2x3 = 3D result)
    let m_t = m.transpose(); // Now 2x3
    let v_mt = Matrix::vecmat(&v, &m_t).unwrap();

    // (A x v)^T = v^T x A^T
    assert_eq!(av.as_slice(), v_mt.as_slice());
}

// ===== 2D Convolution Tests =====

#[test]
fn test_convolve2d_basic_3x3() {
    // Simple 3x3 convolution with identity kernel (should preserve input)
    let input = Matrix::from_vec(3, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]).unwrap();

    // 1x1 identity kernel (should return center pixel)
    let kernel = Matrix::from_vec(1, 1, vec![1.0]).unwrap();

    let result = input.convolve2d(&kernel).unwrap();

    // Result should be 3x3 (same input size with valid padding)
    assert_eq!(result.rows(), 3);
    assert_eq!(result.cols(), 3);
    assert_eq!(result.as_slice(), input.as_slice());
}

#[test]
fn test_convolve2d_edge_detection() {
    // Test edge detection with Sobel-like kernel
    let input = Matrix::from_vec(
        4,
        4,
        vec![
            1.0, 1.0, 1.0, 1.0, //
            1.0, 2.0, 2.0, 1.0, //
            1.0, 2.0, 2.0, 1.0, //
            1.0, 1.0, 1.0, 1.0, //
        ],
    )
    .unwrap();

    // Simple 3x3 horizontal edge detection kernel
    #[rustfmt::skip]
    let kernel = Matrix::from_vec(
        3,
        3,
        vec![
            -1.0, -1.0, -1.0,
             0.0,  0.0,  0.0,
             1.0,  1.0,  1.0,
        ],
    )
    .unwrap();

    let result = input.convolve2d(&kernel).unwrap();

    // Result should be 2x2 (4-3+1 = 2)
    assert_eq!(result.rows(), 2);
    assert_eq!(result.cols(), 2);
}

#[test]
fn test_convolve2d_averaging_filter() {
    // Test averaging filter (blur)
    let input = Matrix::from_vec(
        5,
        5,
        vec![
            0.0, 0.0, 0.0, 0.0, 0.0, //
            0.0, 0.0, 0.0, 0.0, 0.0, //
            0.0, 0.0, 9.0, 0.0, 0.0, // Center pixel
            0.0, 0.0, 0.0, 0.0, 0.0, //
            0.0, 0.0, 0.0, 0.0, 0.0, //
        ],
    )
    .unwrap();

    // 3x3 averaging kernel (all 1/9)
    let kernel_val = 1.0 / 9.0;
    let kernel = Matrix::from_vec(
        3,
        3,
        vec![
            kernel_val, kernel_val, kernel_val, //
            kernel_val, kernel_val, kernel_val, //
            kernel_val, kernel_val, kernel_val, //
        ],
    )
    .unwrap();

    let result = input.convolve2d(&kernel).unwrap();

    // Result should be 3x3
    assert_eq!(result.rows(), 3);
    assert_eq!(result.cols(), 3);

    // Center should be 1.0 (9/9)
    assert!((result.get(1, 1).unwrap() - 1.0).abs() < 1e-5);
}

#[test]
fn test_convolve2d_invalid_kernel() {
    let input = Matrix::from_vec(3, 3, vec![1.0; 9]).unwrap();

    // Kernel larger than input
    let kernel = Matrix::from_vec(4, 4, vec![1.0; 16]).unwrap();

    assert!(input.convolve2d(&kernel).is_err());
}
