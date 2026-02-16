pub(crate) use super::*;

#[test]
fn test_from_vec() {
    let m = Matrix::from_vec(2, 3, vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0])
        .expect("test data has correct dimensions: 2*3=6 elements");
    assert_eq!(m.shape(), (2, 3));
    assert!((m.get(0, 0) - 1.0).abs() < 1e-6);
    assert!((m.get(1, 2) - 6.0).abs() < 1e-6);
}

#[test]
fn test_from_vec_error() {
    let result = Matrix::from_vec(2, 3, vec![1.0_f32, 2.0, 3.0]);
    assert!(result.is_err());
}

#[test]
fn test_zeros() {
    let m = Matrix::<f32>::zeros(2, 3);
    assert_eq!(m.shape(), (2, 3));
    assert!(m.as_slice().iter().all(|&x| x == 0.0));
}

#[test]
fn test_eye() {
    let m = Matrix::<f32>::eye(3);
    assert!((m.get(0, 0) - 1.0).abs() < 1e-6);
    assert!((m.get(1, 1) - 1.0).abs() < 1e-6);
    assert!((m.get(2, 2) - 1.0).abs() < 1e-6);
    assert!((m.get(0, 1) - 0.0).abs() < 1e-6);
}

#[test]
fn test_transpose() {
    let m = Matrix::from_vec(2, 3, vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0])
        .expect("test data has correct dimensions: 2*3=6 elements");
    let t = m.transpose();
    assert_eq!(t.shape(), (3, 2));
    assert!((t.get(0, 0) - 1.0).abs() < 1e-6);
    assert!((t.get(0, 1) - 4.0).abs() < 1e-6);
    assert!((t.get(2, 1) - 6.0).abs() < 1e-6);
}

#[test]
fn test_row() {
    let m = Matrix::from_vec(2, 3, vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0])
        .expect("test data has correct dimensions: 2*3=6 elements");
    let row = m.row(1);
    assert_eq!(row.len(), 3);
    assert!((row[0] - 4.0).abs() < 1e-6);
    assert!((row[1] - 5.0).abs() < 1e-6);
    assert!((row[2] - 6.0).abs() < 1e-6);
}

#[test]
fn test_column() {
    let m = Matrix::from_vec(2, 3, vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0])
        .expect("test data has correct dimensions: 2*3=6 elements");
    let col = m.column(1);
    assert_eq!(col.len(), 2);
    assert!((col[0] - 2.0).abs() < 1e-6);
    assert!((col[1] - 5.0).abs() < 1e-6);
}

#[test]
fn test_matmul() {
    // 2x3 * 3x2 = 2x2
    let a = Matrix::from_vec(2, 3, vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0])
        .expect("test data has correct dimensions: 2*3=6 elements");
    let b = Matrix::from_vec(3, 2, vec![7.0_f32, 8.0, 9.0, 10.0, 11.0, 12.0])
        .expect("test data has correct dimensions: 3*2=6 elements");
    let c = a
        .matmul(&b)
        .expect("matrix dimensions are compatible for multiplication: 2x3 * 3x2");

    assert_eq!(c.shape(), (2, 2));
    // c[0,0] = 1*7 + 2*9 + 3*11 = 7 + 18 + 33 = 58
    assert!((c.get(0, 0) - 58.0).abs() < 1e-6);
    // c[0,1] = 1*8 + 2*10 + 3*12 = 8 + 20 + 36 = 64
    assert!((c.get(0, 1) - 64.0).abs() < 1e-6);
}

#[test]
fn test_matmul_dimension_error() {
    let a = Matrix::from_vec(2, 3, vec![1.0_f32; 6])
        .expect("test data has correct dimensions: 2*3=6 elements");
    let b = Matrix::from_vec(2, 2, vec![1.0_f32; 4])
        .expect("test data has correct dimensions: 2*2=4 elements");
    assert!(a.matmul(&b).is_err());
}

#[test]
fn test_matvec() {
    let m = Matrix::from_vec(2, 3, vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0])
        .expect("test data has correct dimensions: 2*3=6 elements");
    let v = Vector::from_slice(&[1.0_f32, 2.0, 3.0]);
    let result = m
        .matvec(&v)
        .expect("matrix columns match vector length: both 3");

    assert_eq!(result.len(), 2);
    // result[0] = 1*1 + 2*2 + 3*3 = 14
    assert!((result[0] - 14.0).abs() < 1e-6);
    // result[1] = 4*1 + 5*2 + 6*3 = 32
    assert!((result[1] - 32.0).abs() < 1e-6);
}

#[test]
fn test_add() {
    let a = Matrix::from_vec(2, 2, vec![1.0_f32, 2.0, 3.0, 4.0])
        .expect("test data has correct dimensions: 2*2=4 elements");
    let b = Matrix::from_vec(2, 2, vec![5.0_f32, 6.0, 7.0, 8.0])
        .expect("test data has correct dimensions: 2*2=4 elements");
    let c = a.add(&b).expect("both matrices have same dimensions: 2x2");

    assert!((c.get(0, 0) - 6.0).abs() < 1e-6);
    assert!((c.get(1, 1) - 12.0).abs() < 1e-6);
}

#[test]
fn test_add_dimension_mismatch() {
    // Test that mismatched dimensions are detected (catches || → && mutation)
    let a = Matrix::from_vec(2, 2, vec![1.0_f32; 4])
        .expect("test data has correct dimensions: 2*2=4 elements");
    let b = Matrix::from_vec(3, 2, vec![1.0_f32; 6])
        .expect("test data has correct dimensions: 3*2=6 elements");
    assert!(a.add(&b).is_err());

    let c = Matrix::from_vec(2, 3, vec![1.0_f32; 6])
        .expect("test data has correct dimensions: 2*3=6 elements");
    assert!(a.add(&c).is_err());
}

#[test]
fn test_sub() {
    // Test element-wise subtraction
    let a = Matrix::from_vec(2, 2, vec![10.0_f32, 8.0, 6.0, 12.0])
        .expect("test data has correct dimensions: 2*2=4 elements");
    let b = Matrix::from_vec(2, 2, vec![4.0_f32, 3.0, 2.0, 7.0])
        .expect("test data has correct dimensions: 2*2=4 elements");
    let c = a.sub(&b).expect("both matrices have same dimensions: 2x2");

    // Verify all elements: a[i] - b[i]
    assert!((c.get(0, 0) - 6.0).abs() < 1e-6); // 10 - 4 = 6
    assert!((c.get(0, 1) - 5.0).abs() < 1e-6); // 8 - 3 = 5
    assert!((c.get(1, 0) - 4.0).abs() < 1e-6); // 6 - 2 = 4
    assert!((c.get(1, 1) - 5.0).abs() < 1e-6); // 12 - 7 = 5
}

#[test]
fn test_sub_dimension_mismatch_rows() {
    // Test that mismatched rows are detected
    let a = Matrix::from_vec(2, 2, vec![1.0_f32; 4])
        .expect("test data has correct dimensions: 2*2=4 elements");
    let b = Matrix::from_vec(3, 2, vec![1.0_f32; 6])
        .expect("test data has correct dimensions: 3*2=6 elements");
    assert!(a.sub(&b).is_err());
}

#[test]
fn test_sub_dimension_mismatch_cols() {
    // Test that mismatched columns are detected
    let a = Matrix::from_vec(2, 2, vec![1.0_f32; 4])
        .expect("test data has correct dimensions: 2*2=4 elements");
    let b = Matrix::from_vec(2, 3, vec![1.0_f32; 6])
        .expect("test data has correct dimensions: 2*3=6 elements");
    assert!(a.sub(&b).is_err());
}

#[test]
fn test_cholesky_solve() {
    // Solve A*x = b where A is symmetric positive definite
    // A = [[4, 2], [2, 3]]
    // b = [1, 2]
    // Solution: x = [-0.125, 0.75]
    let a = Matrix::from_vec(2, 2, vec![4.0_f32, 2.0, 2.0, 3.0])
        .expect("test data has correct dimensions: 2*2=4 elements");
    let b = Vector::from_slice(&[1.0_f32, 2.0]);
    let x = a
        .cholesky_solve(&b)
        .expect("matrix is square, symmetric positive definite, and vector matches size");

    assert_eq!(x.len(), 2);
    assert!((x[0] - (-0.125)).abs() < 1e-5);
    assert!((x[1] - 0.75).abs() < 1e-5);
}

#[test]
fn test_cholesky_solve_3x3() {
    // A = [[4, 12, -16], [12, 37, -43], [-16, -43, 98]]
    // b = [1, 2, 3]
    let a = Matrix::from_vec(
        3,
        3,
        vec![4.0_f32, 12.0, -16.0, 12.0, 37.0, -43.0, -16.0, -43.0, 98.0],
    )
    .expect("test data has correct dimensions: 3*3=9 elements");
    let b = Vector::from_slice(&[1.0_f32, 2.0, 3.0]);
    let x = a
        .cholesky_solve(&b)
        .expect("matrix is square, symmetric positive definite, and vector matches size");

    // Verify A*x ≈ b
    let result = a
        .matvec(&x)
        .expect("matrix columns match vector length: both 3");
    for i in 0..3 {
        assert!((result[i] - b[i]).abs() < 1e-4);
    }
}

#[test]
fn test_cholesky_solve_strict() {
    // Stricter test to catch arithmetic mutations in cholesky_solve
    // Uses a 4x4 SPD matrix to exercise all accumulation loops
    // A = [[4, 2, 1, 1],
    //      [2, 5, 2, 1],
    //      [1, 2, 6, 2],
    //      [1, 1, 2, 7]]
    // This is symmetric positive definite with non-trivial decomposition
    let a = Matrix::from_vec(
        4,
        4,
        vec![
            4.0_f32, 2.0, 1.0, 1.0, 2.0, 5.0, 2.0, 1.0, 1.0, 2.0, 6.0, 2.0, 1.0, 1.0, 2.0, 7.0,
        ],
    )
    .expect("test data has correct dimensions: 4*4=16 elements");
    let b = Vector::from_slice(&[1.0_f32, 2.0, 3.0, 4.0]);
    let x = a
        .cholesky_solve(&b)
        .expect("matrix is square, symmetric positive definite, and vector matches size");

    // Verify A*x = b with very tight tolerance
    let result = a
        .matvec(&x)
        .expect("matrix columns match vector length: both 4");
    for i in 0..4 {
        assert!(
            (result[i] - b[i]).abs() < 1e-5,
            "Failed at index {}: expected {}, got {}",
            i,
            b[i],
            result[i]
        );
    }

    // Also test a 3x3 case with known solution
    // A = [[9, 3, 3], [3, 5, 1], [3, 1, 4]], b = [15, 9, 8] => x = [1, 1, 1]
    let a3 = Matrix::from_vec(3, 3, vec![9.0_f32, 3.0, 3.0, 3.0, 5.0, 1.0, 3.0, 1.0, 4.0])
        .expect("test data has correct dimensions: 3*3=9 elements");
    let b3 = Vector::from_slice(&[15.0_f32, 9.0, 8.0]);
    let x3 = a3
        .cholesky_solve(&b3)
        .expect("matrix is square, symmetric positive definite, and vector matches size");

    // Verify exact solution [1, 1, 1] with element-by-element check
    assert!((x3[0] - 1.0).abs() < 1e-6);
    assert!((x3[1] - 1.0).abs() < 1e-6);
    assert!((x3[2] - 1.0).abs() < 1e-6);

    // Additional verification: check that A*x3 = b3 with strict tolerance
    let verify3 = a3
        .matvec(&x3)
        .expect("matrix columns match vector length: both 3");
    assert!((verify3[0] - 15.0).abs() < 1e-6);
    assert!((verify3[1] - 9.0).abs() < 1e-6);
    assert!((verify3[2] - 8.0).abs() < 1e-6);
}

#[test]
fn test_mul_scalar() {
    let m = Matrix::from_vec(2, 2, vec![1.0_f32, 2.0, 3.0, 4.0])
        .expect("test data has correct dimensions: 2*2=4 elements");
    let result = m.mul_scalar(2.0);
    assert!((result.get(0, 0) - 2.0).abs() < 1e-6);
    assert!((result.get(1, 1) - 8.0).abs() < 1e-6);
}

#[test]
fn test_set() {
    let mut m = Matrix::<f32>::zeros(2, 2);
    m.set(0, 1, 5.0);
    assert!((m.get(0, 1) - 5.0).abs() < 1e-6);
}
