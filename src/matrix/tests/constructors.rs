use super::*;
use crate::TruenoError;

#[test]
fn test_matrix_new() {
    let m = Matrix::new(3, 4);
    assert_eq!(m.rows(), 3);
    assert_eq!(m.cols(), 4);
    assert_eq!(m.shape(), (3, 4));
    assert_eq!(m.as_slice().len(), 12);
}

#[test]
fn test_matrix_from_vec() {
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let m = Matrix::from_vec(2, 2, data).unwrap();
    assert_eq!(m.rows(), 2);
    assert_eq!(m.cols(), 2);
    assert_eq!(m.get(0, 0), Some(&1.0));
    assert_eq!(m.get(0, 1), Some(&2.0));
    assert_eq!(m.get(1, 0), Some(&3.0));
    assert_eq!(m.get(1, 1), Some(&4.0));
}

#[test]
fn test_matrix_from_vec_invalid_size() {
    let data = vec![1.0, 2.0, 3.0];
    let result = Matrix::from_vec(2, 2, data);
    assert!(matches!(result, Err(TruenoError::InvalidInput(_))));
}

#[test]
fn test_matrix_from_slice() {
    // TRUENO-SPEC-014 coverage test
    let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let m = Matrix::from_slice(2, 3, &data).unwrap();
    assert_eq!(m.rows(), 2);
    assert_eq!(m.cols(), 3);
    assert_eq!(m.get(0, 0), Some(&1.0));
    assert_eq!(m.get(1, 2), Some(&6.0));
}

#[test]
fn test_matrix_from_slice_invalid() {
    // TRUENO-SPEC-014 coverage test - error path
    let data = [1.0, 2.0, 3.0];
    let result = Matrix::from_slice(2, 2, &data);
    assert!(matches!(result, Err(TruenoError::InvalidInput(_))));
}

#[test]
fn test_matrix_zeros() {
    let m = Matrix::zeros(2, 3);
    assert_eq!(m.rows(), 2);
    assert_eq!(m.cols(), 3);
    for &val in m.as_slice() {
        assert_eq!(val, 0.0);
    }
}

#[test]
fn test_matrix_identity() {
    let m = Matrix::identity(3);
    assert_eq!(m.rows(), 3);
    assert_eq!(m.cols(), 3);

    // Check diagonal
    assert_eq!(m.get(0, 0), Some(&1.0));
    assert_eq!(m.get(1, 1), Some(&1.0));
    assert_eq!(m.get(2, 2), Some(&1.0));

    // Check off-diagonal
    assert_eq!(m.get(0, 1), Some(&0.0));
    assert_eq!(m.get(0, 2), Some(&0.0));
    assert_eq!(m.get(1, 0), Some(&0.0));
    assert_eq!(m.get(1, 2), Some(&0.0));
    assert_eq!(m.get(2, 0), Some(&0.0));
    assert_eq!(m.get(2, 1), Some(&0.0));
}

#[test]
fn test_matrix_get_out_of_bounds() {
    let m = Matrix::new(2, 2);
    assert_eq!(m.get(2, 0), None);
    assert_eq!(m.get(0, 2), None);
    assert_eq!(m.get(2, 2), None);
}
