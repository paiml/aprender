use super::*;

#[test]
fn test_matvec_single_element_1x1() {
    let m = Matrix::from_vec_with_backend(1, 1, vec![3.0], Backend::Scalar);
    let v = Vector::from_slice(&[5.0]);
    let result = m.matvec(&v).unwrap();
    assert!((result.as_slice()[0] - 15.0).abs() < 1e-6);
}

#[test]
fn test_matvec_single_row() {
    let m = Matrix::from_vec_with_backend(1, 4, vec![1.0, 2.0, 3.0, 4.0], Backend::Scalar);
    let v = Vector::from_slice(&[1.0, 1.0, 1.0, 1.0]);
    let result = m.matvec(&v).unwrap();
    assert!((result.as_slice()[0] - 10.0).abs() < 1e-6);
}

#[test]
fn test_matvec_single_column() {
    let m = Matrix::from_vec_with_backend(3, 1, vec![2.0, 4.0, 6.0], Backend::Scalar);
    let v = Vector::from_slice(&[3.0]);
    let result = m.matvec(&v).unwrap();
    assert_eq!(result.as_slice(), &[6.0, 12.0, 18.0]);
}

#[test]
fn test_matvec_identity_matrix() {
    let m = Matrix::from_vec(3, 3, vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]).unwrap();
    let v = Vector::from_slice(&[7.0, 11.0, 13.0]);
    let result = m.matvec(&v).unwrap();
    for (i, (&got, &exp)) in result.as_slice().iter().zip([7.0, 11.0, 13.0].iter()).enumerate() {
        assert!((got - exp).abs() < 1e-6, "identity matvec [{i}]: {got} != {exp}");
    }
}

#[test]
fn test_matvec_zero_matrix() {
    let m = Matrix::from_vec(2, 3, vec![0.0; 6]).unwrap();
    let v = Vector::from_slice(&[1.0, 2.0, 3.0]);
    let result = m.matvec(&v).unwrap();
    for &val in result.as_slice() {
        assert!((val - 0.0).abs() < 1e-6);
    }
}

#[test]
fn test_matvec_zero_vector() {
    let m = Matrix::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let v = Vector::from_slice(&[0.0, 0.0, 0.0]);
    let result = m.matvec(&v).unwrap();
    for &val in result.as_slice() {
        assert!((val - 0.0).abs() < 1e-6);
    }
}

#[test]
fn test_matvec_dimension_mismatch_error_message() {
    let m = Matrix::from_vec(2, 3, vec![1.0; 6]).unwrap();
    let v = Vector::from_slice(&[1.0, 2.0]);
    let err = m.matvec(&v).unwrap_err();
    match err {
        TruenoError::InvalidInput(msg) => {
            assert!(msg.contains('2'), "Error should mention vector length 2: {msg}");
            assert!(msg.contains('3'), "Error should mention matrix cols 3: {msg}");
        }
        other => panic!("Expected InvalidInput, got {other:?}"),
    }
}

#[test]
fn test_matvec_negative_values() {
    let m = Matrix::from_vec_with_backend(
        2,
        3,
        vec![-1.0, -2.0, -3.0, -4.0, -5.0, -6.0],
        Backend::Scalar,
    );
    let v = Vector::from_slice(&[-1.0, -2.0, -3.0]);
    let result = m.matvec(&v).unwrap();
    assert!((result.as_slice()[0] - 14.0).abs() < 1e-5);
    assert!((result.as_slice()[1] - 32.0).abs() < 1e-5);
}
