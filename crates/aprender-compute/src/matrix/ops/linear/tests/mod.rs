use crate::{Backend, Matrix, TruenoError, Vector};

mod backend;
mod edge_cases;
mod parallel;

#[test]
fn test_transpose_square() {
    let m = Matrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let t = m.transpose();
    assert_eq!(t.rows(), 2);
    assert_eq!(t.cols(), 2);
    assert_eq!(t.get(0, 0), Some(&1.0));
    assert_eq!(t.get(0, 1), Some(&3.0));
    assert_eq!(t.get(1, 0), Some(&2.0));
    assert_eq!(t.get(1, 1), Some(&4.0));
}

#[test]
fn test_transpose_rect() {
    let m = Matrix::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let t = m.transpose();
    assert_eq!(t.rows(), 3);
    assert_eq!(t.cols(), 2);
    assert_eq!(t.get(0, 0), Some(&1.0));
    assert_eq!(t.get(0, 1), Some(&4.0));
    assert_eq!(t.get(1, 0), Some(&2.0));
}

#[test]
fn test_matvec_basic() {
    let m = Matrix::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let v = Vector::from_slice(&[1.0, 2.0, 3.0]);
    let result = m.matvec(&v).unwrap();
    assert_eq!(result.as_slice(), &[14.0, 32.0]);
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
    assert_eq!(result.as_slice(), &[9.0, 12.0, 15.0]);
}

#[test]
fn test_vecmat_dimension_mismatch() {
    let m = Matrix::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let v = Vector::from_slice(&[1.0, 2.0, 3.0]); // Wrong size
    assert!(Matrix::vecmat(&v, &m).is_err());
}

/// Helper: build a matrix with a specific backend and run matvec, verifying results.
fn assert_matvec_backend(
    rows: usize,
    cols: usize,
    mat_data: Vec<f32>,
    vec_data: &[f32],
    expected: &[f32],
    backend: Backend,
    tolerance: f32,
    label: &str,
) {
    let m = Matrix::from_vec_with_backend(rows, cols, mat_data, backend);
    let v = Vector::from_slice(vec_data);
    let result = m.matvec(&v).unwrap();
    assert_eq!(result.as_slice().len(), expected.len(), "{label}: length mismatch");
    for (i, (&got, &exp)) in result.as_slice().iter().zip(expected.iter()).enumerate() {
        assert!((got - exp).abs() < tolerance, "{label} at index {i}: got {got} expected {exp}",);
    }
}
