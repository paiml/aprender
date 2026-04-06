use super::*;

// ===== Transpose Tests =====

#[test]
fn test_transpose_basic() {
    // [[1, 2, 3],     [[1, 4],
    //  [4, 5, 6]]  ->  [2, 5],
    //                   [3, 6]]
    let m = Matrix::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let t = m.transpose();

    assert_eq!(t.rows(), 3);
    assert_eq!(t.cols(), 2);
    assert_eq!(t.get(0, 0), Some(&1.0));
    assert_eq!(t.get(0, 1), Some(&4.0));
    assert_eq!(t.get(1, 0), Some(&2.0));
    assert_eq!(t.get(1, 1), Some(&5.0));
    assert_eq!(t.get(2, 0), Some(&3.0));
    assert_eq!(t.get(2, 1), Some(&6.0));
}

#[test]
fn test_transpose_square() {
    // [[1, 2],     [[1, 3],
    //  [3, 4]]  ->  [2, 4]]
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
fn test_transpose_single_row() {
    // [[1, 2, 3]] -> [[1],
    //                  [2],
    //                  [3]]
    let m = Matrix::from_vec(1, 3, vec![1.0, 2.0, 3.0]).unwrap();
    let t = m.transpose();

    assert_eq!(t.rows(), 3);
    assert_eq!(t.cols(), 1);
    assert_eq!(t.get(0, 0), Some(&1.0));
    assert_eq!(t.get(1, 0), Some(&2.0));
    assert_eq!(t.get(2, 0), Some(&3.0));
}

#[test]
fn test_transpose_single_col() {
    // [[1],        [[1, 2, 3]]
    //  [2],   ->
    //  [3]]
    let m = Matrix::from_vec(3, 1, vec![1.0, 2.0, 3.0]).unwrap();
    let t = m.transpose();

    assert_eq!(t.rows(), 1);
    assert_eq!(t.cols(), 3);
    assert_eq!(t.get(0, 0), Some(&1.0));
    assert_eq!(t.get(0, 1), Some(&2.0));
    assert_eq!(t.get(0, 2), Some(&3.0));
}

#[test]
fn test_transpose_single_element() {
    // [[5]] -> [[5]]
    let m = Matrix::from_vec(1, 1, vec![5.0]).unwrap();
    let t = m.transpose();

    assert_eq!(t.rows(), 1);
    assert_eq!(t.cols(), 1);
    assert_eq!(t.get(0, 0), Some(&5.0));
}

#[test]
fn test_transpose_identity() {
    // I^T = I
    let identity = Matrix::identity(3);
    let t = identity.transpose();

    assert_eq!(t.rows(), 3);
    assert_eq!(t.cols(), 3);

    // Check it's still identity
    for i in 0..3 {
        for j in 0..3 {
            let expected = if i == j { 1.0 } else { 0.0 };
            assert_eq!(t.get(i, j), Some(&expected));
        }
    }
}
