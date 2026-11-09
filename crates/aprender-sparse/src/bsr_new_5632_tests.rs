//! Coverage for `BsrMatrix::new` validation (#5632).

use crate::{BsrMatrix, SparseError};

#[test]
fn accepts_consistent_structure_and_reports_shape() {
    // 2x3 block grid, block_size 2, two stored blocks.
    let m = BsrMatrix::new(2, 3, 2, vec![0, 1, 2], vec![0, 2], vec![1.0; 8]).expect("valid");
    assert_eq!(m.rows(), 4);
    assert_eq!(m.cols(), 6);
    assert_eq!(m.block_size(), 2);
    assert_eq!(m.nnz_blocks(), 2);
}

#[test]
fn accepts_empty_matrix() {
    let m = BsrMatrix::new(1, 1, 4, vec![0, 0], vec![], vec![]).expect("empty is valid");
    assert_eq!(m.nnz_blocks(), 0);
    assert_eq!(m.rows(), 4);
}

#[test]
fn rejects_offsets_of_wrong_length() {
    let err = BsrMatrix::new(2, 2, 1, vec![0, 1], vec![0], vec![1.0]).expect_err("short offsets");
    assert!(
        matches!(
            err,
            SparseError::InvalidOffsetsLength {
                actual: 2,
                expected: 3
            }
        ),
        "{err}"
    );
    let err = BsrMatrix::new(1, 1, 1, vec![0, 0, 0], vec![], vec![]).expect_err("long offsets");
    assert!(
        matches!(
            err,
            SparseError::InvalidOffsetsLength {
                actual: 3,
                expected: 2
            }
        ),
        "{err}"
    );
}

#[test]
fn rejects_values_not_matching_block_count_times_block_area() {
    // 1 block of 2x2 needs 4 values.
    let err = BsrMatrix::new(1, 1, 2, vec![0, 1], vec![0], vec![1.0; 3]).expect_err("short");
    assert!(
        matches!(
            err,
            SparseError::LengthMismatch {
                col_len: 4,
                val_len: 3
            }
        ),
        "{err}"
    );
    let err = BsrMatrix::new(1, 1, 2, vec![0, 1], vec![0], vec![1.0; 5]).expect_err("long");
    assert!(
        matches!(
            err,
            SparseError::LengthMismatch {
                col_len: 4,
                val_len: 5
            }
        ),
        "{err}"
    );
}

#[test]
fn offsets_length_is_checked_before_values_length() {
    let err = BsrMatrix::new(2, 2, 2, vec![0], vec![0], vec![]).expect_err("both wrong");
    assert!(
        matches!(err, SparseError::InvalidOffsetsLength { .. }),
        "{err}"
    );
}

#[test]
fn new_matrix_round_trips_through_to_csr() {
    // One 2x2 block [[1,2],[3,4]] at block (0,1) of a 1x2 block grid.
    let m = BsrMatrix::new(1, 2, 2, vec![0, 1], vec![1], vec![1.0, 2.0, 3.0, 4.0]).expect("valid");
    let csr = m.to_csr().expect("csr");
    assert_eq!((csr.rows(), csr.cols()), (2, 4));
    assert_eq!(csr.nnz(), 4);
}
