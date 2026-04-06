//! CSR format invariant validation.
//!
//! Implements provable contract `sparse-spmv-v1.yaml` Phase: format_validation.
//! Invariant: offsets[0] == 0 && offsets[m] == nnz && offsets monotonically increasing.

use crate::SparseError;

/// Validate CSR format invariants per provable contract.
///
/// # Contract: sparse-spmv-v1.yaml / format_validation
///
/// Checks:
/// 1. `offsets.len() == rows + 1`
/// 2. `offsets[0] == 0`
/// 3. `offsets` is monotonically non-decreasing
/// 4. `offsets[rows] == col_indices.len() == values.len()`
/// 5. All column indices are in `[0, cols)`
///
/// # Errors
///
/// Returns the first invariant violation found.
pub fn validate_csr_invariants(
    rows: usize,
    cols: usize,
    offsets: &[u32],
    col_indices: &[u32],
    values_len: usize,
) -> Result<(), SparseError> {
    // 1. Offsets length
    if offsets.len() != rows + 1 {
        return Err(SparseError::InvalidOffsetsLength {
            actual: offsets.len(),
            expected: rows + 1,
        });
    }

    // 2. First offset must be 0
    if offsets[0] != 0 {
        return Err(SparseError::NonZeroFirstOffset { value: offsets[0] });
    }

    // 3. Monotonically non-decreasing
    for i in 0..rows {
        if offsets[i] > offsets[i + 1] {
            return Err(SparseError::NonMonotonicOffsets {
                index: i,
                value: offsets[i],
                next_index: i + 1,
                next_value: offsets[i + 1],
            });
        }
    }

    // 4. Last offset matches nnz
    let nnz = col_indices.len();
    if offsets[rows] as usize != nnz {
        return Err(SparseError::OffsetNnzMismatch {
            offset_last: offsets[rows],
            nnz,
        });
    }

    // col_indices and values must match
    if nnz != values_len {
        return Err(SparseError::LengthMismatch {
            col_len: nnz,
            val_len: values_len,
        });
    }

    // 5. Column indices in bounds
    for (i, &col) in col_indices.iter().enumerate() {
        if col as usize >= cols {
            return Err(SparseError::ColumnOutOfBounds {
                col,
                cols,
                position: i,
            });
        }
    }

    Ok(())
}
