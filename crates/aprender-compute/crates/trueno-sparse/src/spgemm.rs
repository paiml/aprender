//! Sparse General Matrix-Matrix Multiply (SpGEMM).
//!
//! # Contract: sparse-spgemm-v1.yaml
//!
//! Computes C = A * B where A and B are sparse CSR matrices.
//! Uses Gustavson's algorithm (row-by-row with hash accumulator).
//!
//! ## Proof obligations
//! - Associativity: (AB)C = A(BC) within tolerance
//! - Identity: AI = A, IA = A
//! - Zero: A×0 = 0

use crate::csr::CsrMatrix;
use crate::error::SparseError;

/// Sparse matrix-matrix multiply: C = A * B (both CSR).
///
/// Uses Gustavson's algorithm: for each row of A, scatter-gather
/// into a dense workspace, then compress into CSR.
///
/// # Errors
///
/// Returns error if A.cols() != B.rows().
pub fn spgemm(a: &CsrMatrix<f32>, b: &CsrMatrix<f32>) -> Result<CsrMatrix<f32>, SparseError> {
    if a.cols() != b.rows() {
        return Err(SparseError::SpMVDimensionMismatch { matrix_cols: a.cols(), x_len: b.rows() });
    }

    let m = a.rows();
    let n = b.cols();

    let mut c_offsets = Vec::with_capacity(m + 1);
    let mut c_col_indices = Vec::new();
    let mut c_values = Vec::new();

    // Dense workspace for accumulating one row of C
    let mut work = vec![0.0_f32; n];
    let mut marker = vec![false; n];
    let mut col_list = Vec::new();

    c_offsets.push(0u32);

    for i in 0..m {
        accumulate_row(a, b, i, &mut work, &mut marker, &mut col_list);
        emit_row(
            &mut c_col_indices,
            &mut c_values,
            &mut c_offsets,
            &mut work,
            &mut marker,
            &mut col_list,
        );
    }

    CsrMatrix::new(m, n, c_offsets, c_col_indices, c_values)
}

/// Accumulate row i of C = A * B into workspace.
fn accumulate_row(
    a: &CsrMatrix<f32>,
    b: &CsrMatrix<f32>,
    i: usize,
    work: &mut [f32],
    marker: &mut [bool],
    col_list: &mut Vec<usize>,
) {
    let a_off = a.offsets();
    let a_cols = a.col_indices();
    let a_vals = a.values();
    let b_off = b.offsets();
    let b_cols = b.col_indices();
    let b_vals = b.values();

    let a_start = a_off[i] as usize;
    let a_end = a_off[i + 1] as usize;

    for a_idx in a_start..a_end {
        let k = a_cols[a_idx] as usize;
        let a_val = a_vals[a_idx];

        let b_start = b_off[k] as usize;
        let b_end = b_off[k + 1] as usize;

        for b_idx in b_start..b_end {
            let j = b_cols[b_idx] as usize;
            if !marker[j] {
                marker[j] = true;
                col_list.push(j);
            }
            work[j] += a_val * b_vals[b_idx];
        }
    }
}

/// Emit accumulated row into CSR arrays and reset workspace.
fn emit_row(
    col_indices: &mut Vec<u32>,
    values: &mut Vec<f32>,
    offsets: &mut Vec<u32>,
    work: &mut [f32],
    marker: &mut [bool],
    col_list: &mut Vec<usize>,
) {
    col_list.sort_unstable();

    for &j in col_list.iter() {
        let val = work[j];
        if val.abs() > f32::EPSILON {
            col_indices.push(j as u32);
            values.push(val);
        }
    }

    // Reset workspace for next row
    for &j in col_list.iter() {
        work[j] = 0.0;
        marker[j] = false;
    }
    col_list.clear();

    let nnz = col_indices.len() as u32;
    offsets.push(nnz);
}
