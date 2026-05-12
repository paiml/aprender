// SHIP-TWO-001 — `sparse-spmv-v1` algorithm-level PARTIAL discharge
// for FALSIFY-SPARSE-001..008 (closes 8/8 sweep).
//
// Contract: `contracts/sparse-spmv-v1.yaml`.
//
// Bundles 8 verdict fns + a stand-alone CSR / SpMV / SpGEMM / COO
// reference implementation.

// ===========================================================================
// Reference CSR sparse matrix (validated constructors)
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CsrError {
    NonMonotonicOffsets,
    NonzeroFirstOffset,
    OffsetsLengthMismatch,
    ColumnOutOfBounds,
    LengthMismatch,
}

#[derive(Debug, Clone)]
pub struct Csr {
    pub n_rows: usize,
    pub n_cols: usize,
    pub offsets: Vec<usize>,
    pub cols: Vec<usize>,
    pub vals: Vec<f32>,
}

impl Csr {
    pub fn new(
        n_rows: usize,
        n_cols: usize,
        offsets: Vec<usize>,
        cols: Vec<usize>,
        vals: Vec<f32>,
    ) -> Result<Self, CsrError> {
        if offsets.len() != n_rows + 1 { return Err(CsrError::OffsetsLengthMismatch); }
        if !offsets.is_empty() && offsets[0] != 0 { return Err(CsrError::NonzeroFirstOffset); }
        for w in offsets.windows(2) {
            if w[1] < w[0] { return Err(CsrError::NonMonotonicOffsets); }
        }
        if cols.len() != vals.len() { return Err(CsrError::LengthMismatch); }
        if let Some(&last) = offsets.last() {
            if last != cols.len() { return Err(CsrError::LengthMismatch); }
        }
        for c in &cols {
            if *c >= n_cols { return Err(CsrError::ColumnOutOfBounds); }
        }
        Ok(Self { n_rows, n_cols, offsets, cols, vals })
    }
}

/// Compute `y = alpha * A * x + beta * y` where A is CSR.
pub fn spmv_axpy(a: &Csr, x: &[f32], y: &mut [f32], alpha: f32, beta: f32) -> Result<(), CsrError> {
    if x.len() != a.n_cols || y.len() != a.n_rows { return Err(CsrError::LengthMismatch); }
    for r in 0..a.n_rows {
        let mut acc = 0.0_f32;
        for k in a.offsets[r]..a.offsets[r + 1] {
            acc += a.vals[k] * x[a.cols[k]];
        }
        y[r] = alpha * acc + beta * y[r];
    }
    Ok(())
}

/// Reference dense matvec: `y = A * x`.
pub fn dense_matvec(a: &[f32], n_rows: usize, n_cols: usize, x: &[f32]) -> Vec<f32> {
    let mut y = vec![0.0_f32; n_rows];
    for r in 0..n_rows {
        for c in 0..n_cols {
            y[r] += a[r * n_cols + c] * x[c];
        }
    }
    y
}

/// SpGEMM A * B for two CSR matrices, result as dense.
pub fn spgemm_dense(a: &Csr, b: &Csr) -> Result<Vec<f32>, CsrError> {
    if a.n_cols != b.n_rows { return Err(CsrError::LengthMismatch); }
    let mut out = vec![0.0_f32; a.n_rows * b.n_cols];
    for r in 0..a.n_rows {
        for k in a.offsets[r]..a.offsets[r + 1] {
            let mid = a.cols[k];
            let v_a = a.vals[k];
            for k2 in b.offsets[mid]..b.offsets[mid + 1] {
                let col = b.cols[k2];
                out[r * b.n_cols + col] += v_a * b.vals[k2];
            }
        }
    }
    Ok(out)
}

/// COO triplets to CSR.
pub fn coo_to_csr(
    n_rows: usize,
    n_cols: usize,
    rows: &[usize],
    cols: &[usize],
    vals: &[f32],
) -> Result<Csr, CsrError> {
    if rows.len() != cols.len() || rows.len() != vals.len() { return Err(CsrError::LengthMismatch); }
    for &c in cols { if c >= n_cols { return Err(CsrError::ColumnOutOfBounds); } }
    for &r in rows { if r >= n_rows { return Err(CsrError::ColumnOutOfBounds); } }
    let mut row_counts = vec![0_usize; n_rows];
    for &r in rows { row_counts[r] += 1; }
    let mut offsets = vec![0_usize; n_rows + 1];
    for i in 0..n_rows { offsets[i + 1] = offsets[i] + row_counts[i]; }
    let mut next = offsets.clone();
    let mut out_cols = vec![0_usize; rows.len()];
    let mut out_vals = vec![0.0_f32; rows.len()];
    for k in 0..rows.len() {
        let r = rows[k];
        let pos = next[r];
        out_cols[pos] = cols[k];
        out_vals[pos] = vals[k];
        next[r] += 1;
    }
    Csr::new(n_rows, n_cols, offsets, out_cols, out_vals)
}

// ===========================================================================
// SPARSE-001 — CSR rejects non-monotonic offsets
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sparse001Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_reject_non_monotonic_offsets() -> Sparse001Verdict {
    let res = Csr::new(2, 3, vec![0, 2, 1], vec![0, 1], vec![1.0, 2.0]);
    if matches!(res, Err(CsrError::NonMonotonicOffsets)) { Sparse001Verdict::Pass } else { Sparse001Verdict::Fail }
}

// ===========================================================================
// SPARSE-002 — CSR rejects nonzero first offset
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sparse002Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_reject_nonzero_first_offset() -> Sparse002Verdict {
    let res = Csr::new(2, 3, vec![1, 2, 3], vec![0, 1, 2], vec![1.0, 2.0, 3.0]);
    if matches!(res, Err(CsrError::NonzeroFirstOffset)) { Sparse002Verdict::Pass } else { Sparse002Verdict::Fail }
}

// ===========================================================================
// SPARSE-003 — CSR rejects column out of bounds
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sparse003Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_reject_column_out_of_bounds() -> Sparse003Verdict {
    let res = Csr::new(2, 3, vec![0, 1, 2], vec![0, 5], vec![1.0, 2.0]);
    if matches!(res, Err(CsrError::ColumnOutOfBounds)) { Sparse003Verdict::Pass } else { Sparse003Verdict::Fail }
}

// ===========================================================================
// SPARSE-004 — SpMV identity matrix matches dense
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sparse004Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_spmv_identity() -> Sparse004Verdict {
    let n = 4;
    let offsets = (0..=n).collect::<Vec<_>>();
    let cols = (0..n).collect::<Vec<_>>();
    let vals = vec![1.0_f32; n];
    let id = match Csr::new(n, n, offsets, cols, vals) { Ok(c) => c, Err(_) => return Sparse004Verdict::Fail };
    let x = vec![3.0_f32, 5.0, 7.0, 11.0];
    let mut y = vec![0.0_f32; n];
    if spmv_axpy(&id, &x, &mut y, 1.0, 0.0).is_err() { return Sparse004Verdict::Fail; }
    if y == x { Sparse004Verdict::Pass } else { Sparse004Verdict::Fail }
}

// ===========================================================================
// SPARSE-005 — SpMV alpha/beta correctness
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sparse005Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_spmv_alpha_beta() -> Sparse005Verdict {
    // 2x2 identity: y = 2*A*x + 3*y_init = 2*x + 3*y_init.
    let id = match Csr::new(2, 2, vec![0, 1, 2], vec![0, 1], vec![1.0, 1.0]) {
        Ok(c) => c, Err(_) => return Sparse005Verdict::Fail,
    };
    let x = [4.0_f32, 7.0];
    let mut y = [10.0_f32, 100.0];
    if spmv_axpy(&id, &x, &mut y, 2.0, 3.0).is_err() { return Sparse005Verdict::Fail; }
    let expected = [2.0 * 4.0 + 3.0 * 10.0, 2.0 * 7.0 + 3.0 * 100.0];
    if y == expected { Sparse005Verdict::Pass } else { Sparse005Verdict::Fail }
}

// ===========================================================================
// SPARSE-006 — SpGEMM identity * identity == identity
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sparse006Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_spgemm_identity() -> Sparse006Verdict {
    let n = 3;
    let make_id = || Csr::new(n, n, (0..=n).collect(), (0..n).collect(), vec![1.0; n]).expect("csr matrix valid");
    let id1 = make_id();
    let id2 = make_id();
    let prod = match spgemm_dense(&id1, &id2) { Ok(d) => d, Err(_) => return Sparse006Verdict::Fail };
    let mut expected = vec![0.0_f32; n * n];
    for i in 0..n { expected[i * n + i] = 1.0; }
    if prod == expected { Sparse006Verdict::Pass } else { Sparse006Verdict::Fail }
}

// ===========================================================================
// SPARSE-007 — COO→CSR roundtrip preserves all entries
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sparse007Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_coo_csr_roundtrip() -> Sparse007Verdict {
    let rows = [0, 0, 1, 2, 2, 2];
    let cols = [0, 2, 1, 0, 1, 2];
    let vals = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let csr = match coo_to_csr(3, 3, &rows, &cols, &vals) {
        Ok(c) => c, Err(_) => return Sparse007Verdict::Fail,
    };
    // Densify and check element preservation.
    let mut dense = vec![0.0_f32; 3 * 3];
    for r in 0..csr.n_rows {
        for k in csr.offsets[r]..csr.offsets[r + 1] {
            dense[r * csr.n_cols + csr.cols[k]] = csr.vals[k];
        }
    }
    let mut expected = vec![0.0_f32; 3 * 3];
    for k in 0..rows.len() { expected[rows[k] * 3 + cols[k]] = vals[k]; }
    if dense == expected { Sparse007Verdict::Pass } else { Sparse007Verdict::Fail }
}

// ===========================================================================
// SPARSE-008 — SpMV linearity: SpMV matches dense for arbitrary input
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sparse008Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_spmv_matches_dense() -> Sparse008Verdict {
    // 3x4 dense:
    //   [1 0 2 0]
    //   [0 3 0 4]
    //   [5 0 0 6]
    let dense = vec![
        1.0_f32, 0.0, 2.0, 0.0,
        0.0,     3.0, 0.0, 4.0,
        5.0,     0.0, 0.0, 6.0,
    ];
    let csr = match Csr::new(
        3, 4,
        vec![0, 2, 4, 6],
        vec![0, 2, 1, 3, 0, 3],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    ) { Ok(c) => c, Err(_) => return Sparse008Verdict::Fail };
    let x = vec![2.0_f32, -1.0, 0.5, 4.0];
    let mut y = vec![0.0_f32; 3];
    if spmv_axpy(&csr, &x, &mut y, 1.0, 0.0).is_err() { return Sparse008Verdict::Fail; }
    let y_dense = dense_matvec(&dense, 3, 4, &x);
    for (a, b) in y.iter().zip(y_dense.iter()) {
        if (a - b).abs() > 1e-6 { return Sparse008Verdict::Fail; }
    }
    Sparse008Verdict::Pass
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test] fn sparse_001_pass() { assert_eq!(verdict_from_reject_non_monotonic_offsets(), Sparse001Verdict::Pass); }
    #[test] fn sparse_002_pass() { assert_eq!(verdict_from_reject_nonzero_first_offset(), Sparse002Verdict::Pass); }
    #[test] fn sparse_003_pass() { assert_eq!(verdict_from_reject_column_out_of_bounds(), Sparse003Verdict::Pass); }
    #[test] fn sparse_004_pass() { assert_eq!(verdict_from_spmv_identity(), Sparse004Verdict::Pass); }
    #[test] fn sparse_005_pass() { assert_eq!(verdict_from_spmv_alpha_beta(), Sparse005Verdict::Pass); }
    #[test] fn sparse_006_pass() { assert_eq!(verdict_from_spgemm_identity(), Sparse006Verdict::Pass); }
    #[test] fn sparse_007_pass() { assert_eq!(verdict_from_coo_csr_roundtrip(), Sparse007Verdict::Pass); }
    #[test] fn sparse_008_pass() { assert_eq!(verdict_from_spmv_matches_dense(), Sparse008Verdict::Pass); }

    // Reference impl spot checks (negative paths that the verdicts encode).

    #[test] fn ref_csr_accepts_valid() {
        let c = Csr::new(2, 3, vec![0, 1, 2], vec![0, 2], vec![1.0, 2.0]);
        assert!(c.is_ok());
    }

    #[test] fn ref_csr_rejects_offsets_length_mismatch() {
        let c = Csr::new(2, 3, vec![0, 1], vec![0], vec![1.0]);
        assert!(matches!(c, Err(CsrError::OffsetsLengthMismatch)));
    }

    #[test] fn ref_csr_rejects_cols_vals_length_mismatch() {
        let c = Csr::new(1, 3, vec![0, 2], vec![0, 1], vec![1.0]);
        assert!(matches!(c, Err(CsrError::LengthMismatch)));
    }

    #[test] fn ref_spmv_dim_mismatch() {
        let id = Csr::new(2, 2, vec![0, 1, 2], vec![0, 1], vec![1.0, 1.0]).expect("csr matrix valid");
        let x = [1.0_f32, 2.0, 3.0]; // wrong length
        let mut y = [0.0_f32; 2];
        assert!(spmv_axpy(&id, &x, &mut y, 1.0, 0.0).is_err());
    }

    #[test] fn ref_dense_matvec_basic() {
        let a = [1.0_f32, 2.0, 3.0, 4.0]; // 2x2: [[1,2],[3,4]]
        let x = [5.0_f32, 6.0];
        let y = dense_matvec(&a, 2, 2, &x);
        assert_eq!(y, vec![1.0 * 5.0 + 2.0 * 6.0, 3.0 * 5.0 + 4.0 * 6.0]);
    }

    #[test] fn ref_spgemm_id_times_dense_via_csr() {
        // 2x2 identity × 2x2 [[2, 3], [4, 5]] (encoded as CSR).
        let id = Csr::new(2, 2, vec![0, 1, 2], vec![0, 1], vec![1.0, 1.0]).expect("csr matrix valid");
        let b = Csr::new(2, 2, vec![0, 2, 4], vec![0, 1, 0, 1], vec![2.0, 3.0, 4.0, 5.0]).expect("csr matrix valid");
        let prod = spgemm_dense(&id, &b).expect("csr matrix valid");
        assert_eq!(prod, vec![2.0, 3.0, 4.0, 5.0]);
    }

    #[test] fn ref_coo_to_csr_empty() {
        let csr = coo_to_csr(2, 2, &[], &[], &[]).expect("csr matrix valid");
        assert_eq!(csr.offsets, vec![0, 0, 0]);
        assert!(csr.cols.is_empty());
        assert!(csr.vals.is_empty());
    }
}
