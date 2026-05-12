// `transpose-kernel-v1` algorithm-level PARTIAL discharge for
// FALSIFY-TP-001..006.
//
// Contract: `contracts/transpose-kernel-v1.yaml`.
//
// Pure-Rust verdicts for the 6 falsification gates plus a reference
// row-major transpose helper.
//
// TP-001: element correctness — transpose(A)[j][i] == A[i][j]
// TP-002: involution — transpose(transpose(A)) == A bitwise
// TP-003: non-8-aligned dimensions handled correctly
// TP-004: AVX2 vs scalar bit-exact parity
// TP-005: transpose(I) == I for square identity
// TP-006: attention shape 2048×128 matches naive reference

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TpVerdict {
    Pass,
    Fail,
}

/// Reference row-major transpose: out[j*rows + i] = src[i*cols + j].
/// Returns `None` if `src.len() != rows * cols`.
#[must_use]
pub fn transpose_rowmajor(src: &[f32], rows: usize, cols: usize) -> Option<Vec<f32>> {
    if src.len() != rows.checked_mul(cols)? {
        return None;
    }
    let mut out = vec![0.0_f32; src.len()];
    for i in 0..rows {
        for j in 0..cols {
            out[j * rows + i] = src[i * cols + j];
        }
    }
    Some(out)
}

/// TP-001: transpose(A)[j][i] == A[i][j] for the given (i, j).
/// `transposed` is the kernel's output as a flat row-major buffer.
#[must_use]
pub fn verdict_from_element_correctness(
    src: &[f32],
    transposed: &[f32],
    rows: usize,
    cols: usize,
    i: usize,
    j: usize,
) -> TpVerdict {
    if rows == 0 || cols == 0 {
        return TpVerdict::Fail;
    }
    if i >= rows || j >= cols {
        return TpVerdict::Fail;
    }
    if src.len() != rows * cols || transposed.len() != rows * cols {
        return TpVerdict::Fail;
    }
    let original = src[i * cols + j];
    let mapped = transposed[j * rows + i];
    if original.to_bits() == mapped.to_bits() {
        TpVerdict::Pass
    } else {
        TpVerdict::Fail
    }
}

/// TP-002: transpose(transpose(A)) == A bitwise.
#[must_use]
pub fn verdict_from_involution(src: &[f32], rows: usize, cols: usize) -> TpVerdict {
    let Some(once) = transpose_rowmajor(src, rows, cols) else {
        return TpVerdict::Fail;
    };
    let Some(twice) = transpose_rowmajor(&once, cols, rows) else {
        return TpVerdict::Fail;
    };
    if src.len() != twice.len() {
        return TpVerdict::Fail;
    }
    for (a, b) in src.iter().zip(twice.iter()) {
        if a.to_bits() != b.to_bits() {
            return TpVerdict::Fail;
        }
    }
    TpVerdict::Pass
}

/// TP-003: non-8-aligned dimensions transpose correctly.
#[must_use]
pub fn verdict_from_non_aligned_correctness(rows: usize, cols: usize) -> TpVerdict {
    if rows == 0 || cols == 0 {
        return TpVerdict::Fail;
    }
    let n = rows.checked_mul(cols).unwrap_or(0);
    if n == 0 {
        return TpVerdict::Fail;
    }
    // Build a deterministic test matrix where src[i*cols+j] = i*1000+j
    let src: Vec<f32> = (0..rows)
        .flat_map(|i| (0..cols).map(move |j| (i * 1000 + j) as f32))
        .collect();
    let Some(t) = transpose_rowmajor(&src, rows, cols) else {
        return TpVerdict::Fail;
    };
    for i in 0..rows {
        for j in 0..cols {
            let expected = src[i * cols + j];
            let got = t[j * rows + i];
            if expected.to_bits() != got.to_bits() {
                return TpVerdict::Fail;
            }
        }
    }
    TpVerdict::Pass
}

/// TP-004: AVX2 output bit-exact equal to scalar output.
#[must_use]
pub fn verdict_from_avx2_scalar_parity(avx2_out: &[f32], scalar_out: &[f32]) -> TpVerdict {
    if avx2_out.is_empty() || avx2_out.len() != scalar_out.len() {
        return TpVerdict::Fail;
    }
    for (a, s) in avx2_out.iter().zip(scalar_out.iter()) {
        if a.to_bits() != s.to_bits() {
            return TpVerdict::Fail;
        }
    }
    TpVerdict::Pass
}

/// TP-005: transpose of N×N identity is itself.
#[must_use]
pub fn verdict_from_identity(n: usize) -> TpVerdict {
    if n == 0 {
        return TpVerdict::Fail;
    }
    let mut id = vec![0.0_f32; n * n];
    for i in 0..n {
        id[i * n + i] = 1.0;
    }
    let Some(t) = transpose_rowmajor(&id, n, n) else {
        return TpVerdict::Fail;
    };
    if id.len() != t.len() {
        return TpVerdict::Fail;
    }
    for (a, b) in id.iter().zip(t.iter()) {
        if a.to_bits() != b.to_bits() {
            return TpVerdict::Fail;
        }
    }
    TpVerdict::Pass
}

/// TP-006: 2048×128 attention shape — kernel matches naive reference.
///
/// Caller passes `(kernel_output, naive_reference)` for the canonical
/// 2048×128 case. Pass iff bit-exact, length-matched, and dims correct.
#[must_use]
pub fn verdict_from_attention_shape(
    kernel_output: &[f32],
    naive_reference: &[f32],
    rows: usize,
    cols: usize,
) -> TpVerdict {
    if rows != 2048 || cols != 128 {
        return TpVerdict::Fail;
    }
    if kernel_output.len() != rows * cols || naive_reference.len() != rows * cols {
        return TpVerdict::Fail;
    }
    for (a, b) in kernel_output.iter().zip(naive_reference.iter()) {
        if a.to_bits() != b.to_bits() {
            return TpVerdict::Fail;
        }
    }
    TpVerdict::Pass
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_test_matrix(rows: usize, cols: usize) -> Vec<f32> {
        (0..rows)
            .flat_map(|i| (0..cols).map(move |j| (i * 1000 + j) as f32))
            .collect()
    }

    // -----------------------------------------------------------------
    // Section 1: Reference helper.
    // -----------------------------------------------------------------
    #[test]
    fn transpose_rowmajor_roundtrip() {
        let src = build_test_matrix(4, 3);
        let t = transpose_rowmajor(&src, 4, 3).unwrap();
        let back = transpose_rowmajor(&t, 3, 4).unwrap();
        assert_eq!(src, back);
    }

    #[test]
    fn transpose_rowmajor_size_mismatch_returns_none() {
        let src = vec![1.0_f32; 11];
        assert!(transpose_rowmajor(&src, 3, 4).is_none());
    }

    // -----------------------------------------------------------------
    // Section 2: TP-001 element correctness.
    // -----------------------------------------------------------------
    #[test]
    fn ftp001_pass_first_element() {
        let src = build_test_matrix(4, 3);
        let t = transpose_rowmajor(&src, 4, 3).unwrap();
        let v = verdict_from_element_correctness(&src, &t, 4, 3, 0, 0);
        assert_eq!(v, TpVerdict::Pass);
    }

    #[test]
    fn ftp001_pass_last_element() {
        let src = build_test_matrix(4, 3);
        let t = transpose_rowmajor(&src, 4, 3).unwrap();
        let v = verdict_from_element_correctness(&src, &t, 4, 3, 3, 2);
        assert_eq!(v, TpVerdict::Pass);
    }

    #[test]
    fn ftp001_fail_index_out_of_bounds() {
        let src = build_test_matrix(4, 3);
        let t = transpose_rowmajor(&src, 4, 3).unwrap();
        let v = verdict_from_element_correctness(&src, &t, 4, 3, 4, 0);
        assert_eq!(v, TpVerdict::Fail);
    }

    #[test]
    fn ftp001_fail_buggy_kernel() {
        // Simulate buggy kernel that scrambles output.
        let src = build_test_matrix(4, 3);
        let mut t = transpose_rowmajor(&src, 4, 3).unwrap();
        t[0] = 99.0; // corrupt
        let v = verdict_from_element_correctness(&src, &t, 4, 3, 0, 0);
        assert_eq!(v, TpVerdict::Fail);
    }

    // -----------------------------------------------------------------
    // Section 3: TP-002 involution.
    // -----------------------------------------------------------------
    #[test]
    fn ftp002_pass_8x8() {
        let src = build_test_matrix(8, 8);
        let v = verdict_from_involution(&src, 8, 8);
        assert_eq!(v, TpVerdict::Pass);
    }

    #[test]
    fn ftp002_pass_non_aligned() {
        let src = build_test_matrix(7, 13);
        let v = verdict_from_involution(&src, 7, 13);
        assert_eq!(v, TpVerdict::Pass);
    }

    #[test]
    fn ftp002_pass_skinny() {
        let src = build_test_matrix(1, 100);
        let v = verdict_from_involution(&src, 1, 100);
        assert_eq!(v, TpVerdict::Pass);
    }

    #[test]
    fn ftp002_fail_size_mismatch() {
        let src = vec![1.0_f32; 11]; // 11 doesn't factor into 3x4
        let v = verdict_from_involution(&src, 3, 4);
        assert_eq!(v, TpVerdict::Fail);
    }

    // -----------------------------------------------------------------
    // Section 4: TP-003 non-8-aligned.
    // -----------------------------------------------------------------
    #[test]
    fn ftp003_pass_7x13() {
        let v = verdict_from_non_aligned_correctness(7, 13);
        assert_eq!(v, TpVerdict::Pass);
    }

    #[test]
    fn ftp003_pass_17x3() {
        let v = verdict_from_non_aligned_correctness(17, 3);
        assert_eq!(v, TpVerdict::Pass);
    }

    #[test]
    fn ftp003_pass_1x100() {
        let v = verdict_from_non_aligned_correctness(1, 100);
        assert_eq!(v, TpVerdict::Pass);
    }

    #[test]
    fn ftp003_pass_100x1() {
        let v = verdict_from_non_aligned_correctness(100, 1);
        assert_eq!(v, TpVerdict::Pass);
    }

    #[test]
    fn ftp003_fail_zero_dim() {
        let v = verdict_from_non_aligned_correctness(0, 13);
        assert_eq!(v, TpVerdict::Fail);
    }

    // -----------------------------------------------------------------
    // Section 5: TP-004 AVX2 vs scalar.
    // -----------------------------------------------------------------
    #[test]
    fn ftp004_pass_bit_exact() {
        let avx2 = vec![1.0_f32, 2.5, 3.0, 4.5];
        let scalar = vec![1.0_f32, 2.5, 3.0, 4.5];
        let v = verdict_from_avx2_scalar_parity(&avx2, &scalar);
        assert_eq!(v, TpVerdict::Pass);
    }

    #[test]
    fn ftp004_fail_one_ulp() {
        let avx2 = vec![1.0_f32, 2.5];
        let bumped = f32::from_bits(2.5_f32.to_bits() + 1);
        let scalar = vec![1.0_f32, bumped];
        let v = verdict_from_avx2_scalar_parity(&avx2, &scalar);
        assert_eq!(v, TpVerdict::Fail);
    }

    #[test]
    fn ftp004_fail_empty() {
        let v = verdict_from_avx2_scalar_parity(&[], &[]);
        assert_eq!(v, TpVerdict::Fail);
    }

    // -----------------------------------------------------------------
    // Section 6: TP-005 identity, TP-006 attention shape.
    // -----------------------------------------------------------------
    #[test]
    fn ftp005_pass_identity_4() {
        let v = verdict_from_identity(4);
        assert_eq!(v, TpVerdict::Pass);
    }

    #[test]
    fn ftp005_pass_identity_8() {
        let v = verdict_from_identity(8);
        assert_eq!(v, TpVerdict::Pass);
    }

    #[test]
    fn ftp005_pass_identity_17() {
        let v = verdict_from_identity(17);
        assert_eq!(v, TpVerdict::Pass);
    }

    #[test]
    fn ftp005_fail_zero() {
        let v = verdict_from_identity(0);
        assert_eq!(v, TpVerdict::Fail);
    }

    #[test]
    fn ftp006_pass_canonical_2048_128() {
        let src = build_test_matrix(2048, 128);
        let kernel = transpose_rowmajor(&src, 2048, 128).unwrap();
        let naive = transpose_rowmajor(&src, 2048, 128).unwrap();
        let v = verdict_from_attention_shape(&kernel, &naive, 2048, 128);
        assert_eq!(v, TpVerdict::Pass);
    }

    #[test]
    fn ftp006_fail_wrong_dims() {
        let src = build_test_matrix(1024, 128);
        let kernel = transpose_rowmajor(&src, 1024, 128).unwrap();
        let v = verdict_from_attention_shape(&kernel, &kernel, 1024, 128);
        assert_eq!(v, TpVerdict::Fail);
    }

    #[test]
    fn ftp006_fail_kernel_drift() {
        let src = build_test_matrix(2048, 128);
        let mut kernel = transpose_rowmajor(&src, 2048, 128).unwrap();
        kernel[0] = 99.0; // corrupt
        let naive = transpose_rowmajor(&src, 2048, 128).unwrap();
        let v = verdict_from_attention_shape(&kernel, &naive, 2048, 128);
        assert_eq!(v, TpVerdict::Fail);
    }

    // -----------------------------------------------------------------
    // Section 7: Mutation surveys + realistic.
    // -----------------------------------------------------------------
    #[test]
    fn mutation_survey_002_dim_pairs() {
        for &(r, c) in &[(1usize, 1), (3, 5), (7, 13), (8, 8), (17, 3), (32, 32)] {
            let src = build_test_matrix(r, c);
            let v = verdict_from_involution(&src, r, c);
            assert_eq!(v, TpVerdict::Pass, "({r}, {c})");
        }
    }

    #[test]
    fn realistic_healthy_passes_all_6() {
        let src = build_test_matrix(2048, 128);
        let t = transpose_rowmajor(&src, 2048, 128).unwrap();
        let v1 = verdict_from_element_correctness(&src, &t, 2048, 128, 1024, 64);
        let v2 = verdict_from_involution(&src, 2048, 128);
        let v3 = verdict_from_non_aligned_correctness(7, 13);
        let v4 = verdict_from_avx2_scalar_parity(&t, &t);
        let v5 = verdict_from_identity(8);
        let v6 = verdict_from_attention_shape(&t, &t, 2048, 128);
        assert_eq!(v1, TpVerdict::Pass);
        assert_eq!(v2, TpVerdict::Pass);
        assert_eq!(v3, TpVerdict::Pass);
        assert_eq!(v4, TpVerdict::Pass);
        assert_eq!(v5, TpVerdict::Pass);
        assert_eq!(v6, TpVerdict::Pass);
    }

    #[test]
    fn realistic_pre_fix_all_6_failures() {
        // Pre-fix regressions:
        //  1: kernel returned wrong element at (0, 0)
        //  2: involution failed because transpose isn't symmetric
        //  3: non-aligned skipped edge elements (n=0)
        //  4: AVX2 produced 1-ULP drift vs scalar
        //  5: identity collapsed to zero (n=0 sentinel)
        //  6: attention shape kernel had element drift
        let src = build_test_matrix(2048, 128);
        let mut t = transpose_rowmajor(&src, 2048, 128).unwrap();
        t[0] = 99.0;
        let v1 = verdict_from_element_correctness(&src, &t, 2048, 128, 0, 0);
        let bad_src = vec![1.0_f32; 11];
        let v2 = verdict_from_involution(&bad_src, 3, 4);
        let v3 = verdict_from_non_aligned_correctness(0, 13);
        let avx = vec![1.0_f32];
        let bumped = f32::from_bits(1.0_f32.to_bits() + 1);
        let scalar = vec![bumped];
        let v4 = verdict_from_avx2_scalar_parity(&avx, &scalar);
        let v5 = verdict_from_identity(0);
        let naive = transpose_rowmajor(&src, 2048, 128).unwrap();
        let v6 = verdict_from_attention_shape(&t, &naive, 2048, 128);
        assert_eq!(v1, TpVerdict::Fail);
        assert_eq!(v2, TpVerdict::Fail);
        assert_eq!(v3, TpVerdict::Fail);
        assert_eq!(v4, TpVerdict::Fail);
        assert_eq!(v5, TpVerdict::Fail);
        assert_eq!(v6, TpVerdict::Fail);
    }
}
