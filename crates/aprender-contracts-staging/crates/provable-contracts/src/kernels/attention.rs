//! Scaled dot-product attention kernel.
//!
//! Matches `attention-kernel-v1.yaml`.
//! Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V
//!
//! Each function provides one of three backends:
//! - `fn attention_scalar(...)` -- Pure Rust scalar reference (ground truth)
//! - `unsafe fn attention_avx2(...)` -- AVX2 SIMD implementation
//! - `fn attention_ptx() -> &'static str` -- PTX assembly source string

use super::ops;

// ────────────────────────────────────────────────────────────────────────────
// Scalar implementation
// ────────────────────────────────────────────────────────────────────────────

/// Scaled dot-product attention (scalar reference).
///
/// Q is n x d_k, K is m x d_k, V is m x d_v, output is n x d_v.
///
/// Step 1: scores = Q * K^T / sqrt(d_k)  -- n x m matrix
/// Step 2: softmax each row of scores
/// Step 3: output = scores * V            -- n x d_v matrix
///
/// # Panics
/// Panics if dimensions do not match expected sizes.
pub fn attention_scalar(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    n: usize,
    m: usize,
    d_k: usize,
    d_v: usize,
    output: &mut [f32],
) {
    assert_eq!(
        q.len(),
        n * d_k,
        "Q dimension mismatch: expected {} got {}",
        n * d_k,
        q.len()
    );
    assert_eq!(
        k.len(),
        m * d_k,
        "K dimension mismatch: expected {} got {}",
        m * d_k,
        k.len()
    );
    assert_eq!(
        v.len(),
        m * d_v,
        "V dimension mismatch: expected {} got {}",
        m * d_v,
        v.len()
    );
    assert_eq!(
        output.len(),
        n * d_v,
        "output dimension mismatch: expected {} got {}",
        n * d_v,
        output.len()
    );

    // Step 1: Compute scores = Q * K^T / sqrt(d_k), shape n x m
    let mut scores = vec![0.0f32; n * m];
    ops::score_matrix(q, k, n, m, d_k, &mut scores);

    // Step 2: Softmax each row
    ops::softmax_rows(&mut scores, n, m);

    // Step 3: output = scores * V, shape n x d_v
    ops::matmul_sv(&scores, v, n, m, d_v, output);
}

// ────────────────────────────────────────────────────────────────────────────
// AVX2 implementation
// ────────────────────────────────────────────────────────────────────────────

/// AVX2 scaled dot-product attention -- delegates to scalar.
///
/// Attention is a composition of matmul and softmax; the scalar implementation
/// is already efficient for the composed operation.
///
/// # Safety
/// Requires AVX2 support. Caller must verify with `is_x86_feature_detected!("avx2")`.
///
/// # Panics
/// Panics if dimensions do not match expected sizes.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn attention_avx2(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    n: usize,
    m: usize,
    d_k: usize,
    d_v: usize,
    output: &mut [f32],
) {
    attention_scalar(q, k, v, n, m, d_k, d_v, output);
}

include!("attention_ptx.rs");

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::super::ops::sequential_floats;
    use super::super::ulp::assert_ulp_eq;
    use super::*;
    use proptest::prelude::*;

    // ── Single query, single key ────────────────────────────────────────

    #[test]
    fn test_attention_single_query_single_key() {
        // n=1 query, m=1 key: softmax of single score = 1.0, output = V
        let d_k = 4;
        let d_v = 3;
        let q = vec![1.0, 0.0, 1.0, 0.0];
        let k = vec![1.0, 0.0, 1.0, 0.0];
        let v = vec![2.0, 3.0, 4.0];
        let mut output = vec![0.0f32; d_v];

        attention_scalar(&q, &k, &v, 1, 1, d_k, d_v, &mut output);

        // softmax of a single element = 1.0, so output = 1.0 * V
        assert_ulp_eq(&output, &v, 0);
    }

    // ── Uniform attention ───────────────────────────────────────────────

    #[test]
    fn test_attention_uniform_scores() {
        // When all scores are equal, softmax gives uniform weights = 1/m.
        // Output should be the mean of V rows.
        let n = 1;
        let m = 3;
        let d_k = 2;
        let d_v = 2;

        // Q and K arranged so all dot products are equal
        let q = vec![1.0, 0.0];
        let k = vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0]; // all same K row
        let v = vec![3.0, 6.0, 6.0, 9.0, 9.0, 12.0]; // V rows
        let mut output = vec![0.0f32; d_v];

        attention_scalar(&q, &k, &v, n, m, d_k, d_v, &mut output);

        // Mean of V rows: [(3+6+9)/3, (6+9+12)/3] = [6.0, 9.0]
        let expected = [6.0, 9.0];
        for (a, b) in output.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-5, "expected ~{b}, got {a}");
        }
    }

    // ── Known 2-query, 2-key attention ──────────────────────────────────

    #[test]
    fn test_attention_two_queries_two_keys() {
        let n = 2;
        let m = 2;
        let d_k = 2;
        let d_v = 2;

        // Q = [[1,0],[0,1]], K = [[1,0],[0,1]]
        // QK^T = [[1,0],[0,1]] (identity before scaling)
        // scale = 1/sqrt(2)
        // scores = [[1/sqrt(2), 0], [0, 1/sqrt(2)]]
        // After softmax: dominant weight on diagonal
        let q = vec![1.0, 0.0, 0.0, 1.0];
        let k = vec![1.0, 0.0, 0.0, 1.0];
        let v = vec![10.0, 20.0, 30.0, 40.0];
        let mut output = vec![0.0f32; n * d_v];

        attention_scalar(&q, &k, &v, n, m, d_k, d_v, &mut output);

        // First query attends more to first key, second to second key
        // Exact values depend on softmax but first row should be closer to [10,20]
        assert!(
            output[0] < 20.0,
            "first query, first dim should lean toward V[0]"
        );
        assert!(
            output[2] > 20.0,
            "second query, first dim should lean toward V[1]"
        );
    }

    // ── Dimension assertions ────────────────────────────────────────────

    #[test]
    #[should_panic(expected = "Q dimension mismatch")]
    fn test_attention_bad_q_dim() {
        let mut output = vec![0.0f32; 2];
        attention_scalar(&[1.0], &[1.0, 2.0], &[1.0, 2.0], 1, 1, 2, 2, &mut output);
    }

    #[test]
    #[should_panic(expected = "K dimension mismatch")]
    fn test_attention_bad_k_dim() {
        let mut output = vec![0.0f32; 2];
        attention_scalar(&[1.0, 2.0], &[1.0], &[1.0, 2.0], 1, 1, 2, 2, &mut output);
    }

    #[test]
    #[should_panic(expected = "V dimension mismatch")]
    fn test_attention_bad_v_dim() {
        let mut output = vec![0.0f32; 2];
        attention_scalar(&[1.0, 2.0], &[1.0, 2.0], &[1.0], 1, 1, 2, 2, &mut output);
    }

    // ── Property-based tests ────────────────────────────────────────────

    proptest! {
        #[test]
        fn prop_attention_output_bounded(
            n in 1usize..4,
            m in 1usize..4,
            d_k in 1usize..4,
            d_v in 1usize..4,
        ) {
            let q = sequential_floats(n*d_k, 0.1);
            let k = sequential_floats(m*d_k, 0.1);
            let v = sequential_floats(m*d_v, 0.1);
            let mut output = vec![0.0f32; n * d_v];

            attention_scalar(&q, &k, &v, n, m, d_k, d_v, &mut output);

            // Output is convex combination of V rows, so each output element
            // must be between min and max of corresponding V column
            for j in 0..d_v {
                let v_col_min = (0..m).map(|r| v[r * d_v + j]).fold(f32::INFINITY, f32::min);
                let v_col_max = (0..m).map(|r| v[r * d_v + j]).fold(f32::NEG_INFINITY, f32::max);
                for i in 0..n {
                    let val = output[i * d_v + j];
                    prop_assert!(
                        val >= v_col_min - 1e-5 && val <= v_col_max + 1e-5,
                        "output[{i},{j}] = {val} not in V column range [{v_col_min}, {v_col_max}]"
                    );
                }
            }
        }

        #[test]
        fn prop_attention_softmax_rows_sum_to_one(
            n in 1usize..3,
            m in 1usize..5,
            d_k in 1usize..4,
        ) {
            let d_v = 1; // use d_v=1 so output = softmax weights * V column
            let q = sequential_floats(n*d_k, 0.1);
            let k = sequential_floats(m*d_k, 0.1);
            // V = all ones => output[i] = sum of softmax weights = 1.0
            let v = vec![1.0f32; m * d_v];
            let mut output = vec![0.0f32; n * d_v];

            attention_scalar(&q, &k, &v, n, m, d_k, d_v, &mut output);

            for i in 0..n {
                prop_assert!(
                    (output[i] - 1.0).abs() < 1e-5,
                    "softmax row {i} should sum to 1.0, got {}",
                    output[i]
                );
            }
        }
    }

    // ── AVX2 parity test ────────────────────────────────────────────────

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_attention_avx2_parity() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let n = 3;
        let m = 4;
        let d_k = 5;
        let d_v = 6;
        let q = sequential_floats(n * d_k, 0.1);
        let k = sequential_floats(m * d_k, 0.2);
        let v = sequential_floats(m * d_v, 0.15);

        let mut scalar_out = vec![0.0f32; n * d_v];
        let mut avx2_out = vec![0.0f32; n * d_v];

        attention_scalar(&q, &k, &v, n, m, d_k, d_v, &mut scalar_out);
        unsafe { attention_avx2(&q, &k, &v, n, m, d_k, d_v, &mut avx2_out) };

        // Composed operations allow up to 8 ULP
        assert_ulp_eq(&scalar_out, &avx2_out, 8);
    }

    // ── PTX structural tests ────────────────────────────────────────────

    #[test]
    fn test_attention_ptx_structure() {
        let ptx = attention_ptx();
        assert!(ptx.contains(".version 8.5"), "missing PTX version");
        assert!(ptx.contains(".target sm_90"), "missing PTX target");
        assert!(
            ptx.contains(".entry attention_kernel"),
            "missing entry point"
        );
        assert!(ptx.contains("ret;"), "missing ret instruction");
        assert!(ptx.contains(".shared"), "missing shared memory declaration");
        assert!(ptx.contains("bar.sync"), "missing barrier synchronization");
        assert!(ptx.contains("ex2.approx.f32"), "missing exp approximation");
        assert!(ptx.contains("fma.rn.f32"), "missing FMA instruction");
        let open = ptx.matches('{').count();
        let close = ptx.matches('}').count();
        assert_eq!(
            open, close,
            "unbalanced braces: {open} open vs {close} close"
        );
    }

    #[test]
    fn test_attention_ptx_nonempty() {
        assert!(!attention_ptx().is_empty());
    }

    // ── Softmax helper test ─────────────────────────────────────────────

    #[test]
    fn test_softmax_row_uniform() {
        let mut row = vec![1.0, 1.0, 1.0, 1.0];
        ops::softmax_row(&mut row);
        for &v in &row {
            assert!(
                (v - 0.25).abs() < 1e-6,
                "uniform input should give 0.25, got {v}"
            );
        }
    }

    #[test]
    fn test_softmax_row_single() {
        let mut row = vec![42.0];
        ops::softmax_row(&mut row);
        assert!(
            (row[0] - 1.0).abs() < 1e-6,
            "single element softmax should be 1.0"
        );
    }

    #[test]
    fn test_softmax_row_sums_to_one() {
        let mut row = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        ops::softmax_row(&mut row);
        let sum: f32 = row.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "softmax should sum to 1.0, got {sum}"
        );
    }

    #[test]
    fn test_softmax_row_monotonic() {
        let mut row = vec![1.0, 2.0, 3.0];
        ops::softmax_row(&mut row);
        assert!(row[0] < row[1], "softmax should preserve order");
        assert!(row[1] < row[2], "softmax should preserve order");
    }
}
