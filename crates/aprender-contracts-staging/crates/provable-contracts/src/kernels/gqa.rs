//! Grouped Query Attention kernel.
//!
//! Matches `gqa-kernel-v1.yaml`.
//! KV head broadcasting: kv_head = query_head / (num_heads / num_kv_heads)
//!
//! Each function provides one of three backends:
//! - `fn gqa_scalar(...)` -- Pure Rust scalar reference (ground truth)
//! - `unsafe fn gqa_avx2(...)` -- AVX2 SIMD implementation
//! - `fn gqa_ptx() -> &'static str` -- PTX assembly source string

use super::ops;

/// Single-head attention helper: computes attention for one query sequence
/// against one KV head.
///
/// Q_head is seq_len x d_k, K_head is seq_len x d_k, V_head is seq_len x d_v,
/// output is seq_len x d_v.
fn single_head_attention(
    q_head: &[f32],
    k_head: &[f32],
    v_head: &[f32],
    seq_len: usize,
    d_k: usize,
    d_v: usize,
    output: &mut [f32],
) {
    // scores = Q_head * K_head^T / sqrt(d_k), shape seq_len x seq_len
    let mut scores = vec![0.0f32; seq_len * seq_len];
    ops::score_matrix(q_head, k_head, seq_len, seq_len, d_k, &mut scores);

    // Softmax each row
    ops::softmax_rows(&mut scores, seq_len, seq_len);

    // output = scores * V_head, shape seq_len x d_v
    ops::matmul_sv(&scores, v_head, seq_len, seq_len, d_v, output);
}

// ────────────────────────────────────────────────────────────────────────────
// Scalar implementation
// ────────────────────────────────────────────────────────────────────────────

/// Grouped Query Attention (scalar reference).
///
/// For each query head h in 0..num_heads:
///   kv_head = h / (num_heads / num_kv_heads)
///   Compute attention(Q\[h\], K\[kv_head\], V\[kv_head\]) -> output\[h\]
///
/// Layout (all row-major):
/// - Q: num_heads * seq_len * d_k
/// - K: num_kv_heads * seq_len * d_k
/// - V: num_kv_heads * seq_len * d_v
/// - output: num_heads * seq_len * d_v
///
/// # Panics
/// Panics if `num_heads % num_kv_heads != 0` or dimensions are inconsistent.
pub fn gqa_scalar(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq_len: usize,
    d_k: usize,
    d_v: usize,
    num_heads: usize,
    num_kv_heads: usize,
    output: &mut [f32],
) {
    assert!(
        num_kv_heads > 0 && num_heads % num_kv_heads == 0,
        "num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
    );
    let q_total = num_heads * seq_len * d_k;
    let k_total = num_kv_heads * seq_len * d_k;
    let v_total = num_kv_heads * seq_len * d_v;
    let o_total = num_heads * seq_len * d_v;
    assert_eq!(
        q.len(),
        q_total,
        "Q dimension mismatch: expected {q_total} got {}",
        q.len()
    );
    assert_eq!(
        k.len(),
        k_total,
        "K dimension mismatch: expected {k_total} got {}",
        k.len()
    );
    assert_eq!(
        v.len(),
        v_total,
        "V dimension mismatch: expected {v_total} got {}",
        v.len()
    );
    assert_eq!(
        output.len(),
        o_total,
        "output dimension mismatch: expected {o_total} got {}",
        output.len()
    );

    let heads_per_kv = num_heads / num_kv_heads;
    let q_head_stride = seq_len * d_k;
    let k_head_stride = seq_len * d_k;
    let v_head_stride = seq_len * d_v;
    let o_head_stride = seq_len * d_v;

    for h in 0..num_heads {
        let kv_head = h / heads_per_kv;

        let q_start = h * q_head_stride;
        let k_start = kv_head * k_head_stride;
        let v_start = kv_head * v_head_stride;
        let o_start = h * o_head_stride;

        let q_head = &q[q_start..q_start + q_head_stride];
        let k_head = &k[k_start..k_start + k_head_stride];
        let v_head = &v[v_start..v_start + v_head_stride];
        let o_head = &mut output[o_start..o_start + o_head_stride];

        single_head_attention(q_head, k_head, v_head, seq_len, d_k, d_v, o_head);
    }
}

// ────────────────────────────────────────────────────────────────────────────
// AVX2 implementation
// ────────────────────────────────────────────────────────────────────────────

/// AVX2 Grouped Query Attention -- delegates to scalar.
///
/// # Safety
/// Requires AVX2 support. Caller must verify with `is_x86_feature_detected!("avx2")`.
///
/// # Panics
/// Panics if dimensions are inconsistent.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn gqa_avx2(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq_len: usize,
    d_k: usize,
    d_v: usize,
    num_heads: usize,
    num_kv_heads: usize,
    output: &mut [f32],
) {
    gqa_scalar(q, k, v, seq_len, d_k, d_v, num_heads, num_kv_heads, output);
}

include!("gqa_ptx.rs");

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::super::ops::sequential_floats;
    use super::super::ulp::assert_ulp_eq;
    use super::*;
    use proptest::prelude::*;

    // ── MHA equivalence (num_heads == num_kv_heads) ─────────────────────

    #[test]
    fn test_gqa_equals_mha_when_heads_match() {
        // When num_heads == num_kv_heads, GQA degenerates to standard MHA.
        // Each query head gets its own unique KV head.
        let seq_len = 2;
        let d_k = 3;
        let d_v = 2;
        let num_heads = 2;
        let num_kv_heads = 2;

        let q = sequential_floats(num_heads * seq_len * d_k, 0.1);
        let k = sequential_floats(num_kv_heads * seq_len * d_k, 0.15);
        let v = sequential_floats(num_kv_heads * seq_len * d_v, 0.2);
        let mut output = vec![0.0f32; num_heads * seq_len * d_v];

        gqa_scalar(
            &q,
            &k,
            &v,
            seq_len,
            d_k,
            d_v,
            num_heads,
            num_kv_heads,
            &mut output,
        );

        // Verify by computing each head independently
        for h in 0..num_heads {
            let q_start = h * seq_len * d_k;
            let k_start = h * seq_len * d_k; // kv_head == h since num_heads == num_kv_heads
            let v_start = h * seq_len * d_v;
            let o_start = h * seq_len * d_v;

            let mut expected = vec![0.0f32; seq_len * d_v];
            single_head_attention(
                &q[q_start..q_start + seq_len * d_k],
                &k[k_start..k_start + seq_len * d_k],
                &v[v_start..v_start + seq_len * d_v],
                seq_len,
                d_k,
                d_v,
                &mut expected,
            );

            assert_ulp_eq(&output[o_start..o_start + seq_len * d_v], &expected, 0);
        }
    }

    // ── KV broadcasting test ────────────────────────────────────────────

    #[test]
    fn test_gqa_kv_broadcasting() {
        // 4 query heads, 2 kv heads: heads 0,1 use kv 0; heads 2,3 use kv 1
        let seq_len = 2;
        let d_k = 2;
        let d_v = 2;
        let num_heads = 4;
        let num_kv_heads = 2;

        let q = sequential_floats(num_heads * seq_len * d_k, 0.1);
        let k = sequential_floats(num_kv_heads * seq_len * d_k, 0.2);
        let v = sequential_floats(num_kv_heads * seq_len * d_v, 0.15);
        let mut output = vec![0.0f32; num_heads * seq_len * d_v];

        gqa_scalar(
            &q,
            &k,
            &v,
            seq_len,
            d_k,
            d_v,
            num_heads,
            num_kv_heads,
            &mut output,
        );

        // Verify: heads 0 and 1 use kv_head=0, heads 2 and 3 use kv_head=1
        let head_stride_o = seq_len * d_v;

        // Head 0 and head 1 both use KV head 0, but with different Q
        // So they should generally produce different outputs (different Q)
        // but both use the same K, V from kv_head 0
        let mut head0_ref = vec![0.0f32; seq_len * d_v];
        let mut head1_ref = vec![0.0f32; seq_len * d_v];
        single_head_attention(
            &q[0..seq_len * d_k],
            &k[0..seq_len * d_k], // kv head 0
            &v[0..seq_len * d_v], // kv head 0
            seq_len,
            d_k,
            d_v,
            &mut head0_ref,
        );
        single_head_attention(
            &q[seq_len * d_k..2 * seq_len * d_k],
            &k[0..seq_len * d_k], // kv head 0 (shared)
            &v[0..seq_len * d_v], // kv head 0 (shared)
            seq_len,
            d_k,
            d_v,
            &mut head1_ref,
        );

        assert_ulp_eq(&output[0..head_stride_o], &head0_ref, 0);
        assert_ulp_eq(&output[head_stride_o..2 * head_stride_o], &head1_ref, 0);

        // Head 2 and head 3 use kv_head 1
        let mut head2_ref = vec![0.0f32; seq_len * d_v];
        let mut head3_ref = vec![0.0f32; seq_len * d_v];
        single_head_attention(
            &q[2 * seq_len * d_k..3 * seq_len * d_k],
            &k[seq_len * d_k..2 * seq_len * d_k], // kv head 1
            &v[seq_len * d_v..2 * seq_len * d_v], // kv head 1
            seq_len,
            d_k,
            d_v,
            &mut head2_ref,
        );
        single_head_attention(
            &q[3 * seq_len * d_k..4 * seq_len * d_k],
            &k[seq_len * d_k..2 * seq_len * d_k], // kv head 1
            &v[seq_len * d_v..2 * seq_len * d_v], // kv head 1
            seq_len,
            d_k,
            d_v,
            &mut head3_ref,
        );

        assert_ulp_eq(&output[2 * head_stride_o..3 * head_stride_o], &head2_ref, 0);
        assert_ulp_eq(&output[3 * head_stride_o..4 * head_stride_o], &head3_ref, 0);
    }

    // ── Single head, single position ────────────────────────────────────

    #[test]
    fn test_gqa_single_head_single_pos() {
        // Minimal case: 1 head, 1 kv head, seq_len=1
        let seq_len = 1;
        let d_k = 2;
        let d_v = 3;
        let num_heads = 1;
        let num_kv_heads = 1;

        let q = vec![1.0, 0.5];
        let k = vec![0.5, 1.0];
        let v = vec![2.0, 3.0, 4.0];
        let mut output = vec![0.0f32; d_v];

        gqa_scalar(
            &q,
            &k,
            &v,
            seq_len,
            d_k,
            d_v,
            num_heads,
            num_kv_heads,
            &mut output,
        );

        // Single query, single key: softmax of single score = 1.0, output = V
        assert_ulp_eq(&output, &v, 0);
    }

    // ── Assertion tests ─────────────────────────────────────────────────

    #[test]
    #[should_panic(expected = "must be divisible")]
    fn test_gqa_bad_head_ratio() {
        let mut output = vec![0.0f32; 4];
        gqa_scalar(&[0.0; 6], &[0.0; 4], &[0.0; 4], 1, 2, 2, 3, 2, &mut output);
    }

    #[test]
    #[should_panic(expected = "Q dimension mismatch")]
    fn test_gqa_bad_q_dim() {
        let mut output = vec![0.0f32; 4];
        gqa_scalar(&[0.0; 3], &[0.0; 2], &[0.0; 2], 1, 2, 2, 2, 2, &mut output);
    }

    // ── Property-based tests ────────────────────────────────────────────

    proptest! {
        #[test]
        fn prop_gqa_output_finite(
            seq_len in 1usize..3,
            d_k in 1usize..4,
            d_v in 1usize..4,
        ) {
            let num_heads = 4usize;
            let num_kv_heads = 2usize;

            let q = sequential_floats(num_heads * seq_len * d_k, 0.1);
            let k = sequential_floats(num_kv_heads * seq_len * d_k, 0.1);
            let v = sequential_floats(num_kv_heads * seq_len * d_v, 0.1);
            let mut output = vec![0.0f32; num_heads * seq_len * d_v];

            gqa_scalar(&q, &k, &v, seq_len, d_k, d_v, num_heads, num_kv_heads, &mut output);

            for (idx, &val) in output.iter().enumerate() {
                prop_assert!(val.is_finite(), "output[{idx}] = {val} is not finite");
            }
        }

        #[test]
        fn prop_gqa_mha_equivalence(
            seq_len in 1usize..3,
            d_k in 1usize..3,
            d_v in 1usize..3,
            num_heads in 1usize..4,
        ) {
            // When num_heads == num_kv_heads, each head is independent
            let num_kv_heads = num_heads;
            let q = sequential_floats(num_heads * seq_len * d_k, 0.1);
            let k = sequential_floats(num_kv_heads * seq_len * d_k, 0.15);
            let v = sequential_floats(num_kv_heads * seq_len * d_v, 0.2);
            let mut output = vec![0.0f32; num_heads * seq_len * d_v];

            gqa_scalar(&q, &k, &v, seq_len, d_k, d_v, num_heads, num_kv_heads, &mut output);

            // Verify each head independently
            for h in 0..num_heads {
                let q_start = h * seq_len * d_k;
                let k_start = h * seq_len * d_k;
                let v_start = h * seq_len * d_v;
                let o_start = h * seq_len * d_v;
                let o_len = seq_len * d_v;

                let mut expected = vec![0.0f32; o_len];
                single_head_attention(
                    &q[q_start..q_start + seq_len * d_k],
                    &k[k_start..k_start + seq_len * d_k],
                    &v[v_start..v_start + seq_len * d_v],
                    seq_len, d_k, d_v, &mut expected,
                );

                for idx in 0..o_len {
                    let diff = (output[o_start + idx] - expected[idx]).abs();
                    prop_assert!(
                        diff < 1e-5,
                        "head {h} idx {idx}: expected {} got {} (diff {diff})",
                        expected[idx], output[o_start + idx]
                    );
                }
            }
        }
    }

    // ── AVX2 parity test ────────────────────────────────────────────────

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_gqa_avx2_parity() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let seq_len = 3;
        let d_k = 4;
        let d_v = 2;
        let num_heads = 4;
        let num_kv_heads = 2;

        let q = sequential_floats(num_heads * seq_len * d_k, 0.1);
        let k = sequential_floats(num_kv_heads * seq_len * d_k, 0.2);
        let v = sequential_floats(num_kv_heads * seq_len * d_v, 0.15);

        let mut scalar_out = vec![0.0f32; num_heads * seq_len * d_v];
        let mut avx2_out = vec![0.0f32; num_heads * seq_len * d_v];

        gqa_scalar(
            &q,
            &k,
            &v,
            seq_len,
            d_k,
            d_v,
            num_heads,
            num_kv_heads,
            &mut scalar_out,
        );
        unsafe {
            gqa_avx2(
                &q,
                &k,
                &v,
                seq_len,
                d_k,
                d_v,
                num_heads,
                num_kv_heads,
                &mut avx2_out,
            );
        }

        assert_ulp_eq(&scalar_out, &avx2_out, 8);
    }

    // ── PTX structural tests ────────────────────────────────────────────

    #[test]
    fn test_gqa_ptx_structure() {
        let ptx = gqa_ptx();
        assert!(ptx.contains(".version 8.5"), "missing PTX version");
        assert!(ptx.contains(".target sm_90"), "missing PTX target");
        assert!(ptx.contains(".entry gqa_kernel"), "missing entry point");
        assert!(ptx.contains("ret;"), "missing ret instruction");
        assert!(ptx.contains(".shared"), "missing shared memory declaration");
        assert!(ptx.contains("bar.sync"), "missing barrier synchronization");
        assert!(
            ptx.contains("div.u32"),
            "missing integer division for head mapping"
        );
        assert!(ptx.contains("ex2.approx.f32"), "missing exp approximation");
        let open = ptx.matches('{').count();
        let close = ptx.matches('}').count();
        assert_eq!(
            open, close,
            "unbalanced braces: {open} open vs {close} close"
        );
    }

    #[test]
    fn test_gqa_ptx_nonempty() {
        assert!(!gqa_ptx().is_empty());
    }
}
