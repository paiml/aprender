//! Tied embeddings kernel (language model head).
//!
//! Matches `tied-embeddings-v1.yaml`.
//! `logits = x @ W_embed^T` — reuse embedding weight matrix as LM head projection.
//!
//! Each function provides one of three backends:
//! - `fn tied_lm_head_scalar(...)` -- Pure Rust scalar reference (ground truth)
//! - `unsafe fn tied_lm_head_avx2(...)` -- AVX2 SIMD implementation
//! - `fn tied_embeddings_ptx() -> &'static str` -- PTX assembly source string

use super::ops;

// ────────────────────────────────────────────────────────────────────────────
// Scalar implementation
// ────────────────────────────────────────────────────────────────────────────

/// Tied embedding LM head: `logits = x @ W_embed^T` (scalar reference).
///
/// `x` is `seq_len x d_model`, `embed_weight` is `vocab_size x d_model`,
/// `output` is `seq_len x vocab_size`. This is identical to a linear projection
/// with no bias, reusing the embedding matrix.
///
/// # Panics
/// Panics if dimensions don't match.
pub fn tied_lm_head_scalar(
    x: &[f32],
    embed_weight: &[f32],
    seq_len: usize,
    d_model: usize,
    vocab_size: usize,
    output: &mut [f32],
) {
    assert_eq!(x.len(), seq_len * d_model, "x dimension mismatch");
    assert_eq!(
        embed_weight.len(),
        vocab_size * d_model,
        "embed_weight dimension mismatch"
    );
    assert_eq!(
        output.len(),
        seq_len * vocab_size,
        "output dimension mismatch"
    );

    for s in 0..seq_len {
        let x_row = &x[s * d_model..(s + 1) * d_model];
        for v in 0..vocab_size {
            let w_row = &embed_weight[v * d_model..(v + 1) * d_model];
            output[s * vocab_size + v] = ops::dot(x_row, w_row);
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// AVX2 implementation
// ────────────────────────────────────────────────────────────────────────────

/// AVX2 tied embedding LM head -- delegates to scalar.
///
/// # Safety
/// Requires AVX2 support.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn tied_lm_head_avx2(
    x: &[f32],
    embed_weight: &[f32],
    seq_len: usize,
    d_model: usize,
    vocab_size: usize,
    output: &mut [f32],
) {
    tied_lm_head_scalar(x, embed_weight, seq_len, d_model, vocab_size, output);
}

// ────────────────────────────────────────────────────────────────────────────
// PTX implementation
// ────────────────────────────────────────────────────────────────────────────

/// PTX assembly for tied embedding LM head.
///
/// One thread per output element. bid = seq position, tid = vocab index.
/// Each thread computes dot(x[bid], embed_weight[tid]).
pub fn tied_embeddings_ptx() -> &'static str {
    r#".version 8.5
.target sm_90
.address_size 64
.visible .entry tied_lm_head_kernel(
    .param .u64 X,
    .param .u64 W_EMBED,
    .param .u64 OUT,
    .param .u32 D_MODEL,
    .param .u32 VOCAB_SIZE
) {
    .reg .u32 %tid, %bid, %d_model, %vocab_size, %k, %tmp32;
    .reg .u64 %x_ptr, %w_ptr, %out_ptr, %addr, %off64;
    .reg .f32 %acc, %x_val, %w_val;
    .reg .pred %p_bound, %p_k;

    mov.u32 %tid, %tid.x;
    mov.u32 %bid, %ctaid.x;

    ld.param.u32 %d_model, [D_MODEL];
    ld.param.u32 %vocab_size, [VOCAB_SIZE];
    ld.param.u64 %x_ptr, [X];
    ld.param.u64 %w_ptr, [W_EMBED];
    ld.param.u64 %out_ptr, [OUT];

    // tid = vocab index
    setp.ge.u32 %p_bound, %tid, %vocab_size;
    @%p_bound bra EXIT;

    // acc = dot(x[bid], w_embed[tid])
    mov.f32 %acc, 0f00000000;
    mov.u32 %k, 0;
DOT_LOOP:
    setp.ge.u32 %p_k, %k, %d_model;
    @%p_k bra DOT_DONE;

    // x[bid * d_model + k]
    mad.lo.u32 %tmp32, %bid, %d_model, %k;
    mul.wide.u32 %off64, %tmp32, 4;
    add.u64 %addr, %x_ptr, %off64;
    ld.global.f32 %x_val, [%addr];

    // w_embed[tid * d_model + k]
    mad.lo.u32 %tmp32, %tid, %d_model, %k;
    mul.wide.u32 %off64, %tmp32, 4;
    add.u64 %addr, %w_ptr, %off64;
    ld.global.f32 %w_val, [%addr];

    fma.rn.f32 %acc, %x_val, %w_val, %acc;
    add.u32 %k, %k, 1;
    bra DOT_LOOP;
DOT_DONE:

    // out[bid * vocab_size + tid] = acc
    mad.lo.u32 %tmp32, %bid, %vocab_size, %tid;
    mul.wide.u32 %off64, %tmp32, 4;
    add.u64 %addr, %out_ptr, %off64;
    st.global.f32 [%addr], %acc;

EXIT:
    ret;
}
"#
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::super::ulp::assert_ulp_eq;
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn test_tied_basic() {
        // x = [[1, 0]], W = [[1, 0], [0, 1], [1, 1]]
        // logits = x @ W^T = [[1, 0, 1]]
        let x = [1.0, 0.0];
        let w = [1.0, 0.0, 0.0, 1.0, 1.0, 1.0]; // 3x2
        let mut output = [0.0f32; 3];

        tied_lm_head_scalar(&x, &w, 1, 2, 3, &mut output);
        assert!((output[0] - 1.0).abs() < 1e-5);
        assert!((output[1] - 0.0).abs() < 1e-5);
        assert!((output[2] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_tied_equals_linear_no_bias() {
        // Tied embedding head is just linear projection with no bias
        let x = [1.0, 2.0, 3.0, 4.0]; // 2x2
        let w = [0.5, 0.3, 0.1, 0.2]; // 2x2
        let mut tied_out = [0.0f32; 4];
        let mut linear_out = [0.0f32; 4];

        tied_lm_head_scalar(&x, &w, 2, 2, 2, &mut tied_out);
        super::super::linear::linear_scalar(&x, &w, &[], 2, 2, 2, &mut linear_out);
        assert_ulp_eq(&tied_out, &linear_out, 0);
    }

    #[test]
    fn test_tied_zero_input() {
        let x = [0.0, 0.0];
        let w = [1.0, 2.0, 3.0, 4.0]; // 2x2
        let mut output = [0.0f32; 2];
        tied_lm_head_scalar(&x, &w, 1, 2, 2, &mut output);
        assert_eq!(&output, &[0.0, 0.0]);
    }

    proptest! {
        #[test]
        fn prop_tied_output_finite(
            seq_len in 1usize..3,
            d_model in 1usize..5,
            vocab_size in 1usize..5,
        ) {
            let x: Vec<f32> = (0..seq_len * d_model).map(|i| (i as f32) * 0.1).collect();
            let w: Vec<f32> = (0..vocab_size * d_model).map(|i| (i as f32) * 0.1).collect();
            let mut output = vec![0.0f32; seq_len * vocab_size];

            tied_lm_head_scalar(&x, &w, seq_len, d_model, vocab_size, &mut output);

            for (idx, &val) in output.iter().enumerate() {
                prop_assert!(val.is_finite(), "output[{idx}] = {val} not finite");
            }
        }
    }

    #[test]
    fn test_tied_ptx_structure() {
        let ptx = tied_embeddings_ptx();
        assert!(ptx.contains(".entry tied_lm_head_kernel"));
        assert!(ptx.contains("fma.rn.f32"));
        assert!(ptx.contains("ret;"));
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_tied_avx2_parity() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let x = [1.0, 2.0, 3.0, 4.0]; // 2x2
        let w = [0.5, 0.3, 0.7, 0.1]; // 2x2
        let mut scalar_out = [0.0f32; 4];
        let mut avx2_out = [0.0f32; 4];
        tied_lm_head_scalar(&x, &w, 2, 2, 2, &mut scalar_out);
        unsafe { tied_lm_head_avx2(&x, &w, 2, 2, 2, &mut avx2_out) };
        assert_ulp_eq(&scalar_out, &avx2_out, 0);
    }
}
