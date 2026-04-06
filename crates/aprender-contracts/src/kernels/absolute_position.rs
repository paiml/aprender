//! Absolute position embeddings kernel.
//!
//! Matches `absolute-position-v1.yaml`.
//! `output[t] = token_embed[t] + pos_embed[t]` — learned additive positional encoding.
//!
//! Each function provides one of three backends:
//! - `fn abs_position_scalar(...)` -- Pure Rust scalar reference (ground truth)
//! - `unsafe fn abs_position_avx2(...)` -- AVX2 SIMD implementation
//! - `fn abs_position_ptx() -> &'static str` -- PTX assembly source string

// ────────────────────────────────────────────────────────────────────────────
// Scalar implementation
// ────────────────────────────────────────────────────────────────────────────

/// Add learned position embeddings to token embeddings (scalar reference).
///
/// `token_embed` is `seq_len x dim` (row-major), `pos_embed` is `max_pos x dim`,
/// `output` is `seq_len x dim`. Each position `t` gets `output[t] = token_embed[t] + pos_embed[t]`.
///
/// # Panics
/// Panics if `seq_len > max_pos` or dimensions don't match.
pub fn abs_position_scalar(
    token_embed: &[f32],
    pos_embed: &[f32],
    seq_len: usize,
    max_pos: usize,
    dim: usize,
    output: &mut [f32],
) {
    assert_eq!(
        token_embed.len(),
        seq_len * dim,
        "token_embed dimension mismatch"
    );
    assert_eq!(
        pos_embed.len(),
        max_pos * dim,
        "pos_embed dimension mismatch"
    );
    assert_eq!(output.len(), seq_len * dim, "output dimension mismatch");
    assert!(
        seq_len <= max_pos,
        "seq_len {seq_len} exceeds max_pos {max_pos}"
    );

    for t in 0..seq_len {
        for d in 0..dim {
            let idx = t * dim + d;
            output[idx] = token_embed[idx] + pos_embed[idx];
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// AVX2 implementation
// ────────────────────────────────────────────────────────────────────────────

/// AVX2 absolute position embeddings -- delegates to scalar.
///
/// # Safety
/// Requires AVX2 support.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn abs_position_avx2(
    token_embed: &[f32],
    pos_embed: &[f32],
    seq_len: usize,
    max_pos: usize,
    dim: usize,
    output: &mut [f32],
) {
    abs_position_scalar(token_embed, pos_embed, seq_len, max_pos, dim, output);
}

// ────────────────────────────────────────────────────────────────────────────
// PTX implementation
// ────────────────────────────────────────────────────────────────────────────

/// PTX assembly for absolute position embeddings.
///
/// One thread per (position, dimension) pair. Each thread adds
/// one element from the position embedding table to the token embedding.
pub fn abs_position_ptx() -> &'static str {
    r#".version 8.5
.target sm_90
.address_size 64
.visible .entry abs_position_kernel(
    .param .u64 TOKEN_EMBED,
    .param .u64 POS_EMBED,
    .param .u64 OUT,
    .param .u32 SEQ_LEN,
    .param .u32 DIM
) {
    .reg .u32 %tid, %bid, %dim, %seq_len, %offset;
    .reg .u64 %te_ptr, %pe_ptr, %out_ptr, %addr, %off64;
    .reg .f32 %te_val, %pe_val, %result;
    .reg .pred %p_bound;

    mov.u32 %tid, %tid.x;
    mov.u32 %bid, %ctaid.x;

    ld.param.u32 %dim, [DIM];
    ld.param.u32 %seq_len, [SEQ_LEN];
    ld.param.u64 %te_ptr, [TOKEN_EMBED];
    ld.param.u64 %pe_ptr, [POS_EMBED];
    ld.param.u64 %out_ptr, [OUT];

    // bid = position index, tid = dimension index
    setp.ge.u32 %p_bound, %tid, %dim;
    @%p_bound bra EXIT;

    // offset = bid * dim + tid
    mad.lo.u32 %offset, %bid, %dim, %tid;
    mul.wide.u32 %off64, %offset, 4;

    // Load token_embed[offset]
    add.u64 %addr, %te_ptr, %off64;
    ld.global.f32 %te_val, [%addr];

    // Load pos_embed[offset] (same offset since pos[t] uses same t*dim+d)
    add.u64 %addr, %pe_ptr, %off64;
    ld.global.f32 %pe_val, [%addr];

    // output = token_embed + pos_embed
    add.f32 %result, %te_val, %pe_val;

    // Store output[offset]
    add.u64 %addr, %out_ptr, %off64;
    st.global.f32 [%addr], %result;

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
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn test_abs_position_basic() {
        let token = [1.0, 2.0, 3.0, 4.0]; // 2x2
        let pos = [0.1, 0.2, 0.3, 0.4]; // 2x2 (max_pos=2)
        let mut output = [0.0f32; 4];

        abs_position_scalar(&token, &pos, 2, 2, 2, &mut output);
        assert!((output[0] - 1.1).abs() < 1e-6);
        assert!((output[1] - 2.2).abs() < 1e-6);
        assert!((output[2] - 3.3).abs() < 1e-6);
        assert!((output[3] - 4.4).abs() < 1e-6);
    }

    #[test]
    fn test_abs_position_zero_pos_is_identity() {
        let token = [1.0, 2.0, 3.0];
        let pos = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]; // max_pos=2, dim=3
        let mut output = [0.0f32; 3];

        abs_position_scalar(&token, &pos, 1, 2, 3, &mut output);
        assert_eq!(&output, &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_abs_position_shape_preservation() {
        let seq_len = 3;
        let dim = 4;
        let max_pos = 5;
        let token = vec![1.0f32; seq_len * dim];
        let pos = vec![0.5f32; max_pos * dim];
        let mut output = vec![0.0f32; seq_len * dim];

        abs_position_scalar(&token, &pos, seq_len, max_pos, dim, &mut output);
        assert_eq!(output.len(), seq_len * dim);
    }

    #[test]
    #[should_panic(expected = "seq_len 5 exceeds max_pos 3")]
    fn test_abs_position_oob() {
        let token = vec![0.0f32; 10];
        let pos = vec![0.0f32; 6]; // max_pos=3, dim=2
        let mut output = vec![0.0f32; 10];
        abs_position_scalar(&token, &pos, 5, 3, 2, &mut output);
    }

    proptest! {
        #[test]
        fn prop_abs_position_finite(
            seq_len in 1usize..5,
            dim in 1usize..5,
        ) {
            let max_pos = seq_len + 2;
            let token: Vec<f32> = (0..seq_len * dim).map(|i| (i as f32) * 0.1).collect();
            let pos: Vec<f32> = (0..max_pos * dim).map(|i| (i as f32) * 0.01).collect();
            let mut output = vec![0.0f32; seq_len * dim];

            abs_position_scalar(&token, &pos, seq_len, max_pos, dim, &mut output);

            for (idx, &val) in output.iter().enumerate() {
                prop_assert!(val.is_finite(), "output[{idx}] = {val} not finite");
            }
        }
    }

    #[test]
    fn test_abs_position_ptx_structure() {
        let ptx = abs_position_ptx();
        assert!(ptx.contains(".entry abs_position_kernel"));
        assert!(ptx.contains("add.f32"));
        assert!(ptx.contains("ret;"));
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_abs_position_avx2_parity() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let token = [1.0, 2.0, 3.0, 4.0];
        let pos = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]; // max_pos=4
        let mut scalar_out = [0.0f32; 4];
        let mut avx2_out = [0.0f32; 4];
        abs_position_scalar(&token, &pos, 2, 4, 2, &mut scalar_out);
        unsafe { abs_position_avx2(&token, &pos, 2, 4, 2, &mut avx2_out) };
        assert_eq!(scalar_out, avx2_out);
    }
}
