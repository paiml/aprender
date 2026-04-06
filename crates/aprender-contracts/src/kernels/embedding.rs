//! Embedding lookup kernel.
//!
//! Matches `embedding-lookup-v1.yaml`.
//! `output[i] = W[token_ids[i]]` — table lookup with bounds checking.
//!
//! Each function provides one of three backends:
//! - `fn embedding_scalar(...)` -- Pure Rust scalar reference (ground truth)
//! - `unsafe fn embedding_avx2(...)` -- AVX2 SIMD implementation
//! - `fn embedding_ptx() -> &'static str` -- PTX assembly source string

// ────────────────────────────────────────────────────────────────────────────
// Scalar implementation
// ────────────────────────────────────────────────────────────────────────────

/// Embedding table lookup (scalar reference).
///
/// `weight` is `vocab_size x dim` (row-major), `token_ids` is `seq_len` indices,
/// `output` is `seq_len x dim`.
///
/// # Panics
/// Panics if any token_id >= vocab_size or output dimensions don't match.
pub fn embedding_scalar(
    weight: &[f32],
    token_ids: &[u32],
    vocab_size: usize,
    dim: usize,
    output: &mut [f32],
) {
    assert_eq!(weight.len(), vocab_size * dim, "weight dimension mismatch");
    assert_eq!(
        output.len(),
        token_ids.len() * dim,
        "output dimension mismatch"
    );

    for (i, &tid) in token_ids.iter().enumerate() {
        let tid = tid as usize;
        assert!(
            tid < vocab_size,
            "token_id {tid} >= vocab_size {vocab_size}"
        );
        let src = &weight[tid * dim..(tid + 1) * dim];
        let dst = &mut output[i * dim..(i + 1) * dim];
        dst.copy_from_slice(src);
    }
}

// ────────────────────────────────────────────────────────────────────────────
// AVX2 implementation
// ────────────────────────────────────────────────────────────────────────────

/// AVX2 embedding lookup -- delegates to scalar (memory-bound, no compute).
///
/// # Safety
/// Requires AVX2 support.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn embedding_avx2(
    weight: &[f32],
    token_ids: &[u32],
    vocab_size: usize,
    dim: usize,
    output: &mut [f32],
) {
    embedding_scalar(weight, token_ids, vocab_size, dim, output);
}

// ────────────────────────────────────────────────────────────────────────────
// PTX implementation
// ────────────────────────────────────────────────────────────────────────────

/// PTX assembly for embedding lookup.
///
/// One thread per (token, dimension) pair. Each thread copies one f32
/// from the embedding weight table to the output.
pub fn embedding_ptx() -> &'static str {
    r#".version 8.5
.target sm_90
.address_size 64
.visible .entry embedding_kernel(
    .param .u64 WEIGHT,
    .param .u64 TOKEN_IDS,
    .param .u64 OUT,
    .param .u32 VOCAB_SIZE,
    .param .u32 DIM
) {
    .reg .u32 %tid, %bid, %dim, %vocab_size, %token_id, %offset;
    .reg .u64 %w_ptr, %t_ptr, %out_ptr, %addr, %off64;
    .reg .f32 %val;
    .reg .pred %p_bound;

    mov.u32 %tid, %tid.x;
    mov.u32 %bid, %ctaid.x;

    ld.param.u32 %dim, [DIM];
    ld.param.u32 %vocab_size, [VOCAB_SIZE];
    ld.param.u64 %w_ptr, [WEIGHT];
    ld.param.u64 %t_ptr, [TOKEN_IDS];
    ld.param.u64 %out_ptr, [OUT];

    // bid = token index, tid = dimension index
    setp.ge.u32 %p_bound, %tid, %dim;
    @%p_bound bra EXIT;

    // Load token_id = TOKEN_IDS[bid]
    mul.wide.u32 %off64, %bid, 4;
    add.u64 %addr, %t_ptr, %off64;
    ld.global.u32 %token_id, [%addr];

    // Load WEIGHT[token_id * dim + tid]
    mad.lo.u32 %offset, %token_id, %dim, %tid;
    mul.wide.u32 %off64, %offset, 4;
    add.u64 %addr, %w_ptr, %off64;
    ld.global.f32 %val, [%addr];

    // Store OUT[bid * dim + tid]
    mad.lo.u32 %offset, %bid, %dim, %tid;
    mul.wide.u32 %off64, %offset, 4;
    add.u64 %addr, %out_ptr, %off64;
    st.global.f32 [%addr], %val;

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
    fn test_embedding_basic() {
        // 3 tokens, dim=2, vocab=4
        let weight = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]; // 4x2
        let ids = [0, 2, 1];
        let mut output = [0.0f32; 6]; // 3x2

        embedding_scalar(&weight, &ids, 4, 2, &mut output);

        assert_eq!(&output[0..2], &[1.0, 2.0]); // token 0
        assert_eq!(&output[2..4], &[5.0, 6.0]); // token 2
        assert_eq!(&output[4..6], &[3.0, 4.0]); // token 1
    }

    #[test]
    fn test_embedding_single() {
        let weight = [10.0, 20.0, 30.0];
        let ids = [2];
        let mut output = [0.0f32; 1];

        embedding_scalar(&weight, &ids, 3, 1, &mut output);
        assert_eq!(output[0], 30.0);
    }

    #[test]
    #[should_panic(expected = "token_id 5 >= vocab_size 3")]
    fn test_embedding_oob() {
        let weight = [0.0f32; 6];
        let ids = [5];
        let mut output = [0.0f32; 2];
        embedding_scalar(&weight, &ids, 3, 2, &mut output);
    }

    #[test]
    fn test_embedding_deterministic() {
        let weight = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let ids = [1, 0, 2];
        let mut out1 = [0.0f32; 6];
        let mut out2 = [0.0f32; 6];
        embedding_scalar(&weight, &ids, 3, 2, &mut out1);
        embedding_scalar(&weight, &ids, 3, 2, &mut out2);
        assert_eq!(out1, out2);
    }

    proptest! {
        #[test]
        fn prop_embedding_output_finite(
            vocab_size in 2usize..8,
            dim in 1usize..5,
            seq_len in 1usize..6,
        ) {
            let weight: Vec<f32> = (0..vocab_size * dim)
                .map(|i| (i as f32) * 0.1)
                .collect();
            let ids: Vec<u32> = (0..seq_len)
                .map(|i| (i % vocab_size) as u32)
                .collect();
            let mut output = vec![0.0f32; seq_len * dim];

            embedding_scalar(&weight, &ids, vocab_size, dim, &mut output);

            for (idx, &val) in output.iter().enumerate() {
                prop_assert!(val.is_finite(), "output[{idx}] = {val} not finite");
            }
        }
    }

    #[test]
    fn test_embedding_ptx_structure() {
        let ptx = embedding_ptx();
        assert!(ptx.contains(".entry embedding_kernel"));
        assert!(ptx.contains("ret;"));
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_embedding_avx2_parity() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let weight = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let ids = [0, 3, 1];
        let mut scalar_out = [0.0f32; 6];
        let mut avx2_out = [0.0f32; 6];
        embedding_scalar(&weight, &ids, 4, 2, &mut scalar_out);
        unsafe { embedding_avx2(&weight, &ids, 4, 2, &mut avx2_out) };
        assert_eq!(scalar_out, avx2_out);
    }
}
