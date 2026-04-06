//! ALiBi (Attention with Linear Biases) kernel.
//!
//! Matches `alibi-kernel-v1.yaml`.
//! `scores[i,j] += -m_h * |i - j|` where `m_h = 2^(-8h/H)`.
//!
//! Each function provides one of three backends:
//! - `fn alibi_bias_scalar(...)` -- Pure Rust scalar reference (ground truth)
//! - `unsafe fn alibi_bias_avx2(...)` -- AVX2 SIMD implementation
//! - `fn alibi_ptx() -> &'static str` -- PTX assembly source string

// ────────────────────────────────────────────────────────────────────────────
// Scalar implementation
// ────────────────────────────────────────────────────────────────────────────

/// Compute ALiBi slope for head `h` of `num_heads` total.
///
/// `m_h = 2^(-8 * (h+1) / num_heads)`
#[inline]
pub fn alibi_slope(h: usize, num_heads: usize) -> f32 {
    let exponent = -8.0 * ((h + 1) as f32) / (num_heads as f32);
    2.0f32.powf(exponent)
}

/// Add ALiBi bias to attention scores (scalar reference).
///
/// `scores` is `num_heads x seq_len x seq_len` (row-major).
/// For head h, position i, position j: `scores[h,i,j] += -m_h * |i - j|`.
///
/// # Panics
/// Panics if dimensions don't match.
pub fn alibi_bias_scalar(scores: &mut [f32], num_heads: usize, seq_len: usize) {
    assert_eq!(
        scores.len(),
        num_heads * seq_len * seq_len,
        "scores dimension mismatch"
    );

    let head_stride = seq_len * seq_len;

    for h in 0..num_heads {
        let slope = alibi_slope(h, num_heads);
        let base = h * head_stride;

        for i in 0..seq_len {
            for j in 0..seq_len {
                let dist = i.abs_diff(j);
                scores[base + i * seq_len + j] -= slope * (dist as f32);
            }
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// AVX2 implementation
// ────────────────────────────────────────────────────────────────────────────

/// AVX2 ALiBi bias -- delegates to scalar.
///
/// # Safety
/// Requires AVX2 support.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn alibi_bias_avx2(scores: &mut [f32], num_heads: usize, seq_len: usize) {
    alibi_bias_scalar(scores, num_heads, seq_len);
}

// ────────────────────────────────────────────────────────────────────────────
// PTX implementation
// ────────────────────────────────────────────────────────────────────────────

/// PTX assembly for ALiBi bias addition.
///
/// One thread block per head. Each thread handles one (i, j) pair.
pub fn alibi_ptx() -> &'static str {
    r#".version 8.5
.target sm_90
.address_size 64
.visible .entry alibi_kernel(
    .param .u64 SCORES,
    .param .u32 NUM_HEADS,
    .param .u32 SEQ_LEN
) {
    .reg .u32 %tid, %bid, %num_heads, %seq_len;
    .reg .u32 %head, %i, %j, %dist, %head_stride, %offset;
    .reg .u64 %scores_ptr, %addr, %off64;
    .reg .f32 %slope, %dist_f, %bias, %score, %exp, %neg8, %h_f, %nh_f;
    .reg .pred %p_bound, %p_ge;

    mov.u32 %tid, %tid.x;
    mov.u32 %bid, %ctaid.x;

    ld.param.u32 %num_heads, [NUM_HEADS];
    ld.param.u32 %seq_len, [SEQ_LEN];
    ld.param.u64 %scores_ptr, [SCORES];

    // bid = head, tid = flattened (i * seq_len + j)
    mov.u32 %head, %bid;
    mul.lo.u32 %head_stride, %seq_len, %seq_len;

    // i = tid / seq_len, j = tid % seq_len
    div.u32 %i, %tid, %seq_len;
    rem.u32 %j, %tid, %seq_len;

    setp.ge.u32 %p_bound, %tid, %head_stride;
    @%p_bound bra EXIT;

    // slope = 2^(-8 * (head+1) / num_heads)
    add.u32 %offset, %head, 1;
    cvt.rn.f32.u32 %h_f, %offset;
    cvt.rn.f32.u32 %nh_f, %num_heads;
    mov.f32 %neg8, 0fC1000000;
    mul.f32 %exp, %neg8, %h_f;
    div.rn.f32 %exp, %exp, %nh_f;
    ex2.approx.f32 %slope, %exp;

    // dist = |i - j|
    setp.ge.u32 %p_ge, %i, %j;
    @%p_ge bra CALC_DIST_FORWARD;
    sub.u32 %dist, %j, %i;
    bra APPLY_BIAS;
CALC_DIST_FORWARD:
    sub.u32 %dist, %i, %j;

APPLY_BIAS:
    cvt.rn.f32.u32 %dist_f, %dist;
    mul.f32 %bias, %slope, %dist_f;

    // scores[head * head_stride + tid] -= bias
    mad.lo.u32 %offset, %head, %head_stride, %tid;
    mul.wide.u32 %off64, %offset, 4;
    add.u64 %addr, %scores_ptr, %off64;
    ld.global.f32 %score, [%addr];
    sub.f32 %score, %score, %bias;
    st.global.f32 [%addr], %score;

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
    fn test_alibi_slopes() {
        // 8 heads: slopes should be 2^(-1), 2^(-2), ..., 2^(-8)
        let slopes: Vec<f32> = (0..8).map(|h| alibi_slope(h, 8)).collect();
        assert!((slopes[0] - 0.5).abs() < 1e-6);
        assert!((slopes[1] - 0.25).abs() < 1e-6);
        assert!((slopes[7] - 1.0 / 256.0).abs() < 1e-6);
        // Slopes must be monotonically decreasing
        for i in 1..8 {
            assert!(slopes[i] < slopes[i - 1], "slopes not decreasing at {i}");
        }
    }

    #[test]
    fn test_alibi_diagonal_zero() {
        // On the diagonal (i==j), bias should be 0 (distance = 0)
        let seq_len = 4;
        let num_heads = 2;
        let mut scores = vec![1.0f32; num_heads * seq_len * seq_len];
        alibi_bias_scalar(&mut scores, num_heads, seq_len);

        for h in 0..num_heads {
            for i in 0..seq_len {
                let idx = h * seq_len * seq_len + i * seq_len + i;
                assert_eq!(
                    scores[idx], 1.0,
                    "diagonal should be unchanged at h={h} i={i}"
                );
            }
        }
    }

    #[test]
    fn test_alibi_negative_bias() {
        // Off-diagonal elements should have scores decreased
        let seq_len = 3;
        let num_heads = 1;
        let mut scores = vec![0.0f32; seq_len * seq_len];
        alibi_bias_scalar(&mut scores, num_heads, seq_len);

        // All off-diagonal should be negative (bias subtracted from 0)
        for i in 0..seq_len {
            for j in 0..seq_len {
                if i != j {
                    assert!(
                        scores[i * seq_len + j] < 0.0,
                        "off-diagonal [{i},{j}] should be negative, got {}",
                        scores[i * seq_len + j]
                    );
                }
            }
        }
    }

    #[test]
    fn test_alibi_symmetry() {
        // |i-j| = |j-i|, so bias is symmetric for each head
        let seq_len = 5;
        let num_heads = 2;
        let mut scores = vec![0.0f32; num_heads * seq_len * seq_len];
        alibi_bias_scalar(&mut scores, num_heads, seq_len);

        for h in 0..num_heads {
            let base = h * seq_len * seq_len;
            for i in 0..seq_len {
                for j in 0..seq_len {
                    let a = scores[base + i * seq_len + j];
                    let b = scores[base + j * seq_len + i];
                    assert!(
                        (a - b).abs() < 1e-6,
                        "asymmetry at h={h} [{i},{j}]: {a} vs {b}"
                    );
                }
            }
        }
    }

    proptest! {
        #[test]
        fn prop_alibi_slopes_positive(num_heads in 1usize..17) {
            for h in 0..num_heads {
                let s = alibi_slope(h, num_heads);
                prop_assert!(s > 0.0, "slope must be positive, got {s} at h={h}");
                prop_assert!(s <= 1.0, "slope must be <= 1, got {s} at h={h}");
            }
        }

        #[test]
        fn prop_alibi_output_finite(
            num_heads in 1usize..5,
            seq_len in 1usize..8,
        ) {
            let mut scores = vec![0.0f32; num_heads * seq_len * seq_len];
            alibi_bias_scalar(&mut scores, num_heads, seq_len);

            for (idx, &val) in scores.iter().enumerate() {
                prop_assert!(val.is_finite(), "scores[{idx}] = {val} not finite");
            }
        }
    }

    #[test]
    fn test_alibi_ptx_structure() {
        let ptx = alibi_ptx();
        assert!(ptx.contains(".entry alibi_kernel"));
        assert!(ptx.contains("ex2.approx.f32"));
        assert!(ptx.contains("ret;"));
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_alibi_avx2_parity() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let num_heads = 2;
        let seq_len = 4;
        let mut scalar_scores = vec![1.0f32; num_heads * seq_len * seq_len];
        let mut avx2_scores = scalar_scores.clone();
        alibi_bias_scalar(&mut scalar_scores, num_heads, seq_len);
        unsafe { alibi_bias_avx2(&mut avx2_scores, num_heads, seq_len) };
        assert_eq!(scalar_scores, avx2_scores);
    }
}
