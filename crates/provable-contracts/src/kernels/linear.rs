//! Linear projection kernel.
//!
//! Matches `linear-projection-v1.yaml`.
//! `y = xW^T + b` — matrix multiply with optional bias.
//!
//! Each function provides one of three backends:
//! - `fn linear_scalar(...)` -- Pure Rust scalar reference (ground truth)
//! - `unsafe fn linear_avx2(...)` -- AVX2 SIMD implementation
//! - `fn linear_ptx() -> &'static str` -- PTX assembly source string

use super::ops;

// ────────────────────────────────────────────────────────────────────────────
// Scalar implementation
// ────────────────────────────────────────────────────────────────────────────

/// Linear projection (scalar reference).
///
/// Computes `y = x @ W^T + bias` where:
/// - `x` is `batch x in_features` (row-major)
/// - `weight` is `out_features x in_features` (row-major, transposed during multiply)
/// - `bias` is `out_features` (optional, pass empty slice for no bias)
/// - `output` is `batch x out_features`
///
/// # Panics
/// Panics if dimensions are inconsistent.
pub fn linear_scalar(
    x: &[f32],
    weight: &[f32],
    bias: &[f32],
    batch: usize,
    in_features: usize,
    out_features: usize,
    output: &mut [f32],
) {
    assert_eq!(x.len(), batch * in_features, "x dimension mismatch");
    assert_eq!(
        weight.len(),
        out_features * in_features,
        "weight dimension mismatch"
    );
    assert_eq!(
        output.len(),
        batch * out_features,
        "output dimension mismatch"
    );
    assert!(
        bias.is_empty() || bias.len() == out_features,
        "bias must be empty or out_features={out_features}, got {}",
        bias.len()
    );

    // y = x @ W^T: for each row of x, dot with each row of W
    for b in 0..batch {
        let x_row = &x[b * in_features..(b + 1) * in_features];
        for o in 0..out_features {
            let w_row = &weight[o * in_features..(o + 1) * in_features];
            let mut val = ops::dot(x_row, w_row);
            if !bias.is_empty() {
                val += bias[o];
            }
            output[b * out_features + o] = val;
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// AVX2 implementation
// ────────────────────────────────────────────────────────────────────────────

/// AVX2 linear projection -- delegates to scalar.
///
/// # Safety
/// Requires AVX2 support.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn linear_avx2(
    x: &[f32],
    weight: &[f32],
    bias: &[f32],
    batch: usize,
    in_features: usize,
    out_features: usize,
    output: &mut [f32],
) {
    linear_scalar(x, weight, bias, batch, in_features, out_features, output);
}

// ────────────────────────────────────────────────────────────────────────────
// PTX implementation
// ────────────────────────────────────────────────────────────────────────────

/// PTX assembly for linear projection.
///
/// One thread per output element (batch_idx, out_feature). Each thread
/// computes one dot product of x_row and w_row, then adds bias.
pub fn linear_ptx() -> &'static str {
    r#".version 8.5
.target sm_90
.address_size 64
.visible .entry linear_kernel(
    .param .u64 X,
    .param .u64 W,
    .param .u64 BIAS,
    .param .u64 OUT,
    .param .u32 BATCH,
    .param .u32 IN_FEAT,
    .param .u32 OUT_FEAT,
    .param .u32 HAS_BIAS
) {
    .reg .u32 %tid, %bid, %batch, %in_feat, %out_feat, %has_bias;
    .reg .u32 %b_idx, %o_idx, %k, %tmp32;
    .reg .u64 %x_ptr, %w_ptr, %bias_ptr, %out_ptr, %addr, %off64;
    .reg .f32 %acc, %x_val, %w_val, %bias_val;
    .reg .pred %p_k, %p_bias, %p_bound;

    mov.u32 %tid, %tid.x;
    mov.u32 %bid, %ctaid.x;

    ld.param.u32 %batch, [BATCH];
    ld.param.u32 %in_feat, [IN_FEAT];
    ld.param.u32 %out_feat, [OUT_FEAT];
    ld.param.u32 %has_bias, [HAS_BIAS];
    ld.param.u64 %x_ptr, [X];
    ld.param.u64 %w_ptr, [W];
    ld.param.u64 %bias_ptr, [BIAS];
    ld.param.u64 %out_ptr, [OUT];

    // bid = batch index, tid = output feature index
    mov.u32 %b_idx, %bid;
    mov.u32 %o_idx, %tid;

    setp.ge.u32 %p_bound, %o_idx, %out_feat;
    @%p_bound bra EXIT;

    // acc = dot(x[b_idx], w[o_idx])
    mov.f32 %acc, 0f00000000;
    mov.u32 %k, 0;
DOT_LOOP:
    setp.ge.u32 %p_k, %k, %in_feat;
    @%p_k bra DOT_DONE;

    // x[b_idx * in_feat + k]
    mad.lo.u32 %tmp32, %b_idx, %in_feat, %k;
    mul.wide.u32 %off64, %tmp32, 4;
    add.u64 %addr, %x_ptr, %off64;
    ld.global.f32 %x_val, [%addr];

    // w[o_idx * in_feat + k]
    mad.lo.u32 %tmp32, %o_idx, %in_feat, %k;
    mul.wide.u32 %off64, %tmp32, 4;
    add.u64 %addr, %w_ptr, %off64;
    ld.global.f32 %w_val, [%addr];

    fma.rn.f32 %acc, %x_val, %w_val, %acc;
    add.u32 %k, %k, 1;
    bra DOT_LOOP;
DOT_DONE:

    // Add bias if present
    setp.eq.u32 %p_bias, %has_bias, 0;
    @%p_bias bra STORE;
    mul.wide.u32 %off64, %o_idx, 4;
    add.u64 %addr, %bias_ptr, %off64;
    ld.global.f32 %bias_val, [%addr];
    add.f32 %acc, %acc, %bias_val;

STORE:
    mad.lo.u32 %tmp32, %b_idx, %out_feat, %o_idx;
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

    /// Verify linear projection with bias produces correct known-answer result
    #[test]
    fn test_linear_basic_with_bias() {
        // x = [[1, 2]], W = [[3, 4], [5, 6]], b = [10, 20]
        // y = x @ W^T + b = [[1*3+2*4+10, 1*5+2*6+20]] = [[21, 37]]
        let x = [1.0, 2.0];
        let w = [3.0, 4.0, 5.0, 6.0]; // 2x2
        let b = [10.0, 20.0];
        let mut output = [0.0f32; 2];

        linear_scalar(&x, &w, &b, 1, 2, 2, &mut output);
        assert!((output[0] - 21.0).abs() < 1e-5);
        assert!((output[1] - 37.0).abs() < 1e-5);
    }

    /// Verify linear projection works correctly with an empty bias slice
    #[test]
    fn test_linear_no_bias() {
        let x = [1.0, 0.0];
        let w = [1.0, 0.0, 0.0, 1.0]; // identity-ish
        let mut output = [0.0f32; 2];

        linear_scalar(&x, &w, &[], 1, 2, 2, &mut output);
        assert!((output[0] - 1.0).abs() < 1e-5);
        assert!((output[1] - 0.0).abs() < 1e-5);
    }

    /// Verify zero input produces output equal to bias
    #[test]
    fn test_linear_zero_input_returns_bias() {
        let x = [0.0, 0.0, 0.0];
        let w = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3
        let b = [7.0, 8.0];
        let mut output = [0.0f32; 2];

        linear_scalar(&x, &w, &b, 1, 3, 2, &mut output);
        assert!((output[0] - 7.0).abs() < 1e-5);
        assert!((output[1] - 8.0).abs() < 1e-5);
    }

    /// Verify linear projection handles batched inputs correctly
    #[test]
    fn test_linear_batch() {
        // batch=2, in=2, out=1, W=[[1,1]], no bias
        let x = [1.0, 2.0, 3.0, 4.0]; // 2x2
        let w = [1.0, 1.0]; // 1x2
        let mut output = [0.0f32; 2]; // 2x1

        linear_scalar(&x, &w, &[], 2, 2, 1, &mut output);
        assert!((output[0] - 3.0).abs() < 1e-5); // 1+2
        assert!((output[1] - 7.0).abs() < 1e-5); // 3+4
    }

    /// Verify linear homogeneity: f(2x) = 2*f(x) when bias is absent
    #[test]
    fn test_linear_linearity() {
        // f(2x) = 2*f(x) when no bias
        let x1 = [1.0, 2.0, 3.0];
        let x2: Vec<f32> = x1.iter().map(|v| v * 2.0).collect();
        let w = [0.5, 0.3, 0.1, 0.2, 0.4, 0.6]; // 2x3
        let mut out1 = [0.0f32; 2];
        let mut out2 = [0.0f32; 2];

        linear_scalar(&x1, &w, &[], 1, 3, 2, &mut out1);
        linear_scalar(&x2, &w, &[], 1, 3, 2, &mut out2);

        for i in 0..2 {
            assert!(
                (out2[i] - 2.0 * out1[i]).abs() < 1e-5,
                "linearity violated at {i}: f(2x)={} vs 2*f(x)={}",
                out2[i],
                2.0 * out1[i]
            );
        }
    }

    proptest! {
        #[test]
        fn prop_linear_output_finite(
            batch in 1usize..3,
            in_f in 1usize..5,
            out_f in 1usize..5,
        ) {
            let x: Vec<f32> = (0..batch * in_f).map(|i| (i as f32) * 0.1).collect();
            let w: Vec<f32> = (0..out_f * in_f).map(|i| (i as f32) * 0.1).collect();
            let b: Vec<f32> = (0..out_f).map(|i| (i as f32) * 0.01).collect();
            let mut output = vec![0.0f32; batch * out_f];

            linear_scalar(&x, &w, &b, batch, in_f, out_f, &mut output);

            for (idx, &val) in output.iter().enumerate() {
                prop_assert!(val.is_finite(), "output[{idx}] = {val} not finite");
            }
        }
    }

    /// Verify linear PTX contains required entry point and instructions
    #[test]
    fn test_linear_ptx_structure() {
        let ptx = linear_ptx();
        assert!(ptx.contains(".entry linear_kernel"));
        assert!(ptx.contains("fma.rn.f32"));
        assert!(ptx.contains("ret;"));
    }

    /// Verify AVX2 linear projection produces identical results to scalar
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_linear_avx2_parity() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let x = [1.0, 2.0, 3.0, 4.0]; // 1x4
        let w = [0.5, 0.3, 0.1, 0.2, 0.4, 0.6, 0.7, 0.8]; // 2x4
        let b = [1.0, 2.0];
        let mut scalar_out = [0.0f32; 2]; // 1x2
        let mut avx2_out = [0.0f32; 2];
        linear_scalar(&x, &w, &b, 1, 4, 2, &mut scalar_out);
        unsafe { linear_avx2(&x, &w, &b, 1, 4, 2, &mut avx2_out) };
        assert_ulp_eq(&scalar_out, &avx2_out, 0);
    }
}
