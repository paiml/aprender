//! Rotary Position Embedding (RoPE) kernel.
//!
//! Matches `rope-kernel-v1.yaml`.
//! Rotate pairs of dimensions by position-dependent angles.
//! theta_k = base^(-2k/d), apply 2D rotation matrix per pair.
//!
//! RoPE(x, m)_{2k}   = x_{2k} * cos(m * theta_k) - x_{2k+1} * sin(m * theta_k)
//! RoPE(x, m)_{2k+1} = x_{2k} * sin(m * theta_k) + x_{2k+1} * cos(m * theta_k)
//!
//! Each function provides one of three backends:
//! - `fn rope_scalar(...)` — Pure Rust scalar reference (ground truth)
//! - `unsafe fn rope_avx2(...)` — AVX2 SIMD implementation
//! - `fn rope_ptx() -> &'static str` — PTX assembly source string

// ────────────────────────────────────────────────────────────────────────────
// Scalar implementation
// ────────────────────────────────────────────────────────────────────────────

/// Apply Rotary Position Embedding to input vector `x`.
///
/// For each pair of dimensions (2k, 2k+1):
///   theta_k = base^(-2k / dim) * position
///   output[2k]   = x[2k] * cos(theta_k) - x[2k+1] * sin(theta_k)
///   output[2k+1] = x[2k] * sin(theta_k) + x[2k+1] * cos(theta_k)
///
/// # Panics
/// Panics if:
/// - `x.len() != dim`
/// - `x.len() != output.len()`
/// - `dim` is odd (must be even for pair-wise rotation)
/// - `dim` is zero
pub fn rope_scalar(x: &[f32], position: u32, dim: usize, base: f32, output: &mut [f32]) {
    assert_eq!(x.len(), dim, "x length must equal dim");
    assert_eq!(x.len(), output.len(), "x/output length mismatch");
    assert!(dim > 0, "dim must be positive");
    assert_eq!(dim % 2, 0, "dim must be even for pair-wise rotation");

    let half_dim = dim / 2;
    for k in 0..half_dim {
        let freq = base.powf(-2.0 * k as f32 / dim as f32);
        let theta = freq * position as f32;
        let cos_t = theta.cos();
        let sin_t = theta.sin();
        let x0 = x[2 * k];
        let x1 = x[2 * k + 1];
        output[2 * k] = x0 * cos_t - x1 * sin_t;
        output[2 * k + 1] = x0 * sin_t + x1 * cos_t;
    }
}

// ────────────────────────────────────────────────────────────────────────────
// AVX2 implementation
// ────────────────────────────────────────────────────────────────────────────

/// AVX2 RoPE — delegates to scalar (no hardware `sin`/`cos` in AVX2).
///
/// # Safety
/// Requires AVX2 support. Caller must verify with `is_x86_feature_detected!("avx2")`.
///
/// # Panics
/// Panics if `x.len() != dim`, `x.len() != output.len()`, `dim` is odd, or `dim` is zero.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn rope_avx2(x: &[f32], position: u32, dim: usize, base: f32, output: &mut [f32]) {
    rope_scalar(x, position, dim, base, output);
}

// ────────────────────────────────────────────────────────────────────────────
// PTX implementation
// ────────────────────────────────────────────────────────────────────────────

/// PTX assembly for the RoPE kernel (1 thread per dimension pair).
///
/// Each thread handles one pair (2k, 2k+1):
/// - Computes angle = position * base^(-2k/dim) using `lg2.approx.f32` and `ex2.approx.f32`
/// - Applies rotation via `sin.approx.f32` and `cos.approx.f32`
pub fn rope_ptx() -> &'static str {
    r#".version 8.5
.target sm_90
.address_size 64
.visible .entry rope_kernel(
    .param .u64 input,
    .param .u64 output,
    .param .u32 position,
    .param .u32 dim,
    .param .f32 base
) {
    .reg .u32 %tid, %ntid, %ctaid, %idx, %half_dim, %dim, %pos;
    .reg .u32 %idx2, %idx2p1;
    .reg .u64 %in_ptr, %out_ptr, %off0, %off1;
    .reg .f32 %x0, %x1, %y0, %y1;
    .reg .f32 %k_f, %dim_f, %neg_exp, %freq, %pos_f, %theta;
    .reg .f32 %cos_t, %sin_t;
    .reg .f32 %base_val, %log_base, %k_two, %k_ln2, %k_rcp_ln2;
    .reg .pred %p;

    mov.u32 %tid, %tid.x;
    mov.u32 %ntid, %ntid.x;
    mov.u32 %ctaid, %ctaid.x;
    mad.lo.u32 %idx, %ctaid, %ntid, %tid;

    ld.param.u32 %dim, [dim];
    shr.u32 %half_dim, %dim, 1;
    setp.ge.u32 %p, %idx, %half_dim;
    @%p bra DONE;

    ld.param.u64 %in_ptr, [input];
    ld.param.u64 %out_ptr, [output];
    ld.param.u32 %pos, [position];
    ld.param.f32 %base_val, [base];

    // Constants
    mov.f32 %k_two, 0f40000000;       // 2.0
    mov.f32 %k_ln2, 0f3F317218;       // ln(2) ~ 0.693147
    mov.f32 %k_rcp_ln2, 0f3FB8AA3B;   // 1/ln(2) ~ 1.442695

    // Compute freq = base^(-2k/dim) using exp2(log2(base) * (-2k/dim))
    cvt.rn.f32.u32 %k_f, %idx;
    cvt.rn.f32.u32 %dim_f, %dim;
    mul.f32 %neg_exp, %k_two, %k_f;
    neg.f32 %neg_exp, %neg_exp;
    div.approx.f32 %neg_exp, %neg_exp, %dim_f;
    lg2.approx.f32 %log_base, %base_val;
    mul.f32 %neg_exp, %log_base, %neg_exp;
    ex2.approx.f32 %freq, %neg_exp;

    // theta = freq * position
    cvt.rn.f32.u32 %pos_f, %pos;
    mul.f32 %theta, %freq, %pos_f;

    // Compute cos and sin
    cos.approx.f32 %cos_t, %theta;
    sin.approx.f32 %sin_t, %theta;

    // Load x[2k] and x[2k+1]
    shl.b32 %idx2, %idx, 1;
    add.u32 %idx2p1, %idx2, 1;
    mul.wide.u32 %off0, %idx2, 4;
    mul.wide.u32 %off1, %idx2p1, 4;
    add.u64 %off0, %in_ptr, %off0;
    add.u64 %off1, %in_ptr, %off1;
    ld.global.f32 %x0, [%off0];
    ld.global.f32 %x1, [%off1];

    // Apply rotation:
    //   y0 = x0 * cos - x1 * sin
    //   y1 = x0 * sin + x1 * cos
    mul.f32 %y0, %x0, %cos_t;
    fma.rn.f32 %y0, %x1, %sin_t, %y0;
    neg.f32 %y0, %y0;
    fma.rn.f32 %y0, %x0, %cos_t, 0f00000000;
    mul.f32 %y0, %x1, %sin_t;
    neg.f32 %y0, %y0;
    fma.rn.f32 %y0, %x0, %cos_t, %y0;

    mul.f32 %y1, %x0, %sin_t;
    fma.rn.f32 %y1, %x1, %cos_t, %y1;

    // Store output[2k] and output[2k+1]
    mul.wide.u32 %off0, %idx2, 4;
    mul.wide.u32 %off1, %idx2p1, 4;
    add.u64 %off0, %out_ptr, %off0;
    add.u64 %off1, %out_ptr, %off1;
    st.global.f32 [%off0], %y0;
    st.global.f32 [%off1], %y1;

DONE:
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

    // ── Known-answer tests ────────────────────────────────────────────────

    #[test]
    fn test_rope_position_zero_identity() {
        // At position 0, all angles are 0: cos(0)=1, sin(0)=0 -> identity
        let x = [1.0f32, 2.0, 3.0, 4.0];
        let mut output = vec![0.0f32; 4];
        rope_scalar(&x, 0, 4, 10000.0, &mut output);
        for i in 0..4 {
            assert!(
                (output[i] - x[i]).abs() < 1e-6,
                "RoPE at position 0 should be identity: x[{i}]={}, output[{i}]={}",
                x[i],
                output[i]
            );
        }
    }

    #[test]
    fn test_rope_preserves_norm() {
        // Rotation preserves vector norm: |output| = |input|
        let x = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut output = vec![0.0f32; 6];
        rope_scalar(&x, 42, 6, 10000.0, &mut output);

        let input_norm: f32 = x.iter().map(|&v| v * v).sum::<f32>().sqrt();
        let output_norm: f32 = output.iter().map(|&v| v * v).sum::<f32>().sqrt();

        assert!(
            (input_norm - output_norm).abs() < 1e-4,
            "RoPE should preserve norm: input={input_norm}, output={output_norm}"
        );
    }

    #[test]
    fn test_rope_pair_norm_preserved() {
        // Each pair (x[2k], x[2k+1]) should have its norm preserved independently
        let x = [3.0f32, 4.0, 1.0, 0.0];
        let mut output = vec![0.0f32; 4];
        rope_scalar(&x, 10, 4, 10000.0, &mut output);

        let pair0_in = (x[0] * x[0] + x[1] * x[1]).sqrt();
        let pair0_out = (output[0] * output[0] + output[1] * output[1]).sqrt();
        assert!(
            (pair0_in - pair0_out).abs() < 1e-5,
            "Pair 0 norm not preserved: in={pair0_in}, out={pair0_out}"
        );

        let pair1_in = (x[2] * x[2] + x[3] * x[3]).sqrt();
        let pair1_out = (output[2] * output[2] + output[3] * output[3]).sqrt();
        assert!(
            (pair1_in - pair1_out).abs() < 1e-5,
            "Pair 1 norm not preserved: in={pair1_in}, out={pair1_out}"
        );
    }

    #[test]
    fn test_rope_known_rotation() {
        // For dim=2 with base=1.0, theta = 1.0^0 * pos = pos
        // At position 1: theta = 1.0, cos(1) ~ 0.5403, sin(1) ~ 0.8415
        let x = [1.0f32, 0.0];
        let mut output = vec![0.0f32; 2];
        rope_scalar(&x, 1, 2, 1.0, &mut output);
        let cos1 = 1.0f32.cos();
        let sin1 = 1.0f32.sin();
        assert!(
            (output[0] - cos1).abs() < 1e-6,
            "RoPE(1,0) at pos=1: expected ({cos1}, {sin1}), got ({}, {})",
            output[0],
            output[1]
        );
        assert!(
            (output[1] - sin1).abs() < 1e-6,
            "RoPE(1,0) at pos=1: expected ({cos1}, {sin1}), got ({}, {})",
            output[0],
            output[1]
        );
    }

    #[test]
    fn test_rope_default_base() {
        // Standard transformer base = 10000
        let x = [1.0f32, 0.0, 0.0, 1.0];
        let mut output = vec![0.0f32; 4];
        rope_scalar(&x, 100, 4, 10000.0, &mut output);

        // theta_0 = 10000^(0/4) * 100 = 1 * 100 = 100
        // theta_1 = 10000^(-2/4) * 100 = 10000^(-0.5) * 100 = 0.01 * 100 = 1.0
        let theta0 = 100.0f32;
        let theta1 = 10000.0f32.powf(-0.5) * 100.0;

        let expected_0 = theta0.cos();
        let expected_1 = theta0.sin();
        assert!(
            (output[0] - expected_0).abs() < 1e-4,
            "pair 0: expected cos({theta0})={expected_0}, got {}",
            output[0]
        );
        assert!(
            (output[1] - expected_1).abs() < 1e-4,
            "pair 0: expected sin({theta0})={expected_1}, got {}",
            output[1]
        );

        let expected_2 = -(theta1.sin());
        let expected_3 = theta1.cos();
        assert!(
            (output[2] - expected_2).abs() < 1e-4,
            "pair 1: expected -sin({theta1})={expected_2}, got {}",
            output[2]
        );
        assert!(
            (output[3] - expected_3).abs() < 1e-4,
            "pair 1: expected cos({theta1})={expected_3}, got {}",
            output[3]
        );
    }

    #[test]
    #[should_panic(expected = "dim must be even")]
    fn test_rope_odd_dim_panics() {
        let x = [1.0f32, 2.0, 3.0];
        let mut output = vec![0.0f32; 3];
        rope_scalar(&x, 1, 3, 10000.0, &mut output);
    }

    #[test]
    #[should_panic(expected = "x length must equal dim")]
    fn test_rope_length_mismatch() {
        let x = [1.0f32, 2.0];
        let mut output = vec![0.0f32; 2];
        rope_scalar(&x, 1, 4, 10000.0, &mut output);
    }

    #[test]
    #[should_panic(expected = "x/output length mismatch")]
    fn test_rope_output_length_mismatch() {
        let x = [1.0f32, 2.0, 3.0, 4.0];
        let mut output = vec![0.0f32; 6];
        rope_scalar(&x, 1, 4, 10000.0, &mut output);
    }

    #[test]
    #[should_panic(expected = "dim must be positive")]
    fn test_rope_zero_dim_panics() {
        let x: [f32; 0] = [];
        let mut output: [f32; 0] = [];
        rope_scalar(&x, 1, 0, 10000.0, &mut output);
    }

    // ── Property-based tests ──────────────────────────────────────────────

    proptest! {
        #[test]
        fn prop_rope_preserves_norm(
            x in proptest::collection::vec(-10.0f32..10.0, 1..16usize)
                .prop_filter("even length", |v| v.len() % 2 == 0 && !v.is_empty()),
            position in 0u32..1000,
        ) {
            let dim = x.len();
            let mut output = vec![0.0f32; dim];
            rope_scalar(&x, position, dim, 10000.0, &mut output);

            let input_norm: f32 = x.iter().map(|&v| v * v).sum::<f32>().sqrt();
            let output_norm: f32 = output.iter().map(|&v| v * v).sum::<f32>().sqrt();

            prop_assert!(
                (input_norm - output_norm).abs() < 1e-3,
                "Norm not preserved: input={input_norm}, output={output_norm}"
            );
        }

        #[test]
        fn prop_rope_position_zero_identity(
            x in proptest::collection::vec(-10.0f32..10.0, 1..16usize)
                .prop_filter("even length", |v| v.len() % 2 == 0 && !v.is_empty()),
        ) {
            let dim = x.len();
            let mut output = vec![0.0f32; dim];
            rope_scalar(&x, 0, dim, 10000.0, &mut output);

            for (i, (&xi, &yi)) in x.iter().zip(output.iter()).enumerate() {
                prop_assert!(
                    (xi - yi).abs() < 1e-6,
                    "RoPE at position 0 should be identity: index {i}, x={xi}, output={yi}"
                );
            }
        }

        #[test]
        fn prop_rope_output_finite(
            x in proptest::collection::vec(-100.0f32..100.0, 1..16usize)
                .prop_filter("even length", |v| v.len() % 2 == 0 && !v.is_empty()),
            position in 0u32..10000,
        ) {
            let dim = x.len();
            let mut output = vec![0.0f32; dim];
            rope_scalar(&x, position, dim, 10000.0, &mut output);

            for (i, &y) in output.iter().enumerate() {
                prop_assert!(
                    y.is_finite(),
                    "RoPE output must be finite at index {i}, got {y}"
                );
            }
        }
    }

    // ── AVX2 parity tests ─────────────────────────────────────────────────

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_rope_avx2_parity() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let x: Vec<f32> = (0..16).map(|i| i as f32 * 0.5).collect();
        let mut scalar_out = vec![0.0f32; x.len()];
        let mut avx2_out = vec![0.0f32; x.len()];

        rope_scalar(&x, 42, 16, 10000.0, &mut scalar_out);
        unsafe { rope_avx2(&x, 42, 16, 10000.0, &mut avx2_out) };

        // Delegates to scalar, so 0 ULP expected
        assert_ulp_eq(&scalar_out, &avx2_out, 0);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_rope_avx2_small_dim() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let x = [1.0f32, 2.0];
        let mut scalar_out = vec![0.0f32; 2];
        let mut avx2_out = vec![0.0f32; 2];

        rope_scalar(&x, 100, 2, 10000.0, &mut scalar_out);
        unsafe { rope_avx2(&x, 100, 2, 10000.0, &mut avx2_out) };

        assert_ulp_eq(&scalar_out, &avx2_out, 0);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_rope_avx2_position_zero() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let x: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let mut scalar_out = vec![0.0f32; 8];
        let mut avx2_out = vec![0.0f32; 8];

        rope_scalar(&x, 0, 8, 10000.0, &mut scalar_out);
        unsafe { rope_avx2(&x, 0, 8, 10000.0, &mut avx2_out) };

        assert_ulp_eq(&scalar_out, &avx2_out, 0);
    }

    // ── PTX structural tests ──────────────────────────────────────────────

    #[test]
    fn test_rope_ptx_structure() {
        let ptx = rope_ptx();
        assert!(ptx.contains(".version 8.5"), "missing PTX version");
        assert!(ptx.contains(".target sm_90"), "missing PTX target");
        assert!(ptx.contains(".entry rope_kernel"), "missing entry point");
        assert!(ptx.contains("ret;"), "missing ret instruction");
        assert!(
            ptx.contains("sin.approx.f32"),
            "missing sin.approx for trig"
        );
        assert!(
            ptx.contains("cos.approx.f32"),
            "missing cos.approx for trig"
        );
        assert!(
            ptx.contains("ex2.approx.f32"),
            "missing ex2.approx for powf"
        );
        assert!(
            ptx.contains("lg2.approx.f32"),
            "missing lg2.approx for powf"
        );
        let open = ptx.matches('{').count();
        let close = ptx.matches('}').count();
        assert_eq!(
            open, close,
            "unbalanced braces: {open} open vs {close} close"
        );
    }

    #[test]
    fn test_rope_ptx_nonempty() {
        assert!(!rope_ptx().is_empty());
    }

    #[test]
    fn test_rope_ptx_has_params() {
        let ptx = rope_ptx();
        assert!(ptx.contains(".param .u64 input"), "missing input param");
        assert!(ptx.contains(".param .u64 output"), "missing output param");
        assert!(
            ptx.contains(".param .u32 position"),
            "missing position param"
        );
        assert!(ptx.contains(".param .u32 dim"), "missing dim param");
        assert!(ptx.contains(".param .f32 base"), "missing base param");
    }
}
