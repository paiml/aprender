//! Standalone `SiLU` kernel with explicit sigmoid.
//!
//! Matches `silu-kernel-v1.yaml`.
//!
//! Each function provides one of three backends:
//! - `fn {name}_scalar(...)` — Pure Rust scalar reference (ground truth)
//! - `unsafe fn {name}_avx2(...)` — AVX2 SIMD implementation
//! - `fn {name}_ptx() -> &'static str` — PTX assembly source string

// ────────────────────────────────────────────────────────────────────────────
// Scalar implementations
// ────────────────────────────────────────────────────────────────────────────

/// Sigmoid: sigma(x) = 1 / (1 + exp(-x))
///
/// # Panics
/// Panics if `input.len() != output.len()`.
pub fn sigmoid_scalar(input: &[f32], output: &mut [f32]) {
    assert_eq!(input.len(), output.len());
    for (x, y) in input.iter().zip(output.iter_mut()) {
        *y = 1.0 / (1.0 + (-x).exp());
    }
}

/// Standalone `SiLU`: x * sigma(x) = x / (1 + exp(-x))
///
/// This computes `SiLU` by explicitly multiplying `x` with `sigmoid(x)`,
/// making the sigmoid intermediate available for inspection and testing.
///
/// # Panics
/// Panics if `input.len() != output.len()`.
pub fn silu_standalone_scalar(input: &[f32], output: &mut [f32]) {
    assert_eq!(input.len(), output.len());
    let mut sigma = vec![0.0f32; input.len()];
    sigmoid_scalar(input, &mut sigma);
    for i in 0..input.len() {
        output[i] = input[i] * sigma[i];
    }
}

// ────────────────────────────────────────────────────────────────────────────
// AVX2 implementations
// ────────────────────────────────────────────────────────────────────────────

/// AVX2 sigmoid — delegates to scalar (no hardware `exp` in AVX2).
///
/// # Safety
/// Requires AVX2 support. Caller must verify with `is_x86_feature_detected!("avx2")`.
///
/// # Panics
/// Panics if `input.len() != output.len()`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn sigmoid_avx2(input: &[f32], output: &mut [f32]) {
    sigmoid_scalar(input, output);
}

/// AVX2 standalone `SiLU` — delegates to scalar (no hardware `exp` in AVX2).
///
/// # Safety
/// Requires AVX2 support. Caller must verify with `is_x86_feature_detected!("avx2")`.
///
/// # Panics
/// Panics if `input.len() != output.len()`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn silu_standalone_avx2(input: &[f32], output: &mut [f32]) {
    silu_standalone_scalar(input, output);
}

// ────────────────────────────────────────────────────────────────────────────
// PTX implementations
// ────────────────────────────────────────────────────────────────────────────

/// PTX assembly for the sigmoid kernel.
///
/// sigma(x) = 1 / (1 + exp(-x)), where exp(-x) = 2^(-x / ln2) via `ex2.approx.f32`.
pub fn sigmoid_ptx() -> &'static str {
    r".version 8.5
.target sm_90
.address_size 64
.visible .entry sigmoid_kernel(
    .param .u64 input,
    .param .u64 output,
    .param .u32 n
) {
    .reg .u32 %tid, %ntid, %ctaid, %idx, %n;
    .reg .u64 %in_ptr, %out_ptr, %off;
    .reg .f32 %x, %y, %neg_x, %scaled, %exp_val, %denom, %rcp_denom;
    .reg .f32 %k_one, %k_rcp_ln2;
    .reg .pred %p;

    mov.u32 %tid, %tid.x;
    mov.u32 %ntid, %ntid.x;
    mov.u32 %ctaid, %ctaid.x;
    mad.lo.u32 %idx, %ctaid, %ntid, %tid;
    ld.param.u32 %n, [n];
    setp.ge.u32 %p, %idx, %n;
    @%p bra DONE;

    ld.param.u64 %in_ptr, [input];
    ld.param.u64 %out_ptr, [output];
    mul.wide.u32 %off, %idx, 4;
    add.u64 %in_ptr, %in_ptr, %off;
    add.u64 %out_ptr, %out_ptr, %off;
    ld.global.f32 %x, [%in_ptr];

    // Constants
    mov.f32 %k_one, 0f3F800000;       // 1.0
    mov.f32 %k_rcp_ln2, 0f3FB8AA3B;   // 1/ln(2) ~ 1.442695

    // exp(-x) = 2^(-x * (1/ln2))
    neg.f32 %neg_x, %x;
    mul.f32 %scaled, %neg_x, %k_rcp_ln2;
    ex2.approx.f32 %exp_val, %scaled;

    // sigmoid = 1 / (1 + exp(-x))
    add.f32 %denom, %k_one, %exp_val;
    rcp.approx.f32 %rcp_denom, %denom;
    mov.f32 %y, %rcp_denom;

    st.global.f32 [%out_ptr], %y;

DONE:
    ret;
}
"
}

/// PTX assembly for the standalone `SiLU` kernel.
///
/// `SiLU`(x) = x * sigma(x) = x / (1 + exp(-x)), with exp via `ex2.approx.f32`.
pub fn silu_standalone_ptx() -> &'static str {
    r".version 8.5
.target sm_90
.address_size 64
.visible .entry silu_standalone_kernel(
    .param .u64 input,
    .param .u64 output,
    .param .u32 n
) {
    .reg .u32 %tid, %ntid, %ctaid, %idx, %n;
    .reg .u64 %in_ptr, %out_ptr, %off;
    .reg .f32 %x, %y, %neg_x, %scaled, %exp_val, %denom, %rcp_denom;
    .reg .f32 %k_one, %k_rcp_ln2;
    .reg .pred %p;

    mov.u32 %tid, %tid.x;
    mov.u32 %ntid, %ntid.x;
    mov.u32 %ctaid, %ctaid.x;
    mad.lo.u32 %idx, %ctaid, %ntid, %tid;
    ld.param.u32 %n, [n];
    setp.ge.u32 %p, %idx, %n;
    @%p bra DONE;

    ld.param.u64 %in_ptr, [input];
    ld.param.u64 %out_ptr, [output];
    mul.wide.u32 %off, %idx, 4;
    add.u64 %in_ptr, %in_ptr, %off;
    add.u64 %out_ptr, %out_ptr, %off;
    ld.global.f32 %x, [%in_ptr];

    // Constants
    mov.f32 %k_one, 0f3F800000;       // 1.0
    mov.f32 %k_rcp_ln2, 0f3FB8AA3B;   // 1/ln(2) ~ 1.442695

    // exp(-x) = 2^(-x * (1/ln2))
    neg.f32 %neg_x, %x;
    mul.f32 %scaled, %neg_x, %k_rcp_ln2;
    ex2.approx.f32 %exp_val, %scaled;

    // silu = x / (1 + exp(-x)) = x * rcp(1 + exp(-x))
    add.f32 %denom, %k_one, %exp_val;
    rcp.approx.f32 %rcp_denom, %denom;
    mul.f32 %y, %x, %rcp_denom;

    st.global.f32 [%out_ptr], %y;

DONE:
    ret;
}
"
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::super::ulp::assert_ulp_eq;
    use super::*;
    use proptest::prelude::*;

    // ── Sigmoid known-answer tests ───────────────────────────────────────

    #[test]
    fn test_sigmoid_zero() {
        let input = [0.0f32];
        let mut output = [0.0f32];
        sigmoid_scalar(&input, &mut output);
        assert!(
            (output[0] - 0.5).abs() < 1e-7,
            "sigmoid(0) should be 0.5, got {}",
            output[0]
        );
    }

    #[test]
    fn test_sigmoid_large_positive() {
        let input = [20.0f32];
        let mut output = [0.0f32];
        sigmoid_scalar(&input, &mut output);
        assert!(
            (output[0] - 1.0).abs() < 1e-6,
            "sigmoid(20) should be ~1.0, got {}",
            output[0]
        );
    }

    #[test]
    fn test_sigmoid_large_negative() {
        let input = [-20.0f32];
        let mut output = [0.0f32];
        sigmoid_scalar(&input, &mut output);
        assert!(
            output[0].abs() < 1e-6,
            "sigmoid(-20) should be ~0.0, got {}",
            output[0]
        );
    }

    #[test]
    fn test_sigmoid_symmetry() {
        // sigmoid(-x) = 1 - sigmoid(x)
        let input_pos = [2.0f32];
        let input_neg = [-2.0f32];
        let mut out_pos = [0.0f32];
        let mut out_neg = [0.0f32];
        sigmoid_scalar(&input_pos, &mut out_pos);
        sigmoid_scalar(&input_neg, &mut out_neg);
        assert!(
            (out_pos[0] + out_neg[0] - 1.0).abs() < 1e-6,
            "sigmoid(x) + sigmoid(-x) should be 1.0, got {} + {} = {}",
            out_pos[0],
            out_neg[0],
            out_pos[0] + out_neg[0]
        );
    }

    // ── SiLU standalone known-answer tests ───────────────────────────────

    #[test]
    fn test_silu_standalone_zero() {
        let input = [0.0f32];
        let mut output = [0.0f32];
        silu_standalone_scalar(&input, &mut output);
        assert!(
            output[0].abs() < 1e-7,
            "SiLU(0) should be 0, got {}",
            output[0]
        );
    }

    #[test]
    fn test_silu_standalone_positive() {
        let input = [1.0f32];
        let mut output = [0.0f32];
        silu_standalone_scalar(&input, &mut output);
        let expected = 1.0 / (1.0 + (-1.0f32).exp());
        assert!(
            (output[0] - expected).abs() < 1e-6,
            "SiLU(1) should be ~{expected}, got {}",
            output[0]
        );
    }

    #[test]
    fn test_silu_standalone_negative() {
        let input = [-1.0f32];
        let mut output = [0.0f32];
        silu_standalone_scalar(&input, &mut output);
        let expected = -1.0 / (1.0 + 1.0f32.exp());
        assert!(
            (output[0] - expected).abs() < 1e-6,
            "SiLU(-1) should be ~{expected}, got {}",
            output[0]
        );
    }

    #[test]
    fn test_silu_standalone_matches_direct() {
        // Verify silu_standalone produces the same result as x/(1+exp(-x)) directly
        let input: Vec<f32> = (-10..=10).map(|i| i as f32 * 0.5).collect();
        let mut output = vec![0.0f32; input.len()];
        silu_standalone_scalar(&input, &mut output);
        for (x, y) in input.iter().zip(output.iter()) {
            let expected = x / (1.0 + (-x).exp());
            assert!(
                (y - expected).abs() < 1e-6,
                "SiLU({x}) mismatch: standalone={y}, direct={expected}"
            );
        }
    }

    // ── Property-based tests ─────────────────────────────────────────────

    proptest! {
        #[test]
        fn prop_sigmoid_in_unit_interval(x in proptest::num::f32::NORMAL) {
            let input = [x];
            let mut output = [0.0f32];
            sigmoid_scalar(&input, &mut output);
            prop_assert!(
                (0.0..=1.0).contains(&output[0]),
                "sigmoid({x}) = {} not in [0,1]",
                output[0]
            );
        }

        #[test]
        fn prop_sigmoid_monotonic(
            a in proptest::num::f32::NORMAL,
            b in proptest::num::f32::NORMAL,
        ) {
            let mut out_a = [0.0f32];
            let mut out_b = [0.0f32];
            sigmoid_scalar(&[a], &mut out_a);
            sigmoid_scalar(&[b], &mut out_b);
            if a < b {
                prop_assert!(
                    out_a[0] <= out_b[0],
                    "sigmoid should be monotonic: sigmoid({a})={} > sigmoid({b})={}",
                    out_a[0],
                    out_b[0]
                );
            }
        }

        #[test]
        fn prop_silu_standalone_sign_preserving(x in proptest::num::f32::NORMAL) {
            let input = [x];
            let mut output = [0.0f32];
            silu_standalone_scalar(&input, &mut output);
            if x > 0.0 {
                prop_assert!(output[0] >= 0.0, "SiLU({x}) should be >= 0, got {}", output[0]);
            } else if x < 0.0 {
                prop_assert!(output[0] <= 0.0, "SiLU({x}) should be <= 0, got {}", output[0]);
            }
        }
    }

    // ── AVX2 parity tests ────────────────────────────────────────────────

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_sigmoid_avx2_parity() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let input: Vec<f32> = (-20..20).map(|i| i as f32 * 0.5).collect();
        let mut scalar_out = vec![0.0f32; input.len()];
        let mut avx2_out = vec![0.0f32; input.len()];

        sigmoid_scalar(&input, &mut scalar_out);
        unsafe { sigmoid_avx2(&input, &mut avx2_out) };

        assert_ulp_eq(&scalar_out, &avx2_out, 2);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_silu_standalone_avx2_parity() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let input: Vec<f32> = (-20..20).map(|i| i as f32 * 0.3).collect();
        let mut scalar_out = vec![0.0f32; input.len()];
        let mut avx2_out = vec![0.0f32; input.len()];

        silu_standalone_scalar(&input, &mut scalar_out);
        unsafe { silu_standalone_avx2(&input, &mut avx2_out) };

        assert_ulp_eq(&scalar_out, &avx2_out, 2);
    }

    // ── PTX structural tests ─────────────────────────────────────────────

    #[test]
    fn test_sigmoid_ptx_structure() {
        let ptx = sigmoid_ptx();
        assert!(ptx.contains(".version 8.5"), "missing PTX version");
        assert!(ptx.contains(".target sm_90"), "missing PTX target");
        assert!(ptx.contains(".entry sigmoid_kernel"), "missing entry point");
        assert!(ptx.contains("ret;"), "missing ret instruction");
        assert!(ptx.contains("ex2.approx.f32"), "missing ex2.approx for exp");
        let open = ptx.matches('{').count();
        let close = ptx.matches('}').count();
        assert_eq!(
            open, close,
            "unbalanced braces: {open} open vs {close} close"
        );
    }

    #[test]
    fn test_silu_standalone_ptx_structure() {
        let ptx = silu_standalone_ptx();
        assert!(ptx.contains(".version 8.5"), "missing PTX version");
        assert!(ptx.contains(".target sm_90"), "missing PTX target");
        assert!(
            ptx.contains(".entry silu_standalone_kernel"),
            "missing entry point"
        );
        assert!(ptx.contains("ret;"), "missing ret instruction");
        assert!(ptx.contains("ex2.approx.f32"), "missing ex2.approx for exp");
        let open = ptx.matches('{').count();
        let close = ptx.matches('}').count();
        assert_eq!(
            open, close,
            "unbalanced braces: {open} open vs {close} close"
        );
    }

    #[test]
    fn test_ptx_kernels_are_nonempty() {
        assert!(!sigmoid_ptx().is_empty());
        assert!(!silu_standalone_ptx().is_empty());
    }
}
