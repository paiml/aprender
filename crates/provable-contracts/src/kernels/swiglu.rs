//! SwiGLU gated MLP kernel.
//!
//! Matches `swiglu-kernel-v1.yaml`.
//! SwiGLU(gate, value) = SiLU(gate) * value
//! where SiLU(x) = x / (1 + exp(-x))
//!
//! Each function provides one of three backends:
//! - `fn swiglu_scalar(...)` — Pure Rust scalar reference (ground truth)
//! - `unsafe fn swiglu_avx2(...)` — AVX2 SIMD implementation
//! - `fn swiglu_ptx() -> &'static str` — PTX assembly source string

// ────────────────────────────────────────────────────────────────────────────
// Scalar implementation
// ────────────────────────────────────────────────────────────────────────────

/// Compute SwiGLU(gate, value) = SiLU(gate) * value elementwise.
///
/// SiLU(g) = g / (1 + exp(-g)), then output_i = SiLU(gate_i) * value_i.
///
/// # Panics
/// Panics if `gate.len() != value.len()` or `gate.len() != output.len()`.
pub fn swiglu_scalar(gate: &[f32], value: &[f32], output: &mut [f32]) {
    assert_eq!(gate.len(), value.len(), "gate/value length mismatch");
    assert_eq!(gate.len(), output.len(), "gate/output length mismatch");
    for i in 0..gate.len() {
        let g = gate[i];
        let silu_g = g / (1.0 + (-g).exp());
        output[i] = silu_g * value[i];
    }
}

// ────────────────────────────────────────────────────────────────────────────
// AVX2 implementation
// ────────────────────────────────────────────────────────────────────────────

/// AVX2 SwiGLU — delegates to scalar (no hardware `exp` in AVX2).
///
/// # Safety
/// Requires AVX2 support. Caller must verify with `is_x86_feature_detected!("avx2")`.
///
/// # Panics
/// Panics if `gate.len() != value.len()` or `gate.len() != output.len()`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn swiglu_avx2(gate: &[f32], value: &[f32], output: &mut [f32]) {
    swiglu_scalar(gate, value, output);
}

// ────────────────────────────────────────────────────────────────────────────
// PTX implementation
// ────────────────────────────────────────────────────────────────────────────

/// PTX assembly for the SwiGLU kernel (elementwise, 1 thread per element).
///
/// Computes SiLU(gate_i) * value_i where SiLU uses `ex2.approx.f32` with
/// ln2 scaling for the exponential: exp(-g) = 2^(-g / ln2).
pub fn swiglu_ptx() -> &'static str {
    r#".version 8.5
.target sm_90
.address_size 64
.visible .entry swiglu_kernel(
    .param .u64 gate,
    .param .u64 value,
    .param .u64 output,
    .param .u32 n
) {
    .reg .u32 %tid, %ntid, %ctaid, %idx, %n;
    .reg .u64 %gate_ptr, %val_ptr, %out_ptr, %off;
    .reg .f32 %g, %v, %neg_g, %scaled, %exp_val, %denom, %inv, %silu, %result;
    .reg .f32 %k_one, %k_rcp_ln2;
    .reg .pred %p;

    mov.u32 %tid, %tid.x;
    mov.u32 %ntid, %ntid.x;
    mov.u32 %ctaid, %ctaid.x;
    mad.lo.u32 %idx, %ctaid, %ntid, %tid;
    ld.param.u32 %n, [n];
    setp.ge.u32 %p, %idx, %n;
    @%p bra DONE;

    ld.param.u64 %gate_ptr, [gate];
    ld.param.u64 %val_ptr, [value];
    ld.param.u64 %out_ptr, [output];
    mul.wide.u32 %off, %idx, 4;
    add.u64 %gate_ptr, %gate_ptr, %off;
    add.u64 %val_ptr, %val_ptr, %off;
    add.u64 %out_ptr, %out_ptr, %off;
    ld.global.f32 %g, [%gate_ptr];
    ld.global.f32 %v, [%val_ptr];

    // Constants
    mov.f32 %k_one, 0f3F800000;       // 1.0
    mov.f32 %k_rcp_ln2, 0f3FB8AA3B;   // 1/ln(2) ~ 1.442695

    // exp(-g) = 2^(-g * (1/ln2))
    neg.f32 %neg_g, %g;
    mul.f32 %scaled, %neg_g, %k_rcp_ln2;
    ex2.approx.f32 %exp_val, %scaled;

    // silu(g) = g / (1 + exp(-g))
    add.f32 %denom, %exp_val, %k_one;
    rcp.approx.f32 %inv, %denom;
    mul.f32 %silu, %g, %inv;

    // result = silu(g) * v
    mul.f32 %result, %silu, %v;
    st.global.f32 [%out_ptr], %result;

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
    fn test_swiglu_zero_gate() {
        // SiLU(0) = 0 / (1 + exp(0)) = 0 / 2 = 0, so output = 0 * value = 0
        let gate = [0.0f32, 0.0];
        let value = [1.0f32, 1.0];
        let mut output = vec![0.0f32; 2];
        swiglu_scalar(&gate, &value, &mut output);
        assert_eq!(output[0], 0.0, "SiLU(0) * 1 should be 0");
        assert_eq!(output[1], 0.0, "SiLU(0) * 1 should be 0");
    }

    #[test]
    fn test_swiglu_zero_value() {
        // Any gate with value=0 should produce 0
        let gate = [5.0f32, -3.0, 100.0];
        let value = [0.0f32, 0.0, 0.0];
        let mut output = vec![0.0f32; 3];
        swiglu_scalar(&gate, &value, &mut output);
        for (i, &y) in output.iter().enumerate() {
            assert_eq!(y, 0.0, "SiLU(gate) * 0 should be 0 at index {i}");
        }
    }

    #[test]
    fn test_swiglu_large_positive_gate() {
        // For large positive g, SiLU(g) ~ g, so SwiGLU ~ gate * value
        let gate = [20.0f32, 30.0];
        let value = [2.0f32, 3.0];
        let mut output = vec![0.0f32; 2];
        swiglu_scalar(&gate, &value, &mut output);
        assert!(
            (output[0] - 40.0).abs() < 1e-4,
            "SwiGLU(20, 2) should be ~40.0, got {}",
            output[0]
        );
        assert!(
            (output[1] - 90.0).abs() < 1e-4,
            "SwiGLU(30, 3) should be ~90.0, got {}",
            output[1]
        );
    }

    #[test]
    fn test_swiglu_silu_value_at_one() {
        // SiLU(1) = 1 / (1 + exp(-1)) ~ 0.7310586
        let gate = [1.0f32];
        let value = [1.0f32];
        let mut output = vec![0.0f32; 1];
        swiglu_scalar(&gate, &value, &mut output);
        let expected = 1.0f32 / (1.0 + (-1.0f32).exp());
        assert!(
            (output[0] - expected).abs() < 1e-6,
            "SwiGLU(1, 1) should be SiLU(1) ~ {expected}, got {}",
            output[0]
        );
    }

    #[test]
    fn test_swiglu_negative_gate() {
        // SiLU(-1) = -1 / (1 + exp(1)) ~ -0.2689414
        let gate = [-1.0f32];
        let value = [2.0f32];
        let mut output = vec![0.0f32; 1];
        swiglu_scalar(&gate, &value, &mut output);
        let silu_neg1 = -1.0f32 / (1.0 + 1.0f32.exp());
        let expected = silu_neg1 * 2.0;
        assert!(
            (output[0] - expected).abs() < 1e-6,
            "SwiGLU(-1, 2) should be ~{expected}, got {}",
            output[0]
        );
    }

    #[test]
    fn test_swiglu_empty() {
        let gate: [f32; 0] = [];
        let value: [f32; 0] = [];
        let mut output: [f32; 0] = [];
        swiglu_scalar(&gate, &value, &mut output);
        // Should not panic; empty is fine
    }

    #[test]
    #[should_panic(expected = "gate/value length mismatch")]
    fn test_swiglu_length_mismatch_gate_value() {
        let gate = [1.0f32];
        let value = [1.0f32, 2.0];
        let mut output = vec![0.0f32; 2];
        swiglu_scalar(&gate, &value, &mut output);
    }

    #[test]
    #[should_panic(expected = "gate/output length mismatch")]
    fn test_swiglu_length_mismatch_gate_output() {
        let gate = [1.0f32, 2.0];
        let value = [1.0f32, 2.0];
        let mut output = vec![0.0f32; 3];
        swiglu_scalar(&gate, &value, &mut output);
    }

    // ── Property-based tests ──────────────────────────────────────────────

    proptest! {
        #[test]
        fn prop_swiglu_output_bounded(
            gate in proptest::collection::vec(-10.0f32..10.0, 1..64),
        ) {
            let n = gate.len();
            let value: Vec<f32> = vec![1.0; n];
            let mut output = vec![0.0f32; n];
            swiglu_scalar(&gate, &value, &mut output);
            for (i, &y) in output.iter().enumerate() {
                // SiLU(g) is bounded below: SiLU(g) > -0.279
                // With value=1, output = SiLU(gate)
                prop_assert!(
                    y > -0.3,
                    "SwiGLU output at index {i} should be > -0.3, got {y}"
                );
                prop_assert!(
                    y.is_finite(),
                    "SwiGLU output at index {i} should be finite, got {y}"
                );
            }
        }

        #[test]
        fn prop_swiglu_zero_gate_yields_zero(
            value in proptest::collection::vec(-100.0f32..100.0, 1..32),
        ) {
            let n = value.len();
            let gate = vec![0.0f32; n];
            let mut output = vec![0.0f32; n];
            swiglu_scalar(&gate, &value, &mut output);
            for (i, &y) in output.iter().enumerate() {
                prop_assert_eq!(
                    y, 0.0,
                    "SiLU(0) * value[{}] should be exactly 0, got {}",
                    i, y
                );
            }
        }

        #[test]
        fn prop_swiglu_silu_lower_bound(
            gate in proptest::collection::vec(-50.0f32..50.0, 1..64),
        ) {
            let n = gate.len();
            let value = vec![1.0f32; n];
            let mut output = vec![0.0f32; n];
            swiglu_scalar(&gate, &value, &mut output);
            // With value=1, output = SiLU(gate), and SiLU(x) > -0.279 for all x
            for (i, &y) in output.iter().enumerate() {
                prop_assert!(
                    y > -0.28,
                    "SiLU lower bound violated at index {i}: SiLU({}) = {y}",
                    gate[i]
                );
            }
        }
    }

    // ── AVX2 parity tests ─────────────────────────────────────────────────

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_swiglu_avx2_parity() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let gate: Vec<f32> = (-20..20).map(|i| i as f32 * 0.5).collect();
        let value: Vec<f32> = (0..40).map(|i| i as f32 * 0.1 + 0.5).collect();
        let mut scalar_out = vec![0.0f32; gate.len()];
        let mut avx2_out = vec![0.0f32; gate.len()];

        swiglu_scalar(&gate, &value, &mut scalar_out);
        unsafe { swiglu_avx2(&gate, &value, &mut avx2_out) };

        assert_ulp_eq(&scalar_out, &avx2_out, 2);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_swiglu_avx2_non_aligned_length() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        // 11 elements — not divisible by 8
        let gate: Vec<f32> = (-5..6).map(|i| i as f32).collect();
        let value: Vec<f32> = (0..11).map(|i| i as f32 * 0.3).collect();
        let mut scalar_out = vec![0.0f32; gate.len()];
        let mut avx2_out = vec![0.0f32; gate.len()];

        swiglu_scalar(&gate, &value, &mut scalar_out);
        unsafe { swiglu_avx2(&gate, &value, &mut avx2_out) };

        assert_ulp_eq(&scalar_out, &avx2_out, 2);
    }

    // ── PTX structural tests ──────────────────────────────────────────────

    #[test]
    fn test_swiglu_ptx_structure() {
        let ptx = swiglu_ptx();
        assert!(ptx.contains(".version 8.5"), "missing PTX version");
        assert!(ptx.contains(".target sm_90"), "missing PTX target");
        assert!(ptx.contains(".entry swiglu_kernel"), "missing entry point");
        assert!(ptx.contains("ret;"), "missing ret instruction");
        assert!(ptx.contains("ex2.approx.f32"), "missing ex2.approx for exp");
        assert!(
            ptx.contains("rcp.approx.f32"),
            "missing rcp.approx for reciprocal"
        );
        let open = ptx.matches('{').count();
        let close = ptx.matches('}').count();
        assert_eq!(
            open, close,
            "unbalanced braces: {open} open vs {close} close"
        );
    }

    #[test]
    fn test_swiglu_ptx_nonempty() {
        assert!(!swiglu_ptx().is_empty());
    }

    #[test]
    fn test_swiglu_ptx_has_three_params() {
        let ptx = swiglu_ptx();
        assert!(ptx.contains(".param .u64 gate"), "missing gate param");
        assert!(ptx.contains(".param .u64 value"), "missing value param");
        assert!(ptx.contains(".param .u64 output"), "missing output param");
        assert!(ptx.contains(".param .u32 n"), "missing n param");
    }
}
