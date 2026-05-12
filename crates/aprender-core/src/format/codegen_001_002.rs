// `codegen-dispatch-v1` algorithm-level PARTIAL discharge for the 2
// runtime-SIMD-dispatch falsifiers (scalar fallback safety, dispatch
// determinism).
//
// Contract: `contracts/codegen-dispatch-v1.yaml`.

/// Tolerance for "scalar matches reference" check (8 ULP — same as
/// avx2-fma-dot-v1).
pub const AC_CODEGEN_FALLBACK_TOLERANCE: f32 = 1.0e-5;

// =============================================================================
// FALSIFY-CODEGEN_DISPATCH_V1_001 — scalar fallback safety
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CodegenFallbackVerdict {
    /// Scalar fallback output matches reference within tolerance for
    /// every input in the test corpus.
    Pass,
    /// Scalar diverges from reference on at least one input.
    Fail,
}

#[must_use]
pub fn verdict_from_codegen_fallback(scalar_outputs: &[f32], reference_outputs: &[f32]) -> CodegenFallbackVerdict {
    if scalar_outputs.len() != reference_outputs.len() {
        return CodegenFallbackVerdict::Fail;
    }
    if scalar_outputs.is_empty() {
        return CodegenFallbackVerdict::Fail;
    }
    for (a, b) in scalar_outputs.iter().zip(reference_outputs.iter()) {
        if !a.is_finite() || !b.is_finite() {
            return CodegenFallbackVerdict::Fail;
        }
        if (a - b).abs() >= AC_CODEGEN_FALLBACK_TOLERANCE {
            return CodegenFallbackVerdict::Fail;
        }
    }
    CodegenFallbackVerdict::Pass
}

// =============================================================================
// FALSIFY-CODEGEN_DISPATCH_V1_002 — dispatch determinism
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CodegenDispatchDeterminismVerdict {
    /// Same hardware fingerprint ⇒ same kernel selected across N calls.
    Pass,
    /// Two calls on identical hardware selected different kernels.
    Fail,
}

#[must_use]
pub fn verdict_from_codegen_dispatch_determinism(selected_kernels: &[&str]) -> CodegenDispatchDeterminismVerdict {
    if selected_kernels.is_empty() {
        return CodegenDispatchDeterminismVerdict::Fail;
    }
    let first = selected_kernels[0];
    for k in selected_kernels.iter().skip(1) {
        if *k != first {
            return CodegenDispatchDeterminismVerdict::Fail;
        }
    }
    CodegenDispatchDeterminismVerdict::Pass
}

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Section 1: Provenance pin.
    // -------------------------------------------------------------------------
    #[test]
    fn provenance_fallback_tolerance_1e_neg5() {
        assert!((AC_CODEGEN_FALLBACK_TOLERANCE - 1.0e-5).abs() < f32::EPSILON);
    }

    // -------------------------------------------------------------------------
    // Section 2: CODEGEN-001 scalar fallback safety.
    // -------------------------------------------------------------------------
    #[test]
    fn fcg001_pass_exact_match() {
        let v = vec![1.0_f32, 2.0, 3.0];
        assert_eq!(
            verdict_from_codegen_fallback(&v, &v),
            CodegenFallbackVerdict::Pass
        );
    }

    #[test]
    fn fcg001_pass_within_tolerance() {
        let scalar = vec![1.000001_f32];
        let reference = vec![1.0_f32];
        assert_eq!(
            verdict_from_codegen_fallback(&scalar, &reference),
            CodegenFallbackVerdict::Pass
        );
    }

    #[test]
    fn fcg001_fail_outside_tolerance() {
        let scalar = vec![1.5_f32];
        let reference = vec![1.0_f32];
        assert_eq!(
            verdict_from_codegen_fallback(&scalar, &reference),
            CodegenFallbackVerdict::Fail
        );
    }

    #[test]
    fn fcg001_fail_length_mismatch() {
        assert_eq!(
            verdict_from_codegen_fallback(&[1.0], &[1.0, 2.0]),
            CodegenFallbackVerdict::Fail
        );
    }

    #[test]
    fn fcg001_fail_nan() {
        assert_eq!(
            verdict_from_codegen_fallback(&[f32::NAN], &[1.0]),
            CodegenFallbackVerdict::Fail
        );
    }

    #[test]
    fn fcg001_fail_empty() {
        assert_eq!(
            verdict_from_codegen_fallback(&[], &[]),
            CodegenFallbackVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 3: CODEGEN-002 dispatch determinism.
    // -------------------------------------------------------------------------
    #[test]
    fn fcg002_pass_all_same_kernel() {
        let k = ["avx512", "avx512", "avx512", "avx512"];
        assert_eq!(
            verdict_from_codegen_dispatch_determinism(&k),
            CodegenDispatchDeterminismVerdict::Pass
        );
    }

    #[test]
    fn fcg002_pass_single_call() {
        let k = ["scalar"];
        assert_eq!(
            verdict_from_codegen_dispatch_determinism(&k),
            CodegenDispatchDeterminismVerdict::Pass
        );
    }

    #[test]
    fn fcg002_fail_kernel_changed() {
        let k = ["avx512", "avx2"];
        assert_eq!(
            verdict_from_codegen_dispatch_determinism(&k),
            CodegenDispatchDeterminismVerdict::Fail
        );
    }

    #[test]
    fn fcg002_fail_third_call_diverged() {
        let k = ["avx512", "avx512", "scalar"];
        assert_eq!(
            verdict_from_codegen_dispatch_determinism(&k),
            CodegenDispatchDeterminismVerdict::Fail
        );
    }

    #[test]
    fn fcg002_fail_empty() {
        let empty: [&str; 0] = [];
        assert_eq!(
            verdict_from_codegen_dispatch_determinism(&empty),
            CodegenDispatchDeterminismVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 4: Realistic — full healthy SIMD dispatch passes both.
    // -------------------------------------------------------------------------
    #[test]
    fn realistic_healthy_dispatch_passes_both() {
        // 1024-element output, scalar matches AVX-512 reference.
        let v: Vec<f32> = (0..1024).map(|i| i as f32 + 1e-7).collect();
        let r: Vec<f32> = (0..1024).map(|i| i as f32).collect();
        assert_eq!(
            verdict_from_codegen_fallback(&v, &r),
            CodegenFallbackVerdict::Pass
        );

        // 100 calls all selected avx2 (uniform hardware).
        let calls = vec!["avx2"; 100];
        assert_eq!(
            verdict_from_codegen_dispatch_determinism(&calls),
            CodegenDispatchDeterminismVerdict::Pass
        );
    }

    #[test]
    fn realistic_pre_fix_both_failures() {
        // Bug class 001: scalar produced different result for one element.
        let mut scalar: Vec<f32> = (0..1024).map(|i| i as f32).collect();
        scalar[512] += 0.5;
        let reference: Vec<f32> = (0..1024).map(|i| i as f32).collect();
        assert_eq!(
            verdict_from_codegen_fallback(&scalar, &reference),
            CodegenFallbackVerdict::Fail
        );

        // Bug class 002: kernel selection raced — call 50 picked scalar.
        let mut calls = vec!["avx2"; 100];
        calls[50] = "scalar";
        assert_eq!(
            verdict_from_codegen_dispatch_determinism(&calls),
            CodegenDispatchDeterminismVerdict::Fail
        );
    }
}
