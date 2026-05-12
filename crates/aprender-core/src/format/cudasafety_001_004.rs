// `cuda-kernel-safety-v1` algorithm-level PARTIAL discharge for
// FALSIFY-CUDA-001..004.
//
// Contract: `contracts/cuda-kernel-safety-v1.yaml`.
//
// Pure Rust verdict for the 4 falsification gates:
//   CUDA-001: __global__ kernel generates extern "C" FFI
//   CUDA-002: host code (no CUDA qualifier) transpiles as normal Rust
//   CUDA-003: CUDA qualifier preserved through transformation chain
//   CUDA-004: __global__ keyword detected in inline source
//
// Each verdict is a pure decision rule on a fixture struct; no
// transpiler dependency.

/// Maximum number of transform stages a qualifier must survive
/// (parse → borrow_gen → array_slice → optimize → codegen).
pub const AC_CUDASAFETY_MIN_TRANSFORM_STAGES: usize = 5;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CudaSafetyVerdict {
    Pass,
    Fail,
}

/// CUDA-001: `__global__` kernel must generate `extern "C"` FFI.
///
/// Pass iff:
/// - input has `__global__` qualifier
/// - output contains `extern "C"` block
/// - output contains `fn <name>` matching kernel name
/// - all pointer params are `*mut T`
#[must_use]
#[allow(clippy::fn_params_excessive_bools)]
pub fn verdict_from_kernel_ffi(
    has_global_qualifier: bool,
    output_has_extern_c: bool,
    output_has_matching_fn_name: bool,
    pointer_params_are_raw_mut: bool,
) -> CudaSafetyVerdict {
    if has_global_qualifier
        && output_has_extern_c
        && output_has_matching_fn_name
        && pointer_params_are_raw_mut
    {
        CudaSafetyVerdict::Pass
    } else {
        CudaSafetyVerdict::Fail
    }
}

/// CUDA-002: host code must transpile as a normal Rust function.
///
/// Pass iff:
/// - no CUDA qualifier present (or only `__host__`)
/// - output is NOT an `extern "C"` block
/// - output is a regular `fn`
#[must_use]
pub fn verdict_from_host_transpilation(
    has_global_or_device: bool,
    output_is_extern_c: bool,
    output_is_regular_fn: bool,
) -> CudaSafetyVerdict {
    if !has_global_or_device && !output_is_extern_c && output_is_regular_fn {
        CudaSafetyVerdict::Pass
    } else {
        CudaSafetyVerdict::Fail
    }
}

/// CUDA-003: qualifier must survive every transform stage.
///
/// Pass iff:
/// - input had qualifier
/// - qualifier observed at every stage in `qualifier_at_stage`
/// - at least `AC_CUDASAFETY_MIN_TRANSFORM_STAGES` stages observed
#[must_use]
pub fn verdict_from_qualifier_preservation(
    input_has_qualifier: bool,
    qualifier_at_stage: &[bool],
) -> CudaSafetyVerdict {
    if !input_has_qualifier {
        return CudaSafetyVerdict::Fail;
    }
    if qualifier_at_stage.len() < AC_CUDASAFETY_MIN_TRANSFORM_STAGES {
        return CudaSafetyVerdict::Fail;
    }
    if qualifier_at_stage.iter().all(|&b| b) {
        CudaSafetyVerdict::Pass
    } else {
        CudaSafetyVerdict::Fail
    }
}

/// CUDA-004: parser must detect `__global__` keyword in inline source.
///
/// Pass iff:
/// - source contains `__global__` token
/// - parser flagged C++ mode
/// - empty macro defs were injected for keywords
#[must_use]
pub fn verdict_from_keyword_detection(
    source_has_global_token: bool,
    parser_in_cpp_mode: bool,
    empty_macros_injected: bool,
) -> CudaSafetyVerdict {
    if source_has_global_token && parser_in_cpp_mode && empty_macros_injected {
        CudaSafetyVerdict::Pass
    } else {
        CudaSafetyVerdict::Fail
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------
    // Section 1: Provenance pin.
    // -----------------------------------------------------------------
    #[test]
    fn provenance_min_transform_stages_is_5() {
        assert_eq!(AC_CUDASAFETY_MIN_TRANSFORM_STAGES, 5);
    }

    // -----------------------------------------------------------------
    // Section 2: CUDA-001 kernel_ffi.
    // -----------------------------------------------------------------
    #[test]
    fn fcuda001_pass_kernel_with_extern_c() {
        let v = verdict_from_kernel_ffi(true, true, true, true);
        assert_eq!(v, CudaSafetyVerdict::Pass);
    }

    #[test]
    fn fcuda001_fail_no_global_qualifier() {
        let v = verdict_from_kernel_ffi(false, true, true, true);
        assert_eq!(v, CudaSafetyVerdict::Fail);
    }

    #[test]
    fn fcuda001_fail_missing_extern_c() {
        let v = verdict_from_kernel_ffi(true, false, true, true);
        assert_eq!(v, CudaSafetyVerdict::Fail);
    }

    #[test]
    fn fcuda001_fail_fn_name_mismatch() {
        let v = verdict_from_kernel_ffi(true, true, false, true);
        assert_eq!(v, CudaSafetyVerdict::Fail);
    }

    #[test]
    fn fcuda001_fail_safe_pointers_not_raw() {
        let v = verdict_from_kernel_ffi(true, true, true, false);
        assert_eq!(v, CudaSafetyVerdict::Fail);
    }

    // -----------------------------------------------------------------
    // Section 3: CUDA-002 host_transpilation.
    // -----------------------------------------------------------------
    #[test]
    fn fcuda002_pass_host_function_normal_rust() {
        let v = verdict_from_host_transpilation(false, false, true);
        assert_eq!(v, CudaSafetyVerdict::Pass);
    }

    #[test]
    fn fcuda002_fail_global_treated_as_host() {
        let v = verdict_from_host_transpilation(true, false, true);
        assert_eq!(v, CudaSafetyVerdict::Fail);
    }

    #[test]
    fn fcuda002_fail_host_wrapped_extern_c() {
        let v = verdict_from_host_transpilation(false, true, false);
        assert_eq!(v, CudaSafetyVerdict::Fail);
    }

    #[test]
    fn fcuda002_fail_no_regular_fn() {
        let v = verdict_from_host_transpilation(false, false, false);
        assert_eq!(v, CudaSafetyVerdict::Fail);
    }

    // -----------------------------------------------------------------
    // Section 4: CUDA-003 qualifier_preservation.
    // -----------------------------------------------------------------
    #[test]
    fn fcuda003_pass_qualifier_survives_5_stages() {
        let v = verdict_from_qualifier_preservation(true, &[true; 5]);
        assert_eq!(v, CudaSafetyVerdict::Pass);
    }

    #[test]
    fn fcuda003_pass_qualifier_survives_more_stages() {
        let v = verdict_from_qualifier_preservation(true, &[true; 8]);
        assert_eq!(v, CudaSafetyVerdict::Pass);
    }

    #[test]
    fn fcuda003_fail_no_input_qualifier() {
        let v = verdict_from_qualifier_preservation(false, &[true; 5]);
        assert_eq!(v, CudaSafetyVerdict::Fail);
    }

    #[test]
    fn fcuda003_fail_qualifier_dropped_mid_chain() {
        let v = verdict_from_qualifier_preservation(true, &[true, true, false, true, true]);
        assert_eq!(v, CudaSafetyVerdict::Fail);
    }

    #[test]
    fn fcuda003_fail_too_few_stages() {
        let v = verdict_from_qualifier_preservation(true, &[true; 3]);
        assert_eq!(v, CudaSafetyVerdict::Fail);
    }

    #[test]
    fn fcuda003_fail_zero_stages() {
        let v = verdict_from_qualifier_preservation(true, &[]);
        assert_eq!(v, CudaSafetyVerdict::Fail);
    }

    // -----------------------------------------------------------------
    // Section 5: CUDA-004 keyword_detection.
    // -----------------------------------------------------------------
    #[test]
    fn fcuda004_pass_global_detected_cpp_mode_macros() {
        let v = verdict_from_keyword_detection(true, true, true);
        assert_eq!(v, CudaSafetyVerdict::Pass);
    }

    #[test]
    fn fcuda004_fail_no_global_token() {
        let v = verdict_from_keyword_detection(false, true, true);
        assert_eq!(v, CudaSafetyVerdict::Fail);
    }

    #[test]
    fn fcuda004_fail_parser_not_cpp_mode() {
        let v = verdict_from_keyword_detection(true, false, true);
        assert_eq!(v, CudaSafetyVerdict::Fail);
    }

    #[test]
    fn fcuda004_fail_empty_macros_not_injected() {
        let v = verdict_from_keyword_detection(true, true, false);
        assert_eq!(v, CudaSafetyVerdict::Fail);
    }

    // -----------------------------------------------------------------
    // Section 6: Mutation survey.
    // -----------------------------------------------------------------
    #[test]
    fn mutation_survey_001_only_when_all_4_inputs_true() {
        // Sweep every 4-bit subset; only 0b1111 should Pass.
        for mask in 0_u8..16 {
            let q = mask & 1 != 0;
            let e = mask & 2 != 0;
            let f = mask & 4 != 0;
            let p = mask & 8 != 0;
            let v = verdict_from_kernel_ffi(q, e, f, p);
            let expected = if q && e && f && p {
                CudaSafetyVerdict::Pass
            } else {
                CudaSafetyVerdict::Fail
            };
            assert_eq!(v, expected, "mask={mask:04b}");
        }
    }

    #[test]
    fn mutation_survey_002_pass_only_when_no_qual_no_extern_yes_fn() {
        for mask in 0_u8..8 {
            let g = mask & 1 != 0;
            let e = mask & 2 != 0;
            let f = mask & 4 != 0;
            let v = verdict_from_host_transpilation(g, e, f);
            let expected = if !g && !e && f {
                CudaSafetyVerdict::Pass
            } else {
                CudaSafetyVerdict::Fail
            };
            assert_eq!(v, expected, "mask={mask:03b}");
        }
    }

    // -----------------------------------------------------------------
    // Section 7: Realistic vectors.
    // -----------------------------------------------------------------
    #[test]
    fn realistic_healthy_decy_transpilation_passes_all_4() {
        // CUDA-001
        let v1 = verdict_from_kernel_ffi(true, true, true, true);
        // CUDA-002
        let v2 = verdict_from_host_transpilation(false, false, true);
        // CUDA-003 (parse + borrow_gen + array_slice + optimize + codegen)
        let v3 = verdict_from_qualifier_preservation(true, &[true; 5]);
        // CUDA-004
        let v4 = verdict_from_keyword_detection(true, true, true);
        assert_eq!(v1, CudaSafetyVerdict::Pass);
        assert_eq!(v2, CudaSafetyVerdict::Pass);
        assert_eq!(v3, CudaSafetyVerdict::Pass);
        assert_eq!(v4, CudaSafetyVerdict::Pass);
    }

    #[test]
    fn realistic_pre_fix_all_4_failures() {
        // The regression class — every gate trips simultaneously.
        let v1 = verdict_from_kernel_ffi(true, false, true, true);
        let v2 = verdict_from_host_transpilation(true, false, true);
        let v3 = verdict_from_qualifier_preservation(true, &[true, true, false, true, true]);
        let v4 = verdict_from_keyword_detection(true, false, true);
        assert_eq!(v1, CudaSafetyVerdict::Fail);
        assert_eq!(v2, CudaSafetyVerdict::Fail);
        assert_eq!(v3, CudaSafetyVerdict::Fail);
        assert_eq!(v4, CudaSafetyVerdict::Fail);
    }
}
