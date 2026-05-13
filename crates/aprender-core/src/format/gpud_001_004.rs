// SHIP-TWO-001 — `apr-gpu-diagnostics-v1` algorithm-level PARTIAL
// discharge for FALSIFY-GPU-001..004.
//
// Contract: `contracts/apr-gpu-diagnostics-v1.yaml`.
// Spec: `docs/specifications/aprender-train/ship-two-models-spec.md`.
//
// ## What this file proves NOW (PARTIAL_ALGORITHM_LEVEL)
//
// Four GPU diagnostics gates from `apr ptx` / `apr ptx-map` / `apr cbtop`:
//
// - GPU-001 (PTX syntactically valid): contains `.version` AND
//   `.target sm_<arch>`, balanced braces.
// - GPU-002 (every layer maps to ≥1 kernel): each layer's kernel count > 0.
// - GPU-003 (NDJSON N-line format): exactly N lines, each parses as
//   JSON with required fields.
// - GPU-004 (GPU memory within 5%): |reported - actual| / actual ≤ 0.05
//   AND temperature ∈ [0, 120].

/// PTX header version directive.
pub const AC_GPU_001_VERSION_DIRECTIVE: &str = ".version";

/// PTX target directive prefix.
pub const AC_GPU_001_TARGET_PREFIX: &str = ".target sm_";

/// GPU-002 minimum kernel count per layer.
pub const AC_GPU_002_MIN_KERNELS_PER_LAYER: usize = 1;

/// GPU-004 memory measurement tolerance (5%).
pub const AC_GPU_004_MEM_TOLERANCE_FRAC: f32 = 0.05;

/// GPU-004 temperature lower bound (Celsius).
pub const AC_GPU_004_TEMP_LOWER_C: f32 = 0.0;

/// GPU-004 temperature upper bound (Celsius).
pub const AC_GPU_004_TEMP_UPPER_C: f32 = 120.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpudVerdict {
    Pass,
    Fail,
}

// -----------------------------------------------------------------------------
// In-module reference helpers.
// -----------------------------------------------------------------------------

/// Count `{` minus `}` to verify balanced braces (a coarse check for
/// PTX `.entry` body closure).
#[must_use]
pub fn brace_balance(s: &str) -> i32 {
    let mut count = 0_i32;
    for c in s.chars() {
        match c {
            '{' => count += 1,
            '}' => count -= 1,
            _ => {}
        }
    }
    count
}

// -----------------------------------------------------------------------------
// Verdict 1: GPU-001 — PTX syntactically valid for target arch.
// -----------------------------------------------------------------------------

/// Pass iff `ptx` contains `.version` AND `.target sm_<arch>` AND
/// braces balance to zero.
#[must_use]
pub fn verdict_from_ptx_syntactically_valid(ptx: &str, arch: &str) -> GpudVerdict {
    if ptx.is_empty() {
        return GpudVerdict::Fail;
    }
    if !ptx.contains(AC_GPU_001_VERSION_DIRECTIVE) {
        return GpudVerdict::Fail;
    }
    let target_directive = format!("{AC_GPU_001_TARGET_PREFIX}{arch}");
    if !ptx.contains(&target_directive) {
        return GpudVerdict::Fail;
    }
    if brace_balance(ptx) != 0 {
        return GpudVerdict::Fail;
    }
    GpudVerdict::Pass
}

// -----------------------------------------------------------------------------
// Verdict 2: GPU-002 — every layer maps to ≥ 1 kernel.
// -----------------------------------------------------------------------------

/// `kernel_counts_per_layer[i]` is the number of GPU kernels mapped
/// to layer i. Pass iff every entry ≥ 1.
#[must_use]
pub fn verdict_from_layer_kernel_mapping(kernel_counts_per_layer: &[usize]) -> GpudVerdict {
    if kernel_counts_per_layer.is_empty() {
        return GpudVerdict::Fail;
    }
    for &c in kernel_counts_per_layer {
        if c < AC_GPU_002_MIN_KERNELS_PER_LAYER {
            return GpudVerdict::Fail;
        }
    }
    GpudVerdict::Pass
}

// -----------------------------------------------------------------------------
// Verdict 3: GPU-003 — NDJSON exactly N lines, each with required fields.
// -----------------------------------------------------------------------------

/// `lines` are output lines from cbtop Headless(N). Each entry is a
/// `(line_str, has_required_fields)` pair where `has_required_fields`
/// is true iff the parsed JSON has all of: timestamp, gpu_mem, cpu_util.
#[must_use]
pub fn verdict_from_ndjson_n_lines(
    line_count: usize,
    expected_n: usize,
    each_has_required_fields: &[bool],
) -> GpudVerdict {
    if expected_n == 0 {
        return GpudVerdict::Fail;
    }
    if line_count != expected_n {
        return GpudVerdict::Fail;
    }
    if each_has_required_fields.len() != expected_n {
        return GpudVerdict::Fail;
    }
    if !each_has_required_fields.iter().all(|&v| v) {
        return GpudVerdict::Fail;
    }
    GpudVerdict::Pass
}

// -----------------------------------------------------------------------------
// Verdict 4: GPU-004 — memory measurement + temperature bounds.
// -----------------------------------------------------------------------------

/// Pass iff:
///   1. `|reported_mb - actual_mb| / actual_mb ≤ 0.05`,
///   2. `temperature_c ∈ [0, 120]`.
#[must_use]
pub fn verdict_from_memory_and_temperature(
    reported_mb: f32,
    actual_mb: f32,
    temperature_c: f32,
) -> GpudVerdict {
    if !reported_mb.is_finite() || !actual_mb.is_finite() || !temperature_c.is_finite() {
        return GpudVerdict::Fail;
    }
    if actual_mb <= 0.0 {
        return GpudVerdict::Fail;
    }
    let frac_diff = (reported_mb - actual_mb).abs() / actual_mb;
    if frac_diff > AC_GPU_004_MEM_TOLERANCE_FRAC {
        return GpudVerdict::Fail;
    }
    if !(AC_GPU_004_TEMP_LOWER_C..=AC_GPU_004_TEMP_UPPER_C).contains(&temperature_c) {
        return GpudVerdict::Fail;
    }
    GpudVerdict::Pass
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_ptx(arch: &str) -> String {
        format!(".version 7.0\n.target sm_{arch}\n.address_size 64\n.entry main() {{\n  ret;\n}}\n")
    }

    // -------------------------------------------------------------------------
    // Section 1: Provenance pins.
    // -------------------------------------------------------------------------
    #[test]
    fn provenance_version_directive() {
        assert_eq!(AC_GPU_001_VERSION_DIRECTIVE, ".version");
    }

    #[test]
    fn provenance_target_prefix() {
        assert_eq!(AC_GPU_001_TARGET_PREFIX, ".target sm_");
    }

    #[test]
    fn provenance_min_kernels_one() {
        assert_eq!(AC_GPU_002_MIN_KERNELS_PER_LAYER, 1);
    }

    #[test]
    fn provenance_mem_tolerance_005() {
        assert_eq!(AC_GPU_004_MEM_TOLERANCE_FRAC, 0.05);
    }

    #[test]
    fn provenance_temp_bounds() {
        assert_eq!(AC_GPU_004_TEMP_LOWER_C, 0.0);
        assert_eq!(AC_GPU_004_TEMP_UPPER_C, 120.0);
    }

    // -------------------------------------------------------------------------
    // Section 2: GPU-001 — PTX validity.
    // -------------------------------------------------------------------------
    #[test]
    fn gpu001_pass_valid_ptx_sm80() {
        let ptx = valid_ptx("80");
        assert_eq!(
            verdict_from_ptx_syntactically_valid(&ptx, "80"),
            GpudVerdict::Pass
        );
    }

    #[test]
    fn gpu001_pass_valid_ptx_sm90() {
        let ptx = valid_ptx("90");
        assert_eq!(
            verdict_from_ptx_syntactically_valid(&ptx, "90"),
            GpudVerdict::Pass
        );
    }

    #[test]
    fn gpu001_fail_missing_version() {
        let ptx = ".target sm_80\n.entry main() {\n}\n";
        assert_eq!(
            verdict_from_ptx_syntactically_valid(ptx, "80"),
            GpudVerdict::Fail
        );
    }

    #[test]
    fn gpu001_fail_wrong_target() {
        let ptx = valid_ptx("80");
        assert_eq!(
            verdict_from_ptx_syntactically_valid(&ptx, "90"),
            GpudVerdict::Fail
        );
    }

    #[test]
    fn gpu001_fail_unbalanced_braces() {
        let ptx = ".version 7.0\n.target sm_80\n.entry main() {\n  ret;\n";
        assert_eq!(
            verdict_from_ptx_syntactically_valid(ptx, "80"),
            GpudVerdict::Fail
        );
    }

    #[test]
    fn gpu001_fail_empty() {
        assert_eq!(
            verdict_from_ptx_syntactically_valid("", "80"),
            GpudVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 3: GPU-002 — layer kernel mapping.
    // -------------------------------------------------------------------------
    #[test]
    fn gpu002_pass_one_per_layer() {
        let counts = vec![1_usize; 28]; // 28-layer Qwen
        assert_eq!(
            verdict_from_layer_kernel_mapping(&counts),
            GpudVerdict::Pass
        );
    }

    #[test]
    fn gpu002_pass_multi_kernel_per_layer() {
        let counts = vec![3_usize, 2, 4, 3]; // attention/FFN/norm/embed
        assert_eq!(
            verdict_from_layer_kernel_mapping(&counts),
            GpudVerdict::Pass
        );
    }

    #[test]
    fn gpu002_fail_zero_kernel_layer() {
        // Bug: one layer (e.g., embedding) didn't get kernels mapped.
        let counts = vec![3_usize, 2, 0, 3];
        assert_eq!(
            verdict_from_layer_kernel_mapping(&counts),
            GpudVerdict::Fail
        );
    }

    #[test]
    fn gpu002_fail_all_zero() {
        let counts = vec![0_usize, 0, 0];
        assert_eq!(
            verdict_from_layer_kernel_mapping(&counts),
            GpudVerdict::Fail
        );
    }

    #[test]
    fn gpu002_fail_empty() {
        let counts: Vec<usize> = vec![];
        assert_eq!(
            verdict_from_layer_kernel_mapping(&counts),
            GpudVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 4: GPU-003 — NDJSON line format.
    // -------------------------------------------------------------------------
    #[test]
    fn gpu003_pass_5_lines_all_valid() {
        assert_eq!(
            verdict_from_ndjson_n_lines(5, 5, &[true, true, true, true, true]),
            GpudVerdict::Pass
        );
    }

    #[test]
    fn gpu003_pass_1_line() {
        assert_eq!(
            verdict_from_ndjson_n_lines(1, 1, &[true]),
            GpudVerdict::Pass
        );
    }

    #[test]
    fn gpu003_fail_wrong_count() {
        // Headless(5) but only 3 lines emitted.
        assert_eq!(
            verdict_from_ndjson_n_lines(3, 5, &[true, true, true]),
            GpudVerdict::Fail
        );
    }

    #[test]
    fn gpu003_fail_one_line_missing_field() {
        assert_eq!(
            verdict_from_ndjson_n_lines(5, 5, &[true, true, false, true, true]),
            GpudVerdict::Fail
        );
    }

    #[test]
    fn gpu003_fail_validation_array_length_wrong() {
        // line_count says 5 but per-line validation has only 3 entries.
        assert_eq!(
            verdict_from_ndjson_n_lines(5, 5, &[true, true, true]),
            GpudVerdict::Fail
        );
    }

    #[test]
    fn gpu003_fail_zero_expected() {
        assert_eq!(
            verdict_from_ndjson_n_lines(0, 0, &[]),
            GpudVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 5: GPU-004 — memory + temperature.
    // -------------------------------------------------------------------------
    #[test]
    fn gpu004_pass_exact_match() {
        // 24576 MB, 65 C.
        assert_eq!(
            verdict_from_memory_and_temperature(24576.0, 24576.0, 65.0),
            GpudVerdict::Pass
        );
    }

    #[test]
    fn gpu004_pass_within_5_percent() {
        // 24576 MB ± 4% (≤ 5%).
        assert_eq!(
            verdict_from_memory_and_temperature(25559.0, 24576.0, 65.0),
            GpudVerdict::Pass
        );
    }

    #[test]
    fn gpu004_pass_at_temp_bounds() {
        assert_eq!(
            verdict_from_memory_and_temperature(24576.0, 24576.0, 0.0),
            GpudVerdict::Pass
        );
        assert_eq!(
            verdict_from_memory_and_temperature(24576.0, 24576.0, 120.0),
            GpudVerdict::Pass
        );
    }

    #[test]
    fn gpu004_fail_memory_above_5_percent() {
        // 24576 + 8% = bug.
        assert_eq!(
            verdict_from_memory_and_temperature(26542.0, 24576.0, 65.0),
            GpudVerdict::Fail
        );
    }

    #[test]
    fn gpu004_fail_temp_above_120() {
        // Sensor returned 150 C — sensor bug.
        assert_eq!(
            verdict_from_memory_and_temperature(24576.0, 24576.0, 150.0),
            GpudVerdict::Fail
        );
    }

    #[test]
    fn gpu004_fail_temp_negative() {
        assert_eq!(
            verdict_from_memory_and_temperature(24576.0, 24576.0, -10.0),
            GpudVerdict::Fail
        );
    }

    #[test]
    fn gpu004_fail_zero_actual() {
        assert_eq!(
            verdict_from_memory_and_temperature(0.0, 0.0, 65.0),
            GpudVerdict::Fail
        );
    }

    #[test]
    fn gpu004_fail_nan_temp() {
        assert_eq!(
            verdict_from_memory_and_temperature(24576.0, 24576.0, f32::NAN),
            GpudVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 6: Domain — brace balance.
    // -------------------------------------------------------------------------
    #[test]
    fn domain_brace_balance_zero() {
        assert_eq!(brace_balance("{ a { b } c }"), 0);
        assert_eq!(brace_balance(""), 0);
    }

    #[test]
    fn domain_brace_balance_unclosed() {
        assert_eq!(brace_balance("{ a { b }"), 1);
    }

    #[test]
    fn domain_brace_balance_extra_close() {
        assert_eq!(brace_balance("} a"), -1);
    }

    // -------------------------------------------------------------------------
    // Section 7: Realistic — contract regression scenarios.
    // -------------------------------------------------------------------------
    #[test]
    fn realistic_ptx_target_mismatch_caught() {
        // GPU-001 if_fails: PTX emitted for sm_80 but caller asked sm_90.
        let ptx = valid_ptx("80");
        assert_eq!(
            verdict_from_ptx_syntactically_valid(&ptx, "90"),
            GpudVerdict::Fail
        );
    }

    #[test]
    fn realistic_orphaned_layer_caught() {
        // GPU-002 if_fails: a model layer didn't get any kernel mapped.
        let counts = vec![3_usize, 4, 0, 2];
        assert_eq!(
            verdict_from_layer_kernel_mapping(&counts),
            GpudVerdict::Fail
        );
    }

    #[test]
    fn realistic_cbtop_truncated_output_caught() {
        // GPU-003 if_fails: NDJSON cut short (e.g., terminal closed).
        assert_eq!(
            verdict_from_ndjson_n_lines(2, 5, &[true, true]),
            GpudVerdict::Fail
        );
    }

    #[test]
    fn realistic_memory_sensor_off_by_10pct_caught() {
        // GPU-004 if_fails: nvidia-smi vs reported diverged 10%.
        assert_eq!(
            verdict_from_memory_and_temperature(27033.6, 24576.0, 65.0),
            GpudVerdict::Fail
        );
    }

    #[test]
    fn realistic_full_diagnostic_session_passes_all_4_gates() {
        // Synthetic apr ptx + ptx-map + cbtop + nvidia-smi snapshot.
        let ptx = valid_ptx("80");
        assert_eq!(
            verdict_from_ptx_syntactically_valid(&ptx, "80"),
            GpudVerdict::Pass
        );
        let counts = vec![1_usize; 28]; // 28-layer mapping
        assert_eq!(
            verdict_from_layer_kernel_mapping(&counts),
            GpudVerdict::Pass
        );
        assert_eq!(
            verdict_from_ndjson_n_lines(5, 5, &[true; 5]),
            GpudVerdict::Pass
        );
        // RTX 4090: 24576 MB, 65 C.
        assert_eq!(
            verdict_from_memory_and_temperature(24576.0, 24576.0, 65.0),
            GpudVerdict::Pass
        );
    }
}
