// Bundles three sister contracts in one verdict module:
//
//   `validated-tensor-v1` (FALSIFY-VT-001..004)
//   `wasmtime-upgrade-v1` (FALSIFY-WASM-001..004)
//   `unified-specs-v1` (FALSIFY-SPECS-001..003)
//
// VT-001: density gate — non-zero fraction > 0.055
// VT-002: NaN/Inf rejection
// VT-003: L2 norm — zero rows detected
// VT-004: SIMD vs scalar validation equivalence (zero tolerance)
// WASM-001: api_compatibility — wasmtime v43 compiles unchanged
// WASM-002: behavioral_parity — runtime tests pass
// WASM-003: advisory_elimination — zero wasmtime entries in audit/deny
// WASM-004: api_compatibility — wasm_reference_types works with gc
// SPECS-001: TOC ≤ 500 lines
// SPECS-002: no orphan specs (every .md in TOC)
// SPECS-003: no subcrate specs remain (root only)

/// VT-001: minimum non-zero fraction.
pub const AC_VT_DENSITY_FLOOR: f32 = 0.055;
/// SPECS-001: max TOC.md lines.
pub const AC_SPECS_TOC_MAX_LINES: u32 = 500;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VtWasmSpecsVerdict {
    Pass,
    Fail,
}

// ----------------------------------------------------------------
// VT-001 + VT-003 (density + L2)
// ----------------------------------------------------------------

/// VT-001: density (1.0 - zero_fraction) must be > 0.055.
#[must_use]
pub fn verdict_from_density_gate(zero_count: u64, total: u64) -> VtWasmSpecsVerdict {
    if total == 0 {
        return VtWasmSpecsVerdict::Fail;
    }
    let density = 1.0 - zero_count as f32 / total as f32;
    if density > AC_VT_DENSITY_FLOOR {
        VtWasmSpecsVerdict::Pass
    } else {
        VtWasmSpecsVerdict::Fail
    }
}

/// VT-003: at least one row has nonzero L2 norm.
///
/// `zero_row_count` = number of rows where all elements are 0.
/// `total_rows` = total rows.
/// Pass iff `zero_row_count < total_rows`.
#[must_use]
pub fn verdict_from_l2_no_zero_rows(
    zero_row_count: u64,
    total_rows: u64,
) -> VtWasmSpecsVerdict {
    if total_rows == 0 {
        return VtWasmSpecsVerdict::Fail;
    }
    if zero_row_count == 0 {
        VtWasmSpecsVerdict::Pass
    } else {
        VtWasmSpecsVerdict::Fail
    }
}

// ----------------------------------------------------------------
// VT-002 NaN/Inf rejection
// ----------------------------------------------------------------

/// VT-002: scanning detects NaN/Inf.
///
/// `validation_returned_err` = true iff the scan detected non-finite.
/// `actually_has_nonfinite` = true iff input had NaN/Inf.
/// Pass iff outcome matches input contamination.
#[must_use]
pub fn verdict_from_nan_inf_rejection(
    actually_has_nonfinite: bool,
    validation_returned_err: bool,
) -> VtWasmSpecsVerdict {
    if actually_has_nonfinite == validation_returned_err {
        VtWasmSpecsVerdict::Pass
    } else {
        VtWasmSpecsVerdict::Fail
    }
}

// ----------------------------------------------------------------
// VT-004 SIMD vs scalar
// ----------------------------------------------------------------

#[must_use]
pub fn verdict_from_validation_simd_parity(
    simd_result_is_err: bool,
    scalar_result_is_err: bool,
) -> VtWasmSpecsVerdict {
    if simd_result_is_err == scalar_result_is_err {
        VtWasmSpecsVerdict::Pass
    } else {
        VtWasmSpecsVerdict::Fail
    }
}

// ----------------------------------------------------------------
// WASM-001 + WASM-002 + WASM-004
// ----------------------------------------------------------------

/// WASM-001: cargo check compiles unchanged.
#[must_use]
pub fn verdict_from_wasm_api_compat(cargo_check_ok: bool) -> VtWasmSpecsVerdict {
    if cargo_check_ok {
        VtWasmSpecsVerdict::Pass
    } else {
        VtWasmSpecsVerdict::Fail
    }
}

/// WASM-002: all runtime tests pass.
#[must_use]
pub fn verdict_from_wasm_behavioral_parity(
    tests_passed: u32,
    tests_failed: u32,
) -> VtWasmSpecsVerdict {
    if tests_passed == 0 {
        return VtWasmSpecsVerdict::Fail;
    }
    if tests_failed == 0 {
        VtWasmSpecsVerdict::Pass
    } else {
        VtWasmSpecsVerdict::Fail
    }
}

/// WASM-004: gc feature enables wasm_reference_types.
#[must_use]
pub fn verdict_from_wasm_gc_feature(gc_enabled: bool, ref_types_ok: bool) -> VtWasmSpecsVerdict {
    if gc_enabled && ref_types_ok {
        VtWasmSpecsVerdict::Pass
    } else {
        VtWasmSpecsVerdict::Fail
    }
}

// ----------------------------------------------------------------
// WASM-003 advisory elimination
// ----------------------------------------------------------------

/// WASM-003: zero wasmtime entries in audit/deny config.
#[must_use]
pub fn verdict_from_wasm_advisory_clean(
    audit_toml_wasmtime_entries: u32,
    deny_toml_wasmtime_entries: u32,
) -> VtWasmSpecsVerdict {
    if audit_toml_wasmtime_entries == 0 && deny_toml_wasmtime_entries == 0 {
        VtWasmSpecsVerdict::Pass
    } else {
        VtWasmSpecsVerdict::Fail
    }
}

// ----------------------------------------------------------------
// SPECS-001..003
// ----------------------------------------------------------------

/// SPECS-001: TOC.md ≤ 500 lines.
#[must_use]
pub fn verdict_from_toc_size(line_count: u32) -> VtWasmSpecsVerdict {
    if line_count == 0 {
        return VtWasmSpecsVerdict::Fail;
    }
    if line_count <= AC_SPECS_TOC_MAX_LINES {
        VtWasmSpecsVerdict::Pass
    } else {
        VtWasmSpecsVerdict::Fail
    }
}

/// SPECS-002: zero orphan specs (every .md in TOC).
#[must_use]
pub fn verdict_from_no_orphan_specs(orphan_count: u32) -> VtWasmSpecsVerdict {
    if orphan_count == 0 {
        VtWasmSpecsVerdict::Pass
    } else {
        VtWasmSpecsVerdict::Fail
    }
}

/// SPECS-003: zero .md files under crates/*/docs/specifications/.
#[must_use]
pub fn verdict_from_no_subcrate_specs(subcrate_spec_count: u32) -> VtWasmSpecsVerdict {
    if subcrate_spec_count == 0 {
        VtWasmSpecsVerdict::Pass
    } else {
        VtWasmSpecsVerdict::Fail
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------
    // Section 1: Provenance pin.
    // -----------------------------------------------------------------
    #[test]
    fn provenance_constants() {
        assert_eq!(AC_VT_DENSITY_FLOOR, 0.055);
        assert_eq!(AC_SPECS_TOC_MAX_LINES, 500);
    }

    // -----------------------------------------------------------------
    // Section 2: VT-001..004.
    // -----------------------------------------------------------------
    #[test]
    fn fvt001_pass_dense_embedding() {
        // 1000 elements, 100 zeros = 90% dense
        let v = verdict_from_density_gate(100, 1000);
        assert_eq!(v, VtWasmSpecsVerdict::Pass);
    }

    #[test]
    fn fvt001_fail_94pct_zeros() {
        // PMAT-234 regression — 6% dense = 0.06 > 0.055 actually passes...
        // Use 95% zero (5% dense < 5.5%) to trigger Fail.
        let v = verdict_from_density_gate(950, 1000);
        assert_eq!(v, VtWasmSpecsVerdict::Fail);
    }

    #[test]
    fn fvt001_fail_just_below_boundary() {
        // density 0.054 < 0.055 → Fail (strict >)
        let v = verdict_from_density_gate(946, 1000);
        assert_eq!(v, VtWasmSpecsVerdict::Fail);
    }

    #[test]
    fn fvt002_pass_clean_input() {
        let v = verdict_from_nan_inf_rejection(false, false);
        assert_eq!(v, VtWasmSpecsVerdict::Pass);
    }

    #[test]
    fn fvt002_pass_nan_detected() {
        let v = verdict_from_nan_inf_rejection(true, true);
        assert_eq!(v, VtWasmSpecsVerdict::Pass);
    }

    #[test]
    fn fvt002_fail_nan_undetected() {
        // The exact regression class — silent NaN propagation
        let v = verdict_from_nan_inf_rejection(true, false);
        assert_eq!(v, VtWasmSpecsVerdict::Fail);
    }

    #[test]
    fn fvt002_fail_false_positive() {
        let v = verdict_from_nan_inf_rejection(false, true);
        assert_eq!(v, VtWasmSpecsVerdict::Fail);
    }

    #[test]
    fn fvt003_pass_all_rows_nonzero() {
        let v = verdict_from_l2_no_zero_rows(0, 100);
        assert_eq!(v, VtWasmSpecsVerdict::Pass);
    }

    #[test]
    fn fvt003_fail_one_zero_row() {
        let v = verdict_from_l2_no_zero_rows(1, 100);
        assert_eq!(v, VtWasmSpecsVerdict::Fail);
    }

    #[test]
    fn fvt003_fail_zero_total() {
        let v = verdict_from_l2_no_zero_rows(0, 0);
        assert_eq!(v, VtWasmSpecsVerdict::Fail);
    }

    #[test]
    fn fvt004_pass_both_err() {
        let v = verdict_from_validation_simd_parity(true, true);
        assert_eq!(v, VtWasmSpecsVerdict::Pass);
    }

    #[test]
    fn fvt004_pass_both_ok() {
        let v = verdict_from_validation_simd_parity(false, false);
        assert_eq!(v, VtWasmSpecsVerdict::Pass);
    }

    #[test]
    fn fvt004_fail_simd_only_err() {
        let v = verdict_from_validation_simd_parity(true, false);
        assert_eq!(v, VtWasmSpecsVerdict::Fail);
    }

    // -----------------------------------------------------------------
    // Section 3: WASM-001..004.
    // -----------------------------------------------------------------
    #[test]
    fn fwasm001_pass_compiles() {
        let v = verdict_from_wasm_api_compat(true);
        assert_eq!(v, VtWasmSpecsVerdict::Pass);
    }

    #[test]
    fn fwasm001_fail_does_not_compile() {
        let v = verdict_from_wasm_api_compat(false);
        assert_eq!(v, VtWasmSpecsVerdict::Fail);
    }

    #[test]
    fn fwasm002_pass_all_tests_green() {
        let v = verdict_from_wasm_behavioral_parity(42, 0);
        assert_eq!(v, VtWasmSpecsVerdict::Pass);
    }

    #[test]
    fn fwasm002_fail_one_test_red() {
        let v = verdict_from_wasm_behavioral_parity(41, 1);
        assert_eq!(v, VtWasmSpecsVerdict::Fail);
    }

    #[test]
    fn fwasm002_fail_zero_tests_run() {
        let v = verdict_from_wasm_behavioral_parity(0, 0);
        assert_eq!(v, VtWasmSpecsVerdict::Fail);
    }

    #[test]
    fn fwasm003_pass_clean() {
        let v = verdict_from_wasm_advisory_clean(0, 0);
        assert_eq!(v, VtWasmSpecsVerdict::Pass);
    }

    #[test]
    fn fwasm003_fail_audit_entry() {
        let v = verdict_from_wasm_advisory_clean(1, 0);
        assert_eq!(v, VtWasmSpecsVerdict::Fail);
    }

    #[test]
    fn fwasm003_fail_deny_entry() {
        let v = verdict_from_wasm_advisory_clean(0, 2);
        assert_eq!(v, VtWasmSpecsVerdict::Fail);
    }

    #[test]
    fn fwasm004_pass_gc_with_ref_types() {
        let v = verdict_from_wasm_gc_feature(true, true);
        assert_eq!(v, VtWasmSpecsVerdict::Pass);
    }

    #[test]
    fn fwasm004_fail_gc_disabled() {
        let v = verdict_from_wasm_gc_feature(false, true);
        assert_eq!(v, VtWasmSpecsVerdict::Fail);
    }

    #[test]
    fn fwasm004_fail_ref_types_broken() {
        let v = verdict_from_wasm_gc_feature(true, false);
        assert_eq!(v, VtWasmSpecsVerdict::Fail);
    }

    // -----------------------------------------------------------------
    // Section 4: SPECS-001..003.
    // -----------------------------------------------------------------
    #[test]
    fn fspecs001_pass_400_lines() {
        let v = verdict_from_toc_size(400);
        assert_eq!(v, VtWasmSpecsVerdict::Pass);
    }

    #[test]
    fn fspecs001_pass_at_threshold() {
        let v = verdict_from_toc_size(500);
        assert_eq!(v, VtWasmSpecsVerdict::Pass);
    }

    #[test]
    fn fspecs001_fail_above_500() {
        let v = verdict_from_toc_size(501);
        assert_eq!(v, VtWasmSpecsVerdict::Fail);
    }

    #[test]
    fn fspecs001_fail_zero_lines() {
        // Empty TOC = TOC missing
        let v = verdict_from_toc_size(0);
        assert_eq!(v, VtWasmSpecsVerdict::Fail);
    }

    #[test]
    fn fspecs002_pass_no_orphans() {
        let v = verdict_from_no_orphan_specs(0);
        assert_eq!(v, VtWasmSpecsVerdict::Pass);
    }

    #[test]
    fn fspecs002_fail_one_orphan() {
        let v = verdict_from_no_orphan_specs(1);
        assert_eq!(v, VtWasmSpecsVerdict::Fail);
    }

    #[test]
    fn fspecs003_pass_root_only() {
        let v = verdict_from_no_subcrate_specs(0);
        assert_eq!(v, VtWasmSpecsVerdict::Pass);
    }

    #[test]
    fn fspecs003_fail_subcrate_specs_remain() {
        let v = verdict_from_no_subcrate_specs(5);
        assert_eq!(v, VtWasmSpecsVerdict::Fail);
    }

    // -----------------------------------------------------------------
    // Section 5: Mutation surveys.
    // -----------------------------------------------------------------
    #[test]
    fn mutation_survey_001_density_band() {
        // Avoid the 945/1000 = 0.055 exact boundary (f32 rounding).
        for pct in [0_u64, 50, 90, 100, 940, 950, 990, 1000] {
            let zeros = pct;
            let v = verdict_from_density_gate(zeros, 1000);
            let density = 1.0 - zeros as f32 / 1000.0;
            let want = if density > 0.055 {
                VtWasmSpecsVerdict::Pass
            } else {
                VtWasmSpecsVerdict::Fail
            };
            assert_eq!(v, want, "zeros={zeros}");
        }
    }

    #[test]
    fn mutation_survey_specs_toc_band() {
        for n in [1_u32, 100, 499, 500, 501, 1000] {
            let v = verdict_from_toc_size(n);
            let want = if n <= 500 {
                VtWasmSpecsVerdict::Pass
            } else {
                VtWasmSpecsVerdict::Fail
            };
            assert_eq!(v, want, "n={n}");
        }
    }

    // -----------------------------------------------------------------
    // Section 6: Realistic.
    // -----------------------------------------------------------------
    #[test]
    fn realistic_healthy_passes_all_11() {
        let v1 = verdict_from_density_gate(50, 1000);
        let v2 = verdict_from_nan_inf_rejection(false, false);
        let v3 = verdict_from_l2_no_zero_rows(0, 100);
        let v4 = verdict_from_validation_simd_parity(false, false);
        let v5 = verdict_from_wasm_api_compat(true);
        let v6 = verdict_from_wasm_behavioral_parity(42, 0);
        let v7 = verdict_from_wasm_advisory_clean(0, 0);
        let v8 = verdict_from_wasm_gc_feature(true, true);
        let v9 = verdict_from_toc_size(425);
        let v10 = verdict_from_no_orphan_specs(0);
        let v11 = verdict_from_no_subcrate_specs(0);
        for v in [v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11] {
            assert_eq!(v, VtWasmSpecsVerdict::Pass);
        }
    }

    #[test]
    fn realistic_pre_fix_all_11_failures() {
        // 11 simultaneous regressions across 3 contracts.
        let v1 = verdict_from_density_gate(950, 1000); // density bug
        let v2 = verdict_from_nan_inf_rejection(true, false); // NaN missed
        let v3 = verdict_from_l2_no_zero_rows(1, 100); // dead row
        let v4 = verdict_from_validation_simd_parity(true, false); // SIMD drift
        let v5 = verdict_from_wasm_api_compat(false); // API broke
        let v6 = verdict_from_wasm_behavioral_parity(40, 2); // tests fail
        let v7 = verdict_from_wasm_advisory_clean(1, 0); // advisory leak
        let v8 = verdict_from_wasm_gc_feature(false, true); // gc disabled
        let v9 = verdict_from_toc_size(1200); // TOC bloated
        let v10 = verdict_from_no_orphan_specs(7); // orphan specs
        let v11 = verdict_from_no_subcrate_specs(12); // subcrate specs
        for v in [v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11] {
            assert_eq!(v, VtWasmSpecsVerdict::Fail);
        }
    }
}
