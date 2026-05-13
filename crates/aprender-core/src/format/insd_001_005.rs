// SHIP-TWO-001 — `apr-inspect-dtype-naming-v1` algorithm-level PARTIAL
// discharge for FALSIFY-INSPECT-DTYPE-001..005.
//
// Contract: `contracts/apr-inspect-dtype-naming-v1.yaml`.
// Spec: `docs/specifications/aprender-train/ship-two-models-spec.md`.
//
// ## What this file proves NOW (PARTIAL_ALGORITHM_LEVEL)
//
// Five `apr inspect` dtype-naming gates (raw GGML discriminant leak):
//
// - INSPECT-DTYPE-001 (text DType column has no pure integers).
// - INSPECT-DTYPE-002 (JSON tensors[*].dtype has no pure integers).
// - INSPECT-DTYPE-003 (rosetta inspect text DType has no pure integers).
// - INSPECT-DTYPE-004 (text dtype set ⊆ tensors --json dtype set).
// - INSPECT-DTYPE-005 (validate_inspect.rs references ggml_dtype_name
//   or equivalent name mapper — root-cause guard).

/// Forbidden source pattern for INSPECT-DTYPE-005 (regression guard).
pub const AC_INSD_005_REQUIRED_NAME_MAPPERS: [&str; 2] =
    ["ggml_dtype_name", "dtype_name"];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InsdVerdict {
    Pass,
    Fail,
}

// -----------------------------------------------------------------------------
// In-module reference helpers.
// -----------------------------------------------------------------------------

/// True iff `s` parses as a non-negative integer (matches `^[0-9]+$`).
#[must_use]
pub fn is_pure_integer(s: &str) -> bool {
    !s.is_empty() && s.chars().all(|c| c.is_ascii_digit())
}

// -----------------------------------------------------------------------------
// Verdict 1, 2, 3: dtype column / json / rosetta has no pure integers.
// -----------------------------------------------------------------------------

/// Pass iff every entry in `dtypes` is a non-empty string AND not a
/// pure integer (e.g., "F32", "Q4_K", but never "0", "14").
#[must_use]
pub fn verdict_from_no_pure_integer_dtypes(dtypes: &[&str]) -> InsdVerdict {
    if dtypes.is_empty() {
        // Vacuous Pass: no dtypes to check.
        return InsdVerdict::Pass;
    }
    for &d in dtypes {
        if d.is_empty() || is_pure_integer(d) {
            return InsdVerdict::Fail;
        }
    }
    InsdVerdict::Pass
}

// -----------------------------------------------------------------------------
// Verdict 4: text dtype set ⊆ JSON dtype set.
// -----------------------------------------------------------------------------

/// Pass iff every entry in `text_dtypes` appears in `json_dtypes`.
#[must_use]
pub fn verdict_from_dtype_vocabulary_subset(
    text_dtypes: &[&str],
    json_dtypes: &[&str],
) -> InsdVerdict {
    let json_set: std::collections::HashSet<&str> = json_dtypes.iter().copied().collect();
    for &d in text_dtypes {
        if !json_set.contains(d) {
            return InsdVerdict::Fail;
        }
    }
    InsdVerdict::Pass
}

// -----------------------------------------------------------------------------
// Verdict 5: ggml_dtype_name (or equivalent) reachable.
// -----------------------------------------------------------------------------

/// Pass iff `code_text` references at least one of
/// `AC_INSD_005_REQUIRED_NAME_MAPPERS`.
#[must_use]
pub fn verdict_from_name_mapper_reachable(code_text: &str) -> InsdVerdict {
    if AC_INSD_005_REQUIRED_NAME_MAPPERS
        .iter()
        .any(|p| code_text.contains(p))
    {
        InsdVerdict::Pass
    } else {
        InsdVerdict::Fail
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Section 1: Provenance pin.
    // -------------------------------------------------------------------------
    #[test]
    fn provenance_required_mappers() {
        assert!(AC_INSD_005_REQUIRED_NAME_MAPPERS.contains(&"ggml_dtype_name"));
        assert!(AC_INSD_005_REQUIRED_NAME_MAPPERS.contains(&"dtype_name"));
    }

    // -------------------------------------------------------------------------
    // Section 2: is_pure_integer reference.
    // -------------------------------------------------------------------------
    #[test]
    fn pure_integer_basic() {
        assert!(is_pure_integer("0"));
        assert!(is_pure_integer("14"));
        assert!(is_pure_integer("100"));
    }

    #[test]
    fn pure_integer_rejects_dtype_names() {
        assert!(!is_pure_integer("F32"));
        assert!(!is_pure_integer("Q4_K"));
        assert!(!is_pure_integer("Q6_K"));
        assert!(!is_pure_integer("BF16"));
    }

    #[test]
    fn pure_integer_rejects_empty_and_mixed() {
        assert!(!is_pure_integer(""));
        assert!(!is_pure_integer("F32 "));
        assert!(!is_pure_integer(" 0"));
        assert!(!is_pure_integer("Q4_0"));
    }

    // -------------------------------------------------------------------------
    // Section 3: INSPECT-DTYPE-001/002/003 — no pure integers.
    // -------------------------------------------------------------------------
    #[test]
    fn no_pure_int_pass_typical_q4km() {
        let dtypes = vec!["F32", "Q4_K", "Q6_K", "F16"];
        assert_eq!(
            verdict_from_no_pure_integer_dtypes(&dtypes),
            InsdVerdict::Pass
        );
    }

    #[test]
    fn no_pure_int_pass_empty_vacuous() {
        let dtypes: Vec<&str> = vec![];
        assert_eq!(
            verdict_from_no_pure_integer_dtypes(&dtypes),
            InsdVerdict::Pass
        );
    }

    #[test]
    fn no_pure_int_fail_one_pure_int() {
        // Bug: raw GGML discriminant leaked.
        let dtypes = vec!["F32", "14", "Q4_K"]; // 14 is GGML_TYPE_Q4_K
        assert_eq!(
            verdict_from_no_pure_integer_dtypes(&dtypes),
            InsdVerdict::Fail
        );
    }

    #[test]
    fn no_pure_int_fail_zero_leaked() {
        // Bug: GGML_TYPE_F32 = 0 leaked.
        let dtypes = vec!["0", "Q4_K"];
        assert_eq!(
            verdict_from_no_pure_integer_dtypes(&dtypes),
            InsdVerdict::Fail
        );
    }

    #[test]
    fn no_pure_int_fail_all_pure_int() {
        // Worst case: validate_inspect.rs:318 stringified raw u32 for every tensor.
        let dtypes = vec!["0", "14", "12"];
        assert_eq!(
            verdict_from_no_pure_integer_dtypes(&dtypes),
            InsdVerdict::Fail
        );
    }

    #[test]
    fn no_pure_int_fail_empty_string_in_list() {
        // Empty string sneaked in (e.g., None.unwrap_or("")).
        let dtypes = vec!["F32", ""];
        assert_eq!(
            verdict_from_no_pure_integer_dtypes(&dtypes),
            InsdVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 4: INSPECT-DTYPE-004 — vocabulary subset.
    // -------------------------------------------------------------------------
    #[test]
    fn dtype_subset_pass_text_subset_of_json() {
        let text = vec!["F32", "Q4_K"];
        let json = vec!["F32", "Q4_K", "Q6_K", "F16"];
        assert_eq!(
            verdict_from_dtype_vocabulary_subset(&text, &json),
            InsdVerdict::Pass
        );
    }

    #[test]
    fn dtype_subset_pass_equal() {
        let v = vec!["F32", "Q4_K"];
        assert_eq!(
            verdict_from_dtype_vocabulary_subset(&v, &v),
            InsdVerdict::Pass
        );
    }

    #[test]
    fn dtype_subset_pass_empty_text() {
        let text: Vec<&str> = vec![];
        let json = vec!["F32"];
        assert_eq!(
            verdict_from_dtype_vocabulary_subset(&text, &json),
            InsdVerdict::Pass
        );
    }

    #[test]
    fn dtype_subset_fail_text_has_extra() {
        // text reports Q5_K but tensors --json doesn't — drift.
        let text = vec!["F32", "Q5_K"];
        let json = vec!["F32", "Q4_K"];
        assert_eq!(
            verdict_from_dtype_vocabulary_subset(&text, &json),
            InsdVerdict::Fail
        );
    }

    #[test]
    fn dtype_subset_fail_text_has_pure_int() {
        // text emits "14", JSON emits "Q4_K" — mismatch even though
        // semantically equivalent.
        let text = vec!["14"];
        let json = vec!["Q4_K"];
        assert_eq!(
            verdict_from_dtype_vocabulary_subset(&text, &json),
            InsdVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 5: INSPECT-DTYPE-005 — name mapper reachable.
    // -------------------------------------------------------------------------
    #[test]
    fn insd005_pass_ggml_dtype_name_referenced() {
        let code = r#"
            use crate::format::tensors::ggml_dtype_name;
            fn validate(...) {
                let name = ggml_dtype_name(t.dtype);
            }
        "#;
        assert_eq!(
            verdict_from_name_mapper_reachable(code),
            InsdVerdict::Pass
        );
    }

    #[test]
    fn insd005_pass_dtype_name_alias() {
        let code = "let name = dtype_name(d);";
        assert_eq!(
            verdict_from_name_mapper_reachable(code),
            InsdVerdict::Pass
        );
    }

    #[test]
    fn insd005_fail_no_mapper_referenced() {
        let code = r#"
            // Bug: stringifies raw u32 directly.
            let s = format!("{}", t.dtype as u32);
        "#;
        assert_eq!(
            verdict_from_name_mapper_reachable(code),
            InsdVerdict::Fail
        );
    }

    #[test]
    fn insd005_fail_empty_code() {
        assert_eq!(
            verdict_from_name_mapper_reachable(""),
            InsdVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 6: Realistic — full bug regression scenarios.
    // -------------------------------------------------------------------------
    #[test]
    fn realistic_pre_fix_validate_inspect_318_caught() {
        // Pre-fix: raw discriminants.
        let text_dtypes = vec!["0", "14", "0", "12"];
        assert_eq!(
            verdict_from_no_pure_integer_dtypes(&text_dtypes),
            InsdVerdict::Fail
        );
    }

    #[test]
    fn realistic_post_fix_q4km_named_dtypes() {
        // Post-fix: dtype names.
        let text_dtypes = vec!["F32", "Q4_K", "F32", "Q6_K"];
        assert_eq!(
            verdict_from_no_pure_integer_dtypes(&text_dtypes),
            InsdVerdict::Pass
        );
    }

    #[test]
    fn realistic_cross_command_drift_caught() {
        // INSPECT-DTYPE-004 if_fails: "inspect and tensors use
        // different dtype vocabularies".
        let inspect_text = vec!["F32", "Q4_K"];
        let tensors_json = vec!["float32", "q4_k"]; // lowercase variant
        assert_eq!(
            verdict_from_dtype_vocabulary_subset(&inspect_text, &tensors_json),
            InsdVerdict::Fail
        );
    }

    #[test]
    fn realistic_root_cause_unfixed_caught() {
        // INSPECT-DTYPE-005 if_fails: "Root-cause fix not applied —
        // validate_inspect.rs still stringifies the raw u32".
        // (NOTE: comment must not mention either of the required
        // mapper names verbatim, or the substring search will
        // false-positive.)
        let buggy_code = r#"
            // Bug: stringifies the raw u32 directly.
            let s = format!("{}", t.dtype as u32);
        "#;
        assert_eq!(
            verdict_from_name_mapper_reachable(buggy_code),
            InsdVerdict::Fail
        );
    }

    #[test]
    fn realistic_full_post_fix_pipeline_passes_all_5_gates() {
        // Synthesize post-fix Qwen2.5-Coder-7B-Q4_K_M:
        let text_dtypes = vec!["F32", "Q4_K", "Q6_K"];
        let json_dtypes = vec!["F32", "Q4_K", "Q6_K"];
        let rosetta_dtypes = vec!["F32", "Q4_K", "Q6_K"];
        let code = "use ggml_dtype_name; fn validate() { ggml_dtype_name(d); }";

        // INSPECT-DTYPE-001 (text):
        assert_eq!(
            verdict_from_no_pure_integer_dtypes(&text_dtypes),
            InsdVerdict::Pass
        );
        // INSPECT-DTYPE-002 (JSON):
        assert_eq!(
            verdict_from_no_pure_integer_dtypes(&json_dtypes),
            InsdVerdict::Pass
        );
        // INSPECT-DTYPE-003 (rosetta):
        assert_eq!(
            verdict_from_no_pure_integer_dtypes(&rosetta_dtypes),
            InsdVerdict::Pass
        );
        // INSPECT-DTYPE-004 (subset):
        assert_eq!(
            verdict_from_dtype_vocabulary_subset(&text_dtypes, &json_dtypes),
            InsdVerdict::Pass
        );
        // INSPECT-DTYPE-005 (mapper reachable):
        assert_eq!(
            verdict_from_name_mapper_reachable(code),
            InsdVerdict::Pass
        );
    }
}
