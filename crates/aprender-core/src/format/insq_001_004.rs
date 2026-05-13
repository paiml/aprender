// SHIP-TWO-001 — `apr-inspect-quantization-v1` algorithm-level PARTIAL
// discharge for FALSIFY-INSPECT-QUANT-001..004.
//
// Contract: `contracts/apr-inspect-quantization-v1.yaml`.
// Spec: `docs/specifications/aprender-train/ship-two-models-spec.md`.
//
// ## What this file proves NOW (PARTIAL_ALGORITHM_LEVEL)
//
// Four `apr inspect` quantization-reporting gates:
//
// - INSPECT-QUANT-001 (text quantization is NOT "F32" / "0" / empty
//   for a quantized GGUF model).
// - INSPECT-QUANT-002 (JSON .quantization is a valid dtype name, never
//   "F32" / "0" / "null" for a quantized model).
// - INSPECT-QUANT-003 (inspect's reported quantization == dominant-by-
//   parameter-count dtype in `apr tensors --json` output, excluding
//   bias/norm tensors).
// - INSPECT-QUANT-004 (`validate_inspect.rs` does not contain
//   `tensors.first().map(|t| t.dtype` — the buggy first-tensor-stub
//   pattern is gone).

/// Disallowed quantization values for a quantized model (FALSIFY-INSPECT-QUANT-001/002).
pub const AC_INSQ_DISALLOWED: [&str; 4] = ["F32", "0", "", "null"];

/// Tensor-name substrings that are EXCLUDED from dominant-dtype computation.
pub const AC_INSQ_003_EXCLUDED_NAME_SUBSTRINGS: [&str; 3] = ["bias", "norm", "ln_"];

/// Forbidden code substring for INSPECT-QUANT-004 (regression guard).
pub const AC_INSQ_004_FORBIDDEN_PATTERN: &str = "tensors.first().map(|t| t.dtype";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InsqVerdict {
    Pass,
    Fail,
}

// -----------------------------------------------------------------------------
// In-module reference helpers.
// -----------------------------------------------------------------------------

/// True iff a tensor name should be excluded from dominant-dtype
/// computation (case-insensitive substring match).
#[must_use]
pub fn is_excluded_tensor(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    for sub in AC_INSQ_003_EXCLUDED_NAME_SUBSTRINGS {
        if lower.contains(sub) {
            return true;
        }
    }
    false
}

/// Compute the dominant dtype across the (name, dtype, n_params) list,
/// excluding bias/norm/ln_ tensors. Returns the dtype with the largest
/// total parameter count.
#[must_use]
pub fn dominant_weight_dtype(tensors: &[(String, String, u64)]) -> Option<String> {
    let mut counts: std::collections::HashMap<String, u64> = std::collections::HashMap::new();
    for (name, dtype, n_params) in tensors {
        if is_excluded_tensor(name) {
            continue;
        }
        *counts.entry(dtype.clone()).or_insert(0) += n_params;
    }
    counts
        .into_iter()
        .max_by_key(|&(_, c)| c)
        .map(|(d, _)| d)
}

// -----------------------------------------------------------------------------
// Verdict 1: INSPECT-QUANT-001 / 002 — quantization not F32 / 0 / empty / null.
// -----------------------------------------------------------------------------

/// Pass iff `quantization` is not in the disallowed set.
#[must_use]
pub fn verdict_from_quantization_not_default(quantization: &str) -> InsqVerdict {
    if AC_INSQ_DISALLOWED.contains(&quantization) {
        InsqVerdict::Fail
    } else {
        InsqVerdict::Pass
    }
}

// -----------------------------------------------------------------------------
// Verdict 3: INSPECT-QUANT-003 — inspect quant == dominant-by-params dtype.
// -----------------------------------------------------------------------------

/// `inspect_quant` is what `apr inspect --json` reports.
/// `tensors` is `[(name, dtype, n_params), ...]` from `apr tensors --json`.
/// Pass iff `inspect_quant == dominant_weight_dtype(tensors)`.
#[must_use]
pub fn verdict_from_dominant_dtype_agreement(
    inspect_quant: &str,
    tensors: &[(String, String, u64)],
) -> InsqVerdict {
    match dominant_weight_dtype(tensors) {
        Some(d) if d == inspect_quant => InsqVerdict::Pass,
        _ => InsqVerdict::Fail,
    }
}

// -----------------------------------------------------------------------------
// Verdict 4: INSPECT-QUANT-004 — no tensors.first() stub in code.
// -----------------------------------------------------------------------------

/// `code_text` is the contents of `validate_inspect.rs`. Pass iff
/// `AC_INSQ_004_FORBIDDEN_PATTERN` is NOT present.
#[must_use]
pub fn verdict_from_no_first_tensor_stub(code_text: &str) -> InsqVerdict {
    if code_text.contains(AC_INSQ_004_FORBIDDEN_PATTERN) {
        InsqVerdict::Fail
    } else {
        InsqVerdict::Pass
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_tensor(name: &str, dtype: &str, n: u64) -> (String, String, u64) {
        (name.to_string(), dtype.to_string(), n)
    }

    // -------------------------------------------------------------------------
    // Section 1: Provenance pins.
    // -------------------------------------------------------------------------
    #[test]
    fn provenance_disallowed_set() {
        assert!(AC_INSQ_DISALLOWED.contains(&"F32"));
        assert!(AC_INSQ_DISALLOWED.contains(&"0"));
        assert!(AC_INSQ_DISALLOWED.contains(&""));
        assert!(AC_INSQ_DISALLOWED.contains(&"null"));
    }

    #[test]
    fn provenance_excluded_substrings() {
        assert!(AC_INSQ_003_EXCLUDED_NAME_SUBSTRINGS.contains(&"bias"));
        assert!(AC_INSQ_003_EXCLUDED_NAME_SUBSTRINGS.contains(&"norm"));
        assert!(AC_INSQ_003_EXCLUDED_NAME_SUBSTRINGS.contains(&"ln_"));
    }

    #[test]
    fn provenance_forbidden_pattern() {
        assert!(AC_INSQ_004_FORBIDDEN_PATTERN.contains("tensors.first"));
    }

    // -------------------------------------------------------------------------
    // Section 2: is_excluded_tensor / dominant_weight_dtype.
    // -------------------------------------------------------------------------
    #[test]
    fn domain_excluded_bias() {
        assert!(is_excluded_tensor("blk.0.attn.q.bias"));
    }

    #[test]
    fn domain_excluded_norm() {
        assert!(is_excluded_tensor("blk.0.attn_norm.weight"));
        assert!(is_excluded_tensor("output_norm.weight"));
    }

    #[test]
    fn domain_excluded_ln_prefix() {
        assert!(is_excluded_tensor("ln_q.weight"));
        assert!(is_excluded_tensor("LN_K.weight")); // case-insensitive
    }

    #[test]
    fn domain_not_excluded_weight() {
        assert!(!is_excluded_tensor("blk.0.attn.q.weight"));
        assert!(!is_excluded_tensor("blk.0.ffn_gate.weight"));
        assert!(!is_excluded_tensor("output.weight"));
    }

    #[test]
    fn domain_dominant_dtype_q4k() {
        // 339 weights at Q4_K, 28 norms at F32. Dominant is Q4_K.
        let tensors = vec![
            make_tensor("blk.0.attn.q.weight", "Q4_K", 100),
            make_tensor("blk.0.attn.k.weight", "Q4_K", 100),
            make_tensor("blk.0.attn_norm.weight", "F32", 50),
            make_tensor("output_norm.weight", "F32", 50),
        ];
        assert_eq!(dominant_weight_dtype(&tensors), Some("Q4_K".to_string()));
    }

    #[test]
    fn domain_dominant_dtype_excludes_bias() {
        // If we counted bias, F32 would dominate. Excluding gives Q6_K.
        let tensors = vec![
            make_tensor("blk.0.ffn_gate.weight", "Q6_K", 200),
            make_tensor("blk.0.ffn_up.bias", "F32", 1000), // huge bias, excluded
        ];
        assert_eq!(dominant_weight_dtype(&tensors), Some("Q6_K".to_string()));
    }

    #[test]
    fn domain_dominant_dtype_empty_returns_none() {
        let tensors = vec![];
        assert_eq!(dominant_weight_dtype(&tensors), None);
    }

    #[test]
    fn domain_dominant_dtype_only_excluded_returns_none() {
        let tensors = vec![
            make_tensor("blk.0.attn_norm.weight", "F32", 100),
            make_tensor("blk.0.attn.q.bias", "F32", 50),
        ];
        assert_eq!(dominant_weight_dtype(&tensors), None);
    }

    // -------------------------------------------------------------------------
    // Section 3: INSPECT-QUANT-001/002 — quantization not default.
    // -------------------------------------------------------------------------
    #[test]
    fn insq001_pass_q4_k() {
        assert_eq!(
            verdict_from_quantization_not_default("Q4_K"),
            InsqVerdict::Pass
        );
    }

    #[test]
    fn insq001_pass_q4_k_m() {
        assert_eq!(
            verdict_from_quantization_not_default("Q4_K_M"),
            InsqVerdict::Pass
        );
    }

    #[test]
    fn insq001_pass_q6_k() {
        assert_eq!(
            verdict_from_quantization_not_default("Q6_K"),
            InsqVerdict::Pass
        );
    }

    #[test]
    fn insq001_pass_mixed() {
        assert_eq!(
            verdict_from_quantization_not_default("mixed"),
            InsqVerdict::Pass
        );
    }

    #[test]
    fn insq001_fail_f32_for_quantized() {
        // The exact regression: bias dtype leaked.
        assert_eq!(
            verdict_from_quantization_not_default("F32"),
            InsqVerdict::Fail
        );
    }

    #[test]
    fn insq001_fail_zero_string() {
        // Bug: raw integer 0.
        assert_eq!(
            verdict_from_quantization_not_default("0"),
            InsqVerdict::Fail
        );
    }

    #[test]
    fn insq001_fail_empty() {
        assert_eq!(verdict_from_quantization_not_default(""), InsqVerdict::Fail);
    }

    #[test]
    fn insq001_fail_null() {
        // JSON null serialized as "null" string.
        assert_eq!(
            verdict_from_quantization_not_default("null"),
            InsqVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 4: INSPECT-QUANT-003 — dominant-dtype agreement.
    // -------------------------------------------------------------------------
    #[test]
    fn insq003_pass_inspect_matches_dominant() {
        let tensors = vec![
            make_tensor("blk.0.attn.q.weight", "Q4_K", 100),
            make_tensor("blk.0.ffn_gate.weight", "Q4_K", 200),
            make_tensor("blk.0.attn_norm.weight", "F32", 1),
        ];
        assert_eq!(
            verdict_from_dominant_dtype_agreement("Q4_K", &tensors),
            InsqVerdict::Pass
        );
    }

    #[test]
    fn insq003_fail_inspect_disagrees() {
        // inspect reports F32 (bug), tensors say Q4_K.
        let tensors = vec![
            make_tensor("blk.0.attn.q.weight", "Q4_K", 100),
            make_tensor("blk.0.attn_norm.weight", "F32", 1),
        ];
        assert_eq!(
            verdict_from_dominant_dtype_agreement("F32", &tensors),
            InsqVerdict::Fail
        );
    }

    #[test]
    fn insq003_fail_no_weight_tensors() {
        // Only excluded tensors → no dominant.
        let tensors = vec![
            make_tensor("blk.0.attn_norm.weight", "F32", 100),
            make_tensor("blk.0.attn.q.bias", "F32", 50),
        ];
        assert_eq!(
            verdict_from_dominant_dtype_agreement("F32", &tensors),
            InsqVerdict::Fail
        );
    }

    #[test]
    fn insq003_pass_q6_k_dominant() {
        let tensors = vec![
            make_tensor("output.weight", "Q6_K", 1000),
            make_tensor("blk.0.attn.q.weight", "Q4_K", 200),
        ];
        assert_eq!(
            verdict_from_dominant_dtype_agreement("Q6_K", &tensors),
            InsqVerdict::Pass
        );
    }

    // -------------------------------------------------------------------------
    // Section 5: INSPECT-QUANT-004 — no tensors.first() stub.
    // -------------------------------------------------------------------------
    #[test]
    fn insq004_pass_clean_code() {
        let code = r#"
            fn validate_inspect(model: &Model) -> Result<()> {
                let dominant = compute_dominant_dtype(&model.tensors);
                Ok(())
            }
        "#;
        assert_eq!(
            verdict_from_no_first_tensor_stub(code),
            InsqVerdict::Pass
        );
    }

    #[test]
    fn insq004_fail_stub_present() {
        let code = r#"
            fn validate_inspect(model: &Model) {
                // FIXME: stub
                let q = tensors.first().map(|t| t.dtype.clone());
            }
        "#;
        assert_eq!(
            verdict_from_no_first_tensor_stub(code),
            InsqVerdict::Fail
        );
    }

    #[test]
    fn insq004_fail_stub_in_comment() {
        // Even commented-out stub fails (reminder of past pattern).
        let code = "// let q = tensors.first().map(|t| t.dtype);";
        assert_eq!(
            verdict_from_no_first_tensor_stub(code),
            InsqVerdict::Fail
        );
    }

    #[test]
    fn insq004_pass_empty() {
        assert_eq!(verdict_from_no_first_tensor_stub(""), InsqVerdict::Pass);
    }

    // -------------------------------------------------------------------------
    // Section 6: Realistic — full pipeline.
    // -------------------------------------------------------------------------
    #[test]
    fn realistic_pre_fix_q4km_reports_f32_caught() {
        // Bug: bias dtype reported as quantization.
        let tensors = vec![
            make_tensor("blk.0.attn.q.weight", "Q4_K", 16_777_216),
            make_tensor("blk.0.attn.q.bias", "F32", 4096),
            make_tensor("blk.0.attn_norm.weight", "F32", 4096),
        ];
        // INSPECT-QUANT-001/002:
        assert_eq!(
            verdict_from_quantization_not_default("F32"),
            InsqVerdict::Fail
        );
        // INSPECT-QUANT-003:
        assert_eq!(
            verdict_from_dominant_dtype_agreement("F32", &tensors),
            InsqVerdict::Fail
        );
    }

    #[test]
    fn realistic_post_fix_q4km_full_pipeline() {
        // Post-fix Qwen2.5-Coder-7B Q4_K_M:
        let tensors = vec![
            make_tensor("blk.0.attn.q.weight", "Q4_K", 16_777_216),
            make_tensor("blk.0.attn.k.weight", "Q4_K", 4_194_304),
            make_tensor("blk.0.ffn_gate.weight", "Q4_K", 58_720_256),
            make_tensor("blk.0.attn_norm.weight", "F32", 4096),
            make_tensor("blk.0.attn.q.bias", "F32", 4096),
        ];

        // INSPECT-QUANT-001:
        assert_eq!(
            verdict_from_quantization_not_default("Q4_K"),
            InsqVerdict::Pass
        );
        // INSPECT-QUANT-002 (JSON):
        assert_eq!(
            verdict_from_quantization_not_default("Q4_K_M"),
            InsqVerdict::Pass
        );
        // INSPECT-QUANT-003:
        assert_eq!(
            verdict_from_dominant_dtype_agreement("Q4_K", &tensors),
            InsqVerdict::Pass
        );
        // INSPECT-QUANT-004:
        let clean_code = "fn validate_inspect() { compute_dominant_dtype(); }";
        assert_eq!(
            verdict_from_no_first_tensor_stub(clean_code),
            InsqVerdict::Pass
        );
    }

    #[test]
    fn realistic_first_tensor_bias_bug_caught() {
        // The classic bug: tensors.first() returned the bias; quant
        // reported as F32.
        // We're given the (buggy) reported quant and the tensor list:
        let tensors = vec![
            make_tensor("blk.0.attn.q.bias", "F32", 4096),
            make_tensor("blk.0.attn.q.weight", "Q4_K", 16_777_216),
        ];
        // Both gates Fail:
        assert_eq!(
            verdict_from_quantization_not_default("F32"),
            InsqVerdict::Fail
        );
        assert_eq!(
            verdict_from_dominant_dtype_agreement("F32", &tensors),
            InsqVerdict::Fail
        );
    }
}
