// `apr-compare-hf-nonvacuous-v1` algorithm-level PARTIAL discharge for
// the 3 vacuous-truth-prevention falsifiers (0-tensor exit non-zero,
// no PASS phrase on 0 tensors, mapping diagnostic on 0 tensors).
//
// Contract: `contracts/apr-compare-hf-nonvacuous-v1.yaml`.
// Refs: paiml/aprender#621 (`compare-hf reports '✓ All tensors match'
// when it compared 0 tensors`).
//
// ## What this file proves NOW (`PARTIAL_ALGORITHM_LEVEL`)
//
// Three pure decision predicates that pin the contract's invariants:
// vacuous truth at the library level (∀ ∅ = true) MUST be guarded at
// the CLI boundary. A user running `apr compare-hf` wants to VERIFY,
// and "verified nothing" is not a successful verification.

/// PASS-phrase tokens that MUST NOT appear on 0-tensor output.
pub const AC_COMPAREHF_FORBIDDEN_PASS_PHRASES: [&str; 2] =
    ["all tensors match", "✓"];

/// Diagnostic-keyword set that MUST appear on 0-tensor output.
pub const AC_COMPAREHF_REQUIRED_DIAGNOSTIC_KEYWORDS: [&str; 4] =
    ["mapping", "no tensors matched", "0 tensors", "name-mapping"];

// =============================================================================
// FALSIFY-COMPARE-HF-001 — 0-tensor comparison exits non-zero
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NonzeroExitOnVacuousVerdict {
    /// total_compared == 0 ⇒ exit_code != 0.
    /// total_compared >= 1 ⇒ exit_code reflects pass/fail (any value).
    Pass,
    /// total_compared == 0 AND exit_code == 0 — vacuous PASS.
    Fail,
}

#[must_use]
pub fn verdict_from_nonzero_exit_on_vacuous(
    total_compared: u32,
    exit_code: i32,
) -> NonzeroExitOnVacuousVerdict {
    if total_compared == 0 && exit_code == 0 {
        NonzeroExitOnVacuousVerdict::Fail
    } else {
        NonzeroExitOnVacuousVerdict::Pass
    }
}

// =============================================================================
// FALSIFY-COMPARE-HF-002 — no PASS phrase on 0-tensor output
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NoPassPhraseOnVacuousVerdict {
    /// 0-tensor case: output does NOT contain "all tensors match" / "✓".
    /// Non-zero compared: any text is acceptable.
    Pass,
    /// 0-tensor case but PASS phrase leaked.
    Fail,
}

#[must_use]
pub fn verdict_from_no_pass_phrase_on_vacuous(
    total_compared: u32,
    output: &str,
) -> NoPassPhraseOnVacuousVerdict {
    if total_compared > 0 {
        return NoPassPhraseOnVacuousVerdict::Pass;
    }
    let lower = output.to_lowercase();
    for phrase in AC_COMPAREHF_FORBIDDEN_PASS_PHRASES {
        if lower.contains(phrase) {
            return NoPassPhraseOnVacuousVerdict::Fail;
        }
    }
    NoPassPhraseOnVacuousVerdict::Pass
}

// =============================================================================
// FALSIFY-COMPARE-HF-003 — mapping diagnostic on 0-tensor output
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MappingDiagnosticVerdict {
    /// 0-tensor case: output contains a mapping/zero-tensors hint.
    /// Non-zero compared: any text is acceptable.
    Pass,
    /// 0-tensor case but no diagnostic keyword found.
    Fail,
}

#[must_use]
pub fn verdict_from_mapping_diagnostic(
    total_compared: u32,
    output: &str,
) -> MappingDiagnosticVerdict {
    if total_compared > 0 {
        return MappingDiagnosticVerdict::Pass;
    }
    let lower = output.to_lowercase();
    for kw in AC_COMPAREHF_REQUIRED_DIAGNOSTIC_KEYWORDS {
        if lower.contains(kw) {
            return MappingDiagnosticVerdict::Pass;
        }
    }
    MappingDiagnosticVerdict::Fail
}

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Section 1: Provenance pins.
    // -------------------------------------------------------------------------
    #[test]
    fn provenance_forbidden_pass_phrases_count_2() {
        assert_eq!(AC_COMPAREHF_FORBIDDEN_PASS_PHRASES.len(), 2);
    }

    #[test]
    fn provenance_required_diagnostics_count_4() {
        assert_eq!(AC_COMPAREHF_REQUIRED_DIAGNOSTIC_KEYWORDS.len(), 4);
    }

    // -------------------------------------------------------------------------
    // Section 2: COMPARE-HF-001 non-zero exit on vacuous.
    // -------------------------------------------------------------------------
    #[test]
    fn fc001_pass_zero_compared_nonzero_exit() {
        assert_eq!(
            verdict_from_nonzero_exit_on_vacuous(0, 1),
            NonzeroExitOnVacuousVerdict::Pass
        );
    }

    #[test]
    fn fc001_pass_one_compared_zero_exit() {
        // Non-vacuous comparison; exit 0 is fine if all passed.
        assert_eq!(
            verdict_from_nonzero_exit_on_vacuous(1, 0),
            NonzeroExitOnVacuousVerdict::Pass
        );
    }

    #[test]
    fn fc001_pass_many_compared_failed_exit() {
        // Non-vacuous comparison with some failures: non-zero exit OK.
        assert_eq!(
            verdict_from_nonzero_exit_on_vacuous(339, 1),
            NonzeroExitOnVacuousVerdict::Pass
        );
    }

    #[test]
    fn fc001_fail_zero_compared_zero_exit() {
        // The exact GH-621 regression class.
        assert_eq!(
            verdict_from_nonzero_exit_on_vacuous(0, 0),
            NonzeroExitOnVacuousVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 3: COMPARE-HF-002 no PASS phrase on vacuous.
    // -------------------------------------------------------------------------
    #[test]
    fn fc002_pass_nonvacuous_with_pass_phrase() {
        // Non-vacuous + "all tensors match" is correct.
        let out = "Total tensors compared: 339\n✓ All tensors match within threshold!";
        assert_eq!(
            verdict_from_no_pass_phrase_on_vacuous(339, out),
            NoPassPhraseOnVacuousVerdict::Pass
        );
    }

    #[test]
    fn fc002_pass_vacuous_with_error_text() {
        // Vacuous with error text (no PASS phrase).
        let out = "Total tensors compared: 0\nError: tensor name mapping failed";
        assert_eq!(
            verdict_from_no_pass_phrase_on_vacuous(0, out),
            NoPassPhraseOnVacuousVerdict::Pass
        );
    }

    #[test]
    fn fc002_fail_vacuous_with_pass_phrase() {
        // The exact regression: 0 compared but "All tensors match" emitted.
        let out = "Total tensors compared: 0\n✓ All tensors match within threshold!";
        assert_eq!(
            verdict_from_no_pass_phrase_on_vacuous(0, out),
            NoPassPhraseOnVacuousVerdict::Fail
        );
    }

    #[test]
    fn fc002_fail_vacuous_with_only_checkmark() {
        let out = "Total: 0\n✓";
        assert_eq!(
            verdict_from_no_pass_phrase_on_vacuous(0, out),
            NoPassPhraseOnVacuousVerdict::Fail
        );
    }

    #[test]
    fn fc002_fail_case_insensitive_pass_phrase() {
        let out = "Total: 0\nALL TENSORS MATCH";
        assert_eq!(
            verdict_from_no_pass_phrase_on_vacuous(0, out),
            NoPassPhraseOnVacuousVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 4: COMPARE-HF-003 mapping diagnostic on vacuous.
    // -------------------------------------------------------------------------
    #[test]
    fn fc003_pass_nonvacuous_no_diagnostic_required() {
        let out = "Total: 339\n✓ All match";
        assert_eq!(
            verdict_from_mapping_diagnostic(339, out),
            MappingDiagnosticVerdict::Pass
        );
    }

    #[test]
    fn fc003_pass_vacuous_with_mapping_keyword() {
        let out = "Total: 0\nError: name mapping issue between GGUF and HF";
        assert_eq!(
            verdict_from_mapping_diagnostic(0, out),
            MappingDiagnosticVerdict::Pass
        );
    }

    #[test]
    fn fc003_pass_vacuous_with_zero_tensors_keyword() {
        let out = "Total: 0\nError: 0 tensors matched any name pattern";
        assert_eq!(
            verdict_from_mapping_diagnostic(0, out),
            MappingDiagnosticVerdict::Pass
        );
    }

    #[test]
    fn fc003_pass_vacuous_with_no_tensors_matched_keyword() {
        let out = "Total: 0\nNo tensors matched between local and HF.";
        assert_eq!(
            verdict_from_mapping_diagnostic(0, out),
            MappingDiagnosticVerdict::Pass
        );
    }

    #[test]
    fn fc003_fail_vacuous_no_diagnostic() {
        // The regression: 0 compared but no hint why.
        let out = "Total: 0\nDone.";
        assert_eq!(
            verdict_from_mapping_diagnostic(0, out),
            MappingDiagnosticVerdict::Fail
        );
    }

    #[test]
    fn fc003_fail_vacuous_empty_output() {
        assert_eq!(
            verdict_from_mapping_diagnostic(0, ""),
            MappingDiagnosticVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 5: Realistic — full healthy run + GH-621 regression.
    // -------------------------------------------------------------------------
    #[test]
    fn realistic_healthy_compare_passes_all_3() {
        // Successful 339-tensor comparison.
        let out = "Total tensors compared: 339\n✓ All tensors match within threshold!";
        assert_eq!(
            verdict_from_nonzero_exit_on_vacuous(339, 0),
            NonzeroExitOnVacuousVerdict::Pass
        );
        assert_eq!(
            verdict_from_no_pass_phrase_on_vacuous(339, out),
            NoPassPhraseOnVacuousVerdict::Pass
        );
        assert_eq!(
            verdict_from_mapping_diagnostic(339, out),
            MappingDiagnosticVerdict::Pass
        );
    }

    #[test]
    fn realistic_healthy_zero_tensor_with_diagnostic_passes_all_3() {
        // Post-fix: 0 tensors with non-zero exit + diagnostic + no PASS phrase.
        let out = "Total tensors compared: 0\nError: tensor name mapping returned 0 matches between local GGUF and HF.";
        assert_eq!(
            verdict_from_nonzero_exit_on_vacuous(0, 1),
            NonzeroExitOnVacuousVerdict::Pass
        );
        assert_eq!(
            verdict_from_no_pass_phrase_on_vacuous(0, out),
            NoPassPhraseOnVacuousVerdict::Pass
        );
        assert_eq!(
            verdict_from_mapping_diagnostic(0, out),
            MappingDiagnosticVerdict::Pass
        );
    }

    #[test]
    fn realistic_pre_fix_gh_621_all_3_fail() {
        // The exact GH-621 symptom string.
        let out = "Total tensors compared: 0\n✓ All tensors match within threshold!";
        assert_eq!(
            verdict_from_nonzero_exit_on_vacuous(0, 0),
            NonzeroExitOnVacuousVerdict::Fail
        );
        assert_eq!(
            verdict_from_no_pass_phrase_on_vacuous(0, out),
            NoPassPhraseOnVacuousVerdict::Fail
        );
        assert_eq!(
            verdict_from_mapping_diagnostic(0, out),
            MappingDiagnosticVerdict::Fail
        );
    }
}
