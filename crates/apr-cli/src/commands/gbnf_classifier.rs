//! Grammar-constrained (GBNF) output classifier (CRUX-C-10).
//!
//! Three pure, deterministic classifiers that discharge
//! FALSIFY-CRUX-C-10-{001,002} at the PARTIAL_ALGORITHM_LEVEL:
//!
//!   * `classify_json_grammar_output` — given an already-emitted completion
//!     string and its finish_reason, the string parses as JSON and the
//!     finish_reason is one of {"stop", "length"}.
//!   * `classify_grammar_error_diagnostic` — given `(exit_code, stderr_text)`,
//!     a malformed-grammar invocation exits non-zero and stderr mentions
//!     "grammar" (case-insensitive).
//!   * `classify_illegal_token_masking` — given a proposed next-token logits
//!     row and a `legal_mask` computed by the grammar parser, every illegal
//!     position has a logit that is strictly `-INFINITY` (NaN and any finite
//!     value rejected).
//!
//! Full discharge blocks on `apr run --grammar-file` and
//! `apr serve POST /v1/chat/completions {"grammar": ...}` surfaces
//! integrating a real GBNF parser.

use serde_json::Value;

/// Allowed finish_reason values for grammar-constrained completions.
pub const GBNF_ALLOWED_FINISH_REASONS: &[&str] = &["stop", "length"];

/// Outcome of `classify_json_grammar_output`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum JsonGrammarOutputOutcome {
    /// Completion parses as JSON and finish_reason is stop|length.
    Ok,
    /// The completion string is empty.
    EmptyOutput,
    /// The completion string does not parse as JSON.
    NotJson { error: String },
    /// finish_reason is not one of {"stop", "length"}.
    WrongFinishReason { got: String },
}

/// Grammar-parity gate: output must be valid JSON and finish must be clean.
pub fn classify_json_grammar_output(output: &str, finish_reason: &str) -> JsonGrammarOutputOutcome {
    if output.is_empty() {
        return JsonGrammarOutputOutcome::EmptyOutput;
    }
    if !GBNF_ALLOWED_FINISH_REASONS.contains(&finish_reason) {
        return JsonGrammarOutputOutcome::WrongFinishReason {
            got: finish_reason.to_string(),
        };
    }
    match serde_json::from_str::<Value>(output) {
        Ok(_) => JsonGrammarOutputOutcome::Ok,
        Err(e) => JsonGrammarOutputOutcome::NotJson {
            error: e.to_string(),
        },
    }
}

/// Outcome of `classify_grammar_error_diagnostic`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GrammarErrorDiagnosticOutcome {
    /// Exit code non-zero AND stderr mentions "grammar".
    Ok,
    /// Exit code was zero — malformed grammar was silently accepted.
    ZeroExitCode,
    /// Exit code non-zero but stderr does not mention "grammar".
    MissingGrammarDiagnostic { stderr_snippet: String },
}

/// Malformed-grammar gate: non-zero exit AND "grammar" mentioned in stderr.
pub fn classify_grammar_error_diagnostic(
    exit_code: i32,
    stderr: &str,
) -> GrammarErrorDiagnosticOutcome {
    if exit_code == 0 {
        return GrammarErrorDiagnosticOutcome::ZeroExitCode;
    }
    if stderr.to_lowercase().contains("grammar") {
        return GrammarErrorDiagnosticOutcome::Ok;
    }
    let snippet = stderr.chars().take(160).collect::<String>();
    GrammarErrorDiagnosticOutcome::MissingGrammarDiagnostic {
        stderr_snippet: snippet,
    }
}

/// Outcome of `classify_illegal_token_masking`.
#[derive(Debug, Clone, PartialEq)]
pub enum IllegalTokenMaskingOutcome {
    /// Every illegal position has logit == -INFINITY.
    Ok,
    /// Logits row and legal_mask have different lengths.
    LengthMismatch { logits_len: usize, mask_len: usize },
    /// No legal tokens in the mask (grammar stuck — different defect class).
    NoLegalTokens,
    /// An illegal position has a finite (or +INF, or NaN) logit.
    IllegalTokenNotMasked { token_index: usize, logit: f32 },
}

/// Logit-masking gate: illegal-at-state-s tokens have logit = `-INFINITY`.
/// Any finite value, `+INFINITY`, or `NaN` at an illegal position is rejected.
pub fn classify_illegal_token_masking(
    logits: &[f32],
    legal_mask: &[bool],
) -> IllegalTokenMaskingOutcome {
    if logits.len() != legal_mask.len() {
        return IllegalTokenMaskingOutcome::LengthMismatch {
            logits_len: logits.len(),
            mask_len: legal_mask.len(),
        };
    }
    if !legal_mask.iter().any(|&b| b) {
        return IllegalTokenMaskingOutcome::NoLegalTokens;
    }
    for (i, (&lg, &ok)) in logits.iter().zip(legal_mask.iter()).enumerate() {
        if !ok && lg != f32::NEG_INFINITY {
            return IllegalTokenMaskingOutcome::IllegalTokenNotMasked {
                token_index: i,
                logit: lg,
            };
        }
    }
    IllegalTokenMaskingOutcome::Ok
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- JSON grammar output ---------------------------------------------

    #[test]
    fn json_grammar_ok_on_object_with_stop() {
        assert_eq!(
            classify_json_grammar_output("{\"a\":1}", "stop"),
            JsonGrammarOutputOutcome::Ok
        );
    }

    #[test]
    fn json_grammar_ok_on_array_with_length() {
        assert_eq!(
            classify_json_grammar_output("[1,2,3]", "length"),
            JsonGrammarOutputOutcome::Ok
        );
    }

    #[test]
    fn json_grammar_rejects_empty_output() {
        assert_eq!(
            classify_json_grammar_output("", "stop"),
            JsonGrammarOutputOutcome::EmptyOutput
        );
    }

    #[test]
    fn json_grammar_rejects_non_json_text() {
        match classify_json_grammar_output("not valid json", "stop") {
            JsonGrammarOutputOutcome::NotJson { error } => {
                assert!(!error.is_empty());
            }
            other => panic!("expected NotJson, got {other:?}"),
        }
    }

    #[test]
    fn json_grammar_rejects_wrong_finish_reason() {
        assert_eq!(
            classify_json_grammar_output("{}", "tool_calls"),
            JsonGrammarOutputOutcome::WrongFinishReason {
                got: "tool_calls".to_string()
            }
        );
    }

    #[test]
    fn json_grammar_rejects_content_filter_reason() {
        assert_eq!(
            classify_json_grammar_output("{}", "content_filter"),
            JsonGrammarOutputOutcome::WrongFinishReason {
                got: "content_filter".to_string()
            }
        );
    }

    #[test]
    fn json_grammar_classifier_is_deterministic() {
        let a = classify_json_grammar_output("{\"k\":null}", "stop");
        let b = classify_json_grammar_output("{\"k\":null}", "stop");
        assert_eq!(a, b);
    }

    // ---- malformed-grammar diagnostic ------------------------------------

    #[test]
    fn grammar_error_ok_on_nonzero_exit_with_keyword() {
        assert_eq!(
            classify_grammar_error_diagnostic(1, "error: invalid grammar at line 1"),
            GrammarErrorDiagnosticOutcome::Ok
        );
    }

    #[test]
    fn grammar_error_ok_case_insensitive() {
        assert_eq!(
            classify_grammar_error_diagnostic(2, "GRAMMAR parse error"),
            GrammarErrorDiagnosticOutcome::Ok
        );
    }

    #[test]
    fn grammar_error_rejects_zero_exit() {
        assert_eq!(
            classify_grammar_error_diagnostic(0, "grammar was fine"),
            GrammarErrorDiagnosticOutcome::ZeroExitCode
        );
    }

    #[test]
    fn grammar_error_rejects_missing_keyword() {
        match classify_grammar_error_diagnostic(1, "unrelated parse error") {
            GrammarErrorDiagnosticOutcome::MissingGrammarDiagnostic { stderr_snippet } => {
                assert!(stderr_snippet.contains("unrelated"));
            }
            other => panic!("expected MissingGrammarDiagnostic, got {other:?}"),
        }
    }

    #[test]
    fn grammar_error_snippet_is_truncated_to_160_chars() {
        let long = "x".repeat(500);
        match classify_grammar_error_diagnostic(1, &long) {
            GrammarErrorDiagnosticOutcome::MissingGrammarDiagnostic { stderr_snippet } => {
                assert_eq!(stderr_snippet.len(), 160);
            }
            other => panic!("expected MissingGrammarDiagnostic, got {other:?}"),
        }
    }

    #[test]
    fn grammar_error_classifier_is_deterministic() {
        let a = classify_grammar_error_diagnostic(1, "grammar bad");
        let b = classify_grammar_error_diagnostic(1, "grammar bad");
        assert_eq!(a, b);
    }

    // ---- illegal-token masking -------------------------------------------

    #[test]
    fn masking_ok_when_illegal_positions_are_neg_infinity() {
        let logits = [1.0, f32::NEG_INFINITY, 2.0, f32::NEG_INFINITY];
        let legal = [true, false, true, false];
        assert_eq!(
            classify_illegal_token_masking(&logits, &legal),
            IllegalTokenMaskingOutcome::Ok
        );
    }

    #[test]
    fn masking_ok_when_all_positions_legal() {
        let logits = [1.0, 2.0, 3.0];
        let legal = [true, true, true];
        assert_eq!(
            classify_illegal_token_masking(&logits, &legal),
            IllegalTokenMaskingOutcome::Ok
        );
    }

    #[test]
    fn masking_rejects_length_mismatch() {
        let logits = [1.0, 2.0];
        let legal = [true, true, false];
        assert_eq!(
            classify_illegal_token_masking(&logits, &legal),
            IllegalTokenMaskingOutcome::LengthMismatch {
                logits_len: 2,
                mask_len: 3
            }
        );
    }

    #[test]
    fn masking_rejects_empty_legal_mask() {
        let logits = [f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY];
        let legal = [false, false, false];
        assert_eq!(
            classify_illegal_token_masking(&logits, &legal),
            IllegalTokenMaskingOutcome::NoLegalTokens
        );
    }

    #[test]
    fn masking_rejects_finite_logit_at_illegal_position() {
        let logits = [1.0, 2.0, 3.0];
        let legal = [true, false, true];
        match classify_illegal_token_masking(&logits, &legal) {
            IllegalTokenMaskingOutcome::IllegalTokenNotMasked { token_index, logit } => {
                assert_eq!(token_index, 1);
                assert!((logit - 2.0).abs() < 1e-9);
            }
            other => panic!("expected IllegalTokenNotMasked, got {other:?}"),
        }
    }

    #[test]
    fn masking_rejects_positive_infinity_at_illegal_position() {
        let logits = [1.0, f32::INFINITY, 3.0];
        let legal = [true, false, true];
        match classify_illegal_token_masking(&logits, &legal) {
            IllegalTokenMaskingOutcome::IllegalTokenNotMasked { token_index, logit } => {
                assert_eq!(token_index, 1);
                assert!(logit.is_infinite() && logit.is_sign_positive());
            }
            other => panic!("expected IllegalTokenNotMasked, got {other:?}"),
        }
    }

    #[test]
    fn masking_rejects_nan_at_illegal_position() {
        let logits = [1.0, f32::NAN, 3.0];
        let legal = [true, false, true];
        match classify_illegal_token_masking(&logits, &legal) {
            IllegalTokenMaskingOutcome::IllegalTokenNotMasked { token_index, logit } => {
                assert_eq!(token_index, 1);
                assert!(logit.is_nan());
            }
            other => panic!("expected IllegalTokenNotMasked, got {other:?}"),
        }
    }

    #[test]
    fn masking_classifier_is_deterministic() {
        let logits = [1.0, f32::NEG_INFINITY];
        let legal = [true, false];
        let a = classify_illegal_token_masking(&logits, &legal);
        let b = classify_illegal_token_masking(&logits, &legal);
        assert_eq!(a, b);
    }

    // ---- constants -------------------------------------------------------

    #[test]
    fn allowed_finish_reasons_are_stop_or_length() {
        assert_eq!(GBNF_ALLOWED_FINISH_REASONS, &["stop", "length"]);
    }
}
