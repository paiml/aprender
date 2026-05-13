// SHIP-TWO-001 — `apr-inspect-flags-v1` algorithm-level PARTIAL
// discharge for FALSIFY-INSPECT-FLAGS-001..004.
//
// Contract: `contracts/apr-inspect-flags-v1.yaml`.
// Spec: `docs/specifications/aprender-train/ship-two-models-spec.md`.
//
// ## What this file proves NOW (PARTIAL_ALGORITHM_LEVEL)
//
// Four `apr inspect` flag-materiality gates (paiml/aprender#604, #609):
//
// - INSPECT-FLAGS-001 (--vocab changes output).
// - INSPECT-FLAGS-002 (--weights changes output).
// - INSPECT-FLAGS-003 (--filters <pat> changes output).
// - INSPECT-FLAGS-004 (dispatcher signature exactly 5 args).

/// `run_rosetta_inspect` must accept exactly these 5 parameters.
pub const AC_INSF_004_REQUIRED_ARGS: &[&str] = &[
    "path",
    "show_vocab",
    "show_filters",
    "show_weights",
    "json_output",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InsfVerdict {
    Pass,
    Fail,
}

// -----------------------------------------------------------------------------
// Verdict 1, 2, 3: --vocab/--weights/--filters materiality.
// -----------------------------------------------------------------------------

/// Pass iff `output_with_flag != output_without_flag`.
#[must_use]
pub fn verdict_from_flag_materiality(
    output_without_flag: &str,
    output_with_flag: &str,
) -> InsfVerdict {
    if output_with_flag != output_without_flag {
        InsfVerdict::Pass
    } else {
        InsfVerdict::Fail
    }
}

// -----------------------------------------------------------------------------
// Verdict 4: dispatcher signature.
// -----------------------------------------------------------------------------

/// Pass iff `actual_args` exactly matches `AC_INSF_004_REQUIRED_ARGS`.
#[must_use]
pub fn verdict_from_dispatcher_signature(actual_args: &[&str]) -> InsfVerdict {
    if actual_args == AC_INSF_004_REQUIRED_ARGS {
        InsfVerdict::Pass
    } else {
        InsfVerdict::Fail
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Section 1: Provenance pin.
    // -------------------------------------------------------------------------
    #[test]
    fn provenance_required_args_5() {
        assert_eq!(AC_INSF_004_REQUIRED_ARGS.len(), 5);
    }

    #[test]
    fn provenance_required_args_order() {
        assert_eq!(
            AC_INSF_004_REQUIRED_ARGS,
            &["path", "show_vocab", "show_filters", "show_weights", "json_output"]
        );
    }

    // -------------------------------------------------------------------------
    // Section 2: INSPECT-FLAGS-001/002/003 — flag materiality.
    // -------------------------------------------------------------------------
    #[test]
    fn flags_pass_vocab_changes_output() {
        let no_flag = "Architecture: qwen2\nLayers: 28\n";
        let with_flag = "Architecture: qwen2\nLayers: 28\nVocab: 32K tokens\n";
        assert_eq!(
            verdict_from_flag_materiality(no_flag, with_flag),
            InsfVerdict::Pass
        );
    }

    #[test]
    fn flags_pass_weights_changes_output() {
        let no_flag = "Header: ...\n";
        let with_flag = "Header: ...\nTensor 0: blk.0.attn.q.weight [4096, 4096]\n";
        assert_eq!(
            verdict_from_flag_materiality(no_flag, with_flag),
            InsfVerdict::Pass
        );
    }

    #[test]
    fn flags_pass_filters_changes_output() {
        let no_flag = "Tensor count: 339\n";
        let with_flag = "Tensor count: 339\nFiltered (blk.0): 9 tensors\n";
        assert_eq!(
            verdict_from_flag_materiality(no_flag, with_flag),
            InsfVerdict::Pass
        );
    }

    #[test]
    fn flags_fail_silently_dropped_flag() {
        // The exact regression: paiml/aprender#604, #609 —
        // dispatcher dropped the flag and produced identical output.
        let identical = "Architecture: qwen2\nLayers: 28\n";
        assert_eq!(
            verdict_from_flag_materiality(identical, identical),
            InsfVerdict::Fail
        );
    }

    #[test]
    fn flags_fail_both_empty() {
        // Both outputs empty — vacuous "no difference".
        assert_eq!(
            verdict_from_flag_materiality("", ""),
            InsfVerdict::Fail
        );
    }

    #[test]
    fn flags_pass_minor_diff() {
        // Even one byte of difference is sufficient evidence the flag
        // was dispatched.
        assert_eq!(
            verdict_from_flag_materiality("a", "b"),
            InsfVerdict::Pass
        );
    }

    // -------------------------------------------------------------------------
    // Section 3: INSPECT-FLAGS-004 — dispatcher signature.
    // -------------------------------------------------------------------------
    #[test]
    fn insf004_pass_canonical_5_args() {
        let args = vec!["path", "show_vocab", "show_filters", "show_weights", "json_output"];
        assert_eq!(
            verdict_from_dispatcher_signature(&args),
            InsfVerdict::Pass
        );
    }

    #[test]
    fn insf004_fail_truncated_to_4_args() {
        // Bug: dispatcher missing show_weights.
        let args = vec!["path", "show_vocab", "show_filters", "json_output"];
        assert_eq!(
            verdict_from_dispatcher_signature(&args),
            InsfVerdict::Fail
        );
    }

    #[test]
    fn insf004_fail_truncated_to_3_args() {
        let args = vec!["path", "show_vocab", "show_filters"];
        assert_eq!(
            verdict_from_dispatcher_signature(&args),
            InsfVerdict::Fail
        );
    }

    #[test]
    fn insf004_fail_extra_arg() {
        let args = vec![
            "path",
            "show_vocab",
            "show_filters",
            "show_weights",
            "json_output",
            "extra_param",
        ];
        assert_eq!(
            verdict_from_dispatcher_signature(&args),
            InsfVerdict::Fail
        );
    }

    #[test]
    fn insf004_fail_wrong_order() {
        // Args swapped — even with same 5 names, order matters.
        let args = vec!["path", "show_filters", "show_vocab", "show_weights", "json_output"];
        assert_eq!(
            verdict_from_dispatcher_signature(&args),
            InsfVerdict::Fail
        );
    }

    #[test]
    fn insf004_fail_renamed_arg() {
        let args = vec!["path", "vocab_flag", "show_filters", "show_weights", "json_output"];
        assert_eq!(
            verdict_from_dispatcher_signature(&args),
            InsfVerdict::Fail
        );
    }

    #[test]
    fn insf004_fail_empty() {
        let args: Vec<&str> = vec![];
        assert_eq!(
            verdict_from_dispatcher_signature(&args),
            InsfVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 4: Realistic — full bug regression scenarios.
    // -------------------------------------------------------------------------
    #[test]
    fn realistic_604_vocab_dropped_caught() {
        // paiml/aprender#604: --vocab silently dropped in dispatcher.
        // Both outputs identical despite flag.
        let same = "Architecture: qwen2\nLayers: 28\nVocab: <not shown>\n";
        assert_eq!(
            verdict_from_flag_materiality(same, same),
            InsfVerdict::Fail
        );
    }

    #[test]
    fn realistic_609_weights_dropped_caught() {
        // paiml/aprender#609: --weights silently dropped.
        let same = "Header: ...\nTensor count: 339\n";
        assert_eq!(
            verdict_from_flag_materiality(same, same),
            InsfVerdict::Fail
        );
    }

    #[test]
    fn realistic_post_fix_full_pipeline_passes() {
        // Post-fix dispatcher accepts all 5 args.
        let args = AC_INSF_004_REQUIRED_ARGS.to_vec();
        assert_eq!(
            verdict_from_dispatcher_signature(&args),
            InsfVerdict::Pass
        );
        // All 3 flags now produce material output diffs.
        assert_eq!(
            verdict_from_flag_materiality("base", "base+vocab"),
            InsfVerdict::Pass
        );
        assert_eq!(
            verdict_from_flag_materiality("base", "base+weights"),
            InsfVerdict::Pass
        );
        assert_eq!(
            verdict_from_flag_materiality("base", "base+filters"),
            InsfVerdict::Pass
        );
    }
}
