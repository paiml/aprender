// SHIP-TWO-001 — `apr-list-quiet-wiring-v1` algorithm-level PARTIAL
// discharge for FALSIFY-LIST-QUIET-001..003.
//
// Contract: `contracts/apr-list-quiet-wiring-v1.yaml`.
// Spec: `docs/specifications/aprender-train/ship-two-models-spec.md`.
//
// ## What this file proves NOW (PARTIAL_ALGORITHM_LEVEL)
//
// Three `apr list --quiet` wiring gates:
//
// - LIST-QUIET-001 (--quiet output differs from no-flag output).
// - LIST-QUIET-002 (--quiet omits help text — "Pull a model with:" /
//   "=== Cached Models ===").
// - LIST-QUIET-003 (--quiet output is terse — every line ≤ 200 chars,
//   no tabular padding).

/// Forbidden help-text substrings in --quiet output.
pub const AC_LISTQ_002_FORBIDDEN_PHRASES: [&str; 2] =
    ["Pull a model with", "=== Cached Models ==="];

/// Maximum line length in --quiet output (per FALSIFY-LIST-QUIET-003).
pub const AC_LISTQ_003_MAX_LINE_LEN: usize = 200;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ListqVerdict {
    Pass,
    Fail,
}

// -----------------------------------------------------------------------------
// Verdict 1: LIST-QUIET-001 — --quiet differs from no-flag.
// -----------------------------------------------------------------------------

/// Pass iff `output_quiet != output_no_flag`.
#[must_use]
pub fn verdict_from_quiet_differs_from_default(
    output_no_flag: &str,
    output_quiet: &str,
) -> ListqVerdict {
    if output_quiet != output_no_flag {
        ListqVerdict::Pass
    } else {
        ListqVerdict::Fail
    }
}

// -----------------------------------------------------------------------------
// Verdict 2: LIST-QUIET-002 — --quiet omits help text.
// -----------------------------------------------------------------------------

/// Pass iff `output_quiet` contains NONE of the forbidden phrases.
#[must_use]
pub fn verdict_from_quiet_omits_help_text(output_quiet: &str) -> ListqVerdict {
    for phrase in AC_LISTQ_002_FORBIDDEN_PHRASES {
        if output_quiet.contains(phrase) {
            return ListqVerdict::Fail;
        }
    }
    ListqVerdict::Pass
}

// -----------------------------------------------------------------------------
// Verdict 3: LIST-QUIET-003 — terse line length.
// -----------------------------------------------------------------------------

/// Pass iff every line in `output_quiet` has length ≤ 200 characters.
/// Empty input passes vacuously.
#[must_use]
pub fn verdict_from_quiet_terse_lines(output_quiet: &str) -> ListqVerdict {
    for line in output_quiet.lines() {
        if line.len() > AC_LISTQ_003_MAX_LINE_LEN {
            return ListqVerdict::Fail;
        }
    }
    ListqVerdict::Pass
}

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Section 1: Provenance pins.
    // -------------------------------------------------------------------------
    #[test]
    fn provenance_forbidden_phrases() {
        assert_eq!(AC_LISTQ_002_FORBIDDEN_PHRASES.len(), 2);
        assert!(AC_LISTQ_002_FORBIDDEN_PHRASES.contains(&"Pull a model with"));
        assert!(AC_LISTQ_002_FORBIDDEN_PHRASES.contains(&"=== Cached Models ==="));
    }

    #[test]
    fn provenance_max_line_len_200() {
        assert_eq!(AC_LISTQ_003_MAX_LINE_LEN, 200);
    }

    // -------------------------------------------------------------------------
    // Section 2: LIST-QUIET-001 — output differs.
    // -------------------------------------------------------------------------
    #[test]
    fn listq001_pass_different_outputs() {
        let no_flag = "=== Cached Models ===\nqwen2.5-coder.gguf\nPull a model with: apr pull <model>";
        let quiet = "qwen2.5-coder.gguf";
        assert_eq!(
            verdict_from_quiet_differs_from_default(no_flag, quiet),
            ListqVerdict::Pass
        );
    }

    #[test]
    fn listq001_pass_minor_diff() {
        // Even single-byte difference suffices.
        assert_eq!(
            verdict_from_quiet_differs_from_default("a", "b"),
            ListqVerdict::Pass
        );
    }

    #[test]
    fn listq001_fail_quiet_is_no_op() {
        // The exact regression: --quiet flag silently dropped.
        let same = "=== Cached Models ===\nqwen2.5-coder.gguf";
        assert_eq!(
            verdict_from_quiet_differs_from_default(same, same),
            ListqVerdict::Fail
        );
    }

    #[test]
    fn listq001_fail_both_empty() {
        assert_eq!(
            verdict_from_quiet_differs_from_default("", ""),
            ListqVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 3: LIST-QUIET-002 — omits help text.
    // -------------------------------------------------------------------------
    #[test]
    fn listq002_pass_clean_quiet_output() {
        let quiet = "qwen2.5-coder.gguf\nqwen3-30b.apr\ntiny-llama.safetensors";
        assert_eq!(
            verdict_from_quiet_omits_help_text(quiet),
            ListqVerdict::Pass
        );
    }

    #[test]
    fn listq002_pass_empty() {
        assert_eq!(
            verdict_from_quiet_omits_help_text(""),
            ListqVerdict::Pass
        );
    }

    #[test]
    fn listq002_fail_pull_a_model_with_leaked() {
        let quiet = "qwen2.5-coder.gguf\nPull a model with: apr pull <name>";
        assert_eq!(
            verdict_from_quiet_omits_help_text(quiet),
            ListqVerdict::Fail
        );
    }

    #[test]
    fn listq002_fail_table_header_leaked() {
        let quiet = "=== Cached Models ===\nqwen2.5-coder.gguf";
        assert_eq!(
            verdict_from_quiet_omits_help_text(quiet),
            ListqVerdict::Fail
        );
    }

    #[test]
    fn listq002_fail_both_phrases_leaked() {
        let quiet = "=== Cached Models ===\nm.gguf\nPull a model with: apr pull";
        assert_eq!(
            verdict_from_quiet_omits_help_text(quiet),
            ListqVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 4: LIST-QUIET-003 — terse lines.
    // -------------------------------------------------------------------------
    #[test]
    fn listq003_pass_short_lines() {
        let quiet = "model1.gguf\nmodel2.apr\nmodel3.safetensors";
        assert_eq!(
            verdict_from_quiet_terse_lines(quiet),
            ListqVerdict::Pass
        );
    }

    #[test]
    fn listq003_pass_empty() {
        assert_eq!(verdict_from_quiet_terse_lines(""), ListqVerdict::Pass);
    }

    #[test]
    fn listq003_pass_single_long_path_under_200() {
        let p = "a".repeat(195);
        assert_eq!(verdict_from_quiet_terse_lines(&p), ListqVerdict::Pass);
    }

    #[test]
    fn listq003_pass_at_boundary_200() {
        let p = "a".repeat(200);
        assert_eq!(verdict_from_quiet_terse_lines(&p), ListqVerdict::Pass);
    }

    #[test]
    fn listq003_fail_line_above_200() {
        let p = "a".repeat(201);
        assert_eq!(verdict_from_quiet_terse_lines(&p), ListqVerdict::Fail);
    }

    #[test]
    fn listq003_fail_tabular_padding_long_line() {
        // Simulated tabular output: 250-char line with table padding.
        let line = "│ qwen2.5-coder ".to_string() + &" ".repeat(220) + " │";
        assert_eq!(
            verdict_from_quiet_terse_lines(&line),
            ListqVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 5: Realistic — full bug regression scenarios.
    // -------------------------------------------------------------------------
    #[test]
    fn realistic_pre_fix_quiet_no_op_caught() {
        // Pre-fix: dispatcher dropped cli.quiet.
        let same = "=== Cached Models ===\n│ Name │ Size │\nqwen2.gguf";
        assert_eq!(
            verdict_from_quiet_differs_from_default(same, same),
            ListqVerdict::Fail
        );
    }

    #[test]
    fn realistic_help_text_leak_caught() {
        let quiet = "qwen2.gguf\n\nPull a model with: apr pull <name>\n";
        assert_eq!(
            verdict_from_quiet_omits_help_text(quiet),
            ListqVerdict::Fail
        );
    }

    #[test]
    fn realistic_tabular_padding_leak_caught() {
        let quiet = "│ ".to_string() + &"x".repeat(250) + " │";
        assert_eq!(
            verdict_from_quiet_terse_lines(&quiet),
            ListqVerdict::Fail
        );
    }

    #[test]
    fn realistic_post_fix_pipeline_passes_all_3_gates() {
        // Post-fix: terse identifier-per-line output.
        let no_flag = "=== Cached Models ===\nqwen2.5-coder.gguf — 4.2GB\nqwen3-30b.apr — 18GB\nPull a model with: apr pull <name>";
        let quiet = "qwen2.5-coder.gguf\nqwen3-30b.apr";

        // Gate 1:
        assert_eq!(
            verdict_from_quiet_differs_from_default(no_flag, quiet),
            ListqVerdict::Pass
        );
        // Gate 2:
        assert_eq!(
            verdict_from_quiet_omits_help_text(quiet),
            ListqVerdict::Pass
        );
        // Gate 3:
        assert_eq!(
            verdict_from_quiet_terse_lines(quiet),
            ListqVerdict::Pass
        );
    }

    #[test]
    fn realistic_pre_fix_all_3_gates_fail() {
        // Pre-fix: --quiet is no-op AND emits help text AND tabular.
        let same_padded = "=== Cached Models ===\n│ ".to_string()
            + &"x".repeat(250)
            + " │\nPull a model with: apr pull";

        // All 3 gates Fail.
        assert_eq!(
            verdict_from_quiet_differs_from_default(&same_padded, &same_padded),
            ListqVerdict::Fail
        );
        assert_eq!(
            verdict_from_quiet_omits_help_text(&same_padded),
            ListqVerdict::Fail
        );
        assert_eq!(
            verdict_from_quiet_terse_lines(&same_padded),
            ListqVerdict::Fail
        );
    }
}
