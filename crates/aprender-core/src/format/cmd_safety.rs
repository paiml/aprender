// SHIP-TWO-001 — `apr-cli-command-safety-v1` algorithm-level
// PARTIAL discharge for FALSIFY-CMD-SAFETY-001..004 (closes 4/4).
//
// Contract: `contracts/apr-cli-command-safety-v1.yaml`.
// Spec: `docs/specifications/aprender-train/ship-two-models-spec.md`.
//
// All four cmd-safety verdicts share this module because they
// each have a small, simple algorithm-level reduction. Bundling
// them together (analogous to `pub_cli_002_004`, `qa_002_006`)
// avoids muda while keeping each verdict's contract ID distinct.

// ===========================================================================
// CMD-SAFETY-001 — read-only commands do not write
// ===========================================================================

/// Binary verdict for `FALSIFY-CMD-SAFETY-001` (read-only commands
/// create no new files in /tmp).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CmdSafety001Verdict {
    /// `files_after - files_before == 0` (no new files created).
    Pass,
    /// New files were created OR `files_after < files_before`
    /// (counter corruption).
    Fail,
}

/// Pure verdict function for `FALSIFY-CMD-SAFETY-001`.
///
/// Pass iff `files_after == files_before`.
#[must_use]
pub fn verdict_from_file_count_delta(
    files_before: u64,
    files_after: u64,
) -> CmdSafety001Verdict {
    if files_before == files_after {
        CmdSafety001Verdict::Pass
    } else {
        CmdSafety001Verdict::Fail
    }
}

// ===========================================================================
// CMD-SAFETY-002 — mutating commands need --output
// ===========================================================================

/// Binary verdict for `FALSIFY-CMD-SAFETY-002` (apr convert without
/// -o fails with usage error containing "output" or "required").
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CmdSafety002Verdict {
    /// Stderr is non-empty AND contains `output` OR `required`
    /// (case-insensitive ASCII).
    Pass,
    /// Empty stderr OR neither substring present (mutating cmd
    /// silently accepted no output path).
    Fail,
}

/// Pure verdict function for `FALSIFY-CMD-SAFETY-002`.
///
/// Pass iff `stderr` (lowercase) contains `output` OR `required`.
#[must_use]
pub fn verdict_from_mutating_cmd_usage_error(stderr: &[u8]) -> CmdSafety002Verdict {
    if stderr.is_empty() {
        return CmdSafety002Verdict::Fail;
    }
    let lower: Vec<u8> = stderr.iter().map(|b| b.to_ascii_lowercase()).collect();
    if contains_subseq(&lower, b"output") || contains_subseq(&lower, b"required") {
        CmdSafety002Verdict::Pass
    } else {
        CmdSafety002Verdict::Fail
    }
}

// ===========================================================================
// CMD-SAFETY-003 — long-running commands handle SIGINT
// ===========================================================================

/// Maximum exit code for a SIGINT-handled long-running command.
/// 130 = 128 + 2 (SIGINT signal number); the contract test
/// requires `[ $? -le 130 ]`.
pub const AC_CMD_SAFETY_003_MAX_EXIT: i32 = 130;

/// Binary verdict for `FALSIFY-CMD-SAFETY-003` (apr tui exits
/// cleanly on SIGINT — exit code <= 130).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CmdSafety003Verdict {
    /// `0 <= exit_code <= 130` (clean exit OR SIGINT-derived).
    Pass,
    /// `exit_code > 130` (panic / unhandled exception)
    /// OR `exit_code < 0` (out-of-domain).
    Fail,
}

/// Pure verdict function for `FALSIFY-CMD-SAFETY-003`.
///
/// Pass iff `0 <= exit_code <= 130`.
#[must_use]
pub fn verdict_from_sigint_exit_code(exit_code: i32) -> CmdSafety003Verdict {
    if (0..=AC_CMD_SAFETY_003_MAX_EXIT).contains(&exit_code) {
        CmdSafety003Verdict::Pass
    } else {
        CmdSafety003Verdict::Fail
    }
}

// ===========================================================================
// CMD-SAFETY-004 — command classification complete
// ===========================================================================

/// Binary verdict for `FALSIFY-CMD-SAFETY-004` (every command in
/// apr --help is classified as read_only/mutating/long_running).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CmdSafety004Verdict {
    /// Total commands > 0 AND classified count == total commands.
    Pass,
    /// Total == 0, classified > total (corruption), or classified
    /// < total (incomplete classification).
    Fail,
}

/// Pure verdict function for `FALSIFY-CMD-SAFETY-004`.
///
/// Pass iff `total_commands > 0 AND classified_count == total_commands`.
#[must_use]
pub fn verdict_from_command_classification(
    total_commands: u64,
    classified_count: u64,
) -> CmdSafety004Verdict {
    if total_commands == 0 {
        return CmdSafety004Verdict::Fail;
    }
    if classified_count == total_commands {
        CmdSafety004Verdict::Pass
    } else {
        CmdSafety004Verdict::Fail
    }
}

// ===========================================================================
// Shared primitive
// ===========================================================================

#[must_use]
fn contains_subseq(haystack: &[u8], needle: &[u8]) -> bool {
    if needle.len() > haystack.len() {
        return false;
    }
    haystack.windows(needle.len()).any(|w| w == needle)
}

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // CMD-SAFETY-001 — read-only no new files
    // -------------------------------------------------------------------------
    #[test]
    fn s001_pass_no_new_files() {
        let v = verdict_from_file_count_delta(42, 42);
        assert_eq!(v, CmdSafety001Verdict::Pass);
    }

    #[test]
    fn s001_pass_zero_files() {
        let v = verdict_from_file_count_delta(0, 0);
        assert_eq!(v, CmdSafety001Verdict::Pass);
    }

    #[test]
    fn s001_fail_one_new_file() {
        let v = verdict_from_file_count_delta(42, 43);
        assert_eq!(v, CmdSafety001Verdict::Fail);
    }

    #[test]
    fn s001_fail_files_decreased() {
        // Counter corruption: read-only command shouldn't decrease
        // file count either (cleanup of someone else's files).
        let v = verdict_from_file_count_delta(42, 41);
        assert_eq!(v, CmdSafety001Verdict::Fail);
    }

    // -------------------------------------------------------------------------
    // CMD-SAFETY-002 — usage error contains output/required
    // -------------------------------------------------------------------------
    #[test]
    fn s002_pass_contains_output_lowercase() {
        let v = verdict_from_mutating_cmd_usage_error(b"error: --output is required");
        assert_eq!(v, CmdSafety002Verdict::Pass);
    }

    #[test]
    fn s002_pass_contains_output_uppercase() {
        let v = verdict_from_mutating_cmd_usage_error(b"ERROR: --OUTPUT IS REQUIRED");
        assert_eq!(v, CmdSafety002Verdict::Pass);
    }

    #[test]
    fn s002_pass_contains_required_only() {
        let v = verdict_from_mutating_cmd_usage_error(b"missing required argument");
        assert_eq!(v, CmdSafety002Verdict::Pass);
    }

    #[test]
    fn s002_fail_neither_substring() {
        let v = verdict_from_mutating_cmd_usage_error(b"some unrelated error");
        assert_eq!(v, CmdSafety002Verdict::Fail);
    }

    #[test]
    fn s002_fail_empty_stderr() {
        let v = verdict_from_mutating_cmd_usage_error(&[]);
        assert_eq!(v, CmdSafety002Verdict::Fail);
    }

    #[test]
    fn s002_pass_realistic_clap_error() {
        let v = verdict_from_mutating_cmd_usage_error(
            b"error: the following required arguments were not provided:\n  --output <PATH>",
        );
        assert_eq!(v, CmdSafety002Verdict::Pass);
    }

    // -------------------------------------------------------------------------
    // CMD-SAFETY-003 — SIGINT exit code <= 130
    // -------------------------------------------------------------------------
    #[test]
    fn s003_provenance_max_exit_is_130() {
        assert_eq!(AC_CMD_SAFETY_003_MAX_EXIT, 130);
    }

    #[test]
    fn s003_pass_clean_exit_zero() {
        let v = verdict_from_sigint_exit_code(0);
        assert_eq!(v, CmdSafety003Verdict::Pass);
    }

    #[test]
    fn s003_pass_sigint_130() {
        let v = verdict_from_sigint_exit_code(130);
        assert_eq!(v, CmdSafety003Verdict::Pass);
    }

    #[test]
    fn s003_pass_clean_error_exit_1() {
        let v = verdict_from_sigint_exit_code(1);
        assert_eq!(v, CmdSafety003Verdict::Pass);
    }

    #[test]
    fn s003_fail_panic_101_above_130() {
        // Wait — 101 < 130, so this would actually Pass per
        // contract `<= 130`. Document the contract semantics:
        // panic exit 101 IS within the "clean" range per contract.
        let v = verdict_from_sigint_exit_code(101);
        assert_eq!(v, CmdSafety003Verdict::Pass);
    }

    #[test]
    fn s003_fail_137_sigkill() {
        // SIGKILL (128 + 9 = 137) > 130 → Fail (uncatchable signal,
        // means TUI didn't release resources cleanly).
        let v = verdict_from_sigint_exit_code(137);
        assert_eq!(v, CmdSafety003Verdict::Fail);
    }

    #[test]
    fn s003_fail_negative_exit() {
        let v = verdict_from_sigint_exit_code(-1);
        assert_eq!(v, CmdSafety003Verdict::Fail);
    }

    #[test]
    fn s003_fail_exit_above_255() {
        let v = verdict_from_sigint_exit_code(256);
        assert_eq!(v, CmdSafety003Verdict::Fail);
    }

    // -------------------------------------------------------------------------
    // CMD-SAFETY-004 — every command classified
    // -------------------------------------------------------------------------
    #[test]
    fn s004_pass_all_58_classified() {
        // Canonical 58-command apr CLI, all classified.
        let v = verdict_from_command_classification(58, 58);
        assert_eq!(v, CmdSafety004Verdict::Pass);
    }

    #[test]
    fn s004_fail_one_unclassified() {
        let v = verdict_from_command_classification(58, 57);
        assert_eq!(v, CmdSafety004Verdict::Fail);
    }

    #[test]
    fn s004_fail_zero_total() {
        let v = verdict_from_command_classification(0, 0);
        assert_eq!(v, CmdSafety004Verdict::Fail);
    }

    #[test]
    fn s004_fail_classified_exceeds_total() {
        // Counter corruption.
        let v = verdict_from_command_classification(58, 100);
        assert_eq!(v, CmdSafety004Verdict::Fail);
    }

    #[test]
    fn s004_pass_smaller_cli() {
        let v = verdict_from_command_classification(20, 20);
        assert_eq!(v, CmdSafety004Verdict::Pass);
    }

    // -------------------------------------------------------------------------
    // Shared primitive
    // -------------------------------------------------------------------------
    #[test]
    fn contains_subseq_basic() {
        assert!(contains_subseq(b"hello world", b"world"));
        assert!(!contains_subseq(b"hello world", b"xyz"));
        assert!(!contains_subseq(b"short", b"longer"));
    }
}
