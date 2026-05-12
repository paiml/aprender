// SHIP-TWO-001 — `apr-cli-publish-v1` algorithm-level PARTIAL
// discharge for FALSIFY-PUB-CLI-003.
//
// Contract: `contracts/apr-cli-publish-v1.yaml`.
// Spec: `docs/specifications/aprender-train/ship-two-models-spec.md`
// (apr CLI publish gate; cross-cutting requirement for MODEL-1 +
// MODEL-2 shipping).
//
// ## What FALSIFY-PUB-CLI-003 says
//
//   rule: apr --help works with minimal features
//   prediction: "apr --help lists all 58 commands"
//   test: "apr --help | wc -l > 50"
//   if_fails: "commands missing from minimal build"
//
// ## What this file proves NOW (`PARTIAL_ALGORITHM_LEVEL`)
//
// Decision rule: given `(help_stdout_line_count, exit_code)`,
// Pass iff:
//
//   exit_code == 0 AND
//   help_stdout_line_count > AC_PUB_CLI_003_MIN_HELP_LINES (50)
//
// Strict `>` matches the contract's `wc -l > 50` test wording.
// `apr --help` output for the canonical 58-command CLI is ~70+
// lines (header, usage, commands, options, footer); a regression
// dropping below 50 indicates dropped commands / suppressed
// output / a panic that printed only the panic banner.

/// Minimum number of lines `apr --help` must emit.
///
/// Per contract `FALSIFY-PUB-CLI-003`: 50-line floor catches a
/// regression where commands are missing from the minimal-features
/// build. The canonical 58-command CLI prints ~70+ lines (header,
/// commands list, options, footer); 50 leaves headroom while
/// still rejecting a stripped-down build.
pub const AC_PUB_CLI_003_MIN_HELP_LINES: u64 = 50;

/// Binary verdict for `FALSIFY-PUB-CLI-003`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PubCli003Verdict {
    /// `apr --help` exited 0 AND emitted strictly more than 50
    /// lines of output.
    Pass,
    /// One or more of:
    /// - `exit_code != 0` (apr panicked or `--help` not wired).
    /// - `help_stdout_line_count <= 50` (commands missing from
    ///   build, or output suppressed).
    Fail,
}

/// Pure verdict function for `FALSIFY-PUB-CLI-003`.
///
/// Inputs:
/// - `help_stdout_line_count`: count of newline-terminated lines
///   emitted by `apr --help` to stdout.
/// - `exit_code`: process exit code of `apr --help`.
///
/// Pass iff:
/// 1. `exit_code == 0`,
/// 2. `help_stdout_line_count > 50` (strict, matches `wc -l > 50`).
///
/// Otherwise `Fail`.
///
/// # Examples
///
/// 75 lines + exit 0 (canonical 58-cmd build) — `Pass`:
/// ```
/// use aprender::format::pub_cli_003::{
///     verdict_from_help_command_listing, PubCli003Verdict,
/// };
/// let v = verdict_from_help_command_listing(75, 0);
/// assert_eq!(v, PubCli003Verdict::Pass);
/// ```
///
/// 30 lines (commands stripped from minimal build) — `Fail`:
/// ```
/// use aprender::format::pub_cli_003::{
///     verdict_from_help_command_listing, PubCli003Verdict,
/// };
/// let v = verdict_from_help_command_listing(30, 0);
/// assert_eq!(v, PubCli003Verdict::Fail);
/// ```
#[must_use]
pub fn verdict_from_help_command_listing(
    help_stdout_line_count: u64,
    exit_code: i32,
) -> PubCli003Verdict {
    if exit_code != 0 {
        return PubCli003Verdict::Fail;
    }
    if help_stdout_line_count > AC_PUB_CLI_003_MIN_HELP_LINES {
        PubCli003Verdict::Pass
    } else {
        PubCli003Verdict::Fail
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Section 1: Provenance pin — 50-line floor.
    // -------------------------------------------------------------------------
    #[test]
    fn provenance_min_help_lines_is_50() {
        assert_eq!(AC_PUB_CLI_003_MIN_HELP_LINES, 50);
    }

    // -------------------------------------------------------------------------
    // Section 2: Pass band — canonical 58-command builds.
    // -------------------------------------------------------------------------
    #[test]
    fn pass_canonical_75_lines() {
        // Realistic apr --help output for 58 commands.
        let v = verdict_from_help_command_listing(75, 0);
        assert_eq!(v, PubCli003Verdict::Pass);
    }

    #[test]
    fn pass_at_51_lines() {
        // Just above the strict floor.
        let v = verdict_from_help_command_listing(51, 0);
        assert_eq!(v, PubCli003Verdict::Pass);
    }

    #[test]
    fn pass_with_full_features_100_lines() {
        let v = verdict_from_help_command_listing(100, 0);
        assert_eq!(v, PubCli003Verdict::Pass);
    }

    #[test]
    fn pass_at_huge_line_count() {
        // Sanity at large scale.
        let v = verdict_from_help_command_listing(10_000, 0);
        assert_eq!(v, PubCli003Verdict::Pass);
    }

    // -------------------------------------------------------------------------
    // Section 3: Fail band — exit code non-zero (apr panicked).
    // -------------------------------------------------------------------------
    #[test]
    fn fail_clap_unknown_arg_exit_2() {
        // Even with plenty of output (e.g., clap printed help-on-error),
        // non-zero exit must Fail.
        let v = verdict_from_help_command_listing(75, 2);
        assert_eq!(
            v,
            PubCli003Verdict::Fail,
            "exit != 0 must Fail (clap rejected --help wiring)"
        );
    }

    #[test]
    fn fail_panic_exit_101() {
        let v = verdict_from_help_command_listing(75, 101);
        assert_eq!(v, PubCli003Verdict::Fail);
    }

    #[test]
    fn fail_negative_exit() {
        let v = verdict_from_help_command_listing(75, -1);
        assert_eq!(v, PubCli003Verdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 4: Fail band — line count at or below floor (commands missing).
    // -------------------------------------------------------------------------
    #[test]
    fn fail_at_exact_floor_50_lines() {
        // Strict `>`: 50 is on the fail side per contract `> 50`.
        let v = verdict_from_help_command_listing(50, 0);
        assert_eq!(
            v,
            PubCli003Verdict::Fail,
            "exact 50 lines must Fail (strict > per contract)"
        );
    }

    #[test]
    fn fail_at_49_lines() {
        let v = verdict_from_help_command_listing(49, 0);
        assert_eq!(v, PubCli003Verdict::Fail);
    }

    #[test]
    fn fail_at_30_lines_minimal_build() {
        // Minimal-features build with most commands stripped.
        let v = verdict_from_help_command_listing(30, 0);
        assert_eq!(v, PubCli003Verdict::Fail);
    }

    #[test]
    fn fail_at_zero_lines() {
        // No output at all.
        let v = verdict_from_help_command_listing(0, 0);
        assert_eq!(v, PubCli003Verdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 5: Fail band — combined failures.
    // -------------------------------------------------------------------------
    #[test]
    fn fail_low_lines_and_nonzero_exit() {
        let v = verdict_from_help_command_listing(10, 1);
        assert_eq!(v, PubCli003Verdict::Fail);
    }

    #[test]
    fn fail_zero_lines_with_nonzero_exit() {
        let v = verdict_from_help_command_listing(0, 2);
        assert_eq!(v, PubCli003Verdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 6: Boundary sweep — line count around the 50-line cutoff.
    // -------------------------------------------------------------------------
    #[test]
    fn line_count_sweep_around_floor() {
        let probes: Vec<(u64, PubCli003Verdict)> = vec![
            (0, PubCli003Verdict::Fail),
            (1, PubCli003Verdict::Fail),
            (10, PubCli003Verdict::Fail),
            (40, PubCli003Verdict::Fail),
            (49, PubCli003Verdict::Fail),
            (50, PubCli003Verdict::Fail), // exact floor → Fail (strict >)
            (51, PubCli003Verdict::Pass), // just above → Pass
            (75, PubCli003Verdict::Pass),
            (100, PubCli003Verdict::Pass),
            (1_000_000, PubCli003Verdict::Pass),
        ];
        for (lines, expected) in probes {
            let v = verdict_from_help_command_listing(lines, 0);
            assert_eq!(v, expected, "lines={lines} expected {expected:?}");
        }
    }

    // -------------------------------------------------------------------------
    // Section 7: Composite — exit_code × line_count matrix.
    // -------------------------------------------------------------------------
    #[test]
    fn matrix_only_zero_exit_with_lines_above_floor_passes() {
        let cases: Vec<(u64, i32, PubCli003Verdict)> = vec![
            (75, 0, PubCli003Verdict::Pass),
            (51, 0, PubCli003Verdict::Pass),
            (50, 0, PubCli003Verdict::Fail),
            (75, 1, PubCli003Verdict::Fail),
            (75, -1, PubCli003Verdict::Fail),
            (75, 101, PubCli003Verdict::Fail),
            (0, 0, PubCli003Verdict::Fail),
            (0, 1, PubCli003Verdict::Fail),
            (u64::MAX, 0, PubCli003Verdict::Pass),
            (u64::MAX, 1, PubCli003Verdict::Fail),
        ];
        for (lines, exit, expected) in cases {
            let v = verdict_from_help_command_listing(lines, exit);
            assert_eq!(
                v, expected,
                "lines={lines} exit={exit} expected {expected:?}"
            );
        }
    }
}
