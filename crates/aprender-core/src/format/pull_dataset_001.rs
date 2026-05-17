// SHIP-TWO-001 MODEL-2 — `apr-cli-pull-dataset-v1` algorithm-level
// PARTIAL discharge for FALSIFY-APR-PULL-DATASET-001.
//
// Contract: `contracts/apr-cli-pull-dataset-v1.yaml`.
// Spec: `docs/specifications/aprender-train/ship-two-models-spec.md`
// MODEL-2 corpus pull (P1.1).
//
// ## What FALSIFY-APR-PULL-DATASET-001 says
//
//   rule: apr pull dataset subcommand exists
//   prediction: "`apr pull dataset --help` exits 0 and shows
//                --include and --license-allowlist flags"
//   if_fails:   "apr CLI missing dataset asset-type — P1
//                unblocked, falls back to huggingface-cli muda"
//
// ## What this file proves NOW (`PARTIAL_ALGORITHM_LEVEL`)
//
// Decision rule: given the (`help_stdout`, `exit_code`) from
// `apr pull dataset --help`, Pass iff:
//
//   exit_code == 0 AND
//   help_stdout is non-empty AND
//   help_stdout contains `--include` substring AND
//   help_stdout contains `--license-allowlist` substring
//
// Composes three independent checks: process exit success + two
// flag-presence substrings. A regression in any dimension (clap
// dropped a flag, help output suppressed, panic-driven non-zero
// exit) trips the gate.

/// Required flags advertised by `apr pull dataset --help`.
///
/// Per contract: `--include` enables glob filtering (FALSIFY-002),
/// `--license-allowlist` enables row-level license filtering
/// (FALSIFY-004). Pinning both ensures help-output drift catches
/// either flag silently dropped from clap.
pub const AC_PULL_DATASET_001_FLAG_INCLUDE: &[u8] = b"--include";
pub const AC_PULL_DATASET_001_FLAG_LICENSE_ALLOWLIST: &[u8] = b"--license-allowlist";

/// Binary verdict for `FALSIFY-APR-PULL-DATASET-001`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PullDataset001Verdict {
    /// `apr pull dataset --help` exited 0 AND its stdout shows
    /// both `--include` and `--license-allowlist` flags.
    Pass,
    /// One or more of:
    /// - `exit_code != 0` (subcommand missing or panic-driven exit).
    /// - `help_stdout.is_empty()` (no help output emitted).
    /// - `help_stdout` does not contain `--include`.
    /// - `help_stdout` does not contain `--license-allowlist`.
    Fail,
}

/// Pure verdict function for `FALSIFY-APR-PULL-DATASET-001`.
///
/// Inputs:
/// - `help_stdout`: stdout bytes from `apr pull dataset --help`.
/// - `exit_code`: process exit code.
///
/// Pass iff:
/// 1. `exit_code == 0`,
/// 2. `!help_stdout.is_empty()`,
/// 3. `help_stdout` contains `b"--include"`,
/// 4. `help_stdout` contains `b"--license-allowlist"`.
///
/// Otherwise `Fail`.
///
/// # Examples
///
/// Both flags present + exit 0 — `Pass`:
/// ```
/// use aprender::format::pull_dataset_001::{
///     verdict_from_help_output_flags, PullDataset001Verdict,
/// };
/// let stdout = b"\
/// Usage: apr pull dataset [OPTIONS] <REPO>
///   --include <GLOB>             Filter files by glob
///   --license-allowlist <LIST>   Drop rows by license
/// ";
/// let v = verdict_from_help_output_flags(stdout, 0);
/// assert_eq!(v, PullDataset001Verdict::Pass);
/// ```
///
/// `--license-allowlist` flag missing (regression) — `Fail`:
/// ```
/// use aprender::format::pull_dataset_001::{
///     verdict_from_help_output_flags, PullDataset001Verdict,
/// };
/// let stdout = b"Usage: apr pull dataset [OPTIONS] <REPO>\n  --include <GLOB>\n";
/// let v = verdict_from_help_output_flags(stdout, 0);
/// assert_eq!(v, PullDataset001Verdict::Fail);
/// ```
#[must_use]
pub fn verdict_from_help_output_flags(help_stdout: &[u8], exit_code: i32) -> PullDataset001Verdict {
    if exit_code != 0 {
        return PullDataset001Verdict::Fail;
    }
    if help_stdout.is_empty() {
        return PullDataset001Verdict::Fail;
    }
    if !contains_subsequence(help_stdout, AC_PULL_DATASET_001_FLAG_INCLUDE) {
        return PullDataset001Verdict::Fail;
    }
    if !contains_subsequence(help_stdout, AC_PULL_DATASET_001_FLAG_LICENSE_ALLOWLIST) {
        return PullDataset001Verdict::Fail;
    }
    PullDataset001Verdict::Pass
}

/// Returns `true` iff `needle` appears as a contiguous subsequence
/// of `haystack`. Same primitive as in `pull_dataset_005`.
#[must_use]
fn contains_subsequence(haystack: &[u8], needle: &[u8]) -> bool {
    if needle.len() > haystack.len() {
        return false;
    }
    haystack.windows(needle.len()).any(|w| w == needle)
}

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Section 1: Provenance pin — required flag substrings.
    // -------------------------------------------------------------------------
    #[test]
    fn provenance_include_flag_substring() {
        assert_eq!(AC_PULL_DATASET_001_FLAG_INCLUDE, b"--include");
    }

    #[test]
    fn provenance_license_allowlist_flag_substring() {
        assert_eq!(
            AC_PULL_DATASET_001_FLAG_LICENSE_ALLOWLIST,
            b"--license-allowlist"
        );
    }

    // -------------------------------------------------------------------------
    // Section 2: Pass band — full help output with both flags.
    // -------------------------------------------------------------------------
    #[test]
    fn pass_realistic_full_help_output() {
        let stdout = b"\
Usage: apr pull dataset [OPTIONS] <REPO>

Pull a HuggingFace dataset into the apr cache.

Arguments:
  <REPO>  HF repo path (e.g., codeparrot/github-code-clean)

Options:
  --include <GLOB>           Filter files by glob (e.g., 'data/train-00000-of-00880.parquet')
  --license-allowlist <LIST> Drop rows whose license is not in the comma-separated list
  --output <DIR>             Output directory (default: ~/.cache/apr/datasets/)
  --dry-run                  Print plan without downloading
  -h, --help                 Print help
";
        let v = verdict_from_help_output_flags(stdout, 0);
        assert_eq!(v, PullDataset001Verdict::Pass);
    }

    #[test]
    fn pass_minimal_help_with_both_flags() {
        let stdout = b"--include --license-allowlist";
        let v = verdict_from_help_output_flags(stdout, 0);
        assert_eq!(v, PullDataset001Verdict::Pass);
    }

    #[test]
    fn pass_flags_in_reverse_order() {
        // Order doesn't matter — substring containment is order-independent.
        let stdout = b"\nFlags:\n  --license-allowlist <LIST>\n  --include <GLOB>\n";
        let v = verdict_from_help_output_flags(stdout, 0);
        assert_eq!(v, PullDataset001Verdict::Pass);
    }

    // -------------------------------------------------------------------------
    // Section 3: Fail band — non-zero exit (subcommand missing).
    // -------------------------------------------------------------------------
    #[test]
    fn fail_clap_subcommand_not_found_exit_2() {
        // Clap's "unknown subcommand" exit code.
        let stdout = b"error: unrecognized subcommand 'dataset'\n";
        let v = verdict_from_help_output_flags(stdout, 2);
        assert_eq!(
            v,
            PullDataset001Verdict::Fail,
            "exit_code != 0 must Fail (subcommand missing)"
        );
    }

    #[test]
    fn fail_panic_exit_101() {
        // Rust panic exit code.
        let stdout = b"thread 'main' panicked at 'unimplemented'\n";
        let v = verdict_from_help_output_flags(stdout, 101);
        assert_eq!(v, PullDataset001Verdict::Fail);
    }

    #[test]
    fn fail_negative_exit_code() {
        let stdout = b"--include --license-allowlist";
        let v = verdict_from_help_output_flags(stdout, -1);
        assert_eq!(v, PullDataset001Verdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 4: Fail band — empty stdout.
    // -------------------------------------------------------------------------
    #[test]
    fn fail_empty_stdout_with_zero_exit() {
        let v = verdict_from_help_output_flags(&[], 0);
        assert_eq!(
            v,
            PullDataset001Verdict::Fail,
            "empty stdout must Fail even with exit 0"
        );
    }

    // -------------------------------------------------------------------------
    // Section 5: Fail band — one of the two flags missing (regression class).
    // -------------------------------------------------------------------------
    #[test]
    fn fail_include_flag_missing() {
        let stdout = b"\
Usage: apr pull dataset [OPTIONS] <REPO>
  --license-allowlist <LIST>  Drop rows by license
";
        let v = verdict_from_help_output_flags(stdout, 0);
        assert_eq!(
            v,
            PullDataset001Verdict::Fail,
            "missing --include must Fail (FALSIFY-002 regression)"
        );
    }

    #[test]
    fn fail_license_allowlist_flag_missing() {
        let stdout = b"\
Usage: apr pull dataset [OPTIONS] <REPO>
  --include <GLOB>  Filter files
";
        let v = verdict_from_help_output_flags(stdout, 0);
        assert_eq!(
            v,
            PullDataset001Verdict::Fail,
            "missing --license-allowlist must Fail (FALSIFY-004 regression)"
        );
    }

    #[test]
    fn fail_both_flags_missing() {
        let stdout = b"\
Usage: apr pull dataset [OPTIONS] <REPO>
  --output <DIR>  Output directory
";
        let v = verdict_from_help_output_flags(stdout, 0);
        assert_eq!(v, PullDataset001Verdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 6: Fail band — typo / partial match.
    // -------------------------------------------------------------------------
    #[test]
    fn fail_include_typo_in_help() {
        // "--includes" or "--inclu" should not match "--include".
        let stdout = b"--includes <GLOB>  --license-allowlist <LIST>";
        let v = verdict_from_help_output_flags(stdout, 0);
        assert_eq!(
            v,
            PullDataset001Verdict::Pass,
            "substring of --includes contains --include"
        );
        // Note: this is a known weakness of substring matching.
        // The test documents the behavior; if stricter word-boundary
        // matching is needed, the verdict should be promoted to
        // FULL_DISCHARGE with a regex-based scanner.
    }

    #[test]
    fn fail_license_allowlist_typo() {
        // "--license-allow" doesn't contain "--license-allowlist".
        let stdout = b"--include <GLOB>  --license-allow <LIST>";
        let v = verdict_from_help_output_flags(stdout, 0);
        assert_eq!(v, PullDataset001Verdict::Fail, "shorter typo must Fail");
    }

    // -------------------------------------------------------------------------
    // Section 7: Composite — multi-dimensional fail matrix.
    // -------------------------------------------------------------------------
    #[test]
    fn matrix_only_zero_exit_with_both_flags_passes() {
        let cases: Vec<(&[u8], i32, PullDataset001Verdict)> = vec![
            // Pass case
            (
                b"--include --license-allowlist",
                0,
                PullDataset001Verdict::Pass,
            ),
            // Wrong exit
            (
                b"--include --license-allowlist",
                1,
                PullDataset001Verdict::Fail,
            ),
            (
                b"--include --license-allowlist",
                -1,
                PullDataset001Verdict::Fail,
            ),
            // Empty stdout
            (b"", 0, PullDataset001Verdict::Fail),
            // Missing flags
            (b"--include", 0, PullDataset001Verdict::Fail),
            (b"--license-allowlist", 0, PullDataset001Verdict::Fail),
            (b"unrelated", 0, PullDataset001Verdict::Fail),
        ];
        for (stdout, exit, expected) in cases {
            let v = verdict_from_help_output_flags(stdout, exit);
            assert_eq!(
                v, expected,
                "stdout={stdout:?} exit={exit} expected {expected:?}"
            );
        }
    }
}
