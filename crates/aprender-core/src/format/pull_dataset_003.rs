// SHIP-TWO-001 MODEL-2 — `apr-cli-pull-dataset-v1` algorithm-level
// PARTIAL discharge for FALSIFY-APR-PULL-DATASET-003.
//
// Contract: `contracts/apr-cli-pull-dataset-v1.yaml`.
// Spec: `docs/specifications/aprender-train/ship-two-models-spec.md`
// MODEL-2 corpus pull (P1.1).
//
// ## What FALSIFY-APR-PULL-DATASET-003 says
//
//   rule: no-match glob fails fast
//   prediction: "`apr pull dataset <repo> --include 'no/such/file/*'`
//                exits non-zero"
//   if_fails:   "silent no-op on bad glob hides typos and produces
//                empty corpus"
//
// ## What this file proves NOW (`PARTIAL_ALGORITHM_LEVEL`)
//
// Decision rule: given the observed `(matched_count, exit_code)`
// pair from `apr pull dataset --include <bad_glob>`, Pass iff:
//
//   matched_count == 0 AND exit_code != 0
//
// AND `exit_code` is in the canonical exit-code range (0..=255).
// The contract requires fail-fast on bad glob: zero matches must
// produce a non-zero exit. A silent-no-op regression (matched=0,
// exit=0) trips the gate. Conversely, if matched>0 then this
// gate is misapplied — the caller should be using FALSIFY-002,
// not -003 — so we refuse `matched_count > 0` as caller error.

/// Maximum legal POSIX exit-code value (`u8`).
///
/// `apr` uses `std::process::exit(code)` which wraps to u8 modulo
/// 256. Pinning the cap catches a regression where a panic-driven
/// exit produces a value > 255, OR where signed exit codes are
/// passed through unchecked.
pub const AC_PULL_DATASET_003_MAX_EXIT_CODE: i32 = 255;

/// Binary verdict for `FALSIFY-APR-PULL-DATASET-003`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PullDataset003Verdict {
    /// Glob matched zero files AND `apr` exited with a non-zero
    /// code (in canonical 1..=255 range).
    Pass,
    /// One or more of:
    /// - `matched_count > 0` (caller error — this gate is for
    ///   no-match scenarios; use FALSIFY-002 for match-count gates).
    /// - `exit_code == 0` (silent no-op — the regression class
    ///   the contract pin guards against).
    /// - `exit_code < 0` (caller error — exit codes are non-negative).
    /// - `exit_code > 255` (caller error — POSIX exit codes are u8).
    Fail,
}

/// Pure verdict function for `FALSIFY-APR-PULL-DATASET-003`.
///
/// Inputs:
/// - `matched_count`: number of files the include glob matched
///   (expected to be 0 for a no-match glob like
///   `'no/such/file/*'`).
/// - `exit_code`: process exit code of `apr pull dataset` after
///   the no-match glob (expected to be non-zero per contract).
///
/// Pass iff:
/// 1. `matched_count == 0` (no-match scenario),
/// 2. `exit_code >= 1` (non-zero per contract),
/// 3. `exit_code <= 255` (POSIX exit-code domain).
///
/// Otherwise `Fail`.
///
/// # Examples
///
/// No-match glob, exit code 2 — `Pass`:
/// ```
/// use aprender::format::pull_dataset_003::{
///     verdict_from_no_match_fail_fast, PullDataset003Verdict,
/// };
/// let v = verdict_from_no_match_fail_fast(0, 2);
/// assert_eq!(v, PullDataset003Verdict::Pass);
/// ```
///
/// Silent no-op (matched=0 but exit=0) — `Fail`:
/// ```
/// use aprender::format::pull_dataset_003::{
///     verdict_from_no_match_fail_fast, PullDataset003Verdict,
/// };
/// let v = verdict_from_no_match_fail_fast(0, 0);
/// assert_eq!(v, PullDataset003Verdict::Fail);
/// ```
#[must_use]
pub fn verdict_from_no_match_fail_fast(
    matched_count: u64,
    exit_code: i32,
) -> PullDataset003Verdict {
    if matched_count > 0 {
        return PullDataset003Verdict::Fail;
    }
    if exit_code <= 0 {
        return PullDataset003Verdict::Fail;
    }
    if exit_code > AC_PULL_DATASET_003_MAX_EXIT_CODE {
        return PullDataset003Verdict::Fail;
    }
    PullDataset003Verdict::Pass
}

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Section 1: Provenance pin — POSIX exit-code cap.
    // -------------------------------------------------------------------------
    #[test]
    fn provenance_max_exit_code_is_255() {
        assert_eq!(AC_PULL_DATASET_003_MAX_EXIT_CODE, 255);
    }

    // -------------------------------------------------------------------------
    // Section 2: Pass band — no-match + non-zero exit.
    // -------------------------------------------------------------------------
    #[test]
    fn pass_canonical_no_match_exit_2() {
        // Conventional clap-style "user error" exit code.
        let v = verdict_from_no_match_fail_fast(0, 2);
        assert_eq!(v, PullDataset003Verdict::Pass);
    }

    #[test]
    fn pass_no_match_exit_1() {
        // Generic "error" exit code.
        let v = verdict_from_no_match_fail_fast(0, 1);
        assert_eq!(v, PullDataset003Verdict::Pass);
    }

    #[test]
    fn pass_no_match_exit_at_max() {
        // 255 is the inclusive POSIX cap.
        let v = verdict_from_no_match_fail_fast(0, 255);
        assert_eq!(v, PullDataset003Verdict::Pass);
    }

    #[test]
    fn pass_no_match_exit_signal_term_128() {
        // 128 + signal_no convention for kill-derived exits;
        // 128 + 8 = 136 (SIGFPE), 128 + 9 = 137 (SIGKILL), etc.
        // Still within 1..=255.
        let v = verdict_from_no_match_fail_fast(0, 128);
        assert_eq!(v, PullDataset003Verdict::Pass);
    }

    // -------------------------------------------------------------------------
    // Section 3: Fail band — silent no-op (the contract regression).
    // -------------------------------------------------------------------------
    #[test]
    fn fail_silent_no_op_exit_zero() {
        // The exact regression the contract guards against: bad
        // glob, zero matches, exit 0 = silent success.
        let v = verdict_from_no_match_fail_fast(0, 0);
        assert_eq!(
            v,
            PullDataset003Verdict::Fail,
            "matched=0 + exit=0 must Fail (silent no-op regression)"
        );
    }

    // -------------------------------------------------------------------------
    // Section 4: Fail band — caller error (matched_count > 0 means wrong gate).
    // -------------------------------------------------------------------------
    #[test]
    fn fail_one_match_with_zero_exit() {
        // Matched 1 file but exit=0 — this is FALSIFY-002's domain
        // (use that gate for match-count). Refuse here.
        let v = verdict_from_no_match_fail_fast(1, 0);
        assert_eq!(v, PullDataset003Verdict::Fail);
    }

    #[test]
    fn fail_one_match_with_nonzero_exit() {
        // Matched 1 file but exit=1 — also FALSIFY-002's domain.
        let v = verdict_from_no_match_fail_fast(1, 1);
        assert_eq!(
            v,
            PullDataset003Verdict::Fail,
            "matched > 0 must Fail (use FALSIFY-002 for match-count)"
        );
    }

    #[test]
    fn fail_many_matches() {
        let v = verdict_from_no_match_fail_fast(880, 0);
        assert_eq!(v, PullDataset003Verdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 5: Fail band — exit-code domain violations.
    // -------------------------------------------------------------------------
    #[test]
    fn fail_negative_exit_code() {
        let v = verdict_from_no_match_fail_fast(0, -1);
        assert_eq!(
            v,
            PullDataset003Verdict::Fail,
            "negative exit must Fail (POSIX domain)"
        );
    }

    #[test]
    fn fail_exit_code_above_255() {
        let v = verdict_from_no_match_fail_fast(0, 256);
        assert_eq!(
            v,
            PullDataset003Verdict::Fail,
            "exit > 255 must Fail (POSIX cap)"
        );
    }

    #[test]
    fn fail_huge_negative_exit() {
        let v = verdict_from_no_match_fail_fast(0, i32::MIN);
        assert_eq!(v, PullDataset003Verdict::Fail);
    }

    #[test]
    fn fail_huge_positive_exit() {
        let v = verdict_from_no_match_fail_fast(0, i32::MAX);
        assert_eq!(v, PullDataset003Verdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 6: Boundary sweep — exit-code from -1 to 256.
    // -------------------------------------------------------------------------
    #[test]
    fn exit_code_sweep_at_zero_match() {
        let probes: Vec<(i32, PullDataset003Verdict)> = vec![
            (-1, PullDataset003Verdict::Fail),
            (0, PullDataset003Verdict::Fail), // silent no-op
            (1, PullDataset003Verdict::Pass), // canonical fail-fast
            (2, PullDataset003Verdict::Pass), // clap-style
            (10, PullDataset003Verdict::Pass),
            (100, PullDataset003Verdict::Pass),
            (255, PullDataset003Verdict::Pass), // inclusive cap
            (256, PullDataset003Verdict::Fail),
            (1000, PullDataset003Verdict::Fail),
        ];
        for (exit, expected) in probes {
            let v = verdict_from_no_match_fail_fast(0, exit);
            assert_eq!(v, expected, "exit={exit} expected {expected:?}");
        }
    }

    // -------------------------------------------------------------------------
    // Section 7: Composite — match_count × exit_code matrix on key cells.
    // -------------------------------------------------------------------------
    #[test]
    fn matrix_only_zero_match_with_canonical_nonzero_exit_passes() {
        let cases: Vec<(u64, i32, PullDataset003Verdict)> = vec![
            (0, 0, PullDataset003Verdict::Fail),        // silent no-op
            (0, 1, PullDataset003Verdict::Pass),        // contract pass
            (0, 2, PullDataset003Verdict::Pass),        // contract pass
            (0, -1, PullDataset003Verdict::Fail),       // bad exit
            (0, 256, PullDataset003Verdict::Fail),      // bad exit
            (1, 0, PullDataset003Verdict::Fail),        // wrong gate
            (1, 1, PullDataset003Verdict::Fail),        // wrong gate
            (100, 2, PullDataset003Verdict::Fail),      // wrong gate
            (u64::MAX, 1, PullDataset003Verdict::Fail), // wrong gate
        ];
        for (matched, exit, expected) in cases {
            let v = verdict_from_no_match_fail_fast(matched, exit);
            assert_eq!(
                v, expected,
                "matched={matched} exit={exit} expected {expected:?}"
            );
        }
    }
}
