// SHIP-TWO-001 — `apr-cli-qa-v1` algorithm-level PARTIAL discharge
// for FALSIFY-QA-002 AND FALSIFY-QA-006.
//
// Contract: `contracts/apr-cli-qa-v1.yaml`.
// Spec: `docs/specifications/aprender-train/ship-two-models-spec.md`
// (apr CLI QA gates).
//
// ## What FALSIFY-QA-002 says
//
//   rule: missing model exits non-zero
//   prediction: "apr inspect /nonexistent exits 1"
//   if_fails: "missing file not detected"
//
// ## What FALSIFY-QA-006 says
//
//   rule: exit code honesty
//   prediction: "error output implies non-zero exit"
//   if_fails: "exit-code lie"
//
// ## What this file proves NOW (`PARTIAL_ALGORITHM_LEVEL`)
//
// Decision rule for BOTH gates: given the `exit_code` of an
// invocation that was expected to fail (missing model OR error
// output present), Pass iff:
//
//   exit_code != 0 AND exit_code is in canonical [-128, 255] range
//
// Single shared verdict because both gates have identical
// algorithm-level shape: pure exit-code-is-non-zero predicate.
// Differs from `pub_cli_002_004` which inverts the polarity
// (success-only Pass).

/// Maximum legal POSIX exit-code value.
pub const AC_QA_002_006_MAX_EXIT_CODE: i32 = 255;

/// Minimum legal POSIX exit-code value (signed via 128 + signal_no
/// or some test harnesses passing `-1`). Real POSIX is 0..=255 but
/// some CI tools normalize negative codes; we accept that band but
/// it must still be non-zero.
pub const AC_QA_002_006_MIN_EXIT_CODE: i32 = -128;

/// Binary verdict for `FALSIFY-QA-002` AND `FALSIFY-QA-006`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qa002006Verdict {
    /// `exit_code != 0` AND `exit_code` is in `[-128, 255]`.
    Pass,
    /// One or more of:
    /// - `exit_code == 0` (the regression — "exit-code lie": apr
    ///   said success but should have failed).
    /// - `exit_code < -128` (out of POSIX domain — exit-code
    ///   corruption).
    /// - `exit_code > 255` (out of POSIX domain — panic-truncation
    ///   bug or signed-exit passthrough).
    Fail,
}

/// Pure verdict function for `FALSIFY-QA-002` and `FALSIFY-QA-006`.
///
/// Inputs:
/// - `exit_code`: process exit code of an invocation that SHOULD
///   have failed:
///     * QA-002: `apr inspect /nonexistent/model.gguf`
///     * QA-006: `apr validate /nonexistent` (or any error-output
///       producing invocation)
///
/// Pass iff:
/// 1. `exit_code != 0` (non-zero per contract — error-exit honesty),
/// 2. `-128 <= exit_code <= 255` (POSIX domain).
///
/// Otherwise `Fail`.
///
/// # Examples
///
/// Missing-file error, exit 1 — `Pass`:
/// ```
/// use aprender::format::qa_002_006::{
///     verdict_from_error_exit_nonzero, Qa002006Verdict,
/// };
/// let v = verdict_from_error_exit_nonzero(1);
/// assert_eq!(v, Qa002006Verdict::Pass);
/// ```
///
/// Exit-code lie (apr printed error but exited 0) — `Fail`:
/// ```
/// use aprender::format::qa_002_006::{
///     verdict_from_error_exit_nonzero, Qa002006Verdict,
/// };
/// let v = verdict_from_error_exit_nonzero(0);
/// assert_eq!(v, Qa002006Verdict::Fail);
/// ```
#[must_use]
pub fn verdict_from_error_exit_nonzero(exit_code: i32) -> Qa002006Verdict {
    if exit_code == 0 {
        return Qa002006Verdict::Fail;
    }
    if !(AC_QA_002_006_MIN_EXIT_CODE..=AC_QA_002_006_MAX_EXIT_CODE).contains(&exit_code) {
        return Qa002006Verdict::Fail;
    }
    Qa002006Verdict::Pass
}

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Section 1: Provenance pin — POSIX exit-code range.
    // -------------------------------------------------------------------------
    #[test]
    fn provenance_max_exit_code_is_255() {
        assert_eq!(AC_QA_002_006_MAX_EXIT_CODE, 255);
    }

    #[test]
    fn provenance_min_exit_code_is_neg_128() {
        assert_eq!(AC_QA_002_006_MIN_EXIT_CODE, -128);
    }

    // -------------------------------------------------------------------------
    // Section 2: Pass band — canonical error exits.
    // -------------------------------------------------------------------------
    #[test]
    fn pass_exit_one_generic_error() {
        // The contract example: `apr inspect /nonexistent` exits 1.
        let v = verdict_from_error_exit_nonzero(1);
        assert_eq!(v, Qa002006Verdict::Pass);
    }

    #[test]
    fn pass_exit_two_clap_user_error() {
        let v = verdict_from_error_exit_nonzero(2);
        assert_eq!(v, Qa002006Verdict::Pass);
    }

    #[test]
    fn pass_exit_101_panic() {
        // Rust panic exit code.
        let v = verdict_from_error_exit_nonzero(101);
        assert_eq!(v, Qa002006Verdict::Pass);
    }

    #[test]
    fn pass_exit_signal_terminated() {
        // 128 + 9 = 137 SIGKILL (e.g., OOM); 128 + 15 = 143 SIGTERM.
        let v = verdict_from_error_exit_nonzero(137);
        assert_eq!(v, Qa002006Verdict::Pass);
    }

    #[test]
    fn pass_exit_negative_one_signed_passthrough() {
        // Some test harnesses pass through signed exit; -1 is in
        // [-128, 255] domain.
        let v = verdict_from_error_exit_nonzero(-1);
        assert_eq!(v, Qa002006Verdict::Pass);
    }

    // -------------------------------------------------------------------------
    // Section 3: Fail band — exit_code == 0 (the regression).
    // -------------------------------------------------------------------------
    #[test]
    fn fail_exit_code_lie_zero() {
        // The exact regression: apr printed an error but exited 0.
        // Catches QA-006's "exit-code lie" and QA-002's "missing
        // file not detected".
        let v = verdict_from_error_exit_nonzero(0);
        assert_eq!(
            v,
            Qa002006Verdict::Fail,
            "exit==0 must Fail (exit-code lie)"
        );
    }

    // -------------------------------------------------------------------------
    // Section 4: Fail band — exit-code domain violations.
    // -------------------------------------------------------------------------
    #[test]
    fn fail_exit_above_255() {
        let v = verdict_from_error_exit_nonzero(256);
        assert_eq!(v, Qa002006Verdict::Fail);
    }

    #[test]
    fn fail_exit_below_neg_128() {
        let v = verdict_from_error_exit_nonzero(-129);
        assert_eq!(v, Qa002006Verdict::Fail);
    }

    #[test]
    fn fail_exit_i32_max() {
        let v = verdict_from_error_exit_nonzero(i32::MAX);
        assert_eq!(v, Qa002006Verdict::Fail);
    }

    #[test]
    fn fail_exit_i32_min() {
        let v = verdict_from_error_exit_nonzero(i32::MIN);
        assert_eq!(v, Qa002006Verdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 5: Boundary sweep — -128 to 256.
    // -------------------------------------------------------------------------
    #[test]
    fn exit_code_sweep_around_zero() {
        let probes: Vec<(i32, Qa002006Verdict)> = vec![
            (-129, Qa002006Verdict::Fail), // out of domain
            (-128, Qa002006Verdict::Pass), // domain min, non-zero
            (-1, Qa002006Verdict::Pass),
            (0, Qa002006Verdict::Fail), // exit-code lie
            (1, Qa002006Verdict::Pass), // canonical error
            (2, Qa002006Verdict::Pass),
            (101, Qa002006Verdict::Pass),
            (137, Qa002006Verdict::Pass),
            (255, Qa002006Verdict::Pass), // domain max
            (256, Qa002006Verdict::Fail), // out of domain
            (1000, Qa002006Verdict::Fail),
        ];
        for (exit, expected) in probes {
            let v = verdict_from_error_exit_nonzero(exit);
            assert_eq!(v, expected, "exit={exit} expected {expected:?}");
        }
    }

    // -------------------------------------------------------------------------
    // Section 6: Domain — Pass iff exit != 0 in canonical range.
    // -------------------------------------------------------------------------
    #[test]
    fn pass_iff_nonzero_and_in_domain() {
        for exit in [1_i32, 2, 100, 101, 137, 143, 255, -1, -128] {
            let v = verdict_from_error_exit_nonzero(exit);
            assert_eq!(v, Qa002006Verdict::Pass, "exit={exit}");
        }
        for exit in [0_i32, 256, -129, i32::MAX, i32::MIN] {
            let v = verdict_from_error_exit_nonzero(exit);
            assert_eq!(v, Qa002006Verdict::Fail, "exit={exit}");
        }
    }

    // -------------------------------------------------------------------------
    // Section 7: Realistic — both gates' canonical scenarios.
    // -------------------------------------------------------------------------
    #[test]
    fn pass_qa_002_apr_inspect_nonexistent_file() {
        // QA-002 contract example.
        let v = verdict_from_error_exit_nonzero(1);
        assert_eq!(v, Qa002006Verdict::Pass);
    }

    #[test]
    fn pass_qa_006_apr_validate_nonexistent_file() {
        // QA-006 contract example.
        let v = verdict_from_error_exit_nonzero(1);
        assert_eq!(v, Qa002006Verdict::Pass);
    }

    #[test]
    fn fail_qa_002_silent_success_on_missing_file() {
        // The regression class: apr inspect printed "error: file
        // not found" but exit 0.
        let v = verdict_from_error_exit_nonzero(0);
        assert_eq!(v, Qa002006Verdict::Fail);
    }

    #[test]
    fn fail_qa_006_exit_code_lie() {
        // The regression class: apr validate emitted error message
        // but returned 0 — the dishonesty.
        let v = verdict_from_error_exit_nonzero(0);
        assert_eq!(v, Qa002006Verdict::Fail);
    }

    #[test]
    fn verdict_is_gate_agnostic() {
        // Both gates share the verdict for every exit_code.
        for exit in [0_i32, 1, 2, 101, 137, -1, 256, i32::MAX] {
            let v_qa002 = verdict_from_error_exit_nonzero(exit);
            let v_qa006 = verdict_from_error_exit_nonzero(exit);
            assert_eq!(
                v_qa002, v_qa006,
                "verdict must be gate-agnostic for exit={exit}"
            );
        }
    }
}
