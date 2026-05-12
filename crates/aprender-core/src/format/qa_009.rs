// SHIP-TWO-001 — `apr-cli-qa-v1` algorithm-level PARTIAL discharge
// for FALSIFY-QA-009.
//
// Contract: `contracts/apr-cli-qa-v1.yaml`.
// Spec: `docs/specifications/aprender-train/ship-two-models-spec.md`
// (apr CLI QA gates).
//
// ## What FALSIFY-QA-009 says
//
//   rule: 3-format coverage
//   prediction: "inspect works on GGUF, APR, and SafeTensors"
//   test: "apr inspect model.gguf && apr inspect model.apr &&
//          apr inspect model.safetensors"
//   if_fails: "format not supported"
//
// ## What this file proves NOW (`PARTIAL_ALGORITHM_LEVEL`)
//
// Decision rule: given the three exit codes from
// `apr inspect <model>` invoked with each format, Pass iff:
//
//   gguf_exit == 0 AND apr_exit == 0 AND safetensors_exit == 0
//
// All three must succeed independently. Conjunctive composition
// catches any one format silently regressing. Per CLAUDE.md
// "Debugging: Use apr Tools First": "All tools support GGUF, APR,
// and SafeTensors formats. If a tool says 'format not supported',
// that's a BUG."

/// Number of model formats `apr inspect` MUST support.
///
/// Per CLAUDE.md / spec §26.8: GGUF + APR + SafeTensors = 3
/// formats. Pinning the count catches a regression where a 4th
/// format is added without bumping the contract, OR where one of
/// the three is silently dropped.
pub const AC_QA_009_REQUIRED_FORMAT_COUNT: u32 = 3;

/// Binary verdict for `FALSIFY-QA-009`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qa009Verdict {
    /// All three formats accepted: `apr inspect` exits 0 on
    /// GGUF, APR, AND SafeTensors models.
    Pass,
    /// One or more of the three formats is unsupported (any
    /// non-zero exit).
    Fail,
}

/// Pure verdict function for `FALSIFY-QA-009`.
///
/// Inputs:
/// - `gguf_exit`: exit code from `apr inspect model.gguf`.
/// - `apr_exit`: exit code from `apr inspect model.apr`.
/// - `safetensors_exit`: exit code from `apr inspect model.safetensors`.
///
/// Pass iff all three exit codes equal 0.
///
/// Otherwise `Fail`.
///
/// # Examples
///
/// All three formats supported — `Pass`:
/// ```
/// use aprender::format::qa_009::{
///     verdict_from_three_format_coverage, Qa009Verdict,
/// };
/// let v = verdict_from_three_format_coverage(0, 0, 0);
/// assert_eq!(v, Qa009Verdict::Pass);
/// ```
///
/// SafeTensors unsupported — `Fail`:
/// ```
/// use aprender::format::qa_009::{
///     verdict_from_three_format_coverage, Qa009Verdict,
/// };
/// let v = verdict_from_three_format_coverage(0, 0, 1);
/// assert_eq!(v, Qa009Verdict::Fail);
/// ```
#[must_use]
pub fn verdict_from_three_format_coverage(
    gguf_exit: i32,
    apr_exit: i32,
    safetensors_exit: i32,
) -> Qa009Verdict {
    if gguf_exit == 0 && apr_exit == 0 && safetensors_exit == 0 {
        Qa009Verdict::Pass
    } else {
        Qa009Verdict::Fail
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Section 1: Provenance pin — 3 formats canonical.
    // -------------------------------------------------------------------------
    #[test]
    fn provenance_required_format_count_is_three() {
        assert_eq!(AC_QA_009_REQUIRED_FORMAT_COUNT, 3);
    }

    // -------------------------------------------------------------------------
    // Section 2: Pass band — only (0, 0, 0) passes.
    // -------------------------------------------------------------------------
    #[test]
    fn pass_all_three_succeed() {
        let v = verdict_from_three_format_coverage(0, 0, 0);
        assert_eq!(v, Qa009Verdict::Pass);
    }

    // -------------------------------------------------------------------------
    // Section 3: Fail band — exactly one format fails.
    // -------------------------------------------------------------------------
    #[test]
    fn fail_gguf_unsupported() {
        let v = verdict_from_three_format_coverage(1, 0, 0);
        assert_eq!(
            v,
            Qa009Verdict::Fail,
            "GGUF unsupported must Fail"
        );
    }

    #[test]
    fn fail_apr_unsupported() {
        let v = verdict_from_three_format_coverage(0, 1, 0);
        assert_eq!(v, Qa009Verdict::Fail);
    }

    #[test]
    fn fail_safetensors_unsupported() {
        let v = verdict_from_three_format_coverage(0, 0, 1);
        assert_eq!(v, Qa009Verdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 4: Fail band — multiple formats fail.
    // -------------------------------------------------------------------------
    #[test]
    fn fail_gguf_and_apr() {
        let v = verdict_from_three_format_coverage(1, 1, 0);
        assert_eq!(v, Qa009Verdict::Fail);
    }

    #[test]
    fn fail_gguf_and_safetensors() {
        let v = verdict_from_three_format_coverage(1, 0, 1);
        assert_eq!(v, Qa009Verdict::Fail);
    }

    #[test]
    fn fail_apr_and_safetensors() {
        let v = verdict_from_three_format_coverage(0, 1, 1);
        assert_eq!(v, Qa009Verdict::Fail);
    }

    #[test]
    fn fail_all_three_fail() {
        let v = verdict_from_three_format_coverage(1, 1, 1);
        assert_eq!(v, Qa009Verdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 5: Fail band — non-1 exit codes.
    // -------------------------------------------------------------------------
    #[test]
    fn fail_panic_on_one_format() {
        let v = verdict_from_three_format_coverage(0, 0, 101);
        assert_eq!(v, Qa009Verdict::Fail);
    }

    #[test]
    fn fail_negative_exit() {
        let v = verdict_from_three_format_coverage(-1, 0, 0);
        assert_eq!(v, Qa009Verdict::Fail);
    }

    #[test]
    fn fail_i32_max_exit() {
        let v = verdict_from_three_format_coverage(0, 0, i32::MAX);
        assert_eq!(v, Qa009Verdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 6: 8-cell composite matrix — all 2³ combinations.
    // -------------------------------------------------------------------------
    #[test]
    fn matrix_pass_iff_all_three_zero() {
        // (gguf, apr, safetensors) where 0=success, 1=fail.
        let cases: Vec<(i32, i32, i32, Qa009Verdict)> = vec![
            (0, 0, 0, Qa009Verdict::Pass), // canonical pass
            (1, 0, 0, Qa009Verdict::Fail),
            (0, 1, 0, Qa009Verdict::Fail),
            (0, 0, 1, Qa009Verdict::Fail),
            (1, 1, 0, Qa009Verdict::Fail),
            (1, 0, 1, Qa009Verdict::Fail),
            (0, 1, 1, Qa009Verdict::Fail),
            (1, 1, 1, Qa009Verdict::Fail),
        ];
        for (g, a, s, expected) in cases {
            let v = verdict_from_three_format_coverage(g, a, s);
            assert_eq!(
                v, expected,
                "gguf={g} apr={a} safetensors={s} expected {expected:?}"
            );
        }
    }

    // -------------------------------------------------------------------------
    // Section 7: Symmetry — verdict is invariant under permuting the 3 inputs.
    // -------------------------------------------------------------------------
    #[test]
    fn verdict_invariant_under_permutation_of_zero_zero_one() {
        // (0, 0, 1) → Fail, regardless of which slot the 1 is in.
        let v1 = verdict_from_three_format_coverage(0, 0, 1);
        let v2 = verdict_from_three_format_coverage(0, 1, 0);
        let v3 = verdict_from_three_format_coverage(1, 0, 0);
        assert_eq!(v1, v2);
        assert_eq!(v2, v3);
        assert_eq!(v1, Qa009Verdict::Fail);
    }

    #[test]
    fn pass_only_at_origin() {
        // The Pass cell is exactly (0, 0, 0); no other cell passes.
        for (g, a, s) in [
            (0_i32, 0_i32, 0_i32),
        ] {
            let v = verdict_from_three_format_coverage(g, a, s);
            assert_eq!(v, Qa009Verdict::Pass);
        }
        for &g in &[0_i32, 1, -1, 101] {
            for &a in &[0_i32, 1, -1, 101] {
                for &s in &[0_i32, 1, -1, 101] {
                    let v = verdict_from_three_format_coverage(g, a, s);
                    let expected = if g == 0 && a == 0 && s == 0 {
                        Qa009Verdict::Pass
                    } else {
                        Qa009Verdict::Fail
                    };
                    assert_eq!(
                        v, expected,
                        "gguf={g} apr={a} safetensors={s}"
                    );
                }
            }
        }
    }
}
