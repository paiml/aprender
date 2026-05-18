// SHIP-TWO-001 — `apr-cli-qa-v1` algorithm-level PARTIAL discharge
// for FALSIFY-QA-004.
//
// Contract: `contracts/apr-cli-qa-v1.yaml`.
// Spec: `docs/specifications/aprender-train/ship-two-models-spec.md`
// (apr CLI QA gates; cross-cutting requirement for MODEL-1 + MODEL-2
// shipping).
//
// ## What FALSIFY-QA-004 says
//
//   rule: no NaN in output
//   prediction: "no command emits NaN or Inf"
//   test: "apr run model 'test' --max-tokens 8 2>&1 |
//          grep -qE 'NaN|Inf' && exit 1 || exit 0"
//   if_fails: "numerical garbage in user output"
//
// ## What this file proves NOW (`PARTIAL_ALGORITHM_LEVEL`)
//
// Decision rule: given (`output_lines`, `nan_inf_match_count`),
// Pass iff:
//
//   output_lines > 0 AND
//   nan_inf_match_count == 0 AND
//   nan_inf_match_count <= output_lines
//
// Zero-tolerance: a single NaN or Inf in user output corrupts the
// downstream tooling that scrapes apr's stdout. Refuse empty output
// because that's a different defect class (the command produced
// nothing at all). Partition violation guard catches counter
// corruption.

/// Binary verdict for `FALSIFY-QA-004`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qa004Verdict {
    /// Output had at least one line AND zero NaN/Inf matches.
    Pass,
    /// One or more of:
    /// - `output_lines == 0` (caller error — command produced no
    ///   output to scan).
    /// - `nan_inf_match_count > 0` (numerical garbage leaked into
    ///   user output).
    /// - `nan_inf_match_count > output_lines` (counter corruption
    ///   — partition violation).
    Fail,
}

/// Pure verdict function for `FALSIFY-QA-004`.
///
/// Inputs:
/// - `output_lines`: number of lines in command stdout/stderr.
/// - `nan_inf_match_count`: number of those lines matching the
///   `NaN|Inf` regex.
///
/// Pass iff:
/// 1. `output_lines > 0`,
/// 2. `nan_inf_match_count == 0`,
/// 3. `nan_inf_match_count <= output_lines` (counter sanity).
///
/// Otherwise `Fail`.
///
/// # Examples
///
/// 100-line clean output — `Pass`:
/// ```
/// use aprender::format::qa_004::{
///     verdict_from_nan_inf_scan, Qa004Verdict,
/// };
/// let v = verdict_from_nan_inf_scan(100, 0);
/// assert_eq!(v, Qa004Verdict::Pass);
/// ```
///
/// One NaN match in 100-line output — `Fail`:
/// ```
/// use aprender::format::qa_004::{
///     verdict_from_nan_inf_scan, Qa004Verdict,
/// };
/// let v = verdict_from_nan_inf_scan(100, 1);
/// assert_eq!(v, Qa004Verdict::Fail);
/// ```
#[must_use]
pub fn verdict_from_nan_inf_scan(output_lines: u64, nan_inf_match_count: u64) -> Qa004Verdict {
    if output_lines == 0 {
        return Qa004Verdict::Fail;
    }
    if nan_inf_match_count > output_lines {
        return Qa004Verdict::Fail;
    }
    if nan_inf_match_count == 0 {
        Qa004Verdict::Pass
    } else {
        Qa004Verdict::Fail
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Section 1: Pass band — clean output at canonical sizes.
    // -------------------------------------------------------------------------
    #[test]
    fn pass_one_line_zero_matches() {
        let v = verdict_from_nan_inf_scan(1, 0);
        assert_eq!(v, Qa004Verdict::Pass);
    }

    #[test]
    fn pass_canonical_100_lines() {
        let v = verdict_from_nan_inf_scan(100, 0);
        assert_eq!(v, Qa004Verdict::Pass);
    }

    #[test]
    fn pass_realistic_apr_run_output() {
        // `apr run model 'test' --max-tokens 8` produces ~50 lines.
        let v = verdict_from_nan_inf_scan(50, 0);
        assert_eq!(v, Qa004Verdict::Pass);
    }

    #[test]
    fn pass_huge_clean_output() {
        let v = verdict_from_nan_inf_scan(1_000_000, 0);
        assert_eq!(v, Qa004Verdict::Pass);
    }

    // -------------------------------------------------------------------------
    // Section 2: Fail band — NaN/Inf appearances (zero-tolerance).
    // -------------------------------------------------------------------------
    #[test]
    fn fail_one_nan_in_100_lines() {
        let v = verdict_from_nan_inf_scan(100, 1);
        assert_eq!(
            v,
            Qa004Verdict::Fail,
            "one NaN match must Fail (no tolerance)"
        );
    }

    #[test]
    fn fail_handful_of_matches() {
        let v = verdict_from_nan_inf_scan(100, 7);
        assert_eq!(v, Qa004Verdict::Fail);
    }

    #[test]
    fn fail_one_in_million() {
        // Even at huge scale, one NaN trips the gate.
        let v = verdict_from_nan_inf_scan(1_000_000, 1);
        assert_eq!(v, Qa004Verdict::Fail);
    }

    #[test]
    fn fail_all_lines_nan() {
        // Catastrophic: every line has NaN.
        let v = verdict_from_nan_inf_scan(100, 100);
        assert_eq!(v, Qa004Verdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 3: Fail band — empty output (caller error).
    // -------------------------------------------------------------------------
    #[test]
    fn fail_zero_output_lines() {
        let v = verdict_from_nan_inf_scan(0, 0);
        assert_eq!(
            v,
            Qa004Verdict::Fail,
            "zero output must Fail (vacuous Pass refused)"
        );
    }

    #[test]
    fn fail_zero_output_with_match_count() {
        // Counter corruption: empty output but matches > 0.
        let v = verdict_from_nan_inf_scan(0, 5);
        assert_eq!(v, Qa004Verdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 4: Fail band — partition violations.
    // -------------------------------------------------------------------------
    #[test]
    fn fail_matches_exceed_lines() {
        // Counter corruption: matches > scanned lines.
        let v = verdict_from_nan_inf_scan(100, 101);
        assert_eq!(
            v,
            Qa004Verdict::Fail,
            "matches > lines must Fail (partition violation)"
        );
    }

    #[test]
    fn fail_huge_matches_with_smaller_output() {
        let v = verdict_from_nan_inf_scan(100, u64::MAX);
        assert_eq!(v, Qa004Verdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 5: Boundary sweep — match-count sweep at fixed output.
    // -------------------------------------------------------------------------
    #[test]
    fn match_count_sweep_at_fixed_output() {
        let lines = 100_u64;
        let probes: Vec<(u64, Qa004Verdict)> = vec![
            (0, Qa004Verdict::Pass),
            (1, Qa004Verdict::Fail),
            (10, Qa004Verdict::Fail),
            (50, Qa004Verdict::Fail),
            (99, Qa004Verdict::Fail),
            (100, Qa004Verdict::Fail),
            (101, Qa004Verdict::Fail), // partition violation
        ];
        for (matches, expected) in probes {
            let v = verdict_from_nan_inf_scan(lines, matches);
            assert_eq!(
                v, expected,
                "lines={lines} matches={matches} expected {expected:?}"
            );
        }
    }

    // -------------------------------------------------------------------------
    // Section 6: Domain — zero-tolerance property at canonical sizes.
    // -------------------------------------------------------------------------
    #[test]
    fn pass_iff_matches_is_exactly_zero() {
        for lines in [1_u64, 10, 100, 10_000, 1_000_000] {
            let v_pass = verdict_from_nan_inf_scan(lines, 0);
            assert_eq!(v_pass, Qa004Verdict::Pass, "lines={lines}");

            let v_fail = verdict_from_nan_inf_scan(lines, 1);
            assert_eq!(v_fail, Qa004Verdict::Fail, "lines={lines} with one match");
        }
    }

    // -------------------------------------------------------------------------
    // Section 7: Realistic — apr run / apr trace scenarios.
    // -------------------------------------------------------------------------
    #[test]
    fn pass_apr_run_clean_inference() {
        // `apr run model 'test' --max-tokens 8` typically emits
        // ~50 lines of output (token-by-token + summary). Clean
        // inference has zero NaN.
        let v = verdict_from_nan_inf_scan(50, 0);
        assert_eq!(v, Qa004Verdict::Pass);
    }

    #[test]
    fn fail_apr_trace_nan_in_layer_stats() {
        // Worst case: `apr trace --json` emits NaN in one layer's
        // statistics due to a numerical regression.
        let v = verdict_from_nan_inf_scan(280, 1); // 28 layers * 10 fields
        assert_eq!(v, Qa004Verdict::Fail, "NaN in apr trace stats must Fail");
    }

    #[test]
    fn fail_inf_in_attention_softmax() {
        // Inf in attention scores due to a softmax regression.
        let v = verdict_from_nan_inf_scan(100, 3);
        assert_eq!(v, Qa004Verdict::Fail);
    }
}
