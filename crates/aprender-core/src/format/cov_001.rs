// SHIP-TWO-001 — `apr-cli-coverage-v1` algorithm-level PARTIAL
// discharge for FALSIFY-COV-001.
//
// Contract: `contracts/apr-cli-coverage-v1.yaml`.
// Spec: `docs/specifications/aprender-train/ship-two-models-spec.md`
// (apr CLI coverage gate).
//
// ## What FALSIFY-COV-001 says
//
//   rule: coverage above 95%
//   prediction: "cargo llvm-cov line coverage >= 95%"
//   if_fails: "apr-cli coverage below 95%"
//
// ## What this file proves NOW (`PARTIAL_ALGORITHM_LEVEL`)
//
// Decision rule: given a measured `line_coverage_percent` (an
// f64 in [0.0, 100.0]), Pass iff:
//
//   line_coverage_percent.is_finite() AND
//   0.0 <= line_coverage_percent <= 100.0 AND
//   line_coverage_percent >= AC_COV_001_MIN_PERCENT (95.0)
//
// Inclusive `>=` matches the contract's `>= 95.0` test wording.
// Domain checks (NaN, ±∞, > 100) catch coverage-tool corruption.

/// Minimum required line coverage percentage.
///
/// Per CLAUDE.md "Quality Standards" + spec §11: 95% line
/// coverage is the published threshold. The original 85% was
/// upgraded to 95% (per CLAUDE.md note "upgraded from 85%").
/// Drift to 80% would silently degrade quality; drift to 99%
/// would over-tighten beyond contract.
pub const AC_COV_001_MIN_PERCENT: f64 = 95.0;

/// Binary verdict for `FALSIFY-COV-001`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cov001Verdict {
    /// `line_coverage_percent` is in [95.0, 100.0] (inclusive).
    Pass,
    /// One or more of:
    /// - `line_coverage_percent` is NaN or ±∞.
    /// - `line_coverage_percent < 0.0` or `> 100.0` (out-of-domain).
    /// - `line_coverage_percent < 95.0` (below contract floor).
    Fail,
}

/// Pure verdict function for `FALSIFY-COV-001`.
///
/// Inputs:
/// - `line_coverage_percent`: parsed from `cargo llvm-cov report
///   --summary-only | grep TOTAL | awk '{print $10}'`. Expected
///   to be an f64 in `[0.0, 100.0]`.
///
/// Pass iff:
/// 1. `line_coverage_percent.is_finite()`,
/// 2. `0.0 <= line_coverage_percent <= 100.0`,
/// 3. `line_coverage_percent >= 95.0`.
///
/// Otherwise `Fail`.
///
/// # Examples
///
/// 96.5% line coverage — `Pass`:
/// ```
/// use aprender::format::cov_001::{
///     verdict_from_line_coverage_percent, Cov001Verdict,
/// };
/// let v = verdict_from_line_coverage_percent(96.5);
/// assert_eq!(v, Cov001Verdict::Pass);
/// ```
///
/// 94.99% (just below floor) — `Fail`:
/// ```
/// use aprender::format::cov_001::{
///     verdict_from_line_coverage_percent, Cov001Verdict,
/// };
/// let v = verdict_from_line_coverage_percent(94.99);
/// assert_eq!(v, Cov001Verdict::Fail);
/// ```
#[must_use]
pub fn verdict_from_line_coverage_percent(line_coverage_percent: f64) -> Cov001Verdict {
    if !line_coverage_percent.is_finite() {
        return Cov001Verdict::Fail;
    }
    if !(0.0..=100.0).contains(&line_coverage_percent) {
        return Cov001Verdict::Fail;
    }
    if line_coverage_percent >= AC_COV_001_MIN_PERCENT {
        Cov001Verdict::Pass
    } else {
        Cov001Verdict::Fail
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Section 1: Provenance pin — 95.0 floor.
    // -------------------------------------------------------------------------
    #[test]
    fn provenance_min_percent_is_95() {
        assert!((AC_COV_001_MIN_PERCENT - 95.0).abs() < 1e-12);
    }

    // -------------------------------------------------------------------------
    // Section 2: Pass band — at and above floor.
    // -------------------------------------------------------------------------
    #[test]
    fn pass_at_exact_floor_95_0() {
        // Inclusive floor: 95.0 itself passes.
        let v = verdict_from_line_coverage_percent(95.0);
        assert_eq!(v, Cov001Verdict::Pass);
    }

    #[test]
    fn pass_at_canonical_96_35_per_claude_md() {
        // CLAUDE.md says "Coverage 96.35%" — that's the canonical claim.
        let v = verdict_from_line_coverage_percent(96.35);
        assert_eq!(v, Cov001Verdict::Pass);
    }

    #[test]
    fn pass_at_perfect_100() {
        let v = verdict_from_line_coverage_percent(100.0);
        assert_eq!(v, Cov001Verdict::Pass);
    }

    #[test]
    fn pass_at_99_99() {
        let v = verdict_from_line_coverage_percent(99.99);
        assert_eq!(v, Cov001Verdict::Pass);
    }

    // -------------------------------------------------------------------------
    // Section 3: Fail band — below floor.
    // -------------------------------------------------------------------------
    #[test]
    fn fail_just_below_95_at_94_99() {
        let v = verdict_from_line_coverage_percent(94.99);
        assert_eq!(
            v,
            Cov001Verdict::Fail,
            "94.99% must Fail (below 95% contract floor)"
        );
    }

    #[test]
    fn fail_at_85_old_threshold() {
        // The pre-CLAUDE.md threshold; must Fail under current contract.
        let v = verdict_from_line_coverage_percent(85.0);
        assert_eq!(
            v,
            Cov001Verdict::Fail,
            "85% (old threshold) must Fail under upgraded 95% contract"
        );
    }

    #[test]
    fn fail_at_zero() {
        let v = verdict_from_line_coverage_percent(0.0);
        assert_eq!(v, Cov001Verdict::Fail);
    }

    #[test]
    fn fail_at_50() {
        let v = verdict_from_line_coverage_percent(50.0);
        assert_eq!(v, Cov001Verdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 4: Fail band — domain violations (NaN, ±∞).
    // -------------------------------------------------------------------------
    #[test]
    fn fail_nan() {
        let v = verdict_from_line_coverage_percent(f64::NAN);
        assert_eq!(v, Cov001Verdict::Fail);
    }

    #[test]
    fn fail_positive_infinity() {
        let v = verdict_from_line_coverage_percent(f64::INFINITY);
        assert_eq!(v, Cov001Verdict::Fail);
    }

    #[test]
    fn fail_negative_infinity() {
        let v = verdict_from_line_coverage_percent(f64::NEG_INFINITY);
        assert_eq!(v, Cov001Verdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 5: Fail band — out-of-domain percentages.
    // -------------------------------------------------------------------------
    #[test]
    fn fail_negative_percent() {
        let v = verdict_from_line_coverage_percent(-1.0);
        assert_eq!(v, Cov001Verdict::Fail);
    }

    #[test]
    fn fail_above_100() {
        // Coverage cannot exceed 100% by definition.
        let v = verdict_from_line_coverage_percent(100.001);
        assert_eq!(v, Cov001Verdict::Fail);
    }

    #[test]
    fn fail_huge_value() {
        let v = verdict_from_line_coverage_percent(1e10);
        assert_eq!(v, Cov001Verdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 6: Boundary sweep around 95.0 floor.
    // -------------------------------------------------------------------------
    #[test]
    fn coverage_sweep_around_floor() {
        let probes: Vec<(f64, Cov001Verdict)> = vec![
            (0.0, Cov001Verdict::Fail),
            (50.0, Cov001Verdict::Fail),
            (90.0, Cov001Verdict::Fail),
            (94.0, Cov001Verdict::Fail),
            (94.99, Cov001Verdict::Fail),
            (95.0, Cov001Verdict::Pass), // inclusive floor
            (95.01, Cov001Verdict::Pass),
            (96.35, Cov001Verdict::Pass), // CLAUDE.md canonical
            (99.0, Cov001Verdict::Pass),
            (100.0, Cov001Verdict::Pass), // inclusive ceiling
            (100.001, Cov001Verdict::Fail), // domain
        ];
        for (pct, expected) in probes {
            let v = verdict_from_line_coverage_percent(pct);
            assert_eq!(v, expected, "pct={pct} expected {expected:?}");
        }
    }

    #[test]
    fn fail_just_below_floor_via_ulp() {
        // ULP just below 95.0 must Fail.
        let just_below = f64::from_bits(95.0_f64.to_bits() - 1);
        assert!(just_below < 95.0);
        let v = verdict_from_line_coverage_percent(just_below);
        assert_eq!(
            v,
            Cov001Verdict::Fail,
            "ULP-below floor must Fail (strict < 95)"
        );
    }

    // -------------------------------------------------------------------------
    // Section 7: Realistic — apr-cli historical coverage values.
    // -------------------------------------------------------------------------
    #[test]
    fn pass_at_canonical_apr_cli_96_35_percent() {
        // Per CLAUDE.md: 96.35% is the published apr-cli line coverage.
        let v = verdict_from_line_coverage_percent(96.35);
        assert_eq!(v, Cov001Verdict::Pass);
    }

    #[test]
    fn fail_when_dropped_below_floor() {
        // Hypothetical regression: refactor without tests; coverage
        // drops to 92%.
        let v = verdict_from_line_coverage_percent(92.0);
        assert_eq!(v, Cov001Verdict::Fail);
    }
}
