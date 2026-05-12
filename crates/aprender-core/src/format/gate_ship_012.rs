// SHIP-TWO-001 §6 Compound Ship Gates — GATE-SHIP-012 algorithm-level
// PARTIAL discharge.
//
// Spec: docs/specifications/aprender-train/ship-two-models-spec.md §6 row
// `GATE-SHIP-012 | Coverage ≥ 95% line on new modules (per
// .pmat-gates.toml) | merge`.
// Contract: contracts/compound-ship-gates-v1.yaml v1.1.0 PROPOSED
// (FALSIFY-GATE-SHIP-012 — wired in the same PR as this file lands).
//
// GATE-SHIP-012 is the *merge-time* line-coverage threshold gate: new
// modules MUST achieve at least 95% line coverage as reported by
// `cargo llvm-cov`. The tool emits a coverage percentage in
// [0.0, 100.0]; the gate enforces that percentage >= 95.0 for merge.
//
// This file discharges the *decision rule* at `PARTIAL_ALGORITHM_LEVEL`:
// given (measured_pct, threshold_pct) coverage f32 percentages, the
// verdict is `Pass` iff both are finite, `measured_pct in [0.0, 100.0]`,
// `threshold_pct in (0.0, 100.0]`, AND `measured_pct >= threshold_pct`.
// The tool-level portion (actually running `cargo llvm-cov report
// --json` against the new-module diff and computing per-module
// coverage) is intentionally out of scope.
//
// Sibling of GATE-SHIP-011 (PMAT TDG) at a different floor and a
// different sensor. Same inclusive-floor shape; separate constants
// so a coverage-policy bump doesn't accidentally drag TDG with it
// (and vice versa).

/// Minimum line coverage percentage required on new modules. 95.0
/// corresponds to the "≥ 95% line coverage" CLAUDE.md policy (Quality
/// Standards "95% minimum test coverage (upgraded from 85%)").
///
/// Derivation: `docs/specifications/aprender-train/ship-two-models-spec.md`
/// §6 row GATE-SHIP-012 + `.pmat-gates.toml` `min_coverage_pct`
/// policy + CLAUDE.md Quality Standards. Softening this to 90 would
/// silently allow coverage regressions.
pub const AC_GATE_SHIP_012_MIN_LINE_COVERAGE_PCT: f32 = 95.0;

/// Binary verdict for FALSIFY-GATE-SHIP-012 / GATE-SHIP-012.
/// `Pass` iff both inputs are well-formed AND `measured_pct >=
/// threshold_pct`. `Fail` otherwise.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GateShip012Verdict {
    /// Well-formed inputs AND measured line coverage ≥ threshold.
    /// New modules meet the 95% coverage floor. Merge gate is green.
    Pass,
    /// Any of: non-finite measured / threshold; measured outside
    /// `[0.0, 100.0]`; threshold outside `(0.0, 100.0]`; measured <
    /// threshold. Merge blocked until coverage recovers.
    Fail,
}

/// Algorithm-level verdict rule for FALSIFY-GATE-SHIP-012 /
/// GATE-SHIP-012: inclusive-floor line-coverage threshold check.
///
/// Conservative-Fail guards:
///
///   - `!measured_pct.is_finite()` OR `!threshold_pct.is_finite()`
///     → Fail (NaN / ±∞ are never valid percentages).
///   - `measured_pct < 0.0 || measured_pct > 100.0` → Fail (coverage
///     is a unit percentage).
///   - `threshold_pct <= 0.0 || threshold_pct > 100.0` → Fail
///     (threshold must be a proper percentage in `(0.0, 100.0]`;
///     zero threshold is nonsense).
///   - `measured_pct < threshold_pct` → Fail (below floor).
///
/// # Examples
///
/// ```
/// use aprender::format::gate_ship_012::{
///     verdict_from_line_coverage_pct, GateShip012Verdict,
///     AC_GATE_SHIP_012_MIN_LINE_COVERAGE_PCT,
/// };
///
/// // Coverage = 96.5% at threshold 95% → Pass.
/// assert_eq!(
///     verdict_from_line_coverage_pct(96.5, AC_GATE_SHIP_012_MIN_LINE_COVERAGE_PCT),
///     GateShip012Verdict::Pass
/// );
///
/// // Coverage = 94.9% at threshold 95% → Fail.
/// assert_eq!(
///     verdict_from_line_coverage_pct(94.9, AC_GATE_SHIP_012_MIN_LINE_COVERAGE_PCT),
///     GateShip012Verdict::Fail
/// );
/// ```
#[must_use]
pub const fn verdict_from_line_coverage_pct(
    measured_pct: f32,
    threshold_pct: f32,
) -> GateShip012Verdict {
    if !measured_pct.is_finite() || !threshold_pct.is_finite() {
        return GateShip012Verdict::Fail;
    }
    if measured_pct < 0.0 || measured_pct > 100.0 {
        return GateShip012Verdict::Fail;
    }
    if threshold_pct <= 0.0 || threshold_pct > 100.0 {
        return GateShip012Verdict::Fail;
    }
    if measured_pct >= threshold_pct {
        GateShip012Verdict::Pass
    } else {
        GateShip012Verdict::Fail
    }
}

// ─────────────────────────────────────────────────────────────
// Unit tests — FALSIFY-GATE-SHIP-012 algorithm-level proof
// ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod gate_ship_012_tests {
    use super::*;

    /// FALSIFY-GATE-SHIP-012 algorithm-level PARTIAL discharge: prove
    /// the inclusive-floor line-coverage threshold rule. Any edit
    /// that widens the 95.0 floor, flips `>=` to `>`, or skips the
    /// range guards must break this test.
    #[test]
    fn falsify_gate_ship_012_line_coverage_threshold() {
        // Section 1: exact-boundary 95.0 → Pass. Inclusive-floor case.
        assert_eq!(
            verdict_from_line_coverage_pct(95.0, AC_GATE_SHIP_012_MIN_LINE_COVERAGE_PCT),
            GateShip012Verdict::Pass,
            "measured = threshold = 95.0 must Pass (inclusive floor)",
        );

        // Section 2: one-ULP-below the boundary → Fail. Sharpest
        // counter-example on the Fail side.
        let just_below = f32::from_bits(AC_GATE_SHIP_012_MIN_LINE_COVERAGE_PCT.to_bits() - 1);
        assert!(
            just_below < AC_GATE_SHIP_012_MIN_LINE_COVERAGE_PCT,
            "harness sanity: one-ULP neighbour is below threshold",
        );
        assert_eq!(
            verdict_from_line_coverage_pct(just_below, AC_GATE_SHIP_012_MIN_LINE_COVERAGE_PCT),
            GateShip012Verdict::Fail,
            "measured = threshold - 1 ULP must Fail (sharpest counter-example)",
        );

        // Section 3: clear Pass band — coverage at or above 95.0.
        for &m in &[96.35_f32, 99.0, 100.0, 95.5] {
            assert_eq!(
                verdict_from_line_coverage_pct(m, AC_GATE_SHIP_012_MIN_LINE_COVERAGE_PCT),
                GateShip012Verdict::Pass,
                "measured = {m}% at threshold 95.0 must Pass",
            );
        }

        // Section 4: clear Fail band — coverage below the 95.0 floor.
        for &m in &[94.9_f32, 85.0, 50.0, 0.0] {
            assert_eq!(
                verdict_from_line_coverage_pct(m, AC_GATE_SHIP_012_MIN_LINE_COVERAGE_PCT),
                GateShip012Verdict::Fail,
                "measured = {m}% at threshold 95.0 must Fail",
            );
        }

        // Section 5: non-finite measured or threshold → Fail.
        for &bad in &[f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert_eq!(
                verdict_from_line_coverage_pct(bad, AC_GATE_SHIP_012_MIN_LINE_COVERAGE_PCT),
                GateShip012Verdict::Fail,
                "non-finite measured ({bad}) must Fail conservatively",
            );
            assert_eq!(
                verdict_from_line_coverage_pct(96.0, bad),
                GateShip012Verdict::Fail,
                "non-finite threshold ({bad}) must Fail conservatively",
            );
        }

        // Section 6: out-of-range inputs → Fail.
        // Measured coverage must be in `[0.0, 100.0]`.
        assert_eq!(
            verdict_from_line_coverage_pct(-0.1, AC_GATE_SHIP_012_MIN_LINE_COVERAGE_PCT),
            GateShip012Verdict::Fail,
            "measured = -0.1% must Fail (below 0.0)",
        );
        assert_eq!(
            verdict_from_line_coverage_pct(100.01, AC_GATE_SHIP_012_MIN_LINE_COVERAGE_PCT),
            GateShip012Verdict::Fail,
            "measured = 100.01% must Fail (above 100.0)",
        );
        assert_eq!(
            verdict_from_line_coverage_pct(200.0, AC_GATE_SHIP_012_MIN_LINE_COVERAGE_PCT),
            GateShip012Verdict::Fail,
            "measured = 200.0% must Fail (nonsensical)",
        );
        // Threshold must be in `(0.0, 100.0]`.
        assert_eq!(
            verdict_from_line_coverage_pct(96.0, 0.0),
            GateShip012Verdict::Fail,
            "threshold = 0.0 must Fail (exclusive lower bound)",
        );
        assert_eq!(
            verdict_from_line_coverage_pct(96.0, -1.0),
            GateShip012Verdict::Fail,
            "negative threshold must Fail",
        );
        assert_eq!(
            verdict_from_line_coverage_pct(96.0, 100.01),
            GateShip012Verdict::Fail,
            "threshold > 100.0 must Fail",
        );
        // Threshold = 100.0 is legal (the ceiling); measured must be
        // exactly 100.0 to Pass.
        assert_eq!(
            verdict_from_line_coverage_pct(100.0, 100.0),
            GateShip012Verdict::Pass,
            "measured = 100.0% at threshold 100.0 must Pass (ceiling)",
        );
        assert_eq!(
            verdict_from_line_coverage_pct(99.999, 100.0),
            GateShip012Verdict::Fail,
            "measured just below ceiling at threshold 100.0 must Fail",
        );

        // Section 7: provenance pin — the 95.0 floor is load-bearing
        // and lockstepped with spec §6 + `.pmat-gates.toml` +
        // CLAUDE.md. The upgrade from 85% to 95% is documented in
        // CLAUDE.md Quality Standards and must not drift back.
        #[allow(clippy::float_cmp)]
        {
            assert_eq!(
                AC_GATE_SHIP_012_MIN_LINE_COVERAGE_PCT, 95.0_f32,
                "min line coverage percentage is 95.0 \
                 (spec §6 GATE-SHIP-012; .pmat-gates.toml min_coverage_pct; \
                 CLAUDE.md Quality Standards)",
            );
        }
    }
}
