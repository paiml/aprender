// SHIP-TWO-001 §6 Compound Ship Gates — GATE-SHIP-011 algorithm-level
// PARTIAL discharge.
//
// Spec: docs/specifications/aprender-train/ship-two-models-spec.md §6 row
// `GATE-SHIP-011 | PMAT quality score ≥ A- (project), TDG ≥ 90 |
// merge`.
// Contract: contracts/compound-ship-gates-v1.yaml v1.1.0 PROPOSED
// (FALSIFY-GATE-SHIP-011 — wired in the same PR as this file lands).
//
// GATE-SHIP-011 is the *merge-time* PMAT-TDG threshold gate: the
// repository's TDG (Technical Debt Grade) score MUST be at or above
// 90.0 (equivalent to the A- project grade). PMAT emits a TDG score
// in [0.0, 100.0]; the gate says that score >= 90.0 for merge.
//
// This file discharges the *decision rule* at `PARTIAL_ALGORITHM_LEVEL`:
// given (measured, threshold) TDG f32 scores, the verdict is `Pass`
// iff both are finite, `measured >= 0.0`, `threshold > 0.0 &&
// threshold <= 100.0` (TDG is bounded to the unit-percentage range
// and a zero threshold is nonsense), AND `measured >= threshold`. The
// tool-level portion (actually running `pmat tdg .` on the workspace
// and parsing the score) is intentionally out of scope.
//
// Inclusive-floor boundary: the spec says "≥ 90", so 90.0 exactly
// Passes. A mutation to strict `>` would flip the boundary case to
// Fail and break the test.

/// Minimum PMAT TDG score required for the project-level merge gate.
/// 90.0 corresponds to the A- project grade per PMAT's quality-grading
/// scale.
///
/// Derivation: `docs/specifications/aprender-train/ship-two-models-spec.md`
/// §6 row GATE-SHIP-011 + `.pmat-gates.toml` `min_tdg_score` policy.
/// Softening this to 85 (B+) would silently allow quality regressions
/// to ship.
pub const AC_GATE_SHIP_011_MIN_PMAT_TDG_SCORE: f32 = 90.0;

/// Binary verdict for FALSIFY-GATE-SHIP-011 / GATE-SHIP-011.
/// `Pass` iff both inputs are well-formed AND `measured >=
/// threshold`. `Fail` otherwise.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GateShip011Verdict {
    /// Well-formed inputs AND measured TDG ≥ threshold. PMAT quality
    /// grade is A- or above. Merge gate is green.
    Pass,
    /// Any of: non-finite measured / threshold; measured < 0.0;
    /// threshold outside `(0.0, 100.0]`; measured < threshold. Merge
    /// blocked until quality recovers.
    Fail,
}

/// Algorithm-level verdict rule for FALSIFY-GATE-SHIP-011 /
/// GATE-SHIP-011: inclusive-floor TDG-score threshold check.
///
/// Conservative-Fail guards:
///
///   - `!measured.is_finite()` OR `!threshold.is_finite()` → Fail
///     (NaN / ±∞ are never valid grades).
///   - `measured < 0.0` → Fail (TDG scores are non-negative).
///   - `threshold <= 0.0 || threshold > 100.0` → Fail (threshold
///     must be a proper percentage in `(0.0, 100.0]`; zero threshold
///     is nonsense).
///   - `measured < threshold` → Fail (below floor).
///
/// # Examples
///
/// ```
/// use aprender::format::gate_ship_011::{
///     verdict_from_tdg_score, GateShip011Verdict,
///     AC_GATE_SHIP_011_MIN_PMAT_TDG_SCORE,
/// };
///
/// // TDG = 92.0 at threshold 90.0 → Pass.
/// assert_eq!(
///     verdict_from_tdg_score(92.0, AC_GATE_SHIP_011_MIN_PMAT_TDG_SCORE),
///     GateShip011Verdict::Pass
/// );
///
/// // TDG = 89.9 at threshold 90.0 → Fail (below floor).
/// assert_eq!(
///     verdict_from_tdg_score(89.9, AC_GATE_SHIP_011_MIN_PMAT_TDG_SCORE),
///     GateShip011Verdict::Fail
/// );
/// ```
#[must_use]
pub const fn verdict_from_tdg_score(measured: f32, threshold: f32) -> GateShip011Verdict {
    if !measured.is_finite() || !threshold.is_finite() {
        return GateShip011Verdict::Fail;
    }
    if measured < 0.0 {
        return GateShip011Verdict::Fail;
    }
    if threshold <= 0.0 || threshold > 100.0 {
        return GateShip011Verdict::Fail;
    }
    if measured >= threshold {
        GateShip011Verdict::Pass
    } else {
        GateShip011Verdict::Fail
    }
}

// ─────────────────────────────────────────────────────────────
// Unit tests — FALSIFY-GATE-SHIP-011 algorithm-level proof
// ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod gate_ship_011_tests {
    use super::*;

    /// FALSIFY-GATE-SHIP-011 algorithm-level PARTIAL discharge: prove
    /// the inclusive-floor TDG-score threshold rule. Any edit that
    /// widens the 90.0 floor, flips `>=` to `>`, or skips the range
    /// guards must break this test.
    #[test]
    fn falsify_gate_ship_011_pmat_tdg_threshold() {
        // Section 1: exact-boundary 90.0 → Pass. Inclusive-floor case;
        // a mutation to strict `>` would flip this to Fail.
        assert_eq!(
            verdict_from_tdg_score(90.0, AC_GATE_SHIP_011_MIN_PMAT_TDG_SCORE),
            GateShip011Verdict::Pass,
            "measured = threshold = 90.0 must Pass (inclusive floor)",
        );

        // Section 2: one-ULP-below the boundary → Fail. Sharpest
        // counter-example on the Fail side.
        let just_below = f32::from_bits(AC_GATE_SHIP_011_MIN_PMAT_TDG_SCORE.to_bits() - 1);
        assert!(
            just_below < AC_GATE_SHIP_011_MIN_PMAT_TDG_SCORE,
            "harness sanity: one-ULP neighbour is below threshold",
        );
        assert_eq!(
            verdict_from_tdg_score(just_below, AC_GATE_SHIP_011_MIN_PMAT_TDG_SCORE),
            GateShip011Verdict::Fail,
            "measured = threshold - 1 ULP must Fail (sharpest counter-example)",
        );

        // Section 3: clear Pass band — high-quality scores at or
        // above threshold.
        for &m in &[95.0_f32, 100.0, 92.5] {
            assert_eq!(
                verdict_from_tdg_score(m, AC_GATE_SHIP_011_MIN_PMAT_TDG_SCORE),
                GateShip011Verdict::Pass,
                "measured = {m} at threshold 90.0 must Pass",
            );
        }

        // Section 4: clear Fail band — scores below the 90.0 floor.
        for &m in &[89.9_f32, 80.0, 0.0, 50.0] {
            assert_eq!(
                verdict_from_tdg_score(m, AC_GATE_SHIP_011_MIN_PMAT_TDG_SCORE),
                GateShip011Verdict::Fail,
                "measured = {m} at threshold 90.0 must Fail",
            );
        }

        // Section 5: non-finite measured or threshold → Fail.
        for &bad in &[f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert_eq!(
                verdict_from_tdg_score(bad, AC_GATE_SHIP_011_MIN_PMAT_TDG_SCORE),
                GateShip011Verdict::Fail,
                "non-finite measured ({bad}) must Fail conservatively",
            );
            assert_eq!(
                verdict_from_tdg_score(95.0, bad),
                GateShip011Verdict::Fail,
                "non-finite threshold ({bad}) must Fail conservatively",
            );
        }
        // Negative measured (can't have negative TDG).
        assert_eq!(
            verdict_from_tdg_score(-0.1, AC_GATE_SHIP_011_MIN_PMAT_TDG_SCORE),
            GateShip011Verdict::Fail,
            "negative measured must Fail (TDG is non-negative)",
        );

        // Section 6: out-of-range threshold → Fail. Threshold must be
        // in `(0.0, 100.0]`.
        assert_eq!(
            verdict_from_tdg_score(95.0, 0.0),
            GateShip011Verdict::Fail,
            "threshold = 0.0 must Fail (exclusive lower bound)",
        );
        assert_eq!(
            verdict_from_tdg_score(95.0, -1.0),
            GateShip011Verdict::Fail,
            "negative threshold must Fail",
        );
        assert_eq!(
            verdict_from_tdg_score(95.0, 100.01),
            GateShip011Verdict::Fail,
            "threshold > 100.0 must Fail (above TDG's unit range)",
        );
        assert_eq!(
            verdict_from_tdg_score(95.0, 1000.0),
            GateShip011Verdict::Fail,
            "threshold = 1000.0 must Fail (nonsensical)",
        );
        // Threshold = 100.0 is legal (the ceiling); measured must be
        // exactly 100.0 to Pass.
        assert_eq!(
            verdict_from_tdg_score(100.0, 100.0),
            GateShip011Verdict::Pass,
            "measured = 100.0 at threshold 100.0 must Pass (ceiling)",
        );
        assert_eq!(
            verdict_from_tdg_score(99.999, 100.0),
            GateShip011Verdict::Fail,
            "measured just below ceiling at threshold 100.0 must Fail",
        );

        // Section 7: provenance pin — the 90.0 floor is load-bearing
        // and lockstepped with spec §6 + `.pmat-gates.toml`. Softening
        // this requires a full quality-policy review.
        #[allow(clippy::float_cmp)]
        {
            assert_eq!(
                AC_GATE_SHIP_011_MIN_PMAT_TDG_SCORE, 90.0_f32,
                "min PMAT TDG score is 90.0 (A-) \
                 (spec §6 GATE-SHIP-011; .pmat-gates.toml min_tdg_score)",
            );
        }
    }
}
