// SHIP-TWO-001 AC-SHIP1-023 / FALSIFY-SHIP-023 algorithm-level PARTIAL discharge.
//
// Spec: docs/specifications/aprender-train/ship-two-models-spec.md §7.1 row
// `FALSIFY-SHIP-023 | Re-run AC-005 on second day; score drift | drift > 1.2 pp`.
// Contract: contracts/qwen2-e2e-verification-v1.yaml (FALSIFY-QW2E-SHIP-023 —
// wired in the same PR as this file lands).
//
// SHIP-TWO-001 §7.1 states that the MODEL-1 teacher must produce
// HumanEval pass@1 scores that are stable across two consecutive
// evaluation days. The spec's falsification rule is phrased "drift
// > 1.2 pp" — a pair-of-runs threshold over `|day1 − day2|`.
//
// This file discharges the *decision rule* at `PARTIAL_ALGORITHM_LEVEL`
// by binding one pure verdict fn:
//
//   `verdict_from_score_drift(day1_pct, day2_pct, tolerance_pp)
//    -> Ship023Verdict` — pair-of-runs drift rule. `Pass` iff all three
//   inputs are `is_finite()`, both `day1_pct` and `day2_pct` live in
//   the valid pass@1 range `[0.0, 100.0]`, `tolerance_pp >= 0.0`, AND
//   `(day1_pct - day2_pct).abs() <= tolerance_pp`. Conservative Fail
//   on any non-finite, out-of-range, negative-tolerance, or over-drift
//   input — a harness bug must never silently ship a drifting teacher.
//
// SHIP-023 pairs numerically with SHIP-005 (`AC_SHIP1_005_NOISE_ALLOWANCE_PP
// = 1.2`) but carries different semantics: SHIP-005 is the noise window
// around the *nominal* 86.00% HumanEval pass@1 floor (tolerated drop),
// while SHIP-023 is the drift tolerance *between two measured runs*
// (tolerated day-over-day fluctuation). Both use the same 1.2 pp budget
// by design — SHIP-023's drift must fit inside SHIP-005's noise allowance
// for the ship gate to compose.
//
// The compute-heavy portion of the AC (actually running `apr eval
// --benchmark humaneval` on two consecutive days and computing the
// absolute drift between the two pass@1 medians) is intentionally out
// of scope here. The threshold rule is what the compute harness must
// emit a Pass on, and changing the bound constant
// (`AC_SHIP1_023_MAX_HUMANEVAL_DRIFT_PP`), the inequality direction,
// the `.abs()` symmetric combinator, or the range guard breaks this
// test before a single second-day eval runs.
//
// Mirrors the single-number-threshold shape of SHIP-007 / SHIP-020
// (decode-tps) and SHIP-005's three-constant approach, but uniquely
// uses a pair-of-runs diff with `.abs()` for order invariance. The
// conservative-range guard `[0.0, 100.0]` on both day values mirrors
// SHIP-005's input validation. SHIP-023 is `ship_blocking: false`
// (spec §7.1 stability test, not in §4.2 AC table).

/// Maximum tolerated cross-run drift in percentage points (pp) for the
/// MODEL-1 HumanEval pass@1 stability gate. Any two measured pass@1
/// scores whose absolute difference exceeds 1.2 pp fails
/// FALSIFY-SHIP-023.
///
/// Lockstep with `docs/specifications/aprender-train/ship-two-models-spec.md`
/// §7.1 row FALSIFY-SHIP-023: "Re-run AC-005 on second day; score
/// drift > 1.2 pp".
///
/// Pairs numerically with `AC_SHIP1_005_NOISE_ALLOWANCE_PP = 1.2` but
/// semantically distinct: SHIP-005's noise allowance is the tolerated
/// drop from the 86.00% nominal floor (one-sided, signed), while
/// SHIP-023's drift tolerance is the symmetric `.abs()` difference
/// between two measured runs (two-sided, unsigned). Sharing the same
/// 1.2 pp budget is intentional: a day-over-day drift that exceeds
/// the noise-allowance window would be incompatible with SHIP-005's
/// ship gate.
///
/// f32-exact: `1.2` in IEEE-754 binary32 is `0x3F99999A` =
/// 1.20000004768... — the ULP-above neighbour (`0x3F99999B`) is the
/// sharp counter-example the mutation survey uses.
pub const AC_SHIP1_023_MAX_HUMANEVAL_DRIFT_PP: f32 = 1.2;

/// Binary verdict for FALSIFY-SHIP-023 / AC-SHIP1-023.
/// `Pass` iff inputs are well-formed AND the absolute drift between
/// day1 and day2 pass@1 scores is at or below the tolerance. `Fail`
/// otherwise.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ship023Verdict {
    /// Well-formed inputs AND `(day1 − day2).abs() ≤ tolerance_pp`.
    /// The two-day HumanEval re-run is stable within the 1.2 pp drift
    /// budget.
    Pass,
    /// Any of: non-finite `day1_pct` / `day2_pct` / `tolerance_pp`;
    /// `day1_pct` or `day2_pct` outside `[0.0, 100.0]`;
    /// `tolerance_pp < 0.0`; `(day1 − day2).abs() > tolerance_pp`.
    Fail,
}

/// Pure decision rule for FALSIFY-SHIP-023 / AC-SHIP1-023:
/// pair-of-runs HumanEval pass@1 drift check.
///
/// Conservative-Fail guards:
///
///   - `!day1_pct.is_finite()` OR `!day2_pct.is_finite()` OR
///     `!tolerance_pp.is_finite()` → Fail (NaN / ±∞ are never valid
///     pass@1 scores or drift budgets).
///   - `day1_pct` or `day2_pct` outside `[0.0, 100.0]` → Fail
///     (pass@1 is a percentage; values outside that range are harness
///     bugs).
///   - `tolerance_pp < 0.0` → Fail (contract drift — a negative drift
///     allowance is nonsense; 0.0 is legal and means "byte-identical").
///   - `(day1_pct − day2_pct).abs() > tolerance_pp` → Fail (drift
///     exceeds tolerance).
///
/// The function is order-invariant because of `.abs()`: swapping day1
/// and day2 yields the same verdict. The mutation survey's Section 4
/// proves this.
///
/// # Examples
///
/// ```
/// use aprender::format::ship_023::{
///     verdict_from_score_drift, Ship023Verdict, AC_SHIP1_023_MAX_HUMANEVAL_DRIFT_PP,
/// };
///
/// // Stable re-run: 86.0% day 1, 86.5% day 2 → drift 0.5 pp ≤ 1.2 pp.
/// assert_eq!(
///     verdict_from_score_drift(86.0, 86.5, AC_SHIP1_023_MAX_HUMANEVAL_DRIFT_PP),
///     Ship023Verdict::Pass
/// );
/// // Drifting re-run: 86.0% day 1, 84.5% day 2 → drift 1.5 pp > 1.2 pp.
/// assert_eq!(
///     verdict_from_score_drift(86.0, 84.5, AC_SHIP1_023_MAX_HUMANEVAL_DRIFT_PP),
///     Ship023Verdict::Fail
/// );
/// ```
#[must_use]
pub fn verdict_from_score_drift(day1_pct: f32, day2_pct: f32, tolerance_pp: f32) -> Ship023Verdict {
    if !day1_pct.is_finite() || !day2_pct.is_finite() || !tolerance_pp.is_finite() {
        return Ship023Verdict::Fail;
    }
    if !(0.0_f32..=100.0_f32).contains(&day1_pct) {
        return Ship023Verdict::Fail;
    }
    if !(0.0_f32..=100.0_f32).contains(&day2_pct) {
        return Ship023Verdict::Fail;
    }
    if tolerance_pp < 0.0 {
        return Ship023Verdict::Fail;
    }
    let drift = (day1_pct - day2_pct).abs();
    if drift <= tolerance_pp {
        Ship023Verdict::Pass
    } else {
        Ship023Verdict::Fail
    }
}

// ─────────────────────────────────────────────────────────────
// Unit tests — FALSIFY-SHIP-023 algorithm-level proof
// ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod ship_023_tests {
    use super::*;

    /// FALSIFY-SHIP-023 algorithm-level PARTIAL discharge: prove the
    /// pair-of-runs drift rule for AC-SHIP1-023. Any edit that changes
    /// the 1.2 pp tolerance, the inequality direction, the `.abs()`
    /// combinator, the range guard `[0.0, 100.0]`, the negative-tolerance
    /// rejection, or the non-finite handling must break this test before
    /// a two-day HumanEval re-run is launched.
    #[test]
    fn falsify_ship_023_score_drift_threshold_logic() {
        // Section 1: exact boundary drift = 1.2 pp → Pass; drift just
        // above the boundary → Fail. `AC_SHIP1_023_MAX_HUMANEVAL_DRIFT_PP`
        // is the stored f32 value `0x3F99999A` so the boundary test
        // `(86.0 - 84.8).abs() == 1.2000002` compared against 1.2 must
        // be handled with IEEE-754 awareness. We use 86.0 vs 87.2 which
        // gives drift exactly at 1.2000003 (not quite boundary-equal)
        // — prefer a precisely-boundary pair: 1.2000005 and 0.0.
        assert_eq!(
            verdict_from_score_drift(
                AC_SHIP1_023_MAX_HUMANEVAL_DRIFT_PP,
                0.0,
                AC_SHIP1_023_MAX_HUMANEVAL_DRIFT_PP
            ),
            Ship023Verdict::Pass,
            "drift exactly at tolerance boundary must Pass (inclusive)",
        );
        // drift = 1.2 + 1e-4 = 1.2001, above boundary → Fail.
        let just_above: f32 = AC_SHIP1_023_MAX_HUMANEVAL_DRIFT_PP + 1e-4;
        assert!(
            just_above > AC_SHIP1_023_MAX_HUMANEVAL_DRIFT_PP,
            "harness sanity: just_above must really exceed tolerance",
        );
        assert_eq!(
            verdict_from_score_drift(just_above, 0.0, AC_SHIP1_023_MAX_HUMANEVAL_DRIFT_PP),
            Ship023Verdict::Fail,
            "drift = 1.2 + 1e-4 must Fail (sharpest counter-example)",
        );

        // Section 2: clear Pass band — drifts well below the 1.2 pp
        // budget.
        for &drift in &[0.0_f32, 0.5, 1.0, 1.199] {
            let day1 = 86.0;
            let day2 = 86.0 - drift;
            assert_eq!(
                verdict_from_score_drift(day1, day2, AC_SHIP1_023_MAX_HUMANEVAL_DRIFT_PP),
                Ship023Verdict::Pass,
                "drift {drift} pp must Pass (below 1.2 tolerance)",
            );
        }

        // Section 3: clear Fail band — drifts well above the 1.2 pp
        // budget, including HumanEval-extreme pairings.
        for &drift in &[1.3_f32, 2.0, 10.0, 86.0] {
            let day1 = 86.0;
            let day2 = 86.0 - drift;
            assert_eq!(
                verdict_from_score_drift(day1, day2, AC_SHIP1_023_MAX_HUMANEVAL_DRIFT_PP),
                Ship023Verdict::Fail,
                "drift {drift} pp must Fail (above 1.2 tolerance)",
            );
        }

        // Section 4: symmetric / order-invariant — swapping day1 and
        // day2 must yield identical verdict because `.abs()`. The spec
        // phrasing "drift > 1.2 pp" has no directional bias.
        let (pass_a, pass_b) = (86.0_f32, 84.8_f32);
        assert_eq!(
            verdict_from_score_drift(pass_a, pass_b, AC_SHIP1_023_MAX_HUMANEVAL_DRIFT_PP),
            verdict_from_score_drift(pass_b, pass_a, AC_SHIP1_023_MAX_HUMANEVAL_DRIFT_PP),
            "verdict must be order-invariant (Pass direction)",
        );
        // Both directions must Pass for this pair (drift is exactly
        // 1.2 pp, but f32 subtraction may yield 1.2000002, which is
        // above the stored threshold of 1.2000001 — so we use 86.0
        // and 85.0 for a safely-in-band pair).
        let (pass_c, pass_d) = (86.0_f32, 85.0_f32);
        assert_eq!(
            verdict_from_score_drift(pass_c, pass_d, AC_SHIP1_023_MAX_HUMANEVAL_DRIFT_PP),
            Ship023Verdict::Pass,
            "1.0 pp drift (within 1.2) must Pass in forward direction",
        );
        assert_eq!(
            verdict_from_score_drift(pass_d, pass_c, AC_SHIP1_023_MAX_HUMANEVAL_DRIFT_PP),
            Ship023Verdict::Pass,
            "1.0 pp drift (within 1.2) must Pass in reverse direction (symmetry)",
        );
        // Fail direction must also be symmetric.
        let (fail_a, fail_b) = (86.0_f32, 82.0_f32);
        assert_eq!(
            verdict_from_score_drift(fail_a, fail_b, AC_SHIP1_023_MAX_HUMANEVAL_DRIFT_PP),
            verdict_from_score_drift(fail_b, fail_a, AC_SHIP1_023_MAX_HUMANEVAL_DRIFT_PP),
            "verdict must be order-invariant (Fail direction)",
        );
        assert_eq!(
            verdict_from_score_drift(fail_a, fail_b, AC_SHIP1_023_MAX_HUMANEVAL_DRIFT_PP),
            Ship023Verdict::Fail,
            "4.0 pp drift must Fail regardless of order",
        );

        // Section 5: non-finite inputs on any of the three f32 args
        // must conservatively Fail. A harness that divides by zero, a
        // telemetry bug that promotes NaN, or a missing-field default
        // must NOT silently ship a Pass.
        for &bad in &[f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert_eq!(
                verdict_from_score_drift(bad, 86.0, AC_SHIP1_023_MAX_HUMANEVAL_DRIFT_PP),
                Ship023Verdict::Fail,
                "non-finite day1 ({bad}) must Fail conservatively",
            );
            assert_eq!(
                verdict_from_score_drift(86.0, bad, AC_SHIP1_023_MAX_HUMANEVAL_DRIFT_PP),
                Ship023Verdict::Fail,
                "non-finite day2 ({bad}) must Fail conservatively",
            );
            assert_eq!(
                verdict_from_score_drift(86.0, 86.0, bad),
                Ship023Verdict::Fail,
                "non-finite tolerance ({bad}) must Fail conservatively",
            );
        }

        // Section 6: out-of-range day values Fail. pass@1 is a
        // percentage in `[0.0, 100.0]`; anything else is a harness
        // bug — negative pass@1 is impossible (SHIP-018 proves floor),
        // >100% is impossible by construction (num-correct ≤ total).
        for &oor in &[-0.1_f32, -1.0, -86.0, 100.1, 101.0, 1_000.0] {
            assert_eq!(
                verdict_from_score_drift(oor, 86.0, AC_SHIP1_023_MAX_HUMANEVAL_DRIFT_PP),
                Ship023Verdict::Fail,
                "out-of-range day1 ({oor}) must Fail",
            );
            assert_eq!(
                verdict_from_score_drift(86.0, oor, AC_SHIP1_023_MAX_HUMANEVAL_DRIFT_PP),
                Ship023Verdict::Fail,
                "out-of-range day2 ({oor}) must Fail",
            );
        }
        // Negative tolerance → Fail (contract drift — the drift budget
        // must be non-negative; 0.0 is legal and means byte-identical).
        for &neg_tol in &[-1e-6_f32, -0.001, -1.0, -100.0] {
            assert_eq!(
                verdict_from_score_drift(86.0, 86.0, neg_tol),
                Ship023Verdict::Fail,
                "negative tolerance ({neg_tol}) must Fail (contract drift)",
            );
        }
        // Zero tolerance → Pass iff day1 == day2 exactly.
        assert_eq!(
            verdict_from_score_drift(86.0, 86.0, 0.0),
            Ship023Verdict::Pass,
            "zero tolerance with identical scores must Pass",
        );
        assert_eq!(
            verdict_from_score_drift(86.0, 86.1, 0.0),
            Ship023Verdict::Fail,
            "zero tolerance with any drift must Fail",
        );
        // Boundary pass@1 values {0.0, 100.0} are legal.
        assert_eq!(
            verdict_from_score_drift(0.0, 0.0, AC_SHIP1_023_MAX_HUMANEVAL_DRIFT_PP),
            Ship023Verdict::Pass,
            "both at 0.0% must Pass (degenerate but legal)",
        );
        assert_eq!(
            verdict_from_score_drift(100.0, 100.0, AC_SHIP1_023_MAX_HUMANEVAL_DRIFT_PP),
            Ship023Verdict::Pass,
            "both at 100.0% must Pass (degenerate but legal)",
        );

        // Section 7: provenance pin — the 1.2 pp constant is load-
        // bearing and lockstepped with the spec. If §7.1 FALSIFY-SHIP-023
        // ever widens the drift tolerance (say to 2.0 pp), this const
        // must move in lockstep and the rest of the survey follows.
        #[allow(clippy::float_cmp)]
        {
            assert_eq!(
                AC_SHIP1_023_MAX_HUMANEVAL_DRIFT_PP, 1.2_f32,
                "max HumanEval drift is 1.2 pp (spec §7.1 FALSIFY-SHIP-023)",
            );
        }
    }
}
