// SHIP-TWO-001 AC-SHIP1-024 / FALSIFY-SHIP-024 algorithm-level PARTIAL discharge.
//
// Spec: docs/specifications/aprender-train/ship-two-models-spec.md §7.1 row
// `FALSIFY-SHIP-024 | Prompt-injection torture suite (50 adversarial
// inputs) | any panic or NaN in logits`.
// Contract: contracts/qwen2-e2e-verification-v1.yaml (FALSIFY-QW2E-SHIP-024 —
// wired in the same PR as this file lands).
//
// SHIP-TWO-001 §7.1 states that the MODEL-1 teacher must survive an
// adversarial prompt-injection torture suite (at least 50 inputs)
// without any panic or NaN in the emitted logits. The spec's
// falsification rule is "any panic or NaN in logits" — a runtime-
// invariant aggregate gate with zero tolerance on either class of
// failure.
//
// This file discharges the *decision rule* at `PARTIAL_ALGORITHM_LEVEL`
// by binding one pure verdict fn:
//
//   `verdict_from_adversarial_suite(inputs_run, panic_count, nan_count)
//    -> Ship024Verdict` — three-counter aggregate rule. `Pass` iff
//   `inputs_run >= AC_SHIP1_024_MIN_ADVERSARIAL_SUITE_SIZE = 50` AND
//   `panic_count == 0` AND `nan_count == 0`. Any of: fewer than 50
//   adversarial prompts exercised (insufficient torture coverage);
//   any panic; any NaN logit → Fail.
//
// The compute-heavy portion of the AC (actually running 50+ crafted
// adversarial prompts through `apr run` and counting panics / NaN
// logits) is intentionally out of scope here. The threshold rule is
// what the compute harness must emit a Pass on, and changing any of
// the three bound constants
// (`AC_SHIP1_024_MIN_ADVERSARIAL_SUITE_SIZE`,
// `AC_SHIP1_024_MAX_TOLERATED_PANIC_COUNT`,
// `AC_SHIP1_024_MAX_TOLERATED_NAN_COUNT`) breaks this test before
// a single adversarial prompt runs.
//
// Mirrors the zero-tolerance shape of SHIP-002
// (`AC_SHIP1_002_MAX_TOLERATED_SYNTAX_ERRORS = 0`) but extended to
// three counters simultaneously, and the minimum-coverage shape
// implied by SHIP-005's `total > 0` div-safety guard (sharpened here
// to a concrete 50-prompt floor). Unlike the single-counter zero-
// tolerance ships (SHIP-002, SHIP-004 exit code), SHIP-024 combines
// a lower-bound coverage requirement with two upper-bound tolerance
// requirements in one decision rule. SHIP-024 is `ship_blocking:
// false` (spec §7.1 stability test, not in §4.2 AC table).

/// Minimum adversarial-suite size per spec §7.1 FALSIFY-SHIP-024:
/// "Prompt-injection torture suite (50 adversarial inputs)". Any
/// discharge that exercises fewer than 50 crafted adversarial prompts
/// fails the coverage floor; the aggregate gate Fails regardless of
/// panic/NaN counts.
///
/// Lockstep with `docs/specifications/aprender-train/ship-two-models-spec.md`
/// §7.1 row FALSIFY-SHIP-024.
pub const AC_SHIP1_024_MIN_ADVERSARIAL_SUITE_SIZE: usize = 50;

/// Maximum tolerated panics across the adversarial suite. Zero
/// tolerance: any single panic in any one adversarial prompt is a
/// ship-blocker. The spec §7.1 phrasing "any panic or NaN in logits"
/// has no noise allowance.
pub const AC_SHIP1_024_MAX_TOLERATED_PANIC_COUNT: u32 = 0;

/// Maximum tolerated NaN-logit emissions across the adversarial suite.
/// Zero tolerance: any single NaN in the emitted logits on any
/// adversarial prompt is a ship-blocker. The spec §7.1 phrasing
/// "any panic or NaN in logits" has no noise allowance.
pub const AC_SHIP1_024_MAX_TOLERATED_NAN_COUNT: u32 = 0;

/// Binary verdict for FALSIFY-SHIP-024 / AC-SHIP1-024.
/// `Pass` iff all three counter gates are satisfied. `Fail` otherwise.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ship024Verdict {
    /// Suite size ≥ 50 AND panic_count == 0 AND nan_count == 0. The
    /// MODEL-1 teacher survived the adversarial torture suite with
    /// zero runtime defects and sufficient coverage.
    Pass,
    /// Any of: suite size below 50 (insufficient coverage); panic
    /// count > 0 (any panic is a ship-blocker); NaN count > 0 (any
    /// NaN-logit emission is a ship-blocker).
    Fail,
}

/// Pure decision rule for FALSIFY-SHIP-024 / AC-SHIP1-024:
/// adversarial-suite aggregate gate.
///
/// `const fn` so the three zero-tolerance boundaries are baked into
/// the compiled binary at link time and cannot be silently relaxed
/// at runtime via config. Any mutation at the constant level
/// (`MIN_SUITE_SIZE` from 50 to 49, `MAX_PANIC_COUNT` from 0 to 1,
/// `MAX_NAN_COUNT` from 0 to 1) breaks the mutation survey's Section 1
/// boundary pin before any adversarial prompt runs.
///
/// # Examples
///
/// ```
/// use aprender::format::ship_024::{
///     verdict_from_adversarial_suite, Ship024Verdict,
///     AC_SHIP1_024_MIN_ADVERSARIAL_SUITE_SIZE,
/// };
///
/// // Full suite, zero defects → Pass.
/// assert_eq!(
///     verdict_from_adversarial_suite(
///         AC_SHIP1_024_MIN_ADVERSARIAL_SUITE_SIZE, 0, 0
///     ),
///     Ship024Verdict::Pass
/// );
/// // One panic → Fail.
/// assert_eq!(
///     verdict_from_adversarial_suite(
///         AC_SHIP1_024_MIN_ADVERSARIAL_SUITE_SIZE, 1, 0
///     ),
///     Ship024Verdict::Fail
/// );
/// // Insufficient suite size → Fail even with zero defects.
/// assert_eq!(
///     verdict_from_adversarial_suite(49, 0, 0),
///     Ship024Verdict::Fail
/// );
/// ```
#[must_use]
pub const fn verdict_from_adversarial_suite(
    inputs_run: usize,
    panic_count: u32,
    nan_count: u32,
) -> Ship024Verdict {
    if inputs_run < AC_SHIP1_024_MIN_ADVERSARIAL_SUITE_SIZE {
        return Ship024Verdict::Fail;
    }
    if panic_count > AC_SHIP1_024_MAX_TOLERATED_PANIC_COUNT {
        return Ship024Verdict::Fail;
    }
    if nan_count > AC_SHIP1_024_MAX_TOLERATED_NAN_COUNT {
        return Ship024Verdict::Fail;
    }
    Ship024Verdict::Pass
}

// ─────────────────────────────────────────────────────────────
// Unit tests — FALSIFY-SHIP-024 algorithm-level proof
// ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod ship_024_tests {
    use super::*;

    /// FALSIFY-SHIP-024 algorithm-level PARTIAL discharge: prove the
    /// three-counter aggregate rule for AC-SHIP1-024. Any edit that
    /// changes the 50-prompt floor, the zero-tolerance panic budget,
    /// the zero-tolerance NaN budget, or the AND-combinator shape must
    /// break this test before an adversarial torture suite runs.
    #[test]
    fn falsify_ship_024_adversarial_suite_threshold_logic() {
        // Section 1: exact zero-tolerance boundaries. Suite size of
        // exactly 50 AND 0 panics AND 0 NaN logits → Pass. Any single
        // deviation from (50, 0, 0) → Fail.
        assert_eq!(
            verdict_from_adversarial_suite(50, 0, 0),
            Ship024Verdict::Pass,
            "canonical (50, 0, 0) must Pass (exact boundary)",
        );
        assert_eq!(
            verdict_from_adversarial_suite(50, 1, 0),
            Ship024Verdict::Fail,
            "(50, 1, 0) one panic must Fail (zero-tolerance)",
        );
        assert_eq!(
            verdict_from_adversarial_suite(50, 0, 1),
            Ship024Verdict::Fail,
            "(50, 0, 1) one NaN must Fail (zero-tolerance)",
        );

        // Section 2: insufficient suite size. The spec's 50-prompt
        // floor is a coverage requirement — fewer exercised inputs is
        // insufficient torture and must Fail even with zero defects.
        // 49 is the sharpest counter-example; 0 is the degenerate
        // no-harness-ran case.
        assert_eq!(
            verdict_from_adversarial_suite(49, 0, 0),
            Ship024Verdict::Fail,
            "(49, 0, 0) one-short of floor must Fail (insufficient coverage)",
        );
        assert_eq!(
            verdict_from_adversarial_suite(0, 0, 0),
            Ship024Verdict::Fail,
            "(0, 0, 0) empty harness must Fail (insufficient coverage)",
        );
        for &n in &[1_usize, 10, 25, 40, 49] {
            assert_eq!(
                verdict_from_adversarial_suite(n, 0, 0),
                Ship024Verdict::Fail,
                "inputs_run={n} below floor must Fail",
            );
        }

        // Section 3: over-size Pass band — exercising more than 50
        // adversarial prompts with zero defects must still Pass. The
        // 50 floor is a minimum, not a fixed point.
        for &n in &[51_usize, 100, 200, 500, 1000, 10_000, usize::MAX] {
            assert_eq!(
                verdict_from_adversarial_suite(n, 0, 0),
                Ship024Verdict::Pass,
                "inputs_run={n} above floor with zero defects must Pass",
            );
        }

        // Section 4: single-failure-class counts. Any nonzero panic
        // count must Fail regardless of NaN count; any nonzero NaN
        // count must Fail regardless of panic count. Covers the
        // isolated-class-failure modes.
        for &p in &[1_u32, 2, 5, 10, 100, 1_000] {
            assert_eq!(
                verdict_from_adversarial_suite(50, p, 0),
                Ship024Verdict::Fail,
                "(50, {p}, 0) must Fail (any panic count > 0)",
            );
        }
        for &n in &[1_u32, 2, 5, 10, 100, 1_000] {
            assert_eq!(
                verdict_from_adversarial_suite(50, 0, n),
                Ship024Verdict::Fail,
                "(50, 0, {n}) must Fail (any NaN count > 0)",
            );
        }

        // Section 5: compound failures — both panic AND NaN counters
        // nonzero must Fail. Catches a combined runtime defect class
        // where both error modes occurred in the same adversarial
        // prompt run.
        assert_eq!(
            verdict_from_adversarial_suite(50, 1, 1),
            Ship024Verdict::Fail,
            "(50, 1, 1) compound failure must Fail",
        );
        assert_eq!(
            verdict_from_adversarial_suite(50, 5, 5),
            Ship024Verdict::Fail,
            "(50, 5, 5) compound failure must Fail",
        );
        // Compound failure below floor is still Fail (triple-compound
        // — insufficient + panic + NaN).
        assert_eq!(
            verdict_from_adversarial_suite(10, 10, 10),
            Ship024Verdict::Fail,
            "(10, 10, 10) triple-compound must Fail",
        );

        // Section 6: u32::MAX on both counters must Fail. Catches a
        // telemetry bug that silently saturates on overflow; we must
        // not silently promote Pass at the wraparound edge.
        assert_eq!(
            verdict_from_adversarial_suite(50, u32::MAX, 0),
            Ship024Verdict::Fail,
            "(50, u32::MAX, 0) panic overflow must Fail",
        );
        assert_eq!(
            verdict_from_adversarial_suite(50, 0, u32::MAX),
            Ship024Verdict::Fail,
            "(50, 0, u32::MAX) NaN overflow must Fail",
        );
        assert_eq!(
            verdict_from_adversarial_suite(50, u32::MAX, u32::MAX),
            Ship024Verdict::Fail,
            "(50, u32::MAX, u32::MAX) both-overflow must Fail",
        );
        // And with sufficient usize::MAX coverage still Fails on any
        // nonzero counter.
        assert_eq!(
            verdict_from_adversarial_suite(usize::MAX, 1, 0),
            Ship024Verdict::Fail,
            "(usize::MAX, 1, 0) any panic must Fail even at max coverage",
        );

        // Section 7: provenance pin — all three constants are load-
        // bearing and lockstepped with the spec. If §7.1 FALSIFY-SHIP-024
        // ever loosens the coverage floor (say from 50 to 30) or
        // introduces a nonzero tolerance (say to allow 1 NaN), these
        // constants must move in lockstep and the rest of the survey
        // follows.
        assert_eq!(
            AC_SHIP1_024_MIN_ADVERSARIAL_SUITE_SIZE, 50_usize,
            "adversarial suite minimum size is 50 (spec §7.1 FALSIFY-SHIP-024)",
        );
        assert_eq!(
            AC_SHIP1_024_MAX_TOLERATED_PANIC_COUNT, 0_u32,
            "panic tolerance is 0 (spec §7.1 FALSIFY-SHIP-024)",
        );
        assert_eq!(
            AC_SHIP1_024_MAX_TOLERATED_NAN_COUNT, 0_u32,
            "NaN-logit tolerance is 0 (spec §7.1 FALSIFY-SHIP-024)",
        );
    }
}
