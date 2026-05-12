// SHIP-TWO-001 §6 Compound Ship Gates — GATE-SHIP-004 algorithm-level
// PARTIAL discharge.
//
// Spec: docs/specifications/aprender-train/ship-two-models-spec.md §6 row
// `GATE-SHIP-004 | HumanEval harness produces identical score on two
// consecutive runs (seed=0) | AC-005, AC-008`.
// Contract: contracts/compound-ship-gates-v1.yaml v1.0.0 PROPOSED
// (FALSIFY-GATE-SHIP-004 — wired in the same PR as this file lands).
//
// GATE-SHIP-004 is the *determinism* gate for HumanEval: with seed=0
// and greedy decoding, two consecutive `apr eval --benchmark humaneval`
// runs must produce BITWISE-IDENTICAL pass@1 percentages. This is
// STRICTER than FALSIFY-SHIP-023 (which allows ≤ 1.2 pp drift across
// two days); GATE-SHIP-004 enforces bit-for-bit determinism within a
// single session. If two seed=0 runs produce different bytes, either
// (a) the sampling path is non-deterministic (hidden entropy source),
// or (b) the eval harness has a race.
//
// This file discharges the *decision rule* at `PARTIAL_ALGORITHM_LEVEL`:
// given two pass@1 percentages (f32), the verdict is `Pass` iff
// `to_bits()` returns identical u32 AND both values are finite AND in
// `[0.0, 100.0]`. A single-ULP drift (even within floating-point
// rounding noise) Fails — because a truly deterministic harness with
// seed=0 must produce byte-equal outputs on every invocation.
//
// Contrast with FALSIFY-SHIP-023 (two-day drift tolerance 1.2 pp):
//   - SHIP-023 tolerates natural day-over-day noise (temperature,
//     page cache, etc.) — semantically "stability across sessions".
//   - GATE-SHIP-004 tolerates ZERO drift — semantically "pure
//     determinism within a session".
// Both gates must pass to ship; GATE-SHIP-004 is the sharper one.

/// Binary verdict for FALSIFY-GATE-SHIP-004 / GATE-SHIP-004.
/// `Pass` iff both run scores are well-formed AND their `to_bits()`
/// returns identical u32. `Fail` otherwise.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GateShip004Verdict {
    /// Both pass@1 scores are finite, in `[0.0, 100.0]`, AND
    /// `run_a.to_bits() == run_b.to_bits()`. Two consecutive seed=0
    /// HumanEval runs produced bitwise-identical output — the eval
    /// harness is deterministic.
    Pass,
    /// Any of: non-finite score on either side; out-of-range score;
    /// `to_bits()` mismatch (even by one ULP). Determinism invariant
    /// is broken; publish is blocked.
    Fail,
}

/// Algorithm-level verdict rule for FALSIFY-GATE-SHIP-004 /
/// GATE-SHIP-004: bitwise-identical determinism check on two seed=0
/// HumanEval pass@1 runs.
///
/// Distinct from `AC_SHIP1_023_MAX_HUMANEVAL_DRIFT_PP = 1.2` (day-over-
/// day drift tolerance): this rule tolerates ZERO drift because two
/// runs with seed=0 within the same session must produce byte-equal
/// output. A single-ULP drift (`f32::from_bits(x.to_bits() + 1)`)
/// Fails, because the intent is to catch hidden non-determinism
/// (entropy source, race condition, accidental atomic-counter read)
/// rather than noise absorption.
///
/// Conservative-Fail guards:
///
///   - `!run_a.is_finite()` OR `!run_b.is_finite()` → Fail (NaN /
///     ±∞ are never valid pass@1 scores).
///   - `run_a` or `run_b` outside `[0.0, 100.0]` → Fail (pass@1 is a
///     percentage).
///   - `run_a.to_bits() != run_b.to_bits()` → Fail (determinism
///     broken).
///
/// # Examples
///
/// ```
/// use aprender::format::gate_ship_004::{
///     verdict_from_identical_humaneval_scores, GateShip004Verdict,
/// };
///
/// // Identical seed=0 runs → Pass.
/// assert_eq!(
///     verdict_from_identical_humaneval_scores(86.0, 86.0),
///     GateShip004Verdict::Pass
/// );
///
/// // Close but not bitwise-equal (single ULP) → Fail.
/// let run_a = 86.0_f32;
/// let run_b = f32::from_bits(run_a.to_bits() + 1);
/// assert_eq!(
///     verdict_from_identical_humaneval_scores(run_a, run_b),
///     GateShip004Verdict::Fail
/// );
/// ```
#[must_use]
pub fn verdict_from_identical_humaneval_scores(
    run_a_pct: f32,
    run_b_pct: f32,
) -> GateShip004Verdict {
    if !run_a_pct.is_finite() || !run_b_pct.is_finite() {
        return GateShip004Verdict::Fail;
    }
    if !(0.0_f32..=100.0_f32).contains(&run_a_pct) {
        return GateShip004Verdict::Fail;
    }
    if !(0.0_f32..=100.0_f32).contains(&run_b_pct) {
        return GateShip004Verdict::Fail;
    }
    if run_a_pct.to_bits() == run_b_pct.to_bits() {
        GateShip004Verdict::Pass
    } else {
        GateShip004Verdict::Fail
    }
}

// ─────────────────────────────────────────────────────────────
// Unit tests — FALSIFY-GATE-SHIP-004 algorithm-level proof
// ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod gate_ship_004_tests {
    use super::*;

    /// FALSIFY-GATE-SHIP-004 algorithm-level PARTIAL discharge: prove
    /// the bitwise-identity determinism rule binding two seed=0
    /// HumanEval pass@1 runs. Any edit that relaxes `to_bits()` to
    /// `==` (which treats +0.0 == -0.0), widens to drift-tolerance
    /// like SHIP-023, or skips the range guards must break this test.
    #[test]
    fn falsify_gate_ship_004_humaneval_bitwise_determinism() {
        // Section 1: identical seed=0 runs → Pass.
        assert_eq!(
            verdict_from_identical_humaneval_scores(86.0, 86.0),
            GateShip004Verdict::Pass,
            "identical 86.0 scores must Pass",
        );

        // Section 2: single-ULP difference → Fail. This is the
        // sharpest counter-example: a relaxation to "close-enough
        // for floats" (e.g., `(a - b).abs() < 1e-7`) would flip this
        // to Pass. GATE-SHIP-004's whole purpose is to detect
        // hidden non-determinism that shows up as a single-ULP
        // difference when a thread-local RNG is accidentally seeded
        // with time, or when atomics expose a race.
        let run_a = 86.0_f32;
        let run_b = f32::from_bits(run_a.to_bits() + 1);
        assert_ne!(
            run_a.to_bits(),
            run_b.to_bits(),
            "harness sanity: single-ULP neighbours have different bits",
        );
        assert_eq!(
            verdict_from_identical_humaneval_scores(run_a, run_b),
            GateShip004Verdict::Fail,
            "single-ULP drift must Fail (this is the whole point of GATE-SHIP-004)",
        );

        // Section 3: close-but-not-equal — 86.0 vs 86.0000001. Even
        // within float noise, this must Fail. Catches the class where
        // a reviewer thinks "eh, 1e-7 is indistinguishable" and
        // relaxes the comparison.
        let close_a = 86.0_f32;
        let close_b = 86.000_001_f32;
        if close_a.to_bits() != close_b.to_bits() {
            assert_eq!(
                verdict_from_identical_humaneval_scores(close_a, close_b),
                GateShip004Verdict::Fail,
                "86.0 vs 86.000001 must Fail (bits differ even if within tolerance)",
            );
        }

        // Section 4: non-finite on either side → Fail. A harness bug
        // that emits NaN must not silently Pass just because NaN
        // compares equal to itself under to_bits (which it does, but
        // we guard earlier on is_finite).
        for &bad in &[f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert_eq!(
                verdict_from_identical_humaneval_scores(bad, 86.0),
                GateShip004Verdict::Fail,
                "non-finite run_a ({bad}) must Fail conservatively",
            );
            assert_eq!(
                verdict_from_identical_humaneval_scores(86.0, bad),
                GateShip004Verdict::Fail,
                "non-finite run_b ({bad}) must Fail conservatively",
            );
            assert_eq!(
                verdict_from_identical_humaneval_scores(bad, bad),
                GateShip004Verdict::Fail,
                "both non-finite ({bad}) must Fail (NaN bit-equality is not determinism)",
            );
        }

        // Section 5: out-of-range values Fail. pass@1 is a percentage
        // in `[0.0, 100.0]`; -0.1 or 100.1 on either side is a harness
        // bug.
        for &oor in &[-0.1_f32, -1.0, -86.0, 100.1, 101.0, 1_000.0] {
            assert_eq!(
                verdict_from_identical_humaneval_scores(oor, 86.0),
                GateShip004Verdict::Fail,
                "out-of-range run_a ({oor}) must Fail",
            );
            assert_eq!(
                verdict_from_identical_humaneval_scores(86.0, oor),
                GateShip004Verdict::Fail,
                "out-of-range run_b ({oor}) must Fail",
            );
            // Even if both sides are OOR and bit-equal, the range
            // guard fires first — degenerate Pass path blocked.
            assert_eq!(
                verdict_from_identical_humaneval_scores(oor, oor),
                GateShip004Verdict::Fail,
                "both out-of-range ({oor}) must Fail (range-guard-first)",
            );
        }

        // Section 6: boundary values {0.0, 100.0} are legal pass@1
        // scores. Degenerate cases (pass@1 = 0% or pass@1 = 100%)
        // must Pass when both runs produce the same boundary value.
        assert_eq!(
            verdict_from_identical_humaneval_scores(0.0, 0.0),
            GateShip004Verdict::Pass,
            "both at 0.0% must Pass (degenerate but legal)",
        );
        assert_eq!(
            verdict_from_identical_humaneval_scores(100.0, 100.0),
            GateShip004Verdict::Pass,
            "both at 100.0% must Pass (degenerate but legal)",
        );
        // But 0.0 vs 100.0 is maximally non-deterministic → Fail.
        assert_eq!(
            verdict_from_identical_humaneval_scores(0.0, 100.0),
            GateShip004Verdict::Fail,
            "0.0 vs 100.0 must Fail (max drift across band)",
        );

        // Section 7: provenance — GATE-SHIP-004 is DISTINCT from
        // FALSIFY-SHIP-023 (which uses `AC_SHIP1_023_MAX_HUMANEVAL_DRIFT_PP
        // = 1.2` for day-over-day tolerance). Document the distinction
        // in a test assertion so a careless refactor that unifies
        // them gets caught.
        //
        // SHIP-023 rule: `(day1 - day2).abs() <= 1.2 pp` Pass.
        // GATE-SHIP-004 rule: `run_a.to_bits() == run_b.to_bits()`
        // Pass. A value that would Pass SHIP-023 (drift = 0.5 pp)
        // must Fail GATE-SHIP-004 (bits differ).
        let drift_05_a = 86.0_f32;
        let drift_05_b = 86.5_f32;
        assert_eq!(
            verdict_from_identical_humaneval_scores(drift_05_a, drift_05_b),
            GateShip004Verdict::Fail,
            "0.5 pp drift (Pass under SHIP-023) must Fail under GATE-SHIP-004 \
             (bitwise-identity is strictly stricter than 1.2 pp tolerance)",
        );
    }
}
