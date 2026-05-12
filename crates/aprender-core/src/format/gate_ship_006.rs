// SHIP-TWO-001 §6 Compound Ship Gates — GATE-SHIP-006 algorithm-level
// PARTIAL discharge.
//
// Spec: docs/specifications/aprender-train/ship-two-models-spec.md §6 row
// `GATE-SHIP-006 | GGUF round-trip: APR → GGUF → load in llama.cpp →
// first-token match (tol ≤ 1e-3) | AC-004, AC-009`.
// Contract: contracts/compound-ship-gates-v1.yaml v1.0.0 PROPOSED
// (FALSIFY-GATE-SHIP-006 — wired in the same PR as this file lands).
//
// GATE-SHIP-006 is the *cross-engine parity* gate for the APR → GGUF
// export path: an APR model converted to GGUF and loaded in llama.cpp
// MUST produce a first-token probability within 1e-3 of what the APR
// inference path produces on the same prompt. This bounds quantization
// drift AND export-pipeline bugs AND llama.cpp dequantization
// differences into a single scalar threshold.
//
// This file discharges the *decision rule* at `PARTIAL_ALGORITHM_LEVEL`:
// given (p_apr, p_gguf) first-token probabilities and a tolerance, the
// verdict is `Pass` iff all three are finite, both probabilities live
// in the valid range `[0.0, 1.0]`, `tolerance >= 0.0`, AND
// `(p_apr - p_gguf).abs() <= tolerance`. The compute-heavy portion
// (actually running the APR model on a canonical prompt, exporting to
// GGUF via `apr export --format gguf`, re-loading in llama.cpp, and
// extracting the first-token logit) is intentionally out of scope
// here.
//
// Mirrors the single-threshold shape of SHIP-003 (cosine similarity
// floor) and SHIP-023 (drift threshold), but specialized for
// probabilities in [0.0, 1.0] rather than percentages in [0.0, 100.0]
// or cosines in [-1.0, 1.0]. The `.abs()` combinator makes this
// symmetric: it doesn't matter whether APR or GGUF is "higher" — only
// the magnitude of the drift matters for cross-engine parity.

/// Maximum tolerated absolute difference in first-token probability
/// between the APR inference path and the GGUF inference path (llama.cpp
/// after `apr export --format gguf`). Lockstep with
/// `docs/specifications/aprender-train/ship-two-models-spec.md` §6
/// row GATE-SHIP-006: "first-token match (tol ≤ 1e-3)".
///
/// f32-exact: `1e-3` in IEEE-754 binary32 is `0x3A83126F` =
/// 0.00099999994... — close to but not exactly 0.001. The mutation
/// survey uses this stored f32 for boundary comparisons.
pub const AC_GATE_SHIP_006_MAX_FIRST_TOKEN_DELTA: f32 = 1e-3;

/// Binary verdict for FALSIFY-GATE-SHIP-006 / GATE-SHIP-006.
/// `Pass` iff all inputs are well-formed AND the absolute first-token
/// probability delta is at or below the tolerance. `Fail` otherwise.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GateShip006Verdict {
    /// Well-formed inputs AND `|p_apr − p_gguf| ≤ tolerance`. The APR
    /// and GGUF inference paths agree on the first-token probability
    /// within the spec's 1e-3 tolerance — cross-engine parity holds.
    Pass,
    /// Any of: non-finite input on any of the 3 args; `p_apr` or
    /// `p_gguf` outside `[0.0, 1.0]`; `tolerance < 0.0`;
    /// `|p_apr − p_gguf| > tolerance`. GGUF export path has drifted
    /// from APR reference; ship is blocked.
    Fail,
}

/// Pure decision rule for FALSIFY-GATE-SHIP-006 / GATE-SHIP-006:
/// APR ↔ GGUF first-token probability parity check.
///
/// Conservative-Fail guards:
///
///   - `!p_apr.is_finite()` OR `!p_gguf.is_finite()` OR
///     `!tolerance.is_finite()` → Fail (NaN / ±∞ are never valid
///     probabilities or tolerances).
///   - `p_apr` or `p_gguf` outside `[0.0, 1.0]` → Fail (probabilities
///     are in the unit interval; anything else is a harness bug).
///   - `tolerance < 0.0` → Fail (contract drift — negative tolerance
///     is nonsense; 0.0 is legal and means "bitwise-identical").
///   - `(p_apr - p_gguf).abs() > tolerance` → Fail (drift exceeds
///     tolerance).
///
/// The `.abs()` combinator makes this symmetric: swapping `p_apr` and
/// `p_gguf` yields the same verdict. Cross-engine parity is
/// directionless — we don't care whether llama.cpp is slightly above
/// or slightly below APR, only whether the magnitude is within
/// tolerance.
///
/// # Examples
///
/// ```
/// use aprender::format::gate_ship_006::{
///     verdict_from_first_token_probability_delta, GateShip006Verdict,
///     AC_GATE_SHIP_006_MAX_FIRST_TOKEN_DELTA,
/// };
///
/// // APR and GGUF agree within tolerance → Pass.
/// assert_eq!(
///     verdict_from_first_token_probability_delta(
///         0.5, 0.5005, AC_GATE_SHIP_006_MAX_FIRST_TOKEN_DELTA,
///     ),
///     GateShip006Verdict::Pass
/// );
///
/// // Drift exceeds tolerance → Fail.
/// assert_eq!(
///     verdict_from_first_token_probability_delta(
///         0.5, 0.502, AC_GATE_SHIP_006_MAX_FIRST_TOKEN_DELTA,
///     ),
///     GateShip006Verdict::Fail
/// );
/// ```
#[must_use]
pub const fn verdict_from_first_token_probability_delta(
    p_apr: f32,
    p_gguf: f32,
    tolerance: f32,
) -> GateShip006Verdict {
    if !p_apr.is_finite() || !p_gguf.is_finite() || !tolerance.is_finite() {
        return GateShip006Verdict::Fail;
    }
    if p_apr < 0.0 || p_apr > 1.0 {
        return GateShip006Verdict::Fail;
    }
    if p_gguf < 0.0 || p_gguf > 1.0 {
        return GateShip006Verdict::Fail;
    }
    if tolerance < 0.0 {
        return GateShip006Verdict::Fail;
    }
    let delta = if p_apr >= p_gguf {
        p_apr - p_gguf
    } else {
        p_gguf - p_apr
    };
    if delta <= tolerance {
        GateShip006Verdict::Pass
    } else {
        GateShip006Verdict::Fail
    }
}

// ─────────────────────────────────────────────────────────────
// Unit tests — FALSIFY-GATE-SHIP-006 algorithm-level proof
// ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod gate_ship_006_tests {
    use super::*;

    /// FALSIFY-GATE-SHIP-006 algorithm-level PARTIAL discharge: prove
    /// the first-token parity threshold rule. Any edit that widens the
    /// 1e-3 tolerance, relaxes the range guards, or flips the
    /// inequality must break this test before a real GGUF round-trip
    /// harness runs.
    #[test]
    fn falsify_gate_ship_006_first_token_probability_delta() {
        // Section 1: exact-boundary delta = 0.0 at p = 0.5 — Pass
        // (degenerate bitwise-match case).
        assert_eq!(
            verdict_from_first_token_probability_delta(
                0.5,
                0.5,
                AC_GATE_SHIP_006_MAX_FIRST_TOKEN_DELTA,
            ),
            GateShip006Verdict::Pass,
            "delta = 0.0 at p = 0.5 must Pass (bitwise-identical case)",
        );

        // Section 2: delta exactly at the tolerance boundary (1e-3)
        // — inclusive Pass. The boundary case must be Pass because the
        // rule is `<=`, not `<`. A mutation that flips to `<` would
        // fail this.
        //
        // Note: f32 subtraction may introduce 1 ULP rounding, so we
        // construct the pair such that the subtraction is exact. With
        // p_apr = 1e-3 and p_gguf = 0.0, the subtraction is just
        // `1e-3`, which matches the tolerance bit-for-bit.
        assert_eq!(
            verdict_from_first_token_probability_delta(
                AC_GATE_SHIP_006_MAX_FIRST_TOKEN_DELTA,
                0.0,
                AC_GATE_SHIP_006_MAX_FIRST_TOKEN_DELTA,
            ),
            GateShip006Verdict::Pass,
            "delta exactly at tolerance must Pass (inclusive `<=`)",
        );

        // Section 3: one-ULP-over — sharpest Fail counter-example. Any
        // mutation that softens `<=` to "close-enough" or widens the
        // tolerance must flip this to Pass.
        let just_above = f32::from_bits(AC_GATE_SHIP_006_MAX_FIRST_TOKEN_DELTA.to_bits() + 1);
        assert!(
            just_above > AC_GATE_SHIP_006_MAX_FIRST_TOKEN_DELTA,
            "harness sanity: one-ULP neighbour exceeds tolerance",
        );
        assert_eq!(
            verdict_from_first_token_probability_delta(
                just_above,
                0.0,
                AC_GATE_SHIP_006_MAX_FIRST_TOKEN_DELTA,
            ),
            GateShip006Verdict::Fail,
            "delta = tolerance + 1 ULP must Fail (sharpest counter-example)",
        );

        // Section 4: clear Pass band — small probabilities with small
        // deltas (the realistic case for quantized inference near
        // low-confidence tokens).
        assert_eq!(
            verdict_from_first_token_probability_delta(
                0.001,
                0.0005,
                AC_GATE_SHIP_006_MAX_FIRST_TOKEN_DELTA,
            ),
            GateShip006Verdict::Pass,
            "small probs with delta = 0.0005 must Pass",
        );
        // Symmetric direction (swapped) — same verdict (`.abs()`).
        assert_eq!(
            verdict_from_first_token_probability_delta(
                0.0005,
                0.001,
                AC_GATE_SHIP_006_MAX_FIRST_TOKEN_DELTA,
            ),
            GateShip006Verdict::Pass,
            "small probs swapped must Pass (symmetry via .abs())",
        );
        // Mid-band probabilities with in-tolerance drift.
        assert_eq!(
            verdict_from_first_token_probability_delta(
                0.5,
                0.5009,
                AC_GATE_SHIP_006_MAX_FIRST_TOKEN_DELTA,
            ),
            GateShip006Verdict::Pass,
            "0.5 vs 0.5009 (delta 0.0009) must Pass",
        );
        // Boundary p = 1.0 (unlikely in sampling but legal after
        // softmax-argmax).
        assert_eq!(
            verdict_from_first_token_probability_delta(
                1.0,
                0.9995,
                AC_GATE_SHIP_006_MAX_FIRST_TOKEN_DELTA,
            ),
            GateShip006Verdict::Pass,
            "p = 1.0 with delta 0.0005 must Pass",
        );

        // Section 5: out-of-range probabilities Fail. Both -0.001 and
        // 1.001 are harness bugs (softmax should produce values in
        // [0.0, 1.0] exactly).
        assert_eq!(
            verdict_from_first_token_probability_delta(
                -0.001,
                0.5,
                AC_GATE_SHIP_006_MAX_FIRST_TOKEN_DELTA,
            ),
            GateShip006Verdict::Fail,
            "p_apr = -0.001 (below 0.0) must Fail",
        );
        assert_eq!(
            verdict_from_first_token_probability_delta(
                0.5,
                1.001,
                AC_GATE_SHIP_006_MAX_FIRST_TOKEN_DELTA,
            ),
            GateShip006Verdict::Fail,
            "p_gguf = 1.001 (above 1.0) must Fail",
        );
        assert_eq!(
            verdict_from_first_token_probability_delta(
                2.0,
                2.0,
                AC_GATE_SHIP_006_MAX_FIRST_TOKEN_DELTA,
            ),
            GateShip006Verdict::Fail,
            "both out-of-range (even if delta = 0) must Fail",
        );
        // Negative tolerance → Fail.
        assert_eq!(
            verdict_from_first_token_probability_delta(0.5, 0.5, -1e-6),
            GateShip006Verdict::Fail,
            "negative tolerance must Fail (contract drift)",
        );

        // Section 6: non-finite on any of the three args → conservative
        // Fail.
        for &bad in &[f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert_eq!(
                verdict_from_first_token_probability_delta(
                    bad,
                    0.5,
                    AC_GATE_SHIP_006_MAX_FIRST_TOKEN_DELTA,
                ),
                GateShip006Verdict::Fail,
                "non-finite p_apr ({bad}) must Fail",
            );
            assert_eq!(
                verdict_from_first_token_probability_delta(
                    0.5,
                    bad,
                    AC_GATE_SHIP_006_MAX_FIRST_TOKEN_DELTA,
                ),
                GateShip006Verdict::Fail,
                "non-finite p_gguf ({bad}) must Fail",
            );
            assert_eq!(
                verdict_from_first_token_probability_delta(0.5, 0.5, bad),
                GateShip006Verdict::Fail,
                "non-finite tolerance ({bad}) must Fail",
            );
        }

        // Section 7: provenance pin — the 1e-3 constant is load-bearing
        // and lockstepped with spec §6. If the tolerance ever widens
        // (say to 5e-3 because llama.cpp's dequantizer drifts), this
        // constant and every test in this module must move together.
        #[allow(clippy::float_cmp)]
        {
            assert_eq!(
                AC_GATE_SHIP_006_MAX_FIRST_TOKEN_DELTA, 1e-3_f32,
                "max first-token delta is 1e-3 (spec §6 GATE-SHIP-006)",
            );
        }
    }
}
