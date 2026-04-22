// SHIP-TWO-001 AC-SHIP1-005 / FALSIFY-SHIP-005 algorithm-level PARTIAL discharge.
//
// Spec: docs/specifications/aprender-train/ship-two-models-spec.md
// Contract: contracts/qwen2-e2e-verification-v1.yaml (GATE-QW2E-SHIP-005 —
// to be wired in the same PR as this file lands).
//
// AC-SHIP1-005 states that the MODEL-1 teacher
// (`paiml/qwen2.5-coder-7b-apache-q4k-v1`) must reproduce at least
// 86.00% pass@1 on HumanEval (`apr eval --benchmark humaneval`). The
// spec's row AC-SHIP1-005 explicitly carves a 1.2% noise allowance
// (sampling variance across seeds / temperature / vLLM engine drift)
// so the effective ship floor is **84.80%** pass@1 — i.e. if the
// measured score is ≥ 84.80 the teacher passes the ship gate even
// though the nominal target is 86.00.
//
// This file discharges the *decision rule* at `PARTIAL_ALGORITHM_LEVEL`:
// given a (correct, total, threshold_pct) triple, the verdict is `Pass`
// iff `correct / total * 100 ≥ threshold_pct` AND the inputs are well-
// formed (non-zero total, correct ≤ total, finite threshold). The
// compute-heavy portion of the AC (running 164× HumanEval prompts
// through the teacher and computing pass@1) is intentionally out of
// scope here; the threshold rule is what `apr eval` must emit a Pass
// on, and changing either side of the bind (the 86.00 nominal /
// 84.80 noise-adjusted floor, or the ratio shape) breaks this test
// before any eval run is launched.
//
// Mirrors the MODEL-2 pattern set by SHIP-018 (task #151 on branch
// `feat/falsify-ship-018-partial-discharge`). SHIP-005 is the MODEL-1
// twin: identical `(correct, total, threshold_pct) -> Ship005Verdict`
// shape, different nominal floor (86.00 vs 30.00) and a different
// effective floor (84.80 vs 30.00, because MODEL-1 alone carries a
// 1.2% noise allowance). Authored self-contained because SHIP-018
// is not yet on main; once it lands the two `verdict_from_pass_at_1_*`
// fns should be deduplicated into a single parameterized helper with
// the floor constant(s) held externally.
//
// MODEL-1 is now at 5/10 AC-SHIP1 items touched (SHIP-008 + SHIP-009
// + SHIP-006 + SHIP-007 + SHIP-005).

/// Nominal HumanEval pass@1 target for the MODEL-1 teacher, in
/// percent. Derived from the distilled Qwen2.5-Coder-7B headline
/// score reported by the upstream model card (87.20%) minus a 0.4
/// percentage-point haircut for our evaluation pipeline.
///
/// Lockstep with `docs/specifications/aprender-train/ship-two-models-spec.md`
/// §4.2 row AC-SHIP1-005: "`apr eval --benchmark humaneval` reproduces
/// ≥86.00% pass@1 (allow 1.2% noise)".
pub const AC_SHIP1_005_NOMINAL_HUMANEVAL_PASS_AT_1_PCT: f32 = 86.00;

/// Spec-authorized noise allowance, in *percentage points*, carved out
/// of the nominal 86.00% floor. Accounts for sampling variance across
/// seeds, temperature, and vLLM/realizar engine drift — see the
/// discussion under AC-SHIP1-005 in the spec.
pub const AC_SHIP1_005_NOISE_ALLOWANCE_PP: f32 = 1.20;

/// Effective pass@1 floor for the MODEL-1 ship gate: nominal minus
/// noise allowance. A measured pass@1 at or above this value clears
/// AC-SHIP1-005 even though the headline number is 86.00. Holding
/// this as a const locks the arithmetic ("nominal minus noise") at
/// compile time and makes any drift — e.g. silently raising the
/// allowance to 2.0 pp — a test-breaking edit.
pub const AC_SHIP1_005_EFFECTIVE_HUMANEVAL_PASS_AT_1_PCT: f32 =
    AC_SHIP1_005_NOMINAL_HUMANEVAL_PASS_AT_1_PCT - AC_SHIP1_005_NOISE_ALLOWANCE_PP;

/// Binary verdict for FALSIFY-SHIP-005 / GATE-QW2E-SHIP-005.
/// `Pass` iff `correct / total * 100` is at or above `threshold_pct`
/// AND the inputs are well-formed. `Fail` otherwise.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ship005Verdict {
    /// `correct / total * 100 ≥ threshold_pct` with well-formed inputs.
    Pass,
    /// Any of: ratio < threshold; `total == 0`; `correct > total`;
    /// non-finite threshold.
    Fail,
}

/// Algorithm-level verdict rule for FALSIFY-SHIP-005 / GATE-QW2E-SHIP-005
/// / AC-SHIP1-005: pass@1 threshold check. Returns
/// [`Ship005Verdict::Fail`] conservatively for any malformed input so
/// that a harness bug cannot silently ship a failing teacher.
///
/// Conservative-Fail guards:
///
///   - `total == 0` → Fail (avoid division-by-zero; an empty run
///     cannot satisfy a positive threshold).
///   - `correct > total` → Fail (nonsensical input; a real harness
///     can never report more passes than attempts).
///   - `!threshold_pct.is_finite()` → Fail (NaN / ±∞ contract drift —
///     no real ship floor can be non-finite).
///
/// The ratio is computed in f32, consistent with the way
/// `apr eval --benchmark humaneval --json` emits `pass_at_1` in the
/// result payload.
#[must_use]
pub fn verdict_from_pass_at_1(
    correct: usize,
    total: usize,
    threshold_pct: f32,
) -> Ship005Verdict {
    if total == 0 {
        return Ship005Verdict::Fail;
    }
    if correct > total {
        return Ship005Verdict::Fail;
    }
    if !threshold_pct.is_finite() {
        return Ship005Verdict::Fail;
    }
    #[allow(clippy::cast_precision_loss)]
    let ratio_pct = (correct as f32 / total as f32) * 100.0_f32;
    if ratio_pct >= threshold_pct {
        Ship005Verdict::Pass
    } else {
        Ship005Verdict::Fail
    }
}

// ─────────────────────────────────────────────────────────────
// Unit tests — FALSIFY-SHIP-005 algorithm-level proof
// ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod ship_005_tests {
    use super::*;

    /// FALSIFY-SHIP-005 algorithm-level PARTIAL discharge: prove the
    /// two-number threshold rule for AC-SHIP1-005 HumanEval pass@1.
    /// Any edit that changes the 86.00 nominal, the 1.2 pp noise
    /// allowance, the 84.80 effective floor, the inequality direction,
    /// the div-safety / sanity guards, or the non-finite handling
    /// must break this test before a teacher `apr eval` gate runs.
    #[test]
    fn falsify_ship_005_humaneval_pass_at_1_threshold_logic() {
        // Section 1: safely above the effective floor (84.80%) passes.
        // f32 cannot represent 84.80 exactly (nominal − noise yields
        // ~84.79999924), and neither can a fractional ratio like
        // 212/250, so we avoid an exact-boundary test here and instead
        // pick a clean integer ratio comfortably above the floor.
        // 85/100 = 85.0 is f32-exact and strictly above 84.80.
        assert_eq!(
            verdict_from_pass_at_1(
                85,
                100,
                AC_SHIP1_005_EFFECTIVE_HUMANEVAL_PASS_AT_1_PCT,
            ),
            Ship005Verdict::Pass,
            "85/100 = 85.0% must Pass the effective floor",
        );

        // Section 2: above the nominal floor (86.00%) — trivially Pass.
        // 87/100 in f32: 0.87 * 100 rounds very close to 87.0, which
        // is strictly above 86.0.
        assert_eq!(
            verdict_from_pass_at_1(
                87,
                100,
                AC_SHIP1_005_NOMINAL_HUMANEVAL_PASS_AT_1_PCT,
            ),
            Ship005Verdict::Pass,
            "87/100 = 87.00% must Pass the nominal floor",
        );

        // Section 3: the noise-allowance window. 85/100 = 85.00% clears
        // the effective floor but MUST fail against the nominal floor.
        // This is exactly the window the spec's "allow 1.2% noise"
        // carves out — it MUST Fail against the nominal (re-test of
        // the same ratio against a stricter threshold). Section 1
        // already covered the Pass half of this window.
        assert_eq!(
            verdict_from_pass_at_1(
                85,
                100,
                AC_SHIP1_005_NOMINAL_HUMANEVAL_PASS_AT_1_PCT,
            ),
            Ship005Verdict::Fail,
            "85% must Fail against the nominal floor (no noise allowance)",
        );

        // Section 4: below the effective floor — must Fail even with
        // noise allowance. 84/100 = 84.00% < 84.80%.
        assert_eq!(
            verdict_from_pass_at_1(
                84,
                100,
                AC_SHIP1_005_EFFECTIVE_HUMANEVAL_PASS_AT_1_PCT,
            ),
            Ship005Verdict::Fail,
            "84/100 = 84.00% must Fail the effective floor",
        );
        // HumanEval canonical (164 prompts): 139 correct = 84.756%
        // — just barely under 84.80. One of the sharpest real-world
        // counter-examples.
        assert_eq!(
            verdict_from_pass_at_1(
                139,
                164,
                AC_SHIP1_005_EFFECTIVE_HUMANEVAL_PASS_AT_1_PCT,
            ),
            Ship005Verdict::Fail,
            "139/164 = 84.756% must Fail the effective floor",
        );
        // 140/164 = 85.365% — just over effective, just under nominal.
        assert_eq!(
            verdict_from_pass_at_1(
                140,
                164,
                AC_SHIP1_005_EFFECTIVE_HUMANEVAL_PASS_AT_1_PCT,
            ),
            Ship005Verdict::Pass,
            "140/164 = 85.365% must Pass the effective floor",
        );

        // Section 5: monotonicity — for fixed total=164, sweeping
        // correct from 0..=164 against the effective floor, the
        // verdict is Fail up to a transition point and Pass after.
        // Once the transition happens, no higher `correct` may flip
        // back to Fail.
        let mut seen_pass = false;
        for correct in 0..=164 {
            let v = verdict_from_pass_at_1(
                correct,
                164,
                AC_SHIP1_005_EFFECTIVE_HUMANEVAL_PASS_AT_1_PCT,
            );
            if v == Ship005Verdict::Pass {
                seen_pass = true;
            } else if seen_pass {
                panic!(
                    "monotonicity broken: correct={correct} flipped back to Fail after Pass"
                );
            }
        }

        // Section 6: div-safety / sanity guards. `total = 0` must
        // Fail (empty run cannot clear a positive threshold), and
        // `correct > total` must Fail (nonsensical harness output).
        assert_eq!(
            verdict_from_pass_at_1(
                0,
                0,
                AC_SHIP1_005_EFFECTIVE_HUMANEVAL_PASS_AT_1_PCT,
            ),
            Ship005Verdict::Fail,
            "total=0 must Fail (div-by-zero guard)",
        );
        assert_eq!(
            verdict_from_pass_at_1(
                200,
                100,
                AC_SHIP1_005_EFFECTIVE_HUMANEVAL_PASS_AT_1_PCT,
            ),
            Ship005Verdict::Fail,
            "correct>total must Fail (sanity guard)",
        );

        // Section 7: non-finite threshold guard. A telemetry or JSON
        // parse bug that produces NaN / ±∞ must NEVER be promoted
        // to Pass (same "conservative Fail" principle as SHIP-007).
        assert_eq!(
            verdict_from_pass_at_1(164, 164, f32::NAN),
            Ship005Verdict::Fail,
            "NaN threshold must Fail conservatively",
        );
        assert_eq!(
            verdict_from_pass_at_1(164, 164, f32::INFINITY),
            Ship005Verdict::Fail,
            "+∞ threshold must Fail conservatively",
        );
        assert_eq!(
            verdict_from_pass_at_1(164, 164, f32::NEG_INFINITY),
            Ship005Verdict::Fail,
            "-∞ threshold must Fail conservatively",
        );

        // Section 8: provenance pin — the nominal, noise, and
        // effective constants are load-bearing and lockstepped with
        // the spec. If AC-SHIP1-005 ever changes "86.00%" or "1.2%
        // noise", all three constants and this test must move
        // together. An edit that, say, silently widens the noise
        // allowance to 2.0 pp (effective → ~84.00) must break this.
        //
        // We check the effective floor via a tight tolerance (≤ 1e-4)
        // against the nominal value 84.80 rather than an exact equality,
        // because f32 arithmetic for `86.0 - 1.2` yields ~84.79999924,
        // not exactly 84.80. The tolerance is well below the spec's
        // 1.2 pp noise allowance, so silent drift (e.g. to 2.0 pp)
        // would still break this test.
        #[allow(clippy::float_cmp)]
        {
            assert_eq!(
                AC_SHIP1_005_NOMINAL_HUMANEVAL_PASS_AT_1_PCT, 86.00,
                "nominal floor is 86.00% (spec §4.2 AC-SHIP1-005)",
            );
            assert_eq!(
                AC_SHIP1_005_NOISE_ALLOWANCE_PP, 1.20,
                "noise allowance is 1.2 pp (spec §4.2 AC-SHIP1-005)",
            );
        }
        assert!(
            (AC_SHIP1_005_EFFECTIVE_HUMANEVAL_PASS_AT_1_PCT - 84.80).abs() < 1e-4,
            "effective floor must be ~84.80% (nominal − noise); got {}",
            AC_SHIP1_005_EFFECTIVE_HUMANEVAL_PASS_AT_1_PCT,
        );
    }
}
