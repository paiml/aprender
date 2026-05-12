//! FALSIFY-GPUTRAIN-005 / INV-GPUTRAIN-005 / GATE-GPUTRAIN-004 — algorithm-level
//! PARTIAL discharge.
//!
//! Spec: `docs/specifications/aprender-train/ship-two-models-spec.md` §14
//! (task #132 CUDA training backend gap).
//!
//! Contract: `contracts/entrenar/gpu-training-backend-v1.yaml` v1.0.0 → v1.1.0
//! adds `GATE-GPUTRAIN-004` PARTIAL-discharge evidence binding the pure
//! f32-threshold decision rule for the 370M Llama scaffold step-time budget
//! on the RTX 4090 reference host.
//!
//! INV-GPUTRAIN-005 states: "370M step time < 500 ms on RTX 4090 (seq_len=2048,
//! batch=1, sm_89 pre-compiled)." This file discharges the *decision rule* at
//! `PARTIAL_ALGORITHM_LEVEL` via a single pure const fn:
//!
//!   `verdict_from_step_time_ms(measured_ms, threshold_ms) -> Gputrain005Verdict`
//!
//! Pass iff every one of these holds:
//!
//!   - `measured_ms.is_finite()` (no NaN / ±∞)
//!   - `threshold_ms.is_finite()` (no NaN / ±∞)
//!   - `measured_ms >= 0.0` (physical: step time can't be negative)
//!   - `threshold_ms > 0.0` (a zero or negative ceiling is nonsense)
//!   - `measured_ms <= threshold_ms`
//!
//! Mirrors SHIP-007 (decode tps) and SHIP-020 (decode tps, 370M target) shape
//! but with an inequality direction flipped (step time is "lower is better,"
//! throughput is "higher is better"), exactly as INV-GPUTRAIN-005's rule text
//! prescribes.
//!
//! The compute-heavy portion (actually running 50 training steps on an RTX
//! 4090 and reading wall times from train.jsonl) is intentionally out of
//! scope here; the threshold rule is what the live gate calls, and
//! changing the 500-ms constant or the `<=` operator breaks this test
//! before any trainer is allocated.

/// Maximum tolerated median step wall time for the 370M Llama scaffold on
/// the RTX 4090 reference host (seq_len=2048, batch=1, sm_89 pre-compiled).
///
/// Derivation: `TransformerTrainer` backward + AdamW step at 370M on sm_89
/// should land well under this ceiling; 500 ms is the ship floor. A
/// measured median above 500 ms is recorded as FM-GPUTRAIN-UNDER-BUDGET
/// (contracts/entrenar/gpu-training-backend-v1.yaml §failure_modes).
pub const AC_GPUTRAIN_005_MAX_STEP_TIME_MS_RTX4090_370M: f32 = 500.0;

/// Binary verdict for FALSIFY-GPUTRAIN-005 / GATE-GPUTRAIN-004.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Gputrain005Verdict {
    /// Measured median step time is within budget.
    Pass,
    /// Over budget, negative, non-finite, or any sanity-rail violation —
    /// all conservatively Fail.
    Fail,
}

/// Algorithm-level verdict rule for INV-GPUTRAIN-005: given a measured
/// median step wall time (ms) and the published threshold (ms), Pass iff
/// both are finite, measured is non-negative, threshold is strictly
/// positive, and measured ≤ threshold.
///
/// `const fn` so the boundary at exactly `AC_GPUTRAIN_005_MAX_STEP_TIME_MS_
/// RTX4090_370M` is const-evaluable; any mutation to the comparison
/// operator or the finiteness guard is caught at compile+test time.
#[must_use]
pub const fn verdict_from_step_time_ms(measured_ms: f32, threshold_ms: f32) -> Gputrain005Verdict {
    if !measured_ms.is_finite() || !threshold_ms.is_finite() {
        return Gputrain005Verdict::Fail;
    }
    if measured_ms < 0.0 {
        return Gputrain005Verdict::Fail;
    }
    if threshold_ms <= 0.0 {
        return Gputrain005Verdict::Fail;
    }
    if measured_ms <= threshold_ms {
        Gputrain005Verdict::Pass
    } else {
        Gputrain005Verdict::Fail
    }
}

// ─────────────────────────────────────────────────────────────
// Unit tests — FALSIFY-GPUTRAIN-005 algorithm-level proof
// ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// FALSIFY-GPUTRAIN-005 algorithm-level PARTIAL discharge: prove the
    /// 500-ms step-time ceiling rule. Any mutation that swaps `<=` for
    /// `<`, relaxes finiteness, or silently accepts a negative step time
    /// must break this test before the live RTX 4090 dispatch runs.
    #[test]
    fn falsify_gputrain_005_step_time_threshold_logic() {
        let threshold = AC_GPUTRAIN_005_MAX_STEP_TIME_MS_RTX4090_370M;

        // Section 1: boundary — exactly at threshold. The contract rule
        // text uses "<" prose but the falsifier prediction binds to "≤"
        // (spec §14.4 Phase-3 evidence: "median wall_ms < 500" is
        // evaluated against an inclusive ceiling per FALSIFY spec). Pass
        // at exactly 500.0 ms catches any mutation that flipped to
        // strict `<`.
        assert_eq!(
            verdict_from_step_time_ms(500.0, threshold),
            Gputrain005Verdict::Pass,
            "measured == threshold (500.0 ms) must Pass per inclusive ceiling",
        );

        // Section 2: one ULP above threshold. `f32::from_bits(threshold
        // .to_bits() + 1)` is the next representable f32 — 500.00006
        // ish. Any mutation that drops the `<=` check (e.g. `< +
        // tolerance`) flips this to Pass.
        let one_ulp_above = f32::from_bits(threshold.to_bits() + 1);
        assert!(one_ulp_above > threshold);
        assert_eq!(
            verdict_from_step_time_ms(one_ulp_above, threshold),
            Gputrain005Verdict::Fail,
            "one ULP above threshold (~500.00006 ms) must Fail",
        );

        // Section 3: clear Pass band. Values safely under budget —
        // these are the target perf regime for a correctly-wired CUDA
        // training dispatch.
        for &measured in &[0.0f32, 1.0, 100.0, 250.0, 400.0, 499.999] {
            assert_eq!(
                verdict_from_step_time_ms(measured, threshold),
                Gputrain005Verdict::Pass,
                "measured={measured} ms (well under {threshold} ms) must Pass",
            );
        }

        // Section 4: clear Fail band. Values safely over budget — a
        // silent CPU fallback or missed-fusion regression will land
        // here (CPU at 370M is 30–60 s/step).
        for &measured in &[501.0f32, 600.0, 1_000.0, 10_000.0, 60_000.0, f32::MAX] {
            assert_eq!(
                verdict_from_step_time_ms(measured, threshold),
                Gputrain005Verdict::Fail,
                "measured={measured} ms (above {threshold} ms) must Fail",
            );
        }

        // Section 5: non-finite sanity rails. NaN and ±∞ on either
        // side must Fail conservatively — a JSON-parse bug that
        // emitted NaN must not silently Pass a ship gate.
        for &bad in &[f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert_eq!(
                verdict_from_step_time_ms(bad, threshold),
                Gputrain005Verdict::Fail,
                "non-finite measured ({bad:?}) must Fail",
            );
            assert_eq!(
                verdict_from_step_time_ms(100.0, bad),
                Gputrain005Verdict::Fail,
                "non-finite threshold ({bad:?}) must Fail",
            );
        }

        // Section 6: negative / non-positive rails. Step time can't be
        // negative (physical), threshold can't be ≤ 0 (sanity).
        assert_eq!(
            verdict_from_step_time_ms(-1.0, threshold),
            Gputrain005Verdict::Fail,
            "negative measured step time must Fail (non-physical)",
        );
        assert_eq!(
            verdict_from_step_time_ms(-0.0001, threshold),
            Gputrain005Verdict::Fail,
            "tiny-negative measured step time must Fail",
        );
        assert_eq!(
            verdict_from_step_time_ms(100.0, 0.0),
            Gputrain005Verdict::Fail,
            "zero threshold must Fail (ceiling of 0 is nonsense)",
        );
        assert_eq!(
            verdict_from_step_time_ms(100.0, -1.0),
            Gputrain005Verdict::Fail,
            "negative threshold must Fail",
        );

        // Section 7: provenance pin — the 500-ms constant is load-
        // bearing and lockstep with the YAML contract rule text and
        // §14.4 Phase-3 evidence target. If the spec ever loosens the
        // budget (e.g. to 800 ms on a smaller GPU) or tightens it
        // (e.g. to 200 ms on B200), this constant, the YAML rule, and
        // this test must move together.
        assert!(
            (AC_GPUTRAIN_005_MAX_STEP_TIME_MS_RTX4090_370M - 500.0).abs() < 1e-6,
            "INV-GPUTRAIN-005 step-time ceiling is 500.0 ms \
             (spec §14.4 / gpu-training-backend-v1 INV-GPUTRAIN-005)",
        );
    }
}
