// SHIP-TWO-001 AC-SHIP1-007 / FALSIFY-SHIP-007 algorithm-level PARTIAL discharge.
//
// Spec: docs/specifications/aprender-train/ship-two-models-spec.md
// Contract: contracts/apr-cli-commands-v1.yaml (GATE-BENCH-SHIP-007 — to be
// wired in the same PR as this file lands) and the AC row itself at
// `AC-SHIP1-007  | apr bench decode throughput ≥30 tok/s on RTX 4090 (7B Q4_K target)`.
//
// AC-SHIP1-007 states that the MODEL-1 teacher
// (`paiml/qwen2.5-coder-7b-apache-q4k-v1`) must sustain a decode median of
// at least 30 tok/s on an RTX 4090 at the 7B Q4_K quantization. That is a
// ship-blocking performance floor for the teacher artifact — below 30 the
// artifact cannot be declared Ollama-parity-class for 7B Q4_K.
//
// This file discharges the *decision rule* at `PARTIAL_ALGORITHM_LEVEL`:
// given a measured decode tok/s, the verdict is `Pass` iff it is finite
// AND at or above the contract floor (30.0 tok/s). The compute-heavy
// portion of the AC (running `apr bench --iterations 5 --max-tokens 128`
// on live teacher weights on an RTX 4090 host) is intentionally out of
// scope here; the threshold rule is what `apr bench` must emit a Pass on,
// and changing either side of the bind (the 30.0 constant, or the `finite
// AND ≥ floor` shape) breaks this test before any bench run is launched.
//
// Mirrors the MODEL-2 pattern set by SHIP-020 (task #150 on branch
// `feat/falsify-ship-020-partial-discharge`, PR #1005 pending merge).
// SHIP-007 is the MODEL-1 twin: identical f32-threshold verdict shape,
// different floor constant (100.0 → 30.0 tok/s — 7B Q4_K is bandwidth-
// bound at ~3.5× the size of the 370M target). Authored self-contained
// because SHIP-020 is not yet on main; once it lands the two
// `verdict_from_decode_tps_*` fns should be deduplicated into a single
// parameterized helper `verdict_from_decode_tps(measured, floor)`.
//
// MODEL-1 is now at 4/10 AC-SHIP1 items touched (SHIP-008 + SHIP-009 +
// SHIP-006 + SHIP-007).

/// Minimum acceptable median decode throughput, in tok/s, for the MODEL-1
/// teacher (`paiml/qwen2.5-coder-7b-apache-q4k-v1`) when measured by
/// `apr bench --iterations 5 --max-tokens 128` on an RTX 4090 host.
///
/// Derivation: spec AC-SHIP1-007 binds the 30 tok/s floor to the 7B Q4_K
/// ship criterion. The constant is pinned here so that contract drift in
/// either direction (weakening to 25, hardening to 35 without updating
/// AC) is caught at compile+test time, not at a production publish.
/// Lockstep with `docs/specifications/aprender-train/ship-two-models-spec.md`
/// §4.2 row AC-SHIP1-007.
pub const AC_SHIP1_007_MIN_DECODE_TPS_RTX4090_7B: f32 = 30.0;

/// Binary verdict for FALSIFY-SHIP-007 / GATE-BENCH-SHIP-007.
/// `Pass` iff the measured decode throughput is finite AND at or above
/// [`AC_SHIP1_007_MIN_DECODE_TPS_RTX4090_7B`]. `Fail` otherwise (including
/// every non-finite value: NaN, +∞, -∞).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ship007Verdict {
    /// Measured tok/s ≥ [`AC_SHIP1_007_MIN_DECODE_TPS_RTX4090_7B`].
    Pass,
    /// Measured tok/s < [`AC_SHIP1_007_MIN_DECODE_TPS_RTX4090_7B`] or non-finite (ship-blocker).
    Fail,
}

/// Algorithm-level verdict rule for FALSIFY-SHIP-007 / GATE-BENCH-SHIP-007
/// / AC-SHIP1-007: a single f32 threshold check against the MODEL-1 7B
/// Q4_K decode floor. Returns [`Ship007Verdict::Fail`] conservatively for
/// NaN, +∞, and -∞ so that a telemetry or JSON-parse bug can never be
/// silently promoted to a Pass. The full discharge (live `apr bench
/// --iterations 5 --max-tokens 128 paiml/qwen2.5-coder-7b-apache-q4k-v1`
/// on RTX 4090 with median ≥ 30.0) remains blocked on hardware evidence
/// collection.
#[must_use]
pub fn verdict_from_decode_tps(measured_tps: f32) -> Ship007Verdict {
    if !measured_tps.is_finite() {
        return Ship007Verdict::Fail;
    }
    if measured_tps >= AC_SHIP1_007_MIN_DECODE_TPS_RTX4090_7B {
        Ship007Verdict::Pass
    } else {
        Ship007Verdict::Fail
    }
}

// ─────────────────────────────────────────────────────────────
// Unit tests — FALSIFY-SHIP-007 algorithm-level proof
// ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod ship_007_tests {
    use super::*;

    /// FALSIFY-SHIP-007 algorithm-level PARTIAL discharge: prove the
    /// f32-threshold decision rule binding the measured decode tok/s
    /// to AC_SHIP1_007_MIN_DECODE_TPS_RTX4090_7B = 30.0. Any edit that
    /// changes the constant, the comparison direction, the non-finite
    /// handling, or the monotonicity must break this test before a
    /// teacher `apr bench` gate runs.
    #[test]
    fn falsify_ship_007_decode_tps_threshold_logic() {
        // Section 1: boundary — exactly 30.0 is a Pass (contract says
        // "≥ 30", not "> 30"). A regression that silently swaps to
        // strict-greater would make the floor unreachable.
        assert_eq!(
            verdict_from_decode_tps(30.0),
            Ship007Verdict::Pass,
            "boundary 30.0 tok/s must Pass (contract is ≥, not >)",
        );

        // Section 2: just-below — the next representable f32 below 30.0
        // must Fail. This is the sharpest possible counter-example to
        // any off-by-one ULP drift.
        let just_below = f32::from_bits(30.0_f32.to_bits() - 1);
        assert!(just_below < 30.0, "just_below must be strictly < 30.0");
        assert_eq!(
            verdict_from_decode_tps(just_below),
            Ship007Verdict::Fail,
            "one ULP below 30.0 must Fail",
        );

        // Section 3: clear Pass above the floor. Mirrors the healthy
        // operating range (teacher on RTX 4090 is expected in the
        // 40-60 tok/s band for 7B Q4_K).
        assert_eq!(verdict_from_decode_tps(45.0), Ship007Verdict::Pass);
        assert_eq!(verdict_from_decode_tps(100.0), Ship007Verdict::Pass);

        // Section 4: clear Fail below the floor. Catches the class of
        // regression where a kernel change drops decode into the
        // sub-30 band (CB-510, PMAT-592 histories).
        assert_eq!(verdict_from_decode_tps(0.0), Ship007Verdict::Fail);
        assert_eq!(verdict_from_decode_tps(10.0), Ship007Verdict::Fail);
        assert_eq!(verdict_from_decode_tps(29.999_999), Ship007Verdict::Fail);

        // Section 5: monotonicity — above the floor, no value ever
        // flips back to Fail; below the floor, no value ever flips
        // back to Pass. Sampled discretely because f32 is finite.
        for t in [30.0_f32, 30.5, 31.0, 50.0, 200.0, 1_000.0, 10_000.0] {
            assert_eq!(
                verdict_from_decode_tps(t),
                Ship007Verdict::Pass,
                "monotonicity broken above floor at {t}",
            );
        }
        for t in [-1_000.0_f32, -1.0, 0.0, 10.0, 20.0, 29.0, 29.999] {
            assert_eq!(
                verdict_from_decode_tps(t),
                Ship007Verdict::Fail,
                "monotonicity broken below floor at {t}",
            );
        }

        // Section 6: non-finite inputs must Fail conservatively. A
        // telemetry or JSON-parse bug that produces NaN/±∞ must NEVER
        // be promoted to Pass — that's how bad perf results ship.
        assert_eq!(
            verdict_from_decode_tps(f32::NAN),
            Ship007Verdict::Fail,
            "NaN must Fail conservatively",
        );
        assert_eq!(
            verdict_from_decode_tps(f32::INFINITY),
            Ship007Verdict::Fail,
            "+∞ must Fail conservatively",
        );
        assert_eq!(
            verdict_from_decode_tps(f32::NEG_INFINITY),
            Ship007Verdict::Fail,
            "-∞ must Fail conservatively",
        );

        // Section 7: provenance pin — the floor is load-bearing and
        // lockstepped with the spec. If AC-SHIP1-007 ever changes
        // the floor (raising for a newer RTX generation, relaxing
        // because the 7B-Q4_K class is retired), this constant and
        // this test must move together.
        #[allow(clippy::float_cmp)]
        {
            assert_eq!(
                AC_SHIP1_007_MIN_DECODE_TPS_RTX4090_7B, 30.0,
                "MODEL-1 ship criterion floor is 30.0 tok/s \
                 (spec §4.2 AC-SHIP1-007; 7B Q4_K on RTX 4090)",
            );
        }
    }
}
