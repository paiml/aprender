// SHIP-TWO-001 — `apr-format-invariants-v1` algorithm-level PARTIAL
// discharge for FALSIFY-APR-001..003.
//
// Contract: `contracts/apr-format-invariants-v1.yaml`.
// Spec: `docs/specifications/aprender-train/ship-two-models-spec.md`.
//
// ## What this file proves NOW (PARTIAL_ALGORITHM_LEVEL)
//
// Three APR ModelEvidence-format gates:
//
// - APR-001 (roundtrip identity): deserialize(serialize(e)) == e.
// - APR-002 (truncated input rejection): truncated bytes ⇒ ValidationError,
//   never panic.
// - APR-003 (regression detection): identical baselines ⇒ zero regressions
//   under FP tolerance.

/// FP comparison tolerance for regression detection.
pub const AC_APR_003_REGRESSION_EPS: f32 = 1e-6;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AprFiVerdict {
    Pass,
    Fail,
}

// -----------------------------------------------------------------------------
// In-module reference: bytewise-roundtrip helper.
// -----------------------------------------------------------------------------

/// Trivial round-trip via clone — stand-in for serialize+deserialize
/// over arbitrary `Vec<u8>` payload.
#[must_use]
pub fn roundtrip(bytes: &[u8]) -> Vec<u8> {
    bytes.to_vec()
}

// -----------------------------------------------------------------------------
// Verdict 1: APR-001 — roundtrip identity.
// -----------------------------------------------------------------------------

/// Pass iff `serialized_input == round_tripped_output` (byte-equal).
#[must_use]
pub fn verdict_from_roundtrip_identity(input: &[u8], output: &[u8]) -> AprFiVerdict {
    if input == output {
        AprFiVerdict::Pass
    } else {
        AprFiVerdict::Fail
    }
}

// -----------------------------------------------------------------------------
// Verdict 2: APR-002 — truncated input rejection.
// -----------------------------------------------------------------------------

/// Models the runtime: `did_panic` is true iff the deserializer panicked.
/// `returned_validation_error` is true iff a ValidationError was raised.
///
/// Pass iff `did_panic == false` AND (`is_truncated` ⇒ `returned_validation_error`).
#[must_use]
pub fn verdict_from_truncated_rejection(
    is_truncated: bool,
    did_panic: bool,
    returned_validation_error: bool,
) -> AprFiVerdict {
    if did_panic {
        return AprFiVerdict::Fail;
    }
    if is_truncated && !returned_validation_error {
        return AprFiVerdict::Fail;
    }
    AprFiVerdict::Pass
}

// -----------------------------------------------------------------------------
// Verdict 3: APR-003 — regression detection on identical baselines.
// -----------------------------------------------------------------------------

/// Compare two metric arrays elementwise within FP tolerance. Returns
/// the count of "regressions" (entries where `current > baseline + eps`).
///
/// Pass iff:
///   - identical inputs (same length and element pairs `≤ eps`) ⇒ count = 0,
///   - the verdict is asked to confirm `count == 0`.
#[must_use]
pub fn count_regressions(baseline: &[f32], current: &[f32]) -> Option<usize> {
    if baseline.len() != current.len() {
        return None;
    }
    let mut count = 0_usize;
    for (b, c) in baseline.iter().zip(current.iter()) {
        if !b.is_finite() || !c.is_finite() {
            return None;
        }
        if (c - b) > AC_APR_003_REGRESSION_EPS {
            count += 1;
        }
    }
    Some(count)
}

/// Pass iff identical baselines produce zero regressions.
#[must_use]
pub fn verdict_from_regression_detection(
    baseline: &[f32],
    current: &[f32],
    expect_zero: bool,
) -> AprFiVerdict {
    match count_regressions(baseline, current) {
        Some(count) => {
            if expect_zero == (count == 0) {
                AprFiVerdict::Pass
            } else {
                AprFiVerdict::Fail
            }
        }
        None => AprFiVerdict::Fail,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Section 1: Provenance pin.
    // -------------------------------------------------------------------------
    #[test]
    fn provenance_regression_eps_1e_6() {
        assert_eq!(AC_APR_003_REGRESSION_EPS, 1e-6);
    }

    // -------------------------------------------------------------------------
    // Section 2: APR-001 — roundtrip identity.
    // -------------------------------------------------------------------------
    #[test]
    fn apr001_pass_byte_identical() {
        let input = b"\x00\xff\xab\xcd\x12\x34";
        let output = roundtrip(input);
        assert_eq!(
            verdict_from_roundtrip_identity(input, &output),
            AprFiVerdict::Pass
        );
    }

    #[test]
    fn apr001_pass_empty() {
        let input: &[u8] = &[];
        let output = roundtrip(input);
        assert_eq!(
            verdict_from_roundtrip_identity(input, &output),
            AprFiVerdict::Pass
        );
    }

    #[test]
    fn apr001_pass_large() {
        let input: Vec<u8> = (0..10_000).map(|i| (i % 256) as u8).collect();
        let output = roundtrip(&input);
        assert_eq!(
            verdict_from_roundtrip_identity(&input, &output),
            AprFiVerdict::Pass
        );
    }

    #[test]
    fn apr001_fail_one_byte_corrupted() {
        let input = b"\x01\x02\x03\x04";
        let mut output = input.to_vec();
        output[2] = 0xFF; // corruption
        assert_eq!(
            verdict_from_roundtrip_identity(input, &output),
            AprFiVerdict::Fail
        );
    }

    #[test]
    fn apr001_fail_length_mismatch() {
        let input = b"\x01\x02\x03";
        let output = b"\x01\x02";
        assert_eq!(
            verdict_from_roundtrip_identity(input, output),
            AprFiVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 3: APR-002 — truncated input rejection.
    // -------------------------------------------------------------------------
    #[test]
    fn apr002_pass_truncated_with_validation_error() {
        // Truncated bytes, no panic, ValidationError raised.
        assert_eq!(
            verdict_from_truncated_rejection(true, false, true),
            AprFiVerdict::Pass
        );
    }

    #[test]
    fn apr002_pass_full_input_no_error() {
        // Full input, no panic, no validation error needed.
        assert_eq!(
            verdict_from_truncated_rejection(false, false, false),
            AprFiVerdict::Pass
        );
    }

    #[test]
    fn apr002_fail_truncated_silent_pass() {
        // Bug: truncated input loaded silently (e.g., garbage tensors).
        assert_eq!(
            verdict_from_truncated_rejection(true, false, false),
            AprFiVerdict::Fail
        );
    }

    #[test]
    fn apr002_fail_panic_on_truncated() {
        // The contract failure: missing bounds check ⇒ panic.
        assert_eq!(
            verdict_from_truncated_rejection(true, true, false),
            AprFiVerdict::Fail
        );
    }

    #[test]
    fn apr002_fail_panic_on_full() {
        // Even full input shouldn't panic.
        assert_eq!(
            verdict_from_truncated_rejection(false, true, false),
            AprFiVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 4: APR-003 — regression detection.
    // -------------------------------------------------------------------------
    #[test]
    fn apr003_pass_identical_baselines_zero_regressions() {
        let b = vec![1.0_f32, 2.0, 3.0];
        let c = vec![1.0_f32, 2.0, 3.0];
        assert_eq!(
            verdict_from_regression_detection(&b, &c, true),
            AprFiVerdict::Pass
        );
    }

    #[test]
    fn apr003_pass_within_tolerance() {
        let b = vec![1.0_f32];
        let c = vec![1.000_000_5_f32]; // 5e-7 < 1e-6
        assert_eq!(
            verdict_from_regression_detection(&b, &c, true),
            AprFiVerdict::Pass
        );
    }

    #[test]
    fn apr003_pass_improvements_dont_count_as_regression() {
        // c < b ⇒ improvement (not a regression).
        let b = vec![10.0_f32, 5.0];
        let c = vec![9.0_f32, 4.0];
        assert_eq!(
            verdict_from_regression_detection(&b, &c, true),
            AprFiVerdict::Pass
        );
    }

    #[test]
    fn apr003_pass_real_regression_when_expected() {
        // c > b ⇒ regression count > 0; verdict is asked for >0 case.
        let b = vec![1.0_f32, 2.0];
        let c = vec![1.5_f32, 3.0]; // 2 regressions
        assert_eq!(
            verdict_from_regression_detection(&b, &c, false),
            AprFiVerdict::Pass
        );
    }

    #[test]
    fn apr003_fail_silent_regression() {
        // Bug: identical baselines but FP comparison without tolerance
        // produced a false regression count.
        let b = vec![1.0_f32];
        let c = vec![1.0001_f32]; // delta = 1e-4 > 1e-6
        assert_eq!(
            verdict_from_regression_detection(&b, &c, true),
            AprFiVerdict::Fail
        );
    }

    #[test]
    fn apr003_fail_zero_regression_when_expected_some() {
        let b = vec![1.0_f32];
        let c = vec![1.0_f32];
        assert_eq!(
            verdict_from_regression_detection(&b, &c, false),
            AprFiVerdict::Fail
        );
    }

    #[test]
    fn apr003_fail_length_mismatch() {
        let b = vec![1.0_f32, 2.0];
        let c = vec![1.0_f32];
        assert_eq!(
            verdict_from_regression_detection(&b, &c, true),
            AprFiVerdict::Fail
        );
    }

    #[test]
    fn apr003_fail_nan() {
        let b = vec![1.0_f32];
        let c = vec![f32::NAN];
        assert_eq!(
            verdict_from_regression_detection(&b, &c, true),
            AprFiVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 5: Domain — count_regressions reference.
    // -------------------------------------------------------------------------
    #[test]
    fn domain_count_regressions_basic() {
        let b = vec![1.0_f32, 2.0, 3.0];
        let c = vec![1.5_f32, 2.0, 4.0];
        assert_eq!(count_regressions(&b, &c), Some(2));
    }

    #[test]
    fn domain_count_regressions_zero_for_identical() {
        let b = vec![1.0_f32, 2.0];
        let c = vec![1.0_f32, 2.0];
        assert_eq!(count_regressions(&b, &c), Some(0));
    }

    #[test]
    fn domain_count_regressions_none_on_length_mismatch() {
        let b = vec![1.0_f32];
        let c = vec![1.0_f32, 2.0];
        assert_eq!(count_regressions(&b, &c), None);
    }

    #[test]
    fn domain_count_regressions_none_on_nan() {
        let b = vec![1.0_f32, f32::NAN];
        let c = vec![1.0_f32, 1.0];
        assert_eq!(count_regressions(&b, &c), None);
    }

    // -------------------------------------------------------------------------
    // Section 6: Sweep — regression bands.
    // -------------------------------------------------------------------------
    #[test]
    fn sweep_regression_around_eps() {
        let b = vec![1.0_f32];
        let test_cases: Vec<(f32, AprFiVerdict)> = vec![
            (1.0, AprFiVerdict::Pass),
            (1.0000001, AprFiVerdict::Pass), // 1e-7 < 1e-6
            (1.0001, AprFiVerdict::Fail),    // 1e-4 > 1e-6
            (1.5, AprFiVerdict::Fail),
            (0.5, AprFiVerdict::Pass), // improvement
        ];
        for (val, expected) in test_cases {
            let c = vec![val];
            let v = verdict_from_regression_detection(&b, &c, true);
            assert_eq!(v, expected, "val={val}");
        }
    }

    // -------------------------------------------------------------------------
    // Section 7: Realistic — contract regression scenarios.
    // -------------------------------------------------------------------------
    #[test]
    fn realistic_serialize_deserialize_loses_field() {
        // APR-001 if_fails: "Serialization loses precision or field" —
        // simulated as one byte dropped at the end.
        let input = b"\x01\x02\x03\x04";
        let truncated = b"\x01\x02\x03";
        assert_eq!(
            verdict_from_roundtrip_identity(input, truncated),
            AprFiVerdict::Fail
        );
    }

    #[test]
    fn realistic_missing_bounds_check_panic_caught() {
        // APR-002 if_fails: "Missing bounds check in deserializer".
        assert_eq!(
            verdict_from_truncated_rejection(true, true, false),
            AprFiVerdict::Fail
        );
    }

    #[test]
    fn realistic_floating_point_no_tolerance_caught() {
        // APR-003 if_fails: "Floating point comparison not using
        // tolerance" — identical-up-to-FP-noise baselines flagged as
        // regression.
        let b = vec![1.234_567_89_f32];
        let c = vec![1.234_568_5_f32]; // ~6e-7 drift; should not regress
                                       // But the bug is that c gets reported as regressed — verdict
                                       // expects zero regressions.
                                       // Our reference uses 1e-6 tolerance, so this case actually
                                       // passes. To model the bug, simulate a 1e-3 drift instead:
        let buggy_c = vec![1.234_5_f32, 1.235_5_f32];
        let buggy_b = vec![1.234_5_f32, 1.234_5_f32];
        assert_eq!(
            verdict_from_regression_detection(&buggy_b, &buggy_c, true),
            AprFiVerdict::Fail
        );
        // Sanity: the within-tolerance case passes.
        assert_eq!(
            verdict_from_regression_detection(&b, &c, true),
            AprFiVerdict::Pass
        );
    }

    #[test]
    fn realistic_full_roundtrip_pipeline_passes_all_3_gates() {
        // Synthesize an APR ModelEvidence serialize → deserialize:
        let evidence_bytes = vec![0xAB, 0xCD, 0xEF, 0x01, 0x23, 0x45];
        let round_tripped = roundtrip(&evidence_bytes);
        assert_eq!(
            verdict_from_roundtrip_identity(&evidence_bytes, &round_tripped),
            AprFiVerdict::Pass
        );
        // Truncated rejection: simulated as truncated input handled
        // correctly (no panic, ValidationError raised).
        assert_eq!(
            verdict_from_truncated_rejection(true, false, true),
            AprFiVerdict::Pass
        );
        // Regression detection on identical baselines.
        let baseline = vec![1.5_f32, 2.5, 3.5];
        let current = vec![1.5_f32, 2.5, 3.5];
        assert_eq!(
            verdict_from_regression_detection(&baseline, &current, true),
            AprFiVerdict::Pass
        );
    }
}
