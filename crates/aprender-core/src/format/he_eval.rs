// SHIP-TWO-001 — `eval-harness-humaneval-v1` algorithm-level
// PARTIAL discharge for FALSIFY-HE-001..006 (closes 6/6).
//
// Contract: `contracts/eval-harness-humaneval-v1.yaml`.
// Spec: AC-SHIP1-005 (MODEL-1 student ≥ 86.0% HumanEval pass@1).

// ===========================================================================
// HE-001 — teacher baseline reproduces in [84.5, 87.0]
// ===========================================================================

pub const AC_HE_001_TEACHER_LOW: f64 = 84.5;
pub const AC_HE_001_TEACHER_HIGH: f64 = 87.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum He001Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_teacher_baseline(pass_at_1: f64) -> He001Verdict {
    if !pass_at_1.is_finite() { return He001Verdict::Fail; }
    if (AC_HE_001_TEACHER_LOW..=AC_HE_001_TEACHER_HIGH).contains(&pass_at_1) {
        He001Verdict::Pass
    } else {
        He001Verdict::Fail
    }
}

// ===========================================================================
// HE-002 — primary student meets ship threshold (≥86.0)
// ===========================================================================

pub const AC_HE_002_SHIP_THRESHOLD: f64 = 86.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum He002Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_student_ship_threshold(pass_at_1: f64) -> He002Verdict {
    if !pass_at_1.is_finite() { return He002Verdict::Fail; }
    if !(0.0..=100.0).contains(&pass_at_1) { return He002Verdict::Fail; }
    if pass_at_1 >= AC_HE_002_SHIP_THRESHOLD {
        He002Verdict::Pass
    } else {
        He002Verdict::Fail
    }
}

// ===========================================================================
// HE-003 — native APR vs GGUF parity (modulo tokenizer rounding)
// ===========================================================================

pub const AC_HE_003_TOLERANCE: f64 = 0.5;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum He003Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_apr_gguf_parity(apr_pass: f64, gguf_pass: f64) -> He003Verdict {
    if !apr_pass.is_finite() || !gguf_pass.is_finite() { return He003Verdict::Fail; }
    if (apr_pass - gguf_pass).abs() <= AC_HE_003_TOLERANCE {
        He003Verdict::Pass
    } else {
        He003Verdict::Fail
    }
}

// ===========================================================================
// HE-004 — merged is upper bound (merged ≥ q4k)
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum He004Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_merged_upper_bound(merged_pass: f64, q4k_pass: f64) -> He004Verdict {
    if !merged_pass.is_finite() || !q4k_pass.is_finite() { return He004Verdict::Fail; }
    if merged_pass >= q4k_pass {
        He004Verdict::Pass
    } else {
        He004Verdict::Fail
    }
}

// ===========================================================================
// HE-005 — held-out problem regression check (≥ 2 fixed problems still pass)
// ===========================================================================

pub const AC_HE_005_MIN_FIXED_PROBLEMS: u64 = 2;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum He005Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_held_out_regression(fixed_still_passing: u64) -> He005Verdict {
    if fixed_still_passing >= AC_HE_005_MIN_FIXED_PROBLEMS {
        He005Verdict::Pass
    } else {
        He005Verdict::Fail
    }
}

// ===========================================================================
// HE-006 — sampling determinism (T=0 byte-identical)
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum He006Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_sampling_determinism(run_a: &[u8], run_b: &[u8]) -> He006Verdict {
    if run_a.is_empty() || run_b.is_empty() { return He006Verdict::Fail; }
    if run_a == run_b { He006Verdict::Pass } else { He006Verdict::Fail }
}

#[cfg(test)]
mod tests {
    use super::*;

    // HE-001
    #[test] fn h001_pass_at_85_98() { assert_eq!(verdict_from_teacher_baseline(85.98), He001Verdict::Pass); }
    #[test] fn h001_pass_at_low_band() { assert_eq!(verdict_from_teacher_baseline(84.5), He001Verdict::Pass); }
    #[test] fn h001_pass_at_high_band() { assert_eq!(verdict_from_teacher_baseline(87.0), He001Verdict::Pass); }
    #[test] fn h001_fail_below_band() { assert_eq!(verdict_from_teacher_baseline(84.0), He001Verdict::Fail); }
    #[test] fn h001_fail_above_band() { assert_eq!(verdict_from_teacher_baseline(87.5), He001Verdict::Fail); }
    #[test] fn h001_fail_nan() { assert_eq!(verdict_from_teacher_baseline(f64::NAN), He001Verdict::Fail); }

    // HE-002
    #[test] fn h002_pass_at_86() { assert_eq!(verdict_from_student_ship_threshold(86.0), He002Verdict::Pass); }
    #[test] fn h002_pass_at_87() { assert_eq!(verdict_from_student_ship_threshold(87.20), He002Verdict::Pass); }
    #[test] fn h002_fail_85_99() { assert_eq!(verdict_from_student_ship_threshold(85.99), He002Verdict::Fail); }
    #[test] fn h002_fail_above_100() { assert_eq!(verdict_from_student_ship_threshold(100.001), He002Verdict::Fail); }
    #[test] fn h002_fail_negative() { assert_eq!(verdict_from_student_ship_threshold(-1.0), He002Verdict::Fail); }
    #[test] fn h002_fail_nan() { assert_eq!(verdict_from_student_ship_threshold(f64::NAN), He002Verdict::Fail); }

    // HE-003
    #[test] fn h003_pass_exact_match() { assert_eq!(verdict_from_apr_gguf_parity(86.0, 86.0), He003Verdict::Pass); }
    #[test] fn h003_pass_within_tolerance() { assert_eq!(verdict_from_apr_gguf_parity(86.0, 86.4), He003Verdict::Pass); }
    #[test] fn h003_pass_at_exact_tolerance() { assert_eq!(verdict_from_apr_gguf_parity(86.0, 86.5), He003Verdict::Pass); }
    #[test] fn h003_fail_above_tolerance() { assert_eq!(verdict_from_apr_gguf_parity(86.0, 86.6), He003Verdict::Fail); }
    #[test] fn h003_fail_nan() { assert_eq!(verdict_from_apr_gguf_parity(f64::NAN, 86.0), He003Verdict::Fail); }

    // HE-004
    #[test] fn h004_pass_merged_higher() { assert_eq!(verdict_from_merged_upper_bound(87.0, 86.0), He004Verdict::Pass); }
    #[test] fn h004_pass_merged_equal() { assert_eq!(verdict_from_merged_upper_bound(86.0, 86.0), He004Verdict::Pass); }
    #[test] fn h004_fail_merged_lower() { assert_eq!(verdict_from_merged_upper_bound(85.0, 86.0), He004Verdict::Fail); }
    #[test] fn h004_fail_nan() { assert_eq!(verdict_from_merged_upper_bound(f64::NAN, 86.0), He004Verdict::Fail); }

    // HE-005
    #[test] fn h005_pass_at_2() { assert_eq!(verdict_from_held_out_regression(2), He005Verdict::Pass); }
    #[test] fn h005_pass_at_5() { assert_eq!(verdict_from_held_out_regression(5), He005Verdict::Pass); }
    #[test] fn h005_fail_at_1() { assert_eq!(verdict_from_held_out_regression(1), He005Verdict::Fail); }
    #[test] fn h005_fail_at_zero() { assert_eq!(verdict_from_held_out_regression(0), He005Verdict::Fail); }

    // HE-006
    #[test] fn h006_pass_byte_identical() { assert_eq!(verdict_from_sampling_determinism(b"abc", b"abc"), He006Verdict::Pass); }
    #[test] fn h006_fail_drift() { assert_eq!(verdict_from_sampling_determinism(b"abc", b"abd"), He006Verdict::Fail); }
    #[test] fn h006_fail_empty() { assert_eq!(verdict_from_sampling_determinism(&[], &[]), He006Verdict::Fail); }

    // Provenance pins
    #[test] fn provenance_he_001_band() { assert_eq!(AC_HE_001_TEACHER_LOW, 84.5); assert_eq!(AC_HE_001_TEACHER_HIGH, 87.0); }
    #[test] fn provenance_he_002_threshold() { assert_eq!(AC_HE_002_SHIP_THRESHOLD, 86.0); }
    #[test] fn provenance_he_003_tolerance() { assert_eq!(AC_HE_003_TOLERANCE, 0.5); }
    #[test] fn provenance_he_005_min_fixed() { assert_eq!(AC_HE_005_MIN_FIXED_PROBLEMS, 2); }
}
