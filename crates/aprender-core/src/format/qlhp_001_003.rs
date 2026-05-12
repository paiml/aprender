// SHIP-TWO-001 — `qlora-hyperparameters-v1` algorithm-level PARTIAL
// discharge for FALSIFY-QLHP-001..003 (closes 3/3 sweep).
//
// Contract: `contracts/qlora-hyperparameters-v1.yaml`.
// Spec: QLoRA fine-tuning hyperparameter bounds — rank ∈ [4, 256],
// alpha/rank ∈ [0.5, 4.0], lr ∈ [1e-6, 1e-3].

// ===========================================================================
// QLHP-001 — Rank bounds: 4 ≤ rank ≤ 256, power-of-2 preferred
// ===========================================================================

pub const AC_QLHP_001_RANK_MIN: u64 = 4;
pub const AC_QLHP_001_RANK_MAX: u64 = 256;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qlhp001Verdict { Pass, Fail }

#[must_use]
pub const fn is_power_of_two(n: u64) -> bool {
    n.is_power_of_two()
}

/// Pass iff rank ∈ [4, 256] AND rank is a power of 2 (the contract says
/// "power of 2 preferred" — the verdict treats this as REQUIRED for
/// algorithm-level discharge to surface the regression class).
#[must_use]
pub const fn verdict_from_rank_bounds(rank: u64) -> Qlhp001Verdict {
    if rank < AC_QLHP_001_RANK_MIN { return Qlhp001Verdict::Fail; }
    if rank > AC_QLHP_001_RANK_MAX { return Qlhp001Verdict::Fail; }
    if !is_power_of_two(rank) { return Qlhp001Verdict::Fail; }
    Qlhp001Verdict::Pass
}

// ===========================================================================
// QLHP-002 — alpha/rank ∈ [0.5, 4.0] for stable gradients
// ===========================================================================

pub const AC_QLHP_002_RATIO_MIN: f32 = 0.5;
pub const AC_QLHP_002_RATIO_MAX: f32 = 4.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qlhp002Verdict { Pass, Fail }

/// Pass iff `alpha/rank ∈ [0.5, 4.0]` AND both alpha, rank > 0 finite.
#[must_use]
pub fn verdict_from_alpha_rank_ratio(alpha: f32, rank: u64) -> Qlhp002Verdict {
    if rank == 0 { return Qlhp002Verdict::Fail; }
    if !alpha.is_finite() || alpha <= 0.0 { return Qlhp002Verdict::Fail; }
    let ratio = alpha / (rank as f32);
    if !ratio.is_finite() { return Qlhp002Verdict::Fail; }
    if ratio < AC_QLHP_002_RATIO_MIN || ratio > AC_QLHP_002_RATIO_MAX {
        return Qlhp002Verdict::Fail;
    }
    Qlhp002Verdict::Pass
}

// ===========================================================================
// QLHP-003 — Learning rate ∈ [1e-6, 1e-3]
// ===========================================================================

pub const AC_QLHP_003_LR_MIN: f32 = 1.0e-6;
pub const AC_QLHP_003_LR_MAX: f32 = 1.0e-3;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qlhp003Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_learning_rate(lr: f32) -> Qlhp003Verdict {
    if !lr.is_finite() { return Qlhp003Verdict::Fail; }
    if lr < AC_QLHP_003_LR_MIN || lr > AC_QLHP_003_LR_MAX { return Qlhp003Verdict::Fail; }
    Qlhp003Verdict::Pass
}

#[cfg(test)]
mod tests {
    use super::*;

    // QLHP-001 (rank bounds + power of 2)
    #[test] fn qlhp001_pass_canonical_8() {
        assert_eq!(verdict_from_rank_bounds(8), Qlhp001Verdict::Pass);
    }
    #[test] fn qlhp001_pass_canonical_16() {
        assert_eq!(verdict_from_rank_bounds(16), Qlhp001Verdict::Pass);
    }
    #[test] fn qlhp001_pass_at_min_4() {
        assert_eq!(verdict_from_rank_bounds(4), Qlhp001Verdict::Pass);
    }
    #[test] fn qlhp001_pass_at_max_256() {
        assert_eq!(verdict_from_rank_bounds(256), Qlhp001Verdict::Pass);
    }
    #[test] fn qlhp001_fail_below_min() {
        assert_eq!(verdict_from_rank_bounds(2), Qlhp001Verdict::Fail);
        assert_eq!(verdict_from_rank_bounds(3), Qlhp001Verdict::Fail);
    }
    #[test] fn qlhp001_fail_above_max() {
        assert_eq!(verdict_from_rank_bounds(512), Qlhp001Verdict::Fail);
    }
    #[test] fn qlhp001_fail_not_power_of_2() {
        // 6, 10, 12 are in [4, 256] but not powers of 2.
        assert_eq!(verdict_from_rank_bounds(6), Qlhp001Verdict::Fail);
        assert_eq!(verdict_from_rank_bounds(10), Qlhp001Verdict::Fail);
        assert_eq!(verdict_from_rank_bounds(100), Qlhp001Verdict::Fail);
    }
    #[test] fn qlhp001_fail_zero() {
        assert_eq!(verdict_from_rank_bounds(0), Qlhp001Verdict::Fail);
    }
    #[test] fn pow2_helper_sanity() {
        assert!(is_power_of_two(1));
        assert!(is_power_of_two(2));
        assert!(is_power_of_two(8));
        assert!(is_power_of_two(256));
        assert!(!is_power_of_two(0));
        assert!(!is_power_of_two(3));
        assert!(!is_power_of_two(255));
    }

    // QLHP-002 (alpha/rank ratio)
    #[test] fn qlhp002_pass_canonical_2x() {
        // Common QLoRA: alpha=16, rank=8 → ratio=2.0 (within [0.5, 4.0]).
        assert_eq!(verdict_from_alpha_rank_ratio(16.0, 8), Qlhp002Verdict::Pass);
    }
    #[test] fn qlhp002_pass_at_lower() {
        // ratio = 0.5 (boundary).
        assert_eq!(verdict_from_alpha_rank_ratio(4.0, 8), Qlhp002Verdict::Pass);
    }
    #[test] fn qlhp002_pass_at_upper() {
        // ratio = 4.0 (boundary).
        assert_eq!(verdict_from_alpha_rank_ratio(32.0, 8), Qlhp002Verdict::Pass);
    }
    #[test] fn qlhp002_fail_too_small() {
        // ratio = 0.25 — below 0.5.
        assert_eq!(verdict_from_alpha_rank_ratio(2.0, 8), Qlhp002Verdict::Fail);
    }
    #[test] fn qlhp002_fail_too_large() {
        // ratio = 8.0 — above 4.0.
        assert_eq!(verdict_from_alpha_rank_ratio(64.0, 8), Qlhp002Verdict::Fail);
    }
    #[test] fn qlhp002_fail_zero_rank() {
        assert_eq!(verdict_from_alpha_rank_ratio(8.0, 0), Qlhp002Verdict::Fail);
    }
    #[test] fn qlhp002_fail_zero_alpha() {
        assert_eq!(verdict_from_alpha_rank_ratio(0.0, 8), Qlhp002Verdict::Fail);
    }
    #[test] fn qlhp002_fail_negative_alpha() {
        assert_eq!(verdict_from_alpha_rank_ratio(-8.0, 8), Qlhp002Verdict::Fail);
    }
    #[test] fn qlhp002_fail_nan() {
        assert_eq!(verdict_from_alpha_rank_ratio(f32::NAN, 8), Qlhp002Verdict::Fail);
    }

    // QLHP-003 (learning rate range)
    #[test] fn qlhp003_pass_canonical() {
        // Common QLoRA lr: 2e-4.
        assert_eq!(verdict_from_learning_rate(2e-4), Qlhp003Verdict::Pass);
    }
    #[test] fn qlhp003_pass_at_lower() {
        assert_eq!(verdict_from_learning_rate(1e-6), Qlhp003Verdict::Pass);
    }
    #[test] fn qlhp003_pass_at_upper() {
        assert_eq!(verdict_from_learning_rate(1e-3), Qlhp003Verdict::Pass);
    }
    #[test] fn qlhp003_fail_too_small() {
        // 1e-7 is below the minimum (training would stall).
        assert_eq!(verdict_from_learning_rate(1e-7), Qlhp003Verdict::Fail);
    }
    #[test] fn qlhp003_fail_too_large() {
        // 1e-2 is above the maximum (training would diverge).
        assert_eq!(verdict_from_learning_rate(1e-2), Qlhp003Verdict::Fail);
    }
    #[test] fn qlhp003_fail_negative() {
        assert_eq!(verdict_from_learning_rate(-1e-4), Qlhp003Verdict::Fail);
    }
    #[test] fn qlhp003_fail_zero() {
        assert_eq!(verdict_from_learning_rate(0.0), Qlhp003Verdict::Fail);
    }
    #[test] fn qlhp003_fail_nan() {
        assert_eq!(verdict_from_learning_rate(f32::NAN), Qlhp003Verdict::Fail);
    }
    #[test] fn qlhp003_fail_inf() {
        assert_eq!(verdict_from_learning_rate(f32::INFINITY), Qlhp003Verdict::Fail);
    }

    // Provenance
    #[test] fn provenance_constants() {
        assert_eq!(AC_QLHP_001_RANK_MIN, 4);
        assert_eq!(AC_QLHP_001_RANK_MAX, 256);
        assert!((AC_QLHP_002_RATIO_MIN - 0.5).abs() < 1e-9);
        assert!((AC_QLHP_002_RATIO_MAX - 4.0).abs() < 1e-9);
        assert!((AC_QLHP_003_LR_MIN - 1e-6).abs() < 1e-12);
        assert!((AC_QLHP_003_LR_MAX - 1e-3).abs() < 1e-9);
    }
}
