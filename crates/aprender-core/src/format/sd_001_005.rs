// SHIP-TWO-001 — `speculative-decoding-v1` algorithm-level PARTIAL
// discharge for FALSIFY-SD-001..005.
//
// Contract: `contracts/speculative-decoding-v1.yaml`.
// Spec: `docs/specifications/aprender-train/ship-two-models-spec.md`.
//
// ## What this file proves NOW (PARTIAL_ALGORITHM_LEVEL)
//
// Five gates from Leviathan et al. (2023) speculative decoding:
//
// - SD-001 (output distribution equivalence): KL(spec || auto) < 0.01
//   over 10K samples (we accept any KL bound the caller provides).
// - SD-002 (acceptance probability bounds): min(1, q/p) ∈ [0, 1].
// - SD-003 (adjusted distribution validity): r(x) = norm(max(0, q-p))
//   sums to 1 when rejection occurs.
// - SD-004 (min token emission): ≥ 1 token per draft-verify cycle.
// - SD-005 (deterministic acceptance given fixed RNG seed).
//
// In-module reference: `acceptance_prob`, `adjusted_distribution`,
// pure functions over (q, p) pairs.

/// Acceptance-probability lower bound (inclusive).
pub const AC_SD_002_ACCEPT_LOWER: f32 = 0.0;

/// Acceptance-probability upper bound (inclusive).
pub const AC_SD_002_ACCEPT_UPPER: f32 = 1.0;

/// SD-001 KL divergence threshold for output equivalence.
pub const AC_SD_001_KL_THRESHOLD: f32 = 0.01;

/// SD-004 minimum tokens emitted per draft-verify cycle.
pub const AC_SD_004_MIN_EMITTED: usize = 1;

/// Tolerance for adjusted-distribution sum normalization.
pub const AC_SD_003_NORMALIZATION_EPS: f32 = 1e-5;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SdVerdict {
    Pass,
    Fail,
}

// -----------------------------------------------------------------------------
// In-module reference functions.
// -----------------------------------------------------------------------------

/// Acceptance probability `min(1, q/p)`.
///
/// Returns `None` if `p ≤ 0` (invalid draft probability — division
/// by zero protected).
#[must_use]
pub fn acceptance_prob(q: f32, p: f32) -> Option<f32> {
    // `p > 0.0` is false for NaN, 0, or negative — all invalid.
    if !p.is_finite() || !q.is_finite() || p <= 0.0 {
        return None;
    }
    let ratio = q / p;
    if ratio.is_nan() {
        return None;
    }
    Some(ratio.clamp(0.0, 1.0))
}

/// Adjusted distribution `r(x) = normalize(max(0, q(x) - p(x)))`.
/// Returns `None` if no positive mass remains (impossible if q ≠ p).
#[must_use]
pub fn adjusted_distribution(q: &[f32], p: &[f32]) -> Option<Vec<f32>> {
    if q.len() != p.len() || q.is_empty() {
        return None;
    }
    let mut diff: Vec<f32> = q.iter().zip(p.iter()).map(|(qi, pi)| (qi - pi).max(0.0)).collect();
    let sum: f32 = diff.iter().sum();
    if !sum.is_finite() || sum <= 0.0 {
        return None;
    }
    for d in &mut diff {
        *d /= sum;
    }
    Some(diff)
}

// -----------------------------------------------------------------------------
// Verdict 1: SD-001 — output distribution equivalence.
// -----------------------------------------------------------------------------

/// Pass iff the empirical KL divergence between speculative and
/// autoregressive output distributions is below `AC_SD_001_KL_THRESHOLD`.
#[must_use]
pub fn verdict_from_output_equivalence_kl(kl_divergence: f32) -> SdVerdict {
    if !kl_divergence.is_finite() || kl_divergence < 0.0 {
        return SdVerdict::Fail;
    }
    if kl_divergence < AC_SD_001_KL_THRESHOLD {
        SdVerdict::Pass
    } else {
        SdVerdict::Fail
    }
}

// -----------------------------------------------------------------------------
// Verdict 2: SD-002 — acceptance probability bounds.
// -----------------------------------------------------------------------------

/// Pass iff `accept_prob ∈ [0, 1]` and finite.
#[must_use]
pub fn verdict_from_acceptance_bounds(accept_prob: f32) -> SdVerdict {
    if !accept_prob.is_finite() {
        return SdVerdict::Fail;
    }
    if (AC_SD_002_ACCEPT_LOWER..=AC_SD_002_ACCEPT_UPPER).contains(&accept_prob) {
        SdVerdict::Pass
    } else {
        SdVerdict::Fail
    }
}

// -----------------------------------------------------------------------------
// Verdict 3: SD-003 — adjusted distribution validity.
// -----------------------------------------------------------------------------

/// Pass iff:
///  1. all entries non-negative,
///  2. sum equals 1.0 within `AC_SD_003_NORMALIZATION_EPS`.
#[must_use]
pub fn verdict_from_adjusted_distribution(r: &[f32]) -> SdVerdict {
    if r.is_empty() {
        return SdVerdict::Fail;
    }
    let mut sum = 0.0_f32;
    for &v in r {
        if !v.is_finite() || v < 0.0 {
            return SdVerdict::Fail;
        }
        sum += v;
    }
    if (sum - 1.0).abs() < AC_SD_003_NORMALIZATION_EPS {
        SdVerdict::Pass
    } else {
        SdVerdict::Fail
    }
}

// -----------------------------------------------------------------------------
// Verdict 4: SD-004 — min token emission.
// -----------------------------------------------------------------------------

/// Pass iff `tokens_emitted >= 1`.
#[must_use]
pub fn verdict_from_min_emission(tokens_emitted: usize) -> SdVerdict {
    if tokens_emitted >= AC_SD_004_MIN_EMITTED {
        SdVerdict::Pass
    } else {
        SdVerdict::Fail
    }
}

// -----------------------------------------------------------------------------
// Verdict 5: SD-005 — deterministic acceptance given fixed seed.
// -----------------------------------------------------------------------------

#[must_use]
pub fn verdict_from_seed_determinism(run1: &[bool], run2: &[bool]) -> SdVerdict {
    if run1.len() != run2.len() {
        return SdVerdict::Fail;
    }
    if run1 == run2 {
        SdVerdict::Pass
    } else {
        SdVerdict::Fail
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Section 1: Provenance pins.
    // -------------------------------------------------------------------------
    #[test]
    fn provenance_accept_lower_zero() {
        assert_eq!(AC_SD_002_ACCEPT_LOWER, 0.0);
    }

    #[test]
    fn provenance_accept_upper_one() {
        assert_eq!(AC_SD_002_ACCEPT_UPPER, 1.0);
    }

    #[test]
    fn provenance_kl_threshold_001() {
        assert_eq!(AC_SD_001_KL_THRESHOLD, 0.01);
    }

    #[test]
    fn provenance_min_emission_one() {
        assert_eq!(AC_SD_004_MIN_EMITTED, 1);
    }

    #[test]
    fn provenance_normalization_eps_1e_5() {
        assert_eq!(AC_SD_003_NORMALIZATION_EPS, 1e-5);
    }

    // -------------------------------------------------------------------------
    // Section 2: SD-001 Pass band.
    // -------------------------------------------------------------------------
    #[test]
    fn sd001_pass_zero_kl() {
        assert_eq!(verdict_from_output_equivalence_kl(0.0), SdVerdict::Pass);
    }

    #[test]
    fn sd001_pass_small_kl() {
        assert_eq!(verdict_from_output_equivalence_kl(0.005), SdVerdict::Pass);
    }

    // -------------------------------------------------------------------------
    // Section 3: SD-001 Fail band.
    // -------------------------------------------------------------------------
    #[test]
    fn sd001_fail_at_threshold() {
        // Threshold is strict: 0.01 NOT inclusive.
        assert_eq!(verdict_from_output_equivalence_kl(0.01), SdVerdict::Fail);
    }

    #[test]
    fn sd001_fail_above_threshold() {
        assert_eq!(verdict_from_output_equivalence_kl(0.5), SdVerdict::Fail);
    }

    #[test]
    fn sd001_fail_negative_kl() {
        // KL is non-negative by definition.
        assert_eq!(verdict_from_output_equivalence_kl(-0.01), SdVerdict::Fail);
    }

    #[test]
    fn sd001_fail_nan_kl() {
        assert_eq!(verdict_from_output_equivalence_kl(f32::NAN), SdVerdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 4: SD-002 Pass band — acceptance bounds.
    // -------------------------------------------------------------------------
    #[test]
    fn sd002_pass_zero_acceptance() {
        assert_eq!(verdict_from_acceptance_bounds(0.0), SdVerdict::Pass);
    }

    #[test]
    fn sd002_pass_one_acceptance() {
        assert_eq!(verdict_from_acceptance_bounds(1.0), SdVerdict::Pass);
    }

    #[test]
    fn sd002_pass_half_acceptance() {
        assert_eq!(verdict_from_acceptance_bounds(0.5), SdVerdict::Pass);
    }

    // -------------------------------------------------------------------------
    // Section 5: SD-002 Fail band.
    // -------------------------------------------------------------------------
    #[test]
    fn sd002_fail_above_one() {
        // Without min(1, q/p), q/p can exceed 1.
        assert_eq!(verdict_from_acceptance_bounds(1.5), SdVerdict::Fail);
    }

    #[test]
    fn sd002_fail_negative() {
        assert_eq!(verdict_from_acceptance_bounds(-0.1), SdVerdict::Fail);
    }

    #[test]
    fn sd002_fail_nan() {
        // Division by zero when p(x) = 0 yields NaN.
        assert_eq!(verdict_from_acceptance_bounds(f32::NAN), SdVerdict::Fail);
    }

    #[test]
    fn sd002_fail_inf() {
        assert_eq!(
            verdict_from_acceptance_bounds(f32::INFINITY),
            SdVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 6: SD-003 — adjusted distribution.
    // -------------------------------------------------------------------------
    #[test]
    fn sd003_pass_uniform_normalized() {
        let r = vec![0.25_f32, 0.25, 0.25, 0.25];
        assert_eq!(verdict_from_adjusted_distribution(&r), SdVerdict::Pass);
    }

    #[test]
    fn sd003_pass_skewed() {
        let r = vec![0.7_f32, 0.2, 0.1];
        assert_eq!(verdict_from_adjusted_distribution(&r), SdVerdict::Pass);
    }

    #[test]
    fn sd003_pass_one_hot() {
        let r = vec![1.0_f32, 0.0, 0.0];
        assert_eq!(verdict_from_adjusted_distribution(&r), SdVerdict::Pass);
    }

    #[test]
    fn sd003_fail_negative_entry() {
        let r = vec![0.5_f32, -0.1, 0.6];
        assert_eq!(verdict_from_adjusted_distribution(&r), SdVerdict::Fail);
    }

    #[test]
    fn sd003_fail_does_not_sum_to_one() {
        let r = vec![0.5_f32, 0.3, 0.1]; // sum = 0.9
        assert_eq!(verdict_from_adjusted_distribution(&r), SdVerdict::Fail);
    }

    #[test]
    fn sd003_fail_empty() {
        let r: Vec<f32> = vec![];
        assert_eq!(verdict_from_adjusted_distribution(&r), SdVerdict::Fail);
    }

    #[test]
    fn sd003_fail_nan_entry() {
        let r = vec![0.5_f32, f32::NAN, 0.5];
        assert_eq!(verdict_from_adjusted_distribution(&r), SdVerdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 7: SD-004 — min emission.
    // -------------------------------------------------------------------------
    #[test]
    fn sd004_pass_one_token() {
        assert_eq!(verdict_from_min_emission(1), SdVerdict::Pass);
    }

    #[test]
    fn sd004_pass_many_tokens() {
        assert_eq!(verdict_from_min_emission(100), SdVerdict::Pass);
    }

    #[test]
    fn sd004_fail_zero_tokens() {
        // The contract failure: "all K drafts rejected and bonus token
        // not emitted" ⇒ 0 tokens.
        assert_eq!(verdict_from_min_emission(0), SdVerdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 8: SD-005 — determinism.
    // -------------------------------------------------------------------------
    #[test]
    fn sd005_pass_identical_decisions() {
        let r1 = vec![true, false, true, true];
        let r2 = vec![true, false, true, true];
        assert_eq!(verdict_from_seed_determinism(&r1, &r2), SdVerdict::Pass);
    }

    #[test]
    fn sd005_pass_empty_both() {
        let r: Vec<bool> = vec![];
        assert_eq!(verdict_from_seed_determinism(&r, &r), SdVerdict::Pass);
    }

    #[test]
    fn sd005_fail_one_off() {
        let r1 = vec![true, false, true, true];
        let r2 = vec![true, false, true, false]; // last differs
        assert_eq!(verdict_from_seed_determinism(&r1, &r2), SdVerdict::Fail);
    }

    #[test]
    fn sd005_fail_length_mismatch() {
        let r1 = vec![true, false];
        let r2 = vec![true, false, true];
        assert_eq!(verdict_from_seed_determinism(&r1, &r2), SdVerdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 9: Domain — reference functions.
    // -------------------------------------------------------------------------
    #[test]
    fn domain_acceptance_prob_q_geq_p_returns_one() {
        // q(x) >= p(x) ⇒ P(accept) = 1 (draft underestimates).
        assert_eq!(acceptance_prob(0.8, 0.4), Some(1.0));
        assert_eq!(acceptance_prob(0.5, 0.5), Some(1.0));
    }

    #[test]
    fn domain_acceptance_prob_q_lt_p_returns_ratio() {
        // q(x) < p(x) ⇒ P(accept) = q/p.
        let prob = acceptance_prob(0.2, 0.8).unwrap();
        assert!((prob - 0.25).abs() < 1e-6);
    }

    #[test]
    fn domain_acceptance_prob_zero_p_rejected() {
        // Division by zero protected.
        assert_eq!(acceptance_prob(0.5, 0.0), None);
    }

    #[test]
    fn domain_acceptance_prob_negative_p_rejected() {
        assert_eq!(acceptance_prob(0.5, -0.1), None);
    }

    #[test]
    fn domain_adjusted_distribution_basic() {
        // q biased toward index 0, p biased toward index 1 ⇒ r should
        // place mass on index 0.
        let q = vec![0.7_f32, 0.2, 0.1];
        let p = vec![0.1_f32, 0.7, 0.2];
        let r = adjusted_distribution(&q, &p).unwrap();
        // r = norm(max(0, q-p)) = norm([0.6, 0.0, 0.0]) = [1, 0, 0]
        assert!((r[0] - 1.0).abs() < 1e-5, "r={r:?}");
        assert!(r[1] < 1e-5);
        assert!(r[2] < 1e-5);
        assert_eq!(verdict_from_adjusted_distribution(&r), SdVerdict::Pass);
    }

    #[test]
    fn domain_adjusted_distribution_sum_zero_returns_none() {
        // q == p ⇒ all max(0, q-p) = 0 ⇒ no positive mass.
        let q = vec![0.5_f32, 0.5];
        let p = vec![0.5_f32, 0.5];
        assert!(adjusted_distribution(&q, &p).is_none());
    }

    // -------------------------------------------------------------------------
    // Section 10: Sweep — q, p combinations.
    // -------------------------------------------------------------------------
    #[test]
    fn sweep_acceptance_prob_in_band() {
        let test_cases = [
            (0.5_f32, 0.5),
            (0.9, 0.1),
            (0.1, 0.9),
            (0.3, 0.7),
            (0.99, 0.01),
        ];
        for (q, p) in test_cases {
            let prob = acceptance_prob(q, p).unwrap();
            assert_eq!(
                verdict_from_acceptance_bounds(prob),
                SdVerdict::Pass,
                "q={q} p={p} prob={prob}"
            );
        }
    }

    // -------------------------------------------------------------------------
    // Section 11: Realistic — contract regression scenarios.
    // -------------------------------------------------------------------------
    #[test]
    fn realistic_division_by_zero_caught() {
        // SD-002 if_fails: "Division by zero when p(x) = 0".
        // Reference acceptance_prob returns None; downstream
        // verdict_from_acceptance_bounds called on NaN must Fail.
        let buggy_unprotected = 0.5_f32 / 0.0; // = Inf
        assert_eq!(
            verdict_from_acceptance_bounds(buggy_unprotected),
            SdVerdict::Fail
        );
    }

    #[test]
    fn realistic_min_clip_required() {
        // SD-002 if_fails: replace min(1, q/p) with q/p ⇒ unbounded.
        // Without min clip, acceptance can exceed 1.
        let buggy_no_min_clip = 2.0_f32; // q=0.8, p=0.4, no min clip
        assert_eq!(
            verdict_from_acceptance_bounds(buggy_no_min_clip),
            SdVerdict::Fail
        );
    }

    #[test]
    fn realistic_normalization_error() {
        // SD-003 if_fails: "normalization error".
        let r_buggy_unnormalized = vec![0.3_f32, 0.3, 0.3]; // sum=0.9
        assert_eq!(
            verdict_from_adjusted_distribution(&r_buggy_unnormalized),
            SdVerdict::Fail
        );
    }

    #[test]
    fn realistic_all_drafts_rejected_no_bonus() {
        // SD-004 if_fails: "Edge case where all K drafts rejected and
        // bonus token not emitted".
        assert_eq!(verdict_from_min_emission(0), SdVerdict::Fail);
    }

    #[test]
    fn realistic_floating_point_ordering_breaks_determinism() {
        // SD-005 if_fails: "Non-determinism from floating-point
        // ordering or uninitialized state".
        let r1 = vec![true, true, false];
        let r2 = vec![true, false, false]; // hidden state changed mid-run
        assert_eq!(verdict_from_seed_determinism(&r1, &r2), SdVerdict::Fail);
    }

    #[test]
    fn realistic_full_speculative_step() {
        // End-to-end pure-Rust speculative step: q biased to 0, p to 2.
        let q = vec![0.6_f32, 0.3, 0.1];
        let p = vec![0.1_f32, 0.3, 0.6];
        // First, acceptance prob for each token:
        for (qi, pi) in q.iter().zip(p.iter()) {
            let prob = acceptance_prob(*qi, *pi).unwrap();
            assert_eq!(
                verdict_from_acceptance_bounds(prob),
                SdVerdict::Pass
            );
        }
        // Then adjusted distribution on rejection:
        let r = adjusted_distribution(&q, &p).unwrap();
        assert_eq!(verdict_from_adjusted_distribution(&r), SdVerdict::Pass);
        // r = norm([0.5, 0.0, 0.0]) = [1, 0, 0]
        assert!((r[0] - 1.0).abs() < 1e-5);
        // Min emission: at least 1 token.
        assert_eq!(verdict_from_min_emission(1), SdVerdict::Pass);
    }
}
