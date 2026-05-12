// SHIP-TWO-001 — `active-learning-v1` algorithm-level PARTIAL
// discharge for FALSIFY-AL-001..006 (closes 6/6 sweep).
//
// Contract: `contracts/active-learning-v1.yaml`.
// Spec: Active learning query strategies (Settles 2012; Lewis & Gale 1994).
//
// NOTE: gate IDs FALSIFY-AL-* clash with alibi-kernel-v1 (different
// active-learning context). Module name `al2_001_006` disambiguates.

// ===========================================================================
// AL-001 — Uncertainty score: u(p) = 1 - max_i(p_i) ∈ [0, 1]
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Al2_001Verdict { Pass, Fail }

#[must_use]
pub fn uncertainty_score(probs: &[f32]) -> Option<f32> {
    if probs.is_empty() { return None; }
    if !probs.iter().all(|p| p.is_finite() && *p >= 0.0 && *p <= 1.0) { return None; }
    let m = probs.iter().fold(0.0_f32, |acc, &p| acc.max(p));
    let u = 1.0 - m;
    if !u.is_finite() { return None; }
    Some(u)
}

#[must_use]
pub fn verdict_from_uncertainty_bounds(probs: &[f32]) -> Al2_001Verdict {
    match uncertainty_score(probs) {
        Some(u) if (0.0..=1.0).contains(&u) => Al2_001Verdict::Pass,
        _ => Al2_001Verdict::Fail,
    }
}

// ===========================================================================
// AL-002 — Margin score: m(p) = 1 - (p_(1) - p_(2)) ∈ [0, 1]
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Al2_002Verdict { Pass, Fail }

#[must_use]
pub fn margin_score(probs: &[f32]) -> Option<f32> {
    if probs.len() < 2 { return None; }
    if !probs.iter().all(|p| p.is_finite() && *p >= 0.0 && *p <= 1.0) { return None; }
    let mut sorted: Vec<f32> = probs.to_vec();
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let margin = 1.0 - (sorted[0] - sorted[1]);
    if !margin.is_finite() { return None; }
    Some(margin)
}

#[must_use]
pub fn verdict_from_margin_bounds(probs: &[f32]) -> Al2_002Verdict {
    match margin_score(probs) {
        Some(m) if (0.0..=1.0).contains(&m) => Al2_002Verdict::Pass,
        _ => Al2_002Verdict::Fail,
    }
}

// ===========================================================================
// AL-003 — Entropy non-negative: H(p) = -Σ p_i ln(p_i) ≥ 0
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Al2_003Verdict { Pass, Fail }

#[must_use]
pub fn entropy_score(probs: &[f32]) -> Option<f32> {
    if probs.is_empty() { return None; }
    if !probs.iter().all(|p| p.is_finite() && *p >= 0.0 && *p <= 1.0) { return None; }
    let mut h = 0.0_f32;
    for &p in probs {
        // 0 * ln(0) is defined as 0 (limit form).
        if p > 0.0 {
            h -= p * p.ln();
        }
    }
    if !h.is_finite() { return None; }
    Some(h)
}

#[must_use]
pub fn verdict_from_entropy_non_negative(probs: &[f32]) -> Al2_003Verdict {
    match entropy_score(probs) {
        // Allow tiny rounding slack for f32 precision near zero.
        Some(h) if h >= -1.0e-6 => Al2_003Verdict::Pass,
        _ => Al2_003Verdict::Fail,
    }
}

// ===========================================================================
// AL-004 — Vote entropy non-negative: H_vote ≥ 0
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Al2_004Verdict { Pass, Fail }

/// Pass iff vote-distribution entropy is ≥ 0. Caller passes vote counts;
/// verdict normalizes to a distribution and computes entropy.
#[must_use]
pub fn verdict_from_vote_entropy_non_negative(vote_counts: &[u64]) -> Al2_004Verdict {
    if vote_counts.is_empty() { return Al2_004Verdict::Fail; }
    let total: u64 = vote_counts.iter().sum();
    if total == 0 { return Al2_004Verdict::Fail; }
    let probs: Vec<f32> = vote_counts.iter().map(|&v| v as f32 / total as f32).collect();
    match entropy_score(&probs) {
        Some(h) if h >= -1.0e-6 => Al2_004Verdict::Pass,
        _ => Al2_004Verdict::Fail,
    }
}

// ===========================================================================
// AL-005 — Uncertainty monotonicity: u(uniform(k)) ≥ u(one_hot(k))
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Al2_005Verdict { Pass, Fail }

/// Pass iff uniform distribution has higher (or equal) uncertainty than
/// any one-hot distribution for the same k.
#[must_use]
pub fn verdict_from_uncertainty_monotonicity(k: u64) -> Al2_005Verdict {
    if k < 2 { return Al2_005Verdict::Fail; }
    let uniform_p = 1.0 / (k as f32);
    let uniform: Vec<f32> = vec![uniform_p; k as usize];
    let mut one_hot: Vec<f32> = vec![0.0; k as usize];
    one_hot[0] = 1.0;
    let u_uniform = match uncertainty_score(&uniform) {
        Some(v) => v,
        None => return Al2_005Verdict::Fail,
    };
    let u_one_hot = match uncertainty_score(&one_hot) {
        Some(v) => v,
        None => return Al2_005Verdict::Fail,
    };
    if u_uniform >= u_one_hot { Al2_005Verdict::Pass } else { Al2_005Verdict::Fail }
}

// ===========================================================================
// AL-006 — Entropy finiteness: H(p) is finite for any valid prob vector
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Al2_006Verdict { Pass, Fail }

/// Pass iff entropy is finite for every probe distribution provided
/// (including near-zero entries that would trigger 0*ln(0) handling).
#[must_use]
pub fn verdict_from_entropy_finiteness(distributions: &[Vec<f32>]) -> Al2_006Verdict {
    if distributions.is_empty() { return Al2_006Verdict::Fail; }
    for dist in distributions {
        match entropy_score(dist) {
            Some(h) if h.is_finite() => {}
            _ => return Al2_006Verdict::Fail,
        }
    }
    Al2_006Verdict::Pass
}

#[cfg(test)]
mod tests {
    use super::*;

    // AL-001 (uncertainty bounds)
    #[test] fn al2_001_pass_canonical() {
        let p = vec![0.7_f32, 0.2, 0.1];
        assert_eq!(verdict_from_uncertainty_bounds(&p), Al2_001Verdict::Pass);
    }
    #[test] fn al2_001_pass_one_hot() {
        // u([1, 0, 0]) = 1 - 1 = 0.
        let p = vec![1.0_f32, 0.0, 0.0];
        let u = uncertainty_score(&p).unwrap();
        assert_eq!(u, 0.0);
        assert_eq!(verdict_from_uncertainty_bounds(&p), Al2_001Verdict::Pass);
    }
    #[test] fn al2_001_pass_uniform_3() {
        // u([1/3, 1/3, 1/3]) = 1 - 1/3 ≈ 0.667.
        let p = vec![1.0_f32 / 3.0; 3];
        assert_eq!(verdict_from_uncertainty_bounds(&p), Al2_001Verdict::Pass);
    }
    #[test] fn al2_001_fail_nan() {
        let p = vec![0.5_f32, f32::NAN];
        assert_eq!(verdict_from_uncertainty_bounds(&p), Al2_001Verdict::Fail);
    }
    #[test] fn al2_001_fail_negative_prob() {
        let p = vec![0.5_f32, -0.1, 0.6];
        assert_eq!(verdict_from_uncertainty_bounds(&p), Al2_001Verdict::Fail);
    }
    #[test] fn al2_001_fail_above_one() {
        let p = vec![1.5_f32];
        assert_eq!(verdict_from_uncertainty_bounds(&p), Al2_001Verdict::Fail);
    }
    #[test] fn al2_001_fail_empty() {
        assert_eq!(verdict_from_uncertainty_bounds(&[]), Al2_001Verdict::Fail);
    }

    // AL-002 (margin bounds)
    #[test] fn al2_002_pass_canonical() {
        let p = vec![0.7_f32, 0.2, 0.1];
        // margin = 1 - (0.7 - 0.2) = 0.5.
        assert_eq!(verdict_from_margin_bounds(&p), Al2_002Verdict::Pass);
    }
    #[test] fn al2_002_pass_one_hot() {
        // margin = 1 - 1 = 0.
        let p = vec![1.0_f32, 0.0];
        assert_eq!(verdict_from_margin_bounds(&p), Al2_002Verdict::Pass);
    }
    #[test] fn al2_002_pass_tie() {
        // margin = 1 - 0 = 1 (max margin = max ambiguity).
        let p = vec![0.5_f32, 0.5];
        assert_eq!(verdict_from_margin_bounds(&p), Al2_002Verdict::Pass);
    }
    #[test] fn al2_002_fail_too_few() {
        // Need ≥ 2 classes.
        let p = vec![1.0_f32];
        assert_eq!(verdict_from_margin_bounds(&p), Al2_002Verdict::Fail);
    }
    #[test] fn al2_002_fail_invalid_prob() {
        let p = vec![0.7_f32, 1.5];
        assert_eq!(verdict_from_margin_bounds(&p), Al2_002Verdict::Fail);
    }

    // AL-003 (entropy non-negative)
    #[test] fn al2_003_pass_canonical() {
        let p = vec![0.5_f32, 0.5];
        // H = -0.5*ln(0.5) - 0.5*ln(0.5) = ln(2) ≈ 0.693.
        assert_eq!(verdict_from_entropy_non_negative(&p), Al2_003Verdict::Pass);
    }
    #[test] fn al2_003_pass_one_hot() {
        // H([1, 0]) = 0 (degenerate).
        let p = vec![1.0_f32, 0.0];
        assert_eq!(verdict_from_entropy_non_negative(&p), Al2_003Verdict::Pass);
    }
    #[test] fn al2_003_pass_uniform_4() {
        // H([0.25; 4]) = ln(4) ≈ 1.386.
        let p = vec![0.25_f32; 4];
        assert_eq!(verdict_from_entropy_non_negative(&p), Al2_003Verdict::Pass);
    }
    #[test] fn al2_003_fail_invalid() {
        let p = vec![-0.1_f32, 1.1];
        assert_eq!(verdict_from_entropy_non_negative(&p), Al2_003Verdict::Fail);
    }

    // AL-004 (vote entropy non-negative)
    #[test] fn al2_004_pass_unanimous() {
        // All votes for class 0 → entropy = 0.
        let v = vec![5_u64, 0, 0];
        assert_eq!(verdict_from_vote_entropy_non_negative(&v), Al2_004Verdict::Pass);
    }
    #[test] fn al2_004_pass_split() {
        // Split votes → positive entropy.
        let v = vec![3_u64, 3, 4];
        assert_eq!(verdict_from_vote_entropy_non_negative(&v), Al2_004Verdict::Pass);
    }
    #[test] fn al2_004_fail_zero_total() {
        let v = vec![0_u64, 0, 0];
        assert_eq!(verdict_from_vote_entropy_non_negative(&v), Al2_004Verdict::Fail);
    }
    #[test] fn al2_004_fail_empty() {
        assert_eq!(verdict_from_vote_entropy_non_negative(&[]), Al2_004Verdict::Fail);
    }

    // AL-005 (uncertainty monotonicity)
    #[test] fn al2_005_pass_k_2() {
        // u(uniform_2) = 0.5; u(one_hot) = 0; 0.5 ≥ 0.
        assert_eq!(verdict_from_uncertainty_monotonicity(2), Al2_005Verdict::Pass);
    }
    #[test] fn al2_005_pass_k_10() {
        // u(uniform_10) = 0.9; u(one_hot) = 0.
        assert_eq!(verdict_from_uncertainty_monotonicity(10), Al2_005Verdict::Pass);
    }
    #[test] fn al2_005_fail_k_1() {
        // Need ≥ 2 classes.
        assert_eq!(verdict_from_uncertainty_monotonicity(1), Al2_005Verdict::Fail);
    }
    #[test] fn al2_005_fail_k_0() {
        assert_eq!(verdict_from_uncertainty_monotonicity(0), Al2_005Verdict::Fail);
    }

    // AL-006 (entropy finiteness)
    #[test] fn al2_006_pass_canonical() {
        let dists = vec![
            vec![0.5_f32, 0.5],
            vec![0.25_f32; 4],
            vec![1.0_f32, 0.0, 0.0], // 0 * ln(0) defined as 0
            vec![0.99_f32, 0.01],
        ];
        assert_eq!(verdict_from_entropy_finiteness(&dists), Al2_006Verdict::Pass);
    }
    #[test] fn al2_006_pass_near_zero_entries() {
        // The contract's stated regression: "0*ln(0) not handled correctly".
        let dists = vec![vec![0.999_f32, 0.001], vec![1.0_f32 - 1e-7, 1e-7]];
        assert_eq!(verdict_from_entropy_finiteness(&dists), Al2_006Verdict::Pass);
    }
    #[test] fn al2_006_fail_invalid_prob() {
        let dists = vec![vec![0.5_f32, 0.5], vec![-0.1_f32, 1.1]];
        assert_eq!(verdict_from_entropy_finiteness(&dists), Al2_006Verdict::Fail);
    }
    #[test] fn al2_006_fail_nan() {
        let dists = vec![vec![0.5_f32, f32::NAN]];
        assert_eq!(verdict_from_entropy_finiteness(&dists), Al2_006Verdict::Fail);
    }
    #[test] fn al2_006_fail_empty() {
        let dists: Vec<Vec<f32>> = vec![];
        assert_eq!(verdict_from_entropy_finiteness(&dists), Al2_006Verdict::Fail);
    }

    // Helper sanity
    #[test] fn entropy_uniform_2_is_ln2() {
        let h = entropy_score(&[0.5_f32, 0.5]).unwrap();
        assert!((h - 2.0_f32.ln()).abs() < 1e-6);
    }
    #[test] fn entropy_one_hot_is_zero() {
        let h = entropy_score(&[1.0_f32, 0.0]).unwrap();
        assert!(h.abs() < 1e-6);
    }
    #[test] fn margin_max_is_one() {
        // p = [0.5, 0.5] → margin = 1 - 0 = 1.
        let m = margin_score(&[0.5_f32, 0.5]).unwrap();
        assert!((m - 1.0).abs() < 1e-6);
    }
}
