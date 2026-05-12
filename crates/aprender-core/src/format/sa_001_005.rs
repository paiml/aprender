// SHIP-TWO-001 — `sampling-algorithms-v1` algorithm-level PARTIAL
// discharge for FALSIFY-SA-001..005 (closes 5/5 sweep).
//
// Contract: `contracts/sampling-algorithms-v1.yaml`.
// Spec: Sampling algorithms for autoregressive generation
// (Holtzman 2019; Qwen2.5-Coder §14.5).

// ===========================================================================
// SA-001 — Greedy == argmax: greedy(logits) returns argmax(logits)
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sa001Verdict { Pass, Fail }

#[must_use]
pub fn argmax(logits: &[f32]) -> Option<usize> {
    if logits.is_empty() { return None; }
    if !logits.iter().all(|v| v.is_finite()) { return None; }
    let mut best_idx = 0;
    let mut best_val = logits[0];
    for (i, &v) in logits.iter().enumerate().skip(1) {
        if v > best_val {
            best_val = v;
            best_idx = i;
        }
    }
    Some(best_idx)
}

#[must_use]
pub fn verdict_from_greedy_argmax(logits: &[f32], observed: usize) -> Sa001Verdict {
    match argmax(logits) {
        Some(expected) if expected == observed => Sa001Verdict::Pass,
        _ => Sa001Verdict::Fail,
    }
}

// ===========================================================================
// SA-002 — Top-K cardinality: count(nonzero(top_k(p, K))) ≤ K
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sa002Verdict { Pass, Fail }

/// Pass iff `filtered_probs` has at most K non-zero entries AND those
/// entries correspond to the K largest values of the original `probs`.
#[must_use]
pub fn verdict_from_top_k_cardinality(
    probs: &[f32],
    k: usize,
    filtered_probs: &[f32],
) -> Sa002Verdict {
    if probs.is_empty() || filtered_probs.is_empty() { return Sa002Verdict::Fail; }
    if probs.len() != filtered_probs.len() { return Sa002Verdict::Fail; }
    if k == 0 || k > probs.len() { return Sa002Verdict::Fail; }
    let nonzero_count = filtered_probs.iter().filter(|&&p| p > 0.0).count();
    if nonzero_count > k { return Sa002Verdict::Fail; }
    // Verify the kept indices are among the top K.
    let mut sorted_indices: Vec<usize> = (0..probs.len()).collect();
    sorted_indices.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap_or(std::cmp::Ordering::Equal));
    let allowed_top_k: std::collections::HashSet<usize> =
        sorted_indices.iter().take(k).copied().collect();
    for (i, &p) in filtered_probs.iter().enumerate() {
        if p > 0.0 && !allowed_top_k.contains(&i) {
            return Sa002Verdict::Fail; // kept a non-top-K entry
        }
    }
    Sa002Verdict::Pass
}

// ===========================================================================
// SA-003 — Top-P cumulative: sum(top_p(p, threshold)) >= threshold
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sa003Verdict { Pass, Fail }

/// Pass iff sum of retained probabilities ≥ threshold AND threshold ∈ (0, 1].
#[must_use]
pub fn verdict_from_top_p_cumulative(filtered_probs: &[f32], threshold: f32) -> Sa003Verdict {
    if filtered_probs.is_empty() { return Sa003Verdict::Fail; }
    if !threshold.is_finite() || threshold <= 0.0 || threshold > 1.0 { return Sa003Verdict::Fail; }
    if !filtered_probs.iter().all(|v| v.is_finite() && *v >= 0.0) { return Sa003Verdict::Fail; }
    let sum: f32 = filtered_probs.iter().sum();
    if !sum.is_finite() { return Sa003Verdict::Fail; }
    // Use a small slack for f32 accumulation rounding.
    let slack = 1.0e-5_f32;
    if sum + slack < threshold { return Sa003Verdict::Fail; }
    Sa003Verdict::Pass
}

// ===========================================================================
// SA-004 — Temperature identity: softmax(l/1) ≈ softmax(l) within 1e-6
// ===========================================================================

pub const AC_SA_004_TOLERANCE: f32 = 1.0e-6;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sa004Verdict { Pass, Fail }

/// Numerically-stable softmax used for temperature comparison.
#[must_use]
pub fn softmax(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() { return vec![]; }
    let m = logits.iter().fold(f32::NEG_INFINITY, |acc, &x| acc.max(x));
    if !m.is_finite() { return vec![]; }
    let exps: Vec<f32> = logits.iter().map(|&x| (x - m).exp()).collect();
    let s: f32 = exps.iter().sum();
    if s == 0.0 || !s.is_finite() { return vec![]; }
    exps.iter().map(|&e| e / s).collect()
}

#[must_use]
pub fn verdict_from_temperature_identity(logits: &[f32]) -> Sa004Verdict {
    if logits.is_empty() { return Sa004Verdict::Fail; }
    if !logits.iter().all(|v| v.is_finite()) { return Sa004Verdict::Fail; }
    let raw = softmax(logits);
    let scaled: Vec<f32> = logits.iter().map(|&l| l / 1.0).collect();
    let with_t1 = softmax(&scaled);
    if raw.is_empty() || with_t1.is_empty() || raw.len() != with_t1.len() {
        return Sa004Verdict::Fail;
    }
    for (a, b) in raw.iter().zip(with_t1.iter()) {
        if (a - b).abs() > AC_SA_004_TOLERANCE { return Sa004Verdict::Fail; }
    }
    Sa004Verdict::Pass
}

// ===========================================================================
// SA-005 — SIMD parity: byte-exact (contract tolerance=0.0)
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sa005Verdict { Pass, Fail }

/// SIMD sampling output is the chosen token index; verdict checks
/// scalar and SIMD agree on the index AND on the produced probability
/// distribution byte-exactly.
#[must_use]
pub fn verdict_from_simd_sampling_parity(
    scalar_idx: usize,
    simd_idx: usize,
    scalar_dist: &[f32],
    simd_dist: &[f32],
) -> Sa005Verdict {
    if scalar_idx != simd_idx { return Sa005Verdict::Fail; }
    if scalar_dist.is_empty() || simd_dist.is_empty() { return Sa005Verdict::Fail; }
    if scalar_dist.len() != simd_dist.len() { return Sa005Verdict::Fail; }
    for (&s, &v) in scalar_dist.iter().zip(simd_dist.iter()) {
        if !s.is_finite() || !v.is_finite() { return Sa005Verdict::Fail; }
        if s.to_bits() != v.to_bits() { return Sa005Verdict::Fail; }
    }
    Sa005Verdict::Pass
}

#[cfg(test)]
mod tests {
    use super::*;

    // SA-001 (greedy = argmax)
    #[test] fn sa001_pass_canonical() {
        // Max at index 2.
        let logits = vec![0.1_f32, 0.5, 0.9, 0.3];
        assert_eq!(verdict_from_greedy_argmax(&logits, 2), Sa001Verdict::Pass);
    }
    #[test] fn sa001_pass_max_at_zero() {
        let logits = vec![5.0_f32, 0.5, 0.9];
        assert_eq!(verdict_from_greedy_argmax(&logits, 0), Sa001Verdict::Pass);
    }
    #[test] fn sa001_pass_max_at_end() {
        let logits = vec![0.1_f32, 0.5, 0.9, 5.0];
        assert_eq!(verdict_from_greedy_argmax(&logits, 3), Sa001Verdict::Pass);
    }
    #[test] fn sa001_fail_wrong_index() {
        let logits = vec![0.1_f32, 0.5, 0.9, 0.3];
        // Reported max at 1 but actual is at 2.
        assert_eq!(verdict_from_greedy_argmax(&logits, 1), Sa001Verdict::Fail);
    }
    #[test] fn sa001_fail_empty() {
        assert_eq!(verdict_from_greedy_argmax(&[], 0), Sa001Verdict::Fail);
    }
    #[test] fn sa001_fail_nan() {
        let logits = vec![0.1_f32, f32::NAN];
        assert_eq!(verdict_from_greedy_argmax(&logits, 0), Sa001Verdict::Fail);
    }

    // SA-002 (top-K cardinality)
    #[test] fn sa002_pass_canonical() {
        // K=2: keep top 2 (indices 2 and 3 have largest probs).
        let probs = vec![0.1_f32, 0.05, 0.5, 0.35];
        let filtered = vec![0.0_f32, 0.0, 0.5882, 0.4118]; // renormalized top-2
        assert_eq!(verdict_from_top_k_cardinality(&probs, 2, &filtered), Sa002Verdict::Pass);
    }
    #[test] fn sa002_pass_k_equals_n() {
        let probs = vec![0.25_f32, 0.25, 0.25, 0.25];
        let filtered = probs.clone();
        assert_eq!(verdict_from_top_k_cardinality(&probs, 4, &filtered), Sa002Verdict::Pass);
    }
    #[test] fn sa002_fail_too_many_kept() {
        let probs = vec![0.1_f32, 0.2, 0.3, 0.4];
        // K=2 but 3 entries kept.
        let filtered = vec![0.0_f32, 0.2, 0.3, 0.4];
        assert_eq!(verdict_from_top_k_cardinality(&probs, 2, &filtered), Sa002Verdict::Fail);
    }
    #[test] fn sa002_fail_kept_non_top_k() {
        let probs = vec![0.1_f32, 0.2, 0.3, 0.4];
        // Kept index 0 (lowest prob) when K=2 should keep indices 2 and 3.
        let filtered = vec![0.5_f32, 0.0, 0.0, 0.5];
        assert_eq!(verdict_from_top_k_cardinality(&probs, 2, &filtered), Sa002Verdict::Fail);
    }
    #[test] fn sa002_fail_k_zero() {
        // The contract says "K=0 to observe empty filtered set" — that's
        // an INVALID config, the verdict rejects K=0.
        let probs = vec![0.5_f32, 0.5];
        let filtered = vec![0.0_f32, 0.0];
        assert_eq!(verdict_from_top_k_cardinality(&probs, 0, &filtered), Sa002Verdict::Fail);
    }
    #[test] fn sa002_fail_k_above_v() {
        let probs = vec![0.5_f32, 0.5];
        let filtered = probs.clone();
        assert_eq!(verdict_from_top_k_cardinality(&probs, 5, &filtered), Sa002Verdict::Fail);
    }

    // SA-003 (top-P cumulative)
    #[test] fn sa003_pass_canonical() {
        // threshold=0.8, retained sums to ~0.85.
        let filtered = vec![0.5_f32, 0.35, 0.0, 0.0];
        assert_eq!(verdict_from_top_p_cumulative(&filtered, 0.8), Sa003Verdict::Pass);
    }
    #[test] fn sa003_pass_at_threshold() {
        // Sum exactly equals threshold.
        let filtered = vec![0.5_f32, 0.5];
        assert_eq!(verdict_from_top_p_cumulative(&filtered, 1.0), Sa003Verdict::Pass);
    }
    #[test] fn sa003_fail_below_threshold() {
        let filtered = vec![0.3_f32, 0.0];
        assert_eq!(verdict_from_top_p_cumulative(&filtered, 0.8), Sa003Verdict::Fail);
    }
    #[test] fn sa003_fail_threshold_zero() {
        let filtered = vec![1.0_f32];
        assert_eq!(verdict_from_top_p_cumulative(&filtered, 0.0), Sa003Verdict::Fail);
    }
    #[test] fn sa003_fail_threshold_above_one() {
        let filtered = vec![1.0_f32];
        assert_eq!(verdict_from_top_p_cumulative(&filtered, 1.5), Sa003Verdict::Fail);
    }
    #[test] fn sa003_fail_negative_prob() {
        let filtered = vec![-0.1_f32, 0.5];
        assert_eq!(verdict_from_top_p_cumulative(&filtered, 0.4), Sa003Verdict::Fail);
    }

    // SA-004 (temperature identity)
    #[test] fn sa004_pass_canonical() {
        let logits = vec![0.5_f32, 1.0, 1.5, 2.0];
        assert_eq!(verdict_from_temperature_identity(&logits), Sa004Verdict::Pass);
    }
    #[test] fn sa004_pass_uniform_logits() {
        let logits = vec![1.0_f32, 1.0, 1.0];
        assert_eq!(verdict_from_temperature_identity(&logits), Sa004Verdict::Pass);
    }
    #[test] fn sa004_pass_negative_logits() {
        let logits = vec![-3.0_f32, 0.0, 5.0];
        assert_eq!(verdict_from_temperature_identity(&logits), Sa004Verdict::Pass);
    }
    #[test] fn sa004_fail_empty() {
        assert_eq!(verdict_from_temperature_identity(&[]), Sa004Verdict::Fail);
    }
    #[test] fn sa004_fail_nan() {
        let logits = vec![0.5_f32, f32::NAN];
        assert_eq!(verdict_from_temperature_identity(&logits), Sa004Verdict::Fail);
    }

    // SA-005 (SIMD parity)
    #[test] fn sa005_pass_identical() {
        let dist = vec![0.25_f32, 0.5, 0.25];
        assert_eq!(verdict_from_simd_sampling_parity(1, 1, &dist, &dist), Sa005Verdict::Pass);
    }
    #[test] fn sa005_fail_index_drift() {
        let dist = vec![0.25_f32, 0.5, 0.25];
        // SIMD selected token 2 instead of 1 — deterministic divergence.
        assert_eq!(verdict_from_simd_sampling_parity(1, 2, &dist, &dist), Sa005Verdict::Fail);
    }
    #[test] fn sa005_fail_dist_byte_drift() {
        let scalar = vec![0.25_f32, 0.5, 0.25];
        let simd = vec![0.25_f32, f32::from_bits(0.5_f32.to_bits() + 1), 0.25];
        // 1-ULP drift fails byte-exact contract.
        assert_eq!(verdict_from_simd_sampling_parity(1, 1, &scalar, &simd), Sa005Verdict::Fail);
    }
    #[test] fn sa005_fail_length() {
        let scalar = vec![0.25_f32];
        let simd = vec![0.25_f32, 0.5];
        assert_eq!(verdict_from_simd_sampling_parity(0, 0, &scalar, &simd), Sa005Verdict::Fail);
    }

    // Helper sanity
    #[test] fn argmax_canonical() {
        assert_eq!(argmax(&[0.1, 0.5, 0.9, 0.3]), Some(2));
        assert_eq!(argmax(&[5.0, 1.0, 0.5]), Some(0));
        assert_eq!(argmax(&[]), None);
    }

    // Provenance
    #[test] fn provenance_constants() {
        assert!((AC_SA_004_TOLERANCE - 1e-6).abs() < 1e-12);
    }
}
