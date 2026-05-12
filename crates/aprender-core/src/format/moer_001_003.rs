// SHIP-TWO-001 — `moe-router-v1` algorithm-level PARTIAL discharge
// for FALSIFY-MOE_ROUTER_V1_001..003.
//
// Contract: `contracts/moe-router-v1.yaml`.
// Spec: `docs/specifications/aprender-train/ship-two-models-spec.md`.
//
// ## What this file proves NOW (PARTIAL_ALGORITHM_LEVEL)
//
// Three router gates from Shazeer et al. (2017):
//
// - MOE-ROUTER-001 (softmax sum): for every token, the router-probability
//   row sums to 1.0.
// - MOE-ROUTER-002 (top-k count): for every token, exactly k experts
//   selected where k = num_experts_per_token.
// - MOE-ROUTER-003 (weight renormalization): for every token, the sum
//   of selected (post-renormalization) weights is 1.0.
//
// In-module reference: `softmax_row`, `top_k_indices`, `renormalize`.

/// Tolerance on Σ router_probs == 1.0 per row (allow softmax round-off).
pub const AC_MOE_ROUTER_001_SOFTMAX_SUM_EPS: f32 = 1e-5;

/// Tolerance on Σ selected_weights == 1.0 per row after renormalization.
pub const AC_MOE_ROUTER_003_RENORM_SUM_EPS: f32 = 1e-5;

/// Lower bound on k (must select at least one expert).
pub const AC_MOE_ROUTER_002_K_MIN: usize = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MoeRouterVerdict {
    Pass,
    Fail,
}

// -----------------------------------------------------------------------------
// In-module reference router.
// -----------------------------------------------------------------------------

/// Numerically stable softmax over a row.
#[must_use]
pub fn softmax_row(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() {
        return Vec::new();
    }
    let mut max = f32::NEG_INFINITY;
    for &v in logits {
        if v > max {
            max = v;
        }
    }
    let mut exps: Vec<f32> = logits.iter().map(|&v| (v - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum > 0.0 {
        for v in &mut exps {
            *v /= sum;
        }
    }
    exps
}

/// Top-k indices by descending probability. Ties broken by smallest
/// index. Returns indices into the original row.
#[must_use]
pub fn top_k_indices(probs: &[f32], k: usize) -> Vec<usize> {
    if probs.is_empty() || k == 0 || k > probs.len() {
        return Vec::new();
    }
    let mut idx: Vec<usize> = (0..probs.len()).collect();
    idx.sort_by(|&a, &b| {
        probs[b]
            .partial_cmp(&probs[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    idx.into_iter().take(k).collect()
}

/// Renormalize a slice of selected weights so they sum to 1.0.
#[must_use]
pub fn renormalize(selected: &[f32]) -> Vec<f32> {
    let sum: f32 = selected.iter().sum();
    if sum > 0.0 {
        selected.iter().map(|w| w / sum).collect()
    } else {
        selected.to_vec()
    }
}

// -----------------------------------------------------------------------------
// Verdict 1: MOE-ROUTER-001 — softmax sum.
// -----------------------------------------------------------------------------

/// Pass iff every row of `router_probs` (laid out row-major as
/// `[n_tokens, n_experts]`) sums to 1.0 within
/// `AC_MOE_ROUTER_001_SOFTMAX_SUM_EPS`.
#[must_use]
pub fn verdict_from_softmax_sum(
    router_probs: &[f32],
    n_tokens: usize,
    n_experts: usize,
) -> MoeRouterVerdict {
    if n_tokens == 0 || n_experts == 0 {
        return MoeRouterVerdict::Fail;
    }
    if router_probs.len() != n_tokens * n_experts {
        return MoeRouterVerdict::Fail;
    }
    for t in 0..n_tokens {
        let row = &router_probs[t * n_experts..(t + 1) * n_experts];
        let mut sum = 0.0_f32;
        for &v in row {
            if !v.is_finite() || v < 0.0 {
                return MoeRouterVerdict::Fail;
            }
            sum += v;
        }
        if (sum - 1.0).abs() >= AC_MOE_ROUTER_001_SOFTMAX_SUM_EPS {
            return MoeRouterVerdict::Fail;
        }
    }
    MoeRouterVerdict::Pass
}

// -----------------------------------------------------------------------------
// Verdict 2: MOE-ROUTER-002 — top-k count.
// -----------------------------------------------------------------------------

/// Pass iff every token row in `selected_indices_per_token` has length
/// `k`, all entries are unique, and all are < `n_experts`.
#[must_use]
pub fn verdict_from_topk_selection(
    selected_indices_per_token: &[Vec<usize>],
    k: usize,
    n_experts: usize,
) -> MoeRouterVerdict {
    if k < AC_MOE_ROUTER_002_K_MIN || k > n_experts {
        return MoeRouterVerdict::Fail;
    }
    if selected_indices_per_token.is_empty() {
        return MoeRouterVerdict::Fail;
    }
    for row in selected_indices_per_token {
        if row.len() != k {
            return MoeRouterVerdict::Fail;
        }
        let mut seen = vec![false; n_experts];
        for &idx in row {
            if idx >= n_experts {
                return MoeRouterVerdict::Fail;
            }
            if seen[idx] {
                // Duplicate: top-k must select distinct experts.
                return MoeRouterVerdict::Fail;
            }
            seen[idx] = true;
        }
    }
    MoeRouterVerdict::Pass
}

// -----------------------------------------------------------------------------
// Verdict 3: MOE-ROUTER-003 — renormalized weights sum to 1.
// -----------------------------------------------------------------------------

/// Pass iff every row in `selected_weights_per_token` sums to 1.0
/// within `AC_MOE_ROUTER_003_RENORM_SUM_EPS`.
#[must_use]
pub fn verdict_from_renorm_sum(
    selected_weights_per_token: &[Vec<f32>],
) -> MoeRouterVerdict {
    if selected_weights_per_token.is_empty() {
        return MoeRouterVerdict::Fail;
    }
    for row in selected_weights_per_token {
        if row.is_empty() {
            return MoeRouterVerdict::Fail;
        }
        let mut sum = 0.0_f32;
        for &v in row {
            if !v.is_finite() || v < 0.0 {
                return MoeRouterVerdict::Fail;
            }
            sum += v;
        }
        if (sum - 1.0).abs() >= AC_MOE_ROUTER_003_RENORM_SUM_EPS {
            return MoeRouterVerdict::Fail;
        }
    }
    MoeRouterVerdict::Pass
}

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Section 1: Provenance pins.
    // -------------------------------------------------------------------------
    #[test]
    fn provenance_softmax_eps_1e_5() {
        assert_eq!(AC_MOE_ROUTER_001_SOFTMAX_SUM_EPS, 1e-5);
    }

    #[test]
    fn provenance_renorm_eps_1e_5() {
        assert_eq!(AC_MOE_ROUTER_003_RENORM_SUM_EPS, 1e-5);
    }

    #[test]
    fn provenance_k_min_is_one() {
        assert_eq!(AC_MOE_ROUTER_002_K_MIN, 1);
    }

    // -------------------------------------------------------------------------
    // Section 2: MOE-ROUTER-001 Pass band.
    // -------------------------------------------------------------------------
    #[test]
    fn moer001_pass_uniform_distribution() {
        let probs = vec![0.25_f32; 8]; // 2 tokens × 4 experts
        assert_eq!(
            verdict_from_softmax_sum(&probs, 2, 4),
            MoeRouterVerdict::Pass
        );
    }

    #[test]
    fn moer001_pass_after_softmax() {
        let logits = vec![1.0_f32, 2.0, 3.0, 4.0];
        let probs = softmax_row(&logits);
        assert_eq!(
            verdict_from_softmax_sum(&probs, 1, 4),
            MoeRouterVerdict::Pass
        );
    }

    #[test]
    fn moer001_pass_skewed_distribution() {
        let probs = vec![0.7_f32, 0.2, 0.1, 0.0];
        assert_eq!(
            verdict_from_softmax_sum(&probs, 1, 4),
            MoeRouterVerdict::Pass
        );
    }

    // -------------------------------------------------------------------------
    // Section 3: MOE-ROUTER-001 Fail band.
    // -------------------------------------------------------------------------
    #[test]
    fn moer001_fail_does_not_sum_to_one() {
        let probs = vec![0.5_f32, 0.3, 0.1, 0.05]; // sum = 0.95
        assert_eq!(
            verdict_from_softmax_sum(&probs, 1, 4),
            MoeRouterVerdict::Fail
        );
    }

    #[test]
    fn moer001_fail_negative_entry() {
        let probs = vec![1.5_f32, -0.5, 0.0, 0.0];
        assert_eq!(
            verdict_from_softmax_sum(&probs, 1, 4),
            MoeRouterVerdict::Fail
        );
    }

    #[test]
    fn moer001_fail_nan_entry() {
        let probs = vec![0.5_f32, 0.5, f32::NAN, 0.0];
        assert_eq!(
            verdict_from_softmax_sum(&probs, 1, 4),
            MoeRouterVerdict::Fail
        );
    }

    #[test]
    fn moer001_fail_buffer_size_mismatch() {
        let probs = vec![0.25_f32; 7]; // not 2*4 = 8
        assert_eq!(
            verdict_from_softmax_sum(&probs, 2, 4),
            MoeRouterVerdict::Fail
        );
    }

    #[test]
    fn moer001_fail_zero_tokens_or_experts() {
        let probs = vec![0.5_f32];
        assert_eq!(
            verdict_from_softmax_sum(&probs, 0, 4),
            MoeRouterVerdict::Fail
        );
        assert_eq!(
            verdict_from_softmax_sum(&probs, 1, 0),
            MoeRouterVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 4: MOE-ROUTER-002 Pass band.
    // -------------------------------------------------------------------------
    #[test]
    fn moer002_pass_top2_of_8() {
        let selections = vec![
            vec![0_usize, 5],
            vec![3_usize, 7],
            vec![1_usize, 2],
        ];
        assert_eq!(
            verdict_from_topk_selection(&selections, 2, 8),
            MoeRouterVerdict::Pass
        );
    }

    #[test]
    fn moer002_pass_top1_minimum_k() {
        let selections = vec![vec![0_usize], vec![3_usize]];
        assert_eq!(
            verdict_from_topk_selection(&selections, 1, 4),
            MoeRouterVerdict::Pass
        );
    }

    // -------------------------------------------------------------------------
    // Section 5: MOE-ROUTER-002 Fail band.
    // -------------------------------------------------------------------------
    #[test]
    fn moer002_fail_too_few_experts() {
        let selections = vec![vec![0_usize]]; // k=2 expected
        assert_eq!(
            verdict_from_topk_selection(&selections, 2, 4),
            MoeRouterVerdict::Fail
        );
    }

    #[test]
    fn moer002_fail_too_many_experts() {
        let selections = vec![vec![0_usize, 1, 2]]; // k=2 expected
        assert_eq!(
            verdict_from_topk_selection(&selections, 2, 4),
            MoeRouterVerdict::Fail
        );
    }

    #[test]
    fn moer002_fail_k_exceeds_n_experts() {
        let selections = vec![vec![0_usize, 1]];
        // k=5 > n_experts=4 — invalid request.
        assert_eq!(
            verdict_from_topk_selection(&selections, 5, 4),
            MoeRouterVerdict::Fail
        );
    }

    #[test]
    fn moer002_fail_k_zero() {
        let selections = vec![vec![0_usize]];
        assert_eq!(
            verdict_from_topk_selection(&selections, 0, 4),
            MoeRouterVerdict::Fail
        );
    }

    #[test]
    fn moer002_fail_index_out_of_range() {
        let selections = vec![vec![0_usize, 99]];
        assert_eq!(
            verdict_from_topk_selection(&selections, 2, 4),
            MoeRouterVerdict::Fail
        );
    }

    #[test]
    fn moer002_fail_duplicate_indices() {
        let selections = vec![vec![1_usize, 1]];
        assert_eq!(
            verdict_from_topk_selection(&selections, 2, 4),
            MoeRouterVerdict::Fail
        );
    }

    #[test]
    fn moer002_fail_empty_token_list() {
        let selections: Vec<Vec<usize>> = vec![];
        assert_eq!(
            verdict_from_topk_selection(&selections, 2, 4),
            MoeRouterVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 6: MOE-ROUTER-003 Pass band.
    // -------------------------------------------------------------------------
    #[test]
    fn moer003_pass_uniform_renormed() {
        let weights = vec![vec![0.5_f32, 0.5], vec![0.25, 0.25, 0.25, 0.25]];
        assert_eq!(verdict_from_renorm_sum(&weights), MoeRouterVerdict::Pass);
    }

    #[test]
    fn moer003_pass_after_renormalize() {
        let raw = vec![0.6_f32, 0.3]; // sums to 0.9 — needs renorm
        let r = renormalize(&raw);
        let weights = vec![r];
        assert_eq!(verdict_from_renorm_sum(&weights), MoeRouterVerdict::Pass);
    }

    #[test]
    fn moer003_pass_skewed_renormed() {
        let weights = vec![vec![0.7_f32, 0.2, 0.1]];
        assert_eq!(verdict_from_renorm_sum(&weights), MoeRouterVerdict::Pass);
    }

    // -------------------------------------------------------------------------
    // Section 7: MOE-ROUTER-003 Fail band.
    // -------------------------------------------------------------------------
    #[test]
    fn moer003_fail_does_not_sum_to_one() {
        let weights = vec![vec![0.4_f32, 0.4]]; // sum 0.8
        assert_eq!(verdict_from_renorm_sum(&weights), MoeRouterVerdict::Fail);
    }

    #[test]
    fn moer003_fail_negative_weight() {
        let weights = vec![vec![1.5_f32, -0.5]];
        assert_eq!(verdict_from_renorm_sum(&weights), MoeRouterVerdict::Fail);
    }

    #[test]
    fn moer003_fail_nan_weight() {
        let weights = vec![vec![0.5_f32, f32::NAN]];
        assert_eq!(verdict_from_renorm_sum(&weights), MoeRouterVerdict::Fail);
    }

    #[test]
    fn moer003_fail_empty_token_list() {
        let weights: Vec<Vec<f32>> = vec![];
        assert_eq!(verdict_from_renorm_sum(&weights), MoeRouterVerdict::Fail);
    }

    #[test]
    fn moer003_fail_empty_row() {
        let weights = vec![vec![]];
        assert_eq!(verdict_from_renorm_sum(&weights), MoeRouterVerdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 8: Domain — reference functions.
    // -------------------------------------------------------------------------
    #[test]
    fn domain_softmax_row_basic() {
        let logits = vec![1.0_f32, 2.0, 3.0];
        let probs = softmax_row(&logits);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        // Probs should be increasing because logits are.
        assert!(probs[0] < probs[1]);
        assert!(probs[1] < probs[2]);
    }

    #[test]
    fn domain_softmax_row_extreme_logits_no_overflow() {
        // The numerical-stability trick: subtract max.
        let logits = vec![1000.0_f32, 1000.0, 1000.0];
        let probs = softmax_row(&logits);
        for &p in &probs {
            assert!(p.is_finite());
            assert!((p - 1.0 / 3.0).abs() < 1e-5);
        }
    }

    #[test]
    fn domain_top_k_indices_descending() {
        // logits=[1, 5, 3, 2, 4] → top-3 indices in descending prob:
        // 1 (5), 4 (4), 2 (3).
        let probs = vec![1.0_f32, 5.0, 3.0, 2.0, 4.0];
        let top = top_k_indices(&probs, 3);
        assert_eq!(top, vec![1, 4, 2]);
    }

    #[test]
    fn domain_top_k_indices_ties_break_smallest() {
        let probs = vec![3.0_f32, 3.0, 3.0, 1.0];
        let top = top_k_indices(&probs, 2);
        // Tied at 3.0: smallest indices first (0, 1).
        assert_eq!(top, vec![0, 1]);
    }

    #[test]
    fn domain_renormalize_basic() {
        let raw = vec![0.4_f32, 0.4]; // sum 0.8
        let r = renormalize(&raw);
        assert!((r[0] - 0.5).abs() < 1e-6);
        assert!((r[1] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn domain_renormalize_all_zero_returns_input() {
        // No renormalization possible — leave as-is (will Fail verdict).
        let raw = vec![0.0_f32, 0.0];
        let r = renormalize(&raw);
        assert_eq!(r, raw);
    }

    // -------------------------------------------------------------------------
    // Section 9: Sweep — k, n_experts.
    // -------------------------------------------------------------------------
    #[test]
    fn sweep_topk_valid_k_values() {
        // n_experts=8, k from 1 to 8 — all should pass given valid
        // selection.
        for k in 1..=8 {
            let mut sel: Vec<usize> = (0..k).collect(); // [0, 1, ..., k-1]
            let _ = sel.split_off(k); // truncate (already correct)
            let selections = vec![sel.clone()];
            assert_eq!(
                verdict_from_topk_selection(&selections, k, 8),
                MoeRouterVerdict::Pass,
                "k={k}"
            );
        }
    }

    // -------------------------------------------------------------------------
    // Section 10: Realistic — end-to-end MoE router.
    // -------------------------------------------------------------------------
    #[test]
    fn realistic_full_router_pipeline() {
        // 2 tokens, 4 experts, top-2.
        let logits = vec![
            // token 0: expert 1 dominant
            1.0_f32, 5.0, 0.5, 0.0,
            // token 1: expert 3 dominant
            0.0, 1.0, 2.0, 5.0,
        ];
        let n_experts = 4;
        let k = 2;

        // Step 1: softmax per row.
        let mut probs = Vec::new();
        for t in 0..2 {
            let row = &logits[t * n_experts..(t + 1) * n_experts];
            probs.extend(softmax_row(row));
        }
        assert_eq!(
            verdict_from_softmax_sum(&probs, 2, n_experts),
            MoeRouterVerdict::Pass
        );

        // Step 2: top-k selection.
        let mut selected_idx: Vec<Vec<usize>> = Vec::new();
        let mut selected_w: Vec<Vec<f32>> = Vec::new();
        for t in 0..2 {
            let row = &probs[t * n_experts..(t + 1) * n_experts];
            let top = top_k_indices(row, k);
            let raw_w: Vec<f32> = top.iter().map(|&i| row[i]).collect();
            let renormed = renormalize(&raw_w);
            selected_idx.push(top);
            selected_w.push(renormed);
        }
        assert_eq!(
            verdict_from_topk_selection(&selected_idx, k, n_experts),
            MoeRouterVerdict::Pass
        );
        assert_eq!(
            verdict_from_renorm_sum(&selected_w),
            MoeRouterVerdict::Pass
        );

        // Sanity: token 0 should pick experts {1, 0} (0 was second
        // highest by virtue of small logit gap).
        // Just check that expert 1 is picked.
        assert!(selected_idx[0].contains(&1));
        assert!(selected_idx[1].contains(&3));
    }
}
