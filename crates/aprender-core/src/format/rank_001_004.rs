// SHIP-TWO-001 — `metrics-ranking-v1` algorithm-level PARTIAL
// discharge for FALSIFY-RANK-001..004.
//
// Contract: `contracts/metrics-ranking-v1.yaml`.
// Spec: `docs/specifications/aprender-train/ship-two-models-spec.md`.
//
// ## What this file proves NOW (PARTIAL_ALGORITHM_LEVEL)
//
// Four ranking-metric gates from Manning et al. (2008) Ch. 8 +
// Järvelin & Käkäläinen (2002) NDCG:
//
// - RANK-001 (hit@k binary): hit@k ∈ {0, 1}.
// - RANK-002 (MRR bounded): MRR ∈ [0, 1].
// - RANK-003 (NDCG perfect ranking): NDCG@k = 1 for sorted ranking.
// - RANK-004 (NDCG bounded): NDCG@k ∈ [0, 1].

/// MRR/NDCG lower bound (inclusive).
pub const AC_RANK_BOUND_LOWER: f32 = 0.0;

/// MRR/NDCG upper bound (inclusive).
pub const AC_RANK_BOUND_UPPER: f32 = 1.0;

/// NDCG = 1.0 tolerance for perfect ranking (FP32 wobble).
pub const AC_RANK_003_PERFECT_EPS: f32 = 1e-5;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RankVerdict {
    Pass,
    Fail,
}

// -----------------------------------------------------------------------------
// In-module reference helpers.
// -----------------------------------------------------------------------------

/// hit@k: 1 if any item in top-k is relevant; else 0.
///
/// `ranked_relevance` — for each rank position, true iff the item at
/// that position is relevant. Length n; we look at first k entries.
#[must_use]
pub fn hit_at_k(ranked_relevance: &[bool], k: usize) -> u32 {
    let upper = k.min(ranked_relevance.len());
    if ranked_relevance[..upper].iter().any(|&r| r) {
        1
    } else {
        0
    }
}

/// Reciprocal rank: 1/rank of first relevant item, or 0 if none.
#[must_use]
pub fn reciprocal_rank(ranked_relevance: &[bool]) -> f32 {
    for (i, &r) in ranked_relevance.iter().enumerate() {
        if r {
            return 1.0 / (i as f32 + 1.0);
        }
    }
    0.0
}

/// MRR = mean reciprocal rank over all queries.
#[must_use]
pub fn mrr(reciprocal_ranks: &[f32]) -> f32 {
    if reciprocal_ranks.is_empty() {
        return 0.0;
    }
    let sum: f32 = reciprocal_ranks.iter().sum();
    sum / reciprocal_ranks.len() as f32
}

/// DCG@k = Σ_{i=1}^{k} rel_i / log2(i+1).
#[must_use]
pub fn dcg_at_k(relevance_scores: &[f32], k: usize) -> f32 {
    let upper = k.min(relevance_scores.len());
    let mut dcg = 0.0_f32;
    for i in 0..upper {
        dcg += relevance_scores[i] / ((i as f32 + 2.0).log2());
    }
    dcg
}

/// IDCG@k = DCG@k for relevance scores sorted descending.
#[must_use]
pub fn idcg_at_k(relevance_scores: &[f32], k: usize) -> f32 {
    let mut sorted = relevance_scores.to_vec();
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    dcg_at_k(&sorted, k)
}

/// NDCG@k = DCG@k / IDCG@k (or 1.0 if IDCG@k == 0 — vacuous).
#[must_use]
pub fn ndcg_at_k(relevance_scores: &[f32], k: usize) -> f32 {
    let idcg = idcg_at_k(relevance_scores, k);
    if idcg == 0.0 {
        // Convention: if all relevance is 0, NDCG is defined as 0
        // (perfect-ranking edge case is captured by the NDCG-perfect
        // gate; here we conservatively report 0.0).
        return 0.0;
    }
    dcg_at_k(relevance_scores, k) / idcg
}

// -----------------------------------------------------------------------------
// Verdict 1: RANK-001 — hit@k binary.
// -----------------------------------------------------------------------------

#[must_use]
pub fn verdict_from_hit_at_k_binary(hit_value: u32) -> RankVerdict {
    if hit_value == 0 || hit_value == 1 {
        RankVerdict::Pass
    } else {
        RankVerdict::Fail
    }
}

// -----------------------------------------------------------------------------
// Verdict 2: RANK-002 — MRR ∈ [0, 1].
// -----------------------------------------------------------------------------

#[must_use]
pub fn verdict_from_mrr_bounded(mrr_value: f32) -> RankVerdict {
    if !mrr_value.is_finite() {
        return RankVerdict::Fail;
    }
    if (AC_RANK_BOUND_LOWER..=AC_RANK_BOUND_UPPER).contains(&mrr_value) {
        RankVerdict::Pass
    } else {
        RankVerdict::Fail
    }
}

// -----------------------------------------------------------------------------
// Verdict 3: RANK-003 — NDCG = 1 for perfect ranking.
// -----------------------------------------------------------------------------

#[must_use]
pub fn verdict_from_ndcg_perfect_ranking(ndcg: f32) -> RankVerdict {
    if !ndcg.is_finite() {
        return RankVerdict::Fail;
    }
    if (ndcg - 1.0).abs() < AC_RANK_003_PERFECT_EPS {
        RankVerdict::Pass
    } else {
        RankVerdict::Fail
    }
}

// -----------------------------------------------------------------------------
// Verdict 4: RANK-004 — NDCG ∈ [0, 1].
// -----------------------------------------------------------------------------

#[must_use]
pub fn verdict_from_ndcg_bounded(ndcg: f32) -> RankVerdict {
    if !ndcg.is_finite() {
        return RankVerdict::Fail;
    }
    if (AC_RANK_BOUND_LOWER..=AC_RANK_BOUND_UPPER).contains(&ndcg) {
        RankVerdict::Pass
    } else {
        RankVerdict::Fail
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Section 1: Provenance pins.
    // -------------------------------------------------------------------------
    #[test]
    fn provenance_bound_lower_zero() {
        assert_eq!(AC_RANK_BOUND_LOWER, 0.0);
    }

    #[test]
    fn provenance_bound_upper_one() {
        assert_eq!(AC_RANK_BOUND_UPPER, 1.0);
    }

    #[test]
    fn provenance_perfect_eps_1e_5() {
        assert_eq!(AC_RANK_003_PERFECT_EPS, 1e-5);
    }

    // -------------------------------------------------------------------------
    // Section 2: Domain — reference helpers.
    // -------------------------------------------------------------------------
    #[test]
    fn domain_hit_at_k_finds_relevant_in_top() {
        let r = vec![false, true, false, false];
        assert_eq!(hit_at_k(&r, 2), 1);
        assert_eq!(hit_at_k(&r, 4), 1);
    }

    #[test]
    fn domain_hit_at_k_misses_when_not_in_top() {
        let r = vec![false, false, false, true];
        assert_eq!(hit_at_k(&r, 2), 0);
        assert_eq!(hit_at_k(&r, 4), 1);
    }

    #[test]
    fn domain_reciprocal_rank_first_relevant() {
        let r = vec![true, false, false];
        assert!((reciprocal_rank(&r) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn domain_reciprocal_rank_third_relevant() {
        let r = vec![false, false, true];
        assert!((reciprocal_rank(&r) - (1.0 / 3.0)).abs() < 1e-6);
    }

    #[test]
    fn domain_reciprocal_rank_none_relevant_returns_zero() {
        let r = vec![false; 5];
        assert_eq!(reciprocal_rank(&r), 0.0);
    }

    #[test]
    fn domain_mrr_average_of_three_queries() {
        // RR = [1.0, 0.5, 0.333...] ⇒ MRR ≈ 0.611.
        let rr = vec![1.0_f32, 0.5, 1.0 / 3.0];
        let m = mrr(&rr);
        assert!((m - 0.611_111).abs() < 1e-4);
    }

    #[test]
    fn domain_dcg_at_k_basic() {
        // rel = [3, 2, 1] ⇒ DCG@3 = 3/log2(2) + 2/log2(3) + 1/log2(4)
        //                       = 3 + 2/1.585 + 1/2 = 3 + 1.262 + 0.5 = 4.762.
        let rel = vec![3.0_f32, 2.0, 1.0];
        let dcg = dcg_at_k(&rel, 3);
        assert!((dcg - 4.762).abs() < 0.01, "dcg={dcg}");
    }

    #[test]
    fn domain_ndcg_perfect_when_sorted() {
        let rel = vec![5.0_f32, 3.0, 1.0]; // already sorted desc
        let ndcg = ndcg_at_k(&rel, 3);
        assert!((ndcg - 1.0).abs() < 1e-5);
    }

    #[test]
    fn domain_ndcg_less_than_one_when_unsorted() {
        let rel = vec![1.0_f32, 5.0, 3.0]; // suboptimal order
        let ndcg = ndcg_at_k(&rel, 3);
        assert!(ndcg < 1.0);
        assert!(ndcg > 0.0);
    }

    // -------------------------------------------------------------------------
    // Section 3: RANK-001 — hit@k binary.
    // -------------------------------------------------------------------------
    #[test]
    fn rank001_pass_zero() {
        assert_eq!(verdict_from_hit_at_k_binary(0), RankVerdict::Pass);
    }

    #[test]
    fn rank001_pass_one() {
        assert_eq!(verdict_from_hit_at_k_binary(1), RankVerdict::Pass);
    }

    #[test]
    fn rank001_fail_above_one() {
        // Bug: hit@k returned multi-relevant count instead of binary.
        assert_eq!(verdict_from_hit_at_k_binary(2), RankVerdict::Fail);
    }

    #[test]
    fn rank001_fail_large_value() {
        assert_eq!(verdict_from_hit_at_k_binary(100), RankVerdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 4: RANK-002 — MRR ∈ [0, 1].
    // -------------------------------------------------------------------------
    #[test]
    fn rank002_pass_zero() {
        assert_eq!(verdict_from_mrr_bounded(0.0), RankVerdict::Pass);
    }

    #[test]
    fn rank002_pass_one() {
        assert_eq!(verdict_from_mrr_bounded(1.0), RankVerdict::Pass);
    }

    #[test]
    fn rank002_pass_half() {
        assert_eq!(verdict_from_mrr_bounded(0.5), RankVerdict::Pass);
    }

    #[test]
    fn rank002_fail_above_one() {
        // Reciprocal rank exceeded 1 due to /0 bug.
        assert_eq!(verdict_from_mrr_bounded(1.5), RankVerdict::Fail);
    }

    #[test]
    fn rank002_fail_negative() {
        assert_eq!(verdict_from_mrr_bounded(-0.1), RankVerdict::Fail);
    }

    #[test]
    fn rank002_fail_nan() {
        assert_eq!(verdict_from_mrr_bounded(f32::NAN), RankVerdict::Fail);
    }

    #[test]
    fn rank002_fail_inf() {
        assert_eq!(verdict_from_mrr_bounded(f32::INFINITY), RankVerdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 5: RANK-003 — NDCG perfect ranking.
    // -------------------------------------------------------------------------
    #[test]
    fn rank003_pass_exact_one() {
        assert_eq!(
            verdict_from_ndcg_perfect_ranking(1.0),
            RankVerdict::Pass
        );
    }

    #[test]
    fn rank003_pass_within_eps() {
        assert_eq!(
            verdict_from_ndcg_perfect_ranking(0.999_999),
            RankVerdict::Pass
        );
    }

    #[test]
    fn rank003_fail_just_below() {
        // Bug: log base error → NDCG slightly below 1 even on perfect.
        assert_eq!(
            verdict_from_ndcg_perfect_ranking(0.99),
            RankVerdict::Fail
        );
    }

    #[test]
    fn rank003_fail_above_one() {
        // Bug: IDCG sort order wrong → NDCG > 1.
        assert_eq!(
            verdict_from_ndcg_perfect_ranking(1.05),
            RankVerdict::Fail
        );
    }

    #[test]
    fn rank003_fail_zero() {
        assert_eq!(
            verdict_from_ndcg_perfect_ranking(0.0),
            RankVerdict::Fail
        );
    }

    #[test]
    fn rank003_fail_nan() {
        assert_eq!(
            verdict_from_ndcg_perfect_ranking(f32::NAN),
            RankVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 6: RANK-004 — NDCG ∈ [0, 1].
    // -------------------------------------------------------------------------
    #[test]
    fn rank004_pass_typical() {
        assert_eq!(verdict_from_ndcg_bounded(0.85), RankVerdict::Pass);
    }

    #[test]
    fn rank004_pass_at_bounds() {
        assert_eq!(verdict_from_ndcg_bounded(0.0), RankVerdict::Pass);
        assert_eq!(verdict_from_ndcg_bounded(1.0), RankVerdict::Pass);
    }

    #[test]
    fn rank004_fail_above_one() {
        // Contract example: "NDCG exceeds 1.0 due to IDCG computation
        // using wrong sort order".
        assert_eq!(verdict_from_ndcg_bounded(1.5), RankVerdict::Fail);
    }

    #[test]
    fn rank004_fail_negative() {
        assert_eq!(verdict_from_ndcg_bounded(-0.1), RankVerdict::Fail);
    }

    #[test]
    fn rank004_fail_nan() {
        // Division by zero when all relevance = 0.
        assert_eq!(verdict_from_ndcg_bounded(f32::NAN), RankVerdict::Fail);
    }

    #[test]
    fn rank004_fail_inf() {
        assert_eq!(verdict_from_ndcg_bounded(f32::INFINITY), RankVerdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 7: Sweep — bounds.
    // -------------------------------------------------------------------------
    #[test]
    fn sweep_mrr_band() {
        let test_cases = [
            (-0.1_f32, RankVerdict::Fail),
            (0.0, RankVerdict::Pass),
            (0.5, RankVerdict::Pass),
            (1.0, RankVerdict::Pass),
            (1.0001, RankVerdict::Fail),
        ];
        for (m, expected) in test_cases {
            let v = verdict_from_mrr_bounded(m);
            assert_eq!(v, expected, "m={m}");
        }
    }

    #[test]
    fn sweep_hit_at_k_band() {
        for h in 0..=1_u32 {
            assert_eq!(
                verdict_from_hit_at_k_binary(h),
                RankVerdict::Pass,
                "h={h}"
            );
        }
        for h in 2..=10_u32 {
            assert_eq!(
                verdict_from_hit_at_k_binary(h),
                RankVerdict::Fail,
                "h={h}"
            );
        }
    }

    // -------------------------------------------------------------------------
    // Section 8: Realistic — contract regression scenarios.
    // -------------------------------------------------------------------------
    #[test]
    fn realistic_hit_returns_count_not_binary_caught() {
        // RANK-001 if_fails: "hit@k returning fractional or unbounded
        // value" — bug returns count of relevant items.
        assert_eq!(verdict_from_hit_at_k_binary(3), RankVerdict::Fail);
    }

    #[test]
    fn realistic_division_by_zero_caught() {
        // RANK-002 if_fails: "Division error or reciprocal rank
        // exceeding 1".
        assert_eq!(verdict_from_mrr_bounded(2.0), RankVerdict::Fail);
    }

    #[test]
    fn realistic_log_base_error_caught() {
        // RANK-003 if_fails: "DCG/IDCG computation mismatch or log
        // base error". With log10 instead of log2, NDCG perfect would
        // not be exactly 1.
        assert_eq!(
            verdict_from_ndcg_perfect_ranking(0.95),
            RankVerdict::Fail
        );
    }

    #[test]
    fn realistic_idcg_wrong_sort_caught() {
        // RANK-004 if_fails: "IDCG computed incorrectly" → NDCG > 1.
        assert_eq!(verdict_from_ndcg_bounded(1.2), RankVerdict::Fail);
    }

    #[test]
    fn realistic_full_metric_pipeline() {
        // 4 queries, each with 5-position ranking and a relevant item.
        let queries_relevance = vec![
            vec![true, false, false, false, false],   // perfect rank
            vec![false, true, false, false, false],   // RR = 0.5
            vec![false, false, true, false, false],   // RR = 0.333
            vec![false, false, false, false, false],  // none ⇒ RR = 0
        ];
        let rrs: Vec<f32> = queries_relevance
            .iter()
            .map(|r| reciprocal_rank(r))
            .collect();
        let m = mrr(&rrs);
        assert_eq!(verdict_from_mrr_bounded(m), RankVerdict::Pass);
        // hit@1 binary check on first query.
        let h1 = hit_at_k(&queries_relevance[0], 1);
        assert_eq!(verdict_from_hit_at_k_binary(h1), RankVerdict::Pass);
        let h4 = hit_at_k(&queries_relevance[3], 5); // none relevant
        assert_eq!(verdict_from_hit_at_k_binary(h4), RankVerdict::Pass);

        // NDCG perfect ranking at k=3.
        let perfect_relevance = vec![3.0_f32, 2.0, 1.0];
        let ndcg = ndcg_at_k(&perfect_relevance, 3);
        assert_eq!(
            verdict_from_ndcg_perfect_ranking(ndcg),
            RankVerdict::Pass
        );
        assert_eq!(verdict_from_ndcg_bounded(ndcg), RankVerdict::Pass);

        // NDCG suboptimal still bounded.
        let suboptimal = vec![1.0_f32, 3.0, 2.0];
        let ndcg2 = ndcg_at_k(&suboptimal, 3);
        assert!(ndcg2 < 1.0);
        assert_eq!(verdict_from_ndcg_bounded(ndcg2), RankVerdict::Pass);
    }
}
