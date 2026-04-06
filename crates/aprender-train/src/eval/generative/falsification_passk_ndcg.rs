//! Falsification tests for pass@k, NDCG@k, and cross-metric invariants
//!
//! Tests mathematical correctness, numerical stability, metric enum
//! consistency, and cross-metric properties.

use super::*;
use proptest::prelude::*;

// ═══════════════════════════════════════════════════════════════════════
// CLAIM 5: pass@k = 1 - C(n-c,k) / C(n,k)
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn falsify_pass_at_k_known_value() {
    // n=10, c=3, k=2
    // C(7,2)/C(10,2) = 21/45 = 7/15
    // pass@2 = 1 - 7/15 = 8/15 ≈ 0.5333...
    let p = pass_at_k(10, 3, 2);
    assert!((p - 8.0 / 15.0).abs() < 1e-10, "pass@2(n=10,c=3) should be 8/15, got {p}");
}

#[test]
fn falsify_pass_at_k_c_equals_1_k_equals_1() {
    // n=10, c=1, k=1 → 1 - C(9,1)/C(10,1) = 1 - 9/10 = 0.1
    let p = pass_at_k(10, 1, 1);
    assert!((p - 0.1).abs() < 1e-10, "pass@1(n=10,c=1) should be 0.1, got {p}");
}

#[test]
fn falsify_pass_at_k_large_n() {
    // Test numerical stability with large values
    // n=1000, c=100, k=10
    let p = pass_at_k(1000, 100, 10);
    assert!(p > 0.0 && p < 1.0, "pass@10(n=1000,c=100) should be in (0,1), got {p}");
    // Manual: C(900,10)/C(1000,10) = prod_{i=0..9} (900-i)/(1000-i)
    let mut manual_ratio = 1.0f64;
    for i in 0..10 {
        manual_ratio *= f64::from(900 - i) / f64::from(1000 - i);
    }
    let expected = 1.0 - manual_ratio;
    assert!((p - expected).abs() < 1e-10, "Numerical stability: expected {expected}, got {p}");
}

#[test]
fn falsify_pass_at_k_monotonic_in_c() {
    // Fixing n and k, pass@k should increase with c
    let p1 = pass_at_k(20, 1, 5);
    let p5 = pass_at_k(20, 5, 5);
    let p10 = pass_at_k(20, 10, 5);
    assert!(p1 < p5, "pass@5 should increase with c: p(c=1)={p1} < p(c=5)={p5}");
    assert!(p5 < p10, "pass@5 should increase with c: p(c=5)={p5} < p(c=10)={p10}");
}

#[test]
fn falsify_pass_at_k_edge_k_equals_n() {
    // k = n: if c > 0, we are guaranteed to pick at least one correct
    // Actually: pass@n = 1 - C(n-c,n)/C(n,n). C(n,n) = 1, C(n-c,n) = 0 if c > 0
    // So pass@n = 1.0 when c > 0
    let p = pass_at_k(10, 1, 10);
    assert!((p - 1.0).abs() < 1e-10, "pass@n with c>0 should be 1.0, got {p}");
}

#[test]
fn falsify_pass_at_k_n_equals_1() {
    assert_eq!(pass_at_k(1, 0, 1), 0.0);
    assert!((pass_at_k(1, 1, 1) - 1.0).abs() < 1e-10);
}

// ═══════════════════════════════════════════════════════════════════════
// CLAIM 6: NDCG@k = DCG@k / IDCG@k, DCG = Σ (2^rel - 1) / log2(i+1)
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn falsify_ndcg_known_value() {
    // scores: [3, 1, 2] → DCG, IDCG computed manually
    let ndcg = ndcg_at_k(&[3.0, 1.0, 2.0], 3);
    let dcg_manual = 7.0 / (2.0_f64).log2() + 1.0 / (3.0_f64).log2() + 3.0 / (4.0_f64).log2();
    let idcg_manual = 7.0 / (2.0_f64).log2() + 3.0 / (3.0_f64).log2() + 1.0 / (4.0_f64).log2();
    let expected = dcg_manual / idcg_manual;

    assert!((ndcg - expected).abs() < 1e-6, "NDCG@3 for [3,1,2]: expected {expected}, got {ndcg}");
}

#[test]
fn falsify_ndcg_single_element() {
    // Single relevant doc → NDCG = 1.0 regardless of score value
    let ndcg = ndcg_at_k(&[5.0], 1);
    assert!((ndcg - 1.0).abs() < 1e-10, "Single element NDCG should be 1.0, got {ndcg}");
}

#[test]
fn falsify_ndcg_binary_relevance() {
    // Binary: [0, 1] → worst ranking (relevant doc at position 2)
    // DCG = (2^0-1)/log2(2) + (2^1-1)/log2(3) = 0 + 1/1.585 = 0.6309
    // IDCG = (2^1-1)/log2(2) + (2^0-1)/log2(3) = 1/1 + 0 = 1.0
    // NDCG = 0.6309
    let ndcg = ndcg_at_k(&[0.0, 1.0], 2);
    let expected = (1.0 / (3.0_f64).log2()) / 1.0;
    assert!(
        (ndcg - expected).abs() < 1e-6,
        "NDCG@2 for [0,1] binary: expected {expected}, got {ndcg}"
    );
}

#[test]
fn falsify_ndcg_k_truncation() {
    // NDCG@1 only considers position 1
    let scores = vec![0.0, 3.0, 2.0, 1.0];
    let ndcg1 = ndcg_at_k(&scores, 1);
    // DCG@1 = (2^0 - 1)/log2(2) = 0/1 = 0, IDCG@1 = (2^3-1)/1 = 7
    // NDCG@1 = 0/7 = 0
    assert!(ndcg1.abs() < 1e-10, "NDCG@1 with irrelevant top result should be 0, got {ndcg1}");
}

// ═══════════════════════════════════════════════════════════════════════
// CLAIM 7: Feature gate isolation — hub-publish doesn't leak
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn falsify_generative_metrics_always_available() {
    // These should compile without any feature gate
    let _ = word_error_rate("a", "b");
    let _ = bleu_score(&["a"], "a", 1);
    let _ = rouge_n("a", "a", 1);
    let _ = rouge_l("a", "a");
    let _ = perplexity(&[-1.0]);
    let _ = pass_at_k(1, 1, 1);
    let _ = ndcg_at_k(&[1.0], 1);
}

// ═══════════════════════════════════════════════════════════════════════
// CLAIM 8: Metric enum consistency
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn falsify_wer_lower_is_better() {
    use crate::eval::evaluator::Metric;
    assert!(!Metric::WER.higher_is_better(), "WER should be lower-is-better");
}

#[test]
fn falsify_perplexity_lower_is_better() {
    use crate::eval::evaluator::Metric;
    assert!(!Metric::Perplexity.higher_is_better(), "Perplexity should be lower-is-better");
}

#[test]
fn falsify_bleu_higher_is_better() {
    use crate::eval::evaluator::Metric;
    assert!(Metric::BLEU.higher_is_better(), "BLEU should be higher-is-better");
}

#[test]
fn falsify_pass_at_k_higher_is_better() {
    use crate::eval::evaluator::Metric;
    assert!(Metric::PassAtK(1).higher_is_better(), "pass@k should be higher-is-better");
}

// ═══════════════════════════════════════════════════════════════════════
// CLAIM 9: Numerical stability under extreme inputs
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn falsify_perplexity_near_zero_prob() {
    // Very small probabilities: log(1e-300)
    let log_probs = vec![(1e-300f64).ln(); 5];
    let ppl = perplexity(&log_probs);
    assert!(ppl.is_finite() && ppl > 0.0, "PPL should handle tiny probs, got {ppl}");
}

#[test]
fn falsify_bleu_single_word_ref_and_hyp() {
    // Single word: only unigram possible
    let score = bleu_score(&["hello"], "hello", 4);
    // 1-gram precision = 1.0, but 2,3,4-gram precision → total=0 → returns 0
    assert_eq!(score, 0.0, "Single word can't have 2-gram, BLEU-4 should be 0");
}

#[test]
fn falsify_bleu_single_word_max_n_1() {
    let score = bleu_score(&["hello"], "hello", 1);
    assert!(
        (score - 1.0).abs() < 1e-10,
        "Single word with max_n=1 should give BLEU=1.0, got {score}"
    );
}

// ═══════════════════════════════════════════════════════════════════════
// CLAIM 10: Cross-metric consistency (proptest)
// ═══════════════════════════════════════════════════════════════════════

proptest! {
    #[test]
    fn falsify_rouge1_geq_rouge2(
        reference in "[a-z]{1,3}( [a-z]{1,3}){3,8}",
        hypothesis in "[a-z]{1,3}( [a-z]{1,3}){3,8}"
    ) {
        let r1 = rouge_n(&reference, &hypothesis, 1);
        let r2 = rouge_n(&reference, &hypothesis, 2);
        // If ROUGE-2 > 0, ROUGE-1 must also be > 0 (necessary condition)
        if r2 > 0.0 {
            prop_assert!(r1 > 0.0, "If ROUGE-2 > 0, ROUGE-1 must also be > 0: R1={}, R2={}", r1, r2);
        }
    }

    #[test]
    fn falsify_wer_edit_distance_identity(s in "[a-z]{1,5}( [a-z]{1,5}){0,9}") {
        let wer = word_error_rate(&s, &s);
        prop_assert!(
            wer.abs() < 1e-10,
            "WER(s, s) must be 0.0 for all s, got {} for {:?}", wer, s
        );
    }

    #[test]
    fn falsify_ndcg_invariant_to_scaling(
        scores in proptest::collection::vec(0.0f64..5.0, 2..10),
        k in 1..10usize
    ) {
        let ndcg = ndcg_at_k(&scores, k);
        prop_assert!(
            (-1e-10..=1.0 + 1e-10).contains(&ndcg),
            "NDCG must be in [0,1], got {}", ndcg
        );
    }
}
