//! Property tests for generative AI evaluation metrics

use super::*;
use proptest::prelude::*;

// ─── WER tests ───────────────────────────────────────────────────────

#[test]
fn test_wer_identical_strings() {
    assert_eq!(word_error_rate("hello world", "hello world"), 0.0);
}

#[test]
fn test_wer_completely_wrong() {
    // 2 substitutions out of 2 words = 1.0
    assert!((word_error_rate("hello world", "foo bar") - 1.0).abs() < 1e-10);
}

#[test]
fn test_wer_empty_both() {
    assert_eq!(word_error_rate("", ""), 0.0);
}

#[test]
fn test_wer_empty_reference() {
    assert!(word_error_rate("", "hello").is_infinite());
}

#[test]
fn test_wer_empty_hypothesis() {
    // All words deleted: WER = N/N = 1.0
    assert!((word_error_rate("hello world", "") - 1.0).abs() < 1e-10);
}

#[test]
fn test_wer_insertion() {
    // "the cat" vs "the big cat" → 1 insertion, WER = 1/2 = 0.5
    assert!((word_error_rate("the cat", "the big cat") - 0.5).abs() < 1e-10);
}

#[test]
fn test_wer_deletion() {
    // "the big cat" vs "the cat" → 1 deletion, WER = 1/3
    let wer = word_error_rate("the big cat", "the cat");
    assert!((wer - 1.0 / 3.0).abs() < 1e-10);
}

proptest! {
    #[test]
    fn prop_wer_identical_is_zero(s in "[a-z ]{1,50}") {
        let wer = word_error_rate(&s, &s);
        prop_assert!((wer - 0.0).abs() < 1e-10, "WER of identical strings should be 0, got {}", wer);
    }

    #[test]
    fn prop_wer_non_negative(
        reference in "[a-z]{1,5}( [a-z]{1,5}){0,9}",
        hypothesis in "[a-z]{1,5}( [a-z]{1,5}){0,9}"
    ) {
        let wer = word_error_rate(&reference, &hypothesis);
        prop_assert!(wer >= 0.0, "WER must be non-negative, got {}", wer);
    }
}

// ─── RTFx tests ──────────────────────────────────────────────────────

#[test]
fn test_rtfx_basic() {
    // 10s audio in 0.1s = 100x real-time
    assert!((real_time_factor_inverse(0.1, 10.0) - 100.0).abs() < 1e-10);
}

#[test]
fn test_rtfx_zero_processing() {
    assert_eq!(real_time_factor_inverse(0.0, 10.0), 0.0);
}

#[test]
fn test_rtfx_negative_processing() {
    assert_eq!(real_time_factor_inverse(-1.0, 10.0), 0.0);
}

// ─── BLEU tests ──────────────────────────────────────────────────────

#[test]
fn test_bleu_identical() {
    let score = bleu_score(&["the cat sat on the mat"], "the cat sat on the mat", 4);
    assert!(score > 0.99, "BLEU of identical strings should be ~1.0, got {score}");
}

#[test]
fn test_bleu_empty_hypothesis() {
    assert_eq!(bleu_score(&["the cat"], "", 4), 0.0);
}

#[test]
fn test_bleu_empty_references() {
    assert_eq!(bleu_score(&[], "the cat", 4), 0.0);
}

#[test]
fn test_bleu_no_overlap() {
    let score = bleu_score(&["the cat sat"], "a dog ran", 4);
    assert_eq!(score, 0.0);
}

#[test]
fn test_bleu_partial_match() {
    // Hypothesis shares most unigrams/bigrams with reference
    let score = bleu_score(
        &["the cat sat on the mat by the door"],
        "the cat sat on the mat by the window",
        4,
    );
    assert!(score > 0.0 && score < 1.0, "Expected BLEU in (0, 1), got {score}");
}

proptest! {
    #[test]
    fn prop_bleu_bounds(
        reference in "[a-z]{1,5}( [a-z]{1,5}){2,9}",
        hypothesis in "[a-z]{1,5}( [a-z]{1,5}){2,9}"
    ) {
        let score = bleu_score(&[&reference], &hypothesis, 4);
        prop_assert!((0.0..=1.0).contains(&score), "BLEU must be in [0,1], got {}", score);
    }
}

// ─── ROUGE tests ─────────────────────────────────────────────────────

#[test]
fn test_rouge1_identical() {
    let score = rouge_n("the cat sat on the mat", "the cat sat on the mat", 1);
    assert!((score - 1.0).abs() < 1e-10);
}

#[test]
fn test_rouge2_identical() {
    let score = rouge_n("the cat sat on the mat", "the cat sat on the mat", 2);
    assert!((score - 1.0).abs() < 1e-10);
}

#[test]
fn test_rouge_l_identical() {
    let score = rouge_l("the cat sat on the mat", "the cat sat on the mat");
    assert!((score - 1.0).abs() < 1e-10);
}

#[test]
fn test_rouge_empty() {
    assert_eq!(rouge_n("", "hello", 1), 0.0);
    assert_eq!(rouge_n("hello", "", 1), 0.0);
    assert_eq!(rouge_l("", "hello"), 0.0);
    assert_eq!(rouge_l("hello", ""), 0.0);
}

#[test]
fn test_rouge_no_overlap() {
    assert_eq!(rouge_n("aaa bbb", "ccc ddd", 1), 0.0);
    assert_eq!(rouge_l("aaa bbb", "ccc ddd"), 0.0);
}

proptest! {
    #[test]
    fn prop_rouge_bounds(
        reference in "[a-z]{1,4}( [a-z]{1,4}){1,8}",
        hypothesis in "[a-z]{1,4}( [a-z]{1,4}){1,8}"
    ) {
        let r1 = rouge_n(&reference, &hypothesis, 1);
        prop_assert!((0.0..=1.0).contains(&r1), "ROUGE-1 must be in [0,1], got {}", r1);

        let rl = rouge_l(&reference, &hypothesis);
        prop_assert!((0.0..=1.0).contains(&rl), "ROUGE-L must be in [0,1], got {}", rl);
    }
}

// ─── Perplexity tests ────────────────────────────────────────────────

#[test]
fn test_perplexity_uniform() {
    // log(1/10) for each of 10 tokens → perplexity = 10
    let log_probs: Vec<f64> = vec![(0.1f64).ln(); 10];
    let ppl = perplexity(&log_probs);
    assert!((ppl - 10.0).abs() < 1e-6);
}

#[test]
fn test_perplexity_perfect() {
    // log(1.0) = 0 for all → perplexity = 1.0
    let log_probs = vec![0.0; 5];
    let ppl = perplexity(&log_probs);
    assert!((ppl - 1.0).abs() < 1e-10);
}

#[test]
fn test_perplexity_empty() {
    assert!(perplexity(&[]).is_infinite());
}

proptest! {
    #[test]
    fn prop_perplexity_at_least_one(log_probs in proptest::collection::vec(-10.0f64..0.0, 1..50)) {
        let ppl = perplexity(&log_probs);
        prop_assert!(ppl >= 1.0, "Perplexity must be >= 1.0 for valid log-probs, got {}", ppl);
    }
}

// ─── pass@k tests ────────────────────────────────────────────────────

#[test]
fn test_pass_at_k_all_correct() {
    assert!((pass_at_k(10, 10, 1) - 1.0).abs() < 1e-10);
}

#[test]
fn test_pass_at_k_none_correct() {
    assert_eq!(pass_at_k(10, 0, 1), 0.0);
}

#[test]
fn test_pass_at_k_half_correct_k1() {
    // n=10, c=5, k=1 → 1 - C(5,1)/C(10,1) = 1 - 5/10 = 0.5
    assert!((pass_at_k(10, 5, 1) - 0.5).abs() < 1e-10);
}

#[test]
fn test_pass_at_k_monotonic_in_k() {
    let p1 = pass_at_k(100, 10, 1);
    let p10 = pass_at_k(100, 10, 10);
    let p100 = pass_at_k(100, 10, 100);
    assert!(p1 <= p10, "pass@1 <= pass@10: {p1} <= {p10}");
    assert!(p10 <= p100, "pass@10 <= pass@100: {p10} <= {p100}");
}

proptest! {
    #[test]
    fn prop_pass_at_k_monotonic(n in 10..100usize, c in 1..10usize) {
        let c = c.min(n);
        let p1 = pass_at_k(n, c, 1);
        let p5 = pass_at_k(n, c, 5);
        let p10 = pass_at_k(n, c, 10.min(n));
        prop_assert!(p1 <= p5 + 1e-10, "pass@1 <= pass@5: {} <= {}", p1, p5);
        prop_assert!(p5 <= p10 + 1e-10, "pass@5 <= pass@10: {} <= {}", p5, p10);
    }

    #[test]
    fn prop_pass_at_k_bounds(n in 1..100usize, c in 0..100usize, k in 1..100usize) {
        let c = c.min(n);
        let p = pass_at_k(n, c, k);
        prop_assert!((0.0..=1.0).contains(&p), "pass@k must be in [0,1], got {}", p);
    }
}

// ─── NDCG@k tests ───────────────────────────────────────────────────

#[test]
fn test_ndcg_perfect_ranking() {
    let scores = vec![3.0, 2.0, 1.0, 0.0];
    assert!((ndcg_at_k(&scores, 4) - 1.0).abs() < 1e-10);
}

#[test]
fn test_ndcg_worst_ranking() {
    let scores = vec![0.0, 0.0, 0.0, 3.0];
    let ndcg = ndcg_at_k(&scores, 4);
    assert!(ndcg < 1.0 && ndcg > 0.0);
}

#[test]
fn test_ndcg_all_zero() {
    let scores = vec![0.0, 0.0, 0.0];
    assert_eq!(ndcg_at_k(&scores, 3), 0.0);
}

#[test]
fn test_ndcg_empty() {
    assert_eq!(ndcg_at_k(&[], 5), 0.0);
}

#[test]
fn test_ndcg_k_zero() {
    assert_eq!(ndcg_at_k(&[1.0, 2.0], 0), 0.0);
}

#[test]
fn test_ndcg_k_larger_than_list() {
    let scores = vec![3.0, 2.0];
    let ndcg = ndcg_at_k(&scores, 10);
    // Should use min(k, len) = 2
    assert!(ndcg > 0.0);
}

proptest! {
    #[test]
    fn prop_ndcg_bounds(scores in proptest::collection::vec(0.0f64..5.0, 1..20), k in 1..20usize) {
        let ndcg = ndcg_at_k(&scores, k);
        prop_assert!((0.0..=1.0 + 1e-10).contains(&ndcg), "NDCG must be in [0,1], got {}", ndcg);
    }

    #[test]
    fn prop_ndcg_perfect_is_one(len in 1..10usize) {
        // Descending scores = perfect ranking
        let scores: Vec<f64> = (0..len).rev().map(|i| i as f64).collect();
        if scores.iter().any(|&s| s > 0.0) {
            let ndcg = ndcg_at_k(&scores, len);
            prop_assert!((ndcg - 1.0).abs() < 1e-10, "Perfect ranking NDCG should be 1.0, got {}", ndcg);
        }
    }
}
