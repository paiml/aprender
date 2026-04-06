//! Falsification tests for ASR and BLEU metrics
//!
//! Tests WER, RTFx, and BLEU score mathematical correctness.

use super::*;

// ═══════════════════════════════════════════════════════════════════════
// CLAIM 1: WER = edit_distance(ref_words, hyp_words) / len(ref_words)
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn falsify_wer_known_substitution() {
    // "a b c" → "a x c" = 1 substitution / 3 words
    let wer = word_error_rate("a b c", "a x c");
    assert!((wer - 1.0 / 3.0).abs() < 1e-10, "1 substitution in 3 words → WER = 1/3, got {wer}");
}

#[test]
fn falsify_wer_known_two_insertions() {
    // "a" → "x a y" = 2 operations (insert x before, insert y after) / 1 word
    let wer = word_error_rate("a", "x a y");
    assert!((wer - 2.0).abs() < 1e-10, "2 insertions in 1-word ref → WER = 2.0, got {wer}");
}

#[test]
fn falsify_wer_exceeds_one() {
    // WER CAN exceed 1.0 — spec says "Can exceed 1.0 when hypothesis is much longer"
    let wer = word_error_rate("a", "b c d e f");
    assert!(wer > 1.0, "WER should exceed 1.0 for heavily inserted hyp, got {wer}");
}

#[test]
fn falsify_wer_asymmetry() {
    // WER is NOT symmetric — swapping ref/hyp changes the denominator
    let wer_ab = word_error_rate("a b c", "a b c d e");
    let wer_ba = word_error_rate("a b c d e", "a b c");
    assert!(
        (wer_ab - wer_ba).abs() > 1e-10,
        "WER must be asymmetric: WER(abc, abcde)={wer_ab} vs WER(abcde, abc)={wer_ba}"
    );
}

#[test]
fn falsify_wer_triangle_inequality() {
    // Edit distance satisfies triangle inequality: d(a,c) ≤ d(a,b) + d(b,c)
    // But WER divides by |ref|, so triangle inequality on WER itself is NOT guaranteed.
    // This test verifies the raw edit distance does satisfy it.
    let a = "the quick brown fox";
    let b = "a quick brown dog";
    let c = "the slow red cat";

    let d_ab = word_error_rate(a, b) * 4.0; // un-normalize: 4 words in ref
    let d_bc = word_error_rate(b, c) * 4.0; // un-normalize: 4 words in ref
    let d_ac = word_error_rate(a, c) * 4.0; // un-normalize: 4 words in ref

    // Triangle inequality on edit distance (not WER)
    assert!(
        d_ac <= d_ab + d_bc + 1e-10,
        "Triangle inequality violated: d(a,c)={d_ac} > d(a,b)={d_ab} + d(b,c)={d_bc}"
    );
}

#[test]
fn falsify_wer_whitespace_only_reference() {
    // A reference of only whitespace → 0 words → same as empty reference
    let wer = word_error_rate("   ", "hello");
    assert!(wer.is_infinite(), "Whitespace-only ref has 0 words → WER should be inf, got {wer}");
}

#[test]
fn falsify_wer_whitespace_only_hypothesis() {
    // All words deleted
    let wer = word_error_rate("hello world", "   ");
    assert!((wer - 1.0).abs() < 1e-10, "Whitespace-only hyp → all deletions → WER=1.0, got {wer}");
}

// ═══════════════════════════════════════════════════════════════════════
// CLAIM 2: BLEU implements Papineni et al. (2002) correctly
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn falsify_bleu_brevity_penalty_applied() {
    // Short hypothesis should be penalized by brevity penalty
    let short = bleu_score(&["a b c d e f g h i j"], "a b c", 1);
    let full = bleu_score(&["a b c d e f g h i j"], "a b c d e f g h i j", 1);
    assert!(
        short < full,
        "Brevity penalty: short hyp ({short}) should score less than full ({full})"
    );
}

#[test]
fn falsify_bleu_clipped_precision() {
    // "the the the the" vs ref "the cat" → unigram precision should be clipped to 1/4
    // (only 1 "the" in ref, but 4 in hyp → clipped to min(4,1) = 1)
    let score = bleu_score(&["the cat"], "the the the the", 1);
    // precision = 1/4, BP = exp(1 - 2/4) = exp(-0.5) ≈ 0.6065
    // BLEU = 0.6065 * 0.25 = 0.1516
    assert!(score < 0.3, "Clipped precision should limit repeated words, got {score}");
    assert!(score > 0.0, "Some unigrams match, should be > 0, got {score}");
}

#[test]
fn falsify_bleu_multi_reference() {
    // Multiple references: max count per n-gram across refs
    let score_single = bleu_score(&["the cat sat"], "the cat sat", 2);
    let score_multi = bleu_score(&["the cat sat", "the dog sat"], "the cat sat", 2);
    assert!(
        (score_single - score_multi).abs() < 1e-10,
        "Adding a non-matching ref should not change score: single={score_single}, multi={score_multi}"
    );
}

#[test]
fn falsify_bleu_max_n_1_equals_unigram() {
    // max_n=1 should only use unigram precision
    let score = bleu_score(&["a b c d"], "a b x d", 1);
    // unigram precision = 3/4 (a,b,d match), BP=1 (equal length)
    assert!((score - 0.75).abs() < 1e-10, "BLEU-1 should be unigram precision = 3/4, got {score}");
}

#[test]
fn falsify_bleu_zero_for_no_bigram_match() {
    // Even with perfect unigram match, zero bigram overlap → BLEU-4 = 0
    // "a b" vs "b a" — unigrams: both have {a, b}, bigrams: "a b" vs "b a" → 0
    let score = bleu_score(&["a b"], "b a", 4);
    assert_eq!(score, 0.0, "No bigram overlap should make BLEU-4 = 0, got {score}");
}
