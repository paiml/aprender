//! Falsification tests for ROUGE and Perplexity metrics
//!
//! Tests ROUGE-N, ROUGE-L, and Perplexity mathematical correctness.

use super::*;

// ═══════════════════════════════════════════════════════════════════════
// CLAIM 3: ROUGE-N computes F1 of n-gram overlap
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn falsify_rouge1_known_value() {
    // ref: "a b c d" (4 unigrams), hyp: "a b e f" (4 unigrams)
    // overlap = {a, b} → 2
    // precision = 2/4 = 0.5, recall = 2/4 = 0.5
    // F1 = 2 * 0.5 * 0.5 / (0.5 + 0.5) = 0.5
    let score = rouge_n("a b c d", "a b e f", 1);
    assert!((score - 0.5).abs() < 1e-10, "ROUGE-1 F1 should be 0.5, got {score}");
}

#[test]
fn falsify_rouge1_asymmetric_lengths() {
    // ref: "a b c" (3 unigrams), hyp: "a" (1 unigram)
    // overlap = {a} → 1
    // precision = 1/1 = 1.0, recall = 1/3
    // F1 = 2 * 1.0 * (1/3) / (1.0 + 1/3) = (2/3) / (4/3) = 0.5
    let score = rouge_n("a b c", "a", 1);
    assert!(
        (score - 0.5).abs() < 1e-10,
        "ROUGE-1 with asymmetric lengths should be 0.5, got {score}"
    );
}

#[test]
fn falsify_rouge_n_is_symmetric() {
    // ROUGE F1 IS symmetric (unlike ROUGE precision or recall alone)
    let r1 = rouge_n("a b c", "a b d", 1);
    let r2 = rouge_n("a b d", "a b c", 1);
    assert!((r1 - r2).abs() < 1e-10, "ROUGE-1 F1 should be symmetric: {r1} vs {r2}");
}

#[test]
fn falsify_rouge2_known_value() {
    // ref: "a b c d" → bigrams: {"a b", "b c", "c d"} (3)
    // hyp: "a b c e" → bigrams: {"a b", "b c", "c e"} (3)
    // overlap: {"a b", "b c"} → 2
    // precision = 2/3, recall = 2/3
    // F1 = 2 * (2/3) * (2/3) / (4/3) = (8/9) / (4/3) = 2/3
    let score = rouge_n("a b c d", "a b c e", 2);
    assert!((score - 2.0 / 3.0).abs() < 1e-10, "ROUGE-2 F1 should be 2/3, got {score}");
}

#[test]
fn falsify_rouge_l_known_value() {
    // ref: "a b c d e" (5 tokens), hyp: "a x b c y" (5 tokens)
    // LCS: "a b c" (length 3)
    // precision = 3/5 = 0.6, recall = 3/5 = 0.6
    // F1 = 0.6
    let score = rouge_l("a b c d e", "a x b c y");
    assert!((score - 0.6).abs() < 1e-10, "ROUGE-L F1 should be 0.6, got {score}");
}

#[test]
fn falsify_rouge_l_subsequence_not_substring() {
    // LCS is subsequence, not substring
    // ref: "a x b x c" (5), hyp: "a b c" (3)
    // LCS: "a b c" (3), NOT "a" (which would be longest common substring at position 0)
    // precision = 3/3 = 1.0, recall = 3/5 = 0.6
    // F1 = 2 * 1.0 * 0.6 / 1.6 = 0.75
    let score = rouge_l("a x b x c", "a b c");
    assert!((score - 0.75).abs() < 1e-10, "ROUGE-L should use subsequence, got {score}");
}

#[test]
fn falsify_rouge_l_geq_rouge2() {
    // ROUGE-L should generally be >= ROUGE-2 for non-degenerate cases
    // (LCS captures long-range matches that bigrams miss)
    let r2 = rouge_n(
        "the quick brown fox jumps over the lazy dog",
        "the quick brown cat jumps over the lazy dog",
        2,
    );
    let rl = rouge_l(
        "the quick brown fox jumps over the lazy dog",
        "the quick brown cat jumps over the lazy dog",
    );
    assert!(rl >= r2 - 1e-10, "ROUGE-L should be >= ROUGE-2 for long sequences: RL={rl}, R2={r2}");
}

// ═══════════════════════════════════════════════════════════════════════
// CLAIM 4: Perplexity = exp(-mean(log_probs))
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn falsify_perplexity_known_binary() {
    // Binary uniform: p = 0.5 → log(0.5) = -ln(2)
    // PPL = exp(-mean(-ln(2))) = exp(ln(2)) = 2.0
    let log_probs = vec![(0.5f64).ln(); 100];
    let ppl = perplexity(&log_probs);
    assert!((ppl - 2.0).abs() < 1e-10, "Binary uniform → PPL=2.0, got {ppl}");
}

#[test]
fn falsify_perplexity_single_token() {
    // Single token with prob 0.01 → PPL = 100
    let ppl = perplexity(&[(0.01f64).ln()]);
    assert!((ppl - 100.0).abs() < 1e-6, "Single token p=0.01 → PPL=100, got {ppl}");
}

#[test]
fn falsify_perplexity_positive_log_probs() {
    // Positive log probs (invalid, but should not panic)
    // log_prob > 0 means prob > 1, which is invalid but PPL should still compute
    let ppl = perplexity(&[1.0, 2.0, 3.0]);
    assert!(ppl < 1.0, "Positive log-probs → PPL < 1.0 (invalid but computable), got {ppl}");
}

#[test]
fn falsify_perplexity_mixed_values() {
    // Mix of good and bad predictions
    let log_probs = vec![(0.9f64).ln(), (0.01f64).ln()];
    let ppl = perplexity(&log_probs);
    // PPL = exp(-(ln(0.9) + ln(0.01)) / 2) = exp(-ln(0.009)/2) = (1/0.009)^0.5 ≈ 10.54
    let expected = (1.0 / (0.9 * 0.01f64)).sqrt();
    assert!((ppl - expected).abs() < 1e-6, "Mixed PPL: expected {expected}, got {ppl}");
}
