//! Text generation evaluation metrics
//!
//! Provides BLEU, ROUGE (1, 2, L), and Perplexity for evaluating
//! text generation, translation, and summarization models.

use std::collections::HashMap;

/// Compute BLEU score with modified n-gram precision and brevity penalty.
///
/// Implements the original BLEU algorithm (Papineni et al., 2002).
/// Returns a value in [0, 1] where 1.0 indicates perfect match.
///
/// # Arguments
/// * `references` - One or more reference translations
/// * `hypothesis` - The candidate translation
/// * `max_n` - Maximum n-gram order (typically 4)
pub fn bleu_score(references: &[&str], hypothesis: &str, max_n: usize) -> f64 {
    if references.is_empty() || hypothesis.is_empty() {
        return 0.0;
    }

    let hyp_tokens: Vec<&str> = hypothesis.split_whitespace().collect();
    if hyp_tokens.is_empty() {
        return 0.0;
    }

    let ref_token_lists: Vec<Vec<&str>> =
        references.iter().map(|r| r.split_whitespace().collect()).collect();

    // Compute modified precision for each n-gram order
    let mut log_precisions = Vec::new();
    for n in 1..=max_n {
        let (clipped, total) = modified_precision(&ref_token_lists, &hyp_tokens, n);
        if total == 0 {
            return 0.0;
        }
        let precision = clipped as f64 / total as f64;
        if precision == 0.0 {
            return 0.0;
        }
        log_precisions.push(precision.max(f64::MIN_POSITIVE).ln());
    }

    // Geometric mean of precisions (uniform weights)
    let avg_log_precision: f64 =
        log_precisions.iter().sum::<f64>() / log_precisions.len().max(1) as f64;

    // Brevity penalty
    let hyp_len = hyp_tokens.len();
    let closest_ref_len = ref_token_lists
        .iter()
        .map(Vec::len)
        .min_by_key(|&len| (len as isize - hyp_len as isize).unsigned_abs())
        .unwrap_or(0);

    let bp = if hyp_len >= closest_ref_len {
        1.0
    } else if closest_ref_len == 0 {
        0.0
    } else {
        (1.0 - closest_ref_len as f64 / hyp_len as f64).exp()
    };

    bp * avg_log_precision.exp()
}

/// Modified n-gram precision: count clipped matches against all references.
fn modified_precision(references: &[Vec<&str>], hypothesis: &[&str], n: usize) -> (usize, usize) {
    let hyp_ngrams = extract_ngrams(hypothesis, n);
    let total: usize = hyp_ngrams.values().sum();

    let mut clipped = 0usize;
    for (ngram, &hyp_count) in &hyp_ngrams {
        let max_ref_count = references
            .iter()
            .map(|r| {
                let ref_ngrams = extract_ngrams(r, n);
                ref_ngrams.get(ngram).copied().unwrap_or(0)
            })
            .max()
            .unwrap_or(0);
        clipped += hyp_count.min(max_ref_count);
    }

    (clipped, total)
}

/// Extract n-grams from a token sequence and count occurrences.
fn extract_ngrams<'a>(tokens: &[&'a str], n: usize) -> HashMap<Vec<&'a str>, usize> {
    let mut counts = HashMap::new();
    if tokens.len() >= n {
        for window in tokens.windows(n) {
            *counts.entry(window.to_vec()).or_insert(0) += 1;
        }
    }
    counts
}

/// Compute ROUGE-N F1 score (n-gram overlap between reference and hypothesis).
///
/// Returns F1 score in [0, 1].
pub fn rouge_n(reference: &str, hypothesis: &str, n: usize) -> f64 {
    let ref_tokens: Vec<&str> = reference.split_whitespace().collect();
    let hyp_tokens: Vec<&str> = hypothesis.split_whitespace().collect();

    if ref_tokens.len() < n || hyp_tokens.len() < n {
        return 0.0;
    }

    let ref_ngrams = extract_ngrams(&ref_tokens, n);
    let hyp_ngrams = extract_ngrams(&hyp_tokens, n);

    let mut overlap = 0usize;
    for (ngram, &hyp_count) in &hyp_ngrams {
        let ref_count = ref_ngrams.get(ngram).copied().unwrap_or(0);
        overlap += hyp_count.min(ref_count);
    }

    let ref_total: usize = ref_ngrams.values().sum();
    let hyp_total: usize = hyp_ngrams.values().sum();

    if ref_total == 0 || hyp_total == 0 {
        return 0.0;
    }

    let precision = overlap as f64 / hyp_total as f64;
    let recall = overlap as f64 / ref_total as f64;

    if precision + recall == 0.0 {
        return 0.0;
    }

    2.0 * precision * recall / (precision + recall)
}

/// Compute ROUGE-L F1 score using longest common subsequence.
///
/// Returns F1 score in [0, 1].
pub fn rouge_l(reference: &str, hypothesis: &str) -> f64 {
    let ref_tokens: Vec<&str> = reference.split_whitespace().collect();
    let hyp_tokens: Vec<&str> = hypothesis.split_whitespace().collect();

    if ref_tokens.is_empty() || hyp_tokens.is_empty() {
        return 0.0;
    }

    let lcs_len = lcs_length(&ref_tokens, &hyp_tokens);

    let precision = lcs_len as f64 / hyp_tokens.len() as f64;
    let recall = lcs_len as f64 / ref_tokens.len() as f64;

    if precision + recall == 0.0 {
        return 0.0;
    }

    2.0 * precision * recall / (precision + recall)
}

/// Compute length of longest common subsequence.
fn lcs_length(a: &[&str], b: &[&str]) -> usize {
    let n = a.len();
    let m = b.len();
    let mut dp = vec![vec![0usize; m + 1]; n + 1];

    for i in 1..=n {
        for j in 1..=m {
            if a[i - 1] == b[j - 1] {
                dp[i][j] = dp[i - 1][j - 1] + 1;
            } else {
                dp[i][j] = dp[i - 1][j].max(dp[i][j - 1]);
            }
        }
    }

    dp[n][m]
}

/// Compute perplexity from log-probabilities.
///
/// Perplexity = exp(-1/N * sum(log_probs))
///
/// Lower is better. Returns >= 1.0 for valid probability distributions.
/// Returns `f64::INFINITY` for empty input.
pub fn perplexity(log_probs: &[f64]) -> f64 {
    if log_probs.is_empty() {
        return f64::INFINITY;
    }

    let avg_neg_log_prob = -log_probs.iter().sum::<f64>() / log_probs.len() as f64;
    avg_neg_log_prob.exp()
}
