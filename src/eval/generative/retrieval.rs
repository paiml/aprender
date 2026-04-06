//! Information retrieval evaluation metrics
//!
//! Provides NDCG@k (Normalized Discounted Cumulative Gain) for evaluating
//! ranking quality in search and retrieval systems.

/// Compute NDCG@k (Normalized Discounted Cumulative Gain).
///
/// Measures ranking quality by comparing the actual ranking against the
/// ideal ranking. Returns a value in [0, 1] where 1.0 indicates a
/// perfect ranking.
///
/// # Arguments
/// * `relevance_scores` - Relevance scores in the order returned by the system
/// * `k` - Number of top results to consider
///
/// # Returns
/// 0.0 if there are no relevant documents or k is 0.
pub fn ndcg_at_k(relevance_scores: &[f64], k: usize) -> f64 {
    contract_pre_ndcg_at_k!();
    if k == 0 || relevance_scores.is_empty() {
        return 0.0;
    }

    let k = k.min(relevance_scores.len());

    let actual_dcg = dcg(&relevance_scores[..k]);

    // Ideal DCG: sort scores descending and compute DCG
    let mut ideal_scores: Vec<f64> = relevance_scores.to_vec();
    ideal_scores.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let idcg = dcg(&ideal_scores[..k]);

    if idcg == 0.0 {
        return 0.0;
    }

    actual_dcg / idcg
}

/// Compute Discounted Cumulative Gain.
///
/// DCG = sum_{i=1}^{k} (2^rel_i - 1) / log2(i + 1)
fn dcg(scores: &[f64]) -> f64 {
    scores
        .iter()
        .enumerate()
        .map(|(i, &rel)| {
            let gain = (2.0f64).powf(rel) - 1.0;
            let discount = ((i + 2) as f64).max(f64::MIN_POSITIVE).log2(); // log2(i+1+1) since i is 0-indexed
            gain / discount
        })
        .sum()
}
