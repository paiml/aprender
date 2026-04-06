//! Efficiency metrics for curriculum learning

/// Compute efficiency score as per CITL spec
///
/// E(T) = Accuracy(T) / log(CorpusSize(T))
///
/// Higher is better - balances accuracy against corpus bloat.
pub fn efficiency_score(accuracy: f32, corpus_size_bytes: usize) -> f32 {
    if corpus_size_bytes <= 1 {
        return accuracy;
    }
    accuracy / (corpus_size_bytes as f32).max(f32::MIN_POSITIVE).ln()
}

/// Compare tiers and select optimal based on efficiency
///
/// Returns (best_tier, efficiency_score)
pub fn select_optimal_tier(tier_results: &[(usize, f32, usize)]) -> Option<(usize, f32)> {
    tier_results
        .iter()
        .map(|&(tier, accuracy, corpus_size)| {
            let eff = efficiency_score(accuracy, corpus_size);
            (tier, eff)
        })
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
}
