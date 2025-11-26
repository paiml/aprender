//! Ranking metrics for recommendation and retrieval systems.
//!
//! These metrics evaluate how well a model ranks relevant items.
//! Common use cases:
//! - Recommendation systems
//! - Information retrieval
//! - Language model evaluation
//! - Search result ranking

/// Hit@K: Whether the correct item appears in top-K predictions.
///
/// Returns 1.0 if the target is in the top-K predictions, 0.0 otherwise.
///
/// # Examples
///
/// ```
/// use aprender::metrics::ranking::hit_at_k;
///
/// let predictions = vec![5, 3, 1, 4, 2];  // Ranked predictions
/// let target = 3;
///
/// assert_eq!(hit_at_k(&predictions, &target, 1), 0.0);  // 3 is not #1
/// assert_eq!(hit_at_k(&predictions, &target, 2), 1.0);  // 3 is in top 2
/// assert_eq!(hit_at_k(&predictions, &target, 5), 1.0);  // 3 is in top 5
/// ```
#[must_use]
pub fn hit_at_k<T: PartialEq>(predictions: &[T], target: &T, k: usize) -> f32 {
    let top_k = predictions.iter().take(k);
    if top_k.into_iter().any(|p| p == target) {
        1.0
    } else {
        0.0
    }
}

/// Mean Hit@K over multiple queries.
///
/// # Examples
///
/// ```
/// use aprender::metrics::ranking::mean_hit_at_k;
///
/// let predictions = vec![
///     vec![1, 2, 3, 4, 5],
///     vec![5, 4, 3, 2, 1],
///     vec![1, 3, 5, 2, 4],
/// ];
/// let targets = vec![1, 1, 1];
///
/// let hit_1 = mean_hit_at_k(&predictions, &targets, 1);
/// assert!((hit_1 - 0.667).abs() < 0.01);  // 2 out of 3 have target at position 1
/// ```
#[must_use]
pub fn mean_hit_at_k<T: PartialEq + Clone>(predictions: &[Vec<T>], targets: &[T], k: usize) -> f32 {
    if predictions.is_empty() || predictions.len() != targets.len() {
        return 0.0;
    }

    let hits: f32 = predictions
        .iter()
        .zip(targets.iter())
        .map(|(preds, target)| hit_at_k(preds, target, k))
        .sum();

    hits / predictions.len() as f32
}

/// Reciprocal Rank: 1/rank of the first correct prediction.
///
/// Returns 0.0 if the target is not in predictions.
///
/// # Examples
///
/// ```
/// use aprender::metrics::ranking::reciprocal_rank;
///
/// let predictions = vec![5, 3, 1, 4, 2];
///
/// assert!((reciprocal_rank(&predictions, &5) - 1.0).abs() < 1e-6);    // rank 1 → 1/1
/// assert!((reciprocal_rank(&predictions, &3) - 0.5).abs() < 1e-6);    // rank 2 → 1/2
/// assert!((reciprocal_rank(&predictions, &1) - 0.333).abs() < 0.01);  // rank 3 → 1/3
/// assert!((reciprocal_rank(&predictions, &99) - 0.0).abs() < 1e-6);   // not found → 0
/// ```
#[must_use]
pub fn reciprocal_rank<T: PartialEq>(predictions: &[T], target: &T) -> f32 {
    for (i, pred) in predictions.iter().enumerate() {
        if pred == target {
            return 1.0 / (i + 1) as f32;
        }
    }
    0.0
}

/// Mean Reciprocal Rank (MRR) over multiple queries.
///
/// MRR is a standard metric for evaluating ranking quality.
///
/// # Examples
///
/// ```
/// use aprender::metrics::ranking::mrr;
///
/// let predictions = vec![
///     vec![1, 2, 3],  // target at rank 1
///     vec![2, 1, 3],  // target at rank 2
///     vec![3, 2, 1],  // target at rank 3
/// ];
/// let targets = vec![1, 1, 1];
///
/// let score = mrr(&predictions, &targets);
/// // MRR = (1/1 + 1/2 + 1/3) / 3 ≈ 0.611
/// assert!((score - 0.611).abs() < 0.01);
/// ```
#[must_use]
pub fn mrr<T: PartialEq + Clone>(predictions: &[Vec<T>], targets: &[T]) -> f32 {
    if predictions.is_empty() || predictions.len() != targets.len() {
        return 0.0;
    }

    let rr_sum: f32 = predictions
        .iter()
        .zip(targets.iter())
        .map(|(preds, target)| reciprocal_rank(preds, target))
        .sum();

    rr_sum / predictions.len() as f32
}

/// Discounted Cumulative Gain at K (DCG@K).
///
/// DCG weights relevant items by their position (higher positions are better).
/// Uses the formula: rel_i / log2(i + 1)
///
/// # Arguments
///
/// * `relevance` - Relevance scores for each prediction (higher = more relevant)
/// * `k` - Number of top predictions to consider
///
/// # Examples
///
/// ```
/// use aprender::metrics::ranking::dcg_at_k;
///
/// // Relevance scores: [3, 2, 3, 0, 1] (higher = more relevant)
/// let relevance = vec![3.0, 2.0, 3.0, 0.0, 1.0];
///
/// let dcg = dcg_at_k(&relevance, 5);
/// // DCG = 3/log2(2) + 2/log2(3) + 3/log2(4) + 0/log2(5) + 1/log2(6)
/// assert!(dcg > 5.0);
/// ```
#[must_use]
pub fn dcg_at_k(relevance: &[f32], k: usize) -> f32 {
    relevance
        .iter()
        .take(k)
        .enumerate()
        .map(|(i, &rel)| rel / (i as f32 + 2.0).log2())
        .sum()
}

/// Normalized Discounted Cumulative Gain at K (NDCG@K).
///
/// NDCG normalizes DCG by the ideal DCG (perfect ranking).
/// Returns a value between 0 and 1, where 1 is a perfect ranking.
///
/// # Examples
///
/// ```
/// use aprender::metrics::ranking::ndcg_at_k;
///
/// // Perfect ranking: highest relevance first
/// let perfect = vec![3.0, 3.0, 2.0, 1.0, 0.0];
/// assert!((ndcg_at_k(&perfect, 5) - 1.0).abs() < 0.01);
///
/// // Suboptimal ranking
/// let suboptimal = vec![0.0, 1.0, 2.0, 3.0, 3.0];
/// assert!(ndcg_at_k(&suboptimal, 5) < 0.8);
/// ```
#[must_use]
pub fn ndcg_at_k(relevance: &[f32], k: usize) -> f32 {
    let dcg = dcg_at_k(relevance, k);

    // Compute ideal DCG (sorted by relevance descending)
    let mut ideal_relevance = relevance.to_vec();
    ideal_relevance.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let idcg = dcg_at_k(&ideal_relevance, k);

    if idcg == 0.0 {
        0.0
    } else {
        dcg / idcg
    }
}

/// Comprehensive ranking evaluation results.
#[derive(Debug, Clone)]
pub struct RankingMetrics {
    /// Hit@1 accuracy
    pub hit_at_1: f32,
    /// Hit@5 accuracy
    pub hit_at_5: f32,
    /// Hit@10 accuracy
    pub hit_at_10: f32,
    /// Mean Reciprocal Rank
    pub mrr: f32,
    /// Number of samples evaluated
    pub n_samples: usize,
}

impl RankingMetrics {
    /// Compute all ranking metrics from predictions and targets.
    ///
    /// # Examples
    ///
    /// ```
    /// use aprender::metrics::ranking::RankingMetrics;
    ///
    /// let predictions = vec![
    ///     vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    ///     vec![2, 1, 3, 4, 5, 6, 7, 8, 9, 10],
    /// ];
    /// let targets = vec![1, 1];
    ///
    /// let metrics = RankingMetrics::compute(&predictions, &targets);
    /// assert_eq!(metrics.hit_at_1, 0.5);  // 1 out of 2
    /// assert_eq!(metrics.hit_at_5, 1.0);  // both within top 5
    /// ```
    #[must_use]
    pub fn compute<T: PartialEq + Clone>(predictions: &[Vec<T>], targets: &[T]) -> Self {
        Self {
            hit_at_1: mean_hit_at_k(predictions, targets, 1),
            hit_at_5: mean_hit_at_k(predictions, targets, 5),
            hit_at_10: mean_hit_at_k(predictions, targets, 10),
            mrr: mrr(predictions, targets),
            n_samples: predictions.len(),
        }
    }

    /// Generate a formatted report string.
    #[must_use]
    pub fn report(&self) -> String {
        format!(
            "Ranking Metrics (n={})\n\
             ─────────────────────\n\
             Hit@1:   {:>6.1}%\n\
             Hit@5:   {:>6.1}%\n\
             Hit@10:  {:>6.1}%\n\
             MRR:     {:>6.3}",
            self.n_samples,
            self.hit_at_1 * 100.0,
            self.hit_at_5 * 100.0,
            self.hit_at_10 * 100.0,
            self.mrr
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hit_at_k_found_first() {
        let predictions = vec![1, 2, 3, 4, 5];
        assert_eq!(hit_at_k(&predictions, &1, 1), 1.0);
        assert_eq!(hit_at_k(&predictions, &1, 5), 1.0);
    }

    #[test]
    fn test_hit_at_k_found_later() {
        let predictions = vec![1, 2, 3, 4, 5];
        assert_eq!(hit_at_k(&predictions, &3, 1), 0.0);
        assert_eq!(hit_at_k(&predictions, &3, 2), 0.0);
        assert_eq!(hit_at_k(&predictions, &3, 3), 1.0);
    }

    #[test]
    fn test_hit_at_k_not_found() {
        let predictions = vec![1, 2, 3, 4, 5];
        assert_eq!(hit_at_k(&predictions, &99, 5), 0.0);
    }

    #[test]
    fn test_mean_hit_at_k() {
        let predictions = vec![vec![1, 2, 3], vec![2, 1, 3], vec![3, 2, 1]];
        let targets = vec![1, 1, 1];

        assert!((mean_hit_at_k(&predictions, &targets, 1) - 0.333).abs() < 0.01);
        assert!((mean_hit_at_k(&predictions, &targets, 2) - 0.667).abs() < 0.01);
        assert!((mean_hit_at_k(&predictions, &targets, 3) - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_reciprocal_rank() {
        let predictions = vec![5, 3, 1, 4, 2];

        assert!((reciprocal_rank(&predictions, &5) - 1.0).abs() < 1e-6);
        assert!((reciprocal_rank(&predictions, &3) - 0.5).abs() < 1e-6);
        assert!((reciprocal_rank(&predictions, &1) - 1.0 / 3.0).abs() < 1e-6);
        assert!((reciprocal_rank(&predictions, &4) - 0.25).abs() < 1e-6);
        assert!((reciprocal_rank(&predictions, &99) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_mrr() {
        let predictions = vec![
            vec![1, 2, 3], // RR = 1
            vec![2, 1, 3], // RR = 0.5
            vec![3, 2, 1], // RR = 0.333
        ];
        let targets = vec![1, 1, 1];

        let score = mrr(&predictions, &targets);
        let expected = (1.0 + 0.5 + 1.0 / 3.0) / 3.0;
        assert!((score - expected).abs() < 0.01);
    }

    #[test]
    fn test_mrr_empty() {
        let predictions: Vec<Vec<i32>> = vec![];
        let targets: Vec<i32> = vec![];
        assert_eq!(mrr(&predictions, &targets), 0.0);
    }

    #[test]
    fn test_dcg_at_k() {
        let relevance = vec![3.0, 2.0, 3.0, 0.0, 1.0];
        let dcg = dcg_at_k(&relevance, 5);

        // DCG = 3/log2(2) + 2/log2(3) + 3/log2(4) + 0/log2(5) + 1/log2(6)
        let expected = 3.0 / 1.0 + 2.0 / 1.585 + 3.0 / 2.0 + 0.0 / 2.322 + 1.0 / 2.585;
        assert!((dcg - expected).abs() < 0.1);
    }

    #[test]
    fn test_ndcg_at_k_perfect() {
        let relevance = vec![3.0, 2.0, 1.0, 0.0];
        let ndcg = ndcg_at_k(&relevance, 4);
        assert!((ndcg - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_ndcg_at_k_reversed() {
        let relevance = vec![0.0, 1.0, 2.0, 3.0];
        let ndcg = ndcg_at_k(&relevance, 4);
        assert!(ndcg < 1.0);
        assert!(ndcg > 0.0);
    }

    #[test]
    fn test_ndcg_at_k_all_zero() {
        let relevance = vec![0.0, 0.0, 0.0];
        let ndcg = ndcg_at_k(&relevance, 3);
        assert_eq!(ndcg, 0.0);
    }

    #[test]
    fn test_ranking_metrics_compute() {
        let predictions = vec![
            vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            vec![2, 1, 3, 4, 5, 6, 7, 8, 9, 10],
        ];
        let targets = vec![1, 1];

        let metrics = RankingMetrics::compute(&predictions, &targets);

        assert_eq!(metrics.hit_at_1, 0.5);
        assert_eq!(metrics.hit_at_5, 1.0);
        assert_eq!(metrics.hit_at_10, 1.0);
        assert!((metrics.mrr - 0.75).abs() < 0.01); // (1 + 0.5) / 2
        assert_eq!(metrics.n_samples, 2);
    }

    #[test]
    fn test_ranking_metrics_report() {
        let metrics = RankingMetrics {
            hit_at_1: 0.5,
            hit_at_5: 0.8,
            hit_at_10: 0.95,
            mrr: 0.65,
            n_samples: 100,
        };

        let report = metrics.report();
        assert!(report.contains("Hit@1"));
        assert!(report.contains("50.0%"));
        assert!(report.contains("MRR"));
    }

    #[test]
    fn test_hit_at_k_with_strings() {
        let predictions = vec!["git commit", "git push", "git pull"];
        assert_eq!(hit_at_k(&predictions, &"git push", 1), 0.0);
        assert_eq!(hit_at_k(&predictions, &"git push", 2), 1.0);
    }
}
