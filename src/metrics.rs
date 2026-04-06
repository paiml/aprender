//! Retrieval evaluation metrics

use crate::ChunkId;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

/// Retrieval metrics for evaluation
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RetrievalMetrics {
    /// Recall@k for various k values
    pub recall: std::collections::HashMap<usize, f32>,
    /// Precision@k for various k values
    pub precision: std::collections::HashMap<usize, f32>,
    /// Mean Reciprocal Rank
    pub mrr: f32,
    /// Normalized Discounted Cumulative Gain@k
    pub ndcg: std::collections::HashMap<usize, f32>,
    /// Mean Average Precision
    pub map: f32,
}

impl RetrievalMetrics {
    /// Compute all metrics for a single query
    pub fn compute(retrieved: &[ChunkId], relevant: &HashSet<ChunkId>, k_values: &[usize]) -> Self {
        // Contract: configuration-v1.yaml precondition (pv codegen)
        contract_pre_configuration!(retrieved);
        let mut metrics = Self::default();

        for &k in k_values {
            metrics.recall.insert(k, Self::recall_at_k(retrieved, relevant, k));
            metrics.precision.insert(k, Self::precision_at_k(retrieved, relevant, k));
            metrics.ndcg.insert(k, Self::ndcg_at_k(retrieved, relevant, k));
        }

        metrics.mrr = Self::mean_reciprocal_rank(retrieved, relevant);
        metrics.map = Self::average_precision(retrieved, relevant);

        metrics
    }

    /// Compute Recall@k
    ///
    /// Recall@k = |relevant ∩ retrieved@k| / |relevant|
    #[must_use]
    pub fn recall_at_k(retrieved: &[ChunkId], relevant: &HashSet<ChunkId>, k: usize) -> f32 {
        if relevant.is_empty() {
            return 0.0;
        }

        // Contract: configuration-v1.yaml precondition (pv codegen)
        contract_pre_configuration!(retrieved);
        let retrieved_k: HashSet<ChunkId> = retrieved.iter().take(k).copied().collect();
        let relevant_retrieved = retrieved_k.intersection(relevant).count();

        relevant_retrieved as f32 / relevant.len() as f32
    }

    /// Compute Precision@k
    ///
    /// Precision@k = |relevant ∩ retrieved@k| / k
    #[must_use]
    pub fn precision_at_k(retrieved: &[ChunkId], relevant: &HashSet<ChunkId>, k: usize) -> f32 {
        if k == 0 {
            return 0.0;
        }

        // Contract: configuration-v1.yaml precondition (pv codegen)
        contract_pre_configuration!(retrieved);
        let retrieved_k: HashSet<ChunkId> = retrieved.iter().take(k).copied().collect();
        let relevant_retrieved = retrieved_k.intersection(relevant).count();

        relevant_retrieved as f32 / k as f32
    }

    /// Compute Mean Reciprocal Rank (MRR)
    ///
    /// MRR = 1 / rank of first relevant result
    #[must_use]
    pub fn mean_reciprocal_rank(retrieved: &[ChunkId], relevant: &HashSet<ChunkId>) -> f32 {
        // Contract: pagerank-kernel-v1.yaml precondition (pv codegen)
        contract_pre_pagerank!(retrieved);
        for (rank, id) in retrieved.iter().enumerate() {
            if relevant.contains(id) {
                return 1.0 / (rank + 1) as f32;
            }
        }
        0.0
    }

    /// Compute Normalized Discounted Cumulative Gain@k
    ///
    /// NDCG@k = DCG@k / IDCG@k
    #[must_use]
    pub fn ndcg_at_k(retrieved: &[ChunkId], relevant: &HashSet<ChunkId>, k: usize) -> f32 {
        // Contract: configuration-v1.yaml precondition (pv codegen)
        contract_pre_configuration!(retrieved);
        let dcg = Self::dcg_at_k(retrieved, relevant, k);
        let idcg = Self::ideal_dcg_at_k(relevant.len(), k);

        if idcg == 0.0 {
            0.0
        } else {
            dcg / idcg
        }
    }

    /// Compute Discounted Cumulative Gain@k
    ///
    /// Note: Each relevant item is counted at most once (at its first occurrence)
    /// to ensure NDCG remains bounded by 1.0.
    fn dcg_at_k(retrieved: &[ChunkId], relevant: &HashSet<ChunkId>, k: usize) -> f32 {
        let mut seen = HashSet::new();
        retrieved
            .iter()
            .take(k)
            .enumerate()
            .filter(|(_, id)| relevant.contains(id) && seen.insert(**id))
            .map(|(rank, _)| 1.0 / (rank as f32 + 2.0).max(f32::EPSILON).log2())
            .sum()
    }

    /// Compute Ideal DCG@k (best possible DCG)
    fn ideal_dcg_at_k(num_relevant: usize, k: usize) -> f32 {
        (0..num_relevant.min(k))
            .map(|rank| 1.0 / (rank as f32 + 2.0).max(f32::EPSILON).log2())
            .sum()
    }

    /// Compute Average Precision (AP)
    ///
    /// AP = (1/|relevant|) * Σ (Precision@k * rel(k))
    #[must_use]
    pub fn average_precision(retrieved: &[ChunkId], relevant: &HashSet<ChunkId>) -> f32 {
        if relevant.is_empty() {
            return 0.0;
        }

        // Contract: configuration-v1.yaml precondition (pv codegen)
        contract_pre_configuration!(retrieved);
        let mut sum_precision = 0.0;
        let mut relevant_count = 0;

        for (rank, id) in retrieved.iter().enumerate() {
            if relevant.contains(id) {
                relevant_count += 1;
                sum_precision += relevant_count as f32 / (rank + 1) as f32;
            }
        }

        sum_precision / relevant.len().max(1) as f32
    }

    /// Compute F1 score at k
    #[must_use]
    pub fn f1_at_k(retrieved: &[ChunkId], relevant: &HashSet<ChunkId>, k: usize) -> f32 {
        // Contract: configuration-v1.yaml precondition (pv codegen)
        contract_pre_configuration!(retrieved);
        let precision = Self::precision_at_k(retrieved, relevant, k);
        let recall = Self::recall_at_k(retrieved, relevant, k);

        if precision + recall == 0.0 {
            0.0
        } else {
            2.0 * precision * recall / (precision + recall)
        }
    }

    /// Compute Hit Rate (1 if any relevant in top-k, else 0)
    #[must_use]
    pub fn hit_rate_at_k(retrieved: &[ChunkId], relevant: &HashSet<ChunkId>, k: usize) -> f32 {
        // Contract: configuration-v1.yaml precondition (pv codegen)
        contract_pre_configuration!(retrieved);
        let retrieved_k: HashSet<ChunkId> = retrieved.iter().take(k).copied().collect();
        if retrieved_k.intersection(relevant).next().is_some() {
            1.0
        } else {
            0.0
        }
    }
}

/// Aggregated metrics across multiple queries
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AggregatedMetrics {
    /// Mean Recall@k
    pub mean_recall: std::collections::HashMap<usize, f32>,
    /// Mean Precision@k
    pub mean_precision: std::collections::HashMap<usize, f32>,
    /// Mean MRR
    pub mean_mrr: f32,
    /// Mean NDCG@k
    pub mean_ndcg: std::collections::HashMap<usize, f32>,
    /// Mean Average Precision (MAP)
    pub map: f32,
    /// Number of queries
    pub query_count: usize,
}

impl AggregatedMetrics {
    /// Aggregate metrics from multiple queries
    pub fn aggregate(metrics: &[RetrievalMetrics]) -> Self {
        if metrics.is_empty() {
            return Self::default();
        }

        let n = metrics.len() as f32;
        let mut agg = Self { query_count: metrics.len(), ..Default::default() };

        // Aggregate MRR and MAP
        agg.mean_mrr = metrics.iter().map(|m| m.mrr).sum::<f32>() / n;
        agg.map = metrics.iter().map(|m| m.map).sum::<f32>() / n;

        // Aggregate k-based metrics
        if let Some(first) = metrics.first() {
            for &k in first.recall.keys() {
                let mean_recall = metrics.iter().filter_map(|m| m.recall.get(&k)).sum::<f32>() / n;
                agg.mean_recall.insert(k, mean_recall);

                let mean_precision =
                    metrics.iter().filter_map(|m| m.precision.get(&k)).sum::<f32>() / n;
                agg.mean_precision.insert(k, mean_precision);

                let mean_ndcg = metrics.iter().filter_map(|m| m.ndcg.get(&k)).sum::<f32>() / n;
                agg.mean_ndcg.insert(k, mean_ndcg);
            }
        }

        agg
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chunk_id(n: u128) -> ChunkId {
        ChunkId(uuid::Uuid::from_u128(n))
    }

    // ============ Recall Tests ============

    #[test]
    fn test_recall_at_k_perfect() {
        let retrieved = vec![chunk_id(1), chunk_id(2), chunk_id(3)];
        let relevant: HashSet<_> = [chunk_id(1), chunk_id(2), chunk_id(3)].into();

        let recall = RetrievalMetrics::recall_at_k(&retrieved, &relevant, 3);
        assert!((recall - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_recall_at_k_partial() {
        let retrieved = vec![chunk_id(1), chunk_id(4), chunk_id(5)];
        let relevant: HashSet<_> = [chunk_id(1), chunk_id(2), chunk_id(3)].into();

        let recall = RetrievalMetrics::recall_at_k(&retrieved, &relevant, 3);
        assert!((recall - 1.0 / 3.0).abs() < 0.001);
    }

    #[test]
    fn test_recall_at_k_none() {
        let retrieved = vec![chunk_id(4), chunk_id(5), chunk_id(6)];
        let relevant: HashSet<_> = [chunk_id(1), chunk_id(2), chunk_id(3)].into();

        let recall = RetrievalMetrics::recall_at_k(&retrieved, &relevant, 3);
        assert!((recall - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_recall_at_k_empty_relevant() {
        let retrieved = vec![chunk_id(1), chunk_id(2)];
        let relevant: HashSet<ChunkId> = HashSet::new();

        let recall = RetrievalMetrics::recall_at_k(&retrieved, &relevant, 2);
        assert!((recall - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_recall_at_k_smaller_k() {
        let retrieved = vec![chunk_id(4), chunk_id(1), chunk_id(2)];
        let relevant: HashSet<_> = [chunk_id(1), chunk_id(2)].into();

        // At k=1, only chunk_id(4) which is not relevant
        let recall = RetrievalMetrics::recall_at_k(&retrieved, &relevant, 1);
        assert!((recall - 0.0).abs() < 0.001);

        // At k=2, chunk_id(1) is relevant
        let recall = RetrievalMetrics::recall_at_k(&retrieved, &relevant, 2);
        assert!((recall - 0.5).abs() < 0.001);
    }

    // ============ Precision Tests ============

    #[test]
    fn test_precision_at_k_perfect() {
        let retrieved = vec![chunk_id(1), chunk_id(2)];
        let relevant: HashSet<_> = [chunk_id(1), chunk_id(2)].into();

        let precision = RetrievalMetrics::precision_at_k(&retrieved, &relevant, 2);
        assert!((precision - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_precision_at_k_half() {
        let retrieved = vec![chunk_id(1), chunk_id(4)];
        let relevant: HashSet<_> = [chunk_id(1), chunk_id(2)].into();

        let precision = RetrievalMetrics::precision_at_k(&retrieved, &relevant, 2);
        assert!((precision - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_precision_at_k_zero() {
        let precision = RetrievalMetrics::precision_at_k(&[], &HashSet::new(), 0);
        assert!((precision - 0.0).abs() < 0.001);
    }

    // ============ MRR Tests ============

    #[test]
    fn test_mrr_first_position() {
        let retrieved = vec![chunk_id(1), chunk_id(2), chunk_id(3)];
        let relevant: HashSet<_> = [chunk_id(1)].into();

        let mrr = RetrievalMetrics::mean_reciprocal_rank(&retrieved, &relevant);
        assert!((mrr - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_mrr_second_position() {
        let retrieved = vec![chunk_id(4), chunk_id(1), chunk_id(3)];
        let relevant: HashSet<_> = [chunk_id(1)].into();

        let mrr = RetrievalMetrics::mean_reciprocal_rank(&retrieved, &relevant);
        assert!((mrr - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_mrr_third_position() {
        let retrieved = vec![chunk_id(4), chunk_id(5), chunk_id(1)];
        let relevant: HashSet<_> = [chunk_id(1)].into();

        let mrr = RetrievalMetrics::mean_reciprocal_rank(&retrieved, &relevant);
        assert!((mrr - 1.0 / 3.0).abs() < 0.001);
    }

    #[test]
    fn test_mrr_not_found() {
        let retrieved = vec![chunk_id(4), chunk_id(5), chunk_id(6)];
        let relevant: HashSet<_> = [chunk_id(1)].into();

        let mrr = RetrievalMetrics::mean_reciprocal_rank(&retrieved, &relevant);
        assert!((mrr - 0.0).abs() < 0.001);
    }

    // ============ NDCG Tests ============

    #[test]
    fn test_ndcg_perfect_order() {
        let retrieved = vec![chunk_id(1), chunk_id(2)];
        let relevant: HashSet<_> = [chunk_id(1), chunk_id(2)].into();

        let ndcg = RetrievalMetrics::ndcg_at_k(&retrieved, &relevant, 2);
        assert!((ndcg - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_ndcg_no_relevant() {
        let retrieved = vec![chunk_id(3), chunk_id(4)];
        let relevant: HashSet<_> = [chunk_id(1), chunk_id(2)].into();

        let ndcg = RetrievalMetrics::ndcg_at_k(&retrieved, &relevant, 2);
        assert!((ndcg - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_ndcg_empty_relevant() {
        let retrieved = vec![chunk_id(1), chunk_id(2)];
        let relevant: HashSet<ChunkId> = HashSet::new();

        let ndcg = RetrievalMetrics::ndcg_at_k(&retrieved, &relevant, 2);
        assert!((ndcg - 0.0).abs() < 0.001);
    }

    // ============ Average Precision Tests ============

    #[test]
    fn test_ap_perfect() {
        let retrieved = vec![chunk_id(1), chunk_id(2), chunk_id(3)];
        let relevant: HashSet<_> = [chunk_id(1), chunk_id(2), chunk_id(3)].into();

        let ap = RetrievalMetrics::average_precision(&retrieved, &relevant);
        // AP = (1/3) * (1/1 + 2/2 + 3/3) = (1/3) * 3 = 1.0
        assert!((ap - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_ap_interleaved() {
        let retrieved = vec![chunk_id(1), chunk_id(4), chunk_id(2)];
        let relevant: HashSet<_> = [chunk_id(1), chunk_id(2)].into();

        let ap = RetrievalMetrics::average_precision(&retrieved, &relevant);
        // AP = (1/2) * (1/1 + 2/3) = (1/2) * (1 + 0.667) = 0.833
        assert!((ap - 5.0 / 6.0).abs() < 0.001);
    }

    #[test]
    fn test_ap_empty_relevant() {
        let retrieved = vec![chunk_id(1), chunk_id(2)];
        let relevant: HashSet<ChunkId> = HashSet::new();

        let ap = RetrievalMetrics::average_precision(&retrieved, &relevant);
        assert!((ap - 0.0).abs() < 0.001);
    }

    // ============ F1 Tests ============

    #[test]
    fn test_f1_perfect() {
        let retrieved = vec![chunk_id(1), chunk_id(2)];
        let relevant: HashSet<_> = [chunk_id(1), chunk_id(2)].into();

        let f1 = RetrievalMetrics::f1_at_k(&retrieved, &relevant, 2);
        assert!((f1 - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_f1_zero() {
        let retrieved = vec![chunk_id(3), chunk_id(4)];
        let relevant: HashSet<_> = [chunk_id(1), chunk_id(2)].into();

        let f1 = RetrievalMetrics::f1_at_k(&retrieved, &relevant, 2);
        assert!((f1 - 0.0).abs() < 0.001);
    }

    // ============ Hit Rate Tests ============

    #[test]
    fn test_hit_rate_hit() {
        let retrieved = vec![chunk_id(3), chunk_id(1), chunk_id(4)];
        let relevant: HashSet<_> = [chunk_id(1), chunk_id(2)].into();

        let hr = RetrievalMetrics::hit_rate_at_k(&retrieved, &relevant, 3);
        assert!((hr - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_hit_rate_miss() {
        let retrieved = vec![chunk_id(3), chunk_id(4)];
        let relevant: HashSet<_> = [chunk_id(1), chunk_id(2)].into();

        let hr = RetrievalMetrics::hit_rate_at_k(&retrieved, &relevant, 2);
        assert!((hr - 0.0).abs() < 0.001);
    }

    // ============ Compute Tests ============

    #[test]
    fn test_compute_all_metrics() {
        let retrieved = vec![chunk_id(1), chunk_id(4), chunk_id(2), chunk_id(5)];
        let relevant: HashSet<_> = [chunk_id(1), chunk_id(2), chunk_id(3)].into();
        let k_values = vec![1, 2, 5, 10];

        let metrics = RetrievalMetrics::compute(&retrieved, &relevant, &k_values);

        assert!(!metrics.recall.is_empty());
        assert!(!metrics.precision.is_empty());
        assert!(!metrics.ndcg.is_empty());
        assert!(metrics.mrr > 0.0);
    }

    // ============ Aggregation Tests ============

    #[test]
    fn test_aggregate_empty() {
        let agg = AggregatedMetrics::aggregate(&[]);
        assert_eq!(agg.query_count, 0);
    }

    #[test]
    fn test_aggregate_single() {
        let retrieved = vec![chunk_id(1), chunk_id(2)];
        let relevant: HashSet<_> = [chunk_id(1), chunk_id(2)].into();
        let metrics = RetrievalMetrics::compute(&retrieved, &relevant, &[1, 2]);

        let agg = AggregatedMetrics::aggregate(&[metrics]);
        assert_eq!(agg.query_count, 1);
        assert!((agg.mean_mrr - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_aggregate_multiple() {
        let metrics1 = RetrievalMetrics {
            mrr: 1.0,
            map: 1.0,
            recall: [(1, 1.0), (2, 1.0)].into(),
            precision: [(1, 1.0), (2, 1.0)].into(),
            ndcg: [(1, 1.0), (2, 1.0)].into(),
        };
        let metrics2 = RetrievalMetrics {
            mrr: 0.5,
            map: 0.5,
            recall: [(1, 0.5), (2, 0.5)].into(),
            precision: [(1, 0.5), (2, 0.5)].into(),
            ndcg: [(1, 0.5), (2, 0.5)].into(),
        };

        let agg = AggregatedMetrics::aggregate(&[metrics1, metrics2]);

        assert_eq!(agg.query_count, 2);
        assert!((agg.mean_mrr - 0.75).abs() < 0.001);
        assert!((agg.map - 0.75).abs() < 0.001);
    }

    // ============ Property-Based Tests ============

    use proptest::prelude::*;

    proptest! {
        #[test]
        fn prop_recall_bounded(
            retrieved_ids in prop::collection::vec(0u128..100, 1..20),
            relevant_ids in prop::collection::vec(0u128..100, 1..10),
            k in 1usize..20
        ) {
            let retrieved: Vec<_> = retrieved_ids.into_iter().map(chunk_id).collect();
            let relevant: HashSet<_> = relevant_ids.into_iter().map(chunk_id).collect();

            let recall = RetrievalMetrics::recall_at_k(&retrieved, &relevant, k);
            prop_assert!(recall >= 0.0);
            prop_assert!(recall <= 1.0);
        }

        #[test]
        fn prop_precision_bounded(
            retrieved_ids in prop::collection::vec(0u128..100, 1..20),
            relevant_ids in prop::collection::vec(0u128..100, 1..10),
            k in 1usize..20
        ) {
            let retrieved: Vec<_> = retrieved_ids.into_iter().map(chunk_id).collect();
            let relevant: HashSet<_> = relevant_ids.into_iter().map(chunk_id).collect();

            let precision = RetrievalMetrics::precision_at_k(&retrieved, &relevant, k);
            prop_assert!(precision >= 0.0);
            prop_assert!(precision <= 1.0);
        }

        #[test]
        fn prop_mrr_bounded(
            retrieved_ids in prop::collection::vec(0u128..100, 1..20),
            relevant_ids in prop::collection::vec(0u128..100, 1..10)
        ) {
            let retrieved: Vec<_> = retrieved_ids.into_iter().map(chunk_id).collect();
            let relevant: HashSet<_> = relevant_ids.into_iter().map(chunk_id).collect();

            let mrr = RetrievalMetrics::mean_reciprocal_rank(&retrieved, &relevant);
            prop_assert!(mrr >= 0.0);
            prop_assert!(mrr <= 1.0);
        }

        #[test]
        fn prop_ndcg_bounded(
            retrieved_ids in prop::collection::vec(0u128..100, 1..20),
            relevant_ids in prop::collection::vec(0u128..100, 1..10),
            k in 1usize..20
        ) {
            let retrieved: Vec<_> = retrieved_ids.into_iter().map(chunk_id).collect();
            let relevant: HashSet<_> = relevant_ids.into_iter().map(chunk_id).collect();

            let ndcg = RetrievalMetrics::ndcg_at_k(&retrieved, &relevant, k);
            prop_assert!(ndcg >= 0.0);
            prop_assert!(ndcg <= 1.0);
        }
    }
}
