//! Score fusion strategies for hybrid retrieval

use crate::ChunkId;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Strategy for fusing dense and sparse retrieval results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FusionStrategy {
    /// Reciprocal Rank Fusion
    RRF {
        /// Constant k for RRF (typically 60)
        k: f32,
    },
    /// Linear combination of normalized scores
    Linear {
        /// Weight for dense results (sparse weight = 1 - dense_weight)
        dense_weight: f32,
    },
    /// Convex combination after min-max normalization
    Convex {
        /// Alpha parameter for convex combination
        alpha: f32,
    },
    /// Distribution-Based Score Fusion
    DBSF,
    /// Take union of results, prefer dense ranking for ties
    Union,
    /// Intersection: only return results in both
    Intersection,
    /// Three-way weighted combination of dense, sparse, and multi-vector results
    #[cfg(feature = "multivector")]
    ThreeWay {
        /// Weight for dense results
        dense_weight: f32,
        /// Weight for sparse results
        sparse_weight: f32,
        /// Weight for multi-vector results
        multivector_weight: f32,
    },
}

impl Default for FusionStrategy {
    fn default() -> Self {
        Self::RRF { k: 60.0 }
    }
}

impl FusionStrategy {
    /// Fuse dense and sparse retrieval results
    #[must_use]
    pub fn fuse(
        &self,
        dense_results: &[(ChunkId, f32)],
        sparse_results: &[(ChunkId, f32)],
    ) -> Vec<(ChunkId, f32)> {
        match self {
            FusionStrategy::RRF { k } => {
                Self::reciprocal_rank_fusion(dense_results, sparse_results, *k)
            }
            FusionStrategy::Linear { dense_weight } => {
                Self::linear_fusion(dense_results, sparse_results, *dense_weight)
            }
            FusionStrategy::Convex { alpha } => {
                Self::convex_fusion(dense_results, sparse_results, *alpha)
            }
            FusionStrategy::DBSF => Self::dbsf_fusion(dense_results, sparse_results),
            FusionStrategy::Union => Self::union_fusion(dense_results, sparse_results),
            FusionStrategy::Intersection => {
                Self::intersection_fusion(dense_results, sparse_results)
            }
            // ThreeWay is for fuse_three(); fallback to RRF for two-way fusion
            #[cfg(feature = "multivector")]
            FusionStrategy::ThreeWay { .. } => {
                Self::reciprocal_rank_fusion(dense_results, sparse_results, 60.0)
            }
        }
    }

    /// Reciprocal Rank Fusion (RRF)
    ///
    /// RRF score = Σ 1 / (k + rank)
    fn reciprocal_rank_fusion(
        dense: &[(ChunkId, f32)],
        sparse: &[(ChunkId, f32)],
        k: f32,
    ) -> Vec<(ChunkId, f32)> {
        let mut scores: HashMap<ChunkId, f32> = HashMap::new();

        for (rank, (id, _)) in dense.iter().enumerate() {
            *scores.entry(*id).or_insert(0.0) += 1.0 / (k + rank as f32 + 1.0);
        }

        for (rank, (id, _)) in sparse.iter().enumerate() {
            *scores.entry(*id).or_insert(0.0) += 1.0 / (k + rank as f32 + 1.0);
        }

        Self::sort_by_score(scores)
    }

    /// Linear fusion with weighted scores
    fn linear_fusion(
        dense: &[(ChunkId, f32)],
        sparse: &[(ChunkId, f32)],
        dense_weight: f32,
    ) -> Vec<(ChunkId, f32)> {
        let sparse_weight = 1.0 - dense_weight;

        // Normalize scores to [0, 1]
        let dense_normalized = Self::min_max_normalize(dense);
        let sparse_normalized = Self::min_max_normalize(sparse);

        let mut scores: HashMap<ChunkId, f32> = HashMap::new();

        for (id, score) in dense_normalized {
            *scores.entry(id).or_insert(0.0) += dense_weight * score;
        }

        for (id, score) in sparse_normalized {
            *scores.entry(id).or_insert(0.0) += sparse_weight * score;
        }

        Self::sort_by_score(scores)
    }

    /// Convex combination fusion
    fn convex_fusion(
        dense: &[(ChunkId, f32)],
        sparse: &[(ChunkId, f32)],
        alpha: f32,
    ) -> Vec<(ChunkId, f32)> {
        // Similar to linear but uses alpha for dense and (1-alpha) for sparse
        Self::linear_fusion(dense, sparse, alpha)
    }

    /// Distribution-Based Score Fusion
    fn dbsf_fusion(dense: &[(ChunkId, f32)], sparse: &[(ChunkId, f32)]) -> Vec<(ChunkId, f32)> {
        // Z-score normalization for each result set
        let dense_normalized = Self::z_score_normalize(dense);
        let sparse_normalized = Self::z_score_normalize(sparse);

        let mut scores: HashMap<ChunkId, f32> = HashMap::new();

        for (id, score) in dense_normalized {
            *scores.entry(id).or_insert(0.0) += score;
        }

        for (id, score) in sparse_normalized {
            *scores.entry(id).or_insert(0.0) += score;
        }

        Self::sort_by_score(scores)
    }

    /// Union fusion: combine all results, preferring higher-ranked source for ties
    fn union_fusion(dense: &[(ChunkId, f32)], sparse: &[(ChunkId, f32)]) -> Vec<(ChunkId, f32)> {
        let mut scores: HashMap<ChunkId, (f32, usize)> = HashMap::new();

        // Dense results get preference (lower rank = better)
        for (rank, (id, score)) in dense.iter().enumerate() {
            scores.insert(*id, (*score, rank));
        }

        // Add sparse results if not already present
        for (rank, (id, score)) in sparse.iter().enumerate() {
            scores.entry(*id).or_insert((*score, dense.len() + rank));
        }

        let mut results: Vec<_> = scores.into_iter().collect();
        results.sort_by(|a, b| a.1 .1.cmp(&b.1 .1)); // Sort by rank
        results.into_iter().map(|(id, (score, _))| (id, score)).collect()
    }

    /// Intersection fusion: only keep results in both sets
    fn intersection_fusion(
        dense: &[(ChunkId, f32)],
        sparse: &[(ChunkId, f32)],
    ) -> Vec<(ChunkId, f32)> {
        let dense_ids: HashMap<ChunkId, f32> = dense.iter().copied().collect();
        let sparse_ids: HashMap<ChunkId, f32> = sparse.iter().copied().collect();

        let mut scores: HashMap<ChunkId, f32> = HashMap::new();

        for (id, dense_score) in &dense_ids {
            if let Some(sparse_score) = sparse_ids.get(id) {
                // Average the scores
                scores.insert(*id, (dense_score + sparse_score) / 2.0);
            }
        }

        Self::sort_by_score(scores)
    }

    /// Min-max normalization to [0, 1]
    fn min_max_normalize(results: &[(ChunkId, f32)]) -> Vec<(ChunkId, f32)> {
        if results.is_empty() {
            return Vec::new();
        }

        let scores: Vec<f32> = results.iter().map(|(_, s)| *s).collect();
        let min = scores.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let range = max - min;

        if range.abs() < f32::EPSILON {
            // All scores are the same
            return results.iter().map(|(id, _)| (*id, 1.0)).collect();
        }

        results.iter().map(|(id, score)| (*id, (score - min) / range)).collect()
    }

    /// Z-score normalization
    fn z_score_normalize(results: &[(ChunkId, f32)]) -> Vec<(ChunkId, f32)> {
        if results.is_empty() {
            return Vec::new();
        }

        let scores: Vec<f32> = results.iter().map(|(_, s)| *s).collect();
        let n = scores.len() as f32;
        let mean: f32 = scores.iter().sum::<f32>() / n;
        let variance: f32 = scores.iter().map(|s| (s - mean).powi(2)).sum::<f32>() / n;
        let std_dev = variance.sqrt();

        if std_dev.abs() < f32::EPSILON {
            return results.iter().map(|(id, _)| (*id, 0.0)).collect();
        }

        results.iter().map(|(id, score)| (*id, (score - mean) / std_dev)).collect()
    }

    /// Sort results by score descending
    fn sort_by_score(scores: HashMap<ChunkId, f32>) -> Vec<(ChunkId, f32)> {
        let mut results: Vec<_> = scores.into_iter().collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results
    }

    /// Fuse three result sets: dense, sparse, and multi-vector.
    ///
    /// This method combines results from all three retrieval methods using
    /// a weighted linear combination after min-max normalization.
    ///
    /// # Arguments
    ///
    /// * `dense` - Dense (single-vector) retrieval results
    /// * `sparse` - Sparse (BM25) retrieval results
    /// * `multivector` - Multi-vector (ColBERT-style) retrieval results
    ///
    /// # Returns
    ///
    /// Fused results sorted by combined score descending.
    #[cfg(feature = "multivector")]
    #[must_use]
    pub fn fuse_three(
        &self,
        dense: &[(ChunkId, f32)],
        sparse: &[(ChunkId, f32)],
        multivector: &[(ChunkId, f32)],
    ) -> Vec<(ChunkId, f32)> {
        if let FusionStrategy::ThreeWay { dense_weight, sparse_weight, multivector_weight } = self {
            Self::three_way_linear(
                dense,
                sparse,
                multivector,
                *dense_weight,
                *sparse_weight,
                *multivector_weight,
            )
        } else {
            // For other strategies, do pairwise fusion
            // First fuse dense and sparse
            let dense_sparse = self.fuse(dense, sparse);
            // Then fuse with multivector (treating combined as "dense")
            self.fuse(&dense_sparse, multivector)
        }
    }

    /// Three-way linear fusion with explicit weights.
    #[cfg(feature = "multivector")]
    fn three_way_linear(
        dense: &[(ChunkId, f32)],
        sparse: &[(ChunkId, f32)],
        multivector: &[(ChunkId, f32)],
        w_dense: f32,
        w_sparse: f32,
        w_multi: f32,
    ) -> Vec<(ChunkId, f32)> {
        let mut scores: HashMap<ChunkId, f32> = HashMap::new();

        // Normalize and weight each source
        let dense_norm = Self::min_max_normalize(dense);
        let sparse_norm = Self::min_max_normalize(sparse);
        let multi_norm = Self::min_max_normalize(multivector);

        for (id, score) in dense_norm {
            *scores.entry(id).or_insert(0.0) += w_dense * score;
        }
        for (id, score) in sparse_norm {
            *scores.entry(id).or_insert(0.0) += w_sparse * score;
        }
        for (id, score) in multi_norm {
            *scores.entry(id).or_insert(0.0) += w_multi * score;
        }

        Self::sort_by_score(scores)
    }

    /// Create a three-way fusion strategy with the given weights.
    ///
    /// Weights should ideally sum to 1.0 but this is not enforced.
    #[cfg(feature = "multivector")]
    #[must_use]
    pub fn three_way(dense_weight: f32, sparse_weight: f32, multivector_weight: f32) -> Self {
        Self::ThreeWay { dense_weight, sparse_weight, multivector_weight }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chunk_id(n: u128) -> ChunkId {
        ChunkId(uuid::Uuid::from_u128(n))
    }

    // ============ FusionStrategy Default Tests ============

    #[test]
    fn test_fusion_strategy_default() {
        let strategy = FusionStrategy::default();
        match strategy {
            FusionStrategy::RRF { k } => assert!((k - 60.0).abs() < 0.01),
            _ => panic!("Expected RRF"),
        }
    }

    #[test]
    fn test_fusion_strategy_serialization() {
        let strategy = FusionStrategy::Linear { dense_weight: 0.7 };
        let json = serde_json::to_string(&strategy).unwrap();
        let deserialized: FusionStrategy = serde_json::from_str(&json).unwrap();

        match deserialized {
            FusionStrategy::Linear { dense_weight } => {
                assert!((dense_weight - 0.7).abs() < 0.01);
            }
            _ => panic!("Wrong strategy type"),
        }
    }

    // ============ RRF Tests ============

    #[test]
    fn test_rrf_empty() {
        let strategy = FusionStrategy::RRF { k: 60.0 };
        let results = strategy.fuse(&[], &[]);
        assert!(results.is_empty());
    }

    #[test]
    fn test_rrf_dense_only() {
        let strategy = FusionStrategy::RRF { k: 60.0 };
        let dense = vec![(chunk_id(1), 0.9), (chunk_id(2), 0.8)];
        let results = strategy.fuse(&dense, &[]);

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, chunk_id(1));
        assert_eq!(results[1].0, chunk_id(2));
    }

    #[test]
    fn test_rrf_sparse_only() {
        let strategy = FusionStrategy::RRF { k: 60.0 };
        let sparse = vec![(chunk_id(1), 0.9), (chunk_id(2), 0.8)];
        let results = strategy.fuse(&[], &sparse);

        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_rrf_combines_ranks() {
        let strategy = FusionStrategy::RRF { k: 60.0 };

        // Chunk 1 is first in both -> highest RRF
        // Chunk 2 is second in dense, first in sparse
        // Chunk 3 is first in dense, second in sparse
        let dense = vec![(chunk_id(1), 0.9), (chunk_id(2), 0.8)];
        let sparse = vec![(chunk_id(1), 0.9), (chunk_id(3), 0.8)];

        let results = strategy.fuse(&dense, &sparse);

        assert_eq!(results.len(), 3);
        // Chunk 1 should be first (appears first in both)
        assert_eq!(results[0].0, chunk_id(1));
    }

    #[test]
    fn test_rrf_score_calculation() {
        let strategy = FusionStrategy::RRF { k: 60.0 };

        let dense = vec![(chunk_id(1), 1.0)]; // rank 0
        let sparse = vec![(chunk_id(1), 1.0)]; // rank 0

        let results = strategy.fuse(&dense, &sparse);

        // RRF score = 1/(60+1) + 1/(60+1) = 2/61
        let expected = 2.0 / 61.0;
        assert!((results[0].1 - expected).abs() < 0.001);
    }

    // ============ Linear Fusion Tests ============

    #[test]
    fn test_linear_empty() {
        let strategy = FusionStrategy::Linear { dense_weight: 0.5 };
        let results = strategy.fuse(&[], &[]);
        assert!(results.is_empty());
    }

    #[test]
    fn test_linear_dense_only() {
        let strategy = FusionStrategy::Linear { dense_weight: 0.7 };
        let dense = vec![(chunk_id(1), 1.0), (chunk_id(2), 0.5)];
        let results = strategy.fuse(&dense, &[]);

        // Scores should be weighted by dense_weight
        assert!(!results.is_empty());
    }

    #[test]
    fn test_linear_equal_weights() {
        let strategy = FusionStrategy::Linear { dense_weight: 0.5 };

        let dense = vec![(chunk_id(1), 1.0)];
        let sparse = vec![(chunk_id(1), 1.0)];

        let results = strategy.fuse(&dense, &sparse);

        // With equal weights and max scores, should get 1.0
        assert!((results[0].1 - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_linear_weight_preference() {
        let strategy = FusionStrategy::Linear { dense_weight: 0.9 };

        // Chunk 1: high in dense, low in sparse
        // Chunk 2: low in dense, high in sparse
        let dense = vec![(chunk_id(1), 1.0), (chunk_id(2), 0.0)];
        let sparse = vec![(chunk_id(2), 1.0), (chunk_id(1), 0.0)];

        let results = strategy.fuse(&dense, &sparse);

        // With 0.9 dense weight, chunk 1 should win
        assert_eq!(results[0].0, chunk_id(1));
    }

    // ============ Convex Fusion Tests ============

    #[test]
    fn test_convex_same_as_linear() {
        let linear = FusionStrategy::Linear { dense_weight: 0.6 };
        let convex = FusionStrategy::Convex { alpha: 0.6 };

        let dense = vec![(chunk_id(1), 0.9), (chunk_id(2), 0.5)];
        let sparse = vec![(chunk_id(2), 0.8), (chunk_id(3), 0.4)];

        let linear_results = linear.fuse(&dense, &sparse);
        let convex_results = convex.fuse(&dense, &sparse);

        assert_eq!(linear_results.len(), convex_results.len());
    }

    // ============ DBSF Tests ============

    #[test]
    fn test_dbsf_empty() {
        let strategy = FusionStrategy::DBSF;
        let results = strategy.fuse(&[], &[]);
        assert!(results.is_empty());
    }

    #[test]
    fn test_dbsf_z_score() {
        let strategy = FusionStrategy::DBSF;

        let dense = vec![(chunk_id(1), 10.0), (chunk_id(2), 5.0), (chunk_id(3), 0.0)];
        let sparse = vec![(chunk_id(1), 100.0), (chunk_id(2), 50.0), (chunk_id(3), 0.0)];

        let results = strategy.fuse(&dense, &sparse);

        // Chunk 1 should still be first (highest z-score in both)
        assert_eq!(results[0].0, chunk_id(1));
    }

    // ============ Union Fusion Tests ============

    #[test]
    fn test_union_combines_all() {
        let strategy = FusionStrategy::Union;

        let dense = vec![(chunk_id(1), 0.9)];
        let sparse = vec![(chunk_id(2), 0.8)];

        let results = strategy.fuse(&dense, &sparse);

        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_union_deduplicates() {
        let strategy = FusionStrategy::Union;

        let dense = vec![(chunk_id(1), 0.9), (chunk_id(2), 0.8)];
        let sparse = vec![(chunk_id(1), 0.7), (chunk_id(3), 0.6)];

        let results = strategy.fuse(&dense, &sparse);

        // Should have 3 unique chunks
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_union_prefers_dense_rank() {
        let strategy = FusionStrategy::Union;

        let dense = vec![(chunk_id(1), 0.9)];
        let sparse = vec![(chunk_id(1), 0.5)]; // Same chunk, different score

        let results = strategy.fuse(&dense, &sparse);

        // Should use dense score
        assert!((results[0].1 - 0.9).abs() < f32::EPSILON);
    }

    // ============ Intersection Fusion Tests ============

    #[test]
    fn test_intersection_empty_no_overlap() {
        let strategy = FusionStrategy::Intersection;

        let dense = vec![(chunk_id(1), 0.9)];
        let sparse = vec![(chunk_id(2), 0.8)];

        let results = strategy.fuse(&dense, &sparse);

        assert!(results.is_empty());
    }

    #[test]
    fn test_intersection_keeps_overlap() {
        let strategy = FusionStrategy::Intersection;

        let dense = vec![(chunk_id(1), 0.8), (chunk_id(2), 0.6)];
        let sparse = vec![(chunk_id(2), 0.9), (chunk_id(3), 0.5)];

        let results = strategy.fuse(&dense, &sparse);

        // Only chunk 2 is in both
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, chunk_id(2));
    }

    #[test]
    fn test_intersection_averages_scores() {
        let strategy = FusionStrategy::Intersection;

        let dense = vec![(chunk_id(1), 0.8)];
        let sparse = vec![(chunk_id(1), 0.4)];

        let results = strategy.fuse(&dense, &sparse);

        // Average of 0.8 and 0.4 = 0.6
        assert!((results[0].1 - 0.6).abs() < 0.001);
    }

    // ============ Normalization Tests ============

    #[test]
    fn test_min_max_normalize_empty() {
        let normalized = FusionStrategy::min_max_normalize(&[]);
        assert!(normalized.is_empty());
    }

    #[test]
    fn test_min_max_normalize_single() {
        let results = vec![(chunk_id(1), 5.0)];
        let normalized = FusionStrategy::min_max_normalize(&results);

        assert_eq!(normalized.len(), 1);
        assert!((normalized[0].1 - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_min_max_normalize_range() {
        let results = vec![(chunk_id(1), 10.0), (chunk_id(2), 5.0), (chunk_id(3), 0.0)];
        let normalized = FusionStrategy::min_max_normalize(&results);

        // Max should be 1.0, min should be 0.0
        assert!((normalized[0].1 - 1.0).abs() < 0.001);
        assert!((normalized[2].1 - 0.0).abs() < 0.001);
        assert!((normalized[1].1 - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_z_score_normalize_empty() {
        let normalized = FusionStrategy::z_score_normalize(&[]);
        assert!(normalized.is_empty());
    }

    #[test]
    fn test_z_score_normalize_same_values() {
        let results = vec![(chunk_id(1), 5.0), (chunk_id(2), 5.0), (chunk_id(3), 5.0)];
        let normalized = FusionStrategy::z_score_normalize(&results);

        // All same -> all should be 0 (mean)
        for (_, score) in normalized {
            assert!(score.abs() < 0.001);
        }
    }

    // ============ Property-Based Tests ============

    use proptest::prelude::*;

    proptest! {
        #[test]
        fn prop_rrf_scores_positive(
            n_dense in 1usize..10,
            n_sparse in 1usize..10
        ) {
            let dense: Vec<_> = (0..n_dense)
                .map(|i| (chunk_id(i as u128), 1.0 - i as f32 * 0.1))
                .collect();
            let sparse: Vec<_> = (100..100 + n_sparse)
                .map(|i| (chunk_id(i as u128), 1.0 - (i - 100) as f32 * 0.1))
                .collect();

            let strategy = FusionStrategy::RRF { k: 60.0 };
            let results = strategy.fuse(&dense, &sparse);

            for (_, score) in results {
                prop_assert!(score > 0.0);
            }
        }

        #[test]
        fn prop_linear_weights_sum_to_one(dense_weight in 0.0f32..1.0) {
            let dense = vec![(chunk_id(1), 1.0)];
            let sparse = vec![(chunk_id(1), 1.0)];

            let strategy = FusionStrategy::Linear { dense_weight };
            let results = strategy.fuse(&dense, &sparse);

            // With both scores at 1.0 (after normalization), result should be 1.0
            prop_assert!((results[0].1 - 1.0).abs() < 0.01);
        }

        #[test]
        fn prop_intersection_subset_of_inputs(
            dense_ids in prop::collection::vec(0u128..100, 1..10),
            sparse_ids in prop::collection::vec(0u128..100, 1..10)
        ) {
            let dense: Vec<_> = dense_ids.iter().map(|&i| (chunk_id(i), 1.0)).collect();
            let sparse: Vec<_> = sparse_ids.iter().map(|&i| (chunk_id(i), 1.0)).collect();

            let strategy = FusionStrategy::Intersection;
            let results = strategy.fuse(&dense, &sparse);

            let dense_set: std::collections::HashSet<_> = dense_ids.iter().copied().collect();
            let sparse_set: std::collections::HashSet<_> = sparse_ids.iter().copied().collect();

            for (id, _) in results {
                let id_val = id.0.as_u128();
                prop_assert!(dense_set.contains(&id_val) && sparse_set.contains(&id_val));
            }
        }

        #[test]
        fn prop_fusion_deterministic(
            n in 1usize..5
        ) {
            let dense: Vec<_> = (0..n).map(|i| (chunk_id(i as u128), 1.0)).collect();
            let sparse: Vec<_> = (0..n).map(|i| (chunk_id(i as u128), 0.5)).collect();

            let strategy = FusionStrategy::RRF { k: 60.0 };
            let results1 = strategy.fuse(&dense, &sparse);
            let results2 = strategy.fuse(&dense, &sparse);

            prop_assert_eq!(results1.len(), results2.len());
            for ((id1, s1), (id2, s2)) in results1.iter().zip(results2.iter()) {
                prop_assert_eq!(id1, id2);
                prop_assert!((s1 - s2).abs() < 0.0001);
            }
        }
    }
}
