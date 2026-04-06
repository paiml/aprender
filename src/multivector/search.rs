//! WARP search algorithm components
//!
//! This module implements the three phases of WARP search:
//!
//! 1. **Centroid Selection** - For each query token, find top-nprobe centroids
//! 2. **Candidate Scoring** - Decompress and score tokens from selected centroids
//! 3. **Score Merging** - Aggregate per-token scores into document scores via MaxSim

use crate::multivector::{codec::ResidualCodec, types::WarpSearchConfig, MultiVectorEmbedding};
use crate::ChunkId;
use std::collections::HashMap;

/// Phase 1: Select top centroids per query token.
///
/// For each query token, compute its similarity with all centroids and
/// select the top-nprobe centroids above the score threshold.
pub struct CentroidSelector;

impl CentroidSelector {
    /// Select top centroids for each query token.
    ///
    /// # Arguments
    ///
    /// * `query` - Query multi-vector embedding
    /// * `centroids` - Flattened centroid vectors [num_centroids × dim]
    /// * `dim` - Token embedding dimension
    /// * `config` - Search configuration
    ///
    /// # Returns
    ///
    /// For each query token, a vector of (centroid_id, centroid_score) pairs
    /// sorted by score descending.
    #[must_use]
    pub fn select(
        query: &MultiVectorEmbedding,
        centroids: &[f32],
        dim: usize,
        config: &WarpSearchConfig,
    ) -> Vec<Vec<(usize, f32)>> {
        if dim == 0 || centroids.is_empty() {
            return query.tokens().map(|_| vec![]).collect();
        }
        let num_centroids = centroids.len() / dim;

        query
            .tokens()
            .map(|query_token| {
                // Compute scores with all centroids
                let mut scores: Vec<(usize, f32)> = (0..num_centroids)
                    .map(|c| {
                        let centroid = &centroids[c * dim..(c + 1) * dim];
                        let score = Self::dot_product(query_token, centroid);
                        (c, score)
                    })
                    .collect();

                // Sort by score descending
                scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

                // Take top nprobe, filtered by threshold
                scores
                    .into_iter()
                    .take(config.nprobe as usize)
                    .filter(|(_, score)| *score >= config.centroid_score_threshold)
                    .collect()
            })
            .collect()
    }

    /// Batch compute centroid scores for a single query token.
    ///
    /// Returns scores for all centroids sorted by score descending.
    #[must_use]
    pub fn batch_scores(query_token: &[f32], centroids: &[f32], dim: usize) -> Vec<(usize, f32)> {
        if dim == 0 || centroids.is_empty() {
            return vec![];
        }
        let num_centroids = centroids.len() / dim;

        let mut scores: Vec<(usize, f32)> = (0..num_centroids)
            .map(|c| {
                let centroid = &centroids[c * dim..(c + 1) * dim];
                let score = Self::dot_product(query_token, centroid);
                (c, score)
            })
            .collect();

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores
    }

    fn dot_product(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }
}

/// Phase 2: Score candidates from a centroid.
///
/// For a single query token and centroid, decompress and score all
/// document tokens assigned to that centroid.
pub struct CandidateScorer;

impl CandidateScorer {
    /// Score candidates from a centroid for one query token.
    ///
    /// # Arguments
    ///
    /// * `query_token` - Query embedding for this token
    /// * `centroid_id` - Selected centroid ID
    /// * `centroid_score` - Precomputed q · c
    /// * `codec` - Residual codec for decompression
    /// * `sizes` - Number of tokens per centroid
    /// * `offsets` - Cumulative offsets per centroid
    /// * `chunk_ids` - Chunk IDs for all tokens
    /// * `token_indices` - Token indices within chunks
    /// * `residuals` - Packed residuals for all tokens
    /// * `bytes_per_residual` - Bytes per packed residual
    ///
    /// # Returns
    ///
    /// Vector of (ChunkId, token_index, score) for all candidates.
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn score(
        query_token: &[f32],
        centroid_id: usize,
        centroid_score: f32,
        codec: &ResidualCodec,
        sizes: &[usize],
        offsets: &[usize],
        chunk_ids: &[ChunkId],
        token_indices: &[u16],
        residuals: &[u8],
        bytes_per_residual: usize,
    ) -> Vec<(ChunkId, u16, f32)> {
        let size = sizes.get(centroid_id).copied().unwrap_or(0);
        if size == 0 {
            return Vec::new();
        }

        let offset = offsets.get(centroid_id).copied().unwrap_or(0);

        (0..size)
            .map(|i| {
                let idx = offset + i;
                let chunk_id = chunk_ids[idx];
                let token_idx = token_indices[idx];

                let residual_start = idx * bytes_per_residual;
                let residual_end = residual_start + bytes_per_residual;
                let residual = &residuals[residual_start..residual_end];

                let score =
                    codec.decompress_score(query_token, centroid_id, centroid_score, residual);

                (chunk_id, token_idx, score)
            })
            .collect()
    }

    /// Score a single candidate.
    #[must_use]
    pub fn score_single(
        query_token: &[f32],
        centroid_id: usize,
        centroid_score: f32,
        codec: &ResidualCodec,
        residual: &[u8],
    ) -> f32 {
        codec.decompress_score(query_token, centroid_id, centroid_score, residual)
    }
}

/// Phase 3: Merge per-token scores into document scores via MaxSim.
///
/// MaxSim computes: score(Q, D) = Σ_i max_j(q_i · d_j)
///
/// For each query token, find the maximum score with any document token,
/// then sum across query tokens.
pub struct ScoreMerger;

impl ScoreMerger {
    /// Merge per-token scores into document scores via MaxSim.
    ///
    /// # Arguments
    ///
    /// * `token_scores` - For each query token: (ChunkId, doc_token_idx, score)
    /// * `k` - Number of top results to return
    ///
    /// # Returns
    ///
    /// Vector of (ChunkId, total_score) sorted by score descending.
    #[must_use]
    pub fn merge(token_scores: Vec<Vec<(ChunkId, u16, f32)>>, k: usize) -> Vec<(ChunkId, f32)> {
        if token_scores.is_empty() {
            return Vec::new();
        }

        let num_query_tokens = token_scores.len();

        // For each document, track max score per query token
        let mut doc_token_maxes: HashMap<ChunkId, Vec<f32>> = HashMap::new();

        for (query_token_idx, scores) in token_scores.into_iter().enumerate() {
            for (chunk_id, _doc_token_idx, score) in scores {
                let maxes = doc_token_maxes
                    .entry(chunk_id)
                    .or_insert_with(|| vec![f32::NEG_INFINITY; num_query_tokens]);

                if score > maxes[query_token_idx] {
                    maxes[query_token_idx] = score;
                }
            }
        }

        // Sum max scores across query tokens
        let mut doc_scores: Vec<(ChunkId, f32)> = doc_token_maxes
            .into_iter()
            .map(|(chunk_id, maxes)| {
                let score: f32 = maxes.into_iter().filter(|&s| s > f32::NEG_INFINITY).sum();
                (chunk_id, score)
            })
            .collect();

        // Sort by score descending
        doc_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Take top-k
        doc_scores.truncate(k);
        doc_scores
    }

    /// Merge scores for a single document across query tokens.
    ///
    /// This is useful when you have per-token scores already grouped by document.
    #[must_use]
    pub fn merge_single_doc(token_max_scores: &[f32]) -> f32 {
        token_max_scores.iter().filter(|&&s| s > f32::NEG_INFINITY).sum()
    }
}

/// Compute exact MaxSim score (for testing/comparison).
///
/// This computes the full MaxSim score without compression:
/// score(Q, D) = Σ_i max_j(q_i · d_j)
#[must_use]
pub fn exact_maxsim(query: &MultiVectorEmbedding, doc: &MultiVectorEmbedding) -> f32 {
    query
        .tokens()
        .map(|q| doc.tokens().map(|d| dot_product(q, d)).fold(f32::NEG_INFINITY, f32::max))
        .filter(|&s| s > f32::NEG_INFINITY)
        .sum()
}

/// Compute dot product between two vectors.
#[inline]
fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn generate_embedding(num_tokens: usize, dim: usize, seed: u64) -> MultiVectorEmbedding {
        let mut embeddings = Vec::with_capacity(num_tokens * dim);
        let mut rng = seed;

        for _ in 0..(num_tokens * dim) {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let val = ((rng >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0;
            embeddings.push(val);
        }

        MultiVectorEmbedding::new(embeddings, num_tokens, dim)
    }

    fn chunk_id(n: u128) -> ChunkId {
        ChunkId(uuid::Uuid::from_u128(n))
    }

    // ============ CentroidSelector Tests ============

    #[test]
    fn test_centroid_selector_basic() {
        let query = generate_embedding(2, 4, 42);

        // Create 4 centroids
        let centroids = vec![
            1.0, 0.0, 0.0, 0.0, // centroid 0
            0.0, 1.0, 0.0, 0.0, // centroid 1
            0.0, 0.0, 1.0, 0.0, // centroid 2
            0.0, 0.0, 0.0, 1.0, // centroid 3
        ];

        let config = WarpSearchConfig::with_k(10).nprobe(2).centroid_score_threshold(-1.0); // Accept all

        let selected = CentroidSelector::select(&query, &centroids, 4, &config);

        assert_eq!(selected.len(), 2); // 2 query tokens
        assert!(selected[0].len() <= 2); // nprobe = 2
    }

    #[test]
    fn test_centroid_selector_threshold() {
        let query = MultiVectorEmbedding::new(vec![1.0, 0.0, 0.0, 0.0], 1, 4);

        let centroids = vec![
            1.0, 0.0, 0.0, 0.0, // centroid 0: score = 1.0
            0.0, 1.0, 0.0, 0.0, // centroid 1: score = 0.0
            0.5, 0.5, 0.0, 0.0, // centroid 2: score = 0.5
            0.0, 0.0, 1.0, 0.0, // centroid 3: score = 0.0
        ];

        let config = WarpSearchConfig::with_k(10).nprobe(4).centroid_score_threshold(0.4);

        let selected = CentroidSelector::select(&query, &centroids, 4, &config);

        // Only centroids with score >= 0.4 should be selected
        assert_eq!(selected.len(), 1);
        assert!(selected[0].len() <= 2); // centroid 0 (1.0) and centroid 2 (0.5)
    }

    #[test]
    fn test_centroid_selector_sorted() {
        let query = MultiVectorEmbedding::new(vec![0.5, 0.5, 0.0, 0.0], 1, 4);

        let centroids = vec![
            1.0, 0.0, 0.0, 0.0, // centroid 0
            0.0, 1.0, 0.0, 0.0, // centroid 1
            0.5, 0.5, 0.0, 0.0, // centroid 2 (best match)
            0.0, 0.0, 1.0, 0.0, // centroid 3
        ];

        let config = WarpSearchConfig::with_k(10).nprobe(4).centroid_score_threshold(-1.0);

        let selected = CentroidSelector::select(&query, &centroids, 4, &config);

        // Results should be sorted by score descending
        assert!(!selected[0].is_empty());
        for i in 1..selected[0].len() {
            assert!(selected[0][i - 1].1 >= selected[0][i].1);
        }
    }

    /// Regression test for paiml/trueno-rag#15: dim=0 must not divide by zero.
    #[test]
    fn test_centroid_selector_dim_zero_no_panic() {
        let query = MultiVectorEmbedding::from_tokens(&[]);
        let centroids: Vec<f32> = vec![];
        let config = WarpSearchConfig::with_k(10);

        let selected = CentroidSelector::select(&query, &centroids, 0, &config);
        assert!(selected.is_empty());
    }

    /// Regression test for paiml/trueno-rag#15: batch_scores with dim=0.
    #[test]
    fn test_batch_scores_dim_zero_no_panic() {
        let scores = CentroidSelector::batch_scores(&[], &[], 0);
        assert!(scores.is_empty());
    }

    #[test]
    fn test_batch_scores() {
        let query_token = vec![1.0, 0.0, 0.0, 0.0];
        let centroids = vec![
            1.0, 0.0, 0.0, 0.0, // centroid 0
            0.0, 1.0, 0.0, 0.0, // centroid 1
        ];

        let scores = CentroidSelector::batch_scores(&query_token, &centroids, 4);

        assert_eq!(scores.len(), 2);
        assert_eq!(scores[0].0, 0); // Best match is centroid 0
        assert!((scores[0].1 - 1.0).abs() < 1e-6);
    }

    // ============ CandidateScorer Tests ============

    #[test]
    fn test_candidate_scorer_empty_centroid() {
        let query_token = vec![1.0, 0.0, 0.0, 0.0];
        let codec = create_test_codec();

        let sizes = vec![0, 5, 3]; // centroid 0 is empty
        let offsets = vec![0, 0, 5];
        let chunk_ids: Vec<ChunkId> = vec![];
        let token_indices: Vec<u16> = vec![];
        let residuals: Vec<u8> = vec![];

        let results = CandidateScorer::score(
            &query_token,
            0, // empty centroid
            0.5,
            &codec,
            &sizes,
            &offsets,
            &chunk_ids,
            &token_indices,
            &residuals,
            2, // bytes per residual
        );

        assert!(results.is_empty());
    }

    fn create_test_codec() -> ResidualCodec {
        // Create a minimal test codec
        let embeddings = vec![0.0f32; 200 * 4]; // 200 samples, dim=4
        ResidualCodec::train(&embeddings, 4, 4, 2, 3).unwrap()
    }

    // ============ ScoreMerger Tests ============

    #[test]
    fn test_score_merger_basic() {
        let token_scores = vec![
            vec![(chunk_id(1), 0, 0.9), (chunk_id(2), 0, 0.8), (chunk_id(1), 1, 0.7)],
            vec![(chunk_id(1), 0, 0.6), (chunk_id(2), 0, 0.5), (chunk_id(3), 0, 0.4)],
        ];

        let results = ScoreMerger::merge(token_scores, 10);

        // chunk_id(1): max(0.9, 0.7) + max(0.6) = 0.9 + 0.6 = 1.5
        // chunk_id(2): max(0.8) + max(0.5) = 0.8 + 0.5 = 1.3
        // chunk_id(3): 0 + max(0.4) = 0.4

        assert_eq!(results.len(), 3);
        assert_eq!(results[0].0, chunk_id(1));
        assert!((results[0].1 - 1.5).abs() < 0.001);
    }

    #[test]
    fn test_score_merger_empty() {
        let token_scores: Vec<Vec<(ChunkId, u16, f32)>> = vec![];
        let results = ScoreMerger::merge(token_scores, 10);
        assert!(results.is_empty());
    }

    #[test]
    fn test_score_merger_respects_k() {
        let token_scores = vec![vec![
            (chunk_id(1), 0, 0.9),
            (chunk_id(2), 0, 0.8),
            (chunk_id(3), 0, 0.7),
            (chunk_id(4), 0, 0.6),
            (chunk_id(5), 0, 0.5),
        ]];

        let results = ScoreMerger::merge(token_scores, 3);
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_score_merger_sorted_descending() {
        let token_scores =
            vec![vec![(chunk_id(1), 0, 0.3), (chunk_id(2), 0, 0.9), (chunk_id(3), 0, 0.6)]];

        let results = ScoreMerger::merge(token_scores, 10);

        assert_eq!(results[0].0, chunk_id(2)); // highest
        assert_eq!(results[1].0, chunk_id(3));
        assert_eq!(results[2].0, chunk_id(1)); // lowest
    }

    #[test]
    fn test_merge_single_doc() {
        let scores = vec![0.9, 0.6, f32::NEG_INFINITY, 0.3];
        let total = ScoreMerger::merge_single_doc(&scores);

        assert!((total - 1.8).abs() < 0.001); // 0.9 + 0.6 + 0.3
    }

    // ============ Exact MaxSim Tests ============

    #[test]
    fn test_exact_maxsim_identical() {
        let emb = generate_embedding(3, 4, 42);
        let score = exact_maxsim(&emb, &emb);

        // Self-similarity: for normalized vectors, this should be num_tokens
        // For non-normalized, just check it's positive
        assert!(score > 0.0);
    }

    #[test]
    fn test_exact_maxsim_orthogonal() {
        let query = MultiVectorEmbedding::new(vec![1.0, 0.0, 0.0, 0.0], 1, 4);
        let doc = MultiVectorEmbedding::new(vec![0.0, 1.0, 0.0, 0.0], 1, 4);

        let score = exact_maxsim(&query, &doc);
        assert!((score - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_exact_maxsim_aligned() {
        let query = MultiVectorEmbedding::new(vec![1.0, 0.0, 0.0, 0.0], 1, 4);
        let doc = MultiVectorEmbedding::new(vec![1.0, 0.0, 0.0, 0.0], 1, 4);

        let score = exact_maxsim(&query, &doc);
        assert!((score - 1.0).abs() < 1e-6);
    }

    // ============ Property-Based Tests ============

    use proptest::prelude::*;

    proptest! {
        #[test]
        fn prop_maxsim_non_negative_for_unit_vectors(
            num_q in 1usize..5,
            num_d in 1usize..5
        ) {
            // Generate unit vectors
            let query = generate_embedding(num_q, 4, 123);
            let doc = generate_embedding(num_d, 4, 456);

            let score = exact_maxsim(&query, &doc);

            // MaxSim can be negative for non-unit vectors, but the test
            // just checks it doesn't panic
            prop_assert!(score.is_finite());
        }

        #[test]
        fn prop_merger_results_count_bounded_by_k(
            k in 1usize..20,
            num_docs in 1usize..50
        ) {
            let token_scores = vec![
                (0..num_docs)
                    .map(|i| (chunk_id(i as u128), 0u16, i as f32 / 100.0))
                    .collect()
            ];

            let results = ScoreMerger::merge(token_scores, k);
            prop_assert!(results.len() <= k);
            prop_assert!(results.len() <= num_docs);
        }

        #[test]
        fn prop_centroid_selector_respects_nprobe(
            nprobe in 1u32..10
        ) {
            let query = generate_embedding(2, 4, 42);
            let centroids = vec![0.5f32; 20 * 4]; // 20 centroids

            let config = WarpSearchConfig::with_k(10)
                .nprobe(nprobe)
                .centroid_score_threshold(-10.0); // Accept all

            let selected = CentroidSelector::select(&query, &centroids, 4, &config);

            for token_selection in selected {
                prop_assert!(token_selection.len() <= nprobe as usize);
            }
        }
    }
}
