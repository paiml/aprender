//! WARP index with IVF structure
//!
//! This module implements the WARP index which organizes compressed token
//! embeddings by centroid for cache-efficient search. The index supports:
//!
//! - Training from sample embeddings
//! - Incremental insertion of documents
//! - Building (compacting) for efficient search
//! - MaxSim-based multi-vector search

use crate::multivector::{
    codec::ResidualCodec,
    search::{CandidateScorer, CentroidSelector, ScoreMerger},
    types::{MultiVectorEmbedding, WarpIndexConfig, WarpSearchConfig},
};
use crate::{Chunk, ChunkId, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// WARP index for efficient multi-vector retrieval.
///
/// The index organizes token embeddings by centroid assignment (IVF structure)
/// for cache-efficient access during search. Each token embedding is stored as:
/// - Centroid assignment
/// - Quantized residual (2-4 bits per dimension)
///
/// # Lifecycle
///
/// 1. Create index with `new(config)`
/// 2. Train codec with `train(samples)`
/// 3. Insert documents with `insert(chunk, embedding)`
/// 4. Build index with `build()` (compacts for efficient search)
/// 5. Search with `search(query, config)`
///
/// # Memory Layout
///
/// After `build()`, data is organized by centroid:
/// ```text
/// Centroid 0: [chunk_ids...] [token_indices...] [residuals...]
/// Centroid 1: [chunk_ids...] [token_indices...] [residuals...]
/// ...
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WarpIndex {
    /// Index configuration
    config: WarpIndexConfig,
    /// Trained residual codec (None until trained)
    codec: Option<ResidualCodec>,
    /// Number of embeddings per centroid
    sizes: Vec<usize>,
    /// Cumulative offset for each centroid's data
    offsets: Vec<usize>,
    /// Chunk IDs, sorted by centroid assignment
    chunk_ids: Vec<ChunkId>,
    /// Token indices within each chunk
    token_indices: Vec<u16>,
    /// Packed residuals, sorted by centroid
    residuals: Vec<u8>,
    /// Original chunks for result retrieval
    #[serde(skip)]
    chunks: HashMap<ChunkId, Chunk>,
    /// Pending embeddings before build
    #[serde(skip)]
    pending: Vec<(ChunkId, MultiVectorEmbedding)>,
    /// Whether the index has been built
    is_built: bool,
}

impl WarpIndex {
    /// Create a new WARP index with the given configuration.
    #[must_use]
    pub fn new(config: WarpIndexConfig) -> Self {
        Self {
            config,
            codec: None,
            sizes: Vec::new(),
            offsets: Vec::new(),
            chunk_ids: Vec::new(),
            token_indices: Vec::new(),
            residuals: Vec::new(),
            chunks: HashMap::new(),
            pending: Vec::new(),
            is_built: false,
        }
    }

    /// Get the index configuration.
    #[must_use]
    pub fn config(&self) -> &WarpIndexConfig {
        &self.config
    }

    /// Get the trained codec (if any).
    #[must_use]
    pub fn codec(&self) -> Option<&ResidualCodec> {
        self.codec.as_ref()
    }

    /// Check if the codec has been trained.
    #[must_use]
    pub fn is_trained(&self) -> bool {
        self.codec.is_some()
    }

    /// Check if the index has been built.
    #[must_use]
    pub fn is_built(&self) -> bool {
        self.is_built
    }

    /// Get the number of indexed chunks.
    #[must_use]
    pub fn num_chunks(&self) -> usize {
        self.chunks.len()
    }

    /// Get the number of indexed tokens.
    #[must_use]
    pub fn num_tokens(&self) -> usize {
        self.chunk_ids.len()
    }

    /// Check if the index is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.chunks.is_empty()
    }

    /// Get a chunk by ID.
    #[must_use]
    pub fn get_chunk(&self, id: &ChunkId) -> Option<&Chunk> {
        self.chunks.get(id)
    }

    /// Get memory usage in bytes (approximate).
    #[must_use]
    pub fn memory_usage(&self) -> usize {
        let codec_size = self
            .codec
            .as_ref()
            .map(|c| {
                c.centroids().len() * 4 // centroids
                    + c.dim() * ((1 << c.nbits()) - 1) * 4 // cutoffs
                    + c.dim() * (1 << c.nbits()) * 4 // weights
            })
            .unwrap_or(0);

        let index_size = self.chunk_ids.len() * size_of::<ChunkId>()
            + self.token_indices.len() * size_of::<u16>()
            + self.residuals.len()
            + self.sizes.len() * size_of::<usize>()
            + self.offsets.len() * size_of::<usize>();

        codec_size + index_size
    }

    /// Train the codec from sample embeddings.
    ///
    /// # Arguments
    ///
    /// * `samples` - Sample multi-vector embeddings for training
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Not enough samples for training
    /// - Configuration is invalid
    pub fn train(&mut self, samples: &[MultiVectorEmbedding]) -> Result<()> {
        // Collect all token embeddings
        let total_tokens: usize = samples.iter().map(|s| s.num_tokens()).sum();
        let min_samples = self.config.effective_min_training_samples();

        if total_tokens < min_samples {
            return Err(crate::Error::InvalidInput(format!(
                "Insufficient training tokens: {total_tokens} < {min_samples} required"
            )));
        }

        // Flatten all embeddings
        let mut all_embeddings = Vec::with_capacity(total_tokens * self.config.token_dim);
        for sample in samples {
            all_embeddings.extend_from_slice(sample.as_slice());
        }

        // Train codec
        let codec = ResidualCodec::train(
            &all_embeddings,
            self.config.token_dim,
            self.config.num_centroids,
            self.config.nbits,
            self.config.kmeans_iterations,
        )?;

        self.codec = Some(codec);
        Ok(())
    }

    /// Insert a chunk with its token embeddings.
    ///
    /// The chunk will be stored in pending state until `build()` is called.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Codec has not been trained
    /// - Index has already been built (call `rebuild()` first)
    pub fn insert(&mut self, chunk: Chunk, embedding: MultiVectorEmbedding) -> Result<()> {
        if self.codec.is_none() {
            return Err(crate::Error::InvalidInput(
                "Codec not trained - call train() first".to_string(),
            ));
        }

        if self.is_built {
            return Err(crate::Error::InvalidInput(
                "Index already built - cannot insert".to_string(),
            ));
        }

        // Contract: embedding-algebra-v1.yaml precondition (pv codegen)
        contract_pre_embedding_lookup!(embedding.as_slice());

        let chunk_id = chunk.id;
        self.chunks.insert(chunk_id, chunk);
        self.pending.push((chunk_id, embedding));

        Ok(())
    }

    /// Build the index for efficient search.
    ///
    /// This compacts all pending embeddings into a centroid-organized
    /// IVF structure optimized for cache-efficient search.
    ///
    /// # Errors
    ///
    /// Returns an error if the codec has not been trained.
    pub fn build(&mut self) -> Result<()> {
        let codec = self.codec.as_ref().ok_or_else(|| {
            crate::Error::InvalidInput("Codec not trained - call train() first".to_string())
        })?;

        // Assign each token to its nearest centroid
        let mut centroid_assignments: Vec<Vec<(ChunkId, u16, Vec<u8>)>> =
            vec![Vec::new(); self.config.num_centroids];

        for (chunk_id, embedding) in &self.pending {
            for (token_idx, token) in embedding.tokens().enumerate() {
                let (centroid_id, residual) = codec.compress(token);
                centroid_assignments[centroid_id].push((*chunk_id, token_idx as u16, residual));
            }
        }

        // Build compacted arrays
        let bytes_per_residual = self.config.packed_residual_size();

        self.sizes = centroid_assignments.iter().map(|v| v.len()).collect();
        self.offsets = self
            .sizes
            .iter()
            .scan(0, |acc, &size| {
                let offset = *acc;
                *acc += size;
                Some(offset)
            })
            .collect();

        let total_tokens: usize = self.sizes.iter().sum();
        self.chunk_ids = Vec::with_capacity(total_tokens);
        self.token_indices = Vec::with_capacity(total_tokens);
        self.residuals = Vec::with_capacity(total_tokens * bytes_per_residual);

        for assignments in centroid_assignments {
            for (chunk_id, token_idx, residual) in assignments {
                self.chunk_ids.push(chunk_id);
                self.token_indices.push(token_idx);
                self.residuals.extend(residual);
            }
        }

        self.pending.clear();
        self.is_built = true;

        Ok(())
    }

    /// Clear the built index to allow new insertions.
    ///
    /// Chunks are preserved, but the IVF structure is cleared.
    /// Call `build()` again after inserting new chunks.
    pub fn clear_index(&mut self) {
        self.sizes.clear();
        self.offsets.clear();
        self.chunk_ids.clear();
        self.token_indices.clear();
        self.residuals.clear();
        self.is_built = false;
    }

    /// Search for relevant chunks using MaxSim scoring.
    ///
    /// # Arguments
    ///
    /// * `query` - Query multi-vector embedding
    /// * `search_config` - Search parameters
    ///
    /// # Returns
    ///
    /// Vector of (ChunkId, score) pairs sorted by score descending.
    ///
    /// # Errors
    ///
    /// Returns an error if the index has not been built.
    pub fn search(
        &self,
        query: &MultiVectorEmbedding,
        search_config: &WarpSearchConfig,
    ) -> Result<Vec<(ChunkId, f32)>> {
        let codec = self
            .codec
            .as_ref()
            .ok_or_else(|| crate::Error::InvalidInput("Codec not trained".to_string()))?;

        if !self.is_built {
            return Err(crate::Error::InvalidInput(
                "Index not built - call build() first".to_string(),
            ));
        }

        // Phase 1: Select centroids per query token
        let selected_centroids = CentroidSelector::select(
            query,
            codec.centroids(),
            self.config.token_dim,
            search_config,
        );

        // Apply bound: limit total centroids examined
        let mut total_centroids = 0;
        let max_tokens = search_config.t_prime.unwrap_or(usize::MAX);
        let bounded_centroids: Vec<Vec<(usize, f32)>> = selected_centroids
            .into_iter()
            .take(max_tokens)
            .map(|centroids| {
                let take =
                    (search_config.bound.saturating_sub(total_centroids)).min(centroids.len());
                total_centroids += take;
                centroids.into_iter().take(take).collect()
            })
            .collect();

        // Phase 2: Score candidates from selected centroids
        let bytes_per_residual = self.config.packed_residual_size();

        let token_scores: Vec<Vec<(ChunkId, u16, f32)>> = bounded_centroids
            .into_iter()
            .enumerate()
            .map(|(query_token_idx, centroids)| {
                let query_token = query.token(query_token_idx);

                centroids
                    .into_iter()
                    .flat_map(|(centroid_id, centroid_score)| {
                        CandidateScorer::score(
                            query_token,
                            centroid_id,
                            centroid_score,
                            codec,
                            &self.sizes,
                            &self.offsets,
                            &self.chunk_ids,
                            &self.token_indices,
                            &self.residuals,
                            bytes_per_residual,
                        )
                    })
                    .collect()
            })
            .collect();

        // Phase 3: Merge via MaxSim
        Ok(ScoreMerger::merge(token_scores, search_config.k))
    }

    /// Get centroid size (number of tokens assigned).
    #[must_use]
    pub fn centroid_size(&self, centroid_id: usize) -> usize {
        self.sizes.get(centroid_id).copied().unwrap_or(0)
    }

    /// Get centroid offset in the compacted arrays.
    #[must_use]
    pub fn centroid_offset(&self, centroid_id: usize) -> usize {
        self.offsets.get(centroid_id).copied().unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DocumentId;

    fn create_test_chunk(content: &str) -> Chunk {
        Chunk::new(DocumentId::new(), content.to_string(), 0, content.len())
    }

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

    // ============ Basic Index Tests ============

    #[test]
    fn test_index_new() {
        let config = WarpIndexConfig::new(2, 16, 32);
        let index = WarpIndex::new(config);

        assert!(!index.is_trained());
        assert!(!index.is_built());
        assert!(index.is_empty());
    }

    #[test]
    fn test_index_config() {
        let config = WarpIndexConfig::new(4, 32, 64);
        let index = WarpIndex::new(config);

        assert_eq!(index.config().nbits, 4);
        assert_eq!(index.config().num_centroids, 32);
        assert_eq!(index.config().token_dim, 64);
    }

    // ============ Training Tests ============

    #[test]
    fn test_index_train() {
        let config = WarpIndexConfig::new(2, 8, 16).with_kmeans_iterations(3);
        let mut index = WarpIndex::new(config);

        // Generate training samples
        let samples: Vec<_> = (0..100).map(|i| generate_embedding(10, 16, i)).collect();

        index.train(&samples).unwrap();

        assert!(index.is_trained());
        assert!(index.codec().is_some());
    }

    #[test]
    fn test_index_train_insufficient_samples() {
        let config = WarpIndexConfig::new(2, 100, 16); // 100 centroids needs 1000+ samples
        let mut index = WarpIndex::new(config);

        let samples: Vec<_> = (0..5).map(|i| generate_embedding(10, 16, i)).collect();

        let result = index.train(&samples);
        assert!(result.is_err());
    }

    // ============ Insert Tests ============

    #[test]
    fn test_index_insert() {
        let config = WarpIndexConfig::new(2, 8, 16).with_kmeans_iterations(3);
        let mut index = WarpIndex::new(config);

        // Train first
        let samples: Vec<_> = (0..100).map(|i| generate_embedding(10, 16, i)).collect();
        index.train(&samples).unwrap();

        // Insert chunk
        let chunk = create_test_chunk("test content");
        let embedding = generate_embedding(5, 16, 999);
        index.insert(chunk, embedding).unwrap();

        assert_eq!(index.num_chunks(), 1);
    }

    #[test]
    fn test_index_insert_without_training() {
        let config = WarpIndexConfig::new(2, 8, 16);
        let mut index = WarpIndex::new(config);

        let chunk = create_test_chunk("test");
        let embedding = generate_embedding(5, 16, 0);

        let result = index.insert(chunk, embedding);
        assert!(result.is_err());
    }

    // ============ Build Tests ============

    #[test]
    fn test_index_build() {
        let config = WarpIndexConfig::new(2, 8, 16).with_kmeans_iterations(3);
        let mut index = WarpIndex::new(config);

        // Train
        let samples: Vec<_> = (0..100).map(|i| generate_embedding(10, 16, i)).collect();
        index.train(&samples).unwrap();

        // Insert
        for i in 0..10 {
            let chunk = create_test_chunk(&format!("document {}", i));
            let embedding = generate_embedding(5, 16, 1000 + i);
            index.insert(chunk, embedding).unwrap();
        }

        // Build
        index.build().unwrap();

        assert!(index.is_built());
        assert_eq!(index.num_chunks(), 10);
        assert_eq!(index.num_tokens(), 50); // 10 chunks × 5 tokens
    }

    #[test]
    fn test_index_cannot_insert_after_build() {
        let config = WarpIndexConfig::new(2, 8, 16).with_kmeans_iterations(3);
        let mut index = WarpIndex::new(config);

        let samples: Vec<_> = (0..100).map(|i| generate_embedding(10, 16, i)).collect();
        index.train(&samples).unwrap();

        let chunk = create_test_chunk("test");
        let embedding = generate_embedding(5, 16, 0);
        index.insert(chunk, embedding).unwrap();

        index.build().unwrap();

        // Try to insert after build
        let chunk2 = create_test_chunk("test2");
        let embedding2 = generate_embedding(5, 16, 1);
        let result = index.insert(chunk2, embedding2);

        assert!(result.is_err());
    }

    // ============ Search Tests ============

    #[test]
    fn test_index_search() {
        let config = WarpIndexConfig::new(2, 8, 16).with_kmeans_iterations(3);
        let mut index = WarpIndex::new(config);

        // Train
        let samples: Vec<_> = (0..100).map(|i| generate_embedding(10, 16, i)).collect();
        index.train(&samples).unwrap();

        // Insert
        for i in 0..20 {
            let chunk = create_test_chunk(&format!("document {}", i));
            let embedding = generate_embedding(5, 16, 1000 + i);
            index.insert(chunk, embedding).unwrap();
        }

        // Build
        index.build().unwrap();

        // Search
        let query = generate_embedding(3, 16, 9999);
        let search_config = WarpSearchConfig::with_k(5);
        let results = index.search(&query, &search_config).unwrap();

        assert!(results.len() <= 5);
        assert!(!results.is_empty());

        // Results should be sorted by score descending
        for i in 1..results.len() {
            assert!(results[i - 1].1 >= results[i].1);
        }
    }

    #[test]
    fn test_index_search_without_build() {
        let config = WarpIndexConfig::new(2, 8, 16).with_kmeans_iterations(3);
        let mut index = WarpIndex::new(config);

        let samples: Vec<_> = (0..100).map(|i| generate_embedding(10, 16, i)).collect();
        index.train(&samples).unwrap();

        let query = generate_embedding(3, 16, 0);
        let search_config = WarpSearchConfig::with_k(5);
        let result = index.search(&query, &search_config);

        assert!(result.is_err());
    }

    // ============ Memory & Stats Tests ============

    #[test]
    fn test_index_memory_usage() {
        let config = WarpIndexConfig::new(2, 8, 16).with_kmeans_iterations(3);
        let mut index = WarpIndex::new(config);

        let samples: Vec<_> = (0..100).map(|i| generate_embedding(10, 16, i)).collect();
        index.train(&samples).unwrap();

        for i in 0..10 {
            let chunk = create_test_chunk(&format!("doc {}", i));
            let embedding = generate_embedding(5, 16, 1000 + i);
            index.insert(chunk, embedding).unwrap();
        }

        index.build().unwrap();

        let memory = index.memory_usage();
        assert!(memory > 0);
    }

    #[test]
    fn test_index_centroid_stats() {
        let config = WarpIndexConfig::new(2, 8, 16).with_kmeans_iterations(3);
        let mut index = WarpIndex::new(config);

        let samples: Vec<_> = (0..100).map(|i| generate_embedding(10, 16, i)).collect();
        index.train(&samples).unwrap();

        for i in 0..10 {
            let chunk = create_test_chunk(&format!("doc {}", i));
            let embedding = generate_embedding(5, 16, 1000 + i);
            index.insert(chunk, embedding).unwrap();
        }

        index.build().unwrap();

        // Total tokens across centroids should equal num_tokens
        let total: usize = (0..8).map(|c| index.centroid_size(c)).sum();
        assert_eq!(total, index.num_tokens());
    }

    // ============ Clear & Rebuild Tests ============

    #[test]
    fn test_index_clear_and_rebuild() {
        let config = WarpIndexConfig::new(2, 8, 16).with_kmeans_iterations(3);
        let mut index = WarpIndex::new(config);

        let samples: Vec<_> = (0..100).map(|i| generate_embedding(10, 16, i)).collect();
        index.train(&samples).unwrap();

        let chunk = create_test_chunk("test");
        let embedding = generate_embedding(5, 16, 0);
        index.insert(chunk, embedding).unwrap();
        index.build().unwrap();

        assert!(index.is_built());

        index.clear_index();

        assert!(!index.is_built());
        assert_eq!(index.num_tokens(), 0);
        // Chunks are preserved
        assert_eq!(index.num_chunks(), 1);
    }

    // ============ Get Chunk Tests ============

    #[test]
    fn test_index_get_chunk() {
        let config = WarpIndexConfig::new(2, 8, 16).with_kmeans_iterations(3);
        let mut index = WarpIndex::new(config);

        let samples: Vec<_> = (0..100).map(|i| generate_embedding(10, 16, i)).collect();
        index.train(&samples).unwrap();

        let chunk = create_test_chunk("test content");
        let chunk_id = chunk.id;
        let embedding = generate_embedding(5, 16, 0);
        index.insert(chunk, embedding).unwrap();

        let retrieved = index.get_chunk(&chunk_id);
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().content, "test content");
    }

    // ============ Property-Based Tests ============

    use proptest::prelude::*;

    proptest! {
        #[test]
        fn prop_search_returns_at_most_k(k in 1usize..20) {
            let config = WarpIndexConfig::new(2, 8, 16).with_kmeans_iterations(3);
            let mut index = WarpIndex::new(config);

            let samples: Vec<_> = (0..100).map(|i| generate_embedding(10, 16, i)).collect();
            index.train(&samples).unwrap();

            for i in 0..30 {
                let chunk = create_test_chunk(&format!("doc {}", i));
                let embedding = generate_embedding(5, 16, 1000 + i as u64);
                index.insert(chunk, embedding).unwrap();
            }

            index.build().unwrap();

            let query = generate_embedding(3, 16, 9999);
            let search_config = WarpSearchConfig::with_k(k);
            let results = index.search(&query, &search_config).unwrap();

            prop_assert!(results.len() <= k);
        }

        #[test]
        fn prop_search_results_sorted_descending(seed in 0u64..1000) {
            let config = WarpIndexConfig::new(2, 8, 16).with_kmeans_iterations(3);
            let mut index = WarpIndex::new(config);

            let samples: Vec<_> = (0..100).map(|i| generate_embedding(10, 16, i)).collect();
            index.train(&samples).unwrap();

            for i in 0..20 {
                let chunk = create_test_chunk(&format!("doc {}", i));
                let embedding = generate_embedding(5, 16, seed + i as u64);
                index.insert(chunk, embedding).unwrap();
            }

            index.build().unwrap();

            let query = generate_embedding(3, 16, seed + 1000);
            let search_config = WarpSearchConfig::with_k(10);
            let results = index.search(&query, &search_config).unwrap();

            for i in 1..results.len() {
                prop_assert!(results[i - 1].1 >= results[i].1);
            }
        }
    }
}
