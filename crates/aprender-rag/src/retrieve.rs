//! Retrieval module for RAG pipelines

use crate::{
    embed::Embedder,
    fusion::FusionStrategy,
    index::{BM25Index, SparseIndex, VectorStore},
    Chunk, ChunkId, Result,
};
use serde::{Deserialize, Serialize};

/// Result of a retrieval operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievalResult {
    /// The retrieved chunk
    pub chunk: Chunk,
    /// Dense retrieval score (if applicable)
    pub dense_score: Option<f32>,
    /// Sparse retrieval score (if applicable)
    pub sparse_score: Option<f32>,
    /// Multi-vector retrieval score (if applicable, ColBERT-style MaxSim)
    #[cfg(feature = "multivector")]
    pub multivector_score: Option<f32>,
    /// Fused score (if hybrid retrieval)
    pub fused_score: Option<f32>,
    /// Reranking score (if reranking applied)
    pub rerank_score: Option<f32>,
}

impl RetrievalResult {
    /// Create a new retrieval result from a chunk
    #[must_use]
    pub fn new(chunk: Chunk) -> Self {
        Self {
            chunk,
            dense_score: None,
            sparse_score: None,
            #[cfg(feature = "multivector")]
            multivector_score: None,
            fused_score: None,
            rerank_score: None,
        }
    }

    /// Set the dense score
    #[must_use]
    pub fn with_dense_score(mut self, score: f32) -> Self {
        self.dense_score = Some(score);
        self
    }

    /// Set the sparse score
    #[must_use]
    pub fn with_sparse_score(mut self, score: f32) -> Self {
        self.sparse_score = Some(score);
        self
    }

    /// Set the fused score
    #[must_use]
    pub fn with_fused_score(mut self, score: f32) -> Self {
        self.fused_score = Some(score);
        self
    }

    /// Set the rerank score
    #[must_use]
    pub fn with_rerank_score(mut self, score: f32) -> Self {
        self.rerank_score = Some(score);
        self
    }

    /// Set the multi-vector (ColBERT-style) score
    #[cfg(feature = "multivector")]
    #[must_use]
    pub fn with_multivector_score(mut self, score: f32) -> Self {
        self.multivector_score = Some(score);
        self
    }

    /// Get the best available score (rerank > fused > multivector > dense > sparse)
    #[must_use]
    pub fn best_score(&self) -> f32 {
        self.rerank_score
            .or(self.fused_score)
            .or(self.dense_score)
            .or(self.sparse_score)
            .unwrap_or(0.0)
    }

    /// Get the best available score including multi-vector score
    #[cfg(feature = "multivector")]
    #[must_use]
    pub fn best_score_with_multivector(&self) -> f32 {
        self.rerank_score
            .or(self.fused_score)
            .or(self.multivector_score)
            .or(self.dense_score)
            .or(self.sparse_score)
            .unwrap_or(0.0)
    }
}

/// Configuration for hybrid retrieval
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HybridRetrieverConfig {
    /// Number of candidates to retrieve from each source
    pub candidates_per_source: usize,
    /// Fusion strategy
    pub fusion: FusionStrategy,
    /// Whether to use dense retrieval
    pub use_dense: bool,
    /// Whether to use sparse retrieval
    pub use_sparse: bool,
}

impl Default for HybridRetrieverConfig {
    fn default() -> Self {
        Self {
            candidates_per_source: 50,
            fusion: FusionStrategy::default(),
            use_dense: true,
            use_sparse: true,
        }
    }
}

/// Hybrid retriever combining dense and sparse retrieval
pub struct HybridRetriever<E: Embedder> {
    /// Dense vector store
    dense: VectorStore,
    /// Sparse BM25 index
    sparse: BM25Index,
    /// Embedder for query embedding
    embedder: E,
    /// Configuration
    config: HybridRetrieverConfig,
}

impl<E: Embedder> HybridRetriever<E> {
    /// Create a new hybrid retriever
    #[must_use]
    pub fn new(dense: VectorStore, sparse: BM25Index, embedder: E) -> Self {
        Self { dense, sparse, embedder, config: HybridRetrieverConfig::default() }
    }

    /// Set the configuration
    #[must_use]
    pub fn with_config(mut self, config: HybridRetrieverConfig) -> Self {
        self.config = config;
        self
    }

    /// Get the dense store
    #[must_use]
    pub fn dense_store(&self) -> &VectorStore {
        &self.dense
    }

    /// Get the dense store mutably
    pub fn dense_store_mut(&mut self) -> &mut VectorStore {
        &mut self.dense
    }

    /// Get the sparse index
    #[must_use]
    pub fn sparse_index(&self) -> &BM25Index {
        &self.sparse
    }

    /// Get the sparse index mutably
    pub fn sparse_index_mut(&mut self) -> &mut BM25Index {
        &mut self.sparse
    }

    /// Index a chunk (adds to both dense and sparse indices)
    pub fn index(&mut self, chunk: Chunk) -> Result<()> {
        // Add to sparse index
        self.sparse.add(&chunk);

        // Add to dense index (requires embedding)
        self.dense.insert(chunk)?;

        Ok(())
    }

    /// Index multiple chunks
    pub fn index_batch(&mut self, chunks: Vec<Chunk>) -> Result<()> {
        for chunk in chunks {
            self.index(chunk)?;
        }
        Ok(())
    }

    /// Retrieve relevant chunks for a query
    pub fn retrieve(&self, query: &str, k: usize) -> Result<Vec<RetrievalResult>> {
        // Contract: configuration-v1.yaml precondition (pv codegen)
        contract_pre_configuration!(query.as_bytes());
        let candidates = self.config.candidates_per_source;

        // Dense retrieval
        let dense_results = if self.config.use_dense {
            let query_embedding = self.embedder.embed_query(query)?;
            self.dense.search(&query_embedding, candidates)?
        } else {
            Vec::new()
        };

        // Sparse retrieval
        let sparse_results =
            if self.config.use_sparse { self.sparse.search(query, candidates) } else { Vec::new() };

        // Fuse results
        let fused = self.config.fusion.fuse(&dense_results, &sparse_results);

        // Build score maps for lookup
        let dense_scores: std::collections::HashMap<ChunkId, f32> =
            dense_results.into_iter().collect();
        let sparse_scores: std::collections::HashMap<ChunkId, f32> =
            sparse_results.into_iter().collect();

        // Build retrieval results
        let mut results = Vec::with_capacity(k.min(fused.len()));
        for (chunk_id, fused_score) in fused.into_iter().take(k) {
            if let Some(chunk) = self.dense.get(chunk_id) {
                let mut result = RetrievalResult::new(chunk.clone()).with_fused_score(fused_score);

                if let Some(&score) = dense_scores.get(&chunk_id) {
                    result = result.with_dense_score(score);
                }
                if let Some(&score) = sparse_scores.get(&chunk_id) {
                    result = result.with_sparse_score(score);
                }

                results.push(result);
            }
        }

        Ok(results)
    }

    /// Retrieve using only dense (vector) search
    pub fn retrieve_dense(&self, query: &str, k: usize) -> Result<Vec<RetrievalResult>> {
        // Contract: configuration-v1.yaml precondition (pv codegen)
        contract_pre_configuration!(query.as_bytes());
        let query_embedding = self.embedder.embed_query(query)?;
        let results = self.dense.search(&query_embedding, k)?;

        let mut retrieval_results = Vec::with_capacity(results.len());
        for (chunk_id, score) in results {
            if let Some(chunk) = self.dense.get(chunk_id) {
                retrieval_results.push(RetrievalResult::new(chunk.clone()).with_dense_score(score));
            }
        }

        Ok(retrieval_results)
    }

    /// Retrieve using only sparse (BM25) search
    pub fn retrieve_sparse(&self, query: &str, k: usize) -> Result<Vec<RetrievalResult>> {
        // Contract: configuration-v1.yaml precondition (pv codegen)
        contract_pre_configuration!(query.as_bytes());
        let results = self.sparse.search(query, k);

        let mut retrieval_results = Vec::with_capacity(results.len());
        for (chunk_id, score) in results {
            if let Some(chunk) = self.dense.get(chunk_id) {
                retrieval_results
                    .push(RetrievalResult::new(chunk.clone()).with_sparse_score(score));
            }
        }

        Ok(retrieval_results)
    }

    /// Get the number of indexed chunks
    #[must_use]
    pub fn len(&self) -> usize {
        self.dense.len()
    }

    /// Check if the retriever is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.dense.is_empty()
    }
}

/// Dense-only retriever (simpler API for vector-only search)
pub struct DenseRetriever<E: Embedder> {
    store: VectorStore,
    embedder: E,
}

impl<E: Embedder> DenseRetriever<E> {
    /// Create a new dense retriever
    #[must_use]
    pub fn new(store: VectorStore, embedder: E) -> Self {
        Self { store, embedder }
    }

    /// Index a chunk
    pub fn index(&mut self, chunk: Chunk) -> Result<()> {
        self.store.insert(chunk)
    }

    /// Retrieve relevant chunks
    pub fn retrieve(&self, query: &str, k: usize) -> Result<Vec<RetrievalResult>> {
        let query_embedding = self.embedder.embed_query(query)?;
        let results = self.store.search(&query_embedding, k)?;

        let mut retrieval_results = Vec::with_capacity(results.len());
        for (chunk_id, score) in results {
            if let Some(chunk) = self.store.get(chunk_id) {
                retrieval_results.push(RetrievalResult::new(chunk.clone()).with_dense_score(score));
            }
        }

        Ok(retrieval_results)
    }
}

/// Sparse-only retriever (BM25)
pub struct SparseRetriever {
    index: BM25Index,
    chunks: std::collections::HashMap<ChunkId, Chunk>,
}

impl SparseRetriever {
    /// Create a new sparse retriever
    #[must_use]
    pub fn new() -> Self {
        Self { index: BM25Index::new(), chunks: std::collections::HashMap::new() }
    }

    /// Index a chunk
    pub fn index(&mut self, chunk: Chunk) {
        self.index.add(&chunk);
        self.chunks.insert(chunk.id, chunk);
    }

    /// Retrieve relevant chunks
    #[must_use]
    pub fn retrieve(&self, query: &str, k: usize) -> Vec<RetrievalResult> {
        let results = self.index.search(query, k);

        results
            .into_iter()
            .filter_map(|(chunk_id, score)| {
                self.chunks
                    .get(&chunk_id)
                    .map(|chunk| RetrievalResult::new(chunk.clone()).with_sparse_score(score))
            })
            .collect()
    }
}

impl Default for SparseRetriever {
    fn default() -> Self {
        Self::new()
    }
}

// ============ Multi-Vector Retriever (WARP) ============

/// Multi-vector retriever using WARP index for ColBERT-style late interaction.
///
/// This retriever uses token-level embeddings and MaxSim scoring for fine-grained
/// semantic matching. Unlike single-vector dense retrieval, multi-vector approaches
/// represent documents and queries as multiple token embeddings.
///
/// # Example
///
/// ```ignore
/// use trueno_rag::multivector::{
///     WarpIndexConfig, WarpSearchConfig,
///     MockMultiVectorEmbedder, MultiVectorRetriever,
/// };
///
/// let config = WarpIndexConfig::new(2, 256, 128);
/// let embedder = MockMultiVectorEmbedder::new(128, 512);
/// let mut retriever = MultiVectorRetriever::new(config, embedder);
///
/// // Train on sample documents
/// retriever.train(&sample_chunks)?;
///
/// // Index documents
/// for chunk in chunks {
///     retriever.index(chunk)?;
/// }
/// retriever.build()?;
///
/// // Search
/// let results = retriever.retrieve("What is machine learning?", 10)?;
/// ```
#[cfg(feature = "multivector")]
pub struct MultiVectorRetriever<E: crate::multivector::MultiVectorEmbedder> {
    /// WARP index for compressed multi-vector storage and search
    index: crate::multivector::WarpIndex,
    /// Multi-vector embedder for token-level embeddings
    embedder: E,
    /// Search configuration
    search_config: crate::multivector::WarpSearchConfig,
}

#[cfg(feature = "multivector")]
impl<E: crate::multivector::MultiVectorEmbedder> MultiVectorRetriever<E> {
    /// Create a new multi-vector retriever with the given configuration and embedder.
    ///
    /// # Arguments
    ///
    /// * `config` - WARP index configuration (nbits, num_centroids, token_dim)
    /// * `embedder` - Multi-vector embedder for generating token embeddings
    #[must_use]
    pub fn new(config: crate::multivector::WarpIndexConfig, embedder: E) -> Self {
        Self {
            index: crate::multivector::WarpIndex::new(config),
            embedder,
            search_config: crate::multivector::WarpSearchConfig::default(),
        }
    }

    /// Set the search configuration.
    #[must_use]
    pub fn with_search_config(mut self, config: crate::multivector::WarpSearchConfig) -> Self {
        self.search_config = config;
        self
    }

    /// Train the WARP index on sample chunks.
    ///
    /// This builds the residual quantization codec by learning centroids from
    /// the provided sample embeddings. Should be called before indexing.
    ///
    /// # Arguments
    ///
    /// * `sample_chunks` - Representative chunks for training the codec
    pub fn train(&mut self, sample_chunks: &[Chunk]) -> Result<()> {
        let texts: Vec<&str> = sample_chunks.iter().map(|c| c.content.as_str()).collect();
        let embeddings = self.embedder.embed_tokens_batch(&texts)?;
        self.index.train(&embeddings)?;
        Ok(())
    }

    /// Index a single chunk.
    ///
    /// The chunk is embedded and compressed using the trained codec.
    /// Call `train()` before indexing.
    pub fn index(&mut self, chunk: Chunk) -> Result<()> {
        let embedding = self.embedder.embed_tokens(&chunk.content)?;
        self.index.insert(chunk, embedding)?;
        Ok(())
    }

    /// Index multiple chunks.
    pub fn index_batch(&mut self, chunks: Vec<Chunk>) -> Result<()> {
        for chunk in chunks {
            self.index(chunk)?;
        }
        Ok(())
    }

    /// Build the index for efficient search.
    ///
    /// This compacts the index by organizing embeddings by centroid (IVF structure).
    /// Call after all chunks have been indexed.
    pub fn build(&mut self) -> Result<()> {
        self.index.build()
    }

    /// Retrieve relevant chunks for a query using multi-vector MaxSim scoring.
    ///
    /// # Arguments
    ///
    /// * `query` - Query text
    /// * `k` - Number of results to return
    pub fn retrieve(&self, query: &str, k: usize) -> Result<Vec<RetrievalResult>> {
        let query_embedding = self.embedder.embed_tokens(query)?;
        let search_config = crate::multivector::WarpSearchConfig::with_k(k)
            .nprobe(self.search_config.nprobe)
            .bound(self.search_config.bound)
            .centroid_score_threshold(self.search_config.centroid_score_threshold);
        let results = self.index.search(&query_embedding, &search_config)?;

        let mut retrieval_results = Vec::with_capacity(results.len());
        for (chunk_id, score) in results {
            if let Some(chunk) = self.index.get_chunk(&chunk_id) {
                retrieval_results
                    .push(RetrievalResult::new(chunk.clone()).with_multivector_score(score));
            }
        }

        Ok(retrieval_results)
    }

    /// Get the number of indexed chunks.
    #[must_use]
    pub fn len(&self) -> usize {
        self.index.num_chunks()
    }

    /// Check if the retriever is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get the underlying WARP index.
    #[must_use]
    pub fn warp_index(&self) -> &crate::multivector::WarpIndex {
        &self.index
    }

    /// Get the embedder.
    #[must_use]
    pub fn embedder(&self) -> &E {
        // Contract: embedding-algebra-v1.yaml precondition (pv codegen)
        contract_pre_embedding_lookup!();
        &self.embedder
    }

    /// Get memory usage statistics.
    #[must_use]
    pub fn memory_usage(&self) -> usize {
        self.index.memory_usage()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{embed::MockEmbedder, DocumentId};

    fn create_test_chunk(content: &str, embedding: Vec<f32>) -> Chunk {
        let mut chunk = Chunk::new(DocumentId::new(), content.to_string(), 0, content.len());
        chunk.set_embedding(embedding);
        chunk
    }

    // ============ RetrievalResult Tests ============

    #[test]
    fn test_retrieval_result_new() {
        let chunk = Chunk::new(DocumentId::new(), "test".to_string(), 0, 4);
        let result = RetrievalResult::new(chunk);

        assert!(result.dense_score.is_none());
        assert!(result.sparse_score.is_none());
        assert!(result.fused_score.is_none());
        assert!(result.rerank_score.is_none());
    }

    #[test]
    fn test_retrieval_result_with_scores() {
        let chunk = Chunk::new(DocumentId::new(), "test".to_string(), 0, 4);
        let result = RetrievalResult::new(chunk)
            .with_dense_score(0.9)
            .with_sparse_score(0.8)
            .with_fused_score(0.85)
            .with_rerank_score(0.95);

        assert_eq!(result.dense_score, Some(0.9));
        assert_eq!(result.sparse_score, Some(0.8));
        assert_eq!(result.fused_score, Some(0.85));
        assert_eq!(result.rerank_score, Some(0.95));
    }

    #[test]
    fn test_retrieval_result_best_score_priority() {
        let chunk = Chunk::new(DocumentId::new(), "test".to_string(), 0, 4);

        // Rerank takes priority
        let result =
            RetrievalResult::new(chunk.clone()).with_dense_score(0.5).with_rerank_score(0.9);
        assert!((result.best_score() - 0.9).abs() < 0.001);

        // Fused takes priority over dense/sparse
        let result =
            RetrievalResult::new(chunk.clone()).with_dense_score(0.5).with_fused_score(0.7);
        assert!((result.best_score() - 0.7).abs() < 0.001);

        // Dense used when nothing else available
        let result = RetrievalResult::new(chunk).with_dense_score(0.5);
        assert!((result.best_score() - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_retrieval_result_best_score_default() {
        let chunk = Chunk::new(DocumentId::new(), "test".to_string(), 0, 4);
        let result = RetrievalResult::new(chunk);
        assert!((result.best_score() - 0.0).abs() < 0.001);
    }

    // ============ HybridRetrieverConfig Tests ============

    #[test]
    fn test_hybrid_config_default() {
        let config = HybridRetrieverConfig::default();
        assert_eq!(config.candidates_per_source, 50);
        assert!(config.use_dense);
        assert!(config.use_sparse);
    }

    // ============ HybridRetriever Tests ============

    #[test]
    fn test_hybrid_retriever_new() {
        let embedder = MockEmbedder::new(64);
        let dense = VectorStore::with_dimension(64);
        let sparse = BM25Index::new();

        let retriever = HybridRetriever::new(dense, sparse, embedder);
        assert!(retriever.is_empty());
    }

    #[test]
    fn test_hybrid_retriever_index() {
        let embedder = MockEmbedder::new(64);
        let dense = VectorStore::with_dimension(64);
        let sparse = BM25Index::new();

        let mut retriever = HybridRetriever::new(dense, sparse, embedder);

        let chunk = create_test_chunk("machine learning is great", vec![0.0; 64]);
        retriever.index(chunk).unwrap();

        assert_eq!(retriever.len(), 1);
    }

    #[test]
    fn test_hybrid_retriever_index_batch() {
        let embedder = MockEmbedder::new(64);
        let dense = VectorStore::with_dimension(64);
        let sparse = BM25Index::new();

        let mut retriever = HybridRetriever::new(dense, sparse, embedder);

        let chunks = vec![
            create_test_chunk("first document", vec![1.0; 64]),
            create_test_chunk("second document", vec![0.5; 64]),
        ];
        retriever.index_batch(chunks).unwrap();

        assert_eq!(retriever.len(), 2);
    }

    #[test]
    fn test_hybrid_retriever_retrieve() {
        let embedder = MockEmbedder::new(3);
        let dense = VectorStore::with_dimension(3);
        let sparse = BM25Index::new();

        let mut retriever = HybridRetriever::new(dense, sparse, embedder);

        // Index some chunks
        retriever
            .index(create_test_chunk("machine learning algorithms", vec![1.0, 0.0, 0.0]))
            .unwrap();
        retriever
            .index(create_test_chunk("deep learning neural networks", vec![0.9, 0.1, 0.0]))
            .unwrap();
        retriever.index(create_test_chunk("cooking recipes", vec![0.0, 0.0, 1.0])).unwrap();

        let results = retriever.retrieve("machine learning", 2).unwrap();

        assert!(!results.is_empty());
        assert!(results.len() <= 2);
    }

    #[test]
    fn test_hybrid_retriever_retrieve_dense_only() {
        let embedder = MockEmbedder::new(3);
        let dense = VectorStore::with_dimension(3);
        let sparse = BM25Index::new();

        let mut retriever = HybridRetriever::new(dense, sparse, embedder);

        retriever.index(create_test_chunk("test doc", vec![1.0, 0.0, 0.0])).unwrap();

        let results = retriever.retrieve_dense("test", 10).unwrap();
        assert!(!results.is_empty());
        assert!(results[0].dense_score.is_some());
        assert!(results[0].sparse_score.is_none());
    }

    #[test]
    fn test_hybrid_retriever_retrieve_sparse_only() {
        let embedder = MockEmbedder::new(3);
        let dense = VectorStore::with_dimension(3);
        let sparse = BM25Index::new();

        let mut retriever = HybridRetriever::new(dense, sparse, embedder);

        retriever.index(create_test_chunk("machine learning test", vec![1.0, 0.0, 0.0])).unwrap();

        let results = retriever.retrieve_sparse("machine", 10).unwrap();
        assert!(!results.is_empty());
        assert!(results[0].sparse_score.is_some());
        assert!(results[0].dense_score.is_none());
    }

    #[test]
    fn test_hybrid_retriever_config() {
        let embedder = MockEmbedder::new(3);
        let dense = VectorStore::with_dimension(3);
        let sparse = BM25Index::new();

        let config = HybridRetrieverConfig {
            candidates_per_source: 100,
            fusion: FusionStrategy::Linear { dense_weight: 0.7 },
            use_dense: true,
            use_sparse: true,
        };

        let retriever = HybridRetriever::new(dense, sparse, embedder).with_config(config);

        assert_eq!(retriever.config.candidates_per_source, 100);
    }

    // ============ DenseRetriever Tests ============

    #[test]
    fn test_dense_retriever() {
        let embedder = MockEmbedder::new(3);
        let store = VectorStore::with_dimension(3);
        let mut retriever = DenseRetriever::new(store, embedder);

        retriever.index(create_test_chunk("test document", vec![1.0, 0.0, 0.0])).unwrap();

        let results = retriever.retrieve("test", 10).unwrap();
        assert_eq!(results.len(), 1);
        assert!(results[0].dense_score.is_some());
    }

    // ============ SparseRetriever Tests ============

    #[test]
    fn test_sparse_retriever_new() {
        let retriever = SparseRetriever::new();
        let results = retriever.retrieve("test", 10);
        assert!(results.is_empty());
    }

    #[test]
    fn test_sparse_retriever_index() {
        let mut retriever = SparseRetriever::new();
        let chunk = Chunk::new(DocumentId::new(), "machine learning test".to_string(), 0, 20);

        retriever.index(chunk);
        let results = retriever.retrieve("machine", 10);

        assert_eq!(results.len(), 1);
        assert!(results[0].sparse_score.is_some());
    }

    #[test]
    fn test_sparse_retriever_multiple() {
        let mut retriever = SparseRetriever::new();

        retriever.index(Chunk::new(
            DocumentId::new(),
            "rust programming language".to_string(),
            0,
            24,
        ));
        retriever.index(Chunk::new(
            DocumentId::new(),
            "python programming language".to_string(),
            0,
            26,
        ));

        let results = retriever.retrieve("programming", 10);
        assert_eq!(results.len(), 2);
    }

    // ============ Additional Coverage Tests ============

    #[test]
    fn test_hybrid_retriever_store_accessors() {
        let embedder = MockEmbedder::new(64);
        let dense = VectorStore::with_dimension(64);
        let sparse = BM25Index::new();

        let mut retriever = HybridRetriever::new(dense, sparse, embedder);

        // Test immutable accessors
        let _dense_store = retriever.dense_store();
        let _sparse_index = retriever.sparse_index();

        // Test mutable accessors
        let dense_mut = retriever.dense_store_mut();
        assert!(dense_mut.is_empty());

        let sparse_mut = retriever.sparse_index_mut();
        let _ = sparse_mut; // Just verify it compiles and works
    }

    #[test]
    fn test_hybrid_retriever_is_empty() {
        let embedder = MockEmbedder::new(64);
        let dense = VectorStore::with_dimension(64);
        let sparse = BM25Index::new();

        let mut retriever = HybridRetriever::new(dense, sparse, embedder);
        assert!(retriever.is_empty());

        retriever.index(create_test_chunk("test", vec![0.0; 64])).unwrap();
        assert!(!retriever.is_empty());
    }

    #[test]
    fn test_sparse_retriever_default() {
        let retriever = SparseRetriever::default();
        let results = retriever.retrieve("test", 10);
        assert!(results.is_empty());
    }

    #[test]
    fn test_retrieval_result_best_score_sparse_fallback() {
        let chunk = Chunk::new(DocumentId::new(), "test".to_string(), 0, 4);

        // Only sparse score available
        let result = RetrievalResult::new(chunk).with_sparse_score(0.75);
        assert!((result.best_score() - 0.75).abs() < 0.001);
    }

    #[test]
    fn test_hybrid_retriever_with_dense_disabled() {
        let embedder = MockEmbedder::new(3);
        let dense = VectorStore::with_dimension(3);
        let sparse = BM25Index::new();

        let config = HybridRetrieverConfig {
            candidates_per_source: 50,
            fusion: FusionStrategy::default(),
            use_dense: false,
            use_sparse: true,
        };

        let mut retriever = HybridRetriever::new(dense, sparse, embedder).with_config(config);

        retriever.index(create_test_chunk("machine learning test", vec![1.0, 0.0, 0.0])).unwrap();

        // Should still work, using only sparse
        let results = retriever.retrieve("machine", 10).unwrap();
        // Results depend on sparse-only fusion
        assert!(results.len() <= 10);
    }

    #[test]
    fn test_hybrid_retriever_with_sparse_disabled() {
        let embedder = MockEmbedder::new(3);
        let dense = VectorStore::with_dimension(3);
        let sparse = BM25Index::new();

        let config = HybridRetrieverConfig {
            candidates_per_source: 50,
            fusion: FusionStrategy::default(),
            use_dense: true,
            use_sparse: false,
        };

        let mut retriever = HybridRetriever::new(dense, sparse, embedder).with_config(config);

        retriever.index(create_test_chunk("test content", vec![1.0, 0.0, 0.0])).unwrap();

        // Should still work, using only dense
        let results = retriever.retrieve("test", 10).unwrap();
        assert!(results.len() <= 10);
    }

    #[test]
    fn test_hybrid_retriever_config_serialization() {
        let config = HybridRetrieverConfig {
            candidates_per_source: 100,
            fusion: FusionStrategy::RRF { k: 60.0 },
            use_dense: true,
            use_sparse: false,
        };

        let json = serde_json::to_string(&config).unwrap();
        let deserialized: HybridRetrieverConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(config.candidates_per_source, deserialized.candidates_per_source);
        assert_eq!(config.use_dense, deserialized.use_dense);
        assert_eq!(config.use_sparse, deserialized.use_sparse);
    }

    #[test]
    fn test_retrieval_result_serialization() {
        let chunk = Chunk::new(DocumentId::new(), "test content".to_string(), 0, 12);
        let result = RetrievalResult::new(chunk)
            .with_dense_score(0.9)
            .with_sparse_score(0.8)
            .with_fused_score(0.85)
            .with_rerank_score(0.95);

        let json = serde_json::to_string(&result).unwrap();
        let deserialized: RetrievalResult = serde_json::from_str(&json).unwrap();

        assert_eq!(result.dense_score, deserialized.dense_score);
        assert_eq!(result.sparse_score, deserialized.sparse_score);
        assert_eq!(result.fused_score, deserialized.fused_score);
        assert_eq!(result.rerank_score, deserialized.rerank_score);
    }

    // ============ Property-Based Tests ============

    use proptest::prelude::*;

    proptest! {
        #[test]
        fn prop_retrieval_result_scores_preserved(
            dense in 0.0f32..1.0,
            sparse in 0.0f32..1.0,
            fused in 0.0f32..1.0
        ) {
            let chunk = Chunk::new(DocumentId::new(), "test".to_string(), 0, 4);
            let result = RetrievalResult::new(chunk)
                .with_dense_score(dense)
                .with_sparse_score(sparse)
                .with_fused_score(fused);

            prop_assert!((result.dense_score.unwrap() - dense).abs() < 0.0001);
            prop_assert!((result.sparse_score.unwrap() - sparse).abs() < 0.0001);
            prop_assert!((result.fused_score.unwrap() - fused).abs() < 0.0001);
        }

        #[test]
        fn prop_hybrid_retriever_respects_k(k in 1usize..10) {
            let embedder = MockEmbedder::new(3);
            let dense = VectorStore::with_dimension(3);
            let sparse = BM25Index::new();

            let mut retriever = HybridRetriever::new(dense, sparse, embedder);

            // Add more chunks than k
            for i in 0..20 {
                let mut emb = vec![0.0; 3];
                emb[i % 3] = 1.0;
                retriever.index(create_test_chunk(
                    &format!("document number {i} about testing"),
                    emb,
                )).unwrap();
            }

            let results = retriever.retrieve("testing", k).unwrap();
            prop_assert!(results.len() <= k);
        }
    }
}
