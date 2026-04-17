//! Core data structures for multi-vector retrieval
//!
//! This module defines the fundamental types used in WARP-based multi-vector
//! retrieval, including embeddings, index configuration, and search parameters.

use serde::{Deserialize, Serialize};

/// A document or query represented as multiple token embeddings.
///
/// In ColBERT-style retrieval, each document and query is represented not by
/// a single embedding vector, but by multiple vectors—one per token. This
/// enables fine-grained "late interaction" scoring via MaxSim.
///
/// # Memory Layout
///
/// Embeddings are stored in a flattened contiguous array for cache efficiency:
/// `[token_0_dim_0, token_0_dim_1, ..., token_1_dim_0, token_1_dim_1, ...]`
///
/// # Example
///
/// ```
/// use trueno_rag::multivector::MultiVectorEmbedding;
///
/// // Create a 3-token embedding with 128 dimensions per token
/// let embeddings = vec![0.0f32; 3 * 128];
/// let mv = MultiVectorEmbedding::new(embeddings, 3, 128);
///
/// assert_eq!(mv.num_tokens(), 3);
/// assert_eq!(mv.dim(), 128);
/// assert_eq!(mv.token(0).len(), 128);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiVectorEmbedding {
    /// Flattened embeddings: [num_tokens * dim]
    embeddings: Vec<f32>,
    /// Number of token embeddings
    num_tokens: usize,
    /// Dimension per token embedding
    dim: usize,
}

impl MultiVectorEmbedding {
    /// Create a new multi-vector embedding.
    ///
    /// # Panics
    ///
    /// Panics if `embeddings.len() != num_tokens * dim`.
    #[must_use]
    pub fn new(embeddings: Vec<f32>, num_tokens: usize, dim: usize) -> Self {
        assert_eq!(
            embeddings.len(),
            num_tokens * dim,
            "Embedding size mismatch: expected {} ({}×{}), got {}",
            num_tokens * dim,
            num_tokens,
            dim,
            embeddings.len()
        );
        // Contract: embedding-algebra-v1.yaml precondition (pv codegen)
        contract_pre_embedding_lookup!(embeddings);
        Self { embeddings, num_tokens, dim }
    }

    /// Create from a vector of token embeddings.
    #[must_use]
    pub fn from_tokens(tokens: &[Vec<f32>]) -> Self {
        if tokens.is_empty() {
            return Self { embeddings: Vec::new(), num_tokens: 0, dim: 0 };
        }

        let dim = tokens[0].len();
        let num_tokens = tokens.len();
        let mut embeddings = Vec::with_capacity(num_tokens * dim);

        for token in tokens {
            assert_eq!(token.len(), dim, "All tokens must have the same dimension");
            embeddings.extend_from_slice(token);
        }

        Self { embeddings, num_tokens, dim }
    }

    /// Get the number of token embeddings.
    #[must_use]
    pub fn num_tokens(&self) -> usize {
        self.num_tokens
    }

    /// Get the dimension of each token embedding.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Get the i-th token embedding as a slice.
    ///
    /// # Panics
    ///
    /// Panics if `i >= num_tokens`.
    #[must_use]
    pub fn token(&self, i: usize) -> &[f32] {
        assert!(i < self.num_tokens, "Token index out of bounds");
        let start = i * self.dim;
        &self.embeddings[start..start + self.dim]
    }

    /// Iterate over token embeddings.
    ///
    /// Returns an empty iterator if `dim == 0` (poka-yoke: prevents
    /// `chunks_exact(0)` panic from uninitialized embedding config).
    pub fn tokens(&self) -> impl Iterator<Item = &[f32]> {
        if self.dim == 0 {
            // chunks_exact(0) panics — return empty iterator instead
            [].chunks_exact(1)
        } else {
            self.embeddings.chunks_exact(self.dim)
        }
    }

    /// Get the raw flattened embeddings.
    #[must_use]
    pub fn as_slice(&self) -> &[f32] {
        &self.embeddings
    }

    /// Get the raw flattened embeddings mutably.
    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        &mut self.embeddings
    }

    /// Memory size in bytes (uncompressed).
    #[must_use]
    pub fn size_bytes(&self) -> usize {
        self.embeddings.len() * size_of::<f32>()
    }

    /// Check if the embedding is empty (no tokens).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.num_tokens == 0
    }
}

/// Configuration for WARP index construction.
///
/// These parameters control the compression quality and index structure.
/// The default values provide a good balance of memory efficiency and
/// retrieval quality for most use cases.
///
/// # Parameter Guidance
///
/// | Corpus Size  | nbits | num_centroids |
/// |--------------|-------|---------------|
/// | < 100K docs  | 4     | 256           |
/// | 100K - 1M    | 2     | 1024          |
/// | > 1M docs    | 2     | 4096          |
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WarpIndexConfig {
    /// Bits per dimension for residual quantization (2 or 4).
    ///
    /// - 2-bit: 16× compression, ~3-5% MRR loss
    /// - 4-bit: 8× compression, ~1-2% MRR loss
    pub nbits: u8,

    /// Number of centroids for IVF clustering.
    ///
    /// More centroids provide finer-grained partitioning but require
    /// more memory for centroid storage. Typical values: 256-4096.
    pub num_centroids: usize,

    /// Token embedding dimension (e.g., 128 for ColBERT).
    pub token_dim: usize,

    /// Minimum training samples for codec training.
    ///
    /// Should be at least 10 × num_centroids for stable clustering.
    /// If None, defaults to 10 × num_centroids.
    pub min_training_samples: Option<usize>,

    /// K-means iterations for centroid training.
    pub kmeans_iterations: usize,
}

impl Default for WarpIndexConfig {
    fn default() -> Self {
        Self {
            nbits: 2,
            num_centroids: 1024,
            token_dim: 128,
            min_training_samples: None,
            kmeans_iterations: 20,
        }
    }
}

impl WarpIndexConfig {
    /// Create a new configuration with the specified parameters.
    #[must_use]
    pub fn new(nbits: u8, num_centroids: usize, token_dim: usize) -> Self {
        Self { nbits, num_centroids, token_dim, ..Default::default() }
    }

    /// Set the minimum training samples.
    #[must_use]
    pub fn with_min_training_samples(mut self, samples: usize) -> Self {
        self.min_training_samples = Some(samples);
        self
    }

    /// Set the k-means iterations.
    #[must_use]
    pub fn with_kmeans_iterations(mut self, iterations: usize) -> Self {
        self.kmeans_iterations = iterations;
        self
    }

    /// Get the effective minimum training samples.
    #[must_use]
    pub fn effective_min_training_samples(&self) -> usize {
        self.min_training_samples.unwrap_or(10 * self.num_centroids)
    }

    /// Calculate packed residual size in bytes.
    #[must_use]
    pub fn packed_residual_size(&self) -> usize {
        (self.token_dim * self.nbits as usize + 7) / 8
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.nbits != 2 && self.nbits != 4 {
            return Err("nbits must be 2 or 4");
        }
        if self.num_centroids == 0 {
            return Err("num_centroids must be > 0");
        }
        if self.token_dim == 0 {
            return Err("token_dim must be > 0");
        }
        if self.kmeans_iterations == 0 {
            return Err("kmeans_iterations must be > 0");
        }
        Ok(())
    }
}

/// Configuration for WARP search.
///
/// These parameters control the trade-off between search speed and
/// recall quality. The defaults are tuned for high recall (>95%).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WarpSearchConfig {
    /// Number of results to return.
    pub k: usize,

    /// Centroids to probe per query token.
    ///
    /// Higher values increase recall but also latency.
    /// Default: 4 (provides ~95% recall on most datasets).
    pub nprobe: u32,

    /// Maximum total centroids examined across all tokens.
    ///
    /// Acts as an upper bound on computation. Default: 128.
    pub bound: usize,

    /// Early termination: skip tokens after this many.
    ///
    /// For very long queries, processing all tokens may be wasteful.
    /// Setting this limits which tokens contribute to scoring.
    pub t_prime: Option<usize>,

    /// Skip tokens with centroid score below threshold.
    ///
    /// Tokens that don't match any centroid well are unlikely to
    /// contribute meaningful scores. Default: 0.4.
    pub centroid_score_threshold: f32,
}

impl Default for WarpSearchConfig {
    fn default() -> Self {
        Self { k: 10, nprobe: 4, bound: 128, t_prime: None, centroid_score_threshold: 0.4 }
    }
}

impl WarpSearchConfig {
    /// Create a search config with the specified k.
    #[must_use]
    pub fn with_k(k: usize) -> Self {
        Self { k, ..Default::default() }
    }

    /// Set nprobe (centroids per token).
    #[must_use]
    pub fn nprobe(mut self, nprobe: u32) -> Self {
        self.nprobe = nprobe;
        self
    }

    /// Set the centroid bound.
    #[must_use]
    pub fn bound(mut self, bound: usize) -> Self {
        self.bound = bound;
        self
    }

    /// Set early termination threshold.
    #[must_use]
    pub fn t_prime(mut self, t_prime: usize) -> Self {
        self.t_prime = Some(t_prime);
        self
    }

    /// Set centroid score threshold.
    #[must_use]
    pub fn centroid_score_threshold(mut self, threshold: f32) -> Self {
        self.centroid_score_threshold = threshold;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ============ MultiVectorEmbedding Tests ============

    #[test]
    fn test_multivector_new() {
        let embeddings = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mv = MultiVectorEmbedding::new(embeddings, 2, 3);

        assert_eq!(mv.num_tokens(), 2);
        assert_eq!(mv.dim(), 3);
        assert_eq!(mv.token(0), &[1.0, 2.0, 3.0]);
        assert_eq!(mv.token(1), &[4.0, 5.0, 6.0]);
    }

    #[test]
    #[should_panic(expected = "Embedding size mismatch")]
    fn test_multivector_size_mismatch() {
        let embeddings = vec![1.0, 2.0, 3.0];
        let _ = MultiVectorEmbedding::new(embeddings, 2, 3); // Should panic
    }

    #[test]
    fn test_multivector_from_tokens() {
        let tokens = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let mv = MultiVectorEmbedding::from_tokens(&tokens);

        assert_eq!(mv.num_tokens(), 3);
        assert_eq!(mv.dim(), 2);
    }

    #[test]
    fn test_multivector_from_tokens_empty() {
        let tokens: Vec<Vec<f32>> = vec![];
        let mv = MultiVectorEmbedding::from_tokens(&tokens);

        assert_eq!(mv.num_tokens(), 0);
        assert!(mv.is_empty());
    }

    /// Regression test for paiml/trueno-rag#15: division by zero when dim is 0.
    /// `from_tokens(&[])` produces dim=0; `tokens()` must not panic.
    #[test]
    fn test_multivector_dim_zero_tokens_no_panic() {
        let mv = MultiVectorEmbedding::from_tokens(&[]);
        assert_eq!(mv.dim(), 0);
        assert_eq!(mv.tokens().count(), 0); // must not panic
    }

    /// Regression: `new(vec![], 0, 0)` is valid (empty embedding) and
    /// iterating tokens must return an empty iterator, not div-by-zero.
    #[test]
    fn test_multivector_new_zero_dim_zero_tokens() {
        let mv = MultiVectorEmbedding::new(vec![], 0, 0);
        assert_eq!(mv.tokens().count(), 0);
        assert!(mv.is_empty());
    }

    #[test]
    fn test_multivector_tokens_iterator() {
        let embeddings = vec![1.0, 2.0, 3.0, 4.0];
        let mv = MultiVectorEmbedding::new(embeddings, 2, 2);

        let tokens: Vec<&[f32]> = mv.tokens().collect();
        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0], &[1.0, 2.0]);
        assert_eq!(tokens[1], &[3.0, 4.0]);
    }

    #[test]
    fn test_multivector_size_bytes() {
        let embeddings = vec![0.0; 100];
        let mv = MultiVectorEmbedding::new(embeddings, 10, 10);

        assert_eq!(mv.size_bytes(), 100 * 4); // 100 f32s × 4 bytes
    }

    #[test]
    fn test_multivector_as_slice() {
        let embeddings = vec![1.0, 2.0, 3.0];
        let mv = MultiVectorEmbedding::new(embeddings.clone(), 1, 3);

        assert_eq!(mv.as_slice(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_multivector_serialization() {
        let mv = MultiVectorEmbedding::new(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
        let json = serde_json::to_string(&mv).unwrap();
        let deserialized: MultiVectorEmbedding = serde_json::from_str(&json).unwrap();

        assert_eq!(mv.num_tokens(), deserialized.num_tokens());
        assert_eq!(mv.dim(), deserialized.dim());
        assert_eq!(mv.as_slice(), deserialized.as_slice());
    }

    // ============ WarpIndexConfig Tests ============

    #[test]
    fn test_index_config_default() {
        let config = WarpIndexConfig::default();

        assert_eq!(config.nbits, 2);
        assert_eq!(config.num_centroids, 1024);
        assert_eq!(config.token_dim, 128);
        assert_eq!(config.kmeans_iterations, 20);
    }

    #[test]
    fn test_index_config_new() {
        let config = WarpIndexConfig::new(4, 256, 64);

        assert_eq!(config.nbits, 4);
        assert_eq!(config.num_centroids, 256);
        assert_eq!(config.token_dim, 64);
    }

    #[test]
    fn test_index_config_builders() {
        let config = WarpIndexConfig::new(2, 512, 128)
            .with_min_training_samples(5000)
            .with_kmeans_iterations(30);

        assert_eq!(config.min_training_samples, Some(5000));
        assert_eq!(config.kmeans_iterations, 30);
    }

    #[test]
    fn test_index_config_effective_min_samples() {
        let config = WarpIndexConfig::new(2, 100, 128);
        assert_eq!(config.effective_min_training_samples(), 1000); // 10 × 100

        let config = config.with_min_training_samples(500);
        assert_eq!(config.effective_min_training_samples(), 500);
    }

    #[test]
    fn test_index_config_packed_size() {
        // 128 dims × 2 bits = 256 bits = 32 bytes
        let config = WarpIndexConfig::new(2, 1024, 128);
        assert_eq!(config.packed_residual_size(), 32);

        // 128 dims × 4 bits = 512 bits = 64 bytes
        let config = WarpIndexConfig::new(4, 1024, 128);
        assert_eq!(config.packed_residual_size(), 64);
    }

    #[test]
    fn test_index_config_validate() {
        let config = WarpIndexConfig::default();
        assert!(config.validate().is_ok());

        let bad_nbits = WarpIndexConfig { nbits: 3, ..Default::default() };
        assert!(bad_nbits.validate().is_err());

        let bad_centroids = WarpIndexConfig { num_centroids: 0, ..Default::default() };
        assert!(bad_centroids.validate().is_err());
    }

    #[test]
    fn test_index_config_serialization() {
        let config = WarpIndexConfig::new(4, 512, 64);
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: WarpIndexConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(config.nbits, deserialized.nbits);
        assert_eq!(config.num_centroids, deserialized.num_centroids);
        assert_eq!(config.token_dim, deserialized.token_dim);
    }

    // ============ WarpSearchConfig Tests ============

    #[test]
    fn test_search_config_default() {
        let config = WarpSearchConfig::default();

        assert_eq!(config.k, 10);
        assert_eq!(config.nprobe, 4);
        assert_eq!(config.bound, 128);
        assert!(config.t_prime.is_none());
        assert!((config.centroid_score_threshold - 0.4).abs() < 0.001);
    }

    #[test]
    fn test_search_config_with_k() {
        let config = WarpSearchConfig::with_k(20);
        assert_eq!(config.k, 20);
    }

    #[test]
    fn test_search_config_builders() {
        let config = WarpSearchConfig::with_k(5)
            .nprobe(8)
            .bound(256)
            .t_prime(10)
            .centroid_score_threshold(0.5);

        assert_eq!(config.k, 5);
        assert_eq!(config.nprobe, 8);
        assert_eq!(config.bound, 256);
        assert_eq!(config.t_prime, Some(10));
        assert!((config.centroid_score_threshold - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_search_config_serialization() {
        let config = WarpSearchConfig::with_k(15).nprobe(6);
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: WarpSearchConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(config.k, deserialized.k);
        assert_eq!(config.nprobe, deserialized.nprobe);
    }

    // ============ Property-Based Tests ============

    use proptest::prelude::*;

    proptest! {
        #[test]
        fn prop_multivector_tokens_count_matches(
            num_tokens in 1usize..20,
            dim in 1usize..64
        ) {
            let embeddings = vec![0.0f32; num_tokens * dim];
            let mv = MultiVectorEmbedding::new(embeddings, num_tokens, dim);

            prop_assert_eq!(mv.num_tokens(), num_tokens);
            prop_assert_eq!(mv.dim(), dim);
            prop_assert_eq!(mv.tokens().count(), num_tokens);
        }

        #[test]
        fn prop_multivector_token_slices_correct_size(
            num_tokens in 1usize..10,
            dim in 1usize..32
        ) {
            let embeddings = vec![0.0f32; num_tokens * dim];
            let mv = MultiVectorEmbedding::new(embeddings, num_tokens, dim);

            for i in 0..num_tokens {
                prop_assert_eq!(mv.token(i).len(), dim);
            }
        }

        #[test]
        fn prop_index_config_packed_size_formula(
            nbits in prop::sample::select(vec![2u8, 4]),
            dim in 1usize..256
        ) {
            let config = WarpIndexConfig::new(nbits, 1024, dim);
            let expected = (dim * nbits as usize + 7) / 8;
            prop_assert_eq!(config.packed_residual_size(), expected);
        }
    }
}
