//! Multi-vector embedder trait and implementations
//!
//! This module defines the trait for models that produce token-level embeddings
//! (like ColBERT) and provides a mock implementation for testing.

use crate::multivector::MultiVectorEmbedding;
use crate::Result;

/// Trait for models that produce token-level embeddings.
///
/// Unlike single-vector embedders (which produce one embedding per text),
/// multi-vector embedders produce one embedding per token, enabling
/// fine-grained late interaction scoring.
///
/// # Example
///
/// ```ignore
/// use trueno_rag::multivector::{MultiVectorEmbedder, MockMultiVectorEmbedder};
///
/// let embedder = MockMultiVectorEmbedder::new(128, 512);
/// let embedding = embedder.embed_tokens("hello world").unwrap();
///
/// assert_eq!(embedding.num_tokens(), 2);
/// assert_eq!(embedding.dim(), 128);
/// ```
pub trait MultiVectorEmbedder: Send + Sync {
    /// Embed text into token-level vectors.
    ///
    /// # Arguments
    ///
    /// * `text` - Input text to embed
    ///
    /// # Returns
    ///
    /// A `MultiVectorEmbedding` containing one vector per token.
    fn embed_tokens(&self, text: &str) -> Result<MultiVectorEmbedding>;

    /// Batch embed multiple texts.
    ///
    /// The default implementation calls `embed_tokens` sequentially.
    /// Implementations may override for more efficient batching.
    fn embed_tokens_batch(&self, texts: &[&str]) -> Result<Vec<MultiVectorEmbedding>> {
        texts.iter().map(|t| self.embed_tokens(t)).collect()
    }

    /// Get the token embedding dimension.
    fn token_dimension(&self) -> usize;

    /// Get the maximum tokens per document.
    fn max_tokens(&self) -> usize;

    /// Get the model identifier.
    fn model_id(&self) -> &str;
}

/// Mock multi-vector embedder for testing.
///
/// Generates deterministic pseudo-random embeddings based on token content.
/// Useful for testing the retrieval pipeline without requiring a real model.
///
/// # Example
///
/// ```
/// use trueno_rag::multivector::MockMultiVectorEmbedder;
/// use trueno_rag::multivector::MultiVectorEmbedder;
///
/// let embedder = MockMultiVectorEmbedder::new(128, 512);
///
/// let emb1 = embedder.embed_tokens("hello world").unwrap();
/// let emb2 = embedder.embed_tokens("hello world").unwrap();
///
/// // Same input produces same output
/// assert_eq!(emb1.as_slice(), emb2.as_slice());
/// ```
#[derive(Debug, Clone)]
pub struct MockMultiVectorEmbedder {
    dim: usize,
    max_tokens: usize,
    seed: u64,
}

impl MockMultiVectorEmbedder {
    /// Create a new mock embedder.
    ///
    /// # Arguments
    ///
    /// * `dim` - Token embedding dimension (e.g., 128 for ColBERT)
    /// * `max_tokens` - Maximum tokens per document
    #[must_use]
    pub fn new(dim: usize, max_tokens: usize) -> Self {
        Self { dim, max_tokens, seed: 42 }
    }

    /// Create with a custom seed for different random sequences.
    #[must_use]
    pub fn with_seed(dim: usize, max_tokens: usize, seed: u64) -> Self {
        Self { dim, max_tokens, seed }
    }

    /// Generate a deterministic unit vector from a seed.
    fn generate_unit_vector(&self, seed: u64) -> Vec<f32> {
        let mut vec = Vec::with_capacity(self.dim);
        let mut rng = seed;

        for _ in 0..self.dim {
            rng = rng.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            let val = ((rng >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0;
            vec.push(val);
        }

        // Normalize to unit length
        let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for v in &mut vec {
                *v /= norm;
            }
        }

        vec
    }

    /// Hash a token to a seed value.
    fn hash_token(&self, token: &str, index: usize) -> u64 {
        let mut hash = self.seed;
        for byte in token.bytes() {
            hash = hash.wrapping_mul(31).wrapping_add(u64::from(byte));
        }
        hash = hash.wrapping_mul(31).wrapping_add(index as u64);
        hash
    }
}

impl MultiVectorEmbedder for MockMultiVectorEmbedder {
    fn embed_tokens(&self, text: &str) -> Result<MultiVectorEmbedding> {
        let tokens: Vec<&str> = text.split_whitespace().collect();
        let num_tokens = tokens.len().min(self.max_tokens);

        if num_tokens == 0 {
            return Ok(MultiVectorEmbedding::new(Vec::new(), 0, self.dim));
        }

        let mut embeddings = Vec::with_capacity(num_tokens * self.dim);

        for (i, token) in tokens.iter().take(num_tokens).enumerate() {
            let token_seed = self.hash_token(token, i);
            embeddings.extend(self.generate_unit_vector(token_seed));
        }

        Ok(MultiVectorEmbedding::new(embeddings, num_tokens, self.dim))
    }

    fn embed_tokens_batch(&self, texts: &[&str]) -> Result<Vec<MultiVectorEmbedding>> {
        texts.iter().map(|t| self.embed_tokens(t)).collect()
    }

    fn token_dimension(&self) -> usize {
        self.dim
    }

    fn max_tokens(&self) -> usize {
        self.max_tokens
    }

    fn model_id(&self) -> &str {
        "mock-multivector"
    }
}

/// Trait implementation for boxed embedders.
impl<E: MultiVectorEmbedder + ?Sized> MultiVectorEmbedder for Box<E> {
    fn embed_tokens(&self, text: &str) -> Result<MultiVectorEmbedding> {
        (**self).embed_tokens(text)
    }

    fn embed_tokens_batch(&self, texts: &[&str]) -> Result<Vec<MultiVectorEmbedding>> {
        (**self).embed_tokens_batch(texts)
    }

    fn token_dimension(&self) -> usize {
        (**self).token_dimension()
    }

    fn max_tokens(&self) -> usize {
        (**self).max_tokens()
    }

    fn model_id(&self) -> &str {
        (**self).model_id()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ============ MockMultiVectorEmbedder Tests ============

    #[test]
    fn test_mock_embedder_new() {
        let embedder = MockMultiVectorEmbedder::new(128, 512);

        assert_eq!(embedder.token_dimension(), 128);
        assert_eq!(embedder.max_tokens(), 512);
        assert_eq!(embedder.model_id(), "mock-multivector");
    }

    #[test]
    fn test_mock_embedder_with_seed() {
        let embedder1 = MockMultiVectorEmbedder::with_seed(128, 512, 123);
        let embedder2 = MockMultiVectorEmbedder::with_seed(128, 512, 456);

        let emb1 = embedder1.embed_tokens("test").unwrap();
        let emb2 = embedder2.embed_tokens("test").unwrap();

        // Different seeds should produce different embeddings
        assert_ne!(emb1.as_slice(), emb2.as_slice());
    }

    #[test]
    fn test_mock_embedder_deterministic() {
        let embedder = MockMultiVectorEmbedder::new(64, 256);

        let emb1 = embedder.embed_tokens("hello world").unwrap();
        let emb2 = embedder.embed_tokens("hello world").unwrap();

        assert_eq!(emb1.num_tokens(), emb2.num_tokens());
        assert_eq!(emb1.as_slice(), emb2.as_slice());
    }

    #[test]
    fn test_mock_embedder_token_count() {
        let embedder = MockMultiVectorEmbedder::new(64, 256);

        let emb = embedder.embed_tokens("one two three four five").unwrap();

        assert_eq!(emb.num_tokens(), 5);
        assert_eq!(emb.dim(), 64);
    }

    #[test]
    fn test_mock_embedder_max_tokens() {
        let embedder = MockMultiVectorEmbedder::new(64, 3);

        let emb = embedder.embed_tokens("one two three four five six").unwrap();

        assert_eq!(emb.num_tokens(), 3); // Capped at max_tokens
    }

    #[test]
    fn test_mock_embedder_empty_text() {
        let embedder = MockMultiVectorEmbedder::new(64, 256);

        let emb = embedder.embed_tokens("").unwrap();

        assert_eq!(emb.num_tokens(), 0);
        assert!(emb.is_empty());
    }

    #[test]
    fn test_mock_embedder_whitespace_only() {
        let embedder = MockMultiVectorEmbedder::new(64, 256);

        let emb = embedder.embed_tokens("   \t\n   ").unwrap();

        assert_eq!(emb.num_tokens(), 0);
    }

    #[test]
    fn test_mock_embedder_unit_vectors() {
        let embedder = MockMultiVectorEmbedder::new(64, 256);

        let emb = embedder.embed_tokens("test token").unwrap();

        // Each token should be approximately unit length
        for token_emb in emb.tokens() {
            let norm: f32 = token_emb.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!((norm - 1.0).abs() < 0.001, "Token not unit length: norm = {}", norm);
        }
    }

    #[test]
    fn test_mock_embedder_different_tokens() {
        let embedder = MockMultiVectorEmbedder::new(64, 256);

        let emb = embedder.embed_tokens("hello world").unwrap();

        // Different tokens should have different embeddings
        let token0 = emb.token(0);
        let token1 = emb.token(1);

        assert_ne!(token0, token1);
    }

    // ============ Batch Embedding Tests ============

    #[test]
    fn test_mock_embedder_batch() {
        let embedder = MockMultiVectorEmbedder::new(64, 256);

        let texts = ["hello", "world", "test"];
        let embeddings = embedder.embed_tokens_batch(&texts).unwrap();

        assert_eq!(embeddings.len(), 3);
        assert_eq!(embeddings[0].num_tokens(), 1);
        assert_eq!(embeddings[1].num_tokens(), 1);
        assert_eq!(embeddings[2].num_tokens(), 1);
    }

    #[test]
    fn test_mock_embedder_batch_consistency() {
        let embedder = MockMultiVectorEmbedder::new(64, 256);

        let texts = ["hello", "world"];
        let batch_result = embedder.embed_tokens_batch(&texts).unwrap();

        let single1 = embedder.embed_tokens("hello").unwrap();
        let single2 = embedder.embed_tokens("world").unwrap();

        assert_eq!(batch_result[0].as_slice(), single1.as_slice());
        assert_eq!(batch_result[1].as_slice(), single2.as_slice());
    }

    // ============ Box<dyn MultiVectorEmbedder> Tests ============

    #[test]
    fn test_boxed_embedder() {
        let embedder: Box<dyn MultiVectorEmbedder> =
            Box::new(MockMultiVectorEmbedder::new(64, 256));

        let emb = embedder.embed_tokens("test").unwrap();

        assert_eq!(emb.num_tokens(), 1);
        assert_eq!(embedder.token_dimension(), 64);
    }

    // ============ Property-Based Tests ============

    use proptest::prelude::*;

    proptest! {
        #[test]
        fn prop_embed_produces_correct_dimensions(
            dim in 16usize..256,
            text in "[a-z ]{1,100}"
        ) {
            let embedder = MockMultiVectorEmbedder::new(dim, 512);
            let emb = embedder.embed_tokens(&text).unwrap();

            prop_assert_eq!(emb.dim(), dim);
            if emb.num_tokens() > 0 {
                prop_assert_eq!(emb.token(0).len(), dim);
            }
        }

        #[test]
        fn prop_embed_respects_max_tokens(
            max_tokens in 1usize..10,
            words in 1usize..20
        ) {
            let text: String = (0..words).map(|i| format!("word{}", i)).collect::<Vec<_>>().join(" ");
            let embedder = MockMultiVectorEmbedder::new(64, max_tokens);

            let emb = embedder.embed_tokens(&text).unwrap();

            prop_assert!(emb.num_tokens() <= max_tokens);
        }

        #[test]
        fn prop_embed_is_deterministic(
            seed in 0u64..10000,
            text in "[a-z ]{1,50}"
        ) {
            let embedder = MockMultiVectorEmbedder::with_seed(64, 256, seed);

            let emb1 = embedder.embed_tokens(&text).unwrap();
            let emb2 = embedder.embed_tokens(&text).unwrap();

            prop_assert_eq!(emb1.as_slice(), emb2.as_slice());
        }

        #[test]
        fn prop_tokens_are_approximately_unit_length(
            dim in 32usize..128,
            text in "[a-z]{3,10}"
        ) {
            let embedder = MockMultiVectorEmbedder::new(dim, 256);
            let emb = embedder.embed_tokens(&text).unwrap();

            for token_emb in emb.tokens() {
                let norm: f32 = token_emb.iter().map(|x| x * x).sum::<f32>().sqrt();
                prop_assert!((norm - 1.0).abs() < 0.01);
            }
        }
    }
}
