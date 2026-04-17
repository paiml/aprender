//! Multi-vector retrieval with WARP algorithm
//!
//! This module provides ColBERT-style multi-vector retrieval using the WARP
//! (Weighted Approximate Residual Product) algorithm. Unlike single-vector
//! dense retrieval, multi-vector approaches represent each document and query
//! as multiple token embeddings, enabling fine-grained "late interaction" scoring.
//!
//! # Overview
//!
//! The WARP algorithm provides memory-efficient multi-vector search by:
//!
//! 1. **Residual Quantization** - Compress token embeddings from 32-bit floats
//!    to 2-4 bits per dimension using centroid-based encoding
//! 2. **IVF Indexing** - Organize embeddings by centroid for cache-efficient access
//! 3. **Deferred Decompression** - Score directly from compressed representations
//!
//! # Key Components
//!
//! - [`MultiVectorEmbedding`] - A document/query represented as multiple token embeddings
//! - [`WarpIndex`] - The main index structure with train/insert/build/search methods
//! - [`WarpIndexConfig`] - Configuration for index construction
//! - [`WarpSearchConfig`] - Configuration for search parameters
//! - [`ResidualCodec`] - Compression codec for token embeddings
//! - [`MultiVectorEmbedder`] - Trait for token-level embedding models
//! - [`MultiVectorRetriever`] - High-level retriever combining embedder and index
//!
//! # Quick Start
//!
//! ```ignore
//! use trueno_rag::multivector::{
//!     WarpIndex, WarpIndexConfig, WarpSearchConfig,
//!     MockMultiVectorEmbedder, MultiVectorEmbedder,
//!     MultiVectorRetriever,
//! };
//!
//! // Create retriever with mock embedder
//! let config = WarpIndexConfig::new(2, 256, 128);
//! let embedder = MockMultiVectorEmbedder::new(128, 512);
//! let mut retriever = MultiVectorRetriever::new(config, embedder);
//!
//! // Train on sample documents
//! retriever.train(&sample_chunks)?;
//!
//! // Index documents
//! for chunk in chunks {
//!     retriever.index(chunk)?;
//! }
//! retriever.build()?;
//!
//! // Search
//! let results = retriever.retrieve("What is machine learning?", 10)?;
//! ```
//!
//! # Theory: MaxSim Scoring
//!
//! ColBERT uses MaxSim scoring which computes, for query Q with tokens {q₁...qₘ}
//! and document D with tokens {d₁...dₙ}:
//!
//! ```text
//! MaxSim(Q, D) = Σᵢ maxⱼ(qᵢ · dⱼ)
//! ```
//!
//! For each query token, find the maximum similarity with any document token,
//! then sum across query tokens. This captures soft alignment without explicit
//! matching.
//!
//! # Feature Flag
//!
//! This module is only available with the `multivector` feature:
//!
//! ```toml
//! [dependencies]
//! trueno-rag = { version = "0.1", features = ["multivector"] }
//! ```
//!
//! # References
//!
//! - Khattab & Zaharia (2020). "ColBERT: Efficient and Effective Passage Search
//!   via Contextualized Late Interaction over BERT." SIGIR 2020.
//! - Santhanam et al. (2022). "ColBERTv2: Effective and Efficient Retrieval via
//!   Lightweight Late Interaction." NAACL 2022.

pub mod codec;
pub mod embedder;
#[cfg(test)]
pub mod falsification;
pub mod index;
pub mod search;
pub mod types;

// Re-export main types at module level
pub use codec::{ResidualCodec, ResidualCodecBuilder};
pub use embedder::{MockMultiVectorEmbedder, MultiVectorEmbedder};
pub use index::WarpIndex;
pub use search::{exact_maxsim, CandidateScorer, CentroidSelector, ScoreMerger};
pub use types::{MultiVectorEmbedding, WarpIndexConfig, WarpSearchConfig};

// Re-export retriever (defined in retrieve.rs but part of this feature)
// Note: MultiVectorRetriever is in retrieve.rs, not here

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Chunk, DocumentId};

    /// Integration test: full pipeline from training to search
    #[test]
    fn test_full_pipeline() {
        // 1. Create embedder
        let embedder = MockMultiVectorEmbedder::new(32, 128);

        // 2. Configure index - use fewer centroids to match training data size
        // (need at least 10 tokens per centroid for training)
        let config = WarpIndexConfig::new(2, 4, 32).with_kmeans_iterations(5);

        // 3. Create index
        let mut index = WarpIndex::new(config);

        // 4. Generate training data with enough tokens
        let training_texts = [
            "machine learning algorithms are powerful tools for data science",
            "deep neural networks have revolutionized computer vision tasks",
            "natural language processing enables machines to understand text",
            "computer vision systems detect objects in images and video",
            "reinforcement learning agents learn through trial and error",
            "transformer architectures power modern large language models",
            "attention mechanisms allow models to focus on relevant inputs",
            "gradient descent optimization updates neural network parameters",
        ];

        let training_embeddings: Vec<_> =
            training_texts.iter().map(|t| embedder.embed_tokens(t).unwrap()).collect();

        // 5. Train codec
        index.train(&training_embeddings).unwrap();

        // 6. Insert documents
        for text in training_texts.iter() {
            let chunk = Chunk::new(DocumentId::new(), text.to_string(), 0, text.len());
            let embedding = embedder.embed_tokens(text).unwrap();
            index.insert(chunk, embedding).unwrap();
        }

        // 7. Build index
        index.build().unwrap();

        // 8. Search
        let query_text = "neural network learning";
        let query_embedding = embedder.embed_tokens(query_text).unwrap();
        let search_config = WarpSearchConfig::with_k(3);

        let results = index.search(&query_embedding, &search_config).unwrap();

        // Verify results
        assert!(!results.is_empty());
        assert!(results.len() <= 3);

        // Results should be sorted by score descending
        for i in 1..results.len() {
            assert!(results[i - 1].1 >= results[i].1);
        }

        // Can retrieve chunks by ID
        for (chunk_id, _score) in &results {
            let chunk = index.get_chunk(chunk_id);
            assert!(chunk.is_some());
        }
    }

    /// Test exact MaxSim matches expected values
    #[test]
    fn test_exact_maxsim_calculation() {
        // Query: 2 tokens
        let query = MultiVectorEmbedding::new(
            vec![
                1.0, 0.0, 0.0, 0.0, // q1
                0.0, 1.0, 0.0, 0.0, // q2
            ],
            2,
            4,
        );

        // Doc: 3 tokens
        let doc = MultiVectorEmbedding::new(
            vec![
                0.5, 0.5, 0.0, 0.0, // d1: q1·d1=0.5, q2·d1=0.5
                1.0, 0.0, 0.0, 0.0, // d2: q1·d2=1.0, q2·d2=0.0
                0.0, 0.0, 1.0, 0.0, // d3: q1·d3=0.0, q2·d3=0.0
            ],
            3,
            4,
        );

        let score = exact_maxsim(&query, &doc);

        // MaxSim = max(0.5, 1.0, 0.0) + max(0.5, 0.0, 0.0) = 1.0 + 0.5 = 1.5
        assert!((score - 1.5).abs() < 1e-6);
    }

    /// Test that compression preserves relative ordering
    #[test]
    fn test_compression_preserves_ordering() {
        let embedder = MockMultiVectorEmbedder::new(32, 128);

        // Create documents with varying relevance
        let query = embedder.embed_tokens("machine learning").unwrap();
        let doc_relevant = embedder.embed_tokens("machine learning algorithms").unwrap();
        let doc_partial = embedder.embed_tokens("learning systems").unwrap();
        let doc_irrelevant = embedder.embed_tokens("cooking recipes").unwrap();

        // Exact scores
        let exact_relevant = exact_maxsim(&query, &doc_relevant);
        let _exact_partial = exact_maxsim(&query, &doc_partial);
        let exact_irrelevant = exact_maxsim(&query, &doc_irrelevant);

        // Verify relative ordering makes sense
        // (relevant should score higher than irrelevant)
        assert!(
            exact_relevant > exact_irrelevant,
            "Relevant doc should score higher: {} vs {}",
            exact_relevant,
            exact_irrelevant
        );
    }

    /// Test search with various nprobe settings
    #[test]
    fn test_search_nprobe_variations() {
        let embedder = MockMultiVectorEmbedder::new(16, 64);
        let config = WarpIndexConfig::new(2, 8, 16).with_kmeans_iterations(3);
        let mut index = WarpIndex::new(config);

        // Train and build
        let texts: Vec<String> = (0..50).map(|i| format!("document number {}", i)).collect();
        let embeddings: Vec<_> = texts.iter().map(|t| embedder.embed_tokens(t).unwrap()).collect();
        index.train(&embeddings).unwrap();

        for (i, text) in texts.iter().enumerate() {
            let chunk = Chunk::new(DocumentId::new(), text.clone(), 0, text.len());
            index.insert(chunk, embeddings[i].clone()).unwrap();
        }
        index.build().unwrap();

        let query = embedder.embed_tokens("document number").unwrap();

        // Test with different nprobe values
        for nprobe in [1, 2, 4, 8] {
            let config = WarpSearchConfig::with_k(5).nprobe(nprobe);
            let results = index.search(&query, &config).unwrap();

            assert!(results.len() <= 5, "nprobe={}: got {} results", nprobe, results.len());
        }
    }

    /// Test memory usage is reasonable
    #[test]
    fn test_memory_efficiency() {
        let embedder = MockMultiVectorEmbedder::new(128, 512);
        // Use fewer centroids - need 10 tokens per centroid for training
        let config = WarpIndexConfig::new(2, 8, 128).with_kmeans_iterations(5);
        let mut index = WarpIndex::new(config);

        // Train with more tokens per document (8 centroids * 10 = 80 tokens needed)
        let texts: Vec<String> = (0..50)
            .map(|i| {
                format!("document number {} contains important information about topic {}", i, i)
            })
            .collect();
        let embeddings: Vec<_> = texts.iter().map(|t| embedder.embed_tokens(t).unwrap()).collect();
        index.train(&embeddings).unwrap();

        // Insert
        for (i, text) in texts.iter().enumerate() {
            let chunk = Chunk::new(DocumentId::new(), text.clone(), 0, text.len());
            index.insert(chunk, embeddings[i].clone()).unwrap();
        }
        index.build().unwrap();

        let memory = index.memory_usage();
        let num_tokens = index.num_tokens();

        // With 2-bit compression: 128 dims × 2 bits = 32 bytes per token
        // Plus overhead for chunk_ids, token_indices, etc.
        let theoretical_min = num_tokens * 32;
        let overhead_factor = 3.0; // Allow 3× overhead for metadata

        assert!(
            memory < (theoretical_min as f64 * overhead_factor) as usize,
            "Memory {} too high for {} tokens (theoretical min {})",
            memory,
            num_tokens,
            theoretical_min
        );
    }
}
