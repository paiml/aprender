use super::*;

// ============ PoolingStrategy Tests ============

#[test]
fn test_pooling_strategy_default() {
    assert_eq!(PoolingStrategy::default(), PoolingStrategy::Mean);
}

#[test]
fn test_pooling_strategy_serialization() {
    let strategy = PoolingStrategy::WeightedMean;
    let json = serde_json::to_string(&strategy).unwrap();
    let deserialized: PoolingStrategy = serde_json::from_str(&json).unwrap();
    assert_eq!(strategy, deserialized);
}

// ============ EmbeddingConfig Tests ============

#[test]
fn test_embedding_config_default() {
    let config = EmbeddingConfig::default();
    assert!(config.normalize);
    assert!(config.query_prefix.is_none());
    assert!(config.document_prefix.is_none());
    assert_eq!(config.max_length, 512);
    assert_eq!(config.pooling, PoolingStrategy::Mean);
}

#[test]
fn test_embedding_config_serialization() {
    let config = EmbeddingConfig {
        normalize: false,
        query_prefix: Some("query: ".to_string()),
        document_prefix: Some("passage: ".to_string()),
        max_length: 256,
        pooling: PoolingStrategy::Cls,
    };

    let json = serde_json::to_string(&config).unwrap();
    let deserialized: EmbeddingConfig = serde_json::from_str(&json).unwrap();

    assert_eq!(config.normalize, deserialized.normalize);
    assert_eq!(config.query_prefix, deserialized.query_prefix);
    assert_eq!(config.max_length, deserialized.max_length);
}

#[test]
fn test_embedding_config_debug() {
    let config = EmbeddingConfig::default();
    let debug_str = format!("{config:?}");
    assert!(debug_str.contains("EmbeddingConfig"));
}

#[test]
fn test_embedding_config_clone() {
    let config = EmbeddingConfig {
        normalize: false,
        query_prefix: Some("q: ".to_string()),
        ..Default::default()
    };
    let cloned = config.clone();
    assert!(!cloned.normalize);
    assert_eq!(cloned.query_prefix, Some("q: ".to_string()));
}

// ============ MockEmbedder Tests ============

#[test]
fn test_mock_embedder_new() {
    let embedder = MockEmbedder::new(384);
    assert_eq!(embedder.dimension(), 384);
    assert_eq!(embedder.model_id(), "mock-embedder");
}

#[test]
fn test_mock_embedder_with_model_id() {
    let embedder = MockEmbedder::new(768).with_model_id("custom-model");
    assert_eq!(embedder.model_id(), "custom-model");
}

#[test]
fn test_mock_embedder_embed() {
    let embedder = MockEmbedder::new(128);
    let embedding = embedder.embed("Hello world").unwrap();

    assert_eq!(embedding.len(), 128);
    // Check normalization
    let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 1e-5);
}

#[test]
fn test_mock_embedder_embed_empty() {
    let embedder = MockEmbedder::new(128);
    let result = embedder.embed("");
    assert!(result.is_err());
}

#[test]
fn test_mock_embedder_embed_batch() {
    let embedder = MockEmbedder::new(64);
    let texts = vec!["Hello", "World", "Test"];
    let embeddings = embedder.embed_batch(&texts).unwrap();

    assert_eq!(embeddings.len(), 3);
    for emb in &embeddings {
        assert_eq!(emb.len(), 64);
    }
}

#[test]
fn test_mock_embedder_deterministic() {
    let embedder = MockEmbedder::new(128);
    let emb1 = embedder.embed("Hello").unwrap();
    let emb2 = embedder.embed("Hello").unwrap();
    assert_eq!(emb1, emb2);
}

#[test]
fn test_mock_embedder_different_texts() {
    let embedder = MockEmbedder::new(128);
    let emb1 = embedder.embed("Hello").unwrap();
    let emb2 = embedder.embed("World").unwrap();
    assert_ne!(emb1, emb2);
}

#[test]
fn test_mock_embedder_query_prefix() {
    let config =
        EmbeddingConfig { query_prefix: Some("query: ".to_string()), ..Default::default() };
    let embedder = MockEmbedder::new(128).with_config(config);

    let query_emb = embedder.embed_query("test").unwrap();
    let doc_emb = embedder.embed_document("test").unwrap();

    // Query and doc embeddings should differ due to prefix
    assert_ne!(query_emb, doc_emb);
}

#[test]
fn test_mock_embedder_embed_chunks() {
    use crate::DocumentId;
    let embedder = MockEmbedder::new(64);

    let doc_id = DocumentId::new();
    let mut chunks = vec![
        Chunk::new(doc_id, "First chunk".to_string(), 0, 11),
        Chunk::new(doc_id, "Second chunk".to_string(), 12, 24),
    ];

    embedder.embed_chunks(&mut chunks).unwrap();

    for chunk in &chunks {
        assert!(chunk.embedding.is_some());
        assert_eq!(chunk.embedding.as_ref().unwrap().len(), 64);
    }
}

#[test]
fn test_mock_embedder_no_normalize() {
    let config = EmbeddingConfig { normalize: false, ..Default::default() };
    let embedder = MockEmbedder::new(128).with_config(config);
    let emb = embedder.embed("test").unwrap();
    assert_eq!(emb.len(), 128);
    // Without normalization, norm is unlikely to be 1.0
    let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
    // Just verify it ran, not necessarily unit length
    assert!(norm > 0.0);
}

#[test]
fn test_mock_embedder_with_document_prefix() {
    let config =
        EmbeddingConfig { document_prefix: Some("doc: ".to_string()), ..Default::default() };
    let embedder = MockEmbedder::new(64).with_config(config);

    let emb1 = embedder.embed("test").unwrap();
    let embedder_no_prefix = MockEmbedder::new(64);
    let emb2 = embedder_no_prefix.embed("test").unwrap();

    // With prefix, embeddings should differ
    assert_ne!(emb1, emb2);
}

#[test]
fn test_mock_embedder_embed_query_empty() {
    let embedder = MockEmbedder::new(64);
    let result = embedder.embed_query("");
    assert!(result.is_err());
}

#[test]
fn test_mock_embedder_normalize_vector_zero() {
    // Test normalizing a zero vector (edge case where norm == 0.0)
    let mut zero_vec = vec![0.0; 10];
    MockEmbedder::normalize_vector(&mut zero_vec);
    // Should remain unchanged (can't normalize a zero vector)
    assert!(zero_vec.iter().all(|&x| x == 0.0));
}

#[test]
fn test_mock_embedder_debug() {
    let embedder = MockEmbedder::new(64);
    let debug_str = format!("{embedder:?}");
    assert!(debug_str.contains("MockEmbedder"));
}

#[test]
fn test_mock_embedder_clone() {
    let embedder = MockEmbedder::new(64).with_model_id("test");
    let cloned = embedder.clone();
    assert_eq!(cloned.model_id(), "test");
    assert_eq!(cloned.dimension(), 64);
}

// ============ Trait Default Implementation Tests ============

// Minimal embedder that uses default trait implementations
struct MinimalEmbedder {
    dim: usize,
}

impl Embedder for MinimalEmbedder {
    fn embed(&self, text: &str) -> Result<Vec<f32>> {
        if text.is_empty() {
            return Err(Error::EmptyDocument("empty".to_string()));
        }
        Ok(vec![1.0; self.dim])
    }

    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        texts.iter().map(|t| self.embed(t)).collect()
    }

    fn dimension(&self) -> usize {
        self.dim
    }

    fn model_id(&self) -> &str {
        "minimal"
    }
}

#[test]
fn test_trait_default_embed_query() {
    let embedder = MinimalEmbedder { dim: 64 };
    let result = embedder.embed_query("test query");
    assert!(result.is_ok());
    assert_eq!(result.unwrap().len(), 64);
}

#[test]
fn test_trait_default_embed_document() {
    let embedder = MinimalEmbedder { dim: 128 };
    let result = embedder.embed_document("test document");
    assert!(result.is_ok());
    assert_eq!(result.unwrap().len(), 128);
}

#[test]
fn test_trait_default_embed_chunks() {
    use crate::DocumentId;
    let embedder = MinimalEmbedder { dim: 32 };
    let doc_id = DocumentId::new();
    let mut chunks = vec![
        Chunk::new(doc_id, "chunk1".to_string(), 0, 6),
        Chunk::new(doc_id, "chunk2".to_string(), 7, 13),
    ];

    embedder.embed_chunks(&mut chunks).unwrap();

    for chunk in &chunks {
        assert!(chunk.embedding.is_some());
        assert_eq!(chunk.embedding.as_ref().unwrap().len(), 32);
    }
}

// ============ PoolingStrategy Tests ============

#[test]
fn test_pooling_strategy_variants() {
    let cls = PoolingStrategy::Cls;
    let mean = PoolingStrategy::Mean;
    let weighted = PoolingStrategy::WeightedMean;
    let last = PoolingStrategy::LastToken;

    assert_ne!(cls, mean);
    assert_ne!(weighted, last);
    assert_eq!(cls, PoolingStrategy::Cls);
}

#[test]
fn test_pooling_strategy_debug() {
    let strategy = PoolingStrategy::LastToken;
    let debug = format!("{strategy:?}");
    assert!(debug.contains("LastToken"));
}

#[test]
fn test_pooling_strategy_clone() {
    let strategy = PoolingStrategy::WeightedMean;
    let cloned = strategy;
    assert_eq!(strategy, cloned);
}

// ============ TfIdfEmbedder Tests ============

#[test]
fn test_tfidf_embedder_new() {
    let embedder = TfIdfEmbedder::new(100);
    assert_eq!(embedder.dimension(), 100);
    assert_eq!(embedder.model_id(), "tfidf");
}

#[test]
fn test_tfidf_embedder_untrained() {
    let embedder = TfIdfEmbedder::new(100);
    let result = embedder.embed("test");
    assert!(result.is_err());
}

#[test]
fn test_tfidf_embedder_fit() {
    let mut embedder = TfIdfEmbedder::new(50);
    let corpus = vec!["the quick brown fox", "the lazy dog", "quick brown dog"];
    embedder.fit(&corpus);

    assert!(!embedder.vocabulary.is_empty());
    assert!(!embedder.idf.is_empty());
}

#[test]
fn test_tfidf_embedder_embed() {
    let mut embedder = TfIdfEmbedder::new(50);
    let corpus = vec!["the quick brown fox", "the lazy dog sleeps", "quick brown lazy fox"];
    embedder.fit(&corpus);

    let embedding = embedder.embed("quick fox").unwrap();
    assert_eq!(embedding.len(), 50);

    // Check normalization
    let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 1e-5 || norm == 0.0);
}

#[test]
fn test_tfidf_embedder_empty() {
    let mut embedder = TfIdfEmbedder::new(50);
    embedder.fit(&["test doc"]);
    let result = embedder.embed("");
    assert!(result.is_err());
}

#[test]
fn test_tfidf_embedder_embed_batch() {
    let mut embedder = TfIdfEmbedder::new(50);
    embedder.fit(&["the quick brown", "lazy fox"]);
    let embeddings = embedder.embed_batch(&["quick", "lazy"]).unwrap();
    assert_eq!(embeddings.len(), 2);
    for emb in &embeddings {
        assert_eq!(emb.len(), 50);
    }
}

#[test]
fn test_tfidf_embedder_oov_terms() {
    // Test with out-of-vocabulary terms (zero norm case)
    let mut embedder = TfIdfEmbedder::new(50);
    embedder.fit(&["alpha beta gamma"]);
    // Query with terms not in vocabulary
    let emb = embedder.embed("xyz unknown terms").unwrap();
    assert_eq!(emb.len(), 50);
    // All values should be 0.0 since no terms matched
    assert!(emb.iter().all(|&x| x == 0.0));
}

#[test]
fn test_tfidf_embedder_dimension_larger_than_vocab() {
    let mut embedder = TfIdfEmbedder::new(1000);
    embedder.fit(&["hello world"]); // Only 2 terms
    let emb = embedder.embed("hello").unwrap();
    assert_eq!(emb.len(), 1000);
}

#[test]
fn test_tfidf_embedder_debug() {
    let embedder = TfIdfEmbedder::new(50);
    let debug_str = format!("{embedder:?}");
    assert!(debug_str.contains("TfIdfEmbedder"));
}

#[test]
fn test_tfidf_embedder_clone() {
    let mut embedder = TfIdfEmbedder::new(50);
    embedder.fit(&["hello world"]);
    let cloned = embedder.clone();
    assert_eq!(cloned.dimension(), 50);
    // Both should embed the same way
    let emb1 = embedder.embed("hello").unwrap();
    let emb2 = cloned.embed("hello").unwrap();
    assert_eq!(emb1, emb2);
}

#[test]
fn test_tfidf_embedder_embed_query_passthrough() {
    // TfIdfEmbedder uses default trait impl for embed_query
    let mut embedder = TfIdfEmbedder::new(50);
    embedder.fit(&["hello world test"]);
    let query_emb = embedder.embed_query("hello").unwrap();
    let doc_emb = embedder.embed_document("hello").unwrap();
    // Should be the same since no special handling
    assert_eq!(query_emb, doc_emb);
}

// ============ Similarity Functions Tests ============

#[test]
fn test_cosine_similarity_identical() {
    let a = vec![1.0, 0.0, 0.0];
    let b = vec![1.0, 0.0, 0.0];
    let sim = cosine_similarity(&a, &b);
    assert!((sim - 1.0).abs() < 1e-5);
}

#[test]
fn test_cosine_similarity_orthogonal() {
    let a = vec![1.0, 0.0, 0.0];
    let b = vec![0.0, 1.0, 0.0];
    let sim = cosine_similarity(&a, &b);
    assert!(sim.abs() < 1e-5);
}

#[test]
fn test_cosine_similarity_opposite() {
    let a = vec![1.0, 0.0, 0.0];
    let b = vec![-1.0, 0.0, 0.0];
    let sim = cosine_similarity(&a, &b);
    assert!((sim + 1.0).abs() < 1e-5);
}

#[test]
fn test_cosine_similarity_different_lengths() {
    let a = vec![1.0, 0.0];
    let b = vec![1.0, 0.0, 0.0];
    let sim = cosine_similarity(&a, &b);
    assert!(sim.abs() < f32::EPSILON);
}

#[test]
fn test_cosine_similarity_zero_vector() {
    let a = vec![0.0, 0.0, 0.0];
    let b = vec![1.0, 0.0, 0.0];
    let sim = cosine_similarity(&a, &b);
    assert!(sim.abs() < f32::EPSILON);
}

#[test]
fn test_dot_product() {
    let a = vec![1.0, 2.0, 3.0];
    let b = vec![4.0, 5.0, 6.0];
    let dot = dot_product(&a, &b);
    assert!((dot - 32.0).abs() < 1e-5);
}

#[test]
fn test_dot_product_orthogonal() {
    let a = vec![1.0, 0.0];
    let b = vec![0.0, 1.0];
    let dot = dot_product(&a, &b);
    assert!(dot.abs() < 1e-5);
}

#[test]
fn test_euclidean_distance_same() {
    let a = vec![1.0, 2.0, 3.0];
    let b = vec![1.0, 2.0, 3.0];
    let dist = euclidean_distance(&a, &b);
    assert!(dist.abs() < 1e-5);
}

#[test]
fn test_euclidean_distance() {
    let a = vec![0.0, 0.0];
    let b = vec![3.0, 4.0];
    let dist = euclidean_distance(&a, &b);
    assert!((dist - 5.0).abs() < 1e-5);
}

// ============ EmbeddingModelType Tests (GH-1) ============

#[cfg(feature = "embeddings")]
mod fastembed_tests {
    use super::*;

    #[test]
    fn test_embedding_model_type_default() {
        assert_eq!(EmbeddingModelType::default(), EmbeddingModelType::AllMiniLmL6V2);
    }

    #[test]
    fn test_embedding_model_type_dimension_mini_lm() {
        assert_eq!(EmbeddingModelType::AllMiniLmL6V2.dimension(), 384);
        assert_eq!(EmbeddingModelType::AllMiniLmL12V2.dimension(), 384);
    }

    #[test]
    fn test_embedding_model_type_dimension_bge() {
        assert_eq!(EmbeddingModelType::BgeSmallEnV15.dimension(), 384);
        assert_eq!(EmbeddingModelType::BgeBaseEnV15.dimension(), 768);
    }

    #[test]
    fn test_embedding_model_type_dimension_nomic() {
        assert_eq!(EmbeddingModelType::NomicEmbedTextV1.dimension(), 768);
    }

    #[test]
    fn test_embedding_model_type_model_name_mini_lm() {
        assert_eq!(
            EmbeddingModelType::AllMiniLmL6V2.model_name(),
            "sentence-transformers/all-MiniLM-L6-v2"
        );
        assert_eq!(
            EmbeddingModelType::AllMiniLmL12V2.model_name(),
            "sentence-transformers/all-MiniLM-L12-v2"
        );
    }

    #[test]
    fn test_embedding_model_type_model_name_bge() {
        assert_eq!(EmbeddingModelType::BgeSmallEnV15.model_name(), "BAAI/bge-small-en-v1.5");
        assert_eq!(EmbeddingModelType::BgeBaseEnV15.model_name(), "BAAI/bge-base-en-v1.5");
    }

    #[test]
    fn test_embedding_model_type_model_name_nomic() {
        assert_eq!(
            EmbeddingModelType::NomicEmbedTextV1.model_name(),
            "nomic-ai/nomic-embed-text-v1"
        );
    }

    #[test]
    fn test_embedding_model_type_to_fastembed() {
        // Verify conversion doesn't panic for any variant
        let _ = EmbeddingModelType::AllMiniLmL6V2.to_fastembed_model();
        let _ = EmbeddingModelType::AllMiniLmL12V2.to_fastembed_model();
        let _ = EmbeddingModelType::BgeSmallEnV15.to_fastembed_model();
        let _ = EmbeddingModelType::BgeBaseEnV15.to_fastembed_model();
        let _ = EmbeddingModelType::NomicEmbedTextV1.to_fastembed_model();
    }

    #[test]
    fn test_embedding_model_type_clone() {
        let model = EmbeddingModelType::BgeBaseEnV15;
        let cloned = model;
        assert_eq!(model, cloned);
    }

    #[test]
    fn test_embedding_model_type_debug() {
        let model = EmbeddingModelType::AllMiniLmL6V2;
        let debug_str = format!("{model:?}");
        assert!(debug_str.contains("AllMiniLmL6V2"));
    }

    // ============ FastEmbedder Integration Tests ============
    // These tests require ONNX Runtime and model downloads.
    // Run with: cargo test --features embeddings -- --ignored

    #[test]
    #[ignore = "Requires ONNX model download"]
    fn test_fastembedder_new() {
        let embedder = FastEmbedder::new(EmbeddingModelType::AllMiniLmL6V2)
            .expect("Failed to create embedder");
        assert_eq!(embedder.dimension(), 384);
        assert_eq!(embedder.model_type(), EmbeddingModelType::AllMiniLmL6V2);
        assert_eq!(embedder.model_id(), "sentence-transformers/all-MiniLM-L6-v2");
    }

    #[test]
    #[ignore = "Requires ONNX model download"]
    fn test_fastembedder_default_model() {
        let embedder = FastEmbedder::default_model().expect("Failed to create embedder");
        assert_eq!(embedder.dimension(), 384);
    }

    #[test]
    #[ignore = "Requires ONNX model download"]
    fn test_fastembedder_embed() {
        let embedder = FastEmbedder::new(EmbeddingModelType::AllMiniLmL6V2)
            .expect("Failed to create embedder");
        let embedding = embedder.embed("Hello world").expect("Failed to embed");
        assert_eq!(embedding.len(), 384);
    }

    #[test]
    #[ignore = "Requires ONNX model download"]
    fn test_fastembedder_embed_empty_error() {
        let embedder = FastEmbedder::new(EmbeddingModelType::AllMiniLmL6V2)
            .expect("Failed to create embedder");
        let result = embedder.embed("");
        assert!(result.is_err());
    }

    #[test]
    #[ignore = "Requires ONNX model download"]
    fn test_fastembedder_embed_batch() {
        let embedder = FastEmbedder::new(EmbeddingModelType::AllMiniLmL6V2)
            .expect("Failed to create embedder");
        let embeddings = embedder.embed_batch(&["Hello", "World"]).expect("Failed to batch embed");
        assert_eq!(embeddings.len(), 2);
        for emb in &embeddings {
            assert_eq!(emb.len(), 384);
        }
    }

    #[test]
    #[ignore = "Requires ONNX model download"]
    fn test_fastembedder_embed_batch_empty() {
        let embedder = FastEmbedder::new(EmbeddingModelType::AllMiniLmL6V2)
            .expect("Failed to create embedder");
        let embeddings = embedder.embed_batch(&[]).expect("Failed to batch embed");
        assert!(embeddings.is_empty());
    }

    #[test]
    #[ignore = "Requires ONNX model download"]
    fn test_fastembedder_embed_batch_all_empty_error() {
        let embedder = FastEmbedder::new(EmbeddingModelType::AllMiniLmL6V2)
            .expect("Failed to create embedder");
        let result = embedder.embed_batch(&["", ""]);
        assert!(result.is_err());
    }

    #[test]
    #[ignore = "Requires ONNX model download"]
    fn test_fastembedder_query_and_document() {
        let embedder = FastEmbedder::new(EmbeddingModelType::AllMiniLmL6V2)
            .expect("Failed to create embedder");
        let query_emb = embedder.embed_query("What is AI?").expect("Failed to embed query");
        let doc_emb =
            embedder.embed_document("AI is artificial intelligence").expect("Failed to embed doc");
        assert_eq!(query_emb.len(), 384);
        assert_eq!(doc_emb.len(), 384);
    }

    #[test]
    #[ignore = "Requires ONNX model download"]
    fn test_fastembedder_debug() {
        let embedder = FastEmbedder::new(EmbeddingModelType::AllMiniLmL6V2)
            .expect("Failed to create embedder");
        let debug_str = format!("{embedder:?}");
        assert!(debug_str.contains("FastEmbedder"));
        assert!(debug_str.contains("AllMiniLmL6V2"));
    }

    #[test]
    #[ignore = "Requires ONNX model download"]
    fn test_fastembedder_clone() {
        let embedder = FastEmbedder::new(EmbeddingModelType::AllMiniLmL6V2)
            .expect("Failed to create embedder");
        let cloned = embedder.clone();
        assert_eq!(cloned.model_type(), EmbeddingModelType::AllMiniLmL6V2);
        // Both should produce the same embedding
        let emb1 = embedder.embed("test").expect("embed1");
        let emb2 = cloned.embed("test").expect("embed2");
        assert_eq!(emb1.len(), emb2.len());
    }
}

// ============ Property-Based Tests ============

use proptest::prelude::*;

proptest! {
    #[test]
    fn prop_mock_embedder_dimension(dim in 1usize..1000) {
        let embedder = MockEmbedder::new(dim);
        let emb = embedder.embed("test").unwrap();
        prop_assert_eq!(emb.len(), dim);
    }

    #[test]
    fn prop_mock_embedder_normalized(text in "[a-zA-Z ]{1,100}") {
        let embedder = MockEmbedder::new(128);
        let emb = embedder.embed(&text).unwrap();
        let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
        prop_assert!((norm - 1.0).abs() < 1e-4);
    }

    #[test]
    fn prop_cosine_similarity_range(
        a in prop::collection::vec(-1.0f32..1.0, 10),
        b in prop::collection::vec(-1.0f32..1.0, 10)
    ) {
        let sim = cosine_similarity(&a, &b);
        prop_assert!(sim >= -1.0 - 1e-5);
        prop_assert!(sim <= 1.0 + 1e-5);
    }

    #[test]
    fn prop_euclidean_distance_non_negative(
        a in prop::collection::vec(-100.0f32..100.0, 5),
        b in prop::collection::vec(-100.0f32..100.0, 5)
    ) {
        let dist = euclidean_distance(&a, &b);
        prop_assert!(dist >= 0.0);
    }
}
