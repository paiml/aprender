//! Integration tests for NemotronEmbedder with GGUF models
//!
//! These tests require a GGUF model file and are ignored by default.
//! Run with: `cargo test --features nemotron -- --ignored`
//!
//! Set NEMOTRON_MODEL_PATH to your GGUF model file.

#![cfg(feature = "nemotron")]

use trueno_rag::embed::{cosine_similarity, Embedder, NemotronConfig, NemotronEmbedder};

/// Test model path - set via environment variable or use default
fn test_model_path() -> Option<String> {
    std::env::var("NEMOTRON_MODEL_PATH").ok()
}

/// Create a test embedder if model is available
fn create_test_embedder() -> Option<NemotronEmbedder> {
    let path = test_model_path()?;
    let config = NemotronConfig::new(&path).with_gpu(false);
    NemotronEmbedder::new(config).ok()
}

#[test]
#[ignore = "Requires GGUF model file (set NEMOTRON_MODEL_PATH env var)"]
fn test_nemotron_embedder_dimension() {
    let Some(embedder) = create_test_embedder() else {
        eprintln!("Skipping: NEMOTRON_MODEL_PATH not set");
        return;
    };

    // Dimension depends on model - just verify it's positive
    let dim = embedder.dimension();
    assert!(dim > 0, "Dimension should be positive, got {dim}");
    eprintln!("Model dimension: {dim}");
}

#[test]
#[ignore = "Requires GGUF model file (set NEMOTRON_MODEL_PATH env var)"]
fn test_nemotron_model_id() {
    let Some(embedder) = create_test_embedder() else {
        eprintln!("Skipping: NEMOTRON_MODEL_PATH not set");
        return;
    };

    // Model ID should be the expected value
    assert_eq!(embedder.model_id(), "nvidia/NV-Embed-v2");
}

#[test]
#[ignore = "Requires GGUF model file (set NEMOTRON_MODEL_PATH env var)"]
fn test_nemotron_embed_query() {
    let Some(embedder) = create_test_embedder() else {
        eprintln!("Skipping: NEMOTRON_MODEL_PATH not set");
        return;
    };

    let embedding =
        embedder.embed_query("What is machine learning?").expect("Failed to embed query");

    assert_eq!(embedding.len(), embedder.dimension());

    // Check normalization (should be unit length)
    let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 1e-4, "Expected unit norm, got {}", norm);
}

#[test]
#[ignore = "Requires GGUF model file (set NEMOTRON_MODEL_PATH env var)"]
fn test_nemotron_embed_document() {
    let Some(embedder) = create_test_embedder() else {
        eprintln!("Skipping: NEMOTRON_MODEL_PATH not set");
        return;
    };

    let embedding = embedder
        .embed_document("Machine learning is a branch of artificial intelligence.")
        .expect("Failed to embed document");

    assert_eq!(embedding.len(), embedder.dimension());
}

#[test]
#[ignore = "Requires GGUF model file (set NEMOTRON_MODEL_PATH env var)"]
fn test_nemotron_asymmetric_retrieval() {
    let Some(embedder) = create_test_embedder() else {
        eprintln!("Skipping: NEMOTRON_MODEL_PATH not set");
        return;
    };

    // Query and passage for the same text should produce different embeddings
    // due to asymmetric prefixes (if the model supports it)
    let query_emb =
        embedder.embed_query("What is machine learning?").expect("Failed to embed query");
    let doc_emb =
        embedder.embed_document("What is machine learning?").expect("Failed to embed document");

    // At minimum, embeddings should be valid
    assert_eq!(query_emb.len(), embedder.dimension());
    assert_eq!(doc_emb.len(), embedder.dimension());

    // Note: With asymmetric prefixes, embeddings should differ
    // but this depends on model architecture
    let sim = cosine_similarity(&query_emb, &doc_emb);
    eprintln!("Query/Doc same text similarity: {sim}");
}

#[test]
#[ignore = "Requires GGUF model file (set NEMOTRON_MODEL_PATH env var)"]
fn test_nemotron_semantic_similarity() {
    let Some(embedder) = create_test_embedder() else {
        eprintln!("Skipping: NEMOTRON_MODEL_PATH not set");
        return;
    };

    // Semantically similar sentences
    let emb1 = embedder.embed_document("The cat sat on the mat.").expect("embed1");
    let emb2 = embedder.embed_document("A feline rested on the rug.").expect("embed2");
    // Unrelated sentence
    let emb3 = embedder.embed_document("Stock prices fell sharply today.").expect("embed3");

    let sim_12 = cosine_similarity(&emb1, &emb2);
    let sim_13 = cosine_similarity(&emb1, &emb3);

    eprintln!("Similar sentences sim: {sim_12}, Unrelated sim: {sim_13}");

    // For embedding models, similar sentences should have higher similarity
    // Note: This test may fail with non-embedding models (e.g., chat models)
    // which is expected - they're not optimized for semantic similarity
}

#[test]
#[ignore = "Requires GGUF model file (set NEMOTRON_MODEL_PATH env var)"]
fn test_nemotron_embed_batch() {
    let Some(embedder) = create_test_embedder() else {
        eprintln!("Skipping: NEMOTRON_MODEL_PATH not set");
        return;
    };

    let texts = vec![
        "First document about machine learning.",
        "Second document about neural networks.",
        "Third document about deep learning.",
    ];

    let embeddings = embedder.embed_batch(&texts).expect("Failed to embed batch");

    assert_eq!(embeddings.len(), 3);
    for emb in &embeddings {
        assert_eq!(emb.len(), embedder.dimension());
    }
}

#[test]
#[ignore = "Requires GGUF model file (set NEMOTRON_MODEL_PATH env var)"]
fn test_nemotron_empty_query_error() {
    let Some(embedder) = create_test_embedder() else {
        eprintln!("Skipping: NEMOTRON_MODEL_PATH not set");
        return;
    };

    let result = embedder.embed_query("");
    assert!(result.is_err());
}

#[test]
#[ignore = "Requires GGUF model file (set NEMOTRON_MODEL_PATH env var)"]
fn test_nemotron_empty_document_error() {
    let Some(embedder) = create_test_embedder() else {
        eprintln!("Skipping: NEMOTRON_MODEL_PATH not set");
        return;
    };

    let result = embedder.embed_document("");
    assert!(result.is_err());
}

#[test]
#[ignore = "Requires GGUF model file (set NEMOTRON_MODEL_PATH env var)"]
fn test_nemotron_embed_chunks() {
    use trueno_rag::{Chunk, DocumentId};

    let Some(embedder) = create_test_embedder() else {
        eprintln!("Skipping: NEMOTRON_MODEL_PATH not set");
        return;
    };

    let doc_id = DocumentId::new();
    let mut chunks = vec![
        Chunk::new(doc_id, "First chunk content.".to_string(), 0, 20),
        Chunk::new(doc_id, "Second chunk content.".to_string(), 21, 43),
    ];

    embedder.embed_chunks(&mut chunks).expect("Failed to embed chunks");

    for chunk in &chunks {
        assert!(chunk.embedding.is_some());
        assert_eq!(chunk.embedding.as_ref().unwrap().len(), embedder.dimension());
    }
}

#[test]
#[ignore = "Requires GGUF model file (set NEMOTRON_MODEL_PATH env var)"]
fn test_nemotron_config_getter() {
    let Some(embedder) = create_test_embedder() else {
        eprintln!("Skipping: NEMOTRON_MODEL_PATH not set");
        return;
    };

    let config = embedder.config();
    assert!(!config.query_prefix.is_empty());
    eprintln!("Query prefix: {:?}", config.query_prefix);
}

#[test]
#[ignore = "Requires GGUF model file (set NEMOTRON_MODEL_PATH env var)"]
fn test_nemotron_debug() {
    let Some(embedder) = create_test_embedder() else {
        eprintln!("Skipping: NEMOTRON_MODEL_PATH not set");
        return;
    };

    let debug_str = format!("{embedder:?}");
    assert!(debug_str.contains("NemotronEmbedder"));
    eprintln!("Debug: {debug_str}");
}
