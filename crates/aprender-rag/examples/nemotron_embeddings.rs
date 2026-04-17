//! Example: NVIDIA Embed Nemotron 8B embeddings
//!
//! Demonstrates using the NemotronEmbedder for semantic embeddings with
//! asymmetric retrieval (different prefixes for queries vs documents).
//!
//! # Requirements
//!
//! 1. Download a Nemotron GGUF model (e.g., from HuggingFace)
//! 2. Set the model path in the code below
//!
//! # Usage
//!
//! ```bash
//! cargo run --example nemotron_embeddings --features nemotron
//! ```

#[cfg(feature = "nemotron")]
use trueno_rag::embed::{cosine_similarity, Embedder, NemotronConfig, NemotronEmbedder};

#[cfg(feature = "nemotron")]
fn main() -> trueno_rag::Result<()> {
    // Configure the embedder
    // Replace with your actual model path
    let model_path = std::env::var("NEMOTRON_MODEL_PATH")
        .unwrap_or_else(|_| "models/NV-Embed-v2-Q4_K.gguf".to_string());

    println!("Loading Nemotron model from: {}", model_path);

    let config = NemotronConfig::new(&model_path)
        .with_gpu(true) // Use GPU if available
        .with_batch_size(8)
        .with_normalize(true);

    let embedder = match NemotronEmbedder::new(config) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("Failed to load model: {}", e);
            eprintln!("\nTo run this example:");
            eprintln!("1. Download a Nemotron GGUF model");
            eprintln!("2. Set NEMOTRON_MODEL_PATH environment variable");
            eprintln!("   export NEMOTRON_MODEL_PATH=/path/to/model.gguf");
            return Ok(());
        }
    };

    println!("Loaded Nemotron embedder: {} dimensions", embedder.dimension());

    // Example documents
    let documents = vec![
        "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
        "Neural networks are computing systems inspired by biological neural networks in the brain.",
        "Deep learning uses multiple layers of neural networks to progressively extract features.",
        "The stock market saw significant gains today as tech companies reported strong earnings.",
    ];

    println!("\n=== Embedding Documents ===\n");
    let doc_embeddings: Vec<_> = documents
        .iter()
        .map(|doc| {
            let emb = embedder.embed_document(doc).expect("Failed to embed");
            println!("Document: {}...", &doc[..50.min(doc.len())]);
            println!("  Embedding dim: {}", emb.len());
            emb
        })
        .collect();

    // Example query with asymmetric retrieval
    let query = "What is machine learning?";
    println!("\n=== Query: {} ===\n", query);

    let query_embedding = embedder.embed_query(query)?;
    println!("Query embedding dim: {}", query_embedding.len());

    // Compute similarities
    println!("\n=== Similarities ===\n");
    let mut scored: Vec<_> = documents
        .iter()
        .zip(doc_embeddings.iter())
        .map(|(doc, emb)| {
            let sim = cosine_similarity(&query_embedding, emb);
            (sim, doc)
        })
        .collect();

    // Sort by similarity (descending)
    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    for (i, (score, doc)) in scored.iter().enumerate() {
        println!("{}. [{:.4}] {}", i + 1, score, &doc[..60.min(doc.len())]);
    }

    // Demonstrate asymmetric retrieval difference
    println!("\n=== Asymmetric Retrieval Demo ===\n");
    let same_text = "What is machine learning?";
    let query_emb = embedder.embed_query(same_text)?;
    let doc_emb = embedder.embed_document(same_text)?;

    let self_sim = cosine_similarity(&query_emb, &doc_emb);
    println!("Same text as query vs document:");
    println!("  Text: \"{}\"", same_text);
    println!("  Cosine similarity: {:.4}", self_sim);
    println!("  (Not 1.0 because query uses instruction prefix for asymmetric retrieval)");

    Ok(())
}

#[cfg(not(feature = "nemotron"))]
fn main() {
    eprintln!("This example requires the 'nemotron' feature.");
    eprintln!("Run with: cargo run --example nemotron_embeddings --features nemotron");
}
