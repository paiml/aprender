//! Semantic Embeddings Example (GH-1)
//!
//! Demonstrates production-quality semantic search using FastEmbedder.
//!
//! Run with: cargo run --example semantic_embeddings --features embeddings
//!
//! This example requires the `embeddings` feature which adds fastembed (ONNX Runtime).
//! The first run will download the model (~90MB for MiniLM).

#[cfg(feature = "embeddings")]
use trueno_rag::{
    chunk::RecursiveChunker, embed::Embedder, fusion::FusionStrategy, pipeline::RagPipelineBuilder,
    rerank::LexicalReranker, Document, EmbeddingModelType, FastEmbedder,
};

#[cfg(feature = "embeddings")]
fn sample_documents() -> Vec<Document> {
    vec![
        Document::new(
            "Retrieval-Augmented Generation (RAG) combines the power of large \
             language models with external knowledge retrieval. Instead of relying \
             solely on parametric knowledge, RAG systems fetch relevant documents \
             from a knowledge base to ground their responses in factual information.",
        )
        .with_title("RAG Overview"),
        Document::new(
            "Vector databases store high-dimensional embeddings and enable fast \
             similarity search. They use approximate nearest neighbor algorithms \
             like HNSW or IVF to find similar vectors in milliseconds, even with \
             billions of vectors.",
        )
        .with_title("Vector Databases"),
        Document::new(
            "Sentence transformers are neural networks trained to produce meaningful \
             sentence embeddings. Unlike word2vec which embeds individual words, \
             sentence transformers capture the semantic meaning of entire sentences, \
             making them ideal for semantic search applications.",
        )
        .with_title("Sentence Transformers"),
        Document::new(
            "The Rust programming language provides memory safety without garbage \
             collection through its ownership system. The borrow checker enforces \
             rules at compile time, preventing data races and use-after-free bugs.",
        )
        .with_title("Rust Memory Safety"),
        Document::new(
            "ONNX Runtime is a high-performance inference engine for machine learning \
             models. It supports models from PyTorch, TensorFlow, and other frameworks, \
             providing optimized execution on CPU, GPU, and specialized hardware.",
        )
        .with_title("ONNX Runtime"),
    ]
}

#[cfg(feature = "embeddings")]
fn demo_similarity() -> trueno_rag::Result<()> {
    println!("=== Embedding Similarity Demo ===\n");
    let embedder = FastEmbedder::new(EmbeddingModelType::AllMiniLmL6V2)?;

    let texts = [
        "The cat sat on the mat.",
        "A feline rested on the rug.",
        "The stock market crashed today.",
    ];

    let embeddings: Vec<Vec<f32>> =
        texts.iter().map(|t| embedder.embed(t)).collect::<Result<_, _>>()?;

    println!("Cosine similarities:");
    println!(
        "  '{}' vs '{}': {:.3}",
        texts[0],
        texts[1],
        cosine_similarity(&embeddings[0], &embeddings[1])
    );
    println!(
        "  '{}' vs '{}': {:.3}",
        texts[0],
        texts[2],
        cosine_similarity(&embeddings[0], &embeddings[2])
    );
    println!("\nNote: Semantically similar sentences have higher scores!");
    Ok(())
}

#[cfg(feature = "embeddings")]
fn demo_retrieval() -> trueno_rag::Result<()> {
    println!("Loading embedding model (first run downloads ~90MB)...");
    let embedder = FastEmbedder::new(EmbeddingModelType::AllMiniLmL6V2)?;
    println!("Model: {} (dimension: {})\n", embedder.model_id(), embedder.dimension());

    let mut pipeline = RagPipelineBuilder::new()
        .chunker(RecursiveChunker::new(256, 32))
        .embedder(embedder)
        .reranker(LexicalReranker::new())
        .fusion(FusionStrategy::RRF { k: 60.0 })
        .max_context_tokens(2048)
        .build()?;

    let documents = sample_documents();
    println!("Indexing {} documents...", documents.len());
    let chunk_count = pipeline.index_documents(&documents)?;
    println!("Created {} chunks with semantic embeddings\n", chunk_count);

    let queries = [
        "How do AI systems access external knowledge?",
        "What technology enables finding similar items quickly?",
        "How can I convert text into numerical representations?",
        "What is the capital of France?",
    ];

    for query in queries {
        run_query(&mut pipeline, query)?;
    }
    Ok(())
}

#[cfg(feature = "embeddings")]
fn main() -> trueno_rag::Result<()> {
    println!("=== Semantic Embeddings Example ===\n");
    demo_retrieval()?;
    demo_similarity()
}

#[cfg(feature = "embeddings")]
fn run_query(
    pipeline: &mut trueno_rag::pipeline::RagPipeline<FastEmbedder, LexicalReranker>,
    query: &str,
) -> trueno_rag::Result<()> {
    println!("Query: \"{}\"\n", query);
    let results = pipeline.query(query, 2)?;
    if results.is_empty() {
        println!("  No relevant results found\n");
    } else {
        for (i, result) in results.iter().enumerate() {
            let title = result.chunk.metadata.title.as_deref().unwrap_or("Untitled");
            let preview = &result.chunk.content[..80.min(result.chunk.content.len())];
            println!(
                "  {}. [Score: {:.3}] {}\n     {}...\n",
                i + 1,
                result.best_score(),
                title,
                preview
            );
        }
    }
    println!("{}\n", "-".repeat(60));
    Ok(())
}

#[cfg(feature = "embeddings")]
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
}

#[cfg(not(feature = "embeddings"))]
fn main() {
    eprintln!("This example requires the 'embeddings' feature.");
    eprintln!("Run with: cargo run --example semantic_embeddings --features embeddings");
    std::process::exit(1);
}
