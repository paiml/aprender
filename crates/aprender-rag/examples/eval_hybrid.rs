//! Hybrid vs Dense vs Sparse Retrieval Comparison
//!
//! Demonstrates Phase 2 hybrid search features:
//! - Building a HybridRetriever with HybridRetrieverConfig
//! - Comparing retrieve() (hybrid), retrieve_dense(), retrieve_sparse()
//! - Computing RetrievalMetrics for each mode
//!
//! Run with: cargo run --example eval_hybrid

use std::collections::HashSet;
use trueno_rag::{
    embed::{Embedder, MockEmbedder},
    fusion::FusionStrategy,
    metrics::RetrievalMetrics,
    retrieve::HybridRetrieverConfig,
    BM25Index, Chunk, ChunkId, DocumentId, HybridRetriever, VectorStore,
};

fn main() -> trueno_rag::Result<()> {
    println!("=== Hybrid vs Dense vs Sparse Retrieval Comparison ===\n");

    // Create sample documents about programming and cloud computing
    let documents = vec![
        "Rust provides memory safety guarantees through its ownership system and borrow checker, eliminating data races at compile time.",
        "AWS Lambda is a serverless compute service that runs code in response to events and automatically manages the underlying compute resources.",
        "Kubernetes orchestrates containerized applications across clusters of machines, handling scaling and failover automatically.",
        "Machine learning models learn patterns from training data to make predictions on new unseen data without explicit programming.",
        "Docker containers package applications with their dependencies into standardized units for consistent deployment across environments.",
        "The Rust compiler enforces strict lifetime rules that prevent use-after-free, double-free, and dangling pointer bugs.",
        "Amazon S3 provides object storage with high availability and durability, commonly used for data lakes and static asset hosting.",
        "Deep learning neural networks with multiple hidden layers can learn hierarchical feature representations from raw data.",
        "Go provides built-in concurrency primitives like goroutines and channels for efficient parallel programming.",
        "Terraform enables infrastructure as code, allowing teams to define and provision cloud resources using declarative configuration files.",
    ];

    // Build hybrid retriever
    let embedder = MockEmbedder::new(384);
    let store = VectorStore::with_dimension(384);
    let bm25 = BM25Index::new();

    let config = HybridRetrieverConfig {
        candidates_per_source: 10,
        fusion: FusionStrategy::RRF { k: 60.0 },
        use_dense: true,
        use_sparse: true,
    };

    let mut retriever = HybridRetriever::new(store, bm25, embedder).with_config(config);

    // Build embedder for pre-computing chunk embeddings
    let embed = MockEmbedder::new(384);

    // Index all documents — track chunk IDs for metrics
    let mut chunk_ids: Vec<ChunkId> = Vec::new();
    let mut content_map: Vec<String> = Vec::new();

    for doc_text in &documents {
        let mut chunk = Chunk::new(DocumentId::new(), (*doc_text).to_string(), 0, doc_text.len());
        chunk.embedding = Some(embed.embed(doc_text)?);
        chunk_ids.push(chunk.id);
        content_map.push((*doc_text).to_string());
        retriever.index(chunk)?;
    }

    println!("Indexed {} documents into hybrid retriever\n", documents.len());

    // Define queries with known relevant documents (by index)
    let queries: Vec<(&str, Vec<usize>)> = vec![
        (
            "memory safety in systems programming",
            vec![0, 5], // Rust ownership + Rust compiler
        ),
        (
            "serverless cloud compute functions",
            vec![1, 6], // Lambda + S3
        ),
        (
            "container orchestration and deployment",
            vec![2, 4], // Kubernetes + Docker
        ),
        (
            "neural network machine learning",
            vec![3, 7], // ML + Deep learning
        ),
    ];

    let k = 5;
    let k_values = vec![1, 3, 5];

    // Compare three retrieval modes
    for (query, relevant_indices) in &queries {
        println!("Query: \"{}\"\n", query);

        let relevant_ids: HashSet<ChunkId> =
            relevant_indices.iter().map(|&i| chunk_ids[i]).collect();

        // Hybrid retrieval (RRF fusion)
        let hybrid_results = retriever.retrieve(query, k)?;
        let hybrid_retrieved: Vec<ChunkId> = hybrid_results.iter().map(|r| r.chunk.id).collect();
        let hybrid_metrics = RetrievalMetrics::compute(&hybrid_retrieved, &relevant_ids, &k_values);

        // Dense-only retrieval
        let dense_results = retriever.retrieve_dense(query, k)?;
        let dense_retrieved: Vec<ChunkId> = dense_results.iter().map(|r| r.chunk.id).collect();
        let dense_metrics = RetrievalMetrics::compute(&dense_retrieved, &relevant_ids, &k_values);

        // Sparse-only retrieval (BM25)
        let sparse_results = retriever.retrieve_sparse(query, k)?;
        let sparse_retrieved: Vec<ChunkId> = sparse_results.iter().map(|r| r.chunk.id).collect();
        let sparse_metrics = RetrievalMetrics::compute(&sparse_retrieved, &relevant_ids, &k_values);

        // Print comparison table
        println!(
            "  {:12} | {:>6} | {:>8} | {:>8} | {:>8}",
            "Mode", "MRR", "NDCG@5", "Recall@5", "Prec@5"
        );
        println!("  {}", "-".repeat(55));

        for (name, metrics) in [
            ("Hybrid/RRF", &hybrid_metrics),
            ("Dense", &dense_metrics),
            ("Sparse/BM25", &sparse_metrics),
        ] {
            println!(
                "  {:12} | {:>6.3} | {:>8.3} | {:>8.3} | {:>8.3}",
                name,
                metrics.mrr,
                metrics.ndcg.get(&5).unwrap_or(&0.0),
                metrics.recall.get(&5).unwrap_or(&0.0),
                metrics.precision.get(&5).unwrap_or(&0.0),
            );
        }

        // Show top-3 hybrid results with scores
        println!("\n  Top-3 hybrid results:");
        for (i, result) in hybrid_results.iter().take(3).enumerate() {
            let preview = &result.chunk.content[..60.min(result.chunk.content.len())];
            println!(
                "    {}. [dense: {:?}, sparse: {:?}, fused: {:.3}]",
                i + 1,
                result.dense_score.map(|s| format!("{:.3}", s)),
                result.sparse_score.map(|s| format!("{:.3}", s)),
                result.best_score()
            );
            println!("       {}...", preview);
        }

        println!();
    }

    // Summary statistics
    println!("=== Summary ===\n");
    println!("HybridRetrieverConfig:");
    println!("  candidates_per_source: 10");
    println!("  fusion: RRF (k=60)");
    println!("  use_dense: true");
    println!("  use_sparse: true");
    println!();
    println!("Hybrid search combines BM25 term matching with dense vector");
    println!("similarity, using Reciprocal Rank Fusion to merge results.");
    println!("This captures both exact keyword matches (sparse) and");
    println!("semantic similarity (dense) for better retrieval quality.");

    Ok(())
}
