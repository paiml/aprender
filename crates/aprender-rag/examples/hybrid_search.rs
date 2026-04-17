//! Hybrid Search Example
//!
//! Run with: cargo run --example hybrid_search

use trueno_rag::{
    embed::MockEmbedder, fusion::FusionStrategy, pipeline::RagPipelineBuilder,
    rerank::NoOpReranker, Document,
};

fn main() -> trueno_rag::Result<()> {
    println!("=== Hybrid Search with Different Fusion Strategies ===\n");

    let documents = vec![
        Document::new("Rust programming language provides memory safety guarantees without garbage collection."),
        Document::new("Python is excellent for data science and machine learning applications."),
        Document::new("Go provides fast compilation and built-in concurrency primitives."),
        Document::new("Memory management in systems programming is crucial for performance."),
        Document::new("The Rust compiler enforces strict borrowing rules at compile time."),
    ];

    let query = "memory safe programming language";
    println!("Query: \"{}\"\n", query);

    let strategies = vec![
        ("RRF (k=60)", FusionStrategy::RRF { k: 60.0 }),
        ("Linear (dense=0.7)", FusionStrategy::Linear { dense_weight: 0.7 }),
        ("Linear (dense=0.3)", FusionStrategy::Linear { dense_weight: 0.3 }),
        ("DBSF", FusionStrategy::DBSF),
        ("Union", FusionStrategy::Union),
    ];

    for (name, strategy) in strategies {
        println!("--- {} ---", name);

        let mut pipeline = RagPipelineBuilder::new()
            .embedder(MockEmbedder::new(384))
            .reranker(NoOpReranker::new())
            .fusion(strategy)
            .build()?;

        pipeline.index_documents(&documents)?;

        let results = pipeline.query(query, 3)?;

        for (i, result) in results.iter().enumerate() {
            let preview = &result.chunk.content[..50.min(result.chunk.content.len())];
            println!(
                "  {}. [dense: {:?}, sparse: {:?}, fused: {:.3}]",
                i + 1,
                result.dense_score.map(|s| format!("{:.3}", s)),
                result.sparse_score.map(|s| format!("{:.3}", s)),
                result.best_score()
            );
            println!("     {}...", preview);
        }
        println!();
    }

    Ok(())
}
