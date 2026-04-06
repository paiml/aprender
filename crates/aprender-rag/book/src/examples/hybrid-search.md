# Hybrid Search

Combine dense and sparse retrieval for better results.

## Basic Hybrid Search

```rust
use trueno_rag::{
    pipeline::RagPipelineBuilder,
    embed::MockEmbedder,
    rerank::LexicalReranker,
    fusion::FusionStrategy,
    Document,
};

fn main() -> trueno_rag::Result<()> {
    // Use RRF fusion (default and most robust)
    let mut pipeline = RagPipelineBuilder::new()
        .embedder(MockEmbedder::new(384))
        .reranker(LexicalReranker::new())
        .fusion(FusionStrategy::RRF { k: 60.0 })
        .build()?;

    // Index documents
    pipeline.index_document(&Document::new(
        "Rust programming language provides memory safety guarantees."
    ))?;

    // Query uses both dense (semantic) and sparse (keyword) search
    let results = pipeline.query("memory safe programming", 5)?;

    for result in &results {
        println!("Dense: {:?}, Sparse: {:?}, Fused: {:?}",
                 result.dense_score,
                 result.sparse_score,
                 result.fused_score);
    }

    Ok(())
}
```

## Comparing Fusion Strategies

```rust
let test_queries = vec![
    "exact keyword match",        // Sparse should help
    "semantic similarity search", // Dense should help
    "mixed query with keywords",  // Both should help
];

let strategies = vec![
    ("RRF", FusionStrategy::RRF { k: 60.0 }),
    ("Linear (0.7)", FusionStrategy::Linear { dense_weight: 0.7 }),
    ("Linear (0.3)", FusionStrategy::Linear { dense_weight: 0.3 }),
    ("DBSF", FusionStrategy::DBSF),
];

for (name, strategy) in strategies {
    let mut pipeline = RagPipelineBuilder::new()
        .embedder(MockEmbedder::new(384))
        .reranker(NoOpReranker::new())
        .fusion(strategy)
        .build()?;

    // Index and query...
    println!("{}: top result score = {:.3}", name, top_score);
}
```

## Tuning Dense vs Sparse Weight

```rust
// Favor dense (semantic) search
let semantic_heavy = FusionStrategy::Linear { dense_weight: 0.8 };

// Favor sparse (keyword) search
let keyword_heavy = FusionStrategy::Linear { dense_weight: 0.2 };

// Balanced
let balanced = FusionStrategy::Linear { dense_weight: 0.5 };
```

## When to Favor Each Approach

| Scenario | Recommendation |
|----------|----------------|
| Technical docs with specific terms | Higher sparse weight |
| Conversational queries | Higher dense weight |
| Domain with synonyms | Higher dense weight |
| Exact phrase matching needed | Higher sparse weight |
| Unknown query types | Use RRF |

## Evaluating Hybrid vs Single Retriever

```rust
use trueno_rag::metrics::RetrievalMetrics;

// Get results from each approach
let dense_only = dense_retriever.retrieve(&query, k)?;
let sparse_only = sparse_retriever.retrieve(&query, k)?;
let hybrid = hybrid_retriever.retrieve(&query, k)?;

// Compute metrics for each
let dense_metrics = RetrievalMetrics::compute(
    &dense_only.iter().map(|r| r.chunk.id).collect::<Vec<_>>(),
    &relevant,
    &[1, 5, 10]
);

// Compare MRR and recall
println!("Dense MRR: {:.3}", dense_metrics.mrr);
println!("Sparse MRR: {:.3}", sparse_metrics.mrr);
println!("Hybrid MRR: {:.3}", hybrid_metrics.mrr);
```
