# Eval Hybrid

Compare hybrid, dense, and sparse retrieval modes with metrics evaluation.

Demonstrates Phase 2 features: `HybridRetriever`, `HybridRetrieverConfig`, `RetrievalMetrics`, and the `retrieve()` / `retrieve_dense()` / `retrieve_sparse()` methods.

```bash
cargo run --example eval_hybrid
```

## Source

```rust
use std::collections::HashSet;
use trueno_rag::{
    embed::MockEmbedder, fusion::FusionStrategy, metrics::RetrievalMetrics,
    retrieve::HybridRetrieverConfig, BM25Index, Chunk, ChunkId, DocumentId,
    HybridRetriever, VectorStore,
};

fn main() -> trueno_rag::Result<()> {
    let embedder = MockEmbedder::new(384);
    let store = VectorStore::with_dimension(384);
    let bm25 = BM25Index::new();

    let config = HybridRetrieverConfig {
        candidates_per_source: 10,
        fusion: FusionStrategy::RRF { k: 60.0 },
        use_dense: true,
        use_sparse: true,
    };

    let mut retriever = HybridRetriever::new(store, bm25, embedder)
        .with_config(config);

    // Index documents
    let documents = vec![
        "Rust provides memory safety through ownership...",
        "AWS Lambda runs code in response to events...",
        // ...
    ];

    let mut chunk_ids = Vec::new();
    for doc in &documents {
        let chunk = Chunk::new(DocumentId::new(), doc.to_string(), 0, doc.len());
        chunk_ids.push(chunk.id);
        retriever.index(chunk)?;
    }

    // Compare retrieval modes
    let query = "memory safety in systems programming";
    let relevant: HashSet<ChunkId> = vec![chunk_ids[0], chunk_ids[5]].into_iter().collect();

    let hybrid = retriever.retrieve(query, 5)?;
    let dense = retriever.retrieve_dense(query, 5)?;
    let sparse = retriever.retrieve_sparse(query, 5)?;

    // Compute metrics for each mode
    let k_values = vec![1, 3, 5];
    for (name, results) in [("Hybrid", hybrid), ("Dense", dense), ("Sparse", sparse)] {
        let ids: Vec<ChunkId> = results.iter().map(|r| r.chunk.id).collect();
        let metrics = RetrievalMetrics::compute(&ids, &relevant, &k_values);
        println!("{}: MRR={:.3}, NDCG@5={:.3}", name, metrics.mrr,
            metrics.ndcg.get(&5).unwrap_or(&0.0));
    }

    Ok(())
}
```

## Expected Output

```
=== Hybrid vs Dense vs Sparse Retrieval Comparison ===

Indexed 10 documents into hybrid retriever

Query: "memory safety in systems programming"

  Mode         |    MRR |   NDCG@5 | Recall@5 |  Prec@5
  -------------------------------------------------------
  Hybrid/RRF   |  1.000 |    0.773 |    1.000 |    0.400
  Dense        |  1.000 |    0.773 |    1.000 |    0.400
  Sparse/BM25  |  0.500 |    0.631 |    1.000 |    0.400

  Top-3 hybrid results:
    1. [dense: Some("0.823"), sparse: Some("4.231"), fused: 0.033]
       Rust provides memory safety guarantees through its ownership...
    2. [dense: Some("0.654"), sparse: Some("2.891"), fused: 0.032]
       The Rust compiler enforces strict lifetime rules that prevent...
    3. [dense: Some("0.412"), sparse: None, fused: 0.016]
       Go provides built-in concurrency primitives like goroutines...
```

## Key Points

- `HybridRetriever` combines `VectorStore` (dense) and `BM25Index` (sparse)
- `retrieve()` fuses both result lists using `FusionStrategy::RRF`
- `retrieve_dense()` and `retrieve_sparse()` isolate individual modes
- `RetrievalMetrics::compute()` calculates MRR, NDCG, Recall, Precision at arbitrary k values
