# Retrieval

Trueno-RAG uses hybrid retrieval combining dense and sparse search.

## Retrieval Types

### Dense Retrieval (Vector Search)

Uses embedding similarity for semantic matching:

```rust
let mut store = VectorStore::with_dimension(384);
store.insert(chunk)?;
let results = store.search(&query_embedding, 10)?;
```

### Sparse Retrieval (BM25)

Uses term-based matching:

```rust
let mut index = BM25Index::new();
index.add(&chunk);
let results = index.search("query terms", 10);
```

### Hybrid Retrieval

Combines both approaches:

```rust
let retriever = HybridRetriever::new(vector_store, bm25_index, embedder)
    .with_config(HybridRetrieverConfig {
        candidates_per_source: 50,
        fusion: FusionStrategy::RRF { k: 60.0 },
        use_dense: true,
        use_sparse: true,
    });

let results = retriever.retrieve("query", 10)?;
```

## Retrieval Results

Each result contains:

```rust
pub struct RetrievalResult {
    pub chunk: Chunk,
    pub dense_score: Option<f32>,
    pub sparse_score: Option<f32>,
    pub fused_score: Option<f32>,
    pub rerank_score: Option<f32>,
}

// Get best score
let score = result.best_score();
```

## BM25 Parameters

Tune BM25 for your use case:

```rust
pub struct BM25Config {
    pub k1: f32,  // Term frequency saturation (default: 1.2)
    pub b: f32,   // Document length normalization (default: 0.75)
}
```

## Performance Tips

1. **Pre-compute embeddings**: Batch embed documents during indexing
2. **Use appropriate k**: Retrieve 2-3x more results than needed before reranking
3. **Tune weights**: Adjust dense/sparse weights based on your domain
4. **Index structure**: Use appropriate index for your corpus size
