# Performance Tuning

Optimize Trueno-RAG for your workload.

## Indexing Performance

### Batch Processing

Index documents in batches:

```rust
let chunk_count = pipeline.index_documents(&documents)?;
```

### Parallel Embedding

Use batch embedding when available:

```rust
let embeddings = embedder.embed_batch(&texts)?;
```

## Retrieval Performance

### Choose Appropriate k

Retrieve just enough candidates:

```rust
// Retrieve 2x what you need for reranking
let candidates = retriever.retrieve(&query, top_k * 2)?;
let results = reranker.rerank(&query, &candidates, top_k)?;
```

### Use Fast Fusion

RRF is faster than DBSF (no z-score normalization):

```rust
FusionStrategy::RRF { k: 60.0 }
```

### Skip Reranking for Speed

Use `NoOpReranker` when latency is critical:

```rust
.reranker(NoOpReranker::new())
```

## Memory Optimization

### Chunk Size Trade-offs

| Chunk Size | Memory | Quality |
|------------|--------|---------|
| Smaller (128-256) | Lower | More precise but fragmented |
| Medium (512) | Moderate | Balanced |
| Larger (1024+) | Higher | More context but coarser |

### Index Size

- Dense index: ~4 bytes * dimension * num_chunks
- Sparse index: Variable, depends on vocabulary

## Profiling

Use Rust profiling tools:

```bash
# Build with debug symbols
cargo build --release

# Profile with perf
perf record ./target/release/your_binary
perf report
```

## Benchmarks

Run the benchmark suite:

```bash
cargo bench
```

Example benchmark output:

```
index_1000_docs     time: [45.2 ms 46.1 ms 47.0 ms]
query_top_10        time: [1.23 ms 1.25 ms 1.28 ms]
rerank_50           time: [0.89 ms 0.91 ms 0.93 ms]
```
