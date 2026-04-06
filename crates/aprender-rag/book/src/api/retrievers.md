# Retrievers API

## HybridRetriever

Combines dense and sparse retrieval:

```rust
pub struct HybridRetriever<E: Embedder> { ... }

impl<E: Embedder> HybridRetriever<E> {
    pub fn new(
        vector_store: VectorStore,
        sparse_index: BM25Index,
        embedder: E
    ) -> Self;
    pub fn with_config(self, config: HybridRetrieverConfig) -> Self;
    pub fn retrieve(&self, query: &str, k: usize) -> Result<Vec<RetrievalResult>>;
    pub fn retrieve_dense(&self, query: &str, k: usize) -> Result<Vec<RetrievalResult>>;
    pub fn retrieve_sparse(&self, query: &str, k: usize) -> Result<Vec<RetrievalResult>>;
    pub fn index(&mut self, chunk: Chunk) -> Result<()>;
    pub fn index_batch(&mut self, chunks: Vec<Chunk>) -> Result<()>;
    pub fn len(&self) -> usize;
}
```

## HybridRetrieverConfig

```rust
pub struct HybridRetrieverConfig {
    pub candidates_per_source: usize,
    pub fusion: FusionStrategy,
    pub use_dense: bool,
    pub use_sparse: bool,
}
```

## VectorStore

Dense vector index:

```rust
pub struct VectorStore { ... }

impl VectorStore {
    pub fn with_dimension(dimension: usize) -> Self;
    pub fn insert(&mut self, chunk: Chunk) -> Result<()>;
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<(ChunkId, f32)>>;
    pub fn len(&self) -> usize;
}
```

## BM25Index

Sparse term-based index:

```rust
pub struct BM25Index { ... }

impl BM25Index {
    pub fn new() -> Self;
    pub fn add(&mut self, chunk: &Chunk);
    pub fn search(&self, query: &str, k: usize) -> Vec<(ChunkId, f32)>;
    pub fn len(&self) -> usize;
}
```

## RetrievalResult

```rust
pub struct RetrievalResult {
    pub chunk: Chunk,
    pub dense_score: Option<f32>,
    pub sparse_score: Option<f32>,
    pub fused_score: Option<f32>,
    pub rerank_score: Option<f32>,
}

impl RetrievalResult {
    pub fn new(chunk: Chunk) -> Self;
    pub fn with_dense_score(self, score: f32) -> Self;
    pub fn with_sparse_score(self, score: f32) -> Self;
    pub fn with_fused_score(self, score: f32) -> Self;
    pub fn best_score(&self) -> f32;
}
```

## FusionStrategy

```rust
pub enum FusionStrategy {
    RRF { k: f32 },
    Linear { dense_weight: f32 },
    Convex { dense_weight: f32 },
    DBSF,
    Union,
    Intersection,
}
```
