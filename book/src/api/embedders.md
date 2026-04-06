# Embedders API

## Embedder Trait

```rust
pub trait Embedder: Send + Sync {
    fn embed(&self, text: &str) -> Result<Vec<f32>>;
    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>>;
    fn embed_chunks(&self, chunks: &mut [Chunk]) -> Result<()>;
    fn dimension(&self) -> usize;
}
```

## MockEmbedder

For testing and development:

```rust
pub struct MockEmbedder { ... }

impl MockEmbedder {
    pub fn new(dimension: usize) -> Self;
}
```

## TfIdfEmbedder

Simple TF-IDF based embeddings:

```rust
pub struct TfIdfEmbedder { ... }

impl TfIdfEmbedder {
    pub fn new(dimension: usize) -> Self;
    pub fn fit(&mut self, documents: &[&str]);
}
```

## Similarity Functions

```rust
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32;
pub fn euclidean_distance(a: &[f32], b: &[f32]) -> f32;
```

## PoolingStrategy

```rust
pub enum PoolingStrategy {
    Mean,
    CLS,
    Max,
}
```

## EmbeddingConfig

```rust
pub struct EmbeddingConfig {
    pub dimension: usize,
    pub normalize: bool,
    pub pooling: PoolingStrategy,
}
```
