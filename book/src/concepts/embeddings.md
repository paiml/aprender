# Embeddings

Embeddings convert text into dense vector representations for semantic search.

## Embedder Trait

All embedders implement the `Embedder` trait:

```rust
pub trait Embedder: Send + Sync {
    fn embed(&self, text: &str) -> Result<Vec<f32>>;
    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>>;
    fn embed_chunks(&self, chunks: &mut [Chunk]) -> Result<()>;
    fn dimension(&self) -> usize;
}
```

## Available Embedders

### MockEmbedder

For testing and development:

```rust
let embedder = MockEmbedder::new(384);
```

### TfIdfEmbedder

Simple TF-IDF based embeddings:

```rust
let mut embedder = TfIdfEmbedder::new(128);
embedder.fit(&["doc1 content", "doc2 content"]);
```

## Similarity Functions

### Cosine Similarity

```rust
use trueno_rag::embed::cosine_similarity;

let sim = cosine_similarity(&vec1, &vec2);
// Returns value between -1.0 and 1.0
```

### Euclidean Distance

```rust
use trueno_rag::embed::euclidean_distance;

let dist = euclidean_distance(&vec1, &vec2);
// Returns non-negative distance
```

## Pooling Strategies

When embedding sequences:

```rust
pub enum PoolingStrategy {
    Mean,    // Average all token embeddings
    CLS,     // Use CLS token (first token)
    Max,     // Max pooling
}
```

## Custom Embedders

Implement the `Embedder` trait for custom models:

```rust
struct MyEmbedder {
    model: MyModel,
}

impl Embedder for MyEmbedder {
    fn embed(&self, text: &str) -> Result<Vec<f32>> {
        Ok(self.model.encode(text))
    }

    fn dimension(&self) -> usize {
        768
    }

    // ... other methods
}
```
