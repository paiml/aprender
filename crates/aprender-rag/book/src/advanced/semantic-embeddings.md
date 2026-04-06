# Semantic Embeddings (GH-1)

Trueno-RAG provides production-quality semantic embeddings via
FastEmbed (ONNX Runtime), enabling real vector similarity search
instead of mock embeddings.

## Quick Start

```rust
use trueno_rag::{
    pipeline::RagPipelineBuilder,
    chunk::RecursiveChunker,
    rerank::LexicalReranker,
    fusion::FusionStrategy,
    FastEmbedder, EmbeddingModelType, Document,
};

// Create embedder with MiniLM model (384 dimensions)
let embedder = FastEmbedder::new(EmbeddingModelType::AllMiniLmL6V2)?;

let mut pipeline = RagPipelineBuilder::new()
    .chunker(RecursiveChunker::new(256, 32))
    .embedder(embedder)
    .reranker(LexicalReranker::new())
    .fusion(FusionStrategy::RRF { k: 60.0 })
    .build()?;

// Index documents with semantic embeddings
let doc = Document::new("RAG combines LLMs with knowledge retrieval...")
    .with_title("RAG Overview");
pipeline.index_document(&doc)?;

// Query with semantic understanding
let results = pipeline.query("How do AI systems access external knowledge?", 5)?;
```

## Available Models

| Model | Dimensions | Size | Speed | Use Case |
|-------|------------|------|-------|----------|
| `AllMiniLmL6V2` | 384 | ~90MB | Fast | General purpose (default) |
| `AllMiniLmL12V2` | 384 | ~120MB | Medium | Better quality |
| `BgeSmallEnV15` | 384 | ~130MB | Fast | MTEB benchmark leader |
| `BgeBaseEnV15` | 768 | ~440MB | Slower | Higher quality |
| `NomicEmbedTextV1` | 768 | ~550MB | Slower | Long context (8192 tokens) |

### Model Selection

```rust
use trueno_rag::{FastEmbedder, EmbeddingModelType};

// Fast general-purpose model (recommended for most use cases)
let embedder = FastEmbedder::new(EmbeddingModelType::AllMiniLmL6V2)?;

// Higher quality at the cost of speed
let embedder = FastEmbedder::new(EmbeddingModelType::BgeBaseEnV15)?;

// Long context support (8192 tokens)
let embedder = FastEmbedder::new(EmbeddingModelType::NomicEmbedTextV1)?;
```

## Semantic vs Lexical Search

### Lexical (BM25)

- Matches exact keywords
- "cat" won't match "feline"
- Fast, no model required
- Good for exact term lookup

### Semantic (Vector)

- Matches meaning
- "cat" matches "feline"
- Requires embedding model
- Good for natural language queries

### Hybrid (Recommended)

Trueno-RAG's hybrid retrieval combines both:

```rust
// Hybrid search with RRF fusion
let results = pipeline.query("How do cats communicate?", 5)?;
// Finds docs about "feline vocalizations" AND "cat sounds"
```

## Embedding Similarity

```rust
use trueno_rag::{FastEmbedder, EmbeddingModelType, embed::Embedder};

let embedder = FastEmbedder::new(EmbeddingModelType::AllMiniLmL6V2)?;

let emb1 = embedder.embed("The cat sat on the mat.")?;
let emb2 = embedder.embed("A feline rested on the rug.")?;
let emb3 = embedder.embed("The stock market crashed today.")?;

// emb1 vs emb2: ~0.85 (semantically similar)
// emb1 vs emb3: ~0.15 (semantically different)
```

## CLI Usage

```bash
# Index with semantic embeddings
trueno-rag index ./docs --embedder semantic --model mini-lm-l6

# Available models via CLI
trueno-rag info

# Query (auto-detects embedder type from index)
trueno-rag query "How does RAG work?"
```

## Batch Embedding

For performance, use batch embedding:

```rust
use trueno_rag::{FastEmbedder, EmbeddingModelType, embed::Embedder};

let embedder = FastEmbedder::new(EmbeddingModelType::AllMiniLmL6V2)?;

let texts = vec![
    "First document text",
    "Second document text",
    "Third document text",
];

// Batch embed is more efficient than embedding one at a time
let embeddings = embedder.embed_batch(&texts)?;
```

## API Reference

### FastEmbedder

```rust
impl FastEmbedder {
    /// Create a new FastEmbedder with the specified model
    pub fn new(model_type: EmbeddingModelType) -> Result<Self>;

    /// Get the model identifier
    pub fn model_id(&self) -> &str;
}

impl Embedder for FastEmbedder {
    /// Embed a single text
    fn embed(&self, text: &str) -> Result<Vec<f32>>;

    /// Batch embed multiple texts (more efficient)
    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>>;

    /// Get embedding dimension
    fn dimension(&self) -> usize;

    /// Get model identifier
    fn model_id(&self) -> &str;
}
```

### EmbeddingModelType

```rust
pub enum EmbeddingModelType {
    AllMiniLmL6V2,    // 384 dims, fast
    AllMiniLmL12V2,   // 384 dims, better quality
    BgeSmallEnV15,    // 384 dims, MTEB leader
    BgeBaseEnV15,     // 768 dims, high quality
    NomicEmbedTextV1, // 768 dims, long context
}

impl EmbeddingModelType {
    pub const fn dimension(&self) -> usize;
    pub const fn model_name(&self) -> &'static str;
}
```

## Feature Flag

Semantic embeddings require the `embeddings` feature:

```toml
[dependencies]
trueno-rag = { version = "0.1.8", features = ["embeddings"] }
```

This adds the FastEmbed dependency (~90MB model download on first run).

## Running the Example

```bash
cargo run --example semantic_embeddings --features embeddings
```

## Best Practices

1. **Start with MiniLM-L6**: Fast, good quality, 384 dimensions
2. **Use batch embedding**: 5-10x faster than individual calls
3. **Hybrid retrieval**: Combine semantic + BM25 for best results
4. **Cache embeddings**: Store with your index for fast restarts
5. **Match dimensions**: Ensure query embeddings match index dimensions

```rust
// Production pattern: hybrid retrieval with semantic embeddings
let embedder = FastEmbedder::new(EmbeddingModelType::AllMiniLmL6V2)?;

let mut pipeline = RagPipelineBuilder::new()
    .chunker(RecursiveChunker::new(256, 32))
    .embedder(embedder)
    .reranker(LexicalReranker::new())
    .fusion(FusionStrategy::RRF { k: 60.0 })  // Hybrid fusion
    .build()?;
```
