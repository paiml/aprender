# Semantic Embeddings

Production-quality semantic search using FastEmbed with ONNX Runtime models.

Requires the `embeddings` feature. First run downloads ~90MB model.

```bash
cargo run --example semantic_embeddings --features embeddings
```

## Source

```rust
#[cfg(feature = "embeddings")]
use trueno_rag::{
    chunk::RecursiveChunker, embed::Embedder, fusion::FusionStrategy,
    pipeline::RagPipelineBuilder, rerank::LexicalReranker, Document,
    EmbeddingModelType, FastEmbedder,
};

fn main() -> trueno_rag::Result<()> {
    let embedder = FastEmbedder::new(EmbeddingModelType::AllMiniLmL6V2)?;
    println!("Model: {} (dim: {})", embedder.model_id(), embedder.dimension());

    let mut pipeline = RagPipelineBuilder::new()
        .chunker(RecursiveChunker::new(256, 32))
        .embedder(embedder)
        .reranker(LexicalReranker::new())
        .fusion(FusionStrategy::RRF { k: 60.0 })
        .build()?;

    let documents = vec![
        Document::new("RAG combines retrieval with generation...").with_title("RAG Overview"),
        Document::new("Vector databases store embeddings...").with_title("Vector Databases"),
        Document::new("Sentence transformers produce meaningful embeddings...")
            .with_title("Sentence Transformers"),
    ];

    pipeline.index_documents(&documents)?;

    let results = pipeline.query("How do AI systems access external knowledge?", 2)?;
    for (i, result) in results.iter().enumerate() {
        let title = result.chunk.metadata.title.as_deref().unwrap_or("Untitled");
        println!("{}. [Score: {:.3}] {}", i + 1, result.best_score(), title);
    }

    Ok(())
}
```

## Available Models

| Model | Enum | Dimension | Notes |
|-------|------|-----------|-------|
| all-MiniLM-L6-v2 | `AllMiniLmL6V2` | 384 | Fast, good quality (default) |
| BGE-small-en-v1.5 | `BgeSmallEnV15` | 384 | Balanced performance |
| BGE-base-en-v1.5 | `BgeBaseEnV15` | 768 | Higher quality, larger |

## Expected Output

```
=== Semantic Embeddings Example ===

Loading embedding model (first run downloads ~90MB)...
Model: sentence-transformers/all-MiniLM-L6-v2 (dimension: 384)

Indexing 5 documents...
Created 5 chunks with semantic embeddings

Query: "How do AI systems access external knowledge?"

  1. [Score: 0.712] RAG Overview
     Retrieval-Augmented Generation (RAG) combines the power of large language...

  2. [Score: 0.534] Vector Databases
     Vector databases store high-dimensional embeddings and enable fast...
```
