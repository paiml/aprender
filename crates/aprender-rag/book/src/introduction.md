# Trueno-RAG

**Pure-Rust Retrieval-Augmented Generation Pipeline**

Trueno-RAG is a high-performance, zero-dependency RAG implementation
built entirely in Rust. It provides a complete pipeline for document
ingestion, chunking, embedding, retrieval, and context assembly
for large language model applications.

## Features

- **Pure Rust**: Zero Python or C++ dependencies
- **SIMD-Accelerated**: Built on Trueno compute primitives for maximum performance
- **Flexible Chunking**: Multiple strategies including recursive, semantic, and structural
- **Hybrid Retrieval**: Combines dense (vector) and sparse (BM25) search
- **Multiple Fusion Strategies**: RRF, Linear, DBSF, and more
- **Reranking**: Lexical and cross-encoder reranking support
- **Comprehensive Testing**: Property-based testing with proptest

## Architecture

```
Document -> Chunker -> Embedder -> Index (Dense + Sparse)
                                        |
Query -> Preprocessor -> Retriever -> Fusion -> Reranker -> Context
```

## Quick Example

```rust
use trueno_rag::{
    pipeline::RagPipelineBuilder,
    chunk::RecursiveChunker,
    embed::MockEmbedder,
    rerank::LexicalReranker,
    fusion::FusionStrategy,
    Document,
};

// Build the pipeline
let mut pipeline = RagPipelineBuilder::new()
    .chunker(RecursiveChunker::new(512, 50))
    .embedder(MockEmbedder::new(384))
    .reranker(LexicalReranker::new())
    .fusion(FusionStrategy::RRF { k: 60.0 })
    .max_context_tokens(4096)
    .build()?;

// Index documents
let doc = Document::new("Your document content here...")
    .with_title("Document Title");
pipeline.index_document(&doc)?;

// Query with context assembly
let (results, context) = pipeline.query_with_context("your query", 5)?;
println!("{}", context.format_with_citations());
```

## Design Philosophy

Trueno-RAG follows the Toyota Way principles of continuous improvement and built-in quality:

1. **Jidoka (Built-in Quality)**: Extensive property-based testing ensures correctness
2. **Kaizen (Continuous Improvement)**: Modular design enables iterative enhancement
3. **Genchi Genbutsu (Go and See)**: Comprehensive metrics and evaluation tools
4. **Respect for People**: Clean, documented APIs for developer experience
