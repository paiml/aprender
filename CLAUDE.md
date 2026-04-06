# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Contract-First Design

This project follows contract-first development with provable-contracts.
Contracts live in `../provable-contracts/contracts/trueno-rag/`.
Run `pmat comply check` to validate contract compliance.

## Project Overview

Trueno-RAG is a pure-Rust implementation of Retrieval-Augmented Generation (RAG) pipelines built on Trueno compute primitives with zero Python/C++ dependencies. The full specification is in `docs/specifications/rag-pipeline-spec.md`.

## Code Search (pmat query)

**NEVER use grep or rg for code discovery.** Use `pmat query` instead -- it returns quality-annotated, ranked results with TDG scores and fault annotations.

```bash
# Find functions by intent
pmat query "document chunking" --limit 10

# Find high-quality code
pmat query "bm25 scoring" --min-grade A --exclude-tests

# Find with fault annotations (unwrap, panic, unsafe, etc.)
pmat query "embedding search" --faults

# Filter by complexity
pmat query "retrieval pipeline" --max-complexity 10

# Cross-project search
pmat query "vector store" --include-project ../trueno

# Git history search (find code by commit intent via RRF fusion)
pmat query "fix rrf fusion" -G
pmat query "fix rrf fusion" --git-history

# Enrichment flags (combine freely)
pmat query "indexer" --churn              # git volatility (commit count, churn score)
pmat query "scorer" --duplicates          # code clone detection (MinHash+LSH)
pmat query "retriever" --entropy             # pattern diversity (repetitive vs unique)
pmat query "rag pipeline" --churn --duplicates --entropy --faults -G  # full audit
```

## Build Commands

```bash
# Build
cargo build

# Run tests (261 tests with property-based testing via proptest)
cargo test

# Fast tests (release mode)
make test-fast

# Run single test
cargo test test_bm25_search

# Linting
cargo clippy -- -D warnings

# Format check
cargo fmt --check

# Run benchmarks
cargo bench

# Run examples
make examples
cargo run --example basic_rag
cargo run --example chunking_strategies
cargo run --example hybrid_search
cargo run --example metrics_evaluation

# Build documentation book
make book

# Check latest trueno version
cargo search trueno
```

## Architecture

### Module Structure

- `src/chunk.rs` - Document chunking (RecursiveChunker, FixedSizeChunker, SentenceChunker, ParagraphChunker, SemanticChunker, StructuralChunker)
- `src/embed.rs` - Embedding generation (MockEmbedder, TfIdfEmbedder, cosine similarity)
- `src/index.rs` - BM25 sparse index and VectorStore for dense retrieval
- `src/fusion.rs` - Score fusion strategies (RRF, Linear, Convex, DBSF, Union, Intersection)
- `src/retrieve.rs` - HybridRetriever, DenseRetriever, SparseRetriever
- `src/rerank.rs` - Reranking (LexicalReranker, MockCrossEncoderReranker, CompositeReranker)
- `src/pipeline.rs` - RagPipeline builder, ContextAssembler with citation tracking
- `src/preprocess.rs` - Query preprocessing (HyDE, multi-query expansion, synonyms)
- `src/metrics.rs` - Retrieval evaluation metrics (Recall, Precision, MRR, NDCG, MAP)
- `src/error.rs` - Error types

### Key Traits

- `Chunker` - `fn chunk(&self, document: &Document) -> Result<Vec<Chunk>>`
- `Embedder` - `fn embed(&self, text: &str) -> Result<Vec<f32>>`
- `SparseIndex` - `fn add(&mut self, chunk: &Chunk)`, `fn search(&self, query: &str, k: usize)`
- `Reranker` - `fn rerank(&self, query: &str, candidates: &[RetrievalResult], top_k: usize)`

### Pipeline Usage

```rust
use trueno_rag::{
    pipeline::RagPipelineBuilder,
    chunk::RecursiveChunker,
    embed::MockEmbedder,
    rerank::LexicalReranker,
    fusion::FusionStrategy,
};

let pipeline = RagPipelineBuilder::new()
    .chunker(RecursiveChunker::new(512, 50))
    .embedder(MockEmbedder::new(384))
    .reranker(LexicalReranker::new())
    .fusion(FusionStrategy::RRF { k: 60.0 })
    .max_context_tokens(4096)
    .build()?;

let (results, context) = pipeline.query_with_context("What is RAG?", 5)?;
println!("{}", context.format_with_citations());
```

### Data Flow

1. **Chunk** - `Document` → `Vec<Chunk>` via `Chunker`
2. **Embed** - `Chunk` → embedding via `Embedder::embed_chunks()`
3. **Index** - Add to `VectorStore` (dense) and `BM25Index` (sparse)
4. **Retrieve** - `HybridRetriever::retrieve()` returns fused results
5. **Rerank** - `Reranker::rerank()` refines ordering
6. **Assemble** - `ContextAssembler::assemble()` builds `AssembledContext` with citations

### Fusion Strategies

- `FusionStrategy::RRF { k: 60.0 }` - Reciprocal Rank Fusion (default)
- `FusionStrategy::Linear { dense_weight: 0.7 }` - Weighted linear combination
- `FusionStrategy::DBSF` - Distribution-Based Score Fusion (z-score normalization)
- `FusionStrategy::Intersection` - Only return results in both dense and sparse

## Dependencies

- `trueno` v0.7.3 - SIMD-accelerated vector operations
- `trueno-db` v0.3.3 - Vector storage
- `tokio` - Async runtime
- `serde/serde_json` - Serialization
- `proptest` (dev) - Property-based testing

## Git Workflow

Work directly on master branch. Do not create feature branches.


## Stack Documentation Search (RAG Oracle)

**IMPORTANT: Proactively use the batuta RAG oracle when:**
- Looking up patterns from other stack components
- Finding cross-language equivalents (Python HuggingFace → Rust)
- Understanding how similar problems are solved elsewhere in the stack

```bash
# Search across the entire Sovereign AI Stack
batuta oracle --rag "your question here"

# Reindex after changes (auto-runs via post-commit hook + ora-fresh)
batuta oracle --rag-index

# Check index freshness (runs automatically on shell login)
ora-fresh
```

The RAG index covers 5000+ documents across the Sovereign AI Stack.
Index auto-updates via post-commit hooks and `ora-fresh` on shell login.
To manually check freshness: `ora-fresh`
To force full reindex: `batuta oracle --rag-index --force`
