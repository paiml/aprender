# RAG Pipeline Overview

Retrieval-Augmented Generation (RAG) enhances LLM responses by retrieving relevant context from a document corpus.

## Pipeline Stages

### 1. Document Ingestion

Documents are loaded and prepared for processing:

```rust
let doc = Document::new("Your document content...")
    .with_title("Document Title")
    .with_source("source.pdf");
```

### 2. Chunking

Documents are split into smaller chunks for indexing:

```rust
let chunker = RecursiveChunker::new(512, 50);
let chunks = chunker.chunk(&document)?;
```

### 3. Embedding

Chunks are converted to vector representations:

```rust
let embedder = MockEmbedder::new(384);
embedder.embed_chunks(&mut chunks)?;
```

### 4. Indexing

Chunks are stored in both dense (vector) and sparse (BM25) indices:

```rust
pipeline.index_document(&doc)?;
```

### 5. Retrieval

Queries retrieve relevant chunks using hybrid search:

```rust
let results = pipeline.query("your query", 10)?;
```

### 6. Reranking

Results are reranked for better relevance:

```rust
let reranked = reranker.rerank(&query, &results, 5)?;
```

### 7. Context Assembly

Retrieved chunks are assembled into a coherent context:

```rust
let context = assembler.assemble(&results);
println!("{}", context.format_with_citations());
```

## Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `chunk_size` | 512 | Maximum chunk size in characters |
| `chunk_overlap` | 50 | Overlap between consecutive chunks |
| `embedding_dimension` | 384 | Vector embedding dimension |
| `max_context_tokens` | 4096 | Maximum tokens in assembled context |
| `fusion_strategy` | RRF | How to combine dense and sparse results |
