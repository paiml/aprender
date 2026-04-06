# Semantic Chunking

Semantic chunking groups text by meaning rather than structure.

## How It Works

1. Split text into sentences
2. Embed each sentence
3. Compare adjacent sentence embeddings
4. Split when similarity drops below threshold

## Usage

```rust
use trueno_rag::chunk::SemanticChunker;
use trueno_rag::embed::MockEmbedder;

let embedder = MockEmbedder::new(384);
let chunker = SemanticChunker::new(
    embedder,
    0.5,   // similarity threshold
    1000,  // max chunk size in chars
);

let chunks = chunker.chunk(&document)?;
```

## Parameters

| Parameter | Description | Typical Range |
|-----------|-------------|---------------|
| `similarity_threshold` | Split when similarity drops below this | 0.3 - 0.7 |
| `max_chunk_size` | Maximum characters per chunk | 500 - 2000 |

## When to Use

Good for:
- Documents with topic shifts
- Content where meaning matters more than structure
- Clustering related information

Avoid when:
- Documents have clear structural boundaries
- Speed is critical (embedding is expensive)
- Documents are already well-organized

## Performance Tips

1. Use a fast embedder for sentence embedding
2. Cache sentence embeddings when reprocessing
3. Consider pre-splitting by paragraphs first
