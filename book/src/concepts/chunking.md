# Document Chunking

Chunking splits documents into smaller pieces for effective retrieval.

## Chunking Strategies

### RecursiveChunker (Default)

Splits text hierarchically using multiple separators:

```rust
let chunker = RecursiveChunker::new(512, 50)
    .with_separators(vec!["\n\n", "\n", ". ", " "]);
```

Best for: General-purpose chunking

### FixedSizeChunker

Splits at exact character boundaries:

```rust
let chunker = FixedSizeChunker::new(256, 32);
```

Best for: Uniform chunk sizes

### SentenceChunker

Groups sentences together:

```rust
let chunker = SentenceChunker::new(3, 1); // 3 sentences, 1 overlap
```

Best for: Preserving sentence boundaries

### ParagraphChunker

Groups paragraphs together:

```rust
let chunker = ParagraphChunker::new(2); // 2 paragraphs per chunk
```

Best for: Paragraph-level retrieval

### SemanticChunker

Splits based on embedding similarity:

```rust
let chunker = SemanticChunker::new(embedder, 0.5, 1000);
```

Best for: Semantically coherent chunks

### StructuralChunker

Respects document structure (headers, sections):

```rust
let chunker = StructuralChunker::new(true, 500);
```

Best for: Markdown and structured documents

## Choosing a Chunking Strategy

| Use Case | Recommended Strategy |
|----------|---------------------|
| General text | RecursiveChunker |
| Technical docs | StructuralChunker |
| Legal documents | SentenceChunker |
| Short passages | FixedSizeChunker |
| Topic clustering | SemanticChunker |

## Chunk Metadata

Each chunk includes metadata:

```rust
chunk.metadata.title      // Document title
chunk.metadata.headers    // Section headers
chunk.metadata.page       // Page number (if applicable)
chunk.start_offset        // Position in original document
chunk.end_offset          // End position
```
