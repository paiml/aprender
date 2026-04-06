# Chunkers API

## Chunker Trait

```rust
pub trait Chunker: Send + Sync {
    fn chunk(&self, document: &Document) -> Result<Vec<Chunk>>;
    fn estimate_chunks(&self, document: &Document) -> usize;
}
```

## RecursiveChunker

```rust
pub struct RecursiveChunker { ... }

impl RecursiveChunker {
    pub fn new(chunk_size: usize, overlap: usize) -> Self;
    pub fn with_separators(self, separators: Vec<String>) -> Self;
}
```

## FixedSizeChunker

```rust
pub struct FixedSizeChunker { ... }

impl FixedSizeChunker {
    pub fn new(chunk_size: usize, overlap: usize) -> Self;
}
```

## SentenceChunker

```rust
pub struct SentenceChunker { ... }

impl SentenceChunker {
    pub fn new(max_sentences: usize, overlap_sentences: usize) -> Self;
}
```

## ParagraphChunker

```rust
pub struct ParagraphChunker { ... }

impl ParagraphChunker {
    pub fn new(max_paragraphs: usize) -> Self;
}
```

## SemanticChunker

```rust
pub struct SemanticChunker<E: Embedder> { ... }

impl<E: Embedder> SemanticChunker<E> {
    pub fn new(
        embedder: E,
        similarity_threshold: f32,
        max_chunk_size: usize
    ) -> Self;
}
```

## StructuralChunker

```rust
pub struct StructuralChunker { ... }

impl StructuralChunker {
    pub fn new(respect_headers: bool, max_section_size: usize) -> Self;
}
```

## Chunk

```rust
pub struct Chunk {
    pub id: ChunkId,
    pub document_id: DocumentId,
    pub content: String,
    pub embedding: Option<Vec<f32>>,
    pub start_offset: usize,
    pub end_offset: usize,
    pub metadata: ChunkMetadata,
}

impl Chunk {
    pub fn new(
        document_id: DocumentId,
        content: String,
        start_offset: usize,
        end_offset: usize
    ) -> Self;
    pub fn len(&self) -> usize;
    pub fn is_empty(&self) -> bool;
    pub fn set_embedding(&mut self, embedding: Vec<f32>);
}
```
