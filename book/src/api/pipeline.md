# Pipeline Builder API

## RagPipelineBuilder

Builder pattern for constructing RAG pipelines.

```rust
pub struct RagPipelineBuilder<E: Embedder, R: Reranker> { ... }

impl<E: Embedder + Clone, R: Reranker> RagPipelineBuilder<E, R> {
    pub fn new() -> Self;
    pub fn chunker(self, chunker: impl Chunker + 'static) -> Self;
    pub fn embedder(self, embedder: E) -> Self;
    pub fn reranker(self, reranker: R) -> Self;
    pub fn fusion(self, fusion: FusionStrategy) -> Self;
    pub fn max_context_tokens(self, max_tokens: usize) -> Self;
    pub fn build(self) -> Result<RagPipeline<E, R>>;
}
```

## RagPipeline

The main pipeline struct.

```rust
pub struct RagPipeline<E: Embedder, R: Reranker> { ... }

impl<E: Embedder + Clone, R: Reranker> RagPipeline<E, R> {
    pub fn index_document(&mut self, doc: &Document) -> Result<Vec<Chunk>>;
    pub fn index_documents(&mut self, docs: &[Document]) -> Result<usize>;
    pub fn document_count(&self) -> usize;
    pub fn chunk_count(&self) -> usize;
    pub fn query(&self, query: &str, k: usize) -> Result<Vec<RetrievalResult>>;
    pub fn query_with_context(
        &self,
        query: &str,
        k: usize
    ) -> Result<(Vec<RetrievalResult>, AssembledContext)>;
}
```

## ContextAssembler

Assembles retrieved chunks into context.

```rust
pub struct ContextAssembler { ... }

impl ContextAssembler {
    pub fn new(config: ContextAssemblerConfig) -> Self;
    pub fn with_max_tokens(max_tokens: usize) -> Self;
    pub fn assemble(&self, results: &[RetrievalResult]) -> AssembledContext;
}
```

## AssembledContext

The assembled context result.

```rust
pub struct AssembledContext {
    pub chunks: Vec<ContextChunk>,
    pub total_tokens: usize,
    pub citations: Vec<Citation>,
}

impl AssembledContext {
    pub fn format_with_citations(&self) -> String;
    pub fn format_plain(&self) -> String;
    pub fn citation_list(&self) -> String;
    pub fn len(&self) -> usize;
    pub fn is_empty(&self) -> bool;
}
```
