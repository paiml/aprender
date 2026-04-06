//! RAG Pipeline implementation with context assembly

use crate::{
    chunk::{Chunk, Chunker, RecursiveChunker},
    embed::{Embedder, MockEmbedder},
    fusion::FusionStrategy,
    index::{BM25Index, VectorStore},
    rerank::{NoOpReranker, Reranker},
    retrieve::{HybridRetriever, HybridRetrieverConfig, RetrievalResult},
    Document, DocumentId, Error, Result,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Default embedding dimension (all-MiniLM-L6-v2 / BGE-small-en-v1.5)
const DEFAULT_EMBEDDING_DIM: usize = 384;

/// Citation for a retrieved chunk
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Citation {
    /// Citation ID (1-indexed for display)
    pub id: usize,
    /// Source document ID
    pub document_id: DocumentId,
    /// Source chunk ID
    pub chunk_id: crate::ChunkId,
    /// Document title (if available)
    pub title: Option<String>,
    /// Source URL (if available)
    pub url: Option<String>,
    /// Page number (if available)
    pub page: Option<usize>,
}

/// A chunk in the assembled context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextChunk {
    /// The chunk content
    pub content: String,
    /// Citation ID
    pub citation_id: usize,
    /// Retrieval score
    pub retrieval_score: f32,
    /// Rerank score (if available)
    pub rerank_score: Option<f32>,
}

/// Assembled context from retrieval results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssembledContext {
    /// Ordered chunks in context
    pub chunks: Vec<ContextChunk>,
    /// Total token count (estimated)
    pub total_tokens: usize,
    /// Source citations
    pub citations: Vec<Citation>,
}

impl AssembledContext {
    /// Create a new empty context
    #[must_use]
    pub fn new() -> Self {
        Self { chunks: Vec::new(), total_tokens: 0, citations: Vec::new() }
    }

    /// Add a chunk to the context
    pub fn add_chunk(&mut self, result: &RetrievalResult, citation_id: usize) {
        let chunk = ContextChunk {
            content: result.chunk.content.clone(),
            citation_id,
            retrieval_score: result.best_score(),
            rerank_score: result.rerank_score,
        };

        // Estimate tokens (rough: ~4 chars per token for English)
        self.total_tokens += result.chunk.content.len() / 4;
        self.chunks.push(chunk);
    }

    /// Add a citation
    pub fn add_citation(&mut self, result: &RetrievalResult) -> usize {
        let id = self.citations.len() + 1;

        let citation = Citation {
            id,
            document_id: result.chunk.document_id,
            chunk_id: result.chunk.id,
            title: result.chunk.metadata.title.clone(),
            url: None, // Would come from document metadata
            page: result.chunk.metadata.page,
        };

        self.citations.push(citation);
        id
    }

    /// Format context with inline citations
    #[must_use]
    pub fn format_with_citations(&self) -> String {
        self.chunks
            .iter()
            .map(|c| format!("{} [{}]", c.content, c.citation_id))
            .collect::<Vec<_>>()
            .join("\n\n")
    }

    /// Format context without citations
    #[must_use]
    pub fn format_plain(&self) -> String {
        self.chunks.iter().map(|c| c.content.as_str()).collect::<Vec<_>>().join("\n\n")
    }

    /// Generate citation list
    #[must_use]
    pub fn citation_list(&self) -> String {
        self.citations
            .iter()
            .map(|c| {
                let title = c.title.as_deref().unwrap_or("Untitled");
                format!("[{}] {}", c.id, title)
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Get the number of chunks
    #[must_use]
    pub fn len(&self) -> usize {
        self.chunks.len()
    }

    /// Check if the context is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.chunks.is_empty()
    }
}

impl Default for AssembledContext {
    fn default() -> Self {
        Self::new()
    }
}

/// Strategy for assembling context from retrieval results
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub enum AssemblyStrategy {
    /// Simple concatenation in rank order
    #[default]
    Sequential,
    /// Group by document, then by rank
    DocumentGrouped,
    /// Interleave chunks for diversity
    Interleaved,
}

/// Context assembler configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextAssemblerConfig {
    /// Maximum context length in tokens (estimated)
    pub max_tokens: usize,
    /// Assembly strategy
    pub strategy: AssemblyStrategy,
    /// Include citations
    pub include_citations: bool,
}

impl Default for ContextAssemblerConfig {
    fn default() -> Self {
        Self { max_tokens: 4096, strategy: AssemblyStrategy::Sequential, include_citations: true }
    }
}

/// Assembles retrieved chunks into a coherent context
#[derive(Debug, Clone)]
pub struct ContextAssembler {
    config: ContextAssemblerConfig,
}

impl ContextAssembler {
    /// Create a new context assembler
    #[must_use]
    pub fn new(config: ContextAssemblerConfig) -> Self {
        Self { config }
    }

    /// Create with default configuration
    #[must_use]
    pub fn with_max_tokens(max_tokens: usize) -> Self {
        Self::new(ContextAssemblerConfig { max_tokens, ..Default::default() })
    }

    /// Assemble context from retrieval results
    #[must_use]
    pub fn assemble(&self, results: &[RetrievalResult]) -> AssembledContext {
        match self.config.strategy {
            AssemblyStrategy::Sequential => self.assemble_sequential(results),
            AssemblyStrategy::DocumentGrouped => self.assemble_grouped(results),
            AssemblyStrategy::Interleaved => self.assemble_interleaved(results),
        }
    }

    fn assemble_sequential(&self, results: &[RetrievalResult]) -> AssembledContext {
        let mut context = AssembledContext::new();
        let mut remaining_tokens = self.config.max_tokens;

        for result in results {
            let chunk_tokens = result.chunk.content.len() / 4; // Rough estimate

            if chunk_tokens > remaining_tokens {
                // Could truncate, but for now we just stop
                break;
            }

            let citation_id =
                if self.config.include_citations { context.add_citation(result) } else { 0 };

            context.add_chunk(result, citation_id);
            remaining_tokens = remaining_tokens.saturating_sub(chunk_tokens);
        }

        context
    }

    fn assemble_grouped(&self, results: &[RetrievalResult]) -> AssembledContext {
        // Group by document
        let mut by_doc: HashMap<DocumentId, Vec<&RetrievalResult>> = HashMap::new();
        for result in results {
            by_doc.entry(result.chunk.document_id).or_default().push(result);
        }

        // Flatten while respecting order within documents
        let mut context = AssembledContext::new();
        let mut remaining_tokens = self.config.max_tokens;

        for (_, doc_results) in by_doc {
            for result in doc_results {
                let chunk_tokens = result.chunk.content.len() / 4;

                if chunk_tokens > remaining_tokens {
                    break;
                }

                let citation_id =
                    if self.config.include_citations { context.add_citation(result) } else { 0 };

                context.add_chunk(result, citation_id);
                remaining_tokens = remaining_tokens.saturating_sub(chunk_tokens);
            }
        }

        context
    }

    fn assemble_interleaved(&self, results: &[RetrievalResult]) -> AssembledContext {
        // For now, same as sequential but could implement round-robin from different docs
        self.assemble_sequential(results)
    }
}

impl Default for ContextAssembler {
    fn default() -> Self {
        Self::new(ContextAssemblerConfig::default())
    }
}

/// RAG Pipeline configuration
#[derive(Debug, Clone)]
pub struct RagPipelineConfig {
    /// Chunking chunk size
    pub chunk_size: usize,
    /// Chunking overlap
    pub chunk_overlap: usize,
    /// Embedding dimension
    pub embedding_dimension: usize,
    /// Retrieval config
    pub retrieval: HybridRetrieverConfig,
    /// Context assembly config
    pub context: ContextAssemblerConfig,
}

impl Default for RagPipelineConfig {
    fn default() -> Self {
        Self {
            chunk_size: 512,
            chunk_overlap: 50,
            embedding_dimension: DEFAULT_EMBEDDING_DIM,
            retrieval: HybridRetrieverConfig::default(),
            context: ContextAssemblerConfig::default(),
        }
    }
}

/// Complete RAG pipeline
pub struct RagPipeline<E: Embedder, R: Reranker> {
    /// Document chunker
    chunker: Box<dyn Chunker>,
    /// Embedder
    embedder: E,
    /// Hybrid retriever
    retriever: HybridRetriever<E>,
    /// Reranker
    reranker: R,
    /// Context assembler
    assembler: ContextAssembler,
    /// Indexed document count
    document_count: usize,
}

impl<E: Embedder + Clone, R: Reranker> RagPipeline<E, R> {
    /// Index a single document
    pub fn index_document(&mut self, document: &Document) -> Result<Vec<Chunk>> {
        // Chunk the document
        let mut chunks = self.chunker.chunk(document)?;

        // Embed the chunks
        self.embedder.embed_chunks(&mut chunks)?;

        // Add to retriever (both dense and sparse indices)
        for chunk in &chunks {
            self.retriever.index(chunk.clone())?;
        }

        self.document_count += 1;
        Ok(chunks)
    }

    /// Index multiple documents
    pub fn index_documents(&mut self, documents: &[Document]) -> Result<usize> {
        let mut total_chunks = 0;
        for doc in documents {
            let chunks = self.index_document(doc)?;
            total_chunks += chunks.len();
        }
        Ok(total_chunks)
    }

    /// Get the number of indexed documents
    #[must_use]
    pub fn document_count(&self) -> usize {
        self.document_count
    }

    /// Get the number of indexed chunks
    #[must_use]
    pub fn chunk_count(&self) -> usize {
        self.retriever.len()
    }

    /// Query the pipeline
    pub fn query(&self, query: &str, k: usize) -> Result<Vec<RetrievalResult>> {
        // Retrieve
        let mut results = self.retriever.retrieve(query, k * 2)?;

        // Rerank
        results = self.reranker.rerank(query, &results, k)?;

        Ok(results)
    }

    /// Query and assemble context
    pub fn query_with_context(
        &self,
        query: &str,
        k: usize,
    ) -> Result<(Vec<RetrievalResult>, AssembledContext)> {
        let results = self.query(query, k)?;
        let context = self.assembler.assemble(&results);
        Ok((results, context))
    }

    /// Get the context assembler
    #[must_use]
    pub fn assembler(&self) -> &ContextAssembler {
        &self.assembler
    }

    /// Assemble context from results
    #[must_use]
    pub fn assemble_context(&self, results: &[RetrievalResult]) -> AssembledContext {
        self.assembler.assemble(results)
    }

    /// Get the chunker
    #[must_use]
    pub fn chunker(&self) -> &dyn Chunker {
        self.chunker.as_ref()
    }

    /// Get the embedder
    #[must_use]
    pub fn embedder(&self) -> &E {
        &self.embedder
    }
}

/// Builder for RAG pipeline
pub struct RagPipelineBuilder<E: Embedder, R: Reranker> {
    chunker: Option<Box<dyn Chunker>>,
    embedder: Option<E>,
    vector_store: Option<VectorStore>,
    sparse_index: Option<BM25Index>,
    reranker: Option<R>,
    fusion: FusionStrategy,
    assembler_config: ContextAssemblerConfig,
}

impl<E: Embedder + Clone, R: Reranker> RagPipelineBuilder<E, R> {
    /// Create a new pipeline builder
    #[must_use]
    pub fn new() -> Self {
        Self {
            chunker: None,
            embedder: None,
            vector_store: None,
            sparse_index: None,
            reranker: None,
            fusion: FusionStrategy::default(),
            assembler_config: ContextAssemblerConfig::default(),
        }
    }

    /// Set the chunker
    #[must_use]
    pub fn chunker(mut self, chunker: impl Chunker + 'static) -> Self {
        self.chunker = Some(Box::new(chunker));
        self
    }

    /// Set the embedder
    #[must_use]
    pub fn embedder(mut self, embedder: E) -> Self {
        self.embedder = Some(embedder);
        self
    }

    /// Set the vector store
    #[must_use]
    pub fn vector_store(mut self, store: VectorStore) -> Self {
        self.vector_store = Some(store);
        self
    }

    /// Set the sparse index
    #[must_use]
    pub fn sparse_index(mut self, index: BM25Index) -> Self {
        self.sparse_index = Some(index);
        self
    }

    /// Set the reranker
    #[must_use]
    pub fn reranker(mut self, reranker: R) -> Self {
        self.reranker = Some(reranker);
        self
    }

    /// Set the fusion strategy
    #[must_use]
    pub fn fusion(mut self, fusion: FusionStrategy) -> Self {
        self.fusion = fusion;
        self
    }

    /// Set max context tokens
    #[must_use]
    pub fn max_context_tokens(mut self, max_tokens: usize) -> Self {
        self.assembler_config.max_tokens = max_tokens;
        self
    }

    /// Build the pipeline
    pub fn build(self) -> Result<RagPipeline<E, R>> {
        let embedder =
            self.embedder.ok_or_else(|| Error::InvalidConfig("embedder required".to_string()))?;

        let reranker =
            self.reranker.ok_or_else(|| Error::InvalidConfig("reranker required".to_string()))?;

        let chunker = self.chunker.unwrap_or_else(|| Box::new(RecursiveChunker::new(512, 50)));

        let vector_store =
            self.vector_store.unwrap_or_else(|| VectorStore::with_dimension(embedder.dimension()));

        let sparse_index = self.sparse_index.unwrap_or_default();

        let retrieval_config = HybridRetrieverConfig { fusion: self.fusion, ..Default::default() };

        let retriever = HybridRetriever::new(vector_store, sparse_index, embedder.clone())
            .with_config(retrieval_config);

        let assembler = ContextAssembler::new(self.assembler_config);

        Ok(RagPipeline { chunker, embedder, retriever, reranker, assembler, document_count: 0 })
    }
}

impl<E: Embedder + Clone, R: Reranker> Default for RagPipelineBuilder<E, R> {
    fn default() -> Self {
        Self::new()
    }
}

/// Simplified pipeline builder with defaults
#[must_use]
pub fn pipeline_builder() -> RagPipelineBuilder<MockEmbedder, NoOpReranker> {
    RagPipelineBuilder::new()
}

#[cfg(test)]
mod tests;
