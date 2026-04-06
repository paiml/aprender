//! Trueno-RAG: Pure-Rust Retrieval-Augmented Generation Pipeline
//!
//! This crate provides a complete RAG stack built on Trueno compute primitives
//! with zero Python/C++ dependencies.
//!
//! # Quick Start
//!
//! ```rust
//! use trueno_rag::{
//!     pipeline::RagPipelineBuilder,
//!     chunk::RecursiveChunker,
//!     embed::MockEmbedder,
//!     rerank::NoOpReranker,
//!     fusion::FusionStrategy,
//!     Document,
//! };
//!
//! // Build a RAG pipeline
//! let mut pipeline = RagPipelineBuilder::new()
//!     .chunker(RecursiveChunker::new(512, 50))
//!     .embedder(MockEmbedder::new(384))
//!     .reranker(NoOpReranker::new())
//!     .fusion(FusionStrategy::RRF { k: 60.0 })
//!     .build()
//!     .unwrap();
//!
//! // Index a document
//! let doc = Document::new("Machine learning enables computers to learn from data.")
//!     .with_title("ML Intro");
//! pipeline.index_document(&doc).unwrap();
//!
//! // Query the pipeline
//! let results = pipeline.query("machine learning", 5).unwrap();
//! assert!(!results.is_empty());
//! ```
//!
//! # Chunking Strategies
//!
//! Multiple chunking strategies are available:
//!
//! - [`RecursiveChunker`] - Hierarchical splitting (default)
//! - [`FixedSizeChunker`] - Character-based splitting
//! - [`SentenceChunker`] - Sentence-boundary aware
//! - [`ParagraphChunker`] - Paragraph grouping
//! - [`SemanticChunker`] - Embedding similarity-based
//! - [`StructuralChunker`] - Header/section-aware
//! - [`TimestampChunker`] - Subtitle/transcript time-boundary aware
//!
//! # Fusion Strategies
//!
//! Combine dense and sparse retrieval results:
//!
//! - [`FusionStrategy::RRF`] - Reciprocal Rank Fusion (recommended)
//! - [`FusionStrategy::Linear`] - Weighted combination
//! - [`FusionStrategy::DBSF`] - Distribution-based score fusion
//!
//! # Example: Custom Chunking
//!
//! ```rust
//! use trueno_rag::{chunk::{ParagraphChunker, Chunker}, Document};
//!
//! let chunker = ParagraphChunker::new(2); // 2 paragraphs per chunk
//! let doc = Document::new("Para 1.\n\nPara 2.\n\nPara 3.");
//! let chunks = chunker.chunk(&doc).unwrap();
//! assert_eq!(chunks.len(), 2);
//! ```
#![deny(missing_docs)]
#![deny(clippy::all)]
#![allow(clippy::disallowed_methods)] // json! macro internally uses unwrap
#![allow(clippy::return_self_not_must_use)]
#![warn(clippy::pedantic)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::missing_panics_doc)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::map_unwrap_or)]
#![allow(clippy::redundant_closure_for_method_calls)]
#![allow(clippy::unnecessary_literal_bound)]
#![allow(clippy::cloned_instead_of_copied)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::assigning_clones)]
#![allow(clippy::manual_div_ceil)]
#![allow(clippy::unnecessary_map_or)]
#![allow(clippy::derivable_impls)]
#[macro_use]
#[allow(unused_macros)]
mod generated_contracts;
pub mod chunk;
#[cfg(feature = "compression")]
pub mod compressed;
pub mod embed;
pub mod error;
#[cfg(feature = "eval")]
pub mod eval;
pub mod fusion;
pub mod index;
pub mod loader;
pub mod media;
pub mod metrics;
#[cfg(feature = "multivector")]
pub mod multivector;
pub mod pipeline;
pub mod preprocess;
pub mod rerank;
pub mod retrieve;
#[cfg(feature = "sqlite")]
#[allow(clippy::cast_possible_wrap, clippy::cast_sign_loss, clippy::cast_possible_truncation)]
pub mod sqlite;
pub use chunk::{
    Chunk, ChunkId, ChunkMetadata, Chunker, ChunkingStrategy, FixedSizeChunker, ParagraphChunker,
    RecursiveChunker, SemanticChunker, SentenceChunker, StructuralChunker, TimestampChunker,
};
#[cfg(feature = "compression")]
pub use compressed::Compression;
pub use embed::{Embedder, EmbeddingConfig, PoolingStrategy};
#[cfg(feature = "embeddings")]
pub use embed::{EmbeddingModelType, FastEmbedder};
pub use error::{Error, Result};
pub use fusion::FusionStrategy;
pub use index::{BM25Index, SparseIndex, VectorStore};
#[cfg(feature = "transcription")]
pub use loader::transcription::{TranscriptionBackend, TranscriptionConfig, TranscriptionLoader};
#[cfg(feature = "ocr")]
pub use loader::ImageLoader;
pub use loader::{DocumentLoader, LoaderRegistry, SubtitleLoader, TextLoader};
pub use media::{parse_subtitles, SubtitleCue, SubtitleFormat, SubtitleTrack};
pub use metrics::{AggregatedMetrics, RetrievalMetrics};
#[cfg(feature = "multivector")]
pub use multivector::{
    exact_maxsim, MockMultiVectorEmbedder, MultiVectorEmbedder, MultiVectorEmbedding,
    ResidualCodec, WarpIndex, WarpIndexConfig, WarpSearchConfig,
};
pub use pipeline::{ContextAssembler, RagPipeline};
pub use rerank::Reranker;
#[cfg(feature = "multivector")]
pub use retrieve::MultiVectorRetriever;
pub use retrieve::{HybridRetriever, RetrievalResult};
#[cfg(feature = "sqlite")]
pub use sqlite::{SqliteIndex, SqliteStore};
/// Document identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct DocumentId(pub uuid::Uuid);
impl DocumentId {
    /// Create a new random document ID
    #[must_use]
    pub fn new() -> Self {
        Self(uuid::Uuid::new_v4())
    }
}
impl Default for DocumentId {
    fn default() -> Self {
        Self::new()
    }
}
impl std::fmt::Display for DocumentId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}
/// A document to be indexed
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Document {
    /// Unique document identifier
    pub id: DocumentId,
    /// Document content
    pub content: String,
    /// Document title
    pub title: Option<String>,
    /// Source URL or path
    pub source: Option<String>,
    /// Custom metadata
    pub metadata: std::collections::HashMap<String, serde_json::Value>,
}
impl Document {
    /// Create a new document with the given content
    #[must_use]
    pub fn new(content: impl Into<String>) -> Self {
        Self {
            id: DocumentId::new(),
            content: content.into(),
            title: None,
            source: None,
            metadata: std::collections::HashMap::new(),
        }
    }
    /// Set the document title
    #[must_use]
    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
    }
    /// Set the document source
    #[must_use]
    pub fn with_source(mut self, source: impl Into<String>) -> Self {
        self.source = Some(source.into());
        self
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_document_id_unique() {
        let id1 = DocumentId::new();
        let id2 = DocumentId::new();
        assert_ne!(id1, id2);
    }
    #[test]
    fn test_document_creation() {
        let doc = Document::new("Hello, world!");
        assert_eq!(doc.content, "Hello, world!");
        assert!(doc.title.is_none());
        assert!(doc.source.is_none());
    }
    #[test]
    fn test_document_builder() {
        let doc =
            Document::new("Content").with_title("Test Title").with_source("https://example.com");
        assert_eq!(doc.content, "Content");
        assert_eq!(doc.title, Some("Test Title".to_string()));
        assert_eq!(doc.source, Some("https://example.com".to_string()));
    }
    #[test]
    fn test_document_id_display() {
        let id = DocumentId::new();
        let display = format!("{id}");
        assert!(!display.is_empty());
        assert!(display.contains('-')); // UUID format
    }
    #[test]
    fn test_document_id_serialization() {
        let id = DocumentId::new();
        let json = serde_json::to_string(&id).unwrap();
        let deserialized: DocumentId = serde_json::from_str(&json).unwrap();
        assert_eq!(id, deserialized);
    }
}
