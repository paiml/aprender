//! RAG Oracle - Intelligent retrieval-augmented generation for Sovereign AI Stack
//!
//! Implements the APR-Powered RAG Oracle specification with:
//! - Content-addressable indexing (BLAKE3)
//! - Hybrid retrieval (BM25 + dense)
//! - Heijunka load-leveled reindexing
//! - Jidoka stop-on-error validation
//!
//! # Toyota Production System Principles
//!
//! - **Jidoka**: Stop-on-error during indexing
//! - **Poka-Yoke**: Content hashing prevents stale indexes
//! - **Heijunka**: Load-leveled incremental reindexing
//! - **Kaizen**: Continuous embedding improvement
//! - **Genchi Genbutsu**: Direct observation of source docs
//! - **Muda**: Delta-only updates eliminate waste

// Allow dead code and unused imports for library implementation
// Full integration will use all exported types
pub mod binary_index;
mod chunker;
mod falsification;
pub mod fingerprint;
mod indexer;
pub mod persistence;
pub mod profiling;
pub mod quantization;
pub mod query_cache;
mod retriever;
pub mod tui;
mod types;
mod validator;

// Binary index exports
#[allow(unused_imports)]
pub use binary_index::{
    BinaryIndexError, BinaryIndexReader, BinaryIndexWriter, DocumentEntry, IndexHeader, Posting,
    MAGIC, VERSION,
};
#[allow(unused_imports)]
pub use chunker::SemanticChunker;
#[allow(unused_imports)]
pub use fingerprint::{blake3_hash, ChunkerConfig, DocumentFingerprint};
#[allow(unused_imports)]
pub use indexer::HeijunkaReindexer;
// Profiling exports
#[allow(unused_imports)]
pub use profiling::{
    get_summary, record_cache_hit, record_cache_miss, record_query_latency, reset_metrics, span,
    Counter, Histogram, HistogramBucket, MetricsSummary, RagMetrics, SpanStats, TimedSpan,
    GLOBAL_METRICS,
};
// Query cache exports
#[allow(unused_imports)]
pub use query_cache::{CacheStats, CachedPlan, QueryPlanCache};
// Scalar Int8 Rescoring exports (specification implementation)
#[allow(unused_imports)]
pub use quantization::{
    CalibrationStats, QuantizationError, QuantizationParams, QuantizedEmbedding, RescoreResult,
    RescoreRetriever, RescoreRetrieverConfig, SimdBackend,
};
#[allow(unused_imports)]
pub use retriever::{HybridRetriever, InvertedIndex};
#[allow(unused_imports)]
pub use types::RetrievalResult;
#[allow(unused_imports)]
pub use types::*;
#[allow(unused_imports)]
pub use validator::JidokaIndexValidator;

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

/// RAG Oracle - Main interface for stack documentation queries
///
/// Dogfoods the Sovereign AI Stack:
/// - `trueno-rag` for chunking and retrieval
/// - `trueno-db` for vector storage
/// - `aprender` for embeddings (.apr format)
/// - `simular` for deterministic testing
#[derive(Debug)]
pub struct RagOracle {
    /// Document index with fingerprints
    index: DocumentIndex,
    /// Hybrid retriever (BM25 + dense)
    retriever: HybridRetriever,
    /// Jidoka validator
    validator: JidokaIndexValidator,
    /// Configuration
    config: RagOracleConfig,
}

/// RAG Oracle configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RagOracleConfig {
    /// Stack component repositories to index
    pub repositories: Vec<PathBuf>,
    /// Document sources to include
    pub sources: Vec<DocumentSource>,
    /// Chunk size in tokens
    pub chunk_size: usize,
    /// Chunk overlap in tokens
    pub chunk_overlap: usize,
    /// Number of results to return
    pub top_k: usize,
    /// Reranking depth
    pub rerank_depth: usize,
}

impl Default for RagOracleConfig {
    fn default() -> Self {
        Self {
            repositories: vec![],
            sources: vec![
                DocumentSource::ClaudeMd,
                DocumentSource::ReadmeMd,
                DocumentSource::CargoToml,
                DocumentSource::DocsDir,
            ],
            chunk_size: 512,
            chunk_overlap: 64,
            top_k: 5,
            rerank_depth: 20,
        }
    }
}

/// Document source types with priority (Genchi Genbutsu)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DocumentSource {
    /// CLAUDE.md - P0 Critical, indexed on every commit
    ClaudeMd,
    /// README.md - P1 High, indexed on release
    ReadmeMd,
    /// Cargo.toml - P1 High, indexed on version bump
    CargoToml,
    /// pyproject.toml - P1 High, Python project metadata
    PyProjectToml,
    /// docs/*.md - P2 Medium, weekly scan
    DocsDir,
    /// examples/*.rs - P3 Low, monthly scan
    ExamplesDir,
    /// Docstrings - P3 Low, on release
    Docstrings,
    /// Python source files - P2 Medium, for ground truth corpora
    PythonSource,
    /// Python test files - P3 Low, for ground truth validation
    PythonTests,
}
impl DocumentSource {
    /// Get priority level (0 = highest)
    pub fn priority(&self) -> u8 {
        match self {
            Self::ClaudeMd => 0,
            Self::ReadmeMd | Self::CargoToml | Self::PyProjectToml => 1,
            Self::DocsDir | Self::PythonSource => 2,
            Self::ExamplesDir | Self::Docstrings | Self::PythonTests => 3,
        }
    }

    /// Get glob pattern for this source
    pub fn glob_pattern(&self) -> &'static str {
        match self {
            Self::ClaudeMd => "CLAUDE.md",
            Self::ReadmeMd => "README.md",
            Self::CargoToml => "Cargo.toml",
            Self::PyProjectToml => "pyproject.toml",
            Self::DocsDir => "docs/**/*.md",
            Self::ExamplesDir => "examples/**/*.rs",
            Self::Docstrings => "src/**/*.rs",
            Self::PythonSource => "src/**/*.py",
            Self::PythonTests => "tests/**/*.py",
        }
    }
}

/// Document index containing all indexed documents
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct DocumentIndex {
    /// Documents by ID
    documents: HashMap<String, IndexedDocument>,
    /// Fingerprints for change detection
    fingerprints: HashMap<String, DocumentFingerprint>,
    /// Total chunks indexed
    total_chunks: usize,
}

/// An indexed document with chunks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexedDocument {
    /// Unique document ID
    pub id: String,
    /// Source component (e.g., "trueno", "aprender")
    pub component: String,
    /// Source file path
    pub path: PathBuf,
    /// Document source type
    pub source_type: DocumentSource,
    /// Document chunks
    pub chunks: Vec<DocumentChunk>,
}

/// A chunk of a document
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentChunk {
    /// Chunk ID (document_id + chunk_index)
    pub id: String,
    /// Chunk content
    pub content: String,
    /// Start line in source document
    pub start_line: usize,
    /// End line in source document
    pub end_line: usize,
    /// Content hash for deduplication
    pub content_hash: [u8; 32],
}
impl RagOracle {
    /// Create a new RAG Oracle with default configuration
    pub fn new() -> Self {
        Self::with_config(RagOracleConfig::default())
    }

    /// Create a new RAG Oracle with custom configuration
    pub fn with_config(config: RagOracleConfig) -> Self {
        Self {
            index: DocumentIndex::default(),
            retriever: HybridRetriever::new(),
            validator: JidokaIndexValidator::new(384), // 384-dim embeddings
            config,
        }
    }

    /// Query the oracle with natural language
    pub fn query(&self, query: &str) -> Vec<RetrievalResult> {
        self.retriever.retrieve(query, &self.index, self.config.top_k)
    }

    /// Get index statistics
    pub fn stats(&self) -> IndexStats {
        IndexStats {
            total_documents: self.index.documents.len(),
            total_chunks: self.index.total_chunks,
            components: self
                .index
                .documents
                .values()
                .map(|d| d.component.clone())
                .collect::<std::collections::HashSet<_>>()
                .len(),
        }
    }

    /// Check if a document needs reindexing (Poka-Yoke)
    pub fn needs_reindex(&self, doc_id: &str, current_hash: [u8; 32]) -> bool {
        self.index
            .fingerprints
            .get(doc_id)
            .map(|fp| fp.content_hash != current_hash)
            .unwrap_or(true)
    }
}

impl Default for RagOracle {
    fn default() -> Self {
        Self::new()
    }
}

/// Index statistics
#[derive(Debug, Clone)]
pub struct IndexStats {
    /// Total documents indexed
    pub total_documents: usize,
    /// Total chunks indexed
    pub total_chunks: usize,
    /// Number of components
    pub components: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rag_oracle_creation() {
        let oracle = RagOracle::new();
        let stats = oracle.stats();
        assert_eq!(stats.total_documents, 0);
        assert_eq!(stats.total_chunks, 0);
    }

    #[test]
    fn test_rag_oracle_default() {
        let oracle = RagOracle::default();
        let stats = oracle.stats();
        assert_eq!(stats.total_documents, 0);
        assert_eq!(stats.components, 0);
    }

    #[test]
    fn test_rag_oracle_with_config() {
        let config = RagOracleConfig {
            repositories: vec![PathBuf::from("/test")],
            sources: vec![DocumentSource::ClaudeMd],
            chunk_size: 256,
            chunk_overlap: 32,
            top_k: 10,
            rerank_depth: 50,
        };
        let oracle = RagOracle::with_config(config);
        let stats = oracle.stats();
        assert_eq!(stats.total_documents, 0);
    }

    #[test]
    fn test_rag_oracle_query_empty_index() {
        let oracle = RagOracle::new();
        let results = oracle.query("test query");
        assert!(results.is_empty());
    }

    #[test]
    fn test_document_source_priority() {
        assert_eq!(DocumentSource::ClaudeMd.priority(), 0);
        assert_eq!(DocumentSource::ReadmeMd.priority(), 1);
        assert_eq!(DocumentSource::CargoToml.priority(), 1);
        assert_eq!(DocumentSource::PyProjectToml.priority(), 1);
        assert_eq!(DocumentSource::DocsDir.priority(), 2);
        assert_eq!(DocumentSource::PythonSource.priority(), 2);
        assert_eq!(DocumentSource::ExamplesDir.priority(), 3);
        assert_eq!(DocumentSource::Docstrings.priority(), 3);
        assert_eq!(DocumentSource::PythonTests.priority(), 3);
    }

    #[test]
    fn test_document_source_glob_patterns() {
        assert_eq!(DocumentSource::ClaudeMd.glob_pattern(), "CLAUDE.md");
        assert_eq!(DocumentSource::ReadmeMd.glob_pattern(), "README.md");
        assert_eq!(DocumentSource::CargoToml.glob_pattern(), "Cargo.toml");
        assert_eq!(DocumentSource::PyProjectToml.glob_pattern(), "pyproject.toml");
        assert_eq!(DocumentSource::DocsDir.glob_pattern(), "docs/**/*.md");
        assert_eq!(DocumentSource::ExamplesDir.glob_pattern(), "examples/**/*.rs");
        assert_eq!(DocumentSource::Docstrings.glob_pattern(), "src/**/*.rs");
        assert_eq!(DocumentSource::PythonSource.glob_pattern(), "src/**/*.py");
        assert_eq!(DocumentSource::PythonTests.glob_pattern(), "tests/**/*.py");
    }

    #[test]
    fn test_config_defaults() {
        let config = RagOracleConfig::default();
        assert_eq!(config.chunk_size, 512);
        assert_eq!(config.chunk_overlap, 64);
        assert_eq!(config.top_k, 5);
        assert_eq!(config.rerank_depth, 20);
        assert!(config.repositories.is_empty());
        assert!(!config.sources.is_empty());
    }

    #[test]
    fn test_config_default_sources() {
        let config = RagOracleConfig::default();
        assert!(config.sources.contains(&DocumentSource::ClaudeMd));
        assert!(config.sources.contains(&DocumentSource::ReadmeMd));
        assert!(config.sources.contains(&DocumentSource::CargoToml));
        assert!(config.sources.contains(&DocumentSource::DocsDir));
    }

    #[test]
    fn test_needs_reindex_new_document() {
        let oracle = RagOracle::new();
        let hash = [0u8; 32];
        assert!(oracle.needs_reindex("new_doc", hash));
    }

    #[test]
    fn test_document_index_default() {
        let index = DocumentIndex::default();
        assert!(index.documents.is_empty());
        assert!(index.fingerprints.is_empty());
        assert_eq!(index.total_chunks, 0);
    }

    #[test]
    fn test_index_stats_components() {
        let oracle = RagOracle::new();
        let stats = oracle.stats();
        assert_eq!(stats.components, 0);
    }

    // Property-based tests for RAG Oracle
    mod proptests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            #![proptest_config(ProptestConfig::with_cases(50))]

            /// Property: Oracle always returns empty results for empty index
            #[test]
            fn prop_empty_oracle_returns_empty(query in "[a-z ]{1,100}") {
                let oracle = RagOracle::new();
                let results = oracle.query(&query);
                prop_assert!(results.is_empty());
            }

            /// Property: Config chunk_overlap is always less than chunk_size
            #[test]
            fn prop_config_overlap_less_than_size(
                chunk_size in 64usize..1024,
                overlap_factor in 0.0f64..0.5
            ) {
                let overlap = (chunk_size as f64 * overlap_factor) as usize;
                let config = RagOracleConfig {
                    chunk_size,
                    chunk_overlap: overlap,
                    ..Default::default()
                };
                prop_assert!(config.chunk_overlap <= config.chunk_size);
            }

            /// Property: needs_reindex always returns true for new documents
            #[test]
            fn prop_needs_reindex_new_doc(doc_id in "[a-z]{3,20}", hash in prop::array::uniform32(0u8..)) {
                let oracle = RagOracle::new();
                prop_assert!(oracle.needs_reindex(&doc_id, hash));
            }

            /// Property: Document source priorities are valid (0-3)
            #[test]
            fn prop_source_priority_valid(source_idx in 0usize..9) {
                let sources = [
                    DocumentSource::ClaudeMd,
                    DocumentSource::ReadmeMd,
                    DocumentSource::CargoToml,
                    DocumentSource::PyProjectToml,
                    DocumentSource::DocsDir,
                    DocumentSource::ExamplesDir,
                    DocumentSource::Docstrings,
                    DocumentSource::PythonSource,
                    DocumentSource::PythonTests,
                ];
                let source = sources[source_idx];
                prop_assert!(source.priority() <= 3);
            }

            /// Property: Glob patterns are non-empty
            #[test]
            fn prop_glob_pattern_nonempty(source_idx in 0usize..9) {
                let sources = [
                    DocumentSource::ClaudeMd,
                    DocumentSource::ReadmeMd,
                    DocumentSource::CargoToml,
                    DocumentSource::PyProjectToml,
                    DocumentSource::DocsDir,
                    DocumentSource::ExamplesDir,
                    DocumentSource::Docstrings,
                    DocumentSource::PythonSource,
                    DocumentSource::PythonTests,
                ];
                let source = sources[source_idx];
                prop_assert!(!source.glob_pattern().is_empty());
            }

            /// Property: Stats are consistent
            #[test]
            fn prop_stats_consistent(_seed in 0u64..1000) {
                let oracle = RagOracle::new();
                let stats = oracle.stats();
                // Empty oracle should have all zeros
                prop_assert_eq!(stats.total_documents, 0);
                prop_assert_eq!(stats.total_chunks, 0);
                prop_assert_eq!(stats.components, 0);
            }
        }
    }
}
