# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2024-11-28

### Added

#### Core Pipeline
- `RagPipelineBuilder` for fluent pipeline construction
- `RagPipeline` with document indexing and querying
- `ContextAssembler` with citation tracking
- `AssembledContext` with formatted output

#### Chunking (6 strategies)
- `RecursiveChunker` - Hierarchical splitting with configurable separators
- `FixedSizeChunker` - Character-based splitting with overlap
- `SentenceChunker` - Sentence-boundary aware chunking
- `ParagraphChunker` - Paragraph grouping
- `SemanticChunker` - Embedding similarity-based chunking
- `StructuralChunker` - Header/section-aware chunking for Markdown

#### Embedding
- `MockEmbedder` - Testing/development embedder
- `TfIdfEmbedder` - Simple TF-IDF based embeddings
- `cosine_similarity` and `euclidean_distance` functions

#### Indexing
- `VectorStore` - Dense vector index
- `BM25Index` - Sparse term-based index with configurable k1/b parameters

#### Retrieval
- `HybridRetriever` - Combined dense + sparse retrieval
- `DenseRetriever` - Vector-only retrieval
- `SparseRetriever` - BM25-only retrieval

#### Fusion Strategies
- `RRF` - Reciprocal Rank Fusion
- `Linear` - Weighted linear combination
- `Convex` - Convex combination
- `DBSF` - Distribution-Based Score Fusion
- `Union` - Union of results
- `Intersection` - Intersection of results

#### Reranking
- `NoOpReranker` - Pass-through reranker
- `LexicalReranker` - Lexical feature scoring
- `MockCrossEncoderReranker` - Cross-encoder pattern testing
- `CompositeReranker` - Combine multiple rerankers

#### Query Preprocessing
- `PassthroughPreprocessor` - No-op preprocessor
- `HydePreprocessor` - Hypothetical Document Embeddings
- `MultiQueryPreprocessor` - Query expansion
- `KeywordExpander` - Keyword-based expansion
- `SynonymExpander` - Synonym-based expansion
- `ChainedPreprocessor` - Combine preprocessors
- `QueryAnalyzer` - Intent detection

#### Evaluation Metrics
- `RetrievalMetrics` - Per-query metrics
- `AggregatedMetrics` - Cross-query aggregation
- Recall@k, Precision@k, MRR, NDCG@k, MAP, F1@k, Hit Rate

#### Documentation
- mdBook documentation with 18 chapters
- 4 runnable examples
- API reference

#### Testing
- 261 tests total
- Property-based testing with proptest
- Integration tests for full pipeline

### Dependencies
- `trueno` v0.7.3 - SIMD-accelerated operations
- `trueno-db` v0.3.3 - Vector storage
- `serde` / `serde_json` - Serialization
- `uuid` - Unique identifiers
- `thiserror` - Error handling
- `proptest` (dev) - Property-based testing
