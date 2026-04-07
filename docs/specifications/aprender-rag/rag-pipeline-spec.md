# Trueno-RAG: Retrieval-Augmented Generation Pipeline Specification

**Version:** 1.0.0
**Status:** Draft
**Authors:** Pragmatic AI Labs
**References:** TRUENO-RAG-001

## Abstract

Trueno-RAG provides a sovereign, pure-Rust implementation of Retrieval-Augmented
Generation (RAG) pipelines. This specification defines the architecture for document
chunking, hybrid retrieval (dense + sparse), cross-encoder reranking, and context
assembly—all built on the Trueno compute primitives with zero Python/C++ dependencies.

## 1. Introduction

### 1.1 Motivation

Retrieval-Augmented Generation (RAG) has emerged as the dominant paradigm for
grounding large language models in factual, up-to-date information [1]. However,
existing RAG frameworks (LangChain, LlamaIndex) suffer from:

1. **Python dependency hell** - Complex dependency graphs, version conflicts
2. **Performance overhead** - GIL limitations, serialization costs
3. **Deployment friction** - Cannot compile to WASM, Lambda, or embedded
4. **Sovereignty concerns** - External API calls, data leaving infrastructure

Trueno-RAG addresses these by providing a complete RAG stack in pure Rust, integrated with the Sovereign AI ecosystem.

> **Toyota Way Review: Elimination of Waste (Muda)**
> By removing "dependency hell," we eliminate the *waste of waiting* and
> *waste of processing* associated with complex environments. However, we must
> ensure `aprender` and `realizar` do not introduce the *waste of overproduction*
> (reinventing wheels that already roll smoothly).

### 1.2 Design Principles

- **Pure Rust** - No FFI, no Python, compiles to any target
- **Modular pipeline** - Swap components without rewriting
- **Hybrid retrieval** - Best of dense (semantic) and sparse (lexical)
- **Production-ready** - Streaming, batching, observability built-in

### 1.3 Integration

Trueno-RAG builds on and integrates with:
- **trueno** - SIMD-accelerated vector operations
- **trueno-db** - Vector storage with HNSW indexing
- **aprender** - Embedding model inference (when local)
- **realizar** - LLM inference for generation
- **alimentar** - Document loading and preprocessing

## 2. Architecture

### 2.1 Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Trueno-RAG Pipeline                                │
├─────────────┬─────────────┬─────────────┬─────────────┬─────────────────────┤
│   Ingest    │   Chunk     │   Embed     │   Index     │      Retrieve       │
│  ─────────  │  ────────   │  ────────   │  ────────   │     ──────────      │
│  Documents  │  Splitting  │  Vectors    │  trueno-db  │  Dense + Sparse     │
│  Extraction │  Strategies │  Generation │  HNSW       │  + Reranking        │
├─────────────┴─────────────┴─────────────┴─────────────┴─────────────────────┤
│                         Context Assembly & Generation                        │
│                     (Citation tracking, prompt formatting)                   │
└─────────────────────────────────────────────────────────────────────────────┘
```

> **Toyota Way Review: Visual Control**
> This value stream map is clear. To fully embrace *Jidoka* (built-in quality),
> consider explicitly visualizing where "defects" (failed chunks, bad embeddings)
> are detected and ejected from the pipeline to prevent downstream pollution (Andon).

### 2.2 Component Interaction

```rust
pub struct RagPipeline {
    /// Document chunker
    chunker: Box<dyn Chunker>,
    /// Embedding generator
    embedder: Box<dyn Embedder>,
    /// Vector store
    vector_store: VectorStore,
    /// Sparse index (BM25)
    sparse_index: SparseIndex,
    /// Reranker (optional)
    reranker: Option<Box<dyn Reranker>>,
    /// Context assembler
    assembler: ContextAssembler,
}
```

## 3. Document Chunking

### 3.1 Chunking Strategies

Effective chunking is critical for RAG performance [2]. Trueno-RAG supports multiple strategies:

```rust
pub enum ChunkingStrategy {
    /// Fixed-size chunks with overlap
    FixedSize {
        chunk_size: usize,
        overlap: usize,
    },
    /// Split on sentence boundaries
    Sentence {
        max_sentences: usize,
        overlap_sentences: usize,
    },
    /// Split on paragraph boundaries
    Paragraph {
        max_paragraphs: usize,
    },
    /// Recursive character splitting (like LangChain)
    Recursive {
        separators: Vec<String>,
        chunk_size: usize,
        overlap: usize,
    },
    /// Semantic chunking based on embedding similarity
    Semantic {
        similarity_threshold: f32,
        max_chunk_size: usize,
    },
    /// Document-structure aware (headers, sections)
    Structural {
        respect_headers: bool,
        max_section_size: usize,
    },
}
```

> **Toyota Way Review: Built-in Quality (Jidoka)**
> *Fixed-size* chunking is prone to "defects" (cutting semantic context).
> *Semantic* chunking is preferred as it stops the line (the chunk) based on
> quality (meaning) rather than an arbitrary quota (size), reducing the
> *waste of correction* later.

### 3.2 Chunk Data Model

```rust
pub struct Chunk {
    /// Unique chunk identifier
    pub id: ChunkId,
    /// Source document reference
    pub document_id: DocumentId,
    /// Chunk text content
    pub content: String,
    /// Character offset in source document
    pub start_offset: usize,
    pub end_offset: usize,
    /// Metadata inherited from document
    pub metadata: ChunkMetadata,
    /// Embedding vector (populated after embedding)
    pub embedding: Option<Vec<f32>>,
}

pub struct ChunkMetadata {
    /// Source document title
    pub title: Option<String>,
    /// Section/header hierarchy
    pub headers: Vec<String>,
    /// Page number (for PDFs)
    pub page: Option<usize>,
    /// Custom metadata
    pub custom: HashMap<String, Value>,
}
```

### 3.3 Chunking Implementation

```rust
pub trait Chunker: Send + Sync {
    /// Split document into chunks
    fn chunk(&self, document: &Document) -> Result<Vec<Chunk>>;

    /// Estimate chunk count without materializing
    fn estimate_chunks(&self, document: &Document) -> usize;
}

impl Chunker for RecursiveChunker {
    fn chunk(&self, document: &Document) -> Result<Vec<Chunk>> {
        let mut chunks = Vec::new();
        let text = &document.content;

        // Try separators in order: \n\n, \n, sentence, word
        for separator in &self.separators {
            if can_split_with(text, separator, self.chunk_size) {
                return self.split_with_separator(text, separator);
            }
        }

        // Fallback to character splitting
        self.split_by_chars(text)
    }
}
```

## 4. Embedding Generation

### 4.1 Embedding Interface

```rust
pub trait Embedder: Send + Sync {
    /// Embed a single text
    fn embed(&self, text: &str) -> Result<Vec<f32>>;

    /// Batch embed multiple texts
    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>>;

    /// Get embedding dimension
    fn dimension(&self) -> usize;

    /// Get model identifier
    fn model_id(&self) -> &str;
}
```

### 4.2 Embedding Providers

Trueno-RAG supports multiple embedding sources:

```rust
pub enum EmbeddingProvider {
    /// Local model via Realizar
    Local {
        model_path: PathBuf,
        device: Device,
    },
    /// Aprender's built-in embeddings
    Aprender {
        model: AprenderEmbeddingModel,
    },
    /// External API (OpenAI-compatible)
    Api {
        endpoint: String,
        model: String,
        api_key: SecretString,
    },
}
```

> **Toyota Way Review: Genchi Genbutsu (Go and See)**
> Local inference (`Local`) allows for *Genchi Genbutsu*—processing data where it
> resides—reducing the *waste of transport* (sending data to external APIs) and
> increasing sovereignty.

```rust
/// Aprender's native embedding models
pub enum AprenderEmbeddingModel {
    /// TF-IDF based (no neural network)
    TfIdf { max_features: usize },
    /// Word2Vec style
    Word2Vec { dimension: usize },
    /// Sentence embeddings
    SentenceTransformer { model: String },
}
```

### 4.3 Embedding Best Practices

Following embedding research [3], Trueno-RAG implements:

```rust
pub struct EmbeddingConfig {
    /// Normalize embeddings to unit length
    pub normalize: bool,
    /// Instruction prefix for asymmetric retrieval
    pub query_prefix: Option<String>,
    pub document_prefix: Option<String>,
    /// Max sequence length
    pub max_length: usize,
    /// Pooling strategy
    pub pooling: PoolingStrategy,
}

pub enum PoolingStrategy {
    /// Use [CLS] token embedding
    Cls,
    /// Mean of all token embeddings
    Mean,
    /// Mean with attention weighting
    WeightedMean,
    /// Last token (for decoder models)
    LastToken,
}
```

## 5. Vector Storage & Indexing

### 5.1 Integration with Trueno-DB

Trueno-RAG uses trueno-db for vector storage with HNSW indexing [4]:

```rust
pub struct VectorStore {
    /// Trueno-DB instance
    db: TruenoDB,
    /// Collection name
    collection: String,
    /// Index configuration
    index_config: HnswConfig,
}

pub struct HnswConfig {
    /// Number of connections per node
    pub m: usize,
    /// Size of dynamic candidate list during construction
    pub ef_construction: usize,
    /// Size of dynamic candidate list during search
    pub ef_search: usize,
    /// Distance metric
    pub metric: DistanceMetric,
}
```

> **Toyota Way Review: Standardized Work**
> HNSW parameters (`m`, `ef_construction`) are critical standards. These should be
> tuned based on *Heijunka* (leveling) principles to balance indexing speed (load)
> with search accuracy (quality), avoiding overburdening (`Muri`) the system
> during ingestion.

```rust
pub enum DistanceMetric {
    Cosine,
    Euclidean,
    DotProduct,
    Manhattan,
}
```

### 5.2 Vector Operations

```rust
impl VectorStore {
    /// Insert chunks with embeddings
    pub async fn insert(&self, chunks: &[Chunk]) -> Result<()> {
        let records: Vec<_> = chunks.iter()
            .filter_map(|c| c.embedding.as_ref().map(|e| {
                VectorRecord {
                    id: c.id.to_string(),
                    vector: e.clone(),
                    metadata: c.metadata.to_json(),
                }
            }))
            .collect();

        self.db.insert_batch(&self.collection, &records).await
    }

    /// Search for similar vectors
    pub async fn search(
        &self,
        query_vector: &[f32],
        k: usize,
        filter: Option<MetadataFilter>,
    ) -> Result<Vec<SearchResult>> {
        self.db.search(
            &self.collection,
            query_vector,
            k,
            filter,
            self.index_config.ef_search,
        ).await
    }
}
```

## 6. Sparse Retrieval (BM25)

### 6.1 BM25 Implementation

Sparse retrieval remains competitive with dense methods, especially for keyword-heavy queries [5]:

```rust
pub struct BM25Index {
    /// Inverted index: term -> [(doc_id, term_freq)]
    inverted_index: HashMap<String, Vec<(ChunkId, u32)>>,
    /// Document frequencies
    doc_freqs: HashMap<String, u32>,
    /// Document lengths
    doc_lengths: HashMap<ChunkId, u32>,
    /// Average document length
    avg_doc_length: f32,
    /// Total document count
    doc_count: u32,
    /// BM25 parameters
    k1: f32,
    b: f32,
}

impl BM25Index {
    /// BM25 scoring function
    pub fn score(&self, query_terms: &[String], chunk_id: ChunkId) -> f32 {
        let doc_len = self.doc_lengths.get(&chunk_id).copied().unwrap_or(0) as f32;

        query_terms.iter().map(|term| {
            let df = self.doc_freqs.get(term).copied().unwrap_or(0) as f32;
            let idf = ((self.doc_count as f32 - df + 0.5) / (df + 0.5) + 1.0).ln();

            let tf = self.term_frequency(term, chunk_id) as f32;
            let tf_norm = (tf * (self.k1 + 1.0)) /
                (tf + self.k1 * (1.0 - self.b + self.b * doc_len / self.avg_doc_length));

            idf * tf_norm
        }).sum()
    }
```

> **Toyota Way Review: Efficiency**
> While BM25 is robust, this iterative scoring calculation must be scrutinized
> for *Muda* (computational waste). Ensure the `trueno` SIMD primitives are
> leveraged here to minimize cycle time.

```rust
    /// Search with BM25
    pub fn search(&self, query: &str, k: usize) -> Vec<(ChunkId, f32)> {
        let terms = self.tokenize(query);
        let mut scores: HashMap<ChunkId, f32> = HashMap::new();

        for term in &terms {
            if let Some(postings) = self.inverted_index.get(term) {
                for (chunk_id, _) in postings {
                    let score = scores.entry(*chunk_id).or_insert(0.0);
                    *score += self.score(&terms, *chunk_id);
                }
            }
        }

        // Top-k by score
        let mut results: Vec<_> = scores.into_iter().collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        results.truncate(k);
        results
    }
}
```

### 6.2 Tokenization for BM25

```rust
pub struct BM25Tokenizer {
    /// Lowercase all terms
    lowercase: bool,
    /// Remove stopwords
    stopwords: HashSet<String>,
    /// Stemmer (Porter)
    stemmer: Option<PorterStemmer>,
    /// N-gram range
    ngram_range: Option<(usize, usize)>,
}

impl BM25Tokenizer {
    pub fn tokenize(&self, text: &str) -> Vec<String> {
        let mut tokens: Vec<String> = text
            .split_whitespace()
            .map(|s| s.trim_matches(|c: char| !c.is_alphanumeric()))
            .filter(|s| !s.is_empty())
            .map(|s| if self.lowercase { s.to_lowercase() } else { s.to_string() })
            .filter(|s| !self.stopwords.contains(s))
            .map(|s| self.stemmer.as_ref().map_or(s.clone(), |st| st.stem(&s)))
            .collect();

        if let Some((min_n, max_n)) = self.ngram_range {
            tokens.extend(self.generate_ngrams(&tokens, min_n, max_n));
        }

        tokens
    }
}
```

## 7. Hybrid Retrieval

### 7.1 Fusion Strategies

Hybrid retrieval combines dense and sparse results [6]:

```rust
pub enum FusionStrategy {
    /// Reciprocal Rank Fusion
    RRF { k: f32 },
    /// Linear combination of normalized scores
    Linear { dense_weight: f32 },
    /// Convex combination after score normalization
    Convex { alpha: f32 },
    /// Distribution-Based Score Fusion
    DBSF,
}

impl FusionStrategy {
    pub fn fuse(
        &self,
        dense_results: &[(ChunkId, f32)],
        sparse_results: &[(ChunkId, f32)],
    ) -> Vec<(ChunkId, f32)> {
        match self {
            FusionStrategy::RRF { k } => {
                self.reciprocal_rank_fusion(dense_results, sparse_results, *k)
            }
            FusionStrategy::Linear { dense_weight } => {
                self.linear_fusion(dense_results, sparse_results, *dense_weight)
            }
            // ...
        }
    }

    fn reciprocal_rank_fusion(
        &self,
        dense: &[(ChunkId, f32)],
        sparse: &[(ChunkId, f32)],
        k: f32,
    ) -> Vec<(ChunkId, f32)> {
        let mut scores: HashMap<ChunkId, f32> = HashMap::new();

        for (rank, (id, _)) in dense.iter().enumerate() {
            *scores.entry(*id).or_insert(0.0) += 1.0 / (k + rank as f32 + 1.0);
        }
        for (rank, (id, _)) in sparse.iter().enumerate() {
            *scores.entry(*id).or_insert(0.0) += 1.0 / (k + rank as f32 + 1.0);
        }

        let mut results: Vec<_> = scores.into_iter().collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        results
    }
}
```

> **Toyota Way Review: Nemawashi (Decision Making)**
> *Reciprocal Rank Fusion* acts as a consensus mechanism, integrating diverse
> "perspectives" (dense and sparse) to make a better decision. This aligns with
> making decisions slowly by consensus, then implementing rapidly.

### 7.2 Hybrid Retriever

```rust
pub struct HybridRetriever {
    dense: VectorStore,
    sparse: BM25Index,
    embedder: Box<dyn Embedder>,
    fusion: FusionStrategy,
}

impl HybridRetriever {
    pub async fn retrieve(
        &self,
        query: &str,
        k: usize,
    ) -> Result<Vec<RetrievalResult>> {
        // Parallel retrieval
        let query_embedding = self.embedder.embed(query)?;

        let (dense_results, sparse_results) = tokio::join!(
            self.dense.search(&query_embedding, k * 2, None),
            async { Ok(self.sparse.search(query, k * 2)) }
        );

        // Fuse results
        let fused = self.fusion.fuse(
            &dense_results?,
            &sparse_results?,
        );

        // Take top-k and fetch chunks
        let top_k: Vec<_> = fused.into_iter().take(k).collect();
        self.fetch_chunks(&top_k).await
    }
}
```

## 8. Reranking

### 8.1 Cross-Encoder Reranking

Reranking with cross-encoders significantly improves retrieval quality [7]:

```rust
pub trait Reranker: Send + Sync {
    /// Rerank candidates given query
    fn rerank(
        &self,
        query: &str,
        candidates: &[RetrievalResult],
        top_k: usize,
    ) -> Result<Vec<RetrievalResult>>;
}

pub struct CrossEncoderReranker {
    /// Model for scoring query-document pairs
    model: CrossEncoderModel,
    /// Batch size for inference
    batch_size: usize,
}

impl Reranker for CrossEncoderReranker {
    fn rerank(
        &self,
        query: &str,
        candidates: &[RetrievalResult],
        top_k: usize,
    ) -> Result<Vec<RetrievalResult>> {
        // Create query-document pairs
        let pairs: Vec<_> = candidates.iter()
            .map(|c| (query, c.chunk.content.as_str()))
            .collect();

        // Score all pairs
        let scores = self.model.predict_batch(&pairs, self.batch_size)?;

        // Sort by cross-encoder score
        let mut scored: Vec<_> = candidates.iter()
            .zip(scores.iter())
            .map(|(c, s)| (c.clone(), *s))
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Return top-k
        Ok(scored.into_iter()
            .take(top_k)
            .map(|(mut r, score)| {
                r.rerank_score = Some(score);
                r
            })
            .collect())
    }
}
```

### 8.2 Lightweight Reranking

For latency-sensitive applications, Trueno-RAG offers lightweight alternatives:

```rust
pub struct ColBERTReranker {
    /// Late interaction model
    model: ColBERTModel,
}

pub struct MonoT5Reranker {
    /// Sequence-to-sequence reranker
    model: T5Model,
}

/// No neural network - uses lexical features
pub struct LexicalReranker {
    /// Weight for exact match
    exact_match_weight: f32,
    /// Weight for query term coverage
    coverage_weight: f32,
    /// Weight for position bias (earlier = better)
    position_weight: f32,
}
```

## 9. Context Assembly

### 9.1 Context Window Optimization

Assembling retrieved chunks into an optimal context window [8]:

```rust
pub struct ContextAssembler {
    /// Maximum context length in tokens
    max_tokens: usize,
    /// Tokenizer for length estimation
    tokenizer: Box<dyn Tokenizer>,
    /// Assembly strategy
    strategy: AssemblyStrategy,
}

pub enum AssemblyStrategy {
    /// Simple concatenation in rank order
    Sequential,
    /// Group by document, then by rank
    DocumentGrouped,
    /// Interleave for diversity
    Interleaved,
    /// Optimize for coverage of query aspects
    CoverageOptimized,
}

impl ContextAssembler {
    pub fn assemble(
        &self,
        query: &str,
        results: &[RetrievalResult],
    ) -> AssembledContext {
        let mut context = AssembledContext::new();
        let mut remaining_tokens = self.max_tokens;

        for result in results {
            let chunk_tokens = self.tokenizer.count_tokens(&result.chunk.content);

            if chunk_tokens <= remaining_tokens {
                context.add_chunk(result.clone());
                remaining_tokens -= chunk_tokens;
            } else if remaining_tokens > 50 {
                // Truncate chunk to fit
                let truncated = self.truncate_chunk(&result.chunk, remaining_tokens);
                context.add_truncated(result.clone(), truncated);
                break;
            } else {
                break;
            }
        }

        context
    }
}
```

> **Toyota Way Review: Stop and Fix (Jidoka)**
> Truncating chunks to fit context is a "stopgap," not a root cause fix. It
> creates "hidden defects" (missing info). A *Kaizen* opportunity exists here:
> summarize or re-rank rather than blindly chopping, which respects the value
> of the information.

### 9.2 Citation Tracking

```rust
pub struct AssembledContext {
    /// Ordered chunks in context
    pub chunks: Vec<ContextChunk>,
    /// Total token count
    pub total_tokens: usize,
    /// Source citations
    pub citations: Vec<Citation>,
}

pub struct ContextChunk {
    pub content: String,
    pub citation_id: usize,
    pub retrieval_score: f32,
    pub rerank_score: Option<f32>,
}

pub struct Citation {
    pub id: usize,
    pub document_id: DocumentId,
    pub chunk_id: ChunkId,
    pub title: Option<String>,
    pub url: Option<String>,
    pub page: Option<usize>,
}

impl AssembledContext {
    /// Format context with inline citations
    pub fn format_with_citations(&self) -> String {
        self.chunks.iter()
            .map(|c| format!("{} [{}]", c.content, c.citation_id))
            .collect::<Vec<_>>()
            .join("\n\n")
    }

    /// Generate citation list
    pub fn citation_list(&self) -> String {
        self.citations.iter()
            .map(|c| format!("[{}] {}", c.id, c.title.as_deref().unwrap_or("Untitled")))
            .collect::<Vec<_>>()
            .join("\n")
    }
}
```

## 10. Query Processing

### 10.1 Query Enhancement

```rust
pub struct QueryProcessor {
    /// Expand query with synonyms
    pub expansion: Option<QueryExpansion>,
    /// Decompose complex queries
    pub decomposition: Option<QueryDecomposition>,
    /// HyDE: Hypothetical Document Embeddings
    pub hyde: Option<HydeConfig>,
}

/// Hypothetical Document Embeddings [9]
pub struct HydeConfig {
    /// LLM for generating hypothetical answer
    pub generator: Box<dyn TextGenerator>,
    /// Number of hypothetical documents
    pub num_hypotheticals: usize,
    /// Aggregate embeddings
    pub aggregation: HydeAggregation,
}

impl QueryProcessor {
    pub async fn process(&self, query: &str) -> ProcessedQuery {
        let mut processed = ProcessedQuery {
            original: query.to_string(),
            expanded: None,
            decomposed: None,
            hypotheticals: None,
        };

        if let Some(hyde) = &self.hyde {
            // Generate hypothetical documents
            let hypotheticals = hyde.generator
                .generate_hypotheticals(query, hyde.num_hypotheticals)
                .await?;
            processed.hypotheticals = Some(hypotheticals);
        }

        processed
    }
}
```

### 10.2 Multi-Query Retrieval

```rust
pub struct MultiQueryRetriever {
    base_retriever: HybridRetriever,
    query_generator: Box<dyn QueryGenerator>,
    num_queries: usize,
}

impl MultiQueryRetriever {
    pub async fn retrieve(
        &self,
        query: &str,
        k: usize,
    ) -> Result<Vec<RetrievalResult>> {
        // Generate query variations
        let queries = self.query_generator
            .generate_variations(query, self.num_queries)
            .await?;

        // Retrieve for each query
        let mut all_results = Vec::new();
        for q in queries {
            let results = self.base_retriever.retrieve(&q, k).await?;
            all_results.extend(results);
        }

        // Deduplicate and rerank
        self.deduplicate_and_rank(all_results, k)
    }
}
```

> **Toyota Way Review: Pull System**
> Generating multiple queries allows the system to "pull" more relevant
> information based on customer (user) intent, rather than just "pushing" what
> matches the literal string. This improves value, though we must watch for
> the *waste of over-processing*.

## 11. Evaluation

### 11.1 Retrieval Metrics

Following standard IR evaluation [10]:

```rust
pub struct RetrievalMetrics {
    /// Recall@k
    pub recall: HashMap<usize, f32>,
    /// Precision@k
    pub precision: HashMap<usize, f32>,
    /// Mean Reciprocal Rank
    pub mrr: f32,
    /// Normalized Discounted Cumulative Gain
    pub ndcg: HashMap<usize, f32>,
    /// Mean Average Precision
    pub map: f32,
}

impl RetrievalMetrics {
    pub fn compute(
        retrieved: &[ChunkId],
        relevant: &HashSet<ChunkId>,
        k_values: &[usize],
    ) -> Self {
        let mut metrics = Self::default();

        for &k in k_values {
            let retrieved_k: HashSet<_> = retrieved.iter().take(k).collect();
            let relevant_retrieved = retrieved_k.intersection(
                &relevant.iter().collect()
            ).count();

            metrics.recall.insert(k, relevant_retrieved as f32 / relevant.len() as f32);
            metrics.precision.insert(k, relevant_retrieved as f32 / k as f32);
            metrics.ndcg.insert(k, Self::compute_ndcg(retrieved, relevant, k));
        }

        metrics.mrr = Self::compute_mrr(retrieved, relevant);
        metrics.map = Self::compute_map(retrieved, relevant);

        metrics
    }
}
```

### 11.2 End-to-End RAG Evaluation

```rust
pub struct RagEvaluator {
    /// Retrieval metrics
    retrieval: RetrievalMetrics,
    /// Generation quality (if ground truth available)
    generation: Option<GenerationMetrics>,
    /// Faithfulness: does answer match retrieved context?
    faithfulness: Option<f32>,
    /// Answer relevance: does answer address query?
    relevance: Option<f32>,
}

pub struct GenerationMetrics {
    /// Exact match
    pub em: f32,
    /// F1 token overlap
    pub f1: f32,
    /// ROUGE-L
    pub rouge_l: f32,
    /// BERTScore
    pub bert_score: Option<f32>,
}
```

## 12. API Design

### 12.1 High-Level API

```rust
use trueno_rag::{RagPipeline, Config};

// Build pipeline
let pipeline = RagPipeline::builder()
    .chunker(RecursiveChunker::new(512, 50))
    .embedder(LocalEmbedder::new("bge-small-en")?)
    .vector_store(TruenoDB::open("./vectors")?)
    .sparse_index(BM25Index::new())
    .fusion(FusionStrategy::RRF { k: 60.0 })
    .reranker(CrossEncoderReranker::new("ms-marco-MiniLM")?)
    .build()?;

// Index documents
pipeline.index_documents(&documents).await?;

// Query
let results = pipeline.query("What is RAG?", 5).await?;

// Get assembled context
let context = pipeline.assemble_context("What is RAG?", &results)?;
println!("{}", context.format_with_citations());
```

### 12.2 Streaming API

```rust
impl RagPipeline {
    /// Stream retrieval results as they become available
    pub fn query_stream(
        &self,
        query: &str,
        k: usize,
    ) -> impl Stream<Item = RetrievalResult> {
        stream! {
            // Stream dense results first (usually faster)
            let dense = self.vector_store.search_stream(&query_embedding, k);
            pin_mut!(dense);
            while let Some(result) = dense.next().await {
                yield result;
            }

            // Then sparse results
            for result in self.sparse_index.search(query, k) {
                yield self.fetch_chunk(result.0).await;
            }
        }
    }
}
```

> **Toyota Way Review: One-Piece Flow**
> Streaming results implements *continuous flow*, reducing the batch size to one.
> This eliminates the *waste of waiting* for the user, delivering value the
> moment it is created.

## 13. References

[1] Lewis, P., et al. (2020). "Retrieval-Augmented Generation for
Knowledge-Intensive NLP Tasks." *Advances in Neural Information Processing
Systems (NeurIPS)*, 33, 9459-9474. arXiv:2005.11401

[2] Gao, Y., et al. (2024). "Retrieval-Augmented Generation for Large
Language Models: A Survey." *arXiv preprint* arXiv:2312.10997

[3] Karpukhin, V., et al. (2020). "Dense Passage Retrieval for Open-Domain
Question Answering." *Proceedings of EMNLP*, 6769-6781.
DOI: 10.18653/v1/2020.emnlp-main.550

[4] Malkov, Y. A., & Yashunin, D. A. (2020). "Efficient and Robust
Approximate Nearest Neighbor Search Using Hierarchical Navigable Small
World Graphs." *IEEE Transactions on Pattern Analysis and Machine
Intelligence*, 42(4), 824-836. DOI: 10.1109/TPAMI.2018.2889473

[5] Robertson, S., & Zaragoza, H. (2009). "The Probabilistic Relevance
Framework: BM25 and Beyond." *Foundations and Trends in Information
Retrieval*, 3(4), 333-389. DOI: 10.1561/1500000019

[6] Ma, X., et al. (2024). "Fine-Tuning LLaMA for Multi-Stage Text Retrieval." *Proceedings of SIGIR*. arXiv:2310.08319

[7] Nogueira, R., & Cho, K. (2019). "Passage Re-ranking with BERT." *arXiv preprint* arXiv:1901.04085

[8] Liu, N. F., et al. (2024). "Lost in the Middle: How Language Models
Use Long Contexts." *Transactions of the Association for Computational
Linguistics*, 12, 157-173. DOI: 10.1162/tacl_a_00638

[9] Gao, L., et al. (2023). "Precise Zero-Shot Dense Retrieval without
Relevance Labels." *Proceedings of ACL*, 1762-1777.
DOI: 10.18653/v1/2023.acl-long.99

[10] Thakur, N., et al. (2021). "BEIR: A Heterogeneous Benchmark for
Zero-shot Evaluation of Information Retrieval Models." *Proceedings of
NeurIPS Datasets and Benchmarks Track*. arXiv:2104.08663

---

## Appendix A: Glossary

| Term | Definition |
|------|------------|
| **RAG** | Retrieval-Augmented Generation - grounding LLMs with retrieved context |
| **Dense Retrieval** | Semantic search using embedding vectors |
| **Sparse Retrieval** | Lexical search using term frequencies (BM25) |
| **Hybrid Retrieval** | Combination of dense and sparse methods |
| **Reranking** | Second-stage scoring of retrieved candidates |
| **HNSW** | Hierarchical Navigable Small World - ANN index |
| **HyDE** | Hypothetical Document Embeddings |

## Appendix B: Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2024-11-28 | Initial specification |