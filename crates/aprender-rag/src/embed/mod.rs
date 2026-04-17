//! Embedding generation for RAG pipelines

#[cfg(feature = "nemotron")]
mod nemotron;
#[cfg(feature = "nemotron")]
pub use nemotron::{NemotronConfig, NemotronEmbedder};

use crate::{Chunk, Error, Result};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

/// Pooling strategy for token embeddings
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
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

impl Default for PoolingStrategy {
    fn default() -> Self {
        Self::Mean
    }
}

/// Configuration for embedding generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingConfig {
    /// Normalize embeddings to unit length
    pub normalize: bool,
    /// Instruction prefix for queries (asymmetric retrieval)
    pub query_prefix: Option<String>,
    /// Instruction prefix for documents
    pub document_prefix: Option<String>,
    /// Maximum sequence length in tokens
    pub max_length: usize,
    /// Pooling strategy
    pub pooling: PoolingStrategy,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            normalize: true,
            query_prefix: None,
            document_prefix: None,
            max_length: 512,
            pooling: PoolingStrategy::Mean,
        }
    }
}

/// Trait for embedding generation
#[async_trait]
pub trait Embedder: Send + Sync {
    /// Embed a single text
    fn embed(&self, text: &str) -> Result<Vec<f32>>;

    /// Batch embed multiple texts
    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>>;

    /// Get embedding dimension
    fn dimension(&self) -> usize;

    /// Get model identifier
    fn model_id(&self) -> &str;

    /// Embed a query (may use query prefix)
    fn embed_query(&self, query: &str) -> Result<Vec<f32>> {
        self.embed(query)
    }

    /// Embed a document (may use document prefix)
    fn embed_document(&self, document: &str) -> Result<Vec<f32>> {
        self.embed(document)
    }

    /// Embed chunks and update them in place
    fn embed_chunks(&self, chunks: &mut [Chunk]) -> Result<()> {
        let texts: Vec<&str> = chunks.iter().map(|c| c.content.as_str()).collect();
        let embeddings = self.embed_batch(&texts)?;

        for (chunk, embedding) in chunks.iter_mut().zip(embeddings) {
            chunk.set_embedding(embedding);
        }

        Ok(())
    }
}

/// Blanket impl so `HybridRetriever<Box<dyn Embedder>>` works without
/// requiring the caller to know the concrete embedder type at compile time.
impl Embedder for Box<dyn Embedder> {
    fn embed(&self, text: &str) -> Result<Vec<f32>> {
        (**self).embed(text)
    }

    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        (**self).embed_batch(texts)
    }

    fn dimension(&self) -> usize {
        (**self).dimension()
    }

    fn model_id(&self) -> &str {
        (**self).model_id()
    }

    fn embed_query(&self, query: &str) -> Result<Vec<f32>> {
        (**self).embed_query(query)
    }

    fn embed_document(&self, document: &str) -> Result<Vec<f32>> {
        (**self).embed_document(document)
    }

    fn embed_chunks(&self, chunks: &mut [Chunk]) -> Result<()> {
        (**self).embed_chunks(chunks)
    }
}

/// Mock embedder for testing (uses simple hash-based vectors)
#[derive(Debug, Clone)]
pub struct MockEmbedder {
    dimension: usize,
    model_id: String,
    config: EmbeddingConfig,
}

impl MockEmbedder {
    /// Create a new mock embedder
    #[must_use]
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            model_id: "mock-embedder".to_string(),
            config: EmbeddingConfig::default(),
        }
    }

    /// Set the model ID
    #[must_use]
    pub fn with_model_id(mut self, model_id: impl Into<String>) -> Self {
        self.model_id = model_id.into();
        self
    }

    /// Set configuration
    #[must_use]
    pub fn with_config(mut self, config: EmbeddingConfig) -> Self {
        self.config = config;
        self
    }

    fn hash_to_vector(&self, text: &str) -> Vec<f32> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut vector = Vec::with_capacity(self.dimension);
        let mut hasher = DefaultHasher::new();

        for i in 0..self.dimension {
            text.hash(&mut hasher);
            i.hash(&mut hasher);
            let hash = hasher.finish();
            // Convert hash to float in range [-1, 1]
            let value = (hash as f32 / u64::MAX as f32) * 2.0 - 1.0;
            vector.push(value);
        }

        if self.config.normalize {
            Self::normalize_vector(&mut vector);
        }

        vector
    }

    fn normalize_vector(vector: &mut [f32]) {
        let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in vector.iter_mut() {
                *x /= norm;
            }
        }
    }
}

impl Embedder for MockEmbedder {
    fn embed(&self, text: &str) -> Result<Vec<f32>> {
        if text.is_empty() {
            return Err(Error::EmptyDocument("empty text for embedding".to_string()));
        }

        let prefixed = if let Some(prefix) = &self.config.document_prefix {
            format!("{prefix}{text}")
        } else {
            text.to_string()
        };

        Ok(self.hash_to_vector(&prefixed))
    }

    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        texts.iter().map(|t| self.embed(t)).collect()
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn embed_query(&self, query: &str) -> Result<Vec<f32>> {
        if query.is_empty() {
            return Err(Error::Query("empty query".to_string()));
        }

        let prefixed = if let Some(prefix) = &self.config.query_prefix {
            format!("{prefix}{query}")
        } else {
            query.to_string()
        };

        Ok(self.hash_to_vector(&prefixed))
    }
}

/// TF-IDF based embedder (sparse-to-dense conversion)
#[derive(Debug, Clone)]
pub struct TfIdfEmbedder {
    dimension: usize,
    vocabulary: std::collections::HashMap<String, usize>,
    idf: Vec<f32>,
}

impl TfIdfEmbedder {
    /// Create a new TF-IDF embedder (untrained)
    #[must_use]
    pub fn new(dimension: usize) -> Self {
        Self { dimension, vocabulary: std::collections::HashMap::new(), idf: Vec::new() }
    }

    /// Train the embedder on a corpus
    pub fn fit(&mut self, documents: &[&str]) {
        use std::collections::{HashMap, HashSet};

        let mut doc_freq: HashMap<String, usize> = HashMap::new();
        let mut all_terms: HashSet<String> = HashSet::new();

        for doc in documents {
            let terms: HashSet<String> = doc.split_whitespace().map(|s| s.to_lowercase()).collect();

            for term in &terms {
                *doc_freq.entry(term.clone()).or_insert(0) += 1;
                all_terms.insert(term.clone());
            }
        }

        // Build vocabulary (top N terms by document frequency)
        let mut terms: Vec<_> = all_terms.into_iter().collect();
        terms.sort_by_key(|t| std::cmp::Reverse(doc_freq.get(t).copied().unwrap_or(0)));
        terms.truncate(self.dimension);

        self.vocabulary = terms.iter().enumerate().map(|(i, t)| (t.clone(), i)).collect();

        // Compute IDF
        let n = documents.len() as f32;
        self.idf = terms
            .iter()
            .map(|t| {
                let df = doc_freq.get(t).copied().unwrap_or(1) as f32;
                (n / df).max(f32::EPSILON).ln() + 1.0
            })
            .collect();
    }

    fn compute_tf(&self, text: &str) -> Vec<f32> {
        let mut tf = vec![0.0f32; self.dimension];
        let terms: Vec<String> = text.split_whitespace().map(|s| s.to_lowercase()).collect();
        let total = terms.len() as f32;

        for term in terms {
            if let Some(&idx) = self.vocabulary.get(&term) {
                tf[idx] += 1.0 / total;
            }
        }

        tf
    }
}

impl Embedder for TfIdfEmbedder {
    fn embed(&self, text: &str) -> Result<Vec<f32>> {
        if text.is_empty() {
            return Err(Error::EmptyDocument("empty text".to_string()));
        }

        if self.vocabulary.is_empty() {
            return Err(Error::InvalidConfig("embedder not trained".to_string()));
        }

        let tf = self.compute_tf(text);
        let mut tfidf: Vec<f32> = tf.iter().zip(self.idf.iter()).map(|(t, i)| t * i).collect();

        // Normalize
        let norm: f32 = tfidf.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in &mut tfidf {
                *x /= norm;
            }
        }

        // Pad to dimension if needed
        tfidf.resize(self.dimension, 0.0);
        Ok(tfidf)
    }

    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        texts.iter().map(|t| self.embed(t)).collect()
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn model_id(&self) -> &str {
        "tfidf"
    }
}

/// Compute cosine similarity between two vectors
#[must_use]
/// Compute L2 (Euclidean) norm of a vector.
fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

/// Divide numerator by denominator, returning 0.0 if denominator is zero.
fn safe_divide(numerator: f32, denominator: f32) -> f32 {
    if denominator == 0.0 {
        0.0
    } else {
        numerator / denominator
    }
}

/// Compute cosine similarity between two vectors.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }
    safe_divide(dot_product(a, b), l2_norm(a) * l2_norm(b))
}

/// Compute dot product between two vectors
#[must_use]
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Compute euclidean distance between two vectors
#[must_use]
pub fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum::<f32>().sqrt()
}

// ============================================================================
// FastEmbed-based Embedder (GH-1: Production-ready semantic embeddings)
// ============================================================================

/// Available embedding models when `embeddings` feature is enabled
#[cfg(feature = "embeddings")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbeddingModelType {
    /// all-MiniLM-L6-v2: Fast, good quality (384 dims)
    AllMiniLmL6V2,
    /// all-MiniLM-L12-v2: Better quality, slightly slower (384 dims)
    AllMiniLmL12V2,
    /// BGE-small-en-v1.5: Balanced performance (384 dims)
    BgeSmallEnV15,
    /// BGE-base-en-v1.5: Higher quality (768 dims)
    BgeBaseEnV15,
    /// NomicEmbed-text-v1: Good for retrieval (768 dims)
    NomicEmbedTextV1,
}

#[cfg(feature = "embeddings")]
impl Default for EmbeddingModelType {
    fn default() -> Self {
        Self::AllMiniLmL6V2
    }
}

#[cfg(feature = "embeddings")]
impl EmbeddingModelType {
    /// Get the fastembed model enum variant
    fn to_fastembed_model(self) -> fastembed::EmbeddingModel {
        match self {
            Self::AllMiniLmL6V2 => fastembed::EmbeddingModel::AllMiniLML6V2,
            Self::AllMiniLmL12V2 => fastembed::EmbeddingModel::AllMiniLML12V2,
            Self::BgeSmallEnV15 => fastembed::EmbeddingModel::BGESmallENV15,
            Self::BgeBaseEnV15 => fastembed::EmbeddingModel::BGEBaseENV15,
            Self::NomicEmbedTextV1 => fastembed::EmbeddingModel::NomicEmbedTextV1,
        }
    }

    /// Get the embedding dimension for this model
    #[must_use]
    pub const fn dimension(self) -> usize {
        match self {
            Self::AllMiniLmL6V2 | Self::AllMiniLmL12V2 | Self::BgeSmallEnV15 => 384,
            Self::BgeBaseEnV15 | Self::NomicEmbedTextV1 => 768,
        }
    }

    /// Get human-readable model name
    #[must_use]
    pub const fn model_name(self) -> &'static str {
        match self {
            Self::AllMiniLmL6V2 => "sentence-transformers/all-MiniLM-L6-v2",
            Self::AllMiniLmL12V2 => "sentence-transformers/all-MiniLM-L12-v2",
            Self::BgeSmallEnV15 => "BAAI/bge-small-en-v1.5",
            Self::BgeBaseEnV15 => "BAAI/bge-base-en-v1.5",
            Self::NomicEmbedTextV1 => "nomic-ai/nomic-embed-text-v1",
        }
    }
}

/// Production-ready semantic embedder using fastembed (ONNX Runtime)
///
/// Requires the `embeddings` feature to be enabled.
///
/// # Example
///
/// ```rust,ignore
/// use trueno_rag::embed::{FastEmbedder, EmbeddingModelType, Embedder};
///
/// let embedder = FastEmbedder::new(EmbeddingModelType::AllMiniLmL6V2)?;
/// let embedding = embedder.embed("Hello, world!")?;
/// assert_eq!(embedding.len(), 384);
/// ```
#[cfg(feature = "embeddings")]
#[derive(Clone)]
pub struct FastEmbedder {
    model: std::sync::Arc<std::sync::Mutex<fastembed::TextEmbedding>>,
    model_type: EmbeddingModelType,
}

#[cfg(feature = "embeddings")]
impl std::fmt::Debug for FastEmbedder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FastEmbedder")
            .field("model_type", &self.model_type)
            .field("dimension", &self.model_type.dimension())
            .finish_non_exhaustive() // model field intentionally omitted (not Debug)
    }
}

#[cfg(feature = "embeddings")]
impl FastEmbedder {
    /// Create a new FastEmbedder with the specified model
    ///
    /// Downloads the model on first use if not cached.
    ///
    /// # Errors
    /// Returns an error if model initialization fails.
    pub fn new(model_type: EmbeddingModelType) -> Result<Self> {
        let options = fastembed::InitOptions::new(model_type.to_fastembed_model())
            .with_show_download_progress(true);

        let model = fastembed::TextEmbedding::try_new(options).map_err(|e| {
            Error::InvalidConfig(format!("Failed to initialize embedding model: {e}"))
        })?;

        Ok(Self { model: std::sync::Arc::new(std::sync::Mutex::new(model)), model_type })
    }

    /// Create with default model (all-MiniLM-L6-v2)
    ///
    /// # Errors
    /// Returns an error if model initialization fails.
    pub fn default_model() -> Result<Self> {
        Self::new(EmbeddingModelType::default())
    }

    /// Get the model type
    #[must_use]
    pub fn model_type(&self) -> EmbeddingModelType {
        self.model_type
    }
}

#[cfg(feature = "embeddings")]
impl Embedder for FastEmbedder {
    fn embed(&self, text: &str) -> Result<Vec<f32>> {
        if text.is_empty() {
            return Err(Error::EmptyDocument("empty text for embedding".to_string()));
        }

        let mut model =
            self.model.lock().map_err(|e| Error::Embedding(format!("lock failed: {e}")))?;

        let embeddings = model
            .embed(vec![text], None)
            .map_err(|e| Error::Embedding(format!("embedding failed: {e}")))?;

        embeddings
            .into_iter()
            .next()
            .ok_or_else(|| Error::Embedding("no embedding returned".to_string()))
    }

    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        // Filter out empty texts
        let non_empty: Vec<&str> = texts.iter().copied().filter(|t| !t.is_empty()).collect();
        if non_empty.is_empty() {
            return Err(Error::EmptyDocument("all texts are empty".to_string()));
        }

        let mut model =
            self.model.lock().map_err(|e| Error::Embedding(format!("lock failed: {e}")))?;

        model
            .embed(non_empty, None)
            .map_err(|e| Error::Embedding(format!("batch embedding failed: {e}")))
    }

    fn dimension(&self) -> usize {
        self.model_type.dimension()
    }

    fn model_id(&self) -> &str {
        self.model_type.model_name()
    }

    fn embed_query(&self, query: &str) -> Result<Vec<f32>> {
        // Some models use query prefixes, but fastembed handles this internally
        self.embed(query)
    }

    fn embed_document(&self, document: &str) -> Result<Vec<f32>> {
        self.embed(document)
    }
}

#[cfg(test)]
mod tests;
