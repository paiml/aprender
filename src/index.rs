//! Indexing for RAG pipelines (BM25 sparse index and vector store)

use crate::{Chunk, ChunkId, Error, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Default embedding dimension (all-MiniLM-L6-v2 / BGE-small-en-v1.5)
const DEFAULT_EMBEDDING_DIM: usize = 384;

/// Sparse index trait for lexical retrieval
pub trait SparseIndex: Send + Sync {
    /// Index a chunk
    fn add(&mut self, chunk: &Chunk);

    /// Index multiple chunks
    fn add_batch(&mut self, chunks: &[Chunk]);

    /// Search for matching chunks
    fn search(&self, query: &str, k: usize) -> Vec<(ChunkId, f32)>;

    /// Remove a chunk from the index
    fn remove(&mut self, chunk_id: ChunkId);

    /// Get the number of indexed documents
    fn len(&self) -> usize;

    /// Check if the index is empty
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// BM25 index implementation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BM25Index {
    /// Inverted index: term -> [(chunk_id, term_freq)]
    inverted_index: HashMap<String, Vec<(ChunkId, u32)>>,
    /// Document frequencies: term -> doc count
    doc_freqs: HashMap<String, u32>,
    /// Document lengths: chunk_id -> length
    doc_lengths: HashMap<ChunkId, u32>,
    /// Average document length
    avg_doc_length: f32,
    /// Total document count
    doc_count: u32,
    /// BM25 k1 parameter (term frequency saturation)
    k1: f32,
    /// BM25 b parameter (length normalization)
    b: f32,
    /// Tokenizer settings
    lowercase: bool,
    /// Stopwords
    stopwords: HashSet<String>,
}

impl Default for BM25Index {
    fn default() -> Self {
        Self::new()
    }
}

impl BM25Index {
    /// Create a new BM25 index with default parameters
    #[must_use]
    pub fn new() -> Self {
        Self {
            inverted_index: HashMap::new(),
            doc_freqs: HashMap::new(),
            doc_lengths: HashMap::new(),
            avg_doc_length: 0.0,
            doc_count: 0,
            k1: 1.2,
            b: 0.75,
            lowercase: true,
            stopwords: Self::default_stopwords(),
        }
    }

    /// Create with custom BM25 parameters
    #[must_use]
    pub fn with_params(k1: f32, b: f32) -> Self {
        Self { k1, b, ..Self::new() }
    }

    /// Set stopwords
    #[must_use]
    pub fn with_stopwords(mut self, stopwords: HashSet<String>) -> Self {
        self.stopwords = stopwords;
        self
    }

    fn default_stopwords() -> HashSet<String> {
        [
            "a", "an", "the", "is", "are", "was", "were", "be", "been", "being", "have", "has",
            "had", "do", "does", "did", "will", "would", "could", "should", "may", "might", "must",
            "shall", "can", "need", "dare", "ought", "used", "to", "of", "in", "for", "on", "with",
            "at", "by", "from", "as", "into", "through", "during", "before", "after", "above",
            "below", "between", "under", "again", "further", "then", "once", "here", "there",
            "when", "where", "why", "how", "all", "each", "few", "more", "most", "other", "some",
            "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "just",
            "and", "but", "if", "or", "because", "until", "while", "this", "that", "these",
            "those", "it", "its",
        ]
        .iter()
        .map(|s| (*s).to_string())
        .collect()
    }

    /// Tokenize text
    pub fn tokenize(&self, text: &str) -> Vec<String> {
        text.split(|c: char| !c.is_alphanumeric())
            .filter(|s| !s.is_empty())
            .map(|s| if self.lowercase { s.to_lowercase() } else { s.to_string() })
            .filter(|s| !self.stopwords.contains(s))
            .filter(|s| s.len() >= 2) // Filter very short tokens
            .collect()
    }

    /// Compute term frequency in a document
    fn term_frequency(&self, term: &str, chunk_id: ChunkId) -> u32 {
        self.inverted_index
            .get(term)
            .and_then(|postings| postings.iter().find(|(id, _)| *id == chunk_id))
            .map(|(_, freq)| *freq)
            .unwrap_or(0)
    }

    /// Compute BM25 score for a single term
    fn score_term(&self, term: &str, chunk_id: ChunkId) -> f32 {
        let tf = self.term_frequency(term, chunk_id) as f32;
        if tf == 0.0 {
            return 0.0;
        }

        let df = self.doc_freqs.get(term).copied().unwrap_or(0) as f32;
        let n = self.doc_count as f32;
        let doc_len = self.doc_lengths.get(&chunk_id).copied().unwrap_or(0) as f32;

        // IDF component: log((N - df + 0.5) / (df + 0.5) + 1)
        let idf = ((n - df + 0.5) / (df + 0.5) + 1.0).max(f32::EPSILON).ln();

        // TF component with length normalization
        let tf_norm = (tf * (self.k1 + 1.0))
            / (tf + self.k1 * (1.0 - self.b + self.b * doc_len / self.avg_doc_length));

        idf * tf_norm
    }

    /// Update average document length
    fn update_avg_doc_length(&mut self) {
        if self.doc_count == 0 {
            self.avg_doc_length = 0.0;
        } else {
            let total: u32 = self.doc_lengths.values().sum();
            self.avg_doc_length = total as f32 / self.doc_count as f32;
        }
    }

    /// Get chunks containing a term
    fn get_chunks_for_term(&self, term: &str) -> Vec<ChunkId> {
        self.inverted_index
            .get(term)
            .map(|postings| postings.iter().map(|(id, _)| *id).collect())
            .unwrap_or_default()
    }
}

impl SparseIndex for BM25Index {
    fn add(&mut self, chunk: &Chunk) {
        let tokens = self.tokenize(&chunk.content);
        let doc_len = tokens.len() as u32;

        // Update document length
        self.doc_lengths.insert(chunk.id, doc_len);
        self.doc_count += 1;

        // Count term frequencies
        let mut term_freqs: HashMap<String, u32> = HashMap::new();
        for token in &tokens {
            *term_freqs.entry(token.clone()).or_insert(0) += 1;
        }

        // Update inverted index and document frequencies
        let mut seen_terms: HashSet<String> = HashSet::new();
        for (term, freq) in term_freqs {
            self.inverted_index.entry(term.clone()).or_default().push((chunk.id, freq));

            if seen_terms.insert(term.clone()) {
                *self.doc_freqs.entry(term).or_insert(0) += 1;
            }
        }

        self.update_avg_doc_length();
    }

    fn add_batch(&mut self, chunks: &[Chunk]) {
        for chunk in chunks {
            self.add(chunk);
        }
    }

    fn search(&self, query: &str, k: usize) -> Vec<(ChunkId, f32)> {
        let query_terms = self.tokenize(query);
        if query_terms.is_empty() {
            return Vec::new();
        }

        // Collect candidate documents
        let mut candidates: HashSet<ChunkId> = HashSet::new();
        for term in &query_terms {
            for chunk_id in self.get_chunks_for_term(term) {
                candidates.insert(chunk_id);
            }
        }

        // Score candidates
        let mut scores: Vec<(ChunkId, f32)> = candidates
            .into_iter()
            .map(|chunk_id| {
                let score: f32 =
                    query_terms.iter().map(|term| self.score_term(term, chunk_id)).sum();
                (chunk_id, score)
            })
            .filter(|(_, score)| *score > 0.0)
            .collect();

        // Sort by score descending
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(k);
        scores
    }

    fn remove(&mut self, chunk_id: ChunkId) {
        // Remove from document lengths
        if self.doc_lengths.remove(&chunk_id).is_some() {
            self.doc_count = self.doc_count.saturating_sub(1);
        }

        // Remove from inverted index
        let mut terms_to_remove: Vec<String> = Vec::new();
        for (term, postings) in &mut self.inverted_index {
            let original_len = postings.len();
            postings.retain(|(id, _)| *id != chunk_id);

            if postings.len() < original_len {
                // Document contained this term
                if let Some(df) = self.doc_freqs.get_mut(term) {
                    *df = df.saturating_sub(1);
                    if *df == 0 {
                        terms_to_remove.push(term.clone());
                    }
                }
            }
        }

        // Clean up empty terms
        for term in terms_to_remove {
            self.inverted_index.remove(&term);
            self.doc_freqs.remove(&term);
        }

        self.update_avg_doc_length();
    }

    fn len(&self) -> usize {
        self.doc_count as usize
    }
}

/// Vector store configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorStoreConfig {
    /// Embedding dimension
    pub dimension: usize,
    /// Distance metric
    pub metric: DistanceMetric,
    /// HNSW M parameter (connections per node)
    pub hnsw_m: usize,
    /// HNSW ef_construction parameter
    pub hnsw_ef_construction: usize,
    /// HNSW ef_search parameter
    pub hnsw_ef_search: usize,
}

impl Default for VectorStoreConfig {
    fn default() -> Self {
        Self {
            dimension: DEFAULT_EMBEDDING_DIM,
            metric: DistanceMetric::Cosine,
            hnsw_m: 16,
            hnsw_ef_construction: 100,
            hnsw_ef_search: 50,
        }
    }
}

/// Distance metric for vector search
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum DistanceMetric {
    /// Cosine similarity
    #[default]
    Cosine,
    /// Euclidean distance
    Euclidean,
    /// Dot product
    DotProduct,
}

/// Vector store for dense retrieval
#[derive(Debug, Clone)]
pub struct VectorStore {
    /// Configuration
    config: VectorStoreConfig,
    /// Stored vectors: chunk_id -> embedding
    vectors: HashMap<ChunkId, Vec<f32>>,
    /// Chunk content cache: chunk_id -> content
    chunks: HashMap<ChunkId, Chunk>,
}

impl VectorStore {
    /// Create a new vector store
    #[must_use]
    pub fn new(config: VectorStoreConfig) -> Self {
        Self { config, vectors: HashMap::new(), chunks: HashMap::new() }
    }

    /// Create with default configuration
    #[must_use]
    pub fn with_dimension(dimension: usize) -> Self {
        Self::new(VectorStoreConfig { dimension, ..Default::default() })
    }

    /// Get the configuration
    #[must_use]
    pub fn config(&self) -> &VectorStoreConfig {
        &self.config
    }

    /// Insert a chunk with its embedding
    pub fn insert(&mut self, chunk: Chunk) -> Result<()> {
        let embedding = chunk
            .embedding
            .as_ref()
            .ok_or_else(|| Error::InvalidConfig("chunk must have embedding".to_string()))?;

        if embedding.len() != self.config.dimension {
            return Err(Error::DimensionMismatch {
                expected: self.config.dimension,
                actual: embedding.len(),
            });
        }

        self.vectors.insert(chunk.id, embedding.clone());
        self.chunks.insert(chunk.id, chunk);
        Ok(())
    }

    /// Insert multiple chunks
    pub fn insert_batch(&mut self, chunks: Vec<Chunk>) -> Result<()> {
        for chunk in chunks {
            self.insert(chunk)?;
        }
        Ok(())
    }

    /// Search for similar vectors
    pub fn search(&self, query_vector: &[f32], k: usize) -> Result<Vec<(ChunkId, f32)>> {
        if query_vector.len() != self.config.dimension {
            return Err(Error::DimensionMismatch {
                expected: self.config.dimension,
                actual: query_vector.len(),
            });
        }

        let mut scores: Vec<(ChunkId, f32)> = self
            .vectors
            .iter()
            .map(|(id, vec)| {
                let score = match self.config.metric {
                    DistanceMetric::Cosine => cosine_similarity(query_vector, vec),
                    DistanceMetric::Euclidean => -euclidean_distance(query_vector, vec),
                    DistanceMetric::DotProduct => dot_product(query_vector, vec),
                };
                (*id, score)
            })
            .collect();

        // Sort by score descending (higher is better)
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(k);

        Ok(scores)
    }

    /// Get a chunk by ID
    #[must_use]
    pub fn get(&self, chunk_id: ChunkId) -> Option<&Chunk> {
        self.chunks.get(&chunk_id)
    }

    /// Remove a chunk
    pub fn remove(&mut self, chunk_id: ChunkId) -> Option<Chunk> {
        self.vectors.remove(&chunk_id);
        self.chunks.remove(&chunk_id)
    }

    /// Get the number of stored vectors
    #[must_use]
    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    /// Check if the store is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }
}

// Helper functions from embed module
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
}

fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum::<f32>().sqrt()
}

fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DocumentId;

    fn create_test_chunk(content: &str) -> Chunk {
        Chunk::new(DocumentId::new(), content.to_string(), 0, content.len())
    }

    fn create_test_chunk_with_embedding(content: &str, embedding: Vec<f32>) -> Chunk {
        let mut chunk = create_test_chunk(content);
        chunk.set_embedding(embedding);
        chunk
    }

    // ============ BM25Index Tests ============

    #[test]
    fn test_bm25_index_new() {
        let index = BM25Index::new();
        assert_eq!(index.len(), 0);
        assert!(index.is_empty());
        assert!((index.k1 - 1.2).abs() < 0.01);
        assert!((index.b - 0.75).abs() < 0.01);
    }

    #[test]
    fn test_bm25_index_with_params() {
        let index = BM25Index::with_params(1.5, 0.5);
        assert!((index.k1 - 1.5).abs() < 0.01);
        assert!((index.b - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_bm25_tokenize() {
        let index = BM25Index::new();
        let tokens = index.tokenize("Hello World! This is a test.");

        assert!(tokens.contains(&"hello".to_string()));
        assert!(tokens.contains(&"world".to_string()));
        assert!(tokens.contains(&"test".to_string()));
        // Stopwords should be removed
        assert!(!tokens.contains(&"this".to_string()));
        assert!(!tokens.contains(&"is".to_string()));
        assert!(!tokens.contains(&"a".to_string()));
    }

    #[test]
    fn test_bm25_tokenize_lowercase() {
        let index = BM25Index::new();
        let tokens = index.tokenize("HELLO World");
        assert!(tokens.contains(&"hello".to_string()));
        assert!(tokens.contains(&"world".to_string()));
    }

    #[test]
    fn test_bm25_add_chunk() {
        let mut index = BM25Index::new();
        let chunk = create_test_chunk("Machine learning is fascinating");

        index.add(&chunk);

        assert_eq!(index.len(), 1);
        assert!(!index.is_empty());
        assert!(index.inverted_index.contains_key("machine"));
        assert!(index.inverted_index.contains_key("learning"));
    }

    #[test]
    fn test_bm25_add_batch() {
        let mut index = BM25Index::new();
        let chunks = vec![
            create_test_chunk("First document about AI"),
            create_test_chunk("Second document about ML"),
            create_test_chunk("Third document about deep learning"),
        ];

        index.add_batch(&chunks);

        assert_eq!(index.len(), 3);
    }

    #[test]
    fn test_bm25_search_basic() {
        let mut index = BM25Index::new();
        let chunk1 = create_test_chunk("Machine learning algorithms");
        let chunk2 = create_test_chunk("Deep learning neural networks");
        let chunk3 = create_test_chunk("Natural language processing");

        index.add(&chunk1);
        index.add(&chunk2);
        index.add(&chunk3);

        let results = index.search("machine learning", 10);

        assert!(!results.is_empty());
        // Chunk with "machine learning" should score highest
        assert!(results.iter().any(|(id, _)| *id == chunk1.id));
    }

    #[test]
    fn test_bm25_search_empty_query() {
        let mut index = BM25Index::new();
        index.add(&create_test_chunk("Test document"));

        let results = index.search("", 10);
        assert!(results.is_empty());
    }

    #[test]
    fn test_bm25_search_stopwords_only() {
        let mut index = BM25Index::new();
        index.add(&create_test_chunk("Test document"));

        let results = index.search("the a an", 10);
        assert!(results.is_empty());
    }

    #[test]
    fn test_bm25_search_no_match() {
        let mut index = BM25Index::new();
        index.add(&create_test_chunk("Cats and dogs"));

        let results = index.search("quantum physics", 10);
        assert!(results.is_empty());
    }

    #[test]
    fn test_bm25_search_ranking() {
        let mut index = BM25Index::new();

        // Document with more term matches should rank higher
        let chunk1 = create_test_chunk("python programming language");
        let chunk2 = create_test_chunk("python python python programming");

        index.add(&chunk1);
        index.add(&chunk2);

        let results = index.search("python programming", 10);

        assert_eq!(results.len(), 2);
        // Chunk2 should rank higher due to more "python" occurrences
        assert_eq!(results[0].0, chunk2.id);
    }

    #[test]
    fn test_bm25_search_top_k() {
        let mut index = BM25Index::new();
        for i in 0..10 {
            index.add(&create_test_chunk(&format!("document {i} about rust")));
        }

        let results = index.search("rust", 3);
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_bm25_remove() {
        let mut index = BM25Index::new();
        let chunk = create_test_chunk("Test document");
        let chunk_id = chunk.id;

        index.add(&chunk);
        assert_eq!(index.len(), 1);

        index.remove(chunk_id);
        assert_eq!(index.len(), 0);

        let results = index.search("test", 10);
        assert!(results.is_empty());
    }

    #[test]
    fn test_bm25_avg_doc_length() {
        let mut index = BM25Index::new();

        index.add(&create_test_chunk("short text")); // ~2 tokens
        index.add(&create_test_chunk("this is a longer piece of text about programming")); // ~5 tokens

        assert!(index.avg_doc_length > 0.0);
    }

    #[test]
    fn test_bm25_idf_calculation() {
        let mut index = BM25Index::new();

        // Add documents with varying term frequencies
        index.add(&create_test_chunk("common rare"));
        index.add(&create_test_chunk("common word"));
        index.add(&create_test_chunk("common term"));

        // Search for rare term should give higher score
        let rare_results = index.search("rare", 10);
        let common_results = index.search("common", 10);

        // "rare" appears in 1 doc, "common" in 3 docs
        // IDF of "rare" should be higher
        assert!(!rare_results.is_empty());
        assert!(!common_results.is_empty());
    }

    // ============ VectorStore Tests ============

    #[test]
    fn test_vector_store_new() {
        let store = VectorStore::with_dimension(384);
        assert_eq!(store.config().dimension, 384);
        assert!(store.is_empty());
    }

    #[test]
    fn test_vector_store_config() {
        let config = VectorStoreConfig {
            dimension: 768,
            metric: DistanceMetric::DotProduct,
            hnsw_m: 32,
            hnsw_ef_construction: 200,
            hnsw_ef_search: 100,
        };
        let store = VectorStore::new(config.clone());

        assert_eq!(store.config().dimension, 768);
        assert_eq!(store.config().metric, DistanceMetric::DotProduct);
    }

    #[test]
    fn test_vector_store_insert() {
        let mut store = VectorStore::with_dimension(3);
        let chunk = create_test_chunk_with_embedding("test", vec![1.0, 0.0, 0.0]);

        store.insert(chunk.clone()).unwrap();

        assert_eq!(store.len(), 1);
        assert!(!store.is_empty());
        assert!(store.get(chunk.id).is_some());
    }

    #[test]
    fn test_vector_store_insert_no_embedding() {
        let mut store = VectorStore::with_dimension(3);
        let chunk = create_test_chunk("no embedding");

        let result = store.insert(chunk);
        assert!(result.is_err());
    }

    #[test]
    fn test_vector_store_insert_wrong_dimension() {
        let mut store = VectorStore::with_dimension(3);
        let chunk = create_test_chunk_with_embedding("test", vec![1.0, 0.0]); // Wrong dimension

        let result = store.insert(chunk);
        assert!(result.is_err());
        match result {
            Err(Error::DimensionMismatch { expected, actual }) => {
                assert_eq!(expected, 3);
                assert_eq!(actual, 2);
            }
            _ => panic!("Expected DimensionMismatch error"),
        }
    }

    #[test]
    fn test_vector_store_insert_batch() {
        let mut store = VectorStore::with_dimension(3);
        let chunks = vec![
            create_test_chunk_with_embedding("a", vec![1.0, 0.0, 0.0]),
            create_test_chunk_with_embedding("b", vec![0.0, 1.0, 0.0]),
            create_test_chunk_with_embedding("c", vec![0.0, 0.0, 1.0]),
        ];

        store.insert_batch(chunks).unwrap();
        assert_eq!(store.len(), 3);
    }

    #[test]
    fn test_vector_store_search_cosine() {
        let mut store = VectorStore::with_dimension(3);

        let chunk1 = create_test_chunk_with_embedding("north", vec![1.0, 0.0, 0.0]);
        let chunk2 = create_test_chunk_with_embedding("east", vec![0.0, 1.0, 0.0]);
        let chunk3 = create_test_chunk_with_embedding(
            "diagonal",
            vec![std::f32::consts::FRAC_1_SQRT_2, std::f32::consts::FRAC_1_SQRT_2, 0.0],
        );

        let id1 = chunk1.id;
        let id3 = chunk3.id;

        store.insert(chunk1).unwrap();
        store.insert(chunk2).unwrap();
        store.insert(chunk3).unwrap();

        // Search for vector pointing mostly north
        let query = vec![0.9, 0.1, 0.0];
        let results = store.search(&query, 10).unwrap();

        assert_eq!(results.len(), 3);
        // chunk1 (north) should be most similar
        assert_eq!(results[0].0, id1);
        // chunk3 (diagonal) should be second
        assert_eq!(results[1].0, id3);
    }

    #[test]
    fn test_vector_store_search_top_k() {
        let mut store = VectorStore::with_dimension(3);

        for i in 0..10 {
            let embedding = vec![i as f32, 0.0, 0.0];
            store
                .insert(create_test_chunk_with_embedding(&format!("chunk {i}"), embedding))
                .unwrap();
        }

        let results = store.search(&[9.0, 0.0, 0.0], 3).unwrap();
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_vector_store_search_wrong_dimension() {
        let store = VectorStore::with_dimension(3);
        let result = store.search(&[1.0, 0.0], 10);
        assert!(result.is_err());
    }

    #[test]
    fn test_vector_store_remove() {
        let mut store = VectorStore::with_dimension(3);
        let chunk = create_test_chunk_with_embedding("test", vec![1.0, 0.0, 0.0]);
        let chunk_id = chunk.id;

        store.insert(chunk).unwrap();
        assert_eq!(store.len(), 1);

        let removed = store.remove(chunk_id);
        assert!(removed.is_some());
        assert_eq!(store.len(), 0);
        assert!(store.get(chunk_id).is_none());
    }

    #[test]
    fn test_vector_store_remove_nonexistent() {
        let mut store = VectorStore::with_dimension(3);
        let removed = store.remove(ChunkId::new());
        assert!(removed.is_none());
    }

    #[test]
    fn test_distance_metric_euclidean() {
        let config = VectorStoreConfig {
            dimension: 2,
            metric: DistanceMetric::Euclidean,
            ..Default::default()
        };
        let mut store = VectorStore::new(config);

        let chunk1 = create_test_chunk_with_embedding("origin", vec![0.0, 0.0]);
        let chunk2 = create_test_chunk_with_embedding("near", vec![1.0, 0.0]);
        let chunk3 = create_test_chunk_with_embedding("far", vec![10.0, 0.0]);

        let id2 = chunk2.id;
        let id1 = chunk1.id;

        store.insert(chunk1).unwrap();
        store.insert(chunk2).unwrap();
        store.insert(chunk3).unwrap();

        // Search from origin - near should be closest
        let results = store.search(&[0.0, 0.0], 10).unwrap();
        assert_eq!(results[0].0, id1); // Exact match
        assert_eq!(results[1].0, id2); // Nearest neighbor
    }

    #[test]
    fn test_distance_metric_dot_product() {
        let config = VectorStoreConfig {
            dimension: 2,
            metric: DistanceMetric::DotProduct,
            ..Default::default()
        };
        let mut store = VectorStore::new(config);

        let chunk1 = create_test_chunk_with_embedding("small", vec![1.0, 0.0]);
        let chunk2 = create_test_chunk_with_embedding("large", vec![10.0, 0.0]);

        let id2 = chunk2.id;

        store.insert(chunk1).unwrap();
        store.insert(chunk2).unwrap();

        // Dot product prefers larger magnitude vectors
        let results = store.search(&[1.0, 0.0], 10).unwrap();
        assert_eq!(results[0].0, id2);
    }

    // ============ Property-Based Tests ============

    use proptest::prelude::*;

    proptest! {
        #[test]
        fn prop_bm25_add_increases_count(content in "[a-zA-Z ]{10,100}") {
            let mut index = BM25Index::new();
            let initial = index.len();
            index.add(&create_test_chunk(&content));
            prop_assert_eq!(index.len(), initial + 1);
        }

        #[test]
        fn prop_bm25_search_results_within_k(
            content in prop::collection::vec("[a-zA-Z]{3,10}", 5..20),
            k in 1usize..10
        ) {
            let mut index = BM25Index::new();
            for c in &content {
                index.add(&create_test_chunk(c));
            }

            let results = index.search("test", k);
            prop_assert!(results.len() <= k);
        }

        #[test]
        fn prop_bm25_scores_non_negative(
            docs in prop::collection::vec("[a-zA-Z ]{5,50}", 3..10),
            query in "[a-zA-Z]{3,10}"
        ) {
            let mut index = BM25Index::new();
            for doc in &docs {
                index.add(&create_test_chunk(doc));
            }

            let results = index.search(&query, 100);
            for (_, score) in results {
                prop_assert!(score >= 0.0);
            }
        }

        #[test]
        fn prop_vector_store_search_returns_stored(
            dim in 2usize..10,
            n_chunks in 1usize..20
        ) {
            let mut store = VectorStore::with_dimension(dim);
            let mut ids = Vec::new();

            for i in 0..n_chunks {
                let mut embedding = vec![0.0f32; dim];
                embedding[i % dim] = 1.0;
                let chunk = create_test_chunk_with_embedding(&format!("chunk {i}"), embedding);
                ids.push(chunk.id);
                store.insert(chunk).unwrap();
            }

            let query = vec![1.0f32; dim];
            let results = store.search(&query, n_chunks).unwrap();

            // All results should be from stored chunks
            for (id, _) in results {
                prop_assert!(ids.contains(&id));
            }
        }
    }
}
