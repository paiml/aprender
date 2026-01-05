//! Indexing data structures for efficient nearest neighbor search.
//!
//! This module provides approximate nearest neighbor search algorithms
//! optimized for production ML workloads.
//!
//! # Algorithms
//!
//! - **HNSW** (Hierarchical Navigable Small World): O(log n) approximate search
//!
//! # Quick Start
//!
//! ```
//! use aprender::index::hnsw::HNSWIndex;
//! use aprender::primitives::Vector;
//!
//! // Create index with M=16 connections per node
//! let mut index = HNSWIndex::new(16, 200, 0.0);
//!
//! // Add vectors at different angles (cosine distance measures angle)
//! index.add("horizontal", Vector::from_slice(&[1.0, 0.0, 0.0]));
//! index.add("diagonal", Vector::from_slice(&[1.0, 1.0, 0.0]));
//! index.add("vertical", Vector::from_slice(&[0.0, 1.0, 0.0]));
//!
//! // Search for 2 nearest neighbors to nearly horizontal vector
//! let query = Vector::from_slice(&[0.9, 0.1, 0.0]);
//! let results = index.search(&query, 2);
//!
//! assert_eq!(results.len(), 2);
//! // Results are sorted by cosine distance (closest first)
//! assert!(results[0].1 <= results[1].1);
//! ```

pub mod hnsw;

pub use hnsw::HNSWIndex;

/// Cross-Encoder for reranking search results.
///
/// Takes (query, document) pairs and produces relevance scores.
/// More accurate than bi-encoders but slower (can't pre-compute).
#[derive(Debug, Clone)]
pub struct CrossEncoder<F> {
    score_fn: F,
}

impl<F> CrossEncoder<F>
where
    F: Fn(&[f32], &[f32]) -> f32,
{
    /// Create cross-encoder with custom scoring function.
    pub fn new(score_fn: F) -> Self {
        Self { score_fn }
    }

    /// Score a single (query, document) pair.
    pub fn score(&self, query: &[f32], document: &[f32]) -> f32 {
        (self.score_fn)(query, document)
    }

    /// Rerank candidates by cross-encoder score.
    pub fn rerank<'a, T>(
        &self,
        query: &[f32],
        candidates: &'a [(T, Vec<f32>)],
        top_k: usize,
    ) -> Vec<(&'a T, f32)> {
        let mut scored: Vec<(&T, f32)> = candidates
            .iter()
            .map(|(id, doc)| (id, self.score(query, doc)))
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k);
        scored
    }
}

/// Default cross-encoder using cosine similarity.
#[must_use] 
pub fn default_cross_encoder() -> CrossEncoder<impl Fn(&[f32], &[f32]) -> f32> {
    CrossEncoder::new(|q, d| {
        let dot: f32 = q.iter().zip(d).map(|(&a, &b)| a * b).sum();
        let nq: f32 = q.iter().map(|&x| x * x).sum::<f32>().sqrt();
        let nd: f32 = d.iter().map(|&x| x * x).sum::<f32>().sqrt();
        dot / (nq * nd + 1e-10)
    })
}

/// Hybrid Search combining dense and sparse retrieval.
#[derive(Debug, Clone)]
pub struct HybridSearch {
    /// Weight for dense (semantic) scores
    dense_weight: f32,
    /// Weight for sparse (lexical) scores
    sparse_weight: f32,
}

impl HybridSearch {
    /// Create hybrid search with dense/sparse weights.
    #[must_use] 
    pub fn new(dense_weight: f32, sparse_weight: f32) -> Self {
        Self {
            dense_weight,
            sparse_weight,
        }
    }

    /// Fuse dense and sparse scores using linear combination.
    pub fn fuse_scores(
        &self,
        dense_results: &[(String, f32)],
        sparse_results: &[(String, f32)],
        top_k: usize,
    ) -> Vec<(String, f32)> {
        use std::collections::HashMap;

        let mut scores: HashMap<String, f32> = HashMap::new();

        // Normalize and add dense scores
        let dense_max = dense_results
            .iter()
            .map(|(_, s)| *s)
            .fold(0.0_f32, f32::max);
        for (id, score) in dense_results {
            let norm = if dense_max > 0.0 {
                score / dense_max
            } else {
                0.0
            };
            *scores.entry(id.clone()).or_insert(0.0) += self.dense_weight * norm;
        }

        // Normalize and add sparse scores
        let sparse_max = sparse_results
            .iter()
            .map(|(_, s)| *s)
            .fold(0.0_f32, f32::max);
        for (id, score) in sparse_results {
            let norm = if sparse_max > 0.0 {
                score / sparse_max
            } else {
                0.0
            };
            *scores.entry(id.clone()).or_insert(0.0) += self.sparse_weight * norm;
        }

        let mut results: Vec<_> = scores.into_iter().collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(top_k);
        results
    }

    /// Reciprocal Rank Fusion (RRF) for combining rankings.
    #[must_use] 
    pub fn rrf_fuse(&self, rankings: &[Vec<String>], k: f32, top_n: usize) -> Vec<(String, f32)> {
        use std::collections::HashMap;

        let mut scores: HashMap<String, f32> = HashMap::new();

        for ranking in rankings {
            for (rank, id) in ranking.iter().enumerate() {
                *scores.entry(id.clone()).or_insert(0.0) += 1.0 / (k + rank as f32 + 1.0);
            }
        }

        let mut results: Vec<_> = scores.into_iter().collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(top_n);
        results
    }

    #[must_use] 
    pub fn dense_weight(&self) -> f32 {
        self.dense_weight
    }
    #[must_use] 
    pub fn sparse_weight(&self) -> f32 {
        self.sparse_weight
    }
}

impl Default for HybridSearch {
    fn default() -> Self {
        Self::new(0.7, 0.3) // Default: 70% dense, 30% sparse
    }
}

/// Bi-Encoder for efficient dense retrieval.
///
/// Encodes queries and documents separately, allowing pre-computation
/// of document embeddings for fast retrieval.
///
/// # Architecture
///
/// ```text
/// Query  ─┬─> Encoder ─> Query Embedding  ─┐
///         │                                 ├─> Similarity Score
/// Document ─> Encoder ─> Doc Embedding   ─┘
/// ```
#[derive(Debug)]
pub struct BiEncoder<F> {
    encode_fn: F,
    similarity: SimilarityMetric,
}

/// Similarity metric for comparing embeddings.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SimilarityMetric {
    Cosine,
    DotProduct,
    Euclidean,
}

impl<F> BiEncoder<F>
where
    F: Fn(&[f32]) -> Vec<f32>,
{
    /// Create bi-encoder with custom encoding function.
    pub fn new(encode_fn: F, similarity: SimilarityMetric) -> Self {
        Self {
            encode_fn,
            similarity,
        }
    }

    /// Encode a single input.
    pub fn encode(&self, input: &[f32]) -> Vec<f32> {
        (self.encode_fn)(input)
    }

    /// Encode a batch of inputs.
    pub fn encode_batch(&self, inputs: &[Vec<f32>]) -> Vec<Vec<f32>> {
        inputs.iter().map(|x| self.encode(x)).collect()
    }

    /// Compute similarity between two embeddings.
    pub fn similarity(&self, a: &[f32], b: &[f32]) -> f32 {
        match self.similarity {
            SimilarityMetric::Cosine => {
                let dot: f32 = a.iter().zip(b).map(|(&x, &y)| x * y).sum();
                let na: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
                let nb: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
                dot / (na * nb + 1e-10)
            }
            SimilarityMetric::DotProduct => a.iter().zip(b).map(|(&x, &y)| x * y).sum(),
            SimilarityMetric::Euclidean => {
                let dist_sq: f32 = a.iter().zip(b).map(|(&x, &y)| (x - y).powi(2)).sum();
                -dist_sq.sqrt() // Negative for sorting (higher = more similar)
            }
        }
    }

    /// Retrieve top-k most similar documents.
    pub fn retrieve<T: Clone>(
        &self,
        query: &[f32],
        corpus: &[(T, Vec<f32>)],
        top_k: usize,
    ) -> Vec<(T, f32)> {
        let query_emb = self.encode(query);
        let mut scores: Vec<(T, f32)> = corpus
            .iter()
            .map(|(id, doc_emb)| (id.clone(), self.similarity(&query_emb, doc_emb)))
            .collect();

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(top_k);
        scores
    }
}

/// ColBERT-style late interaction retrieval.
///
/// Computes fine-grained token-level interactions between queries and documents,
/// using `MaxSim` (maximum similarity per query token).
///
/// # Architecture
///
/// ```text
/// Query Tokens  ─> [q1, q2, ..., qn]  ─┐
///                                       ├─> MaxSim aggregation ─> Score
/// Doc Tokens    ─> [d1, d2, ..., dm]  ─┘
/// ```
///
/// # Reference
///
/// Khattab, O., & Zaharia, M. (2020). `ColBERT`: Efficient and Effective Passage
/// Search via Contextualized Late Interaction over BERT. SIGIR.
#[derive(Debug)]
pub struct ColBERT {
    embedding_dim: usize,
}

impl ColBERT {
    /// Create `ColBERT` with specified embedding dimension.
    #[must_use] 
    pub fn new(embedding_dim: usize) -> Self {
        Self { embedding_dim }
    }

    /// Compute `MaxSim` score between query and document token embeddings.
    ///
    /// For each query token, finds maximum similarity with any doc token,
    /// then sums across query tokens.
    pub fn maxsim(&self, query_tokens: &[Vec<f32>], doc_tokens: &[Vec<f32>]) -> f32 {
        if query_tokens.is_empty() || doc_tokens.is_empty() {
            return 0.0;
        }

        let mut total = 0.0_f32;

        for q in query_tokens {
            let max_sim = doc_tokens
                .iter()
                .map(|d| cosine_sim(q, d))
                .fold(f32::NEG_INFINITY, f32::max);
            total += max_sim;
        }

        total
    }

    /// Score a batch of documents against a query.
    #[must_use] 
    pub fn score_documents(
        &self,
        query_tokens: &[Vec<f32>],
        documents: &[Vec<Vec<f32>>],
    ) -> Vec<f32> {
        documents
            .iter()
            .map(|doc| self.maxsim(query_tokens, doc))
            .collect()
    }

    /// Retrieve top-k documents using `MaxSim`.
    pub fn retrieve<T: Clone>(
        &self,
        query_tokens: &[Vec<f32>],
        corpus: &[(T, Vec<Vec<f32>>)],
        top_k: usize,
    ) -> Vec<(T, f32)> {
        let mut scores: Vec<(T, f32)> = corpus
            .iter()
            .map(|(id, doc_tokens)| (id.clone(), self.maxsim(query_tokens, doc_tokens)))
            .collect();

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(top_k);
        scores
    }

    #[must_use] 
    pub fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }
}

fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(&x, &y)| x * y).sum();
    let na: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
    dot / (na * nb + 1e-10)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cross_encoder_score() {
        let ce = default_cross_encoder();
        let query = vec![1.0, 0.0];
        let doc1 = vec![1.0, 0.0];
        let doc2 = vec![0.0, 1.0];

        let score1 = ce.score(&query, &doc1);
        let score2 = ce.score(&query, &doc2);
        assert!(score1 > score2);
    }

    #[test]
    fn test_cross_encoder_rerank() {
        let ce = default_cross_encoder();
        let query = vec![1.0, 0.0];
        let candidates = vec![
            ("doc1", vec![0.0, 1.0]),
            ("doc2", vec![1.0, 0.0]),
            ("doc3", vec![0.5, 0.5]),
        ];

        let reranked = ce.rerank(&query, &candidates, 2);
        assert_eq!(reranked.len(), 2);
        assert_eq!(*reranked[0].0, "doc2");
    }

    #[test]
    fn test_hybrid_search_fuse() {
        let hs = HybridSearch::new(0.6, 0.4);
        let dense = vec![("a".to_string(), 0.9), ("b".to_string(), 0.5)];
        let sparse = vec![("b".to_string(), 1.0), ("c".to_string(), 0.7)];

        let fused = hs.fuse_scores(&dense, &sparse, 3);
        assert!(fused.len() <= 3);
    }

    #[test]
    fn test_hybrid_search_rrf() {
        let hs = HybridSearch::default();
        let rankings = vec![
            vec!["a".to_string(), "b".to_string(), "c".to_string()],
            vec!["b".to_string(), "a".to_string(), "d".to_string()],
        ];

        let fused = hs.rrf_fuse(&rankings, 60.0, 3);
        assert_eq!(fused.len(), 3);
    }

    #[test]
    fn test_hybrid_search_default() {
        let hs = HybridSearch::default();
        assert!((hs.dense_weight() - 0.7).abs() < 1e-6);
        assert!((hs.sparse_weight() - 0.3).abs() < 1e-6);
    }

    // BiEncoder Tests

    #[test]
    fn test_bi_encoder_cosine() {
        let encoder = BiEncoder::new(|x: &[f32]| x.to_vec(), SimilarityMetric::Cosine);
        let a = vec![1.0, 0.0];
        let b = vec![1.0, 0.0];
        let c = vec![0.0, 1.0];

        let sim_ab = encoder.similarity(&a, &b);
        let sim_ac = encoder.similarity(&a, &c);

        assert!((sim_ab - 1.0).abs() < 1e-6); // Same direction
        assert!(sim_ac.abs() < 1e-6); // Orthogonal
    }

    #[test]
    fn test_bi_encoder_dot_product() {
        let encoder = BiEncoder::new(|x: &[f32]| x.to_vec(), SimilarityMetric::DotProduct);
        let a = vec![2.0, 3.0];
        let b = vec![1.0, 2.0];

        let sim = encoder.similarity(&a, &b);
        assert!((sim - 8.0).abs() < 1e-6); // 2*1 + 3*2 = 8
    }

    #[test]
    fn test_bi_encoder_euclidean() {
        let encoder = BiEncoder::new(|x: &[f32]| x.to_vec(), SimilarityMetric::Euclidean);
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];

        let sim = encoder.similarity(&a, &b);
        assert!((sim - (-5.0)).abs() < 1e-6); // -sqrt(9+16) = -5
    }

    #[test]
    fn test_bi_encoder_encode() {
        let encoder = BiEncoder::new(
            |x: &[f32]| x.iter().map(|&v| v * 2.0).collect(),
            SimilarityMetric::Cosine,
        );

        let input = vec![1.0, 2.0, 3.0];
        let encoded = encoder.encode(&input);

        assert_eq!(encoded, vec![2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_bi_encoder_encode_batch() {
        let encoder = BiEncoder::new(|x: &[f32]| x.to_vec(), SimilarityMetric::Cosine);
        let inputs = vec![vec![1.0, 0.0], vec![0.0, 1.0]];

        let encoded = encoder.encode_batch(&inputs);
        assert_eq!(encoded.len(), 2);
    }

    #[test]
    fn test_bi_encoder_retrieve() {
        let encoder = BiEncoder::new(|x: &[f32]| x.to_vec(), SimilarityMetric::Cosine);
        let corpus = vec![
            ("doc1", vec![1.0, 0.0]),
            ("doc2", vec![0.0, 1.0]),
            ("doc3", vec![0.707, 0.707]),
        ];

        let query = vec![1.0, 0.0];
        let results = encoder.retrieve(&query, &corpus, 2);

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, "doc1"); // Exact match
    }

    // ColBERT Tests

    #[test]
    fn test_colbert_creation() {
        let colbert = ColBERT::new(128);
        assert_eq!(colbert.embedding_dim(), 128);
    }

    #[test]
    fn test_colbert_maxsim_identical() {
        let colbert = ColBERT::new(4);
        let query = vec![vec![1.0, 0.0, 0.0, 0.0], vec![0.0, 1.0, 0.0, 0.0]];
        let doc = query.clone();

        let score = colbert.maxsim(&query, &doc);
        assert!((score - 2.0).abs() < 1e-5); // 1.0 + 1.0
    }

    #[test]
    fn test_colbert_maxsim_different() {
        let colbert = ColBERT::new(4);
        let query = vec![vec![1.0, 0.0, 0.0, 0.0]];
        let doc = vec![vec![0.0, 1.0, 0.0, 0.0]];

        let score = colbert.maxsim(&query, &doc);
        assert!(score.abs() < 1e-5); // Orthogonal
    }

    #[test]
    fn test_colbert_maxsim_empty() {
        let colbert = ColBERT::new(4);
        let empty: Vec<Vec<f32>> = vec![];
        let doc = vec![vec![1.0, 0.0, 0.0, 0.0]];

        assert_eq!(colbert.maxsim(&empty, &doc), 0.0);
        assert_eq!(colbert.maxsim(&doc, &empty), 0.0);
    }

    #[test]
    fn test_colbert_score_documents() {
        let colbert = ColBERT::new(2);
        let query = vec![vec![1.0, 0.0]];
        let docs = vec![vec![vec![1.0, 0.0]], vec![vec![0.0, 1.0]]];

        let scores = colbert.score_documents(&query, &docs);
        assert_eq!(scores.len(), 2);
        assert!(scores[0] > scores[1]); // First doc matches better
    }

    #[test]
    fn test_colbert_retrieve() {
        let colbert = ColBERT::new(2);
        let query = vec![vec![1.0, 0.0], vec![0.707, 0.707]];
        let corpus = vec![
            ("doc1", vec![vec![1.0, 0.0], vec![0.0, 1.0]]),
            ("doc2", vec![vec![0.0, 1.0]]),
        ];

        let results = colbert.retrieve(&query, &corpus, 2);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_similarity_metric_equality() {
        assert_eq!(SimilarityMetric::Cosine, SimilarityMetric::Cosine);
        assert_ne!(SimilarityMetric::Cosine, SimilarityMetric::DotProduct);
    }
}
