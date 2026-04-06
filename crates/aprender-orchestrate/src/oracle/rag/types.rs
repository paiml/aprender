//! RAG Oracle Types
//!
//! Core types for the retrieval-augmented generation oracle.

use serde::{Deserialize, Serialize};
use std::cmp::Ordering;

/// Query result from RAG retrieval
#[derive(Debug, Clone)]
pub struct QueryResult {
    /// Retrieved documents with scores
    pub results: Vec<RetrievalResult>,
    /// Query latency in milliseconds
    pub latency_ms: u64,
    /// Total documents searched
    pub docs_searched: usize,
}

/// A single retrieval result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievalResult {
    /// Document/chunk ID
    pub id: String,
    /// Source component (e.g., "trueno", "aprender")
    pub component: String,
    /// Source file path
    pub source: String,
    /// Content snippet
    pub content: String,
    /// Relevance score (0.0 - 1.0)
    pub score: f64,
    /// Start line in source
    pub start_line: usize,
    /// End line in source
    pub end_line: usize,
    /// Score breakdown
    pub score_breakdown: ScoreBreakdown,
}

impl PartialEq for RetrievalResult {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for RetrievalResult {}

impl PartialOrd for RetrievalResult {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for RetrievalResult {
    fn cmp(&self, other: &Self) -> Ordering {
        // Higher scores first
        other.score.partial_cmp(&self.score).unwrap_or(Ordering::Equal)
    }
}

/// Score breakdown for transparency
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ScoreBreakdown {
    /// BM25 sparse retrieval score
    pub bm25_score: f64,
    /// Dense embedding similarity score
    pub dense_score: f64,
    /// RRF fusion score
    pub rrf_score: f64,
    /// Cross-encoder rerank score (if applied)
    pub rerank_score: Option<f64>,
}

/// BM25 configuration (Robertson-Walker parameters)
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Bm25Config {
    /// Term frequency saturation (typically 1.2-2.0)
    pub k1: f32,
    /// Length normalization (typically 0.75)
    pub b: f32,
}

impl Default for Bm25Config {
    fn default() -> Self {
        // Tuned for technical documentation per Trotman et al. (2014)
        Self { k1: 1.5, b: 0.75 }
    }
}

/// Reciprocal Rank Fusion parameters
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct RrfConfig {
    /// RRF constant k (typically 60)
    pub k: usize,
}

impl Default for RrfConfig {
    fn default() -> Self {
        // Standard RRF k=60 from Cormack et al. (2009)
        Self { k: 60 }
    }
}

/// Index health metrics for monitoring
#[derive(Debug, Clone, Default)]
pub struct IndexHealthMetrics {
    /// Coverage percentage (0-100)
    pub coverage_percent: u16,
    /// Documents per component
    pub docs_per_component: Vec<(String, usize)>,
    /// Component names for display
    pub component_names: Vec<String>,
    /// Query latency samples (ms)
    pub latency_samples: Vec<u64>,
    /// MRR history for quality tracking
    pub mrr_history: Vec<f64>,
    /// NDCG history for quality tracking
    pub ndcg_history: Vec<f64>,
    /// Index freshness score (0-100)
    pub freshness_score: f64,
}

/// Retrieval metrics for evaluation (IR standard metrics)
#[derive(Debug, Clone, Default)]
pub struct RelevanceMetrics {
    /// Mean Reciprocal Rank
    pub mrr: f64,
    /// Normalized Discounted Cumulative Gain at K
    pub ndcg_at_k: f64,
    /// Recall at K
    pub recall_at_k: f64,
    /// Precision at K
    pub precision_at_k: f64,
}

impl RelevanceMetrics {
    /// Calculate MRR from ranked results and relevance judgments
    pub fn calculate_mrr(results: &[&str], relevant: &[&str]) -> f64 {
        for (rank, result) in results.iter().enumerate() {
            if relevant.contains(result) {
                return 1.0 / (rank + 1) as f64;
            }
        }
        0.0
    }

    /// Calculate Recall@K
    pub fn calculate_recall_at_k(results: &[&str], relevant: &[&str], k: usize) -> f64 {
        if relevant.is_empty() {
            return 0.0;
        }
        let hits = results.iter().take(k).filter(|r| relevant.contains(r)).count();
        hits as f64 / relevant.len() as f64
    }

    /// Calculate Precision@K
    pub fn calculate_precision_at_k(results: &[&str], relevant: &[&str], k: usize) -> f64 {
        if k == 0 {
            return 0.0;
        }
        let hits = results.iter().take(k).filter(|r| relevant.contains(r)).count();
        hits as f64 / k as f64
    }
}

/// Jidoka halt reasons
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum JidokaHalt {
    /// Embedding dimension mismatch
    DimensionMismatch { expected: usize, actual: usize },
    /// Corrupted embedding (NaN/Inf)
    CorruptedEmbedding { doc_id: String },
    /// Content integrity violation
    IntegrityViolation { doc_id: String },
    /// Model hash mismatch
    ModelMismatch { expected: String, actual: String },
}

impl std::fmt::Display for JidokaHalt {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DimensionMismatch { expected, actual } => {
                write!(f, "Embedding dimension mismatch: expected {}, got {}", expected, actual)
            }
            Self::CorruptedEmbedding { doc_id } => {
                write!(f, "Corrupted embedding detected in document: {}", doc_id)
            }
            Self::IntegrityViolation { doc_id } => {
                write!(f, "Content integrity violation in document: {}", doc_id)
            }
            Self::ModelMismatch { expected, actual } => {
                write!(f, "Model hash mismatch: expected {}, got {}", expected, actual)
            }
        }
    }
}

impl std::error::Error for JidokaHalt {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_retrieval_result_ordering() {
        let r1 = RetrievalResult {
            id: "1".to_string(),
            component: "test".to_string(),
            source: "test.md".to_string(),
            content: "content".to_string(),
            score: 0.9,
            start_line: 1,
            end_line: 1,
            score_breakdown: ScoreBreakdown::default(),
        };
        let r2 = RetrievalResult {
            id: "2".to_string(),
            component: "test".to_string(),
            source: "test.md".to_string(),
            content: "content".to_string(),
            score: 0.5,
            start_line: 1,
            end_line: 1,
            score_breakdown: ScoreBreakdown::default(),
        };

        // Higher score should come first
        assert!(r1 < r2); // In Ord, we reverse for descending
    }

    #[test]
    fn test_bm25_config_defaults() {
        let config = Bm25Config::default();
        assert!((config.k1 - 1.5).abs() < 0.001);
        assert!((config.b - 0.75).abs() < 0.001);
    }

    #[test]
    fn test_rrf_config_defaults() {
        let config = RrfConfig::default();
        assert_eq!(config.k, 60);
    }

    #[test]
    fn test_mrr_calculation() {
        let results = vec!["doc1", "doc2", "doc3"];
        let relevant = vec!["doc2"];

        let mrr = RelevanceMetrics::calculate_mrr(&results, &relevant);
        assert!((mrr - 0.5).abs() < 0.001); // doc2 is at rank 2, so MRR = 1/2
    }

    #[test]
    fn test_mrr_first_result() {
        let results = vec!["doc1", "doc2", "doc3"];
        let relevant = vec!["doc1"];

        let mrr = RelevanceMetrics::calculate_mrr(&results, &relevant);
        assert!((mrr - 1.0).abs() < 0.001); // doc1 is at rank 1, so MRR = 1
    }

    #[test]
    fn test_mrr_not_found() {
        let results = vec!["doc1", "doc2", "doc3"];
        let relevant = vec!["doc4"];

        let mrr = RelevanceMetrics::calculate_mrr(&results, &relevant);
        assert!((mrr - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_recall_at_k() {
        let results = vec!["doc1", "doc2", "doc3", "doc4", "doc5"];
        let relevant = vec!["doc2", "doc4", "doc6"];

        let recall = RelevanceMetrics::calculate_recall_at_k(&results, &relevant, 3);
        assert!((recall - 1.0 / 3.0).abs() < 0.001); // Only doc2 in top 3, 1/3 relevant found
    }

    #[test]
    fn test_precision_at_k() {
        let results = vec!["doc1", "doc2", "doc3", "doc4", "doc5"];
        let relevant = vec!["doc2", "doc4"];

        let precision = RelevanceMetrics::calculate_precision_at_k(&results, &relevant, 5);
        assert!((precision - 2.0 / 5.0).abs() < 0.001); // 2 relevant in top 5
    }

    #[test]
    fn test_jidoka_halt_display() {
        let halt = JidokaHalt::DimensionMismatch { expected: 384, actual: 768 };
        assert!(halt.to_string().contains("384"));
        assert!(halt.to_string().contains("768"));
    }

    #[test]
    fn test_jidoka_halt_corrupted() {
        let halt = JidokaHalt::CorruptedEmbedding { doc_id: "test_doc".to_string() };
        assert!(halt.to_string().contains("test_doc"));
    }

    #[test]
    fn test_jidoka_halt_integrity() {
        let halt = JidokaHalt::IntegrityViolation { doc_id: "integrity_doc".to_string() };
        assert!(halt.to_string().contains("integrity_doc"));
        assert!(halt.to_string().contains("integrity"));
    }

    #[test]
    fn test_jidoka_halt_model_mismatch() {
        let halt = JidokaHalt::ModelMismatch {
            expected: "abc123".to_string(),
            actual: "def456".to_string(),
        };
        let display = halt.to_string();
        assert!(display.contains("abc123"));
        assert!(display.contains("def456"));
    }

    #[test]
    fn test_jidoka_halt_error_trait() {
        let halt = JidokaHalt::DimensionMismatch { expected: 384, actual: 768 };
        // JidokaHalt implements Error trait
        let error: &dyn std::error::Error = &halt;
        assert!(!error.to_string().is_empty());
    }

    #[test]
    fn test_retrieval_result_equality() {
        let r1 = RetrievalResult {
            id: "same_id".to_string(),
            component: "test".to_string(),
            source: "test.md".to_string(),
            content: "content1".to_string(),
            score: 0.9,
            start_line: 1,
            end_line: 1,
            score_breakdown: ScoreBreakdown::default(),
        };
        let r2 = RetrievalResult {
            id: "same_id".to_string(),
            component: "different".to_string(),
            source: "different.md".to_string(),
            content: "content2".to_string(),
            score: 0.5,
            start_line: 10,
            end_line: 20,
            score_breakdown: ScoreBreakdown::default(),
        };
        // Equality is based on ID only
        assert_eq!(r1, r2);
    }

    #[test]
    fn test_retrieval_result_partial_ord() {
        let r1 = RetrievalResult {
            id: "1".to_string(),
            component: "test".to_string(),
            source: "test.md".to_string(),
            content: "content".to_string(),
            score: 0.9,
            start_line: 1,
            end_line: 1,
            score_breakdown: ScoreBreakdown::default(),
        };
        let r2 = RetrievalResult {
            id: "2".to_string(),
            component: "test".to_string(),
            source: "test.md".to_string(),
            content: "content".to_string(),
            score: 0.5,
            start_line: 1,
            end_line: 1,
            score_breakdown: ScoreBreakdown::default(),
        };

        // partial_cmp should return Some
        assert!(r1.partial_cmp(&r2).is_some());
    }

    #[test]
    fn test_score_breakdown_default() {
        let breakdown = ScoreBreakdown::default();
        assert_eq!(breakdown.bm25_score, 0.0);
        assert_eq!(breakdown.dense_score, 0.0);
        assert_eq!(breakdown.rrf_score, 0.0);
        assert!(breakdown.rerank_score.is_none());
    }

    #[test]
    fn test_recall_empty_relevant() {
        let results = vec!["doc1", "doc2"];
        let relevant: Vec<&str> = vec![];
        let recall = RelevanceMetrics::calculate_recall_at_k(&results, &relevant, 5);
        assert_eq!(recall, 0.0);
    }

    #[test]
    fn test_precision_at_zero() {
        let results = vec!["doc1", "doc2"];
        let relevant = vec!["doc1"];
        let precision = RelevanceMetrics::calculate_precision_at_k(&results, &relevant, 0);
        assert_eq!(precision, 0.0);
    }

    #[test]
    fn test_index_health_metrics_default() {
        let metrics = IndexHealthMetrics::default();
        assert_eq!(metrics.coverage_percent, 0);
        assert!(metrics.docs_per_component.is_empty());
        assert!(metrics.latency_samples.is_empty());
        assert_eq!(metrics.freshness_score, 0.0);
    }

    #[test]
    fn test_relevance_metrics_default() {
        let metrics = RelevanceMetrics::default();
        assert_eq!(metrics.mrr, 0.0);
        assert_eq!(metrics.ndcg_at_k, 0.0);
        assert_eq!(metrics.recall_at_k, 0.0);
        assert_eq!(metrics.precision_at_k, 0.0);
    }

    #[test]
    fn test_query_result_struct() {
        let result = QueryResult { results: vec![], latency_ms: 42, docs_searched: 100 };
        assert_eq!(result.latency_ms, 42);
        assert_eq!(result.docs_searched, 100);
    }

    // Property-based tests for types
    mod proptests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            #![proptest_config(ProptestConfig::with_cases(50))]

            /// Property: RetrievalResult ordering is consistent (higher score = earlier)
            #[test]
            fn prop_retrieval_result_ordering(
                score1 in 0.0f64..1.0,
                score2 in 0.0f64..1.0
            ) {
                let r1 = RetrievalResult {
                    id: "1".to_string(),
                    component: "test".to_string(),
                    source: "test.md".to_string(),
                    content: "".to_string(),
                    score: score1,
                    start_line: 1,
                    end_line: 1,
                    score_breakdown: ScoreBreakdown::default(),
                };
                let r2 = RetrievalResult {
                    id: "2".to_string(),
                    component: "test".to_string(),
                    source: "test.md".to_string(),
                    content: "".to_string(),
                    score: score2,
                    start_line: 1,
                    end_line: 1,
                    score_breakdown: ScoreBreakdown::default(),
                };

                // Verify ordering is descending by score
                use std::cmp::Ordering;
                let cmp = r1.cmp(&r2);
                if score1 > score2 {
                    prop_assert!(cmp == Ordering::Less || (score1 - score2).abs() < 1e-10);
                } else if score1 < score2 {
                    prop_assert!(cmp == Ordering::Greater || (score1 - score2).abs() < 1e-10);
                }
            }

            /// Property: MRR is in range [0, 1]
            #[test]
            fn prop_mrr_range(
                results in prop::collection::vec("[a-z]{3}", 0..10),
                relevant in prop::collection::vec("[a-z]{3}", 0..5)
            ) {
                let results_refs: Vec<&str> = results.iter().map(|s| s.as_str()).collect();
                let relevant_refs: Vec<&str> = relevant.iter().map(|s| s.as_str()).collect();
                let mrr = RelevanceMetrics::calculate_mrr(&results_refs, &relevant_refs);
                prop_assert!((0.0..=1.0).contains(&mrr), "MRR {} not in [0, 1]", mrr);
            }

            /// Property: Recall is in range [0, 1]
            #[test]
            fn prop_recall_range(
                results in prop::collection::vec("[a-z]{3}", 1..10),
                relevant in prop::collection::vec("[a-z]{3}", 1..5),
                k in 1usize..20
            ) {
                let results_refs: Vec<&str> = results.iter().map(|s| s.as_str()).collect();
                let relevant_refs: Vec<&str> = relevant.iter().map(|s| s.as_str()).collect();
                let recall = RelevanceMetrics::calculate_recall_at_k(&results_refs, &relevant_refs, k);
                prop_assert!((0.0..=1.0).contains(&recall), "Recall {} not in [0, 1]", recall);
            }

            /// Property: Precision is in range [0, 1]
            #[test]
            fn prop_precision_range(
                results in prop::collection::vec("[a-z]{3}", 1..10),
                relevant in prop::collection::vec("[a-z]{3}", 1..5),
                k in 1usize..20
            ) {
                let results_refs: Vec<&str> = results.iter().map(|s| s.as_str()).collect();
                let relevant_refs: Vec<&str> = relevant.iter().map(|s| s.as_str()).collect();
                let precision = RelevanceMetrics::calculate_precision_at_k(&results_refs, &relevant_refs, k);
                prop_assert!((0.0..=1.0).contains(&precision), "Precision {} not in [0, 1]", precision);
            }

            /// Property: BM25 config values are positive
            #[test]
            fn prop_bm25_config_positive(k1 in 0.1f32..10.0, b in 0.0f32..1.0) {
                let config = Bm25Config { k1, b };
                prop_assert!(config.k1 > 0.0);
                prop_assert!(config.b >= 0.0 && config.b <= 1.0);
            }

            /// Property: JidokaHalt display is non-empty
            #[test]
            fn prop_jidoka_halt_display_nonempty(
                expected in 1usize..1000,
                actual in 1usize..1000
            ) {
                let halt = JidokaHalt::DimensionMismatch { expected, actual };
                prop_assert!(!halt.to_string().is_empty());
            }
        }
    }
}
