//! Reranking module for RAG pipelines

use crate::{retrieve::RetrievalResult, Result};
use serde::{Deserialize, Serialize};

/// Trait for reranking retrieved results
pub trait Reranker: Send + Sync {
    /// Rerank candidates given a query
    fn rerank(
        &self,
        query: &str,
        candidates: &[RetrievalResult],
        top_k: usize,
    ) -> Result<Vec<RetrievalResult>>;
}

/// Lexical reranker using simple text matching features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LexicalReranker {
    /// Weight for exact query match
    pub exact_match_weight: f32,
    /// Weight for query term coverage
    pub coverage_weight: f32,
    /// Weight for position bias (earlier terms = better)
    pub position_weight: f32,
    /// Whether to lowercase for matching
    pub case_insensitive: bool,
}

impl Default for LexicalReranker {
    fn default() -> Self {
        Self {
            exact_match_weight: 0.3,
            coverage_weight: 0.5,
            position_weight: 0.2,
            case_insensitive: true,
        }
    }
}

impl LexicalReranker {
    /// Create a new lexical reranker
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set weights
    #[must_use]
    pub fn with_weights(mut self, exact_match: f32, coverage: f32, position: f32) -> Self {
        self.exact_match_weight = exact_match;
        self.coverage_weight = coverage;
        self.position_weight = position;
        self
    }

    /// Calculate rerank score for a single candidate
    fn score(&self, query: &str, content: &str) -> f32 {
        let (query, content) = if self.case_insensitive {
            (query.to_lowercase(), content.to_lowercase())
        } else {
            (query.to_string(), content.to_string())
        };

        let query_terms: Vec<&str> = query.split_whitespace().collect();
        if query_terms.is_empty() {
            return 0.0;
        }

        // Exact match score
        let exact_match = if content.contains(&query) { 1.0 } else { 0.0 };

        // Coverage score: what fraction of query terms appear in content
        let matches = query_terms.iter().filter(|term| content.contains(*term)).count() as f32;
        let coverage = matches / query_terms.len().max(1) as f32;

        // Position score: how early do query terms appear
        let position_score = query_terms
            .iter()
            .filter_map(|term| content.find(term))
            .map(|pos| 1.0 / (1.0 + pos as f32 / 100.0))
            .sum::<f32>()
            / query_terms.len().max(1) as f32;

        self.exact_match_weight * exact_match
            + self.coverage_weight * coverage
            + self.position_weight * position_score
    }
}

impl Reranker for LexicalReranker {
    fn rerank(
        &self,
        query: &str,
        candidates: &[RetrievalResult],
        top_k: usize,
    ) -> Result<Vec<RetrievalResult>> {
        let mut scored: Vec<(RetrievalResult, f32)> = candidates
            .iter()
            .map(|c| {
                let score = self.score(query, &c.chunk.content);
                (c.clone(), score)
            })
            .collect();

        // Sort by rerank score descending
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Take top_k and set rerank score
        Ok(scored
            .into_iter()
            .take(top_k)
            .map(|(mut result, score)| {
                result.rerank_score = Some(score);
                result
            })
            .collect())
    }
}

/// Mock cross-encoder reranker for testing
#[derive(Debug, Clone)]
pub struct MockCrossEncoderReranker {
    /// Model identifier
    model_id: String,
}

impl MockCrossEncoderReranker {
    /// Create a new mock cross-encoder
    #[must_use]
    pub fn new(model_id: impl Into<String>) -> Self {
        Self { model_id: model_id.into() }
    }

    /// Get the model ID
    #[must_use]
    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    /// Score a query-document pair (mock implementation)
    #[allow(clippy::unused_self)]
    fn score_pair(&self, query: &str, document: &str) -> f32 {
        // Simple mock: based on term overlap
        let query_lower = query.to_lowercase();
        let doc_lower = document.to_lowercase();

        let query_terms: std::collections::HashSet<&str> = query_lower.split_whitespace().collect();
        let doc_terms: std::collections::HashSet<&str> = doc_lower.split_whitespace().collect();

        if query_terms.is_empty() || doc_terms.is_empty() {
            return 0.0;
        }

        let overlap = query_terms.intersection(&doc_terms).count();
        overlap as f32 / query_terms.len() as f32
    }
}

impl Reranker for MockCrossEncoderReranker {
    fn rerank(
        &self,
        query: &str,
        candidates: &[RetrievalResult],
        top_k: usize,
    ) -> Result<Vec<RetrievalResult>> {
        let mut scored: Vec<(RetrievalResult, f32)> = candidates
            .iter()
            .map(|c| {
                let score = self.score_pair(query, &c.chunk.content);
                (c.clone(), score)
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        Ok(scored
            .into_iter()
            .take(top_k)
            .map(|(mut result, score)| {
                result.rerank_score = Some(score);
                result
            })
            .collect())
    }
}

/// Composite reranker that combines multiple rerankers
pub struct CompositeReranker {
    rerankers: Vec<(Box<dyn Reranker>, f32)>,
}

impl CompositeReranker {
    /// Create a new composite reranker
    #[must_use]
    pub fn new() -> Self {
        Self { rerankers: Vec::new() }
    }

    /// Add a reranker with a weight
    #[must_use]
    pub fn with_reranker(mut self, reranker: Box<dyn Reranker>, weight: f32) -> Self {
        self.rerankers.push((reranker, weight));
        self
    }
}

impl Default for CompositeReranker {
    fn default() -> Self {
        Self::new()
    }
}

impl Reranker for CompositeReranker {
    fn rerank(
        &self,
        query: &str,
        candidates: &[RetrievalResult],
        top_k: usize,
    ) -> Result<Vec<RetrievalResult>> {
        if self.rerankers.is_empty() {
            return Ok(candidates.iter().take(top_k).cloned().collect());
        }

        // Get scores from each reranker
        let mut combined_scores: std::collections::HashMap<usize, f32> =
            std::collections::HashMap::new();

        for (reranker, weight) in &self.rerankers {
            let reranked = reranker.rerank(query, candidates, candidates.len())?;
            for result in &reranked {
                // Find original index
                for (orig_idx, orig) in candidates.iter().enumerate() {
                    if orig.chunk.id == result.chunk.id {
                        let score = result.rerank_score.unwrap_or(0.0);
                        *combined_scores.entry(orig_idx).or_insert(0.0) += weight * score;
                        break;
                    }
                }
            }
        }

        // Sort by combined score
        let mut indexed: Vec<_> = combined_scores.into_iter().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        Ok(indexed
            .into_iter()
            .take(top_k)
            .map(|(idx, score)| {
                let mut result = candidates[idx].clone();
                result.rerank_score = Some(score);
                result
            })
            .collect())
    }
}

/// No-op reranker that just returns candidates in original order
#[derive(Debug, Clone, Default)]
pub struct NoOpReranker;

impl NoOpReranker {
    /// Create a new no-op reranker
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl Reranker for NoOpReranker {
    fn rerank(
        &self,
        _query: &str,
        candidates: &[RetrievalResult],
        top_k: usize,
    ) -> Result<Vec<RetrievalResult>> {
        Ok(candidates.iter().take(top_k).cloned().collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Chunk, DocumentId};

    fn create_result(content: &str) -> RetrievalResult {
        let chunk = Chunk::new(DocumentId::new(), content.to_string(), 0, content.len());
        RetrievalResult::new(chunk)
    }

    fn create_result_with_score(content: &str, dense: f32) -> RetrievalResult {
        create_result(content).with_dense_score(dense)
    }

    // ============ LexicalReranker Tests ============

    #[test]
    fn test_lexical_reranker_default() {
        let reranker = LexicalReranker::default();
        assert!((reranker.exact_match_weight - 0.3).abs() < 0.01);
        assert!((reranker.coverage_weight - 0.5).abs() < 0.01);
        assert!((reranker.position_weight - 0.2).abs() < 0.01);
        assert!(reranker.case_insensitive);
    }

    #[test]
    fn test_lexical_reranker_with_weights() {
        let reranker = LexicalReranker::new().with_weights(0.5, 0.3, 0.2);
        assert!((reranker.exact_match_weight - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_lexical_reranker_exact_match() {
        let reranker = LexicalReranker::new();
        let candidates = vec![
            create_result("This contains the exact query machine learning"),
            create_result("This mentions machine and learning separately"),
        ];

        let results = reranker.rerank("machine learning", &candidates, 2).unwrap();

        // Exact match should score higher
        assert!(results[0].rerank_score.unwrap() > results[1].rerank_score.unwrap());
    }

    #[test]
    fn test_lexical_reranker_coverage() {
        let reranker = LexicalReranker::new();
        let candidates =
            vec![create_result("machine learning algorithms"), create_result("machine only here")];

        let results = reranker.rerank("machine learning neural networks", &candidates, 2).unwrap();

        // First has 2 matches, second has 1
        assert!(results[0].rerank_score.unwrap() >= results[1].rerank_score.unwrap());
    }

    #[test]
    fn test_lexical_reranker_top_k() {
        let reranker = LexicalReranker::new();
        let candidates: Vec<_> = (0..10).map(|i| create_result(&format!("doc {i}"))).collect();

        let results = reranker.rerank("doc", &candidates, 3).unwrap();
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_lexical_reranker_empty_query() {
        let reranker = LexicalReranker::new();
        let candidates = vec![create_result("test content")];

        let results = reranker.rerank("", &candidates, 10).unwrap();
        assert_eq!(results.len(), 1);
        assert!((results[0].rerank_score.unwrap() - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_lexical_reranker_case_insensitive() {
        let reranker = LexicalReranker::new();
        let candidates = vec![create_result("MACHINE LEARNING"), create_result("machine learning")];

        let results = reranker.rerank("Machine Learning", &candidates, 2).unwrap();

        // Both should score the same (case insensitive)
        let diff = (results[0].rerank_score.unwrap() - results[1].rerank_score.unwrap()).abs();
        assert!(diff < 0.01);
    }

    // ============ MockCrossEncoderReranker Tests ============

    #[test]
    fn test_mock_cross_encoder_new() {
        let reranker = MockCrossEncoderReranker::new("ms-marco-MiniLM");
        assert_eq!(reranker.model_id(), "ms-marco-MiniLM");
    }

    #[test]
    fn test_mock_cross_encoder_rerank() {
        let reranker = MockCrossEncoderReranker::new("test-model");
        let candidates =
            vec![create_result("machine learning algorithms"), create_result("cooking recipes")];

        let results = reranker.rerank("machine learning", &candidates, 2).unwrap();

        // First should score higher (more term overlap)
        assert!(results[0].rerank_score.unwrap() > results[1].rerank_score.unwrap());
    }

    #[test]
    fn test_mock_cross_encoder_empty() {
        let reranker = MockCrossEncoderReranker::new("test-model");
        let results = reranker.rerank("test", &[], 10).unwrap();
        assert!(results.is_empty());
    }

    // ============ CompositeReranker Tests ============

    #[test]
    fn test_composite_reranker_empty() {
        let reranker = CompositeReranker::new();
        let candidates = vec![create_result("test")];

        let results = reranker.rerank("test", &candidates, 10).unwrap();
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_composite_reranker_single() {
        let lexical = Box::new(LexicalReranker::new());
        let reranker = CompositeReranker::new().with_reranker(lexical, 1.0);

        let candidates =
            vec![create_result("exact match query here"), create_result("no match at all")];

        let results = reranker.rerank("query", &candidates, 2).unwrap();
        assert_eq!(results.len(), 2);
        assert!(results[0].rerank_score.is_some());
    }

    #[test]
    fn test_composite_reranker_multiple() {
        let lexical = Box::new(LexicalReranker::new());
        let cross = Box::new(MockCrossEncoderReranker::new("test"));

        let reranker =
            CompositeReranker::new().with_reranker(lexical, 0.5).with_reranker(cross, 0.5);

        let candidates =
            vec![create_result("machine learning test"), create_result("unrelated content")];

        let results = reranker.rerank("machine learning", &candidates, 2).unwrap();
        assert_eq!(results.len(), 2);
    }

    // ============ NoOpReranker Tests ============

    #[test]
    fn test_noop_reranker() {
        let reranker = NoOpReranker::new();
        let candidates =
            vec![create_result_with_score("first", 0.9), create_result_with_score("second", 0.8)];

        let results = reranker.rerank("query", &candidates, 10).unwrap();

        assert_eq!(results.len(), 2);
        // Order should be preserved
        assert!(results[0].chunk.content.contains("first"));
    }

    #[test]
    fn test_noop_reranker_top_k() {
        let reranker = NoOpReranker::new();
        let candidates: Vec<_> = (0..10).map(|i| create_result(&format!("doc {i}"))).collect();

        let results = reranker.rerank("query", &candidates, 3).unwrap();
        assert_eq!(results.len(), 3);
    }

    // ============ Property-Based Tests ============

    use proptest::prelude::*;

    proptest! {
        #[test]
        fn prop_lexical_rerank_scores_bounded(
            query in "[a-zA-Z ]{1,20}",
            content in "[a-zA-Z ]{1,100}"
        ) {
            let reranker = LexicalReranker::new();
            let candidates = vec![create_result(&content)];

            let results = reranker.rerank(&query, &candidates, 1).unwrap();
            let score = results[0].rerank_score.unwrap();

            prop_assert!(score >= 0.0);
            prop_assert!(score <= 1.0);
        }

        #[test]
        fn prop_rerank_respects_top_k(k in 1usize..10, n in 1usize..20) {
            let reranker = LexicalReranker::new();
            let candidates: Vec<_> = (0..n)
                .map(|i| create_result(&format!("document {i}")))
                .collect();

            let results = reranker.rerank("document", &candidates, k).unwrap();
            prop_assert!(results.len() <= k);
            prop_assert!(results.len() <= n);
        }

        #[test]
        fn prop_noop_preserves_order(n in 1usize..10) {
            let reranker = NoOpReranker::new();
            let candidates: Vec<_> = (0..n)
                .map(|i| create_result(&format!("doc {i}")))
                .collect();

            let results = reranker.rerank("query", &candidates, n).unwrap();

            for (i, result) in results.iter().enumerate() {
                let expected = format!("doc {i}");
                prop_assert!(result.chunk.content.contains(&expected));
            }
        }
    }
}
