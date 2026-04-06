//! Decision pattern storage with cosine similarity search.
//!
//! Stores `DecisionPattern` instances indexed by a unique `pattern_id`,
//! and retrieves the top-k most similar patterns to a query feature vector
//! using cosine similarity over the pattern's `feature_weights`.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A decision pattern with feature weights and metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionPattern {
    /// Unique identifier for this pattern.
    pub pattern_id: String,
    /// Human-readable description of this pattern.
    pub description: String,
    /// Feature weight vector used for similarity search.
    pub feature_weights: Vec<f32>,
    /// Confidence score in range [0.0, 1.0].
    pub confidence: f32,
    /// Category label for this pattern.
    pub category: String,
}

impl DecisionPattern {
    /// Create a new decision pattern.
    #[must_use]
    pub fn new(
        pattern_id: impl Into<String>,
        description: impl Into<String>,
        feature_weights: Vec<f32>,
        confidence: f32,
        category: impl Into<String>,
    ) -> Self {
        Self {
            pattern_id: pattern_id.into(),
            description: description.into(),
            feature_weights,
            confidence: confidence.clamp(0.0, 1.0),
            category: category.into(),
        }
    }
}

/// Store for decision patterns with cosine-similarity retrieval.
///
/// Patterns are stored in a `HashMap` keyed by `pattern_id`.
/// The `search` method computes cosine similarity between the query
/// feature vector and every stored pattern, returning the top-k results
/// sorted by descending similarity.
///
/// # Example
///
/// ```
/// use entrenar::decision::{DecisionPattern, PatternStore};
///
/// let mut store = PatternStore::new();
/// store.add_pattern(DecisionPattern::new(
///     "p1", "type fix", vec![1.0, 0.0, 0.0], 0.9, "type_error",
/// ));
///
/// let results = store.search(&[1.0, 0.0, 0.0], 5);
/// assert_eq!(results.len(), 1);
/// assert_eq!(results[0].pattern_id, "p1");
/// ```
#[derive(Debug, Clone, Default)]
pub struct PatternStore {
    patterns: HashMap<String, DecisionPattern>,
}

impl PatternStore {
    /// Create an empty pattern store.
    #[must_use]
    pub fn new() -> Self {
        Self { patterns: HashMap::new() }
    }

    /// Add a pattern to the store.
    ///
    /// If a pattern with the same `pattern_id` already exists it is replaced.
    pub fn add_pattern(&mut self, pattern: DecisionPattern) {
        self.patterns.insert(pattern.pattern_id.clone(), pattern);
    }

    /// Retrieve a pattern by its id.
    #[must_use]
    pub fn get_pattern(&self, id: &str) -> Option<&DecisionPattern> {
        self.patterns.get(id)
    }

    /// Search for the top-k patterns most similar to `query_features`.
    ///
    /// Similarity is measured by cosine similarity between `query_features`
    /// and each pattern's `feature_weights`. Patterns whose feature vector
    /// has a different length than the query, or whose norm is zero, receive
    /// a similarity of zero.
    ///
    /// Returns results sorted by descending similarity, limited to `top_k`.
    #[must_use]
    pub fn search(&self, query_features: &[f32], top_k: usize) -> Vec<&DecisionPattern> {
        let mut scored: Vec<(f32, &DecisionPattern)> = self
            .patterns
            .values()
            .map(|p| {
                let sim = cosine_similarity(query_features, &p.feature_weights);
                (sim, p)
            })
            .collect();

        // Sort descending by similarity (NaN-safe: treat NaN as less than everything).
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k);
        scored.into_iter().map(|(_, p)| p).collect()
    }

    /// List all patterns in the store (unordered).
    #[must_use]
    pub fn list_patterns(&self) -> Vec<&DecisionPattern> {
        self.patterns.values().collect()
    }

    /// Remove a pattern by id. Returns the removed pattern if it existed.
    pub fn remove_pattern(&mut self, id: &str) -> Option<DecisionPattern> {
        self.patterns.remove(id)
    }

    /// Return the number of stored patterns.
    #[must_use]
    pub fn len(&self) -> usize {
        self.patterns.len()
    }

    /// Return whether the store is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.patterns.is_empty()
    }
}

/// Compute cosine similarity between two vectors.
///
/// Returns 0.0 if the vectors differ in length or either has zero norm.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }

    let mut dot = 0.0_f32;
    let mut norm_a = 0.0_f32;
    let mut norm_b = 0.0_f32;

    for i in 0..a.len() {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom == 0.0 {
        0.0
    } else {
        dot / denom
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_pattern(id: &str, weights: Vec<f32>, category: &str) -> DecisionPattern {
        DecisionPattern::new(id, format!("desc_{id}"), weights, 0.8, category)
    }

    #[test]
    fn test_add_and_get_pattern() {
        let mut store = PatternStore::new();
        let p = make_pattern("p1", vec![1.0, 0.0], "cat_a");
        store.add_pattern(p);

        let retrieved = store.get_pattern("p1");
        assert!(retrieved.is_some());
        assert_eq!(retrieved.expect("operation should succeed").pattern_id, "p1");
        assert_eq!(retrieved.expect("operation should succeed").category, "cat_a");
    }

    #[test]
    fn test_get_nonexistent_pattern() {
        let store = PatternStore::new();
        assert!(store.get_pattern("missing").is_none());
    }

    #[test]
    fn test_add_replaces_existing() {
        let mut store = PatternStore::new();
        store.add_pattern(make_pattern("p1", vec![1.0], "old"));
        store.add_pattern(make_pattern("p1", vec![2.0], "new"));

        assert_eq!(store.len(), 1);
        assert_eq!(store.get_pattern("p1").expect("operation should succeed").category, "new");
    }

    #[test]
    fn test_remove_pattern() {
        let mut store = PatternStore::new();
        store.add_pattern(make_pattern("p1", vec![1.0], "a"));
        store.add_pattern(make_pattern("p2", vec![2.0], "b"));

        let removed = store.remove_pattern("p1");
        assert!(removed.is_some());
        assert_eq!(removed.expect("operation should succeed").pattern_id, "p1");
        assert_eq!(store.len(), 1);
        assert!(store.get_pattern("p1").is_none());
    }

    #[test]
    fn test_remove_nonexistent() {
        let mut store = PatternStore::new();
        assert!(store.remove_pattern("ghost").is_none());
    }

    #[test]
    fn test_list_patterns() {
        let mut store = PatternStore::new();
        store.add_pattern(make_pattern("p1", vec![1.0], "a"));
        store.add_pattern(make_pattern("p2", vec![2.0], "b"));

        let list = store.list_patterns();
        assert_eq!(list.len(), 2);
        let ids: Vec<&str> = list.iter().map(|p| p.pattern_id.as_str()).collect();
        assert!(ids.contains(&"p1"));
        assert!(ids.contains(&"p2"));
    }

    #[test]
    fn test_len_and_is_empty() {
        let mut store = PatternStore::new();
        assert!(store.is_empty());
        assert_eq!(store.len(), 0);

        store.add_pattern(make_pattern("p1", vec![1.0], "a"));
        assert!(!store.is_empty());
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let sim = cosine_similarity(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0]);
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let sim = cosine_similarity(&[1.0, 0.0], &[0.0, 1.0]);
        assert!(sim.abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let sim = cosine_similarity(&[1.0, 0.0], &[-1.0, 0.0]);
        assert!((sim - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_different_lengths() {
        let sim = cosine_similarity(&[1.0, 2.0], &[1.0]);
        assert_eq!(sim, 0.0);
    }

    #[test]
    fn test_cosine_similarity_zero_vector() {
        let sim = cosine_similarity(&[0.0, 0.0], &[1.0, 2.0]);
        assert_eq!(sim, 0.0);
    }

    #[test]
    fn test_search_returns_top_k() {
        let mut store = PatternStore::new();
        store.add_pattern(make_pattern("close", vec![0.9, 0.1, 0.0], "a"));
        store.add_pattern(make_pattern("exact", vec![1.0, 0.0, 0.0], "b"));
        store.add_pattern(make_pattern("far", vec![0.0, 0.0, 1.0], "c"));

        let results = store.search(&[1.0, 0.0, 0.0], 2);
        assert_eq!(results.len(), 2);
        // Most similar first
        assert_eq!(results[0].pattern_id, "exact");
        assert_eq!(results[1].pattern_id, "close");
    }

    #[test]
    fn test_search_top_k_larger_than_store() {
        let mut store = PatternStore::new();
        store.add_pattern(make_pattern("p1", vec![1.0, 0.0], "a"));

        let results = store.search(&[1.0, 0.0], 10);
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_search_empty_store() {
        let store = PatternStore::new();
        let results = store.search(&[1.0, 0.0], 5);
        assert!(results.is_empty());
    }

    #[test]
    fn test_search_with_mismatched_dimensions() {
        let mut store = PatternStore::new();
        // Pattern with 3D weights, query with 2D
        store.add_pattern(make_pattern("p1", vec![1.0, 0.0, 0.0], "a"));

        let results = store.search(&[1.0, 0.0], 5);
        // Should return the pattern but with zero similarity
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_confidence_clamped() {
        let p = DecisionPattern::new("id", "desc", vec![], 1.5, "cat");
        assert_eq!(p.confidence, 1.0);

        let p2 = DecisionPattern::new("id2", "desc", vec![], -0.5, "cat");
        assert_eq!(p2.confidence, 0.0);
    }

    #[test]
    fn test_default_store() {
        let store = PatternStore::default();
        assert!(store.is_empty());
    }
}
