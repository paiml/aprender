//! Incremental IDF (Inverse Document Frequency) tracking.
//!
//! Traditional TF-IDF assumes a static corpus, but in production systems
//! documents arrive continuously. This module provides incremental IDF updates
//! with exponential decay to handle concept drift.
//!
//! # Algorithm
//!
//! For each term, we track:
//! - `doc_freq`: Number of documents containing the term
//! - `total_docs`: Total number of documents seen
//!
//! IDF is computed as: `log((total_docs + 1) / (doc_freq + 1)) + 1`
//!
//! With exponential decay (half-life parameter):
//! - Old frequencies decay over time
//! - Recent documents have more weight
//!
//! # References
//!
//! Sculley et al. (2015). "Hidden Technical Debt in Machine Learning Systems."
//! NIPS 2015. <https://papers.nips.cc/paper/2015/file/86df7dcfd896fcaf2674f757a2463eba-Paper.pdf>
//!
//! # Examples
//!
//! ```
//! use aprender::text::incremental_idf::IncrementalIDF;
//!
//! let mut idf = IncrementalIDF::new(0.95);
//!
//! // Add documents
//! idf.update(&["machine", "learning"]);
//! idf.update(&["machine", "intelligence"]);
//! idf.update(&["deep", "learning"]);
//!
//! // Get IDF scores
//! let idf_machine = idf.idf("machine");
//! let idf_learning = idf.idf("learning");
//! let idf_deep = idf.idf("deep");
//!
//! // "machine" appears in 2/3 docs, "deep" in 1/3
//! assert!(idf_deep > idf_machine);
//! ```

use std::collections::HashMap;

/// Incremental IDF tracker with exponential decay.
///
/// # Configuration
///
/// - `decay_factor`: Exponential decay factor (0.9-0.99 recommended)
///   - 0.95 = half-life of ~14 documents
///   - 0.99 = half-life of ~69 documents
#[derive(Debug, Clone)]
pub struct IncrementalIDF {
    /// Document frequency for each term
    doc_freq: HashMap<String, f64>,
    /// Total number of documents processed
    total_docs: f64,
    /// Exponential decay factor (0 < decay < 1)
    decay_factor: f64,
}

impl IncrementalIDF {
    /// Create a new incremental IDF tracker.
    ///
    /// # Arguments
    ///
    /// * `decay_factor` - Exponential decay factor (0.9-0.99 recommended)
    ///
    /// # Examples
    ///
    /// ```
    /// use aprender::text::incremental_idf::IncrementalIDF;
    ///
    /// let idf = IncrementalIDF::new(0.95);
    /// ```
    #[must_use]
    pub fn new(decay_factor: f64) -> Self {
        assert!(
            (0.0..1.0).contains(&decay_factor),
            "Decay factor must be in (0, 1)"
        );

        Self {
            doc_freq: HashMap::new(),
            total_docs: 0.0,
            decay_factor,
        }
    }

    /// Update IDF with a new document.
    ///
    /// # Arguments
    ///
    /// * `terms` - Unique terms in the document
    ///
    /// # Examples
    ///
    /// ```
    /// use aprender::text::incremental_idf::IncrementalIDF;
    ///
    /// let mut idf = IncrementalIDF::new(0.95);
    /// idf.update(&["hello", "world"]);
    /// idf.update(&["hello", "rust"]);
    /// ```
    pub fn update(&mut self, terms: &[&str]) {
        // Apply decay to all existing frequencies
        self.total_docs *= self.decay_factor;
        for freq in self.doc_freq.values_mut() {
            *freq *= self.decay_factor;
        }

        // Increment document count
        self.total_docs += 1.0;

        // Update document frequencies
        for &term in terms {
            *self.doc_freq.entry(term.to_string()).or_insert(0.0) += 1.0;
        }
    }

    /// Get IDF score for a term.
    ///
    /// Returns the IDF value, or a default high value for unseen terms.
    ///
    /// # Arguments
    ///
    /// * `term` - Term to get IDF for
    ///
    /// # Examples
    ///
    /// ```
    /// use aprender::text::incremental_idf::IncrementalIDF;
    ///
    /// let mut idf = IncrementalIDF::new(0.95);
    /// idf.update(&["hello", "world"]);
    ///
    /// let hello_idf = idf.idf("hello");
    /// assert!(hello_idf > 0.0);
    /// ```
    #[must_use]
    pub fn idf(&self, term: &str) -> f64 {
        let df = self.doc_freq.get(term).copied().unwrap_or(0.0);
        ((self.total_docs + 1.0) / (df + 1.0)).ln() + 1.0
    }

    /// Get all tracked terms and their IDFs.
    ///
    /// # Examples
    ///
    /// ```
    /// use aprender::text::incremental_idf::IncrementalIDF;
    ///
    /// let mut idf = IncrementalIDF::new(0.95);
    /// idf.update(&["hello", "world"]);
    ///
    /// let terms = idf.terms();
    /// assert!(terms.contains_key("hello"));
    /// assert!(terms.contains_key("world"));
    /// ```
    #[must_use]
    pub fn terms(&self) -> HashMap<String, f64> {
        self.doc_freq
            .keys()
            .map(|term| (term.clone(), self.idf(term)))
            .collect()
    }

    /// Number of terms tracked.
    ///
    /// # Examples
    ///
    /// ```
    /// use aprender::text::incremental_idf::IncrementalIDF;
    ///
    /// let mut idf = IncrementalIDF::new(0.95);
    /// assert_eq!(idf.len(), 0);
    ///
    /// idf.update(&["hello", "world"]);
    /// assert_eq!(idf.len(), 2);
    /// ```
    #[must_use]
    pub fn len(&self) -> usize {
        self.doc_freq.len()
    }

    /// Check if tracker is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use aprender::text::incremental_idf::IncrementalIDF;
    ///
    /// let idf = IncrementalIDF::new(0.95);
    /// assert!(idf.is_empty());
    /// ```
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.doc_freq.is_empty()
    }

    /// Total number of documents processed (with decay).
    ///
    /// # Examples
    ///
    /// ```
    /// use aprender::text::incremental_idf::IncrementalIDF;
    ///
    /// let mut idf = IncrementalIDF::new(0.95);
    /// idf.update(&["hello"]);
    /// idf.update(&["world"]);
    ///
    /// // With decay=0.95, total_docs = 1*0.95 + 1 = 1.95
    /// assert!((idf.total_docs() - 1.95).abs() < 0.01);
    /// ```
    #[must_use]
    pub fn total_docs(&self) -> f64 {
        self.total_docs
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_tracker() {
        let idf = IncrementalIDF::new(0.95);
        assert!(idf.is_empty());
        assert_eq!(idf.len(), 0);
        assert!((idf.total_docs() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_single_document() {
        let mut idf = IncrementalIDF::new(0.95);
        idf.update(&["hello", "world"]);

        assert_eq!(idf.len(), 2);
        assert!((idf.total_docs() - 1.0).abs() < 1e-10);

        // Both terms appear in same document
        let hello_idf = idf.idf("hello");
        let world_idf = idf.idf("world");
        assert!((hello_idf - world_idf).abs() < 1e-10);
    }

    #[test]
    fn test_multiple_documents() {
        let mut idf = IncrementalIDF::new(0.95);

        idf.update(&["machine", "learning"]);
        idf.update(&["machine", "intelligence"]);
        idf.update(&["deep", "learning"]);

        // "machine" appears in 2 docs, "deep" in 1 doc
        let machine_idf = idf.idf("machine");
        let deep_idf = idf.idf("deep");

        // Less frequent terms have higher IDF
        assert!(
            deep_idf > machine_idf,
            "deep_idf={deep_idf}, machine_idf={machine_idf}"
        );
    }

    #[test]
    fn test_exponential_decay() {
        let mut idf = IncrementalIDF::new(0.5); // Strong decay for testing

        idf.update(&["old"]);

        // Add many documents without "old"
        for _ in 0..10 {
            idf.update(&["new"]);
        }

        // "old" should have decayed significantly
        let old_df = idf.doc_freq.get("old").copied().unwrap_or(0.0);
        assert!(old_df < 0.01, "Old term should have decayed, df={old_df}");
    }

    #[test]
    fn test_unseen_term() {
        let mut idf = IncrementalIDF::new(0.95);
        idf.update(&["hello"]);

        // Unseen term should have high IDF
        let unseen_idf = idf.idf("unseen");
        assert!(unseen_idf > 0.0);
    }

    #[test]
    fn test_terms_map() {
        let mut idf = IncrementalIDF::new(0.95);
        idf.update(&["alpha", "beta"]);

        let terms = idf.terms();
        assert_eq!(terms.len(), 2);
        assert!(terms.contains_key("alpha"));
        assert!(terms.contains_key("beta"));
    }

    #[test]
    #[should_panic(expected = "Decay factor must be in (0, 1)")]
    fn test_invalid_decay_factor() {
        let _ = IncrementalIDF::new(1.5);
    }

    #[test]
    fn test_idf_monotonicity() {
        // Terms with lower document frequency should have higher IDF
        let mut idf = IncrementalIDF::new(0.95);

        // "common" appears in all 3 documents
        idf.update(&["common", "doc1"]);
        idf.update(&["common", "doc2"]);
        idf.update(&["common", "doc3"]);

        let common_idf = idf.idf("common");
        let doc1_idf = idf.idf("doc1");

        // "doc1" appears in fewer documents -> higher IDF
        assert!(doc1_idf > common_idf);
    }
}
