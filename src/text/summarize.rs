//! Extractive text summarization.
//!
//! This module provides algorithms for extractive summarization:
//! - `TextRank`: Graph-based sentence ranking (like `PageRank`)
//! - TF-IDF: Sentence scoring based on term importance
//! - Hybrid: Combination of multiple methods
//!
//! # Quick Start
//!
//! ```
//! use aprender::text::summarize::{TextSummarizer, SummarizationMethod};
//!
//! let text = "First sentence. Second sentence. Third sentence.";
//! let summarizer = TextSummarizer::new(SummarizationMethod::TfIdf, 2);
//!
//! let summary = summarizer.summarize(text).expect("summarize should succeed");
//! assert_eq!(summary.len(), 2);  // Top 2 sentences
//! ```

use crate::AprenderError;
use std::collections::{HashMap, HashSet};

/// Summarization method.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SummarizationMethod {
    /// `TextRank` algorithm (graph-based)
    TextRank,
    /// TF-IDF sentence scoring
    TfIdf,
    /// Hybrid approach (average of `TextRank` and TF-IDF)
    Hybrid,
}

/// Extractive text summarizer.
///
/// Selects the most important sentences from the input text to create
/// a summary. Does not generate new text.
///
/// # Examples
///
/// ```
/// use aprender::text::summarize::{TextSummarizer, SummarizationMethod};
///
/// let text = "Machine learning is great. It solves many problems. \
///             Deep learning is a subset of machine learning.";
///
/// let summarizer = TextSummarizer::new(SummarizationMethod::TfIdf, 2);
/// let summary = summarizer.summarize(text).expect("summarize should succeed");
///
/// assert_eq!(summary.len(), 2);
/// ```
#[derive(Debug)]
pub struct TextSummarizer {
    /// Summarization method to use
    method: SummarizationMethod,
    /// Maximum number of sentences in summary
    max_sentences: usize,
    /// Damping factor for `TextRank` (default: 0.85)
    damping_factor: f64,
    /// Number of `TextRank` iterations (default: 100)
    max_iterations: usize,
    /// Convergence threshold for `TextRank` (default: 0.0001)
    convergence_threshold: f64,
}

impl TextSummarizer {
    /// Create a new text summarizer.
    ///
    /// # Arguments
    ///
    /// * `method` - Summarization method to use
    /// * `max_sentences` - Maximum number of sentences in summary
    ///
    /// # Examples
    ///
    /// ```
    /// use aprender::text::summarize::{TextSummarizer, SummarizationMethod};
    ///
    /// let summarizer = TextSummarizer::new(SummarizationMethod::TextRank, 3);
    /// ```
    #[must_use]
    pub fn new(method: SummarizationMethod, max_sentences: usize) -> Self {
        Self {
            method,
            max_sentences,
            damping_factor: 0.85,
            max_iterations: 100,
            convergence_threshold: 0.0001,
        }
    }

    /// Set damping factor for `TextRank`.
    ///
    /// # Examples
    ///
    /// ```
    /// use aprender::text::summarize::{TextSummarizer, SummarizationMethod};
    ///
    /// let summarizer = TextSummarizer::new(SummarizationMethod::TextRank, 2)
    ///     .with_damping_factor(0.9);
    /// ```
    #[must_use]
    pub fn with_damping_factor(mut self, factor: f64) -> Self {
        self.damping_factor = factor;
        self
    }

    /// Set maximum iterations for `TextRank`.
    ///
    /// # Examples
    ///
    /// ```
    /// use aprender::text::summarize::{TextSummarizer, SummarizationMethod};
    ///
    /// let summarizer = TextSummarizer::new(SummarizationMethod::TextRank, 2)
    ///     .with_max_iterations(50);
    /// ```
    #[must_use]
    pub fn with_max_iterations(mut self, max_iter: usize) -> Self {
        self.max_iterations = max_iter;
        self
    }

    /// Summarize text.
    ///
    /// Returns a list of the most important sentences from the input text.
    ///
    /// # Arguments
    ///
    /// * `text` - Input text to summarize
    ///
    /// # Examples
    ///
    /// ```
    /// use aprender::text::summarize::{TextSummarizer, SummarizationMethod};
    ///
    /// let text = "First important sentence. Second key point. \
    ///             Third detail. Fourth major finding.";
    ///
    /// let summarizer = TextSummarizer::new(SummarizationMethod::TfIdf, 2);
    /// let summary = summarizer.summarize(text).expect("summarize should succeed");
    ///
    /// assert!(summary.len() <= 2);
    /// ```
    pub fn summarize(&self, text: &str) -> Result<Vec<String>, AprenderError> {
        let sentences = Self::split_sentences(text);

        if sentences.is_empty() {
            return Ok(Vec::new());
        }

        if sentences.len() <= self.max_sentences {
            return Ok(sentences);
        }

        // Compute sentence scores based on method
        let scores = match self.method {
            SummarizationMethod::TextRank => self.textrank_scores(&sentences),
            SummarizationMethod::TfIdf => self.tfidf_scores(&sentences),
            SummarizationMethod::Hybrid => self.hybrid_scores(&sentences),
        };

        // Select top-k sentences
        let top_indices = Self::select_top_sentences(&scores, self.max_sentences);

        // Return sentences in original order
        let mut selected: Vec<(usize, String)> = top_indices
            .into_iter()
            .map(|idx| (idx, sentences[idx].clone()))
            .collect();
        selected.sort_by_key(|(idx, _)| *idx);

        Ok(selected.into_iter().map(|(_, sent)| sent).collect())
    }

    /// Split text into sentences.
    fn split_sentences(text: &str) -> Vec<String> {
        text.split(['.', '!', '?'])
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(String::from)
            .collect()
    }

    /// Compute `TextRank` scores for sentences.
    fn textrank_scores(&self, sentences: &[String]) -> Vec<f64> {
        let n = sentences.len();
        if n == 0 {
            return Vec::new();
        }

        // Build similarity matrix
        let similarity = self.build_similarity_matrix(sentences);

        // Initialize scores uniformly
        let mut scores = vec![1.0 / n as f64; n];
        let mut new_scores = vec![0.0; n];

        // Power iteration
        for _ in 0..self.max_iterations {
            let mut converged = true;

            for i in 0..n {
                let mut score = (1.0 - self.damping_factor) / n as f64;

                for j in 0..n {
                    if i != j {
                        let outbound_sum: f64 =
                            (0..n).filter(|&k| k != j).map(|k| similarity[j][k]).sum();

                        if outbound_sum > 1e-10 {
                            score +=
                                self.damping_factor * (similarity[j][i] / outbound_sum) * scores[j];
                        }
                    }
                }

                new_scores[i] = score;

                if (new_scores[i] - scores[i]).abs() > self.convergence_threshold {
                    converged = false;
                }
            }

            scores.clone_from_slice(&new_scores);

            if converged {
                break;
            }
        }

        scores
    }

    /// Compute TF-IDF scores for sentences.
    #[allow(clippy::unused_self)]
    fn tfidf_scores(&self, sentences: &[String]) -> Vec<f64> {
        if sentences.is_empty() {
            return Vec::new();
        }

        // Tokenize all sentences
        let tokenized: Vec<Vec<String>> = sentences.iter().map(|s| Self::tokenize(s)).collect();

        // Compute IDF values
        let idf = Self::compute_idf(&tokenized);

        // Compute TF-IDF score for each sentence
        let scores: Vec<f64> = tokenized
            .iter()
            .map(|tokens| {
                if tokens.is_empty() {
                    return 0.0;
                }

                // Term frequency in sentence
                let mut tf: HashMap<&str, f64> = HashMap::new();
                for token in tokens {
                    *tf.entry(token.as_str()).or_insert(0.0) += 1.0;
                }

                // Normalize TF
                let max_tf = tf.values().copied().fold(0.0, f64::max);
                if max_tf > 0.0 {
                    for value in tf.values_mut() {
                        *value /= max_tf;
                    }
                }

                // Sum TF-IDF scores
                tf.iter()
                    .map(|(term, &tf_val)| {
                        let idf_val = idf.get(*term).copied().unwrap_or(0.0);
                        tf_val * idf_val
                    })
                    .sum()
            })
            .collect();

        scores
    }

    /// Compute hybrid scores (average of `TextRank` and TF-IDF).
    fn hybrid_scores(&self, sentences: &[String]) -> Vec<f64> {
        let textrank = self.textrank_scores(sentences);
        let tfidf = self.tfidf_scores(sentences);

        // Normalize both scores to [0, 1]
        let textrank_norm = Self::normalize(&textrank);
        let tfidf_norm = Self::normalize(&tfidf);

        // Average
        let scores: Vec<f64> = textrank_norm
            .iter()
            .zip(tfidf_norm.iter())
            .map(|(tr, tf)| (tr + tf) / 2.0)
            .collect();

        scores
    }

    /// Build sentence similarity matrix based on word overlap.
    #[allow(clippy::unused_self)]
    fn build_similarity_matrix(&self, sentences: &[String]) -> Vec<Vec<f64>> {
        let n = sentences.len();
        let mut similarity = vec![vec![0.0; n]; n];

        let tokenized: Vec<HashSet<String>> = sentences
            .iter()
            .map(|s| Self::tokenize(s).into_iter().collect())
            .collect();

        for i in 0..n {
            for j in 0..n {
                if i != j {
                    let intersection: f64 = tokenized[i].intersection(&tokenized[j]).count() as f64;

                    let union_size = tokenized[i].len() + tokenized[j].len();

                    if union_size > 0 {
                        similarity[i][j] = (2.0 * intersection) / union_size as f64;
                    }
                }
            }
        }

        similarity
    }

    /// Tokenize text into words.
    fn tokenize(text: &str) -> Vec<String> {
        text.to_lowercase()
            .split(|c: char| c.is_whitespace() || c.is_ascii_punctuation())
            .filter(|s| !s.is_empty())
            .map(String::from)
            .collect()
    }

    /// Compute IDF values for all terms.
    fn compute_idf(documents: &[Vec<String>]) -> HashMap<String, f64> {
        let n = documents.len() as f64;
        let mut document_freq: HashMap<String, usize> = HashMap::new();

        for doc in documents {
            let unique_terms: HashSet<&str> = doc.iter().map(String::as_str).collect();
            for term in unique_terms {
                *document_freq.entry(term.to_string()).or_insert(0) += 1;
            }
        }

        document_freq
            .into_iter()
            .map(|(term, df)| {
                let idf = ((n + 1.0) / (df as f64 + 1.0)).ln() + 1.0;
                (term, idf)
            })
            .collect()
    }

    /// Normalize scores to [0, 1] range.
    fn normalize(scores: &[f64]) -> Vec<f64> {
        if scores.is_empty() {
            return Vec::new();
        }

        let min_score = scores.iter().copied().fold(f64::INFINITY, f64::min);
        let max_score = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);

        let range = max_score - min_score;

        if range < 1e-10 {
            return vec![0.5; scores.len()];
        }

        scores.iter().map(|&s| (s - min_score) / range).collect()
    }

    /// Select top-k sentences by score.
    fn select_top_sentences(scores: &[f64], k: usize) -> Vec<usize> {
        let mut indexed: Vec<(usize, f64)> = scores
            .iter()
            .enumerate()
            .map(|(idx, &score)| (idx, score))
            .collect();

        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        indexed.into_iter().take(k).map(|(idx, _)| idx).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tfidf_summarization() {
        let text = "Machine learning is great. It solves many problems. \
                    Deep learning is a subset of machine learning. \
                    Neural networks are powerful tools.";

        let summarizer = TextSummarizer::new(SummarizationMethod::TfIdf, 2);
        let summary = summarizer.summarize(text).expect("should succeed");

        assert_eq!(summary.len(), 2);
    }

    #[test]
    fn test_textrank_summarization() {
        let text = "First sentence about AI. Second sentence about ML. \
                    Third sentence about AI and ML. Fourth unrelated sentence.";

        let summarizer = TextSummarizer::new(SummarizationMethod::TextRank, 2);
        let summary = summarizer.summarize(text).expect("should succeed");

        assert_eq!(summary.len(), 2);
    }

    #[test]
    fn test_hybrid_summarization() {
        let text = "Alpha beta gamma. Delta epsilon zeta. \
                    Eta theta iota. Kappa lambda mu.";

        let summarizer = TextSummarizer::new(SummarizationMethod::Hybrid, 2);
        let summary = summarizer.summarize(text).expect("should succeed");

        assert_eq!(summary.len(), 2);
    }

    #[test]
    fn test_empty_text() {
        let summarizer = TextSummarizer::new(SummarizationMethod::TfIdf, 2);
        let summary = summarizer.summarize("").expect("should succeed");

        assert_eq!(summary.len(), 0);
    }

    #[test]
    fn test_fewer_sentences_than_max() {
        let text = "Only one sentence here.";

        let summarizer = TextSummarizer::new(SummarizationMethod::TfIdf, 5);
        let summary = summarizer.summarize(text).expect("should succeed");

        assert_eq!(summary.len(), 1);
    }

    #[test]
    fn test_sentence_order_preserved() {
        let text = "First. Second. Third. Fourth. Fifth.";

        let summarizer = TextSummarizer::new(SummarizationMethod::TfIdf, 3);
        let summary = summarizer.summarize(text).expect("should succeed");

        // Check that sentences appear in original order
        for i in 0..summary.len().saturating_sub(1) {
            let idx1 = text.find(&summary[i]).expect("sentence should exist");
            let idx2 = text.find(&summary[i + 1]).expect("sentence should exist");
            assert!(idx1 < idx2, "Sentences should maintain original order");
        }
    }

    // ====================================================================
    // Additional coverage tests for uncovered branches
    // ====================================================================

    #[test]
    fn test_with_damping_factor() {
        let text = "First sentence about topic. Second sentence about topic. \
                    Third sentence different. Fourth sentence topic again.";

        let summarizer =
            TextSummarizer::new(SummarizationMethod::TextRank, 2).with_damping_factor(0.5);
        let summary = summarizer.summarize(text).expect("should succeed");
        assert_eq!(summary.len(), 2);
    }

    #[test]
    fn test_with_max_iterations() {
        let text = "Alpha beta gamma. Delta epsilon zeta. \
                    Eta theta iota. Kappa lambda mu.";

        let summarizer =
            TextSummarizer::new(SummarizationMethod::TextRank, 2).with_max_iterations(5);
        let summary = summarizer.summarize(text).expect("should succeed");
        assert_eq!(summary.len(), 2);
    }

    #[test]
    fn test_normalize_equal_scores() {
        // When all scores are the same, normalize should return 0.5 for each
        let scores = vec![3.0, 3.0, 3.0, 3.0];
        let normalized = TextSummarizer::normalize(&scores);
        for val in &normalized {
            assert!(
                (*val - 0.5).abs() < 1e-10,
                "Equal scores should normalize to 0.5"
            );
        }
    }

    #[test]
    fn test_normalize_empty() {
        let scores: Vec<f64> = Vec::new();
        let normalized = TextSummarizer::normalize(&scores);
        assert!(normalized.is_empty());
    }

    #[test]
    fn test_normalize_two_values() {
        let scores = vec![0.0, 1.0];
        let normalized = TextSummarizer::normalize(&scores);
        assert!((normalized[0] - 0.0).abs() < 1e-10);
        assert!((normalized[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_summarization_method_debug() {
        let debug_tr = format!("{:?}", SummarizationMethod::TextRank);
        let debug_tf = format!("{:?}", SummarizationMethod::TfIdf);
        let debug_hy = format!("{:?}", SummarizationMethod::Hybrid);

        assert!(debug_tr.contains("TextRank"));
        assert!(debug_tf.contains("TfIdf"));
        assert!(debug_hy.contains("Hybrid"));
    }

    #[test]
    fn test_summarization_method_clone_copy_eq() {
        let m1 = SummarizationMethod::TextRank;
        let m2 = m1; // Copy
        let m3 = m1.clone(); // Clone
        assert_eq!(m1, m2);
        assert_eq!(m1, m3);
        assert_ne!(SummarizationMethod::TextRank, SummarizationMethod::TfIdf);
    }

    #[test]
    fn test_summarizer_debug() {
        let summarizer = TextSummarizer::new(SummarizationMethod::TfIdf, 2);
        let debug = format!("{:?}", summarizer);
        assert!(debug.contains("TextSummarizer"));
    }

    #[test]
    fn test_textrank_early_convergence() {
        // With very few iterations and high convergence threshold,
        // TextRank should converge quickly
        let text = "Same sentence. Same sentence. Same sentence. Same sentence.";

        let summarizer =
            TextSummarizer::new(SummarizationMethod::TextRank, 2).with_max_iterations(1000);
        let summary = summarizer.summarize(text).expect("should succeed");
        // All sentences are identical, so any 2 can be selected
        assert_eq!(summary.len(), 2);
    }

    #[test]
    fn test_tfidf_with_empty_sentence_tokens() {
        // A sentence that becomes empty after tokenization (only punctuation)
        let text = "Hello world. !!! ???. This is a sentence. Another one here.";

        let summarizer = TextSummarizer::new(SummarizationMethod::TfIdf, 2);
        let summary = summarizer.summarize(text).expect("should succeed");
        assert_eq!(summary.len(), 2);
    }

    #[test]
    fn test_hybrid_with_many_sentences() {
        let text = "Sentence one about AI. Sentence two about ML. \
                    Sentence three about AI and ML. Sentence four about data. \
                    Sentence five about AI models. Sentence six conclusion.";

        let summarizer = TextSummarizer::new(SummarizationMethod::Hybrid, 3);
        let summary = summarizer.summarize(text).expect("should succeed");
        assert_eq!(summary.len(), 3);
    }

    #[test]
    fn test_split_sentences_with_exclamation_and_question() {
        let text = "First! Second? Third.";

        let summarizer = TextSummarizer::new(SummarizationMethod::TfIdf, 3);
        let summary = summarizer.summarize(text).expect("should succeed");
        assert_eq!(summary.len(), 3);
    }

    #[test]
    fn test_two_sentence_input_one_output() {
        let text = "First sentence. Second sentence.";

        let summarizer = TextSummarizer::new(SummarizationMethod::TfIdf, 1);
        let summary = summarizer.summarize(text).expect("should succeed");
        assert_eq!(summary.len(), 1);
    }

    #[test]
    fn test_select_top_sentences_ordering() {
        let scores = vec![0.1, 0.5, 0.3, 0.9, 0.2];
        let top2 = TextSummarizer::select_top_sentences(&scores, 2);
        // Should select indices of highest scores: index 3 (0.9) and index 1 (0.5)
        assert!(top2.contains(&3));
        assert!(top2.contains(&1));
    }

    #[test]
    fn test_textrank_no_overlap_sentences() {
        // Sentences with no word overlap (similarity matrix is all zeros)
        let text = "Alpha beta. Gamma delta. Epsilon zeta. Eta theta.";

        let summarizer = TextSummarizer::new(SummarizationMethod::TextRank, 2);
        let summary = summarizer.summarize(text).expect("should succeed");
        assert_eq!(summary.len(), 2);
    }

    #[test]
    fn test_exactly_max_sentences() {
        // Exactly max_sentences count -- should return all
        let text = "First. Second.";

        let summarizer = TextSummarizer::new(SummarizationMethod::TfIdf, 2);
        let summary = summarizer.summarize(text).expect("should succeed");
        assert_eq!(summary.len(), 2);
    }

    #[test]
    fn test_builder_chain() {
        let summarizer = TextSummarizer::new(SummarizationMethod::TextRank, 3)
            .with_damping_factor(0.75)
            .with_max_iterations(50);

        let text = "One about topic. Two about topic. Three topic. Four topic. Five topic.";
        let summary = summarizer.summarize(text).expect("should succeed");
        assert_eq!(summary.len(), 3);
    }
}
