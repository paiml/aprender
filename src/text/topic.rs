//! Topic modeling with Latent Dirichlet Allocation (LDA).
//!
//! This module provides unsupervised topic discovery:
//! - LDA: Discover latent topics in document collections
//! - Topic distribution per document
//! - Word distribution per topic
//!
//! # Quick Start
//!
//! ```
//! use aprender::text::topic::LatentDirichletAllocation;
//! use aprender::primitives::Matrix;
//!
//! // Document-term matrix (3 docs × 5 terms)
//! let dtm = Matrix::from_vec(3, 5, vec![
//!     2.0, 1.0, 0.0, 0.0, 0.0,
//!     0.0, 0.0, 2.0, 1.0, 0.0,
//!     1.0, 0.0, 0.0, 1.0, 2.0,
//! ]).unwrap();
//!
//! let mut lda = LatentDirichletAllocation::new(2);  // 2 topics
//! lda.fit(&dtm, 10).unwrap();  // 10 iterations
//! ```

use crate::primitives::Matrix;
use crate::AprenderError;

/// Latent Dirichlet Allocation for topic modeling.
///
/// LDA discovers latent topics in a collection of documents by modeling:
/// - Each document as a mixture of topics
/// - Each topic as a distribution over words
///
/// # Examples
///
/// ```
/// use aprender::text::topic::LatentDirichletAllocation;
/// use aprender::primitives::Matrix;
///
/// // Create document-term matrix
/// let dtm = Matrix::from_vec(2, 3, vec![
///     1.0, 2.0, 0.0,
///     0.0, 1.0, 2.0,
/// ]).unwrap();
///
/// let mut lda = LatentDirichletAllocation::new(2);
/// lda.fit(&dtm, 5).unwrap();
/// ```
#[derive(Debug)]
pub struct LatentDirichletAllocation {
    /// Number of topics
    n_topics: usize,
    /// Document-topic distribution (`n_docs` × `n_topics`)
    doc_topic: Option<Matrix<f64>>,
    /// Topic-word distribution (`n_topics` × `n_terms`)
    topic_word: Option<Matrix<f64>>,
    /// Random seed
    random_seed: u64,
}

impl LatentDirichletAllocation {
    /// Create a new LDA model.
    ///
    /// # Arguments
    ///
    /// * `n_topics` - Number of topics to discover
    ///
    /// # Examples
    ///
    /// ```
    /// use aprender::text::topic::LatentDirichletAllocation;
    ///
    /// let lda = LatentDirichletAllocation::new(5);  // 5 topics
    /// ```
    #[must_use]
    pub fn new(n_topics: usize) -> Self {
        Self {
            n_topics,
            doc_topic: None,
            topic_word: None,
            random_seed: 42,
        }
    }

    /// Set random seed.
    ///
    /// # Examples
    ///
    /// ```
    /// use aprender::text::topic::LatentDirichletAllocation;
    ///
    /// let lda = LatentDirichletAllocation::new(3).with_random_seed(123);
    /// ```
    #[must_use]
    pub fn with_random_seed(mut self, seed: u64) -> Self {
        self.random_seed = seed;
        self
    }

    /// Fit LDA model to document-term matrix.
    ///
    /// Uses simplified variational inference to learn topic distributions.
    ///
    /// # Arguments
    ///
    /// * `dtm` - Document-term matrix (`n_docs` × `n_terms`)
    /// * `max_iter` - Maximum iterations
    ///
    /// # Examples
    ///
    /// ```
    /// use aprender::text::topic::LatentDirichletAllocation;
    /// use aprender::primitives::Matrix;
    ///
    /// let dtm = Matrix::from_vec(2, 3, vec![
    ///     1.0, 2.0, 0.0,
    ///     0.0, 1.0, 2.0,
    /// ]).unwrap();
    ///
    /// let mut lda = LatentDirichletAllocation::new(2);
    /// lda.fit(&dtm, 10).unwrap();
    /// ```
    pub fn fit(&mut self, dtm: &Matrix<f64>, max_iter: usize) -> Result<(), AprenderError> {
        let n_docs = dtm.n_rows();
        let n_terms = dtm.n_cols();

        if n_docs == 0 || n_terms == 0 {
            return Err(AprenderError::Other(
                "Document-term matrix cannot be empty".to_string(),
            ));
        }

        // Initialize with uniform + small random noise
        let mut doc_topic_data = vec![0.0; n_docs * self.n_topics];
        let mut topic_word_data = vec![0.0; self.n_topics * n_terms];

        // Simple initialization: uniform distribution + noise
        let doc_topic_init = 1.0 / self.n_topics as f64;
        let topic_word_init = 1.0 / n_terms as f64;

        for i in 0..n_docs {
            for k in 0..self.n_topics {
                let idx = i * self.n_topics + k;
                doc_topic_data[idx] = doc_topic_init + self.pseudo_random(idx) * 0.01;
            }
        }

        for k in 0..self.n_topics {
            for v in 0..n_terms {
                let idx = k * n_terms + v;
                topic_word_data[idx] = topic_word_init + self.pseudo_random(idx + 1000) * 0.01;
            }
        }

        // Normalize
        Self::normalize_rows(&mut doc_topic_data, n_docs, self.n_topics);
        Self::normalize_rows(&mut topic_word_data, self.n_topics, n_terms);

        // Simple iterative update (simplified EM-like algorithm)
        for _ in 0..max_iter {
            // E-step: Compute expected topic assignments
            let mut new_doc_topic = vec![0.0; n_docs * self.n_topics];
            let mut new_topic_word = vec![0.0; self.n_topics * n_terms];

            for d in 0..n_docs {
                for v in 0..n_terms {
                    let count = dtm.get(d, v);
                    if count > 0.0 {
                        // Compute p(z|d,w) for each topic
                        let mut topic_probs = vec![0.0; self.n_topics];
                        let mut sum = 0.0;

                        for k in 0..self.n_topics {
                            let doc_topic_prob = doc_topic_data[d * self.n_topics + k];
                            let topic_word_prob = topic_word_data[k * n_terms + v];
                            topic_probs[k] = doc_topic_prob * topic_word_prob;
                            sum += topic_probs[k];
                        }

                        // Normalize and accumulate
                        if sum > 1e-10 {
                            for k in 0..self.n_topics {
                                let prob = topic_probs[k] / sum;
                                new_doc_topic[d * self.n_topics + k] += count * prob;
                                new_topic_word[k * n_terms + v] += count * prob;
                            }
                        }
                    }
                }
            }

            // M-step: Normalize distributions
            Self::normalize_rows(&mut new_doc_topic, n_docs, self.n_topics);
            Self::normalize_rows(&mut new_topic_word, self.n_topics, n_terms);

            doc_topic_data = new_doc_topic;
            topic_word_data = new_topic_word;
        }

        self.doc_topic = Some(
            Matrix::from_vec(n_docs, self.n_topics, doc_topic_data)
                .map_err(|e: &str| AprenderError::Other(e.to_string()))?,
        );
        self.topic_word = Some(
            Matrix::from_vec(self.n_topics, n_terms, topic_word_data)
                .map_err(|e: &str| AprenderError::Other(e.to_string()))?,
        );

        Ok(())
    }

    /// Get document-topic distribution.
    ///
    /// Returns matrix where each row is a document's topic distribution.
    ///
    /// # Examples
    ///
    /// ```
    /// use aprender::text::topic::LatentDirichletAllocation;
    /// use aprender::primitives::Matrix;
    ///
    /// let dtm = Matrix::from_vec(2, 3, vec![
    ///     1.0, 2.0, 0.0,
    ///     0.0, 1.0, 2.0,
    /// ]).unwrap();
    ///
    /// let mut lda = LatentDirichletAllocation::new(2);
    /// lda.fit(&dtm, 5).unwrap();
    ///
    /// let doc_topics = lda.document_topics().unwrap();
    /// assert_eq!(doc_topics.n_rows(), 2);  // 2 documents
    /// assert_eq!(doc_topics.n_cols(), 2);  // 2 topics
    /// ```
    pub fn document_topics(&self) -> Result<&Matrix<f64>, AprenderError> {
        self.doc_topic
            .as_ref()
            .ok_or_else(|| AprenderError::Other("Model not fitted. Call fit() first".to_string()))
    }

    /// Get topic-word distribution.
    ///
    /// Returns matrix where each row is a topic's word distribution.
    ///
    /// # Examples
    ///
    /// ```
    /// use aprender::text::topic::LatentDirichletAllocation;
    /// use aprender::primitives::Matrix;
    ///
    /// let dtm = Matrix::from_vec(2, 3, vec![
    ///     1.0, 2.0, 0.0,
    ///     0.0, 1.0, 2.0,
    /// ]).unwrap();
    ///
    /// let mut lda = LatentDirichletAllocation::new(2);
    /// lda.fit(&dtm, 5).unwrap();
    ///
    /// let topic_words = lda.topic_words().unwrap();
    /// assert_eq!(topic_words.n_rows(), 2);  // 2 topics
    /// assert_eq!(topic_words.n_cols(), 3);  // 3 terms
    /// ```
    pub fn topic_words(&self) -> Result<&Matrix<f64>, AprenderError> {
        self.topic_word
            .as_ref()
            .ok_or_else(|| AprenderError::Other("Model not fitted. Call fit() first".to_string()))
    }

    /// Get top words for each topic.
    ///
    /// # Arguments
    ///
    /// * `vocabulary` - List of words corresponding to term indices
    /// * `n_words` - Number of top words to return per topic
    ///
    /// # Examples
    ///
    /// ```
    /// use aprender::text::topic::LatentDirichletAllocation;
    /// use aprender::primitives::Matrix;
    ///
    /// let dtm = Matrix::from_vec(2, 3, vec![
    ///     2.0, 1.0, 0.0,
    ///     0.0, 1.0, 2.0,
    /// ]).unwrap();
    ///
    /// let mut lda = LatentDirichletAllocation::new(2);
    /// lda.fit(&dtm, 5).unwrap();
    ///
    /// let vocab = vec!["word1".to_string(), "word2".to_string(), "word3".to_string()];
    /// let top_words = lda.top_words(&vocab, 2).unwrap();
    /// assert_eq!(top_words.len(), 2);  // 2 topics
    /// ```
    pub fn top_words(
        &self,
        vocabulary: &[String],
        n_words: usize,
    ) -> Result<Vec<Vec<(String, f64)>>, AprenderError> {
        let topic_word = self.topic_words()?;

        if vocabulary.len() != topic_word.n_cols() {
            return Err(AprenderError::Other(
                "Vocabulary size must match number of terms".to_string(),
            ));
        }

        let mut result = Vec::new();

        for topic_idx in 0..self.n_topics {
            let mut word_scores: Vec<(String, f64)> = vocabulary
                .iter()
                .enumerate()
                .map(|(word_idx, word)| {
                    let score = topic_word.get(topic_idx, word_idx);
                    (word.clone(), score)
                })
                .collect();

            // Sort by score descending
            word_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            // Take top n_words
            word_scores.truncate(n_words);

            result.push(word_scores);
        }

        Ok(result)
    }

    /// Normalize rows to sum to 1.
    fn normalize_rows(data: &mut [f64], n_rows: usize, n_cols: usize) {
        for i in 0..n_rows {
            let row_start = i * n_cols;
            let row_end = row_start + n_cols;
            let row_sum: f64 = data[row_start..row_end].iter().sum();

            if row_sum > 1e-10 {
                for val in &mut data[row_start..row_end] {
                    *val /= row_sum;
                }
            }
        }
    }

    /// Pseudo-random number generator (simple LCG for reproducibility).
    fn pseudo_random(&self, idx: usize) -> f64 {
        let a: u64 = 1664525;
        let c: u64 = 1013904223;
        let m: u64 = 2_u64.pow(32);

        let x = ((a.wrapping_mul(self.random_seed.wrapping_add(idx as u64))).wrapping_add(c)) % m;
        x as f64 / m as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lda_fit() {
        let dtm = Matrix::from_vec(
            3,
            5,
            vec![
                2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 2.0,
            ],
        )
        .expect("matrix should succeed");

        let mut lda = LatentDirichletAllocation::new(2);
        lda.fit(&dtm, 10).expect("fit should succeed");

        let doc_topics = lda.document_topics().expect("should have doc topics");
        assert_eq!(doc_topics.n_rows(), 3);
        assert_eq!(doc_topics.n_cols(), 2);
    }

    #[test]
    fn test_lda_top_words() {
        let dtm = Matrix::from_vec(2, 3, vec![2.0, 1.0, 0.0, 0.0, 1.0, 2.0])
            .expect("matrix should succeed");

        let mut lda = LatentDirichletAllocation::new(2);
        lda.fit(&dtm, 10).expect("fit should succeed");

        let vocab = vec![
            "word1".to_string(),
            "word2".to_string(),
            "word3".to_string(),
        ];
        let top_words = lda.top_words(&vocab, 2).expect("top words should succeed");

        assert_eq!(top_words.len(), 2); // 2 topics
        assert_eq!(top_words[0].len(), 2); // 2 words per topic
    }

    // =========================================================================
    // Extended coverage tests
    // =========================================================================

    #[test]
    fn test_lda_with_random_seed() {
        let lda = LatentDirichletAllocation::new(3).with_random_seed(12345);
        assert_eq!(lda.random_seed, 12345);
        assert_eq!(lda.n_topics, 3);
    }

    #[test]
    fn test_lda_empty_matrix_error() {
        let dtm = Matrix::from_vec(0, 0, vec![]).expect("empty matrix");
        let mut lda = LatentDirichletAllocation::new(2);
        let result = lda.fit(&dtm, 10);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("empty"));
    }

    #[test]
    fn test_document_topics_not_fitted() {
        let lda = LatentDirichletAllocation::new(2);
        let result = lda.document_topics();
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("not fitted"));
    }

    #[test]
    fn test_topic_words_not_fitted() {
        let lda = LatentDirichletAllocation::new(2);
        let result = lda.topic_words();
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("not fitted"));
    }

    #[test]
    fn test_top_words_vocab_mismatch() {
        let dtm = Matrix::from_vec(2, 3, vec![2.0, 1.0, 0.0, 0.0, 1.0, 2.0])
            .expect("matrix should succeed");

        let mut lda = LatentDirichletAllocation::new(2);
        lda.fit(&dtm, 5).expect("fit should succeed");

        // Vocabulary with wrong size (should be 3, providing 2)
        let vocab = vec!["word1".to_string(), "word2".to_string()];
        let result = lda.top_words(&vocab, 2);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("Vocabulary size"));
    }

    #[test]
    fn test_lda_debug() {
        let lda = LatentDirichletAllocation::new(5);
        let debug_str = format!("{:?}", lda);
        assert!(debug_str.contains("LatentDirichletAllocation"));
        assert!(debug_str.contains("n_topics"));
    }

    #[test]
    fn test_lda_topic_words_access() {
        let dtm = Matrix::from_vec(2, 3, vec![2.0, 1.0, 0.0, 0.0, 1.0, 2.0])
            .expect("matrix should succeed");

        let mut lda = LatentDirichletAllocation::new(2);
        lda.fit(&dtm, 5).expect("fit should succeed");

        let topic_words = lda.topic_words().expect("should have topic words");
        assert_eq!(topic_words.n_rows(), 2); // 2 topics
        assert_eq!(topic_words.n_cols(), 3); // 3 terms

        // Each row should sum to approximately 1 (normalized)
        for row in 0..topic_words.n_rows() {
            let mut sum = 0.0;
            for col in 0..topic_words.n_cols() {
                sum += topic_words.get(row, col);
            }
            assert!((sum - 1.0).abs() < 0.01, "Row {} sum: {}", row, sum);
        }
    }

    #[test]
    fn test_lda_single_topic() {
        let dtm = Matrix::from_vec(2, 3, vec![2.0, 1.0, 0.0, 0.0, 1.0, 2.0])
            .expect("matrix should succeed");

        let mut lda = LatentDirichletAllocation::new(1);
        lda.fit(&dtm, 5).expect("fit should succeed");

        let doc_topics = lda.document_topics().expect("should have doc topics");
        assert_eq!(doc_topics.n_cols(), 1);

        // With one topic, all docs should have 1.0 weight for that topic
        for row in 0..doc_topics.n_rows() {
            assert!((doc_topics.get(row, 0) - 1.0).abs() < 0.01);
        }
    }

    #[test]
    fn test_lda_many_topics() {
        let dtm = Matrix::from_vec(3, 4, vec![
            2.0, 1.0, 0.0, 1.0,
            0.0, 1.0, 2.0, 0.0,
            1.0, 0.0, 1.0, 2.0,
        ])
        .expect("matrix should succeed");

        let mut lda = LatentDirichletAllocation::new(3);
        lda.fit(&dtm, 5).expect("fit should succeed");

        let doc_topics = lda.document_topics().expect("should have doc topics");
        assert_eq!(doc_topics.n_rows(), 3);
        assert_eq!(doc_topics.n_cols(), 3);
    }

    #[test]
    fn test_lda_sparse_matrix() {
        // Matrix with many zeros
        let dtm = Matrix::from_vec(3, 5, vec![
            1.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 1.0,
        ])
        .expect("matrix should succeed");

        let mut lda = LatentDirichletAllocation::new(2);
        lda.fit(&dtm, 5).expect("fit should succeed");

        // Should still work with sparse data
        let doc_topics = lda.document_topics().expect("should have doc topics");
        assert_eq!(doc_topics.n_rows(), 3);
    }

    #[test]
    fn test_lda_top_words_all() {
        let dtm = Matrix::from_vec(2, 4, vec![
            2.0, 1.0, 0.0, 0.5,
            0.0, 1.0, 2.0, 0.5,
        ])
        .expect("matrix should succeed");

        let mut lda = LatentDirichletAllocation::new(2);
        lda.fit(&dtm, 10).expect("fit should succeed");

        let vocab = vec![
            "alpha".to_string(),
            "beta".to_string(),
            "gamma".to_string(),
            "delta".to_string(),
        ];

        // Request all words
        let top_words = lda.top_words(&vocab, 4).expect("top words should succeed");
        assert_eq!(top_words.len(), 2);
        assert_eq!(top_words[0].len(), 4);
        assert_eq!(top_words[1].len(), 4);
    }

    #[test]
    fn test_lda_reproducibility() {
        let dtm = Matrix::from_vec(2, 3, vec![2.0, 1.0, 0.0, 0.0, 1.0, 2.0])
            .expect("matrix should succeed");

        // Fit with same seed twice
        let mut lda1 = LatentDirichletAllocation::new(2).with_random_seed(42);
        lda1.fit(&dtm, 5).expect("fit should succeed");

        let mut lda2 = LatentDirichletAllocation::new(2).with_random_seed(42);
        lda2.fit(&dtm, 5).expect("fit should succeed");

        let topics1 = lda1.document_topics().expect("topics");
        let topics2 = lda2.document_topics().expect("topics");

        // Same seed should give same results
        for row in 0..topics1.n_rows() {
            for col in 0..topics1.n_cols() {
                let diff = (topics1.get(row, col) - topics2.get(row, col)).abs();
                assert!(diff < 1e-10, "Results differ at ({}, {})", row, col);
            }
        }
    }

    #[test]
    fn test_normalize_rows_zero_sum() {
        let mut data = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        LatentDirichletAllocation::normalize_rows(&mut data, 2, 3);
        // Zero rows should stay zero
        assert!(data.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_pseudo_random_range() {
        let lda = LatentDirichletAllocation::new(2).with_random_seed(123);
        for idx in 0..100 {
            let val = lda.pseudo_random(idx);
            assert!(val >= 0.0 && val < 1.0, "Value {} out of range: {}", idx, val);
        }
    }

    #[test]
    fn test_lda_default_random_seed() {
        let lda = LatentDirichletAllocation::new(3);
        assert_eq!(lda.random_seed, 42); // Default seed
    }

    #[test]
    fn test_lda_fit_with_zero_counts() {
        // Matrix where some documents have zero counts in certain terms
        let dtm = Matrix::from_vec(3, 4, vec![
            0.0, 0.0, 1.0, 2.0,  // Doc 0: no words in first two terms
            1.0, 2.0, 0.0, 0.0,  // Doc 1: no words in last two terms
            0.0, 1.0, 1.0, 0.0,  // Doc 2: sparse
        ])
        .expect("matrix");

        let mut lda = LatentDirichletAllocation::new(2);
        lda.fit(&dtm, 10).expect("fit should succeed");

        let doc_topics = lda.document_topics().expect("should have topics");
        // Each row should sum to 1
        for row in 0..doc_topics.n_rows() {
            let mut sum = 0.0;
            for col in 0..doc_topics.n_cols() {
                sum += doc_topics.get(row, col);
            }
            assert!((sum - 1.0).abs() < 0.01);
        }
    }

    #[test]
    fn test_lda_fit_single_iteration() {
        let dtm = Matrix::from_vec(2, 3, vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
            .expect("matrix");

        let mut lda = LatentDirichletAllocation::new(2);
        lda.fit(&dtm, 1).expect("fit should succeed");

        // Even with 1 iteration, we should have valid distributions
        let topics = lda.document_topics().expect("topics");
        assert_eq!(topics.n_rows(), 2);
    }

    #[test]
    fn test_lda_fit_zero_iterations() {
        let dtm = Matrix::from_vec(2, 3, vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
            .expect("matrix");

        let mut lda = LatentDirichletAllocation::new(2);
        // With zero iterations, only initialization happens
        lda.fit(&dtm, 0).expect("fit should succeed");

        let topics = lda.document_topics().expect("topics");
        assert_eq!(topics.n_rows(), 2);
    }

    #[test]
    fn test_lda_fit_many_iterations() {
        let dtm = Matrix::from_vec(3, 4, vec![
            3.0, 0.0, 0.0, 0.0,
            0.0, 3.0, 0.0, 0.0,
            0.0, 0.0, 3.0, 0.0,
        ])
        .expect("matrix");

        let mut lda = LatentDirichletAllocation::new(3);
        // More iterations should converge
        lda.fit(&dtm, 50).expect("fit should succeed");

        let topics = lda.topic_words().expect("topic words");
        assert_eq!(topics.n_rows(), 3);
        assert_eq!(topics.n_cols(), 4);
    }

    #[test]
    fn test_lda_large_document_term_matrix() {
        // Larger matrix to test performance
        let n_docs = 10;
        let n_terms = 20;
        let mut data = vec![0.0; n_docs * n_terms];
        for i in 0..n_docs {
            for j in 0..n_terms {
                // Create some pattern
                data[i * n_terms + j] = if (i + j) % 3 == 0 { 2.0 } else { 0.0 };
            }
        }

        let dtm = Matrix::from_vec(n_docs, n_terms, data).expect("matrix");

        let mut lda = LatentDirichletAllocation::new(4);
        lda.fit(&dtm, 5).expect("fit should succeed");

        let doc_topics = lda.document_topics().expect("doc topics");
        assert_eq!(doc_topics.n_rows(), n_docs);
        assert_eq!(doc_topics.n_cols(), 4);
    }

    #[test]
    fn test_top_words_more_than_available() {
        let dtm = Matrix::from_vec(2, 3, vec![1.0, 2.0, 0.0, 0.0, 1.0, 2.0])
            .expect("matrix");

        let mut lda = LatentDirichletAllocation::new(2);
        lda.fit(&dtm, 5).expect("fit should succeed");

        let vocab = vec!["a".to_string(), "b".to_string(), "c".to_string()];

        // Request more words than available
        let top_words = lda.top_words(&vocab, 10).expect("top words");
        assert_eq!(top_words.len(), 2);
        // Should truncate to available count
        assert_eq!(top_words[0].len(), 3);
        assert_eq!(top_words[1].len(), 3);
    }

    #[test]
    fn test_top_words_request_one() {
        let dtm = Matrix::from_vec(2, 3, vec![2.0, 1.0, 0.0, 0.0, 1.0, 2.0])
            .expect("matrix");

        let mut lda = LatentDirichletAllocation::new(2);
        lda.fit(&dtm, 5).expect("fit should succeed");

        let vocab = vec!["first".to_string(), "second".to_string(), "third".to_string()];

        let top_words = lda.top_words(&vocab, 1).expect("top words");
        assert_eq!(top_words.len(), 2);
        assert_eq!(top_words[0].len(), 1);
        assert_eq!(top_words[1].len(), 1);
    }

    #[test]
    fn test_top_words_request_zero() {
        let dtm = Matrix::from_vec(2, 3, vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
            .expect("matrix");

        let mut lda = LatentDirichletAllocation::new(2);
        lda.fit(&dtm, 5).expect("fit should succeed");

        let vocab = vec!["a".to_string(), "b".to_string(), "c".to_string()];

        let top_words = lda.top_words(&vocab, 0).expect("top words");
        assert_eq!(top_words.len(), 2);
        assert_eq!(top_words[0].len(), 0);
        assert_eq!(top_words[1].len(), 0);
    }

    #[test]
    fn test_lda_different_seeds_initialization() {
        // Test that different seeds initialize differently (before convergence)
        let lda1 = LatentDirichletAllocation::new(2).with_random_seed(1);
        let lda2 = LatentDirichletAllocation::new(2).with_random_seed(999);

        // Different seeds should give different pseudo-random values
        let mut any_different = false;
        for idx in 0..10 {
            if (lda1.pseudo_random(idx) - lda2.pseudo_random(idx)).abs() > 1e-10 {
                any_different = true;
                break;
            }
        }
        assert!(any_different, "Different seeds should produce different initializations");
    }

    #[test]
    fn test_normalize_rows_single_row() {
        let mut data = vec![1.0, 2.0, 3.0];
        LatentDirichletAllocation::normalize_rows(&mut data, 1, 3);
        let sum: f64 = data.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_normalize_rows_multiple_rows() {
        let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        LatentDirichletAllocation::normalize_rows(&mut data, 2, 3);

        // First row: 1+2+3=6, so 1/6, 2/6, 3/6
        assert!((data[0] - 1.0 / 6.0).abs() < 1e-10);
        assert!((data[1] - 2.0 / 6.0).abs() < 1e-10);
        assert!((data[2] - 3.0 / 6.0).abs() < 1e-10);

        // Second row: 4+5+6=15, so 4/15, 5/15, 6/15
        assert!((data[3] - 4.0 / 15.0).abs() < 1e-10);
        assert!((data[4] - 5.0 / 15.0).abs() < 1e-10);
        assert!((data[5] - 6.0 / 15.0).abs() < 1e-10);
    }

    #[test]
    fn test_normalize_rows_very_small_sum() {
        let mut data = vec![1e-15, 1e-15, 1e-15];
        LatentDirichletAllocation::normalize_rows(&mut data, 1, 3);
        // Sum is 3e-15 which is < 1e-10, so values should remain unchanged
        assert!(data[0] < 1e-14);
    }

    #[test]
    fn test_pseudo_random_deterministic() {
        let lda1 = LatentDirichletAllocation::new(2).with_random_seed(42);
        let lda2 = LatentDirichletAllocation::new(2).with_random_seed(42);

        for idx in 0..50 {
            assert_eq!(lda1.pseudo_random(idx), lda2.pseudo_random(idx));
        }
    }

    #[test]
    fn test_pseudo_random_different_seeds() {
        let lda1 = LatentDirichletAllocation::new(2).with_random_seed(1);
        let lda2 = LatentDirichletAllocation::new(2).with_random_seed(2);

        // At least one value should differ
        let mut any_different = false;
        for idx in 0..10 {
            if (lda1.pseudo_random(idx) - lda2.pseudo_random(idx)).abs() > 1e-10 {
                any_different = true;
                break;
            }
        }
        assert!(any_different);
    }

    #[test]
    fn test_lda_fit_all_zero_document() {
        // One document has all zeros
        let dtm = Matrix::from_vec(3, 3, vec![
            1.0, 2.0, 0.0,
            0.0, 0.0, 0.0,  // All zeros
            0.0, 1.0, 2.0,
        ])
        .expect("matrix");

        let mut lda = LatentDirichletAllocation::new(2);
        lda.fit(&dtm, 10).expect("fit should succeed");

        let doc_topics = lda.document_topics().expect("topics");
        assert_eq!(doc_topics.n_rows(), 3);
    }

    #[test]
    fn test_lda_fit_uniform_distribution() {
        // All documents have same word distribution
        let dtm = Matrix::from_vec(3, 4, vec![
            1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0,
        ])
        .expect("matrix");

        let mut lda = LatentDirichletAllocation::new(2);
        lda.fit(&dtm, 10).expect("fit should succeed");

        let topic_words = lda.topic_words().expect("topic words");
        // Each topic should have similar word distributions
        for topic in 0..2 {
            let mut sum = 0.0;
            for word in 0..4 {
                sum += topic_words.get(topic, word);
            }
            assert!((sum - 1.0).abs() < 0.01);
        }
    }

    #[test]
    fn test_lda_top_words_with_ties() {
        // Create a situation where multiple words have same probability
        let dtm = Matrix::from_vec(2, 4, vec![
            1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0,
        ])
        .expect("matrix");

        let mut lda = LatentDirichletAllocation::new(1);
        lda.fit(&dtm, 10).expect("fit");

        let vocab = vec!["a".to_string(), "b".to_string(), "c".to_string(), "d".to_string()];
        let top_words = lda.top_words(&vocab, 2).expect("top words");

        assert_eq!(top_words.len(), 1);
        assert_eq!(top_words[0].len(), 2);
    }

    #[test]
    fn test_lda_single_document() {
        let dtm = Matrix::from_vec(1, 3, vec![1.0, 2.0, 3.0])
            .expect("matrix");

        let mut lda = LatentDirichletAllocation::new(2);
        lda.fit(&dtm, 5).expect("fit");

        let doc_topics = lda.document_topics().expect("topics");
        assert_eq!(doc_topics.n_rows(), 1);
        assert_eq!(doc_topics.n_cols(), 2);
    }

    #[test]
    fn test_lda_single_term() {
        let dtm = Matrix::from_vec(3, 1, vec![1.0, 2.0, 3.0])
            .expect("matrix");

        let mut lda = LatentDirichletAllocation::new(2);
        lda.fit(&dtm, 5).expect("fit");

        let topic_words = lda.topic_words().expect("topic words");
        assert_eq!(topic_words.n_cols(), 1);
        // Each topic should have weight 1.0 for the only word
        for topic in 0..2 {
            assert!((topic_words.get(topic, 0) - 1.0).abs() < 0.01);
        }
    }

    #[test]
    fn test_lda_topics_equals_terms() {
        // Edge case: number of topics equals number of terms
        let dtm = Matrix::from_vec(3, 3, vec![
            2.0, 0.0, 0.0,
            0.0, 2.0, 0.0,
            0.0, 0.0, 2.0,
        ])
        .expect("matrix");

        let mut lda = LatentDirichletAllocation::new(3);
        lda.fit(&dtm, 20).expect("fit");

        let doc_topics = lda.document_topics().expect("topics");
        assert_eq!(doc_topics.n_cols(), 3);
    }

    #[test]
    fn test_lda_topics_exceeds_terms() {
        // More topics than terms
        let dtm = Matrix::from_vec(2, 2, vec![1.0, 2.0, 2.0, 1.0])
            .expect("matrix");

        let mut lda = LatentDirichletAllocation::new(5);
        lda.fit(&dtm, 5).expect("fit");

        let doc_topics = lda.document_topics().expect("topics");
        assert_eq!(doc_topics.n_cols(), 5);
    }

    #[test]
    fn test_lda_high_count_values() {
        // Test with larger count values
        let dtm = Matrix::from_vec(2, 3, vec![
            100.0, 50.0, 10.0,
            10.0, 50.0, 100.0,
        ])
        .expect("matrix");

        let mut lda = LatentDirichletAllocation::new(2);
        lda.fit(&dtm, 10).expect("fit");

        let doc_topics = lda.document_topics().expect("topics");
        for row in 0..2 {
            let mut sum = 0.0;
            for col in 0..2 {
                let val = doc_topics.get(row, col);
                assert!(val >= 0.0 && val <= 1.0);
                sum += val;
            }
            assert!((sum - 1.0).abs() < 0.01);
        }
    }

    #[test]
    fn test_lda_small_count_values() {
        // Test with very small count values
        let dtm = Matrix::from_vec(2, 3, vec![
            0.001, 0.002, 0.001,
            0.002, 0.001, 0.002,
        ])
        .expect("matrix");

        let mut lda = LatentDirichletAllocation::new(2);
        lda.fit(&dtm, 10).expect("fit");

        let topic_words = lda.topic_words().expect("topic words");
        assert_eq!(topic_words.n_rows(), 2);
    }

    #[test]
    fn test_empty_rows_error() {
        // Matrix with 0 rows but some columns
        let dtm = Matrix::from_vec(0, 3, vec![]).expect("matrix");
        let mut lda = LatentDirichletAllocation::new(2);
        let result = lda.fit(&dtm, 5);
        assert!(result.is_err());
    }

    #[test]
    fn test_empty_cols_error() {
        // Matrix with some rows but 0 columns
        let dtm = Matrix::from_vec(3, 0, vec![]).expect("matrix");
        let mut lda = LatentDirichletAllocation::new(2);
        let result = lda.fit(&dtm, 5);
        assert!(result.is_err());
    }

    #[test]
    fn test_lda_n_topics_getter() {
        let lda = LatentDirichletAllocation::new(7);
        assert_eq!(lda.n_topics, 7);
    }
}
