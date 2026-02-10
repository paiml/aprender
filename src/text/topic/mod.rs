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
//! ]).expect("matrix creation should succeed");
//!
//! let mut lda = LatentDirichletAllocation::new(2);  // 2 topics
//! lda.fit(&dtm, 10).expect("fit should succeed");  // 10 iterations
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
/// ]).expect("matrix creation should succeed");
///
/// let mut lda = LatentDirichletAllocation::new(2);
/// lda.fit(&dtm, 5).expect("fit should succeed");
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
    /// ]).expect("matrix creation should succeed");
    ///
    /// let mut lda = LatentDirichletAllocation::new(2);
    /// lda.fit(&dtm, 10).expect("fit should succeed");
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
    /// ]).expect("matrix creation should succeed");
    ///
    /// let mut lda = LatentDirichletAllocation::new(2);
    /// lda.fit(&dtm, 5).expect("fit should succeed");
    ///
    /// let doc_topics = lda.document_topics().expect("model should be fitted");
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
    /// ]).expect("matrix creation should succeed");
    ///
    /// let mut lda = LatentDirichletAllocation::new(2);
    /// lda.fit(&dtm, 5).expect("fit should succeed");
    ///
    /// let topic_words = lda.topic_words().expect("model should be fitted");
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
    /// ]).expect("matrix creation should succeed");
    ///
    /// let mut lda = LatentDirichletAllocation::new(2);
    /// lda.fit(&dtm, 5).expect("fit should succeed");
    ///
    /// let vocab = vec!["word1".to_string(), "word2".to_string(), "word3".to_string()];
    /// let top_words = lda.top_words(&vocab, 2).expect("top_words should succeed");
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
mod tests;
