//! Text vectorization for machine learning.
//!
//! This module provides vectorization tools to convert text documents into numerical
//! feature vectors suitable for machine learning models:
//!
//! - **CountVectorizer**: Bag of Words representation (word counts)
//! - **TfidfVectorizer**: TF-IDF weighted features (term frequency-inverse document frequency)
//!
//! # Design Principles
//!
//! - Zero `unwrap()` calls (Cloudflare-class safety)
//! - Result-based error handling
//! - Comprehensive test coverage (≥95%)
//! - Integration with tokenizers and stop words
//!
//! # Quick Start
//!
//! ```
//! use aprender::text::vectorize::CountVectorizer;
//! use aprender::text::tokenize::WhitespaceTokenizer;
//!
//! let documents = vec![
//!     "the cat sat on the mat",
//!     "the dog sat on the log",
//! ];
//!
//! let mut vectorizer = CountVectorizer::new()
//!     .with_tokenizer(Box::new(WhitespaceTokenizer::new()));
//!
//! let matrix = vectorizer.fit_transform(&documents).unwrap();
//! // matrix shape: (2 documents, vocabulary_size features)
//! ```

use crate::primitives::Matrix;
use crate::text::Tokenizer;
use crate::AprenderError;
use std::collections::HashMap;

/// Bag of Words vectorizer that converts text to word count matrix.
///
/// Transforms a collection of text documents into a matrix of token counts.
/// Each row represents a document, each column represents a token in the vocabulary.
///
/// # Examples
///
/// ```
/// use aprender::text::vectorize::CountVectorizer;
/// use aprender::text::tokenize::WhitespaceTokenizer;
///
/// let docs = vec!["cat dog", "dog bird", "cat bird bird"];
///
/// let mut vectorizer = CountVectorizer::new()
///     .with_tokenizer(Box::new(WhitespaceTokenizer::new()));
///
/// let matrix = vectorizer.fit_transform(&docs).unwrap();
/// assert_eq!(matrix.n_rows(), 3);  // 3 documents
/// assert_eq!(matrix.n_cols(), 3);  // 3 unique words
/// ```
#[allow(missing_debug_implementations)]
pub struct CountVectorizer {
    /// Tokenizer for splitting text
    tokenizer: Option<Box<dyn Tokenizer>>,
    /// Vocabulary: word -> index mapping
    vocabulary: HashMap<String, usize>,
    /// Whether to convert to lowercase
    lowercase: bool,
    /// Maximum features (vocabulary size limit)
    max_features: Option<usize>,
}

impl CountVectorizer {
    /// Create a new `CountVectorizer`.
    ///
    /// # Examples
    ///
    /// ```
    /// use aprender::text::vectorize::CountVectorizer;
    ///
    /// let vectorizer = CountVectorizer::new();
    /// ```
    pub fn new() -> Self {
        Self {
            tokenizer: None,
            vocabulary: HashMap::new(),
            lowercase: true,
            max_features: None,
        }
    }

    /// Set the tokenizer to use.
    ///
    /// # Examples
    ///
    /// ```
    /// use aprender::text::vectorize::CountVectorizer;
    /// use aprender::text::tokenize::WhitespaceTokenizer;
    ///
    /// let vectorizer = CountVectorizer::new()
    ///     .with_tokenizer(Box::new(WhitespaceTokenizer::new()));
    /// ```
    pub fn with_tokenizer(mut self, tokenizer: Box<dyn Tokenizer>) -> Self {
        self.tokenizer = Some(tokenizer);
        self
    }

    /// Set whether to convert to lowercase.
    ///
    /// # Examples
    ///
    /// ```
    /// use aprender::text::vectorize::CountVectorizer;
    ///
    /// let vectorizer = CountVectorizer::new().with_lowercase(false);
    /// ```
    pub fn with_lowercase(mut self, lowercase: bool) -> Self {
        self.lowercase = lowercase;
        self
    }

    /// Set maximum vocabulary size.
    ///
    /// # Examples
    ///
    /// ```
    /// use aprender::text::vectorize::CountVectorizer;
    ///
    /// let vectorizer = CountVectorizer::new().with_max_features(1000);
    /// ```
    pub fn with_max_features(mut self, max_features: usize) -> Self {
        self.max_features = Some(max_features);
        self
    }

    /// Learn vocabulary from documents and transform to count matrix.
    ///
    /// # Arguments
    ///
    /// * `documents` - Collection of text documents
    ///
    /// # Returns
    ///
    /// * `Ok(Matrix<f64>)` - Count matrix (n_documents × vocabulary_size)
    /// * `Err(AprenderError)` - If vectorization fails
    ///
    /// # Examples
    ///
    /// ```
    /// use aprender::text::vectorize::CountVectorizer;
    /// use aprender::text::tokenize::WhitespaceTokenizer;
    ///
    /// let docs = vec!["hello world", "hello rust"];
    ///
    /// let mut vectorizer = CountVectorizer::new()
    ///     .with_tokenizer(Box::new(WhitespaceTokenizer::new()));
    ///
    /// let matrix = vectorizer.fit_transform(&docs).unwrap();
    /// assert_eq!(matrix.n_rows(), 2);
    /// ```
    pub fn fit_transform<S: AsRef<str>>(
        &mut self,
        documents: &[S],
    ) -> Result<Matrix<f64>, AprenderError> {
        self.fit(documents)?;
        self.transform(documents)
    }

    /// Learn vocabulary from documents.
    ///
    /// # Arguments
    ///
    /// * `documents` - Collection of text documents
    ///
    /// # Examples
    ///
    /// ```
    /// use aprender::text::vectorize::CountVectorizer;
    /// use aprender::text::tokenize::WhitespaceTokenizer;
    ///
    /// let docs = vec!["hello world"];
    /// let mut vectorizer = CountVectorizer::new()
    ///     .with_tokenizer(Box::new(WhitespaceTokenizer::new()));
    ///
    /// vectorizer.fit(&docs).unwrap();
    /// assert_eq!(vectorizer.vocabulary_size(), 2);
    /// ```
    pub fn fit<S: AsRef<str>>(&mut self, documents: &[S]) -> Result<(), AprenderError> {
        if documents.is_empty() {
            return Err(AprenderError::Other(
                "Cannot fit on empty documents".to_string(),
            ));
        }

        let tokenizer = self.tokenizer.as_ref().ok_or_else(|| {
            AprenderError::Other("Tokenizer not set. Use with_tokenizer()".to_string())
        })?;

        // Build vocabulary from all documents
        let mut word_counts: HashMap<String, usize> = HashMap::new();

        for doc in documents {
            let text = doc.as_ref();
            let tokens = tokenizer.tokenize(text)?;

            for token in tokens {
                let word = if self.lowercase {
                    token.to_lowercase()
                } else {
                    token
                };
                *word_counts.entry(word).or_insert(0) += 1;
            }
        }

        // Sort by frequency and limit vocabulary size
        let mut sorted_words: Vec<(String, usize)> = word_counts.into_iter().collect();
        sorted_words.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

        if let Some(max_features) = self.max_features {
            sorted_words.truncate(max_features);
        }

        // Build vocabulary mapping
        self.vocabulary = sorted_words
            .into_iter()
            .enumerate()
            .map(|(idx, (word, _))| (word, idx))
            .collect();

        Ok(())
    }

    /// Transform documents to count matrix using learned vocabulary.
    ///
    /// # Arguments
    ///
    /// * `documents` - Collection of text documents
    ///
    /// # Returns
    ///
    /// * `Ok(Matrix<f64>)` - Count matrix
    /// * `Err(AprenderError)` - If transformation fails
    ///
    /// # Examples
    ///
    /// ```
    /// use aprender::text::vectorize::CountVectorizer;
    /// use aprender::text::tokenize::WhitespaceTokenizer;
    ///
    /// let docs = vec!["hello world"];
    /// let mut vectorizer = CountVectorizer::new()
    ///     .with_tokenizer(Box::new(WhitespaceTokenizer::new()));
    ///
    /// vectorizer.fit(&docs).unwrap();
    /// let matrix = vectorizer.transform(&docs).unwrap();
    /// assert_eq!(matrix.n_rows(), 1);
    /// ```
    pub fn transform<S: AsRef<str>>(&self, documents: &[S]) -> Result<Matrix<f64>, AprenderError> {
        if documents.is_empty() {
            return Err(AprenderError::Other(
                "Cannot transform empty documents".to_string(),
            ));
        }

        if self.vocabulary.is_empty() {
            return Err(AprenderError::Other(
                "Vocabulary is empty. Call fit() first".to_string(),
            ));
        }

        let tokenizer = self
            .tokenizer
            .as_ref()
            .ok_or_else(|| AprenderError::Other("Tokenizer not set".to_string()))?;

        let n_docs = documents.len();
        let vocab_size = self.vocabulary.len();
        let mut data = vec![0.0; n_docs * vocab_size];

        for (doc_idx, doc) in documents.iter().enumerate() {
            let text = doc.as_ref();
            let tokens = tokenizer.tokenize(text)?;

            for token in tokens {
                let word = if self.lowercase {
                    token.to_lowercase()
                } else {
                    token
                };

                if let Some(&word_idx) = self.vocabulary.get(&word) {
                    let idx = doc_idx * vocab_size + word_idx;
                    data[idx] += 1.0;
                }
            }
        }

        Matrix::from_vec(n_docs, vocab_size, data)
            .map_err(|e: &str| AprenderError::Other(e.to_string()))
    }

    /// Get the learned vocabulary.
    ///
    /// # Examples
    ///
    /// ```
    /// use aprender::text::vectorize::CountVectorizer;
    /// use aprender::text::tokenize::WhitespaceTokenizer;
    ///
    /// let docs = vec!["hello world"];
    /// let mut vectorizer = CountVectorizer::new()
    ///     .with_tokenizer(Box::new(WhitespaceTokenizer::new()));
    ///
    /// vectorizer.fit(&docs).unwrap();
    /// let vocab = vectorizer.vocabulary();
    /// assert!(vocab.contains_key("hello"));
    /// assert!(vocab.contains_key("world"));
    /// ```
    pub fn vocabulary(&self) -> &HashMap<String, usize> {
        &self.vocabulary
    }

    /// Get the vocabulary size.
    ///
    /// # Examples
    ///
    /// ```
    /// use aprender::text::vectorize::CountVectorizer;
    /// use aprender::text::tokenize::WhitespaceTokenizer;
    ///
    /// let docs = vec!["cat dog bird"];
    /// let mut vectorizer = CountVectorizer::new()
    ///     .with_tokenizer(Box::new(WhitespaceTokenizer::new()));
    ///
    /// vectorizer.fit(&docs).unwrap();
    /// assert_eq!(vectorizer.vocabulary_size(), 3);
    /// ```
    pub fn vocabulary_size(&self) -> usize {
        self.vocabulary.len()
    }
}

impl Default for CountVectorizer {
    fn default() -> Self {
        Self::new()
    }
}

/// TF-IDF vectorizer that converts text to TF-IDF weighted matrix.
///
/// Transforms text documents into a matrix of TF-IDF (Term Frequency-Inverse Document Frequency)
/// features. TF-IDF reflects how important a word is to a document in a collection.
///
/// **TF-IDF Formula:**
/// ```text
/// tfidf(t, d) = tf(t, d) × idf(t)
/// tf(t, d) = count of term t in document d
/// idf(t) = log(N / df(t))
/// where N = total documents, df(t) = documents containing term t
/// ```
///
/// # Examples
///
/// ```
/// use aprender::text::vectorize::TfidfVectorizer;
/// use aprender::text::tokenize::WhitespaceTokenizer;
///
/// let docs = vec![
///     "the cat sat on the mat",
///     "the dog sat on the log",
/// ];
///
/// let mut vectorizer = TfidfVectorizer::new()
///     .with_tokenizer(Box::new(WhitespaceTokenizer::new()));
///
/// let matrix = vectorizer.fit_transform(&docs).unwrap();
/// assert_eq!(matrix.n_rows(), 2);  // 2 documents
/// ```
#[allow(missing_debug_implementations)]
pub struct TfidfVectorizer {
    /// Count vectorizer for term frequencies
    count_vectorizer: CountVectorizer,
    /// Inverse document frequencies
    idf_values: Vec<f64>,
}

impl TfidfVectorizer {
    /// Create a new `TfidfVectorizer`.
    ///
    /// # Examples
    ///
    /// ```
    /// use aprender::text::vectorize::TfidfVectorizer;
    ///
    /// let vectorizer = TfidfVectorizer::new();
    /// ```
    pub fn new() -> Self {
        Self {
            count_vectorizer: CountVectorizer::new(),
            idf_values: Vec::new(),
        }
    }

    /// Set the tokenizer to use.
    ///
    /// # Examples
    ///
    /// ```
    /// use aprender::text::vectorize::TfidfVectorizer;
    /// use aprender::text::tokenize::WhitespaceTokenizer;
    ///
    /// let vectorizer = TfidfVectorizer::new()
    ///     .with_tokenizer(Box::new(WhitespaceTokenizer::new()));
    /// ```
    pub fn with_tokenizer(mut self, tokenizer: Box<dyn Tokenizer>) -> Self {
        self.count_vectorizer = self.count_vectorizer.with_tokenizer(tokenizer);
        self
    }

    /// Set whether to convert to lowercase.
    ///
    /// # Examples
    ///
    /// ```
    /// use aprender::text::vectorize::TfidfVectorizer;
    ///
    /// let vectorizer = TfidfVectorizer::new().with_lowercase(false);
    /// ```
    pub fn with_lowercase(mut self, lowercase: bool) -> Self {
        self.count_vectorizer = self.count_vectorizer.with_lowercase(lowercase);
        self
    }

    /// Set maximum vocabulary size.
    ///
    /// # Examples
    ///
    /// ```
    /// use aprender::text::vectorize::TfidfVectorizer;
    ///
    /// let vectorizer = TfidfVectorizer::new().with_max_features(1000);
    /// ```
    pub fn with_max_features(mut self, max_features: usize) -> Self {
        self.count_vectorizer = self.count_vectorizer.with_max_features(max_features);
        self
    }

    /// Learn vocabulary and IDF from documents, transform to TF-IDF matrix.
    ///
    /// # Arguments
    ///
    /// * `documents` - Collection of text documents
    ///
    /// # Returns
    ///
    /// * `Ok(Matrix<f64>)` - TF-IDF matrix (n_documents × vocabulary_size)
    /// * `Err(AprenderError)` - If vectorization fails
    ///
    /// # Examples
    ///
    /// ```
    /// use aprender::text::vectorize::TfidfVectorizer;
    /// use aprender::text::tokenize::WhitespaceTokenizer;
    ///
    /// let docs = vec!["hello world", "hello rust", "world programming"];
    ///
    /// let mut vectorizer = TfidfVectorizer::new()
    ///     .with_tokenizer(Box::new(WhitespaceTokenizer::new()));
    ///
    /// let matrix = vectorizer.fit_transform(&docs).unwrap();
    /// assert_eq!(matrix.n_rows(), 3);
    /// ```
    pub fn fit_transform<S: AsRef<str>>(
        &mut self,
        documents: &[S],
    ) -> Result<Matrix<f64>, AprenderError> {
        self.fit(documents)?;
        self.transform(documents)
    }

    /// Learn vocabulary and IDF values from documents.
    ///
    /// # Arguments
    ///
    /// * `documents` - Collection of text documents
    ///
    /// # Examples
    ///
    /// ```
    /// use aprender::text::vectorize::TfidfVectorizer;
    /// use aprender::text::tokenize::WhitespaceTokenizer;
    ///
    /// let docs = vec!["hello world"];
    /// let mut vectorizer = TfidfVectorizer::new()
    ///     .with_tokenizer(Box::new(WhitespaceTokenizer::new()));
    ///
    /// vectorizer.fit(&docs).unwrap();
    /// assert_eq!(vectorizer.vocabulary_size(), 2);
    /// ```
    pub fn fit<S: AsRef<str>>(&mut self, documents: &[S]) -> Result<(), AprenderError> {
        // Fit count vectorizer to build vocabulary
        self.count_vectorizer.fit(documents)?;

        // Get count matrix to compute IDF
        let count_matrix = self.count_vectorizer.transform(documents)?;

        // Compute document frequency for each term
        let vocab_size = self.count_vectorizer.vocabulary_size();
        let n_docs = documents.len() as f64;

        let mut doc_freq = vec![0.0; vocab_size];

        #[allow(clippy::needless_range_loop)]
        for col in 0..vocab_size {
            for row in 0..count_matrix.n_rows() {
                if count_matrix.get(row, col) > 0.0 {
                    doc_freq[col] += 1.0;
                }
            }
        }

        // Compute IDF: log(N / df) with smoothing
        self.idf_values = doc_freq
            .iter()
            .map(|&df| ((n_docs + 1.0) / (df + 1.0)).ln() + 1.0)
            .collect();

        Ok(())
    }

    /// Transform documents to TF-IDF matrix using learned vocabulary and IDF.
    ///
    /// # Arguments
    ///
    /// * `documents` - Collection of text documents
    ///
    /// # Returns
    ///
    /// * `Ok(Matrix<f64>)` - TF-IDF matrix
    /// * `Err(AprenderError)` - If transformation fails
    ///
    /// # Examples
    ///
    /// ```
    /// use aprender::text::vectorize::TfidfVectorizer;
    /// use aprender::text::tokenize::WhitespaceTokenizer;
    ///
    /// let docs = vec!["hello world"];
    /// let mut vectorizer = TfidfVectorizer::new()
    ///     .with_tokenizer(Box::new(WhitespaceTokenizer::new()));
    ///
    /// vectorizer.fit(&docs).unwrap();
    /// let matrix = vectorizer.transform(&docs).unwrap();
    /// assert_eq!(matrix.n_rows(), 1);
    /// ```
    pub fn transform<S: AsRef<str>>(&self, documents: &[S]) -> Result<Matrix<f64>, AprenderError> {
        if self.idf_values.is_empty() {
            return Err(AprenderError::Other(
                "IDF not computed. Call fit() first".to_string(),
            ));
        }

        // Get count matrix (TF)
        let tf_matrix = self.count_vectorizer.transform(documents)?;

        // Apply IDF weighting
        let n_docs = tf_matrix.n_rows();
        let vocab_size = tf_matrix.n_cols();
        let mut tfidf_data = Vec::with_capacity(n_docs * vocab_size);

        for row in 0..n_docs {
            for col in 0..vocab_size {
                let tf = tf_matrix.get(row, col);
                let idf = self.idf_values[col];
                tfidf_data.push(tf * idf);
            }
        }

        Matrix::from_vec(n_docs, vocab_size, tfidf_data)
            .map_err(|e: &str| AprenderError::Other(e.to_string()))
    }

    /// Get the learned vocabulary.
    ///
    /// # Examples
    ///
    /// ```
    /// use aprender::text::vectorize::TfidfVectorizer;
    /// use aprender::text::tokenize::WhitespaceTokenizer;
    ///
    /// let docs = vec!["hello world"];
    /// let mut vectorizer = TfidfVectorizer::new()
    ///     .with_tokenizer(Box::new(WhitespaceTokenizer::new()));
    ///
    /// vectorizer.fit(&docs).unwrap();
    /// let vocab = vectorizer.vocabulary();
    /// assert!(vocab.contains_key("hello"));
    /// ```
    pub fn vocabulary(&self) -> &HashMap<String, usize> {
        self.count_vectorizer.vocabulary()
    }

    /// Get the vocabulary size.
    ///
    /// # Examples
    ///
    /// ```
    /// use aprender::text::vectorize::TfidfVectorizer;
    /// use aprender::text::tokenize::WhitespaceTokenizer;
    ///
    /// let docs = vec!["cat dog bird"];
    /// let mut vectorizer = TfidfVectorizer::new()
    ///     .with_tokenizer(Box::new(WhitespaceTokenizer::new()));
    ///
    /// vectorizer.fit(&docs).unwrap();
    /// assert_eq!(vectorizer.vocabulary_size(), 3);
    /// ```
    pub fn vocabulary_size(&self) -> usize {
        self.count_vectorizer.vocabulary_size()
    }

    /// Get the IDF values.
    ///
    /// # Examples
    ///
    /// ```
    /// use aprender::text::vectorize::TfidfVectorizer;
    /// use aprender::text::tokenize::WhitespaceTokenizer;
    ///
    /// let docs = vec!["hello world", "hello rust"];
    /// let mut vectorizer = TfidfVectorizer::new()
    ///     .with_tokenizer(Box::new(WhitespaceTokenizer::new()));
    ///
    /// vectorizer.fit(&docs).unwrap();
    /// assert_eq!(vectorizer.idf_values().len(), 3);  // 3 unique words
    /// ```
    pub fn idf_values(&self) -> &[f64] {
        &self.idf_values
    }
}

impl Default for TfidfVectorizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::text::tokenize::WhitespaceTokenizer;

    #[test]
    fn test_count_vectorizer_basic() {
        let docs = vec!["cat dog", "dog bird", "cat bird bird"];

        let mut vectorizer =
            CountVectorizer::new().with_tokenizer(Box::new(WhitespaceTokenizer::new()));

        let matrix = vectorizer
            .fit_transform(&docs)
            .expect("fit_transform should succeed");

        assert_eq!(matrix.n_rows(), 3);
        assert_eq!(matrix.n_cols(), 3); // 3 unique words
    }

    #[test]
    fn test_count_vectorizer_vocabulary() {
        let docs = vec!["hello world", "hello rust"];

        let mut vectorizer =
            CountVectorizer::new().with_tokenizer(Box::new(WhitespaceTokenizer::new()));

        vectorizer.fit(&docs).expect("fit should succeed");

        let vocab = vectorizer.vocabulary();
        assert_eq!(vocab.len(), 3);
        assert!(vocab.contains_key("hello"));
        assert!(vocab.contains_key("world"));
        assert!(vocab.contains_key("rust"));
    }

    #[test]
    fn test_tfidf_vectorizer_basic() {
        let docs = vec!["hello world", "hello rust", "world programming"];

        let mut vectorizer =
            TfidfVectorizer::new().with_tokenizer(Box::new(WhitespaceTokenizer::new()));

        let matrix = vectorizer
            .fit_transform(&docs)
            .expect("fit_transform should succeed");

        assert_eq!(matrix.n_rows(), 3);
        assert_eq!(vectorizer.vocabulary_size(), 4);
    }

    #[test]
    fn test_tfidf_idf_values() {
        let docs = vec!["hello world", "hello rust"];

        let mut vectorizer =
            TfidfVectorizer::new().with_tokenizer(Box::new(WhitespaceTokenizer::new()));

        vectorizer.fit(&docs).expect("fit should succeed");

        let idf = vectorizer.idf_values();
        assert_eq!(idf.len(), 3);
        // All IDF values should be positive
        for &value in idf {
            assert!(value > 0.0);
        }
    }
}
