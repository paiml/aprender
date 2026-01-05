//! Text vectorization for machine learning.
//!
//! This module provides vectorization tools to convert text documents into numerical
//! feature vectors suitable for machine learning models:
//!
//! - **`CountVectorizer`**: Bag of Words representation (word counts)
//! - **`TfidfVectorizer`**: TF-IDF weighted features (term frequency-inverse document frequency)
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
use crate::text::stopwords::StopWordsFilter;
use crate::text::Tokenizer;
use crate::AprenderError;
use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

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
    /// N-gram range (min, max)
    ngram_range: (usize, usize),
    /// Minimum document frequency (absolute count)
    min_df: usize,
    /// Maximum document frequency (ratio 0.0-1.0)
    max_df: f32,
    /// Stop words filter
    stop_words: Option<StopWordsFilter>,
    /// Strip accents (unicode normalization)
    strip_accents: bool,
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
    #[must_use]
    pub fn new() -> Self {
        Self {
            tokenizer: None,
            vocabulary: HashMap::new(),
            lowercase: true,
            max_features: None,
            ngram_range: (1, 1),
            min_df: 1,
            max_df: 1.0,
            stop_words: None,
            strip_accents: false,
        }
    }

    /// Use English stop words (removes common words like "the", "and", "is").
    #[must_use]
    pub fn with_stop_words_english(mut self) -> Self {
        self.stop_words = Some(StopWordsFilter::english());
        self
    }

    /// Use custom stop words.
    #[must_use]
    pub fn with_stop_words(mut self, words: &[&str]) -> Self {
        self.stop_words = Some(StopWordsFilter::new(words));
        self
    }

    /// Strip accents/diacritics (e.g., "café" → "cafe").
    #[must_use]
    pub fn with_strip_accents(mut self, enable: bool) -> Self {
        self.strip_accents = enable;
        self
    }

    /// Set n-gram range for feature extraction.
    ///
    /// # Examples
    ///
    /// ```
    /// use aprender::text::vectorize::CountVectorizer;
    ///
    /// // Extract unigrams, bigrams, and trigrams
    /// let vectorizer = CountVectorizer::new().with_ngram_range(1, 3);
    /// ```
    #[must_use]
    pub fn with_ngram_range(mut self, min_n: usize, max_n: usize) -> Self {
        self.ngram_range = (min_n.max(1), max_n.max(1));
        self
    }

    /// Set minimum document frequency threshold.
    ///
    /// Terms appearing in fewer than `min_df` documents are ignored.
    #[must_use]
    pub fn with_min_df(mut self, min_df: usize) -> Self {
        self.min_df = min_df;
        self
    }

    /// Set maximum document frequency threshold (0.0-1.0).
    ///
    /// Terms appearing in more than `max_df` fraction of documents are ignored.
    #[must_use]
    pub fn with_max_df(mut self, max_df: f32) -> Self {
        self.max_df = max_df.clamp(0.0, 1.0);
        self
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
    #[must_use]
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
    #[must_use]
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
    #[must_use]
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
    /// * `Ok(Matrix<f64>)` - Count matrix (`n_documents` × `vocabulary_size`)
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

        let n_docs = documents.len();
        // Track term frequency and document frequency
        let mut term_freq: HashMap<String, usize> = HashMap::new();
        let mut doc_freq: HashMap<String, usize> = HashMap::new();

        for doc in documents {
            let text = doc.as_ref();
            let tokens = tokenizer.tokenize(text)?;

            // Process tokens: lowercase, strip accents, filter stop words
            let tokens: Vec<String> = tokens
                .into_iter()
                .map(|t| {
                    let mut t = if self.lowercase { t.to_lowercase() } else { t };
                    if self.strip_accents {
                        t = strip_accents_unicode(&t);
                    }
                    t
                })
                .filter(|t| {
                    self.stop_words
                        .as_ref()
                        .map_or(true, |sw| !sw.is_stop_word(t))
                })
                .collect();

            // Generate n-grams and track per-document unique terms
            let mut doc_terms: std::collections::HashSet<String> = std::collections::HashSet::new();

            for n in self.ngram_range.0..=self.ngram_range.1 {
                for ngram in tokens.windows(n) {
                    let term = ngram.join("_");
                    *term_freq.entry(term.clone()).or_insert(0) += 1;
                    doc_terms.insert(term);
                }
            }

            // Update document frequency
            for term in doc_terms {
                *doc_freq.entry(term).or_insert(0) += 1;
            }
        }

        // Filter by document frequency
        let max_df_count = (self.max_df * n_docs as f32).ceil() as usize;
        let filtered: Vec<(String, usize)> = term_freq
            .into_iter()
            .filter(|(term, _)| {
                let df = doc_freq.get(term).copied().unwrap_or(0);
                df >= self.min_df && df <= max_df_count
            })
            .collect();

        // Sort by frequency and limit vocabulary size
        let mut sorted_words: Vec<(String, usize)> = filtered;
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

            // Process tokens matching fit()
            let tokens: Vec<String> = tokens
                .into_iter()
                .map(|t| {
                    let mut t = if self.lowercase { t.to_lowercase() } else { t };
                    if self.strip_accents {
                        t = strip_accents_unicode(&t);
                    }
                    t
                })
                .filter(|t| {
                    self.stop_words
                        .as_ref()
                        .map_or(true, |sw| !sw.is_stop_word(t))
                })
                .collect();

            // Generate n-grams matching fit()
            for n in self.ngram_range.0..=self.ngram_range.1 {
                for ngram in tokens.windows(n) {
                    let term = ngram.join("_");
                    if let Some(&word_idx) = self.vocabulary.get(&term) {
                        let idx = doc_idx * vocab_size + word_idx;
                        data[idx] += 1.0;
                    }
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
    #[must_use]
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
    #[must_use]
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
    /// Use sublinear TF scaling: tf = 1 + log(tf) if tf > 0
    sublinear_tf: bool,
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
    #[must_use]
    pub fn new() -> Self {
        Self {
            count_vectorizer: CountVectorizer::new(),
            idf_values: Vec::new(),
            sublinear_tf: false,
        }
    }

    /// Enable sublinear TF scaling: tf = 1 + log(tf) if tf > 0.
    ///
    /// Dampens the effect of high term frequencies.
    #[must_use]
    pub fn with_sublinear_tf(mut self, enable: bool) -> Self {
        self.sublinear_tf = enable;
        self
    }

    /// Set n-gram range for feature extraction.
    #[must_use]
    pub fn with_ngram_range(mut self, min_n: usize, max_n: usize) -> Self {
        self.count_vectorizer = self.count_vectorizer.with_ngram_range(min_n, max_n);
        self
    }

    /// Set minimum document frequency threshold.
    #[must_use]
    pub fn with_min_df(mut self, min_df: usize) -> Self {
        self.count_vectorizer = self.count_vectorizer.with_min_df(min_df);
        self
    }

    /// Set maximum document frequency threshold (0.0-1.0).
    #[must_use]
    pub fn with_max_df(mut self, max_df: f32) -> Self {
        self.count_vectorizer = self.count_vectorizer.with_max_df(max_df);
        self
    }

    /// Use English stop words.
    #[must_use]
    pub fn with_stop_words_english(mut self) -> Self {
        self.count_vectorizer = self.count_vectorizer.with_stop_words_english();
        self
    }

    /// Use custom stop words.
    #[must_use]
    pub fn with_custom_stop_words(mut self, words: &[&str]) -> Self {
        self.count_vectorizer = self.count_vectorizer.with_stop_words(words);
        self
    }

    /// Strip accents/diacritics.
    #[must_use]
    pub fn with_strip_accents(mut self, enable: bool) -> Self {
        self.count_vectorizer = self.count_vectorizer.with_strip_accents(enable);
        self
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
    #[must_use]
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
    #[must_use]
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
    #[must_use]
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
    /// * `Ok(Matrix<f64>)` - TF-IDF matrix (`n_documents` × `vocabulary_size`)
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

        // Apply IDF weighting with optional sublinear TF scaling
        let n_docs = tf_matrix.n_rows();
        let vocab_size = tf_matrix.n_cols();
        let mut tfidf_data = Vec::with_capacity(n_docs * vocab_size);

        for row in 0..n_docs {
            for col in 0..vocab_size {
                let raw_tf = tf_matrix.get(row, col);
                // Apply sublinear TF: tf = 1 + log(tf) if tf > 0
                let tf = if self.sublinear_tf && raw_tf > 0.0 {
                    1.0 + raw_tf.ln()
                } else {
                    raw_tf
                };
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
    #[must_use]
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
    #[must_use]
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
    #[must_use]
    pub fn idf_values(&self) -> &[f64] {
        &self.idf_values
    }
}

impl Default for TfidfVectorizer {
    fn default() -> Self {
        Self::new()
    }
}

/// Strip accents/diacritics from text using Unicode normalization.
///
/// Converts characters like "café" to "cafe", "naïve" to "naive".
fn strip_accents_unicode(text: &str) -> String {
    text.chars()
        .map(|c| {
            // Simple ASCII folding for common accented characters
            match c {
                'á' | 'à' | 'â' | 'ä' | 'ã' | 'å' => 'a',
                'é' | 'è' | 'ê' | 'ë' => 'e',
                'í' | 'ì' | 'î' | 'ï' => 'i',
                'ó' | 'ò' | 'ô' | 'ö' | 'õ' => 'o',
                'ú' | 'ù' | 'û' | 'ü' => 'u',
                'ý' | 'ÿ' => 'y',
                'ñ' => 'n',
                'ç' => 'c',
                'Á' | 'À' | 'Â' | 'Ä' | 'Ã' | 'Å' => 'A',
                'É' | 'È' | 'Ê' | 'Ë' => 'E',
                'Í' | 'Ì' | 'Î' | 'Ï' => 'I',
                'Ó' | 'Ò' | 'Ô' | 'Ö' | 'Õ' => 'O',
                'Ú' | 'Ù' | 'Û' | 'Ü' => 'U',
                'Ý' => 'Y',
                'Ñ' => 'N',
                'Ç' => 'C',
                _ => c,
            }
        })
        .collect()
}

/// Stateless hashing vectorizer for streaming/large-scale text.
///
/// Maps tokens to feature indices using a hash function. Does not store
/// a vocabulary, making it memory-efficient but irreversible.
///
/// # Examples
///
/// ```
/// use aprender::text::vectorize::HashingVectorizer;
/// use aprender::text::tokenize::WhitespaceTokenizer;
///
/// let docs = vec!["hello world", "hello rust"];
///
/// let vectorizer = HashingVectorizer::new(1000)
///     .with_tokenizer(Box::new(WhitespaceTokenizer::new()));
///
/// let matrix = vectorizer.transform(&docs).unwrap();
/// assert_eq!(matrix.n_rows(), 2);
/// assert_eq!(matrix.n_cols(), 1000);
/// ```
#[allow(missing_debug_implementations)]
pub struct HashingVectorizer {
    tokenizer: Option<Box<dyn Tokenizer>>,
    n_features: usize,
    lowercase: bool,
    ngram_range: (usize, usize),
    stop_words: Option<StopWordsFilter>,
}

impl HashingVectorizer {
    /// Create a new `HashingVectorizer` with specified number of features.
    #[must_use]
    pub fn new(n_features: usize) -> Self {
        Self {
            tokenizer: None,
            n_features,
            lowercase: true,
            ngram_range: (1, 1),
            stop_words: None,
        }
    }

    /// Set the tokenizer.
    #[must_use]
    pub fn with_tokenizer(mut self, tokenizer: Box<dyn Tokenizer>) -> Self {
        self.tokenizer = Some(tokenizer);
        self
    }

    /// Set lowercase.
    #[must_use]
    pub fn with_lowercase(mut self, lowercase: bool) -> Self {
        self.lowercase = lowercase;
        self
    }

    /// Set n-gram range.
    #[must_use]
    pub fn with_ngram_range(mut self, min_n: usize, max_n: usize) -> Self {
        self.ngram_range = (min_n.max(1), max_n.max(1));
        self
    }

    /// Use English stop words.
    #[must_use]
    pub fn with_stop_words_english(mut self) -> Self {
        self.stop_words = Some(StopWordsFilter::english());
        self
    }

    /// Transform documents to hashed feature matrix (stateless).
    pub fn transform<S: AsRef<str>>(&self, documents: &[S]) -> Result<Matrix<f64>, AprenderError> {
        if documents.is_empty() {
            return Err(AprenderError::Other(
                "Cannot transform empty documents".to_string(),
            ));
        }

        let tokenizer = self
            .tokenizer
            .as_ref()
            .ok_or_else(|| AprenderError::Other("Tokenizer not set".to_string()))?;

        let n_docs = documents.len();
        let mut data = vec![0.0; n_docs * self.n_features];

        for (doc_idx, doc) in documents.iter().enumerate() {
            let tokens = tokenizer.tokenize(doc.as_ref())?;
            let tokens: Vec<String> = tokens
                .into_iter()
                .map(|t| if self.lowercase { t.to_lowercase() } else { t })
                .filter(|t| {
                    self.stop_words
                        .as_ref()
                        .map_or(true, |sw| !sw.is_stop_word(t))
                })
                .collect();

            for n in self.ngram_range.0..=self.ngram_range.1 {
                for ngram in tokens.windows(n) {
                    let term = ngram.join("_");
                    let hash_idx = hash_term(&term, self.n_features);
                    data[doc_idx * self.n_features + hash_idx] += 1.0;
                }
            }
        }

        Matrix::from_vec(n_docs, self.n_features, data)
            .map_err(|e: &str| AprenderError::Other(e.to_string()))
    }
}

/// Hash a term to a feature index.
fn hash_term(term: &str, n_features: usize) -> usize {
    let mut hasher = DefaultHasher::new();
    term.hash(&mut hasher);
    (hasher.finish() as usize) % n_features
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

    #[test]
    fn test_ngram_extraction() {
        let docs = vec!["the quick brown fox"];

        let mut vectorizer = CountVectorizer::new()
            .with_tokenizer(Box::new(WhitespaceTokenizer::new()))
            .with_ngram_range(1, 2); // unigrams and bigrams

        vectorizer.fit(&docs).expect("fit should succeed");

        let vocab = vectorizer.vocabulary();
        // Should have 4 unigrams + 3 bigrams = 7 terms
        assert_eq!(vocab.len(), 7);
        assert!(vocab.contains_key("the"));
        assert!(vocab.contains_key("the_quick")); // bigram
        assert!(vocab.contains_key("brown_fox")); // bigram
    }

    #[test]
    fn test_min_df_filtering() {
        let docs = vec!["cat dog", "cat bird", "fish"]; // cat appears in 2 docs

        let mut vectorizer = CountVectorizer::new()
            .with_tokenizer(Box::new(WhitespaceTokenizer::new()))
            .with_min_df(2); // require term in at least 2 docs

        vectorizer.fit(&docs).expect("fit should succeed");

        let vocab = vectorizer.vocabulary();
        // Only "cat" appears in 2+ docs
        assert_eq!(vocab.len(), 1);
        assert!(vocab.contains_key("cat"));
    }

    #[test]
    fn test_max_df_filtering() {
        let docs = vec!["the cat", "the dog", "the bird"]; // "the" in 100% of docs

        let mut vectorizer = CountVectorizer::new()
            .with_tokenizer(Box::new(WhitespaceTokenizer::new()))
            .with_max_df(0.5); // exclude terms in >50% of docs

        vectorizer.fit(&docs).expect("fit should succeed");

        let vocab = vectorizer.vocabulary();
        // "the" should be excluded (appears in 100% of docs)
        assert!(!vocab.contains_key("the"));
        assert_eq!(vocab.len(), 3); // cat, dog, bird
    }

    #[test]
    fn test_sublinear_tf() {
        let docs = vec!["word word word word"]; // word appears 4 times

        let mut vectorizer_normal =
            TfidfVectorizer::new().with_tokenizer(Box::new(WhitespaceTokenizer::new()));

        let mut vectorizer_sublinear = TfidfVectorizer::new()
            .with_tokenizer(Box::new(WhitespaceTokenizer::new()))
            .with_sublinear_tf(true);

        let matrix_normal = vectorizer_normal
            .fit_transform(&docs)
            .expect("fit should succeed");
        let matrix_sublinear = vectorizer_sublinear
            .fit_transform(&docs)
            .expect("fit should succeed");

        // With sublinear TF, the score should be lower (1 + ln(4) ≈ 2.39 vs 4)
        assert!(matrix_sublinear.get(0, 0) < matrix_normal.get(0, 0));
    }

    #[test]
    fn test_tfidf_full_pipeline() {
        let docs = vec![
            "machine learning is great",
            "deep learning is powerful",
            "machine learning and deep learning",
        ];

        let mut vectorizer = TfidfVectorizer::new()
            .with_tokenizer(Box::new(WhitespaceTokenizer::new()))
            .with_ngram_range(1, 2)
            .with_sublinear_tf(true);

        let matrix = vectorizer.fit_transform(&docs).expect("fit should succeed");
        assert_eq!(matrix.n_rows(), 3);
        assert!(vectorizer.vocabulary_size() > 0);
    }

    #[test]
    fn test_count_vectorizer_stop_words_english() {
        let docs = vec!["the cat and dog", "a bird is flying"];
        let mut vectorizer = CountVectorizer::new()
            .with_tokenizer(Box::new(WhitespaceTokenizer::new()))
            .with_stop_words_english();

        let _matrix = vectorizer.fit_transform(&docs).expect("fit should succeed");
        // "the", "and", "a", "is" should be filtered out
        let vocab = vectorizer.vocabulary();
        assert!(!vocab.contains_key("the"));
        assert!(!vocab.contains_key("and"));
        assert!(vocab.contains_key("cat") || vocab.contains_key("dog"));
    }

    #[test]
    fn test_count_vectorizer_custom_stop_words() {
        let docs = vec!["hello world hello", "world test"];
        let mut vectorizer = CountVectorizer::new()
            .with_tokenizer(Box::new(WhitespaceTokenizer::new()))
            .with_stop_words(&["hello"]);

        let _matrix = vectorizer.fit_transform(&docs).expect("fit should succeed");
        let vocab = vectorizer.vocabulary();
        assert!(!vocab.contains_key("hello"));
        assert!(vocab.contains_key("world"));
    }

    #[test]
    fn test_count_vectorizer_strip_accents() {
        let vectorizer = CountVectorizer::new().with_strip_accents(true);
        assert!(vectorizer.strip_accents);
    }

    #[test]
    fn test_tfidf_stop_words_english() {
        let docs = vec!["the quick brown fox", "a lazy dog"];
        let mut vectorizer = TfidfVectorizer::new()
            .with_tokenizer(Box::new(WhitespaceTokenizer::new()))
            .with_stop_words_english();

        let _matrix = vectorizer.fit_transform(&docs).expect("fit should succeed");
        let vocab = vectorizer.vocabulary();
        assert!(!vocab.contains_key("the"));
        assert!(!vocab.contains_key("a"));
    }

    #[test]
    fn test_tfidf_custom_stop_words() {
        let docs = vec!["foo bar baz", "bar qux"];
        let mut vectorizer = TfidfVectorizer::new()
            .with_tokenizer(Box::new(WhitespaceTokenizer::new()))
            .with_custom_stop_words(&["foo", "baz"]);

        vectorizer.fit(&docs).expect("fit should succeed");
        let vocab = vectorizer.vocabulary();
        assert!(!vocab.contains_key("foo"));
        assert!(!vocab.contains_key("baz"));
        assert!(vocab.contains_key("bar"));
    }

    #[test]
    fn test_tfidf_strip_accents_builder() {
        let _vectorizer = TfidfVectorizer::new().with_strip_accents(true);
        // Just verify it compiles and doesn't panic
    }

    #[test]
    fn test_hashing_vectorizer_n_features() {
        let vectorizer =
            HashingVectorizer::new(1024).with_tokenizer(Box::new(WhitespaceTokenizer::new()));
        assert_eq!(vectorizer.n_features, 1024);
    }

    #[test]
    fn test_hashing_vectorizer_ngram_range() {
        let vectorizer = HashingVectorizer::new(2048).with_ngram_range(1, 3);
        assert_eq!(vectorizer.ngram_range, (1, 3));
    }

    #[test]
    fn test_hashing_vectorizer_transform() {
        let docs = vec!["hello world", "world hello hello"];

        let vectorizer =
            HashingVectorizer::new(100).with_tokenizer(Box::new(WhitespaceTokenizer::new()));

        let matrix = vectorizer
            .transform(&docs)
            .expect("transform should succeed");

        assert_eq!(matrix.n_rows(), 2);
        assert_eq!(matrix.n_cols(), 100);
    }

    #[test]
    fn test_hashing_vectorizer_with_lowercase() {
        let vectorizer = HashingVectorizer::new(100).with_lowercase(false);
        assert!(!vectorizer.lowercase);
    }

    #[test]
    fn test_hashing_vectorizer_with_stop_words() {
        let docs = vec!["the cat and dog", "a bird"];

        let vectorizer = HashingVectorizer::new(100)
            .with_tokenizer(Box::new(WhitespaceTokenizer::new()))
            .with_stop_words_english();

        let matrix = vectorizer
            .transform(&docs)
            .expect("transform should succeed");
        assert_eq!(matrix.n_rows(), 2);
    }

    #[test]
    fn test_hashing_vectorizer_empty_docs_error() {
        let docs: Vec<&str> = vec![];

        let vectorizer =
            HashingVectorizer::new(100).with_tokenizer(Box::new(WhitespaceTokenizer::new()));

        let result = vectorizer.transform(&docs);
        assert!(result.is_err());
    }

    #[test]
    fn test_hashing_vectorizer_no_tokenizer_error() {
        let docs = vec!["hello"];

        let vectorizer = HashingVectorizer::new(100);

        let result = vectorizer.transform(&docs);
        assert!(result.is_err());
    }

    #[test]
    fn test_count_vectorizer_empty_docs_error() {
        let docs: Vec<&str> = vec![];

        let mut vectorizer =
            CountVectorizer::new().with_tokenizer(Box::new(WhitespaceTokenizer::new()));

        let result = vectorizer.fit(&docs);
        assert!(result.is_err());
    }

    #[test]
    fn test_count_vectorizer_no_tokenizer_error() {
        let docs = vec!["hello"];

        let mut vectorizer = CountVectorizer::new();

        let result = vectorizer.fit(&docs);
        assert!(result.is_err());
    }

    #[test]
    fn test_count_vectorizer_transform_empty_vocab_error() {
        let docs = vec!["hello"];

        let vectorizer =
            CountVectorizer::new().with_tokenizer(Box::new(WhitespaceTokenizer::new()));

        let result = vectorizer.transform(&docs);
        assert!(result.is_err());
    }

    #[test]
    fn test_count_vectorizer_transform_empty_docs_error() {
        let docs = vec!["hello"];
        let empty_docs: Vec<&str> = vec![];

        let mut vectorizer =
            CountVectorizer::new().with_tokenizer(Box::new(WhitespaceTokenizer::new()));

        vectorizer.fit(&docs).expect("fit should succeed");

        let result = vectorizer.transform(&empty_docs);
        assert!(result.is_err());
    }

    #[test]
    fn test_tfidf_transform_without_fit_error() {
        let docs = vec!["hello"];

        let vectorizer =
            TfidfVectorizer::new().with_tokenizer(Box::new(WhitespaceTokenizer::new()));

        let result = vectorizer.transform(&docs);
        assert!(result.is_err());
    }

    #[test]
    fn test_strip_accents_unicode() {
        assert_eq!(strip_accents_unicode("café"), "cafe");
        assert_eq!(strip_accents_unicode("naïve"), "naive");
        assert_eq!(strip_accents_unicode("résumé"), "resume");
        assert_eq!(strip_accents_unicode("CAFÉ"), "CAFE");
        assert_eq!(strip_accents_unicode("señor"), "senor");
        assert_eq!(strip_accents_unicode("façade"), "facade");
        assert_eq!(strip_accents_unicode("über"), "uber");
        assert_eq!(strip_accents_unicode("hello"), "hello");
    }

    #[test]
    fn test_strip_accents_all_characters() {
        // Test all accent mappings
        assert_eq!(strip_accents_unicode("àáâäãå"), "aaaaaa");
        assert_eq!(strip_accents_unicode("èéêë"), "eeee");
        assert_eq!(strip_accents_unicode("ìíîï"), "iiii");
        assert_eq!(strip_accents_unicode("òóôöõ"), "ooooo");
        assert_eq!(strip_accents_unicode("ùúûü"), "uuuu");
        assert_eq!(strip_accents_unicode("ýÿ"), "yy");
        assert_eq!(strip_accents_unicode("ñ"), "n");
        assert_eq!(strip_accents_unicode("ç"), "c");
        // Uppercase
        assert_eq!(strip_accents_unicode("ÀÁÂÄÃÅ"), "AAAAAA");
        assert_eq!(strip_accents_unicode("ÈÉÊË"), "EEEE");
        assert_eq!(strip_accents_unicode("ÌÍÎÏ"), "IIII");
        assert_eq!(strip_accents_unicode("ÒÓÔÖÕ"), "OOOOO");
        assert_eq!(strip_accents_unicode("ÙÚÛÜ"), "UUUU");
        assert_eq!(strip_accents_unicode("Ý"), "Y");
        assert_eq!(strip_accents_unicode("Ñ"), "N");
        assert_eq!(strip_accents_unicode("Ç"), "C");
    }

    #[test]
    fn test_count_vectorizer_with_strip_accents_integration() {
        let docs = vec!["café résumé", "cafe resume"];

        let mut vectorizer = CountVectorizer::new()
            .with_tokenizer(Box::new(WhitespaceTokenizer::new()))
            .with_strip_accents(true);

        vectorizer.fit(&docs).expect("fit should succeed");

        // After stripping accents, "café" and "cafe" should be the same
        let vocab = vectorizer.vocabulary();
        assert!(vocab.contains_key("cafe"));
        assert!(vocab.contains_key("resume"));
        // Should not have the accented versions as separate entries
        assert!(!vocab.contains_key("café"));
    }

    #[test]
    fn test_count_vectorizer_max_features() {
        let docs = vec!["a b c d e f g h i j"];

        let mut vectorizer = CountVectorizer::new()
            .with_tokenizer(Box::new(WhitespaceTokenizer::new()))
            .with_max_features(5);

        vectorizer.fit(&docs).expect("fit should succeed");

        // Should be limited to 5 features
        assert_eq!(vectorizer.vocabulary_size(), 5);
    }

    #[test]
    fn test_count_vectorizer_default() {
        let vectorizer = CountVectorizer::default();
        assert!(vectorizer.lowercase);
        assert_eq!(vectorizer.ngram_range, (1, 1));
    }

    #[test]
    fn test_tfidf_vectorizer_default() {
        let vectorizer = TfidfVectorizer::default();
        assert!(!vectorizer.sublinear_tf);
    }

    #[test]
    fn test_tfidf_with_min_df() {
        let docs = vec!["cat", "cat dog", "dog bird"]; // cat in 2, dog in 2, bird in 1

        let mut vectorizer = TfidfVectorizer::new()
            .with_tokenizer(Box::new(WhitespaceTokenizer::new()))
            .with_min_df(2);

        vectorizer.fit(&docs).expect("fit should succeed");

        let vocab = vectorizer.vocabulary();
        assert!(vocab.contains_key("cat"));
        assert!(vocab.contains_key("dog"));
        assert!(!vocab.contains_key("bird"));
    }

    #[test]
    fn test_tfidf_with_max_df() {
        let docs = vec!["the cat", "the dog", "the bird"];

        let mut vectorizer = TfidfVectorizer::new()
            .with_tokenizer(Box::new(WhitespaceTokenizer::new()))
            .with_max_df(0.5);

        vectorizer.fit(&docs).expect("fit should succeed");

        let vocab = vectorizer.vocabulary();
        assert!(!vocab.contains_key("the"));
    }

    #[test]
    fn test_tfidf_ngram_range() {
        let docs = vec!["hello world"];

        let mut vectorizer = TfidfVectorizer::new()
            .with_tokenizer(Box::new(WhitespaceTokenizer::new()))
            .with_ngram_range(1, 2);

        vectorizer.fit(&docs).expect("fit should succeed");

        let vocab = vectorizer.vocabulary();
        assert!(vocab.contains_key("hello"));
        assert!(vocab.contains_key("world"));
        assert!(vocab.contains_key("hello_world"));
    }

    #[test]
    fn test_tfidf_max_features() {
        let docs = vec!["a b c d e f g"];

        let mut vectorizer = TfidfVectorizer::new()
            .with_tokenizer(Box::new(WhitespaceTokenizer::new()))
            .with_max_features(3);

        vectorizer.fit(&docs).expect("fit should succeed");

        assert_eq!(vectorizer.vocabulary_size(), 3);
    }

    #[test]
    fn test_tfidf_lowercase() {
        let docs = vec!["Hello WORLD"];

        let mut vectorizer = TfidfVectorizer::new()
            .with_tokenizer(Box::new(WhitespaceTokenizer::new()))
            .with_lowercase(true);

        vectorizer.fit(&docs).expect("fit should succeed");

        let vocab = vectorizer.vocabulary();
        assert!(vocab.contains_key("hello"));
        assert!(vocab.contains_key("world"));
        assert!(!vocab.contains_key("Hello"));
    }

    #[test]
    fn test_hash_term_deterministic() {
        let hash1 = hash_term("test", 1000);
        let hash2 = hash_term("test", 1000);
        assert_eq!(hash1, hash2);

        // Different terms should (usually) hash differently
        let hash3 = hash_term("other", 1000);
        assert_ne!(hash1, hash3);
    }

    #[test]
    fn test_hash_term_within_range() {
        for term in &["a", "test", "hello world", "12345"] {
            let hash = hash_term(term, 100);
            assert!(hash < 100);
        }
    }

    #[test]
    fn test_hashing_vectorizer_with_ngrams() {
        let docs = vec!["a b c"];

        let vectorizer = HashingVectorizer::new(1000)
            .with_tokenizer(Box::new(WhitespaceTokenizer::new()))
            .with_ngram_range(1, 2);

        let matrix = vectorizer
            .transform(&docs)
            .expect("transform should succeed");

        // Should have features for unigrams and bigrams
        let row_sum: f64 = (0..1000).map(|i| matrix.get(0, i)).sum();
        assert!(row_sum > 3.0); // at least 3 unigrams + 2 bigrams
    }

    #[test]
    fn test_count_vectorizer_transform_unseen_words() {
        let train_docs = vec!["cat dog"];
        let test_docs = vec!["cat elephant"]; // elephant not in vocab

        let mut vectorizer =
            CountVectorizer::new().with_tokenizer(Box::new(WhitespaceTokenizer::new()));

        vectorizer.fit(&train_docs).expect("fit should succeed");

        let matrix = vectorizer
            .transform(&test_docs)
            .expect("transform should succeed");

        // Should only count "cat", not "elephant"
        let row_sum: f64 = (0..matrix.n_cols()).map(|i| matrix.get(0, i)).sum();
        assert!((row_sum - 1.0).abs() < 1e-6); // Only cat is counted
    }
}
