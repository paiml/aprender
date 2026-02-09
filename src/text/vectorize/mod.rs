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
//! let matrix = vectorizer.fit_transform(&documents).expect("vectorization should succeed");
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
/// let matrix = vectorizer.fit_transform(&docs).expect("fit_transform should succeed");
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
    /// let matrix = vectorizer.fit_transform(&docs).expect("fit_transform should succeed");
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
    /// vectorizer.fit(&docs).expect("fit should succeed");
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
    /// vectorizer.fit(&docs).expect("fit should succeed");
    /// let matrix = vectorizer.transform(&docs).expect("transform should succeed");
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
    /// vectorizer.fit(&docs).expect("fit should succeed");
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
    /// vectorizer.fit(&docs).expect("fit should succeed");
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
/// let matrix = vectorizer.fit_transform(&docs).expect("fit_transform should succeed");
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
    /// let matrix = vectorizer.fit_transform(&docs).expect("fit_transform should succeed");
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
    /// vectorizer.fit(&docs).expect("fit should succeed");
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
    /// vectorizer.fit(&docs).expect("fit should succeed");
    /// let matrix = vectorizer.transform(&docs).expect("transform should succeed");
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
    /// vectorizer.fit(&docs).expect("fit should succeed");
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
    /// vectorizer.fit(&docs).expect("fit should succeed");
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
    /// vectorizer.fit(&docs).expect("fit should succeed");
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
/// let matrix = vectorizer.transform(&docs).expect("transform should succeed");
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
mod tests;
