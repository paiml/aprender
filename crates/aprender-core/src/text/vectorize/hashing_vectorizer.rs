
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
