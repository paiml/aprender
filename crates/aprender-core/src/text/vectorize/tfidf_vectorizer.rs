
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
