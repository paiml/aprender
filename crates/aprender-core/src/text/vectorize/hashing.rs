
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

// Vectorization contract falsification (FALSIFY-VEC-001..007)
// Refs: NLP spec §2.1.2, PMAT-348
#[cfg(test)]
mod vectorize_contract_falsify;
