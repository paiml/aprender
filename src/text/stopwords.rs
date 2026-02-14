//! Stop words filtering for text preprocessing.
//!
//! Stop words are common words (like "the", "is", "at") that carry little semantic
//! meaning and are often removed in NLP tasks to reduce noise and improve efficiency.
//!
//! This module provides:
//! - Default English stop words list (common words from NLTK/sklearn)
//! - `StopWordsFilter` for removing stop words from token lists
//! - Case-insensitive matching
//! - Customizable stop word sets
//!
//! # Examples
//!
//! ```
//! use aprender::text::stopwords::StopWordsFilter;
//!
//! let filter = StopWordsFilter::english();
//!
//! // Filter stop words from tokens
//! let tokens = vec!["the", "quick", "brown", "fox"];
//! let filtered = filter.filter(&tokens).expect("filter should succeed");
//! assert_eq!(filtered, vec!["quick", "brown", "fox"]);
//! ```

use crate::AprenderError;
use std::collections::HashSet;

/// Stop words filter that removes common words from token lists.
///
/// Stop words are case-insensitive and checked using a `HashSet` for O(1) lookup.
///
/// # Examples
///
/// ```
/// use aprender::text::stopwords::StopWordsFilter;
///
/// // Use default English stop words
/// let filter = StopWordsFilter::english();
/// let tokens = vec!["the", "cat", "is", "happy"];
/// let filtered = filter.filter(&tokens).expect("filter should succeed");
/// assert_eq!(filtered, vec!["cat", "happy"]);
///
/// // Custom stop words
/// let custom_filter = StopWordsFilter::new(vec!["foo", "bar"]);
/// let tokens = vec!["foo", "test", "bar", "data"];
/// let filtered = custom_filter.filter(&tokens).expect("filter should succeed");
/// assert_eq!(filtered, vec!["test", "data"]);
/// ```
#[derive(Debug, Clone)]
pub struct StopWordsFilter {
    /// Set of stop words (stored in lowercase for case-insensitive matching)
    stop_words: HashSet<String>,
}

impl StopWordsFilter {
    /// Create a new stop words filter with custom stop words.
    ///
    /// # Arguments
    ///
    /// * `words` - Collection of stop words (will be converted to lowercase)
    ///
    /// # Examples
    ///
    /// ```
    /// use aprender::text::stopwords::StopWordsFilter;
    ///
    /// let filter = StopWordsFilter::new(vec!["custom", "stop", "words"]);
    /// let tokens = vec!["custom", "text", "stop"];
    /// let filtered = filter.filter(&tokens).expect("filter should succeed");
    /// assert_eq!(filtered, vec!["text"]);
    /// ```
    pub fn new<I, S>(words: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let stop_words = words
            .into_iter()
            .map(|s| s.as_ref().to_lowercase())
            .collect();

        Self { stop_words }
    }

    /// Create a filter with English stop words.
    ///
    /// Uses a comprehensive list of 179 common English stop words based on
    /// NLTK and scikit-learn stop word lists.
    ///
    /// # Examples
    ///
    /// ```
    /// use aprender::text::stopwords::StopWordsFilter;
    ///
    /// let filter = StopWordsFilter::english();
    /// let tokens = vec!["the", "machine", "learning", "is", "awesome"];
    /// let filtered = filter.filter(&tokens).expect("filter should succeed");
    /// assert_eq!(filtered, vec!["machine", "learning", "awesome"]);
    /// ```
    #[must_use]
    pub fn english() -> Self {
        Self::new(ENGLISH_STOP_WORDS)
    }

    /// Filter stop words from a list of tokens.
    ///
    /// # Arguments
    ///
    /// * `tokens` - Input tokens (strings)
    ///
    /// # Returns
    ///
    /// * `Ok(Vec<String>)` - Filtered tokens (stop words removed)
    /// * `Err(AprenderError)` - If filtering fails
    ///
    /// # Examples
    ///
    /// ```
    /// use aprender::text::stopwords::StopWordsFilter;
    ///
    /// let filter = StopWordsFilter::english();
    ///
    /// // Case-insensitive filtering
    /// let tokens = vec!["The", "Cat", "IS", "happy"];
    /// let filtered = filter.filter(&tokens).expect("filter should succeed");
    /// assert_eq!(filtered, vec!["Cat", "happy"]);
    ///
    /// // Preserves original case
    /// let tokens = vec!["Machine", "learning", "the", "FUTURE"];
    /// let filtered = filter.filter(&tokens).expect("filter should succeed");
    /// assert_eq!(filtered, vec!["Machine", "learning", "FUTURE"]);
    /// ```
    pub fn filter<S: AsRef<str>>(&self, tokens: &[S]) -> Result<Vec<String>, AprenderError> {
        let filtered: Vec<String> = tokens
            .iter()
            .filter(|token| !self.is_stop_word(token.as_ref()))
            .map(|token| token.as_ref().to_string())
            .collect();

        Ok(filtered)
    }

    /// Filter stop words from a list of owned strings.
    ///
    /// This is a zero-copy version that avoids cloning non-stop-word tokens.
    ///
    /// # Arguments
    ///
    /// * `tokens` - Input tokens (owned strings)
    ///
    /// # Returns
    ///
    /// * `Ok(Vec<String>)` - Filtered tokens (stop words removed)
    /// * `Err(AprenderError)` - If filtering fails
    ///
    /// # Examples
    ///
    /// ```
    /// use aprender::text::stopwords::StopWordsFilter;
    ///
    /// let filter = StopWordsFilter::english();
    /// let tokens = vec![
    ///     "the".to_string(),
    ///     "cat".to_string(),
    ///     "is".to_string(),
    ///     "happy".to_string(),
    /// ];
    /// let filtered = filter.filter_owned(tokens).expect("filter should succeed");
    /// assert_eq!(filtered, vec!["cat", "happy"]);
    /// ```
    pub fn filter_owned(&self, tokens: Vec<String>) -> Result<Vec<String>, AprenderError> {
        let filtered: Vec<String> = tokens
            .into_iter()
            .filter(|token| !self.is_stop_word(token))
            .collect();

        Ok(filtered)
    }

    /// Check if a word is a stop word (case-insensitive).
    ///
    /// # Arguments
    ///
    /// * `word` - Word to check
    ///
    /// # Returns
    ///
    /// * `true` if the word is a stop word
    /// * `false` otherwise
    ///
    /// # Examples
    ///
    /// ```
    /// use aprender::text::stopwords::StopWordsFilter;
    ///
    /// let filter = StopWordsFilter::english();
    /// assert!(filter.is_stop_word("the"));
    /// assert!(filter.is_stop_word("THE"));
    /// assert!(!filter.is_stop_word("machine"));
    /// ```
    #[must_use]
    pub fn is_stop_word(&self, word: &str) -> bool {
        self.stop_words.contains(&word.to_lowercase())
    }

    /// Get the number of stop words in the filter.
    ///
    /// # Examples
    ///
    /// ```
    /// use aprender::text::stopwords::StopWordsFilter;
    ///
    /// let filter = StopWordsFilter::english();
    /// assert_eq!(filter.len(), 171);
    ///
    /// let custom = StopWordsFilter::new(vec!["foo", "bar"]);
    /// assert_eq!(custom.len(), 2);
    /// ```
    #[must_use]
    pub fn len(&self) -> usize {
        self.stop_words.len()
    }

    /// Check if the filter is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use aprender::text::stopwords::StopWordsFilter;
    ///
    /// let empty = StopWordsFilter::new(Vec::<String>::new());
    /// assert!(empty.is_empty());
    ///
    /// let english = StopWordsFilter::english();
    /// assert!(!english.is_empty());
    /// ```
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.stop_words.is_empty()
    }
}

/// Default English stop words (171 common words).
///
/// Based on NLTK and scikit-learn stop word lists, covering:
/// - Articles: a, an, the
/// - Pronouns: i, you, he, she, it, we, they
/// - Prepositions: in, on, at, by, for, with, to, from
/// - Conjunctions: and, or, but, if, because
/// - Common verbs: is, are, was, were, be, been, being, have, has, had, do, does, did
/// - Common adverbs: not, no, yes, very, too, so
/// - Question words: what, when, where, why, how, who, which
///
/// # Examples
///
/// ```
/// use aprender::text::stopwords::ENGLISH_STOP_WORDS;
///
/// assert!(ENGLISH_STOP_WORDS.contains(&"the"));
/// assert!(ENGLISH_STOP_WORDS.contains(&"and"));
/// assert!(!ENGLISH_STOP_WORDS.contains(&"machine"));
/// ```
pub const ENGLISH_STOP_WORDS: &[&str] = &[
    // Articles
    "a",
    "an",
    "the",
    // Pronouns
    "i",
    "me",
    "my",
    "myself",
    "we",
    "our",
    "ours",
    "ourselves",
    "you",
    "your",
    "yours",
    "yourself",
    "yourselves",
    "he",
    "him",
    "his",
    "himself",
    "she",
    "her",
    "hers",
    "herself",
    "it",
    "its",
    "itself",
    "they",
    "them",
    "their",
    "theirs",
    "themselves",
    // Question words
    "what",
    "which",
    "who",
    "whom",
    "whose",
    "why",
    "when",
    "where",
    "how",
    // Prepositions
    "about",
    "above",
    "across",
    "after",
    "against",
    "along",
    "among",
    "around",
    "at",
    "before",
    "behind",
    "below",
    "beneath",
    "beside",
    "between",
    "beyond",
    "by",
    "down",
    "during",
    "for",
    "from",
    "in",
    "inside",
    "into",
    "near",
    "of",
    "off",
    "on",
    "onto",
    "out",
    "outside",
    "over",
    "through",
    "throughout",
    "to",
    "toward",
    "under",
    "underneath",
    "until",
    "up",
    "upon",
    "with",
    "within",
    "without",
    // Conjunctions
    "and",
    "as",
    "because",
    "but",
    "if",
    "or",
    "since",
    "so",
    "than",
    "that",
    "though",
    "unless",
    "while",
    // Common verbs
    "am",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "having",
    "do",
    "does",
    "did",
    "doing",
    "would",
    "should",
    "could",
    "ought",
    "can",
    "may",
    "might",
    "must",
    "will",
    "shall",
    // Common adverbs/adjectives
    "all",
    "any",
    "both",
    "each",
    "every",
    "few",
    "more",
    "most",
    "much",
    "neither",
    "no",
    "none",
    "not",
    "one",
    "other",
    "same",
    "several",
    "some",
    "such",
    "very",
    "too",
    "only",
    "own",
    "then",
    "there",
    "these",
    "this",
    "those",
    "just",
    "now",
    "here",
    // Additional common words
    "again",
    "also",
    "another",
    "back",
    "even",
    "ever",
    "get",
    "give",
    "go",
    "got",
    "made",
    "make",
    "say",
    "see",
    "take",
    "way",
];

#[cfg(test)]
#[path = "stopwords_tests.rs"]
mod tests;
