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
//! // English stop words: filter, check, count
//! let filter = StopWordsFilter::english();
//! let tokens = vec!["the", "quick", "brown", "fox"];
//! let filtered = filter.filter(&tokens).expect("filter should succeed");
//! assert_eq!(filtered, vec!["quick", "brown", "fox"]);
//! assert!(filter.is_stop_word("the"));
//! assert_eq!(filter.len(), 171);
//!
//! // Custom stop words
//! let custom = StopWordsFilter::new(vec!["foo", "bar"]);
//! assert_eq!(custom.len(), 2);
//! assert!(custom.is_stop_word("FOO")); // case-insensitive
//! ```

use crate::AprenderError;
use std::collections::HashSet;

/// Stop words filter that removes common words from token lists.
///
/// Stop words are case-insensitive and checked using a `HashSet` for O(1) lookup.
/// Supports both borrowed (`filter`) and owned (`filter_owned`) token lists.
///
/// # Examples
///
/// ```
/// use aprender::text::stopwords::StopWordsFilter;
///
/// let filter = StopWordsFilter::english();
/// let result = filter.filter(&["the", "cat", "is", "happy"]).unwrap();
/// assert_eq!(result, vec!["cat", "happy"]);
/// ```
#[derive(Debug, Clone)]
pub struct StopWordsFilter {
    /// Set of stop words (stored in lowercase for case-insensitive matching)
    stop_words: HashSet<String>,
}

impl StopWordsFilter {
    /// Create a new stop words filter with custom stop words.
    ///
    /// Words are converted to lowercase for case-insensitive matching.
    ///
    /// ```
    /// # use aprender::text::stopwords::StopWordsFilter;
    /// let f = StopWordsFilter::new(vec!["custom", "STOP"]);
    /// assert!(f.is_stop_word("Custom")); // case-insensitive
    /// assert_eq!(f.len(), 2);
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

    /// Create a filter with the default English stop words (171 words from NLTK/sklearn).
    ///
    /// ```
    /// # use aprender::text::stopwords::StopWordsFilter;
    /// let f = StopWordsFilter::english();
    /// assert_eq!(f.len(), 171);
    /// assert!(f.is_stop_word("the"));
    /// assert!(!f.is_stop_word("machine"));
    /// ```
    #[must_use]
    pub fn english() -> Self {
        Self::new(ENGLISH_STOP_WORDS)
    }

    /// Shared filter implementation: retains non-stop-word tokens from any
    /// iterator whose items can be converted to `String` via the provided closure.
    fn retain_non_stop<I, F>(&self, iter: I, to_string: F) -> Result<Vec<String>, AprenderError>
    where
        I: Iterator,
        F: Fn(I::Item) -> String,
    {
        Ok(iter
            .map(to_string)
            .filter(|s| !self.is_stop_word(s))
            .collect())
    }

    /// Filter stop words from borrowed token slices (case-insensitive, preserves original case).
    ///
    /// ```
    /// # use aprender::text::stopwords::StopWordsFilter;
    /// let f = StopWordsFilter::english();
    /// // Mixed case: stop words removed, content words kept with original case
    /// let out = f.filter(&["The", "Cat", "IS", "happy"]).unwrap();
    /// assert_eq!(out, vec!["Cat", "happy"]);
    /// ```
    pub fn filter<S: AsRef<str>>(&self, tokens: &[S]) -> Result<Vec<String>, AprenderError> {
        self.retain_non_stop(tokens.iter(), |token| token.as_ref().to_string())
    }

    /// Filter stop words from owned strings (avoids cloning non-stop-word tokens).
    ///
    /// ```
    /// # use aprender::text::stopwords::StopWordsFilter;
    /// let owned = vec!["the".into(), "cat".into(), "is".into(), "happy".into()];
    /// let out = StopWordsFilter::english().filter_owned(owned).unwrap();
    /// assert_eq!(out, vec!["cat", "happy"]);
    /// ```
    pub fn filter_owned(&self, tokens: Vec<String>) -> Result<Vec<String>, AprenderError> {
        self.retain_non_stop(tokens.into_iter(), |token| token)
    }

    /// Check if a word is a stop word (case-insensitive).
    ///
    /// ```
    /// # use aprender::text::stopwords::StopWordsFilter;
    /// let f = StopWordsFilter::english();
    /// assert!(f.is_stop_word("THE"));   // case-insensitive
    /// assert!(!f.is_stop_word("rust")); // not a stop word
    /// ```
    #[must_use]
    pub fn is_stop_word(&self, word: &str) -> bool {
        self.stop_words.contains(&word.to_lowercase())
    }

    /// Number of stop words in this filter.
    #[must_use]
    pub fn len(&self) -> usize {
        self.stop_words.len()
    }

    /// Returns `true` if this filter contains no stop words.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.stop_words.is_empty()
    }
}

/// Default English stop words (171 common words).
///
/// Based on NLTK and scikit-learn stop word lists, covering articles, pronouns,
/// prepositions, conjunctions, common verbs, adverbs/adjectives, and question words.
///
/// ```
/// use aprender::text::stopwords::ENGLISH_STOP_WORDS;
/// assert!(ENGLISH_STOP_WORDS.contains(&"the"));
/// assert!(!ENGLISH_STOP_WORDS.contains(&"machine"));
/// ```
pub const ENGLISH_STOP_WORDS: &[&str] = &[
    // Articles + pronouns (34)
    "a",
    "an",
    "the",
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
    // Questions (9) + prepositions (45)
    "what",
    "which",
    "who",
    "whom",
    "whose",
    "why",
    "when",
    "where",
    "how",
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
    // Conjunctions (13) + verbs (26)
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
    // Adverbs/adjectives (30) + common (16)
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
