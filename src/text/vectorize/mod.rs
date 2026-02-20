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

include!("tfidf_vectorizer.rs");
include!("hashing_vectorizer.rs");
include!("mod_part_04.rs");
