//! Text processing and NLP utilities.
//!
//! This module provides text preprocessing tools for Natural Language Processing:
//! - Tokenization (word-level, character-level)
//! - BPE tokenization (Byte Pair Encoding for LLMs/speech models)
//! - Stop words filtering
//! - Stemming (Porter stemmer)
//! - Vectorization (Bag of Words, TF-IDF)
//! - Sentiment analysis (lexicon-based)
//! - Topic modeling (LDA)
//! - Document similarity (cosine, Jaccard, edit distance)
//! - Entity extraction (emails, URLs, mentions, hashtags)
//! - Text summarization (`TextRank`, TF-IDF extractive)
//!
//! # Design Principles
//!
//! Following the Toyota Way and aprender's quality standards:
//! - Zero `unwrap()` calls (Cloudflare-class safety)
//! - Result-based error handling with `AprenderError`
//! - Comprehensive test coverage (≥95%)
//! - Property-based testing with proptest
//! - Pure Rust implementation (no external NLP dependencies)
//!
//! # Quick Start
//!
//! ```
//! use aprender::text::tokenize::WhitespaceTokenizer;
//! use aprender::text::Tokenizer;
//!
//! let tokenizer = WhitespaceTokenizer::new();
//! let tokens = tokenizer.tokenize("Hello, world! This is aprender.").unwrap();
//! assert_eq!(tokens, vec!["Hello,", "world!", "This", "is", "aprender."]);
//! ```
//!
//! # References
//!
//! Based on the comprehensive NLP specification:
//! `docs/specifications/nlp-models-techniques-spec.md`

pub mod bpe;
pub mod chat_template;
pub mod entities;

// Re-export key chat_template types for convenience
pub use chat_template::{
    auto_detect_template, contains_injection_patterns, create_template, detect_format_from_name,
    sanitize_user_content, AlpacaTemplate, ChatMLTemplate, ChatMessage, ChatTemplateEngine,
    HuggingFaceTemplate, Llama2Template, MistralTemplate, PhiTemplate, RawTemplate, SpecialTokens,
    TemplateFormat,
};
pub mod incremental_idf;
pub mod llama_tokenizer;
pub mod sentiment;
pub mod similarity;
pub mod stem;
pub mod stopwords;
pub mod summarize;
pub mod tokenize;
pub mod topic;
pub mod vectorize;

use crate::AprenderError;

/// Trait for text tokenization.
///
/// Tokenizers split text into smaller units (tokens) such as words or characters.
/// All tokenizers must handle edge cases gracefully and return Result for error handling.
///
/// # Examples
///
/// ```
/// use aprender::text::{Tokenizer, tokenize::WhitespaceTokenizer};
///
/// let tokenizer = WhitespaceTokenizer::new();
/// let tokens = tokenizer.tokenize("Hello world").unwrap();
/// assert_eq!(tokens, vec!["Hello", "world"]);
/// ```
pub trait Tokenizer {
    /// Tokenize the input text into a vector of string tokens.
    ///
    /// # Arguments
    ///
    /// * `text` - Input text to tokenize
    ///
    /// # Returns
    ///
    /// * `Ok(Vec<String>)` - Successfully tokenized strings
    /// * `Err(AprenderError)` - If tokenization fails (e.g., invalid encoding)
    ///
    /// # Examples
    ///
    /// ```
    /// use aprender::text::{Tokenizer, tokenize::WhitespaceTokenizer};
    ///
    /// let tokenizer = WhitespaceTokenizer::new();
    /// assert_eq!(
    ///     tokenizer.tokenize("foo bar").unwrap(),
    ///     vec!["foo", "bar"]
    /// );
    /// ```
    fn tokenize(&self, text: &str) -> Result<Vec<String>, AprenderError>;
}

// ============================================================================
// trueno-rag integration (GH-125)
// ============================================================================

/// Re-export trueno-rag types when the `rag` feature is enabled.
///
/// Provides document chunking, retrieval, and RAG pipeline capabilities
/// for document-based ML workflows.
///
/// # Example
///
/// ```ignore
/// use aprender::text::rag::{Chunker, ChunkingStrategy};
///
/// let chunker = Chunker::new(ChunkingStrategy::Recursive {
///     chunk_size: 512,
///     overlap: 64,
/// });
/// let chunks = chunker.chunk(&document)?;
/// ```
#[cfg(feature = "rag")]
pub mod rag {
    // RAG (Retrieval-Augmented Generation) pipeline integration.
    // Re-exports from `trueno-rag` for document chunking, retrieval, and metrics.

    pub use trueno_rag::*;
}
