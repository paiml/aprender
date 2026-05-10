//! HELIX-IDEA-005 Phase 4 — pluggable tokenizer trait.
//!
//! Contract: `contracts/apr-hybrid-retrieval-v1.yaml` (FALSIFY-HYBRID-003).
//!
//! Pre-Phase-4, `BM25Index` had a hardcoded internal `tokenize()`
//! method that split on non-alphanumeric boundaries with built-in
//! lowercasing + stopword filtering. The §2.5 sketch called this
//! out as drift risk: a future `apr serve` inference path that uses
//! a different tokenizer (BPE / SentencePiece) would silently
//! disagree with BM25 on what counts as a "term", with no
//! compile-time anchor pinning the two together.
//!
//! Phase 4 extracts the contract: a public `Tokenizer` trait that
//! BM25Index can OPTIONALLY accept via [`BM25Index::with_tokenizer`].
//! The default behaviour is unchanged (the existing internal
//! tokenization stays the fallback when no override is set), so
//! existing callers don't break. Future callers — including a
//! shared inference tokenizer — implement [`Tokenizer`] and plug
//! in.
//!
//! Note on scope: this PR ships the trait surface plus the
//! BM25 plumbing to use it. Wiring the actual inference-path
//! tokenizer (Qwen / Llama BPE) into the same trait is a separate
//! effort; the gate in §2.5 only asserts the trait is pluggable
//! today, not that any specific inference tokenizer is wired.

/// Decompose a string into BM25 search terms.
///
/// Implementors choose the tokenization rule (whitespace +
/// lowercase, BPE, sentencepiece, etc.) — `BM25Index` will index
/// and search using whatever tokens this returns.
///
/// `Send + Sync` so a `BM25Index` carrying a `dyn Tokenizer` can be
/// shared across threads. `Debug` so the index's `Debug` derive
/// works without manual impl.
pub trait Tokenizer: Send + Sync + std::fmt::Debug {
    /// Decompose `text` into a sequence of indexable terms.
    fn tokenize(&self, text: &str) -> Vec<String>;
}

/// Whitespace-and-non-alphanumeric tokenizer with optional
/// lowercasing, stopword filtering, and minimum-length filtering.
/// Matches the rule that lived inside `BM25Index` pre-Phase-4 and
/// is the default when no other tokenizer is supplied.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct WhitespaceTokenizer {
    /// If true, all tokens are lowercased before being emitted.
    pub lowercase: bool,
    /// Tokens of length strictly less than this are dropped.
    pub min_token_len: usize,
    /// Tokens contained in this set are dropped (after lowercase).
    pub stopwords: std::collections::BTreeSet<String>,
}

impl Default for WhitespaceTokenizer {
    fn default() -> Self {
        Self { lowercase: true, min_token_len: 2, stopwords: std::collections::BTreeSet::new() }
    }
}

impl WhitespaceTokenizer {
    /// Construct with default settings (lowercase, min-length 2,
    /// no stopwords). Use the public fields to customise.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

impl Tokenizer for WhitespaceTokenizer {
    fn tokenize(&self, text: &str) -> Vec<String> {
        text.split(|c: char| !c.is_alphanumeric())
            .filter(|s| !s.is_empty())
            .map(|s| if self.lowercase { s.to_lowercase() } else { s.to_string() })
            .filter(|s| !self.stopwords.contains(s))
            .filter(|s| s.len() >= self.min_token_len)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn whitespace_tokenizer_default_lowercases_and_drops_short() {
        let tok = WhitespaceTokenizer::default();
        let out = tok.tokenize("Hello, World! a b c");
        assert_eq!(out, vec!["hello", "world"]);
    }

    #[test]
    fn whitespace_tokenizer_can_disable_lowercase() {
        let tok = WhitespaceTokenizer { lowercase: false, ..Default::default() };
        let out = tok.tokenize("Hello World");
        assert_eq!(out, vec!["Hello", "World"]);
    }

    #[test]
    fn whitespace_tokenizer_stopwords_drop_match() {
        let mut stopwords = std::collections::BTreeSet::new();
        stopwords.insert("hello".to_string());
        let tok = WhitespaceTokenizer { stopwords, ..Default::default() };
        let out = tok.tokenize("Hello World");
        assert_eq!(out, vec!["world"]);
    }
}
