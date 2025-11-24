//! Tokenization algorithms for text preprocessing.
//!
//! This module provides various tokenization strategies:
//! - Whitespace tokenization (splits on Unicode whitespace)
//! - Word tokenization (alphanumeric + punctuation handling)
//! - Character tokenization (splits into individual characters)
//!
//! All tokenizers implement the `Tokenizer` trait and follow zero-unwrap safety.

use crate::text::Tokenizer;
use crate::AprenderError;

/// Whitespace tokenizer that splits text on Unicode whitespace characters.
///
/// This is the simplest tokenizer, splitting on any Unicode whitespace
/// (spaces, tabs, newlines, etc.). It preserves punctuation attached to words.
///
/// # Examples
///
/// ```
/// use aprender::text::{Tokenizer, tokenize::WhitespaceTokenizer};
///
/// let tokenizer = WhitespaceTokenizer::new();
///
/// // Basic tokenization
/// let tokens = tokenizer.tokenize("Hello, world!").expect("tokenize should succeed");
/// assert_eq!(tokens, vec!["Hello,", "world!"]);
///
/// // Handles multiple spaces
/// let tokens = tokenizer.tokenize("foo   bar").expect("tokenize should succeed");
/// assert_eq!(tokens, vec!["foo", "bar"]);
///
/// // Handles newlines and tabs
/// let tokens = tokenizer.tokenize("line1\nline2\ttab").expect("tokenize should succeed");
/// assert_eq!(tokens, vec!["line1", "line2", "tab"]);
/// ```
#[derive(Debug, Clone, Default)]
pub struct WhitespaceTokenizer;

impl WhitespaceTokenizer {
    /// Create a new whitespace tokenizer.
    ///
    /// # Examples
    ///
    /// ```
    /// use aprender::text::tokenize::WhitespaceTokenizer;
    ///
    /// let tokenizer = WhitespaceTokenizer::new();
    /// ```
    pub fn new() -> Self {
        Self
    }
}

impl Tokenizer for WhitespaceTokenizer {
    fn tokenize(&self, text: &str) -> Result<Vec<String>, AprenderError> {
        // Split on whitespace and filter out empty strings
        let tokens: Vec<String> = text.split_whitespace().map(ToString::to_string).collect();

        Ok(tokens)
    }
}

/// Word tokenizer that splits on whitespace and separates punctuation.
///
/// This tokenizer is more sophisticated than whitespace splitting:
/// - Splits on Unicode whitespace
/// - Separates punctuation from words
/// - Preserves contractions (e.g., "don't" stays together)
///
/// # Examples
///
/// ```
/// use aprender::text::{Tokenizer, tokenize::WordTokenizer};
///
/// let tokenizer = WordTokenizer::new();
///
/// // Separates punctuation
/// let tokens = tokenizer.tokenize("Hello, world!").expect("tokenize should succeed");
/// assert_eq!(tokens, vec!["Hello", ",", "world", "!"]);
///
/// // Preserves contractions
/// let tokens = tokenizer.tokenize("I don't know.").expect("tokenize should succeed");
/// assert_eq!(tokens, vec!["I", "don't", "know", "."]);
/// ```
#[derive(Debug, Clone, Default)]
pub struct WordTokenizer;

impl WordTokenizer {
    /// Create a new word tokenizer.
    ///
    /// # Examples
    ///
    /// ```
    /// use aprender::text::tokenize::WordTokenizer;
    ///
    /// let tokenizer = WordTokenizer::new();
    /// ```
    pub fn new() -> Self {
        Self
    }

    /// Check if a character should be treated as a separator.
    ///
    /// Separators include most punctuation except apostrophes (for contractions).
    fn is_separator(c: char) -> bool {
        c.is_ascii_punctuation() && c != '\''
    }
}

impl Tokenizer for WordTokenizer {
    fn tokenize(&self, text: &str) -> Result<Vec<String>, AprenderError> {
        let mut tokens = Vec::new();
        let mut current = String::new();

        for ch in text.chars() {
            if ch.is_whitespace() {
                // End current token on whitespace
                if !current.is_empty() {
                    tokens.push(current.clone());
                    current.clear();
                }
            } else if Self::is_separator(ch) {
                // Push current word, then push punctuation as separate token
                if !current.is_empty() {
                    tokens.push(current.clone());
                    current.clear();
                }
                tokens.push(ch.to_string());
            } else {
                // Accumulate alphanumeric and apostrophes
                current.push(ch);
            }
        }

        // Don't forget the last token
        if !current.is_empty() {
            tokens.push(current);
        }

        Ok(tokens)
    }
}

/// Character tokenizer that splits text into individual characters.
///
/// This tokenizer is useful for character-level NLP models.
/// It preserves all characters including whitespace and punctuation.
///
/// # Examples
///
/// ```
/// use aprender::text::{Tokenizer, tokenize::CharTokenizer};
///
/// let tokenizer = CharTokenizer::new();
///
/// let tokens = tokenizer.tokenize("Hi!").expect("tokenize should succeed");
/// assert_eq!(tokens, vec!["H", "i", "!"]);
/// ```
#[derive(Debug, Clone, Default)]
pub struct CharTokenizer;

impl CharTokenizer {
    /// Create a new character tokenizer.
    ///
    /// # Examples
    ///
    /// ```
    /// use aprender::text::tokenize::CharTokenizer;
    ///
    /// let tokenizer = CharTokenizer::new();
    /// ```
    pub fn new() -> Self {
        Self
    }
}

impl Tokenizer for CharTokenizer {
    fn tokenize(&self, text: &str) -> Result<Vec<String>, AprenderError> {
        let tokens: Vec<String> = text.chars().map(|c| c.to_string()).collect();
        Ok(tokens)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ========== WhitespaceTokenizer Tests ==========

    #[test]
    fn test_whitespace_tokenizer_basic() {
        let tokenizer = WhitespaceTokenizer::new();

        let tokens = tokenizer
            .tokenize("Hello world")
            .expect("tokenize should succeed");
        assert_eq!(tokens, vec!["Hello", "world"]);
    }

    #[test]
    fn test_whitespace_tokenizer_preserves_punctuation() {
        let tokenizer = WhitespaceTokenizer::new();

        let tokens = tokenizer
            .tokenize("Hello, world!")
            .expect("tokenize should succeed");
        assert_eq!(tokens, vec!["Hello,", "world!"]);
    }

    #[test]
    fn test_whitespace_tokenizer_multiple_spaces() {
        let tokenizer = WhitespaceTokenizer::new();

        let tokens = tokenizer
            .tokenize("foo   bar")
            .expect("tokenize should succeed");
        assert_eq!(tokens, vec!["foo", "bar"]);
    }

    #[test]
    fn test_whitespace_tokenizer_newlines_tabs() {
        let tokenizer = WhitespaceTokenizer::new();

        let tokens = tokenizer
            .tokenize("line1\nline2\ttab")
            .expect("tokenize should succeed");
        assert_eq!(tokens, vec!["line1", "line2", "tab"]);
    }

    #[test]
    fn test_whitespace_tokenizer_empty_string() {
        let tokenizer = WhitespaceTokenizer::new();

        let tokens = tokenizer.tokenize("").expect("tokenize should succeed");
        assert_eq!(tokens, Vec::<String>::new());
    }

    #[test]
    fn test_whitespace_tokenizer_only_whitespace() {
        let tokenizer = WhitespaceTokenizer::new();

        let tokens = tokenizer
            .tokenize("   \n\t  ")
            .expect("tokenize should succeed");
        assert_eq!(tokens, Vec::<String>::new());
    }

    #[test]
    fn test_whitespace_tokenizer_unicode() {
        let tokenizer = WhitespaceTokenizer::new();

        let tokens = tokenizer
            .tokenize("Hello мир 世界")
            .expect("tokenize should succeed");
        assert_eq!(tokens, vec!["Hello", "мир", "世界"]);
    }

    #[test]
    fn test_whitespace_tokenizer_leading_trailing_whitespace() {
        let tokenizer = WhitespaceTokenizer::new();

        let tokens = tokenizer
            .tokenize("  Hello world  ")
            .expect("tokenize should succeed");
        assert_eq!(tokens, vec!["Hello", "world"]);
    }

    // ========== WordTokenizer Tests ==========

    #[test]
    fn test_word_tokenizer_basic() {
        let tokenizer = WordTokenizer::new();

        let tokens = tokenizer
            .tokenize("Hello world")
            .expect("tokenize should succeed");
        assert_eq!(tokens, vec!["Hello", "world"]);
    }

    #[test]
    fn test_word_tokenizer_separates_punctuation() {
        let tokenizer = WordTokenizer::new();

        let tokens = tokenizer
            .tokenize("Hello, world!")
            .expect("tokenize should succeed");
        assert_eq!(tokens, vec!["Hello", ",", "world", "!"]);
    }

    #[test]
    fn test_word_tokenizer_preserves_contractions() {
        let tokenizer = WordTokenizer::new();

        let tokens = tokenizer
            .tokenize("I don't know.")
            .expect("tokenize should succeed");
        assert_eq!(tokens, vec!["I", "don't", "know", "."]);
    }

    #[test]
    fn test_word_tokenizer_multiple_punctuation() {
        let tokenizer = WordTokenizer::new();

        let tokens = tokenizer
            .tokenize("Wait... what?!")
            .expect("tokenize should succeed");
        assert_eq!(tokens, vec!["Wait", ".", ".", ".", "what", "?", "!"]);
    }

    #[test]
    fn test_word_tokenizer_empty_string() {
        let tokenizer = WordTokenizer::new();

        let tokens = tokenizer.tokenize("").expect("tokenize should succeed");
        assert_eq!(tokens, Vec::<String>::new());
    }

    #[test]
    fn test_word_tokenizer_only_punctuation() {
        let tokenizer = WordTokenizer::new();

        let tokens = tokenizer.tokenize(".,!?").expect("tokenize should succeed");
        assert_eq!(tokens, vec![".", ",", "!", "?"]);
    }

    #[test]
    fn test_word_tokenizer_numbers() {
        let tokenizer = WordTokenizer::new();

        let tokens = tokenizer
            .tokenize("I have 3 apples.")
            .expect("tokenize should succeed");
        assert_eq!(tokens, vec!["I", "have", "3", "apples", "."]);
    }

    #[test]
    fn test_word_tokenizer_hyphenated() {
        let tokenizer = WordTokenizer::new();

        let tokens = tokenizer
            .tokenize("state-of-the-art AI")
            .expect("tokenize should succeed");
        assert_eq!(
            tokens,
            vec!["state", "-", "of", "-", "the", "-", "art", "AI"]
        );
    }

    // ========== CharTokenizer Tests ==========

    #[test]
    fn test_char_tokenizer_basic() {
        let tokenizer = CharTokenizer::new();

        let tokens = tokenizer.tokenize("Hi").expect("tokenize should succeed");
        assert_eq!(tokens, vec!["H", "i"]);
    }

    #[test]
    fn test_char_tokenizer_with_punctuation() {
        let tokenizer = CharTokenizer::new();

        let tokens = tokenizer.tokenize("Hi!").expect("tokenize should succeed");
        assert_eq!(tokens, vec!["H", "i", "!"]);
    }

    #[test]
    fn test_char_tokenizer_with_spaces() {
        let tokenizer = CharTokenizer::new();

        let tokens = tokenizer.tokenize("a b").expect("tokenize should succeed");
        assert_eq!(tokens, vec!["a", " ", "b"]);
    }

    #[test]
    fn test_char_tokenizer_empty_string() {
        let tokenizer = CharTokenizer::new();

        let tokens = tokenizer.tokenize("").expect("tokenize should succeed");
        assert_eq!(tokens, Vec::<String>::new());
    }

    #[test]
    fn test_char_tokenizer_unicode() {
        let tokenizer = CharTokenizer::new();

        let tokens = tokenizer.tokenize("世界").expect("tokenize should succeed");
        assert_eq!(tokens, vec!["世", "界"]);
    }

    #[test]
    fn test_char_tokenizer_emoji() {
        let tokenizer = CharTokenizer::new();

        let tokens = tokenizer.tokenize("Hi👋").expect("tokenize should succeed");
        assert_eq!(tokens, vec!["H", "i", "👋"]);
    }

    // ========== Default Trait Tests ==========

    #[test]
    fn test_whitespace_tokenizer_default() {
        let tokenizer = WhitespaceTokenizer;
        let tokens = tokenizer.tokenize("test").expect("tokenize should succeed");
        assert_eq!(tokens, vec!["test"]);
    }

    #[test]
    fn test_word_tokenizer_default() {
        let tokenizer = WordTokenizer;
        let tokens = tokenizer.tokenize("test").expect("tokenize should succeed");
        assert_eq!(tokens, vec!["test"]);
    }

    #[test]
    fn test_char_tokenizer_default() {
        let tokenizer = CharTokenizer;
        let tokens = tokenizer.tokenize("ab").expect("tokenize should succeed");
        assert_eq!(tokens, vec!["a", "b"]);
    }
}
