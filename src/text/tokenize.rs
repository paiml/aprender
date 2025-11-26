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

/// Sentence tokenizer that splits text into sentences.
///
/// Uses punctuation-based rules to detect sentence boundaries,
/// handling common abbreviations and edge cases.
///
/// # Examples
///
/// ```
/// use aprender::text::tokenize::SentenceTokenizer;
///
/// let tokenizer = SentenceTokenizer::new();
///
/// let sentences = tokenizer.split("Hello world. How are you? I'm fine!");
/// assert_eq!(sentences, vec!["Hello world.", "How are you?", "I'm fine!"]);
/// ```
#[derive(Debug, Clone, Default)]
pub struct SentenceTokenizer {
    /// Common abbreviations that don't end sentences
    abbreviations: Vec<&'static str>,
}

impl SentenceTokenizer {
    /// Create a new sentence tokenizer with default abbreviations.
    pub fn new() -> Self {
        Self {
            abbreviations: vec![
                "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "vs", "etc", "inc", "ltd", "corp",
                "st", "ave", "blvd", "rd", "dept", "gov", "gen", "col", "lt", "sgt", "rev", "hon",
                "pres", "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "oct", "nov",
                "dec", "i.e", "e.g", "cf", "al", "vol", "no", "fig", "pp", "ph.d", "m.d", "b.a",
                "m.a", "d.d.s",
            ],
        }
    }

    /// Split text into sentences.
    pub fn split(&self, text: &str) -> Vec<String> {
        if text.is_empty() {
            return Vec::new();
        }

        let mut sentences = Vec::new();
        let mut current = String::new();
        let chars: Vec<char> = text.chars().collect();
        let len = chars.len();

        let mut i = 0;
        while i < len {
            let c = chars[i];
            current.push(c);

            // Check for sentence-ending punctuation
            if c == '.' || c == '?' || c == '!' {
                // Look ahead to see if this is really a sentence end
                let is_end = if i + 1 < len {
                    let next = chars[i + 1];
                    // End if followed by space + uppercase, or end of text
                    if next.is_whitespace() {
                        // Check if followed by uppercase
                        let mut j = i + 2;
                        while j < len && chars[j].is_whitespace() {
                            j += 1;
                        }
                        j >= len || chars[j].is_uppercase()
                    } else {
                        false
                    }
                } else {
                    true // End of text
                };

                // Check for abbreviation (for periods only)
                let is_abbrev = if c == '.' {
                    self.is_abbreviation(&current)
                } else {
                    false
                };

                if is_end && !is_abbrev {
                    let trimmed = current.trim().to_string();
                    if !trimmed.is_empty() {
                        sentences.push(trimmed);
                    }
                    current.clear();
                }
            }
            i += 1;
        }

        // Add remaining text
        let trimmed = current.trim().to_string();
        if !trimmed.is_empty() {
            sentences.push(trimmed);
        }

        sentences
    }

    fn is_abbreviation(&self, text: &str) -> bool {
        // Extract the last word before the period
        let text = text.trim_end_matches('.');
        let last_word = text.split_whitespace().last().unwrap_or("");
        let lower = last_word.to_lowercase();
        self.abbreviations.contains(&lower.as_str())
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
