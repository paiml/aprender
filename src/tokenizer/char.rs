//! Character-level tokenizer implementation.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use super::config::TokenizerConfig;
use super::error::{Result, TokenizerError};
use super::traits::{TokenId, Tokenizer};

/// Character-level tokenizer (simple baseline)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CharTokenizer {
    config: TokenizerConfig,
    vocab: HashMap<char, TokenId>,
    id_to_char: HashMap<TokenId, char>,
    trained: bool,
}

impl CharTokenizer {
    /// Create a new character tokenizer
    pub fn new(config: TokenizerConfig) -> Self {
        Self { config, vocab: HashMap::new(), id_to_char: HashMap::new(), trained: false }
    }
}

impl Tokenizer for CharTokenizer {
    fn train(&mut self, corpus: &[&str]) -> Result<()> {
        let mut id: TokenId = 0;

        // Count character frequencies
        let mut char_counts: HashMap<char, usize> = HashMap::new();
        for text in corpus {
            let processed =
                if self.config.lowercase { text.to_lowercase() } else { text.to_string() };
            for c in processed.chars() {
                *char_counts.entry(c).or_insert(0) += 1;
            }
        }

        // Sort by frequency and take top vocab_size
        let mut chars: Vec<_> = char_counts.into_iter().collect();
        chars.sort_by(|a, b| b.1.cmp(&a.1));

        for (c, count) in chars.into_iter().take(self.config.vocab_size) {
            if count >= self.config.min_frequency {
                self.vocab.insert(c, id);
                self.id_to_char.insert(id, c);
                id += 1;
            }
        }

        self.trained = true;
        Ok(())
    }

    fn encode(&self, text: &str) -> Result<Vec<TokenId>> {
        if !self.trained {
            return Err(TokenizerError::NotTrained);
        }

        let processed = if self.config.lowercase { text.to_lowercase() } else { text.to_string() };

        let mut ids = Vec::new();
        for c in processed.chars() {
            if let Some(&id) = self.vocab.get(&c) {
                ids.push(id);
            }
            // Unknown characters are skipped
        }

        Ok(ids)
    }

    fn decode(&self, ids: &[TokenId]) -> Result<String> {
        if !self.trained {
            return Err(TokenizerError::NotTrained);
        }

        let mut result = String::new();
        for &id in ids {
            if let Some(&c) = self.id_to_char.get(&id) {
                result.push(c);
            }
        }

        Ok(result)
    }

    fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    fn is_trained(&self) -> bool {
        self.trained
    }

    fn id_to_token(&self, _id: TokenId) -> Option<&str> {
        // Characters are not stored as strings
        None
    }

    fn token_to_id(&self, token: &str) -> Option<TokenId> {
        if token.len() == 1 {
            self.vocab
                .get(&token.chars().next().expect("single-char token must have a char"))
                .copied()
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_char_new() {
        let config = TokenizerConfig::char();
        let tokenizer = CharTokenizer::new(config);
        assert!(!tokenizer.is_trained());
    }

    #[test]
    fn test_char_train() {
        let config = TokenizerConfig::char().with_min_frequency(1);
        let mut tokenizer = CharTokenizer::new(config);

        let corpus = vec!["hello", "world"];
        tokenizer.train(&corpus).expect("operation should succeed");

        assert!(tokenizer.is_trained());
        // h, e, l, o, w, r, d = 7 unique chars
        assert_eq!(tokenizer.vocab_size(), 7);
    }

    #[test]
    fn test_char_encode_decode() {
        let config = TokenizerConfig::char().with_min_frequency(1);
        let mut tokenizer = CharTokenizer::new(config);

        let corpus = vec!["hello"];
        tokenizer.train(&corpus).expect("operation should succeed");

        let text = "hello";
        let encoded = tokenizer.encode(text).expect("encoding should succeed");
        let decoded = tokenizer.decode(&encoded).expect("encoding should succeed");

        assert_eq!(decoded, text);
    }

    #[test]
    fn test_char_unknown_chars() {
        let config = TokenizerConfig::char().with_min_frequency(1);
        let mut tokenizer = CharTokenizer::new(config);

        let corpus = vec!["abc"];
        tokenizer.train(&corpus).expect("operation should succeed");

        // 'x' is not in vocabulary, should be skipped
        let encoded = tokenizer.encode("axbc").expect("encoding should succeed");
        let decoded = tokenizer.decode(&encoded).expect("encoding should succeed");

        assert_eq!(decoded, "abc");
    }

    #[test]
    fn test_char_lowercase() {
        let config = TokenizerConfig::char().with_min_frequency(1).with_lowercase(true);
        let mut tokenizer = CharTokenizer::new(config);

        let corpus = vec!["Hello"];
        tokenizer.train(&corpus).expect("operation should succeed");

        let encoded = tokenizer.encode("HELLO").expect("encoding should succeed");
        let decoded = tokenizer.decode(&encoded).expect("encoding should succeed");

        assert_eq!(decoded, "hello");
    }
}

#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(50))]

        #[test]
        fn prop_char_roundtrip(text in "[a-z]{1,20}") {
            let config = TokenizerConfig::char().with_min_frequency(1);
            let mut tokenizer = CharTokenizer::new(config);
            tokenizer.train(&[&text]).expect("operation should succeed");

            let encoded = tokenizer.encode(&text).expect("encoding should succeed");
            let decoded = tokenizer.decode(&encoded).expect("encoding should succeed");

            prop_assert_eq!(decoded, text);
        }

        #[test]
        fn prop_char_vocab_size_matches_unique_chars(text in "[a-z]{5,30}") {
            let config = TokenizerConfig::char()
                .with_min_frequency(1)
                .with_vocab_size(256);
            let mut tokenizer = CharTokenizer::new(config);
            tokenizer.train(&[&text]).expect("operation should succeed");

            let unique_chars: std::collections::HashSet<char> = text.chars().collect();
            prop_assert_eq!(tokenizer.vocab_size(), unique_chars.len());
        }
    }
}
