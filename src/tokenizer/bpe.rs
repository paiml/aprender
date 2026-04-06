//! BPE (Byte Pair Encoding) tokenizer implementation.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use super::config::TokenizerConfig;
use super::error::{Result, TokenizerError};
use super::traits::{TokenId, Tokenizer};

/// BPE (Byte Pair Encoding) tokenizer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BPETokenizer {
    config: TokenizerConfig,
    /// Token to ID mapping
    vocab: HashMap<String, TokenId>,
    /// ID to token mapping
    id_to_token_map: HashMap<TokenId, String>,
    /// Merge rules (pair -> merged token)
    merges: Vec<(String, String)>,
    /// Whether the tokenizer is trained
    trained: bool,
}

impl BPETokenizer {
    /// Create a new BPE tokenizer
    pub fn new(config: TokenizerConfig) -> Self {
        Self {
            config,
            vocab: HashMap::new(),
            id_to_token_map: HashMap::new(),
            merges: Vec::new(),
            trained: false,
        }
    }

    /// Initialize vocabulary with special tokens and bytes
    fn init_vocab(&mut self) {
        let mut id: TokenId = 0;

        // Add special tokens
        let special = [
            &self.config.special_tokens.unk,
            &self.config.special_tokens.bos,
            &self.config.special_tokens.eos,
            &self.config.special_tokens.pad,
            &self.config.special_tokens.mask,
        ];

        for token in special {
            self.vocab.insert(token.clone(), id);
            self.id_to_token_map.insert(id, token.clone());
            id += 1;
        }

        // Add all single bytes as base vocabulary
        for byte in 0..=255u8 {
            let token = format!("{byte:02x}");
            if !self.vocab.contains_key(&token) {
                self.vocab.insert(token.clone(), id);
                self.id_to_token_map.insert(id, token);
                id += 1;
            }
        }
    }

    /// Get pair frequencies from tokenized corpus
    fn get_pair_freqs(&self, tokenized: &[Vec<String>]) -> HashMap<(String, String), usize> {
        let mut freqs = HashMap::new();

        for tokens in tokenized {
            for pair in tokens.windows(2) {
                let key = (pair[0].clone(), pair[1].clone());
                *freqs.entry(key).or_insert(0) += 1;
            }
        }

        freqs
    }

    /// Merge the most frequent pair
    fn merge_pair(&self, tokenized: &mut [Vec<String>], pair: &(String, String), merged: &str) {
        for tokens in tokenized.iter_mut() {
            let mut i = 0;
            while i < tokens.len().saturating_sub(1) {
                if tokens[i] == pair.0 && tokens[i + 1] == pair.1 {
                    tokens[i] = merged.to_string();
                    tokens.remove(i + 1);
                }
                i += 1;
            }
        }
    }

    /// Tokenize text to bytes (initial tokenization)
    fn to_bytes(&self, text: &str) -> Vec<String> {
        text.as_bytes().iter().map(|b| format!("{b:02x}")).collect()
    }

    /// Apply all learned merges
    fn apply_merges(&self, mut tokens: Vec<String>) -> Vec<String> {
        for (a, b) in &self.merges {
            let merged = format!("{a}{b}");
            let mut i = 0;
            while i < tokens.len().saturating_sub(1) {
                if &tokens[i] == a && &tokens[i + 1] == b {
                    tokens[i] = merged.clone();
                    tokens.remove(i + 1);
                } else {
                    i += 1;
                }
            }
        }
        tokens
    }

    /// Save tokenizer to file
    pub fn save(&self, path: &str) -> Result<()> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| TokenizerError::Serialization(e.to_string()))?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Load tokenizer from file
    pub fn load(path: &str) -> Result<Self> {
        let json = std::fs::read_to_string(path)?;
        serde_json::from_str(&json).map_err(|e| TokenizerError::Serialization(e.to_string()))
    }
}

impl Tokenizer for BPETokenizer {
    fn train(&mut self, corpus: &[&str]) -> Result<()> {
        self.init_vocab();

        // Tokenize corpus to bytes
        let mut tokenized: Vec<Vec<String>> = corpus
            .iter()
            .map(|text| {
                let t = if self.config.lowercase { text.to_lowercase() } else { text.to_string() };
                self.to_bytes(&t)
            })
            .collect();

        // Learn merges until we reach target vocab size
        let target = self.config.vocab_size;
        while self.vocab.len() < target {
            let freqs = self.get_pair_freqs(&tokenized);

            // Find most frequent pair
            let best = freqs
                .iter()
                .filter(|(_, &count)| count >= self.config.min_frequency)
                .max_by_key(|(_, count)| *count);

            match best {
                Some((pair, _)) => {
                    let merged = format!("{}{}", pair.0, pair.1);

                    // Add to vocabulary
                    let id = self.vocab.len() as TokenId;
                    self.vocab.insert(merged.clone(), id);
                    self.id_to_token_map.insert(id, merged.clone());

                    // Record merge
                    self.merges.push(pair.clone());

                    // Apply merge
                    self.merge_pair(&mut tokenized, pair, &merged);
                }
                None => break, // No more pairs meet frequency threshold
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

        let tokens = self.to_bytes(&processed);
        let tokens = self.apply_merges(tokens);

        let unk_id = *self
            .vocab
            .get(&self.config.special_tokens.unk)
            .expect("UNK token must exist in trained vocabulary");

        let ids: Vec<TokenId> =
            tokens.iter().map(|t| *self.vocab.get(t).unwrap_or(&unk_id)).collect();

        Ok(ids)
    }

    fn decode(&self, ids: &[TokenId]) -> Result<String> {
        if !self.trained {
            return Err(TokenizerError::NotTrained);
        }

        let mut hex_string = String::new();

        for &id in ids {
            if let Some(token) = self.id_to_token_map.get(&id) {
                // Skip special tokens
                if token.starts_with('<') && token.ends_with('>') {
                    continue;
                }
                hex_string.push_str(token);
            }
        }

        // Convert hex string back to bytes
        let bytes: Vec<u8> = (0..hex_string.len())
            .step_by(2)
            .filter_map(|i| {
                if i + 2 <= hex_string.len() {
                    u8::from_str_radix(&hex_string[i..i + 2], 16).ok()
                } else {
                    None
                }
            })
            .collect();

        String::from_utf8(bytes).map_err(|e| TokenizerError::Training(e.to_string()))
    }

    fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    fn is_trained(&self) -> bool {
        self.trained
    }

    fn id_to_token(&self, id: TokenId) -> Option<&str> {
        self.id_to_token_map.get(&id).map(String::as_str)
    }

    fn token_to_id(&self, token: &str) -> Option<TokenId> {
        self.vocab.get(token).copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bpe_new() {
        let config = TokenizerConfig::bpe();
        let tokenizer = BPETokenizer::new(config);
        assert!(!tokenizer.is_trained());
    }

    #[test]
    fn test_bpe_train() {
        let config = TokenizerConfig::bpe().with_vocab_size(300).with_min_frequency(1);
        let mut tokenizer = BPETokenizer::new(config);

        let corpus = vec!["hello hello", "hello world", "world hello"];
        tokenizer.train(&corpus).expect("operation should succeed");

        assert!(tokenizer.is_trained());
        assert!(tokenizer.vocab_size() > 256); // Base bytes + some merges
    }

    #[test]
    fn test_bpe_encode_not_trained() {
        let config = TokenizerConfig::bpe();
        let tokenizer = BPETokenizer::new(config);

        let result = tokenizer.encode("hello");
        assert!(result.is_err());
    }

    #[test]
    fn test_bpe_encode_decode() {
        let config = TokenizerConfig::bpe().with_vocab_size(300).with_min_frequency(1);
        let mut tokenizer = BPETokenizer::new(config);

        let corpus = vec!["hello world", "hello there"];
        tokenizer.train(&corpus).expect("operation should succeed");

        let text = "hello";
        let encoded = tokenizer.encode(text).expect("encoding should succeed");
        let decoded = tokenizer.decode(&encoded).expect("encoding should succeed");

        assert_eq!(decoded, text);
    }

    #[test]
    fn test_bpe_lowercase() {
        let config =
            TokenizerConfig::bpe().with_vocab_size(300).with_min_frequency(1).with_lowercase(true);
        let mut tokenizer = BPETokenizer::new(config);

        let corpus = vec!["Hello World"];
        tokenizer.train(&corpus).expect("operation should succeed");

        let encoded = tokenizer.encode("HELLO").expect("encoding should succeed");
        let decoded = tokenizer.decode(&encoded).expect("encoding should succeed");

        assert_eq!(decoded, "hello");
    }

    #[test]
    fn test_bpe_id_to_token() {
        let config = TokenizerConfig::bpe().with_vocab_size(300).with_min_frequency(1);
        let mut tokenizer = BPETokenizer::new(config);

        let corpus = vec!["test"];
        tokenizer.train(&corpus).expect("operation should succeed");

        // ID 0 should be <unk>
        assert_eq!(tokenizer.id_to_token(0), Some("<unk>"));
    }

    #[test]
    fn test_bpe_token_to_id() {
        let config = TokenizerConfig::bpe().with_vocab_size(300).with_min_frequency(1);
        let mut tokenizer = BPETokenizer::new(config);

        let corpus = vec!["test"];
        tokenizer.train(&corpus).expect("operation should succeed");

        assert_eq!(tokenizer.token_to_id("<unk>"), Some(0));
    }
}

#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(50))]

        #[test]
        fn prop_bpe_encode_produces_valid_ids(text in "[a-zA-Z ]{1,20}") {
            let config = TokenizerConfig::bpe()
                .with_vocab_size(300)
                .with_min_frequency(1);
            let mut tokenizer = BPETokenizer::new(config);
            tokenizer.train(&[&text]).expect("operation should succeed");

            let encoded = tokenizer.encode(&text).expect("encoding should succeed");

            for id in encoded {
                prop_assert!(tokenizer.id_to_token(id).is_some());
            }
        }

        #[test]
        fn prop_vocab_size_bounded(target_size in 261usize..500) {
            let config = TokenizerConfig::bpe()
                .with_vocab_size(target_size)
                .with_min_frequency(1);
            let mut tokenizer = BPETokenizer::new(config);

            let corpus = vec!["hello world hello world test test"];
            tokenizer.train(&corpus).expect("operation should succeed");

            prop_assert!(tokenizer.vocab_size() <= target_size);
        }
    }
}
