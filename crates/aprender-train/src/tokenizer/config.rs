//! Tokenizer configuration types.

use serde::{Deserialize, Serialize};

/// Default vocabulary size (LLaMA/Mistral family)
const DEFAULT_VOCAB_SIZE: usize = 32000;

/// Special tokens
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpecialTokens {
    /// Unknown token
    pub unk: String,
    /// Beginning of sequence
    pub bos: String,
    /// End of sequence
    pub eos: String,
    /// Padding token
    pub pad: String,
    /// Mask token (for MLM)
    pub mask: String,
}

impl Default for SpecialTokens {
    fn default() -> Self {
        Self {
            unk: "<unk>".to_string(),
            bos: "<s>".to_string(),
            eos: "</s>".to_string(),
            pad: "<pad>".to_string(),
            mask: "<mask>".to_string(),
        }
    }
}

/// Tokenizer type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TokenizerType {
    /// Byte Pair Encoding
    BPE,
    /// WordPiece (BERT-style)
    WordPiece,
    /// Character-level
    Char,
}

/// Tokenizer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenizerConfig {
    /// Target vocabulary size
    pub vocab_size: usize,
    /// Minimum token frequency for training
    pub min_frequency: usize,
    /// Special tokens
    pub special_tokens: SpecialTokens,
    /// Whether to lowercase input
    pub lowercase: bool,
    /// Tokenizer type
    pub tokenizer_type: TokenizerType,
}

impl Default for TokenizerConfig {
    fn default() -> Self {
        Self {
            vocab_size: DEFAULT_VOCAB_SIZE,
            min_frequency: 2,
            special_tokens: SpecialTokens::default(),
            lowercase: false,
            tokenizer_type: TokenizerType::BPE,
        }
    }
}

impl TokenizerConfig {
    /// Create a BPE tokenizer config
    pub fn bpe() -> Self {
        Self { tokenizer_type: TokenizerType::BPE, ..Default::default() }
    }

    /// Create a WordPiece tokenizer config
    pub fn wordpiece() -> Self {
        Self { tokenizer_type: TokenizerType::WordPiece, ..Default::default() }
    }

    /// Create a character-level tokenizer config
    pub fn char() -> Self {
        Self { tokenizer_type: TokenizerType::Char, vocab_size: 256, ..Default::default() }
    }

    /// Set vocabulary size
    pub fn with_vocab_size(mut self, size: usize) -> Self {
        self.vocab_size = size;
        self
    }

    /// Set minimum frequency
    pub fn with_min_frequency(mut self, freq: usize) -> Self {
        self.min_frequency = freq;
        self
    }

    /// Enable lowercase preprocessing
    pub fn with_lowercase(mut self, lowercase: bool) -> Self {
        self.lowercase = lowercase;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenizer_config_default() {
        let config = TokenizerConfig::default();
        assert_eq!(config.vocab_size, 32000);
        assert_eq!(config.tokenizer_type, TokenizerType::BPE);
    }

    #[test]
    fn test_tokenizer_config_bpe() {
        let config = TokenizerConfig::bpe().with_vocab_size(1000);
        assert_eq!(config.vocab_size, 1000);
        assert_eq!(config.tokenizer_type, TokenizerType::BPE);
    }

    #[test]
    fn test_tokenizer_config_wordpiece() {
        let config = TokenizerConfig::wordpiece();
        assert_eq!(config.tokenizer_type, TokenizerType::WordPiece);
    }

    #[test]
    fn test_tokenizer_config_char() {
        let config = TokenizerConfig::char();
        assert_eq!(config.tokenizer_type, TokenizerType::Char);
        assert_eq!(config.vocab_size, 256);
    }

    #[test]
    fn test_special_tokens_default() {
        let special = SpecialTokens::default();
        assert_eq!(special.unk, "<unk>");
        assert_eq!(special.bos, "<s>");
        assert_eq!(special.eos, "</s>");
    }
}
