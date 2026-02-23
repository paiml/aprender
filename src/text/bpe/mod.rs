//! Byte Pair Encoding (BPE) tokenizer (GH-128).
//!
//! Provides BPE tokenization for LLMs and speech models:
//! - `HuggingFace` tokenizer loading (tokenizer.json)
//! - Encode text to token IDs
//! - Decode token IDs to text
//! - Special token handling
//!
//! # Architecture
//!
//! ```text
//! Text → Pre-tokenize → BPE Merge → Token IDs
//!                         ↓
//!              vocab.json + merges.txt
//! ```
//!
//! # Example
//!
//! ```rust
//! use aprender::text::bpe::{BpeTokenizer, BpeConfig};
//!
//! // Create tokenizer with empty vocabulary (default)
//! let config = BpeConfig::default();
//! let tokenizer = BpeTokenizer::new(config);
//!
//! // Empty vocab returns empty tokens - real usage requires loading vocab
//! let tokens = tokenizer.encode("Hello");
//! assert!(tokens.is_empty()); // No vocab loaded yet
//!
//! // For real tokenization, load from HuggingFace:
//! // let tokenizer = BpeTokenizer::from_huggingface("path/to/tokenizer.json")?;
//! // let tokens = tokenizer.encode("Hello world");
//! // assert!(!tokens.is_empty());
//! ```
//!
//! # References
//!
//! - Sennrich, R., et al. (2016). Neural Machine Translation of Rare Words with Subword Units.
//! - Radford, A., et al. (2019). Language Models are Unsupervised Multitask Learners (GPT-2).
//!
//! # PMAT Compliance
//!
//! - Zero `unwrap()` calls
//! - All public APIs return `Result<T, E>` where fallible

use std::collections::HashMap;

// ============================================================================
// Configuration
// ============================================================================

/// BPE tokenizer configuration.
#[derive(Debug, Clone)]
pub struct BpeConfig {
    /// Unknown token
    pub unk_token: String,
    /// Beginning of sentence token
    pub bos_token: Option<String>,
    /// End of sentence token
    pub eos_token: Option<String>,
    /// Padding token
    pub pad_token: Option<String>,
    /// Whether to add prefix space
    pub add_prefix_space: bool,
    /// Maximum vocabulary size
    pub vocab_size: usize,
}

impl Default for BpeConfig {
    fn default() -> Self {
        Self {
            unk_token: "<unk>".to_string(),
            bos_token: Some("<|endoftext|>".to_string()),
            eos_token: Some("<|endoftext|>".to_string()),
            pad_token: None,
            add_prefix_space: true,
            vocab_size: 50257, // GPT-2 default
        }
    }
}

impl BpeConfig {
    /// Create config for Whisper tokenizer
    #[must_use]
    pub fn whisper() -> Self {
        Self {
            unk_token: "<|endoftext|>".to_string(),
            bos_token: Some("<|startoftranscript|>".to_string()),
            eos_token: Some("<|endoftext|>".to_string()),
            pad_token: None,
            add_prefix_space: false,
            vocab_size: 51865, // Whisper vocab size
        }
    }

    /// Create config for GPT-2 tokenizer
    #[must_use]
    pub fn gpt2() -> Self {
        Self::default()
    }

    /// Create config for Llama tokenizer
    #[must_use]
    pub fn llama() -> Self {
        Self {
            unk_token: "<unk>".to_string(),
            bos_token: Some("<s>".to_string()),
            eos_token: Some("</s>".to_string()),
            pad_token: None,
            add_prefix_space: false,
            vocab_size: 32000,
        }
    }

    /// Create config for Qwen2 tokenizer (GH-128)
    ///
    /// Qwen2 uses a 151936 token vocabulary with special chat tokens.
    #[must_use]
    pub fn qwen2() -> Self {
        Self {
            unk_token: "<|endoftext|>".to_string(),
            bos_token: Some("<|im_start|>".to_string()),
            eos_token: Some("<|im_end|>".to_string()),
            pad_token: Some("<|endoftext|>".to_string()),
            add_prefix_space: false,
            vocab_size: crate::demo::Qwen2Config::VOCAB_SIZE,
        }
    }
}

// ============================================================================
// BPE Merge Rule
// ============================================================================

/// A BPE merge rule (pair → merged token).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MergeRule {
    /// First token in pair
    pub first: String,
    /// Second token in pair
    pub second: String,
}

impl MergeRule {
    /// Create a new merge rule
    #[must_use]
    pub fn new(first: impl Into<String>, second: impl Into<String>) -> Self {
        Self {
            first: first.into(),
            second: second.into(),
        }
    }

    /// Get merged token
    #[must_use]
    pub fn merged(&self) -> String {
        format!("{}{}", self.first, self.second)
    }
}

// ============================================================================
// BPE Tokenizer
// ============================================================================

/// Byte Pair Encoding tokenizer.
///
/// Implements subword tokenization using BPE algorithm.
#[derive(Debug, Clone)]
pub struct BpeTokenizer {
    /// Configuration
    config: BpeConfig,
    /// Token to ID mapping
    vocab: HashMap<String, u32>,
    /// ID to token mapping
    id_to_token: HashMap<u32, String>,
    /// BPE merge rules (in order of priority)
    merges: Vec<MergeRule>,
    /// Merge rule to rank (lower = higher priority)
    merge_ranks: HashMap<(String, String), usize>,
    /// Special tokens
    special_tokens: HashMap<String, u32>,
    /// Byte encoder for UTF-8 handling
    byte_encoder: HashMap<u8, char>,
    /// Byte decoder
    byte_decoder: HashMap<char, u8>,
}

mod qwen2bpe_tokenizer;
pub use qwen2bpe_tokenizer::*;
mod qwen2;
pub use qwen2::*;
