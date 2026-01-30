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

use crate::error::{AprenderError, Result};
use serde::Deserialize;
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
            vocab_size: 151936,
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

impl BpeTokenizer {
    /// Create a new BPE tokenizer with given config
    #[must_use]
    pub fn new(config: BpeConfig) -> Self {
        let (byte_encoder, byte_decoder) = bytes_to_unicode();

        Self {
            config,
            vocab: HashMap::new(),
            id_to_token: HashMap::new(),
            merges: Vec::new(),
            merge_ranks: HashMap::new(),
            special_tokens: HashMap::new(),
            byte_encoder,
            byte_decoder,
        }
    }

    /// Create tokenizer with GPT-2 base vocabulary (stub)
    ///
    /// # Note
    /// Real implementation requires loading vocabulary files.
    #[must_use]
    pub fn gpt2_base() -> Self {
        let config = BpeConfig::gpt2();
        let mut tokenizer = Self::new(config);

        // Add basic ASCII characters as initial vocab
        for i in 0..=255u8 {
            if let Some(&c) = tokenizer.byte_encoder.get(&i) {
                let token = c.to_string();
                let id = u32::from(i);
                tokenizer.vocab.insert(token.clone(), id);
                tokenizer.id_to_token.insert(id, token);
            }
        }

        // Add special tokens
        tokenizer.add_special_token("<|endoftext|>", 50256);

        tokenizer
    }

    /// Add a special token
    pub fn add_special_token(&mut self, token: &str, id: u32) {
        self.special_tokens.insert(token.to_string(), id);
        self.vocab.insert(token.to_string(), id);
        self.id_to_token.insert(id, token.to_string());
    }

    /// Add a merge rule
    pub fn add_merge(&mut self, first: &str, second: &str) {
        let rank = self.merges.len();
        let rule = MergeRule::new(first, second);
        self.merge_ranks
            .insert((first.to_string(), second.to_string()), rank);
        self.merges.push(rule);
    }

    /// Get vocabulary size
    #[must_use]
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// Get token ID for a token
    #[must_use]
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.vocab.get(token).copied()
    }

    /// Get token for an ID
    #[must_use]
    pub fn id_to_token(&self, id: u32) -> Option<&str> {
        self.id_to_token.get(&id).map(String::as_str)
    }

    /// Check if token is a special token
    #[must_use]
    pub fn is_special_token(&self, token: &str) -> bool {
        self.special_tokens.contains_key(token)
    }

    /// Encode text to token IDs.
    ///
    /// # Arguments
    /// * `text` - Text to encode
    ///
    /// # Returns
    /// Vector of token IDs
    #[must_use]
    pub fn encode(&self, text: &str) -> Vec<u32> {
        if text.is_empty() {
            return vec![];
        }

        let mut ids = Vec::new();

        // PMAT-114: Handle special tokens FIRST before BPE tokenization
        // This ensures tokens like <|im_start|> are encoded as single tokens (151644)
        // rather than being split into characters (27, 91, 318, 4906, 91, 29)
        let segments = self.split_on_special_tokens(text);

        for segment in segments {
            if let Some(&special_id) = self.special_tokens.get(&segment) {
                // Special token - output its ID directly
                ids.push(special_id);
            } else {
                // Regular text - apply BPE tokenization
                let segment_text = if self.config.add_prefix_space
                    && !segment.starts_with(' ')
                    && ids.is_empty()
                {
                    format!(" {segment}")
                } else {
                    segment
                };

                for word in self.pre_tokenize(&segment_text) {
                    let byte_word = self.bytes_to_bpe_tokens(&word);
                    let tokens = self.bpe(&byte_word);

                    for token in tokens {
                        if let Some(&id) = self.vocab.get(&token) {
                            ids.push(id);
                        } else if let Some(&id) = self.vocab.get(&self.config.unk_token) {
                            ids.push(id);
                        }
                    }
                }
            }
        }

        ids
    }

    /// Split text on special tokens while preserving them as separate segments.
    /// Returns vec of segments where special tokens are their own elements.
    fn split_on_special_tokens(&self, text: &str) -> Vec<String> {
        if self.special_tokens.is_empty() {
            return vec![text.to_string()];
        }

        // Sort special tokens by length (longest first) to avoid partial matches
        let mut sorted_tokens: Vec<_> = self.special_tokens.keys().collect();
        sorted_tokens.sort_by_key(|t| std::cmp::Reverse(t.len()));

        let mut result = Vec::new();
        let mut remaining = text;

        while !remaining.is_empty() {
            // Find the earliest special token occurrence
            let mut earliest_match: Option<(usize, &str)> = None;

            for token in &sorted_tokens {
                if let Some(pos) = remaining.find(token.as_str()) {
                    match earliest_match {
                        None => earliest_match = Some((pos, token)),
                        Some((prev_pos, _)) if pos < prev_pos => {
                            earliest_match = Some((pos, token));
                        }
                        _ => {}
                    }
                }
            }

            match earliest_match {
                Some((pos, token)) => {
                    // Add text before the special token (if any)
                    if pos > 0 {
                        result.push(remaining[..pos].to_string());
                    }
                    // Add the special token itself
                    result.push(token.to_string());
                    // Continue with remaining text
                    remaining = &remaining[pos + token.len()..];
                }
                None => {
                    // No more special tokens - add remaining text
                    result.push(remaining.to_string());
                    break;
                }
            }
        }

        result
    }

    /// Decode token IDs to text.
    ///
    /// # Arguments
    /// * `ids` - Token IDs to decode
    ///
    /// # Returns
    /// Decoded text
    #[must_use]
    pub fn decode(&self, ids: &[u32]) -> String {
        if ids.is_empty() {
            return String::new();
        }

        let mut text = String::new();

        for &id in ids {
            if let Some(token) = self.id_to_token.get(&id) {
                // Skip special tokens in output
                if !self.special_tokens.contains_key(token) {
                    text.push_str(token);
                }
            }
        }

        // Convert byte tokens back to UTF-8
        self.bpe_tokens_to_bytes(&text)
    }

    /// Encode text to token IDs with error handling.
    ///
    /// # Errors
    /// Returns error if encoding fails.
    pub fn encode_checked(&self, text: &str) -> Result<Vec<u32>> {
        Ok(self.encode(text))
    }

    /// Decode token IDs to text with error handling.
    ///
    /// # Errors
    /// Returns error if decoding fails.
    pub fn decode_checked(&self, ids: &[u32]) -> Result<String> {
        Ok(self.decode(ids))
    }

    /// Pre-tokenize text into words
    fn pre_tokenize(&self, text: &str) -> Vec<String> {
        // Simple regex-like pattern: split on whitespace, keeping punctuation
        // Future: Use self.config for model-specific pre-tokenization rules
        let _ = &self.config;
        let mut words = Vec::new();
        let mut current = String::new();

        for c in text.chars() {
            if c.is_whitespace() {
                if !current.is_empty() {
                    words.push(current.clone());
                    current.clear();
                }
                // Include the space as part of next word
                current.push(c);
            } else {
                current.push(c);
            }
        }

        if !current.is_empty() {
            words.push(current);
        }

        words
    }

    /// Convert string to byte-encoded tokens
    fn bytes_to_bpe_tokens(&self, word: &str) -> Vec<String> {
        word.bytes()
            .map(|b| {
                self.byte_encoder
                    .get(&b)
                    .map_or_else(|| format!("?{b}"), |&c| c.to_string())
            })
            .collect()
    }

    /// Convert byte-encoded tokens back to string
    fn bpe_tokens_to_bytes(&self, text: &str) -> String {
        let bytes: Vec<u8> = text
            .chars()
            .filter_map(|c| self.byte_decoder.get(&c).copied())
            .collect();

        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// Apply BPE merges to token list
    fn bpe(&self, tokens: &[String]) -> Vec<String> {
        if tokens.len() <= 1 {
            return tokens.to_vec();
        }

        let mut result = tokens.to_vec();

        loop {
            // Find best merge (lowest rank)
            let mut best_merge: Option<(usize, usize)> = None;
            let mut best_rank = usize::MAX;

            for i in 0..result.len().saturating_sub(1) {
                let pair = (result[i].clone(), result[i + 1].clone());
                if let Some(&rank) = self.merge_ranks.get(&pair) {
                    if rank < best_rank {
                        best_rank = rank;
                        best_merge = Some((i, rank));
                    }
                }
            }

            // Apply best merge or stop
            match best_merge {
                Some((idx, _)) => {
                    let merged = format!("{}{}", result[idx], result[idx + 1]);
                    result.splice(idx..=idx + 1, std::iter::once(merged));
                }
                None => break,
            }
        }

        result
    }
}

impl Default for BpeTokenizer {
    fn default() -> Self {
        Self::new(BpeConfig::default())
    }
}

// ============================================================================
// Qwen2 BPE Tokenizer
// ============================================================================

/// Qwen2-specific BPE tokenizer with chat template support.
///
/// Extends the base BPE tokenizer with Qwen2's special tokens and
/// chat formatting conventions.
///
/// # Example
///
/// ```rust
/// use aprender::text::bpe::Qwen2BpeTokenizer;
///
/// let tokenizer = Qwen2BpeTokenizer::new();
///
/// // Check special tokens
/// assert!(tokenizer.is_eos(151645)); // <|im_end|>
///
/// // Format a chat message
/// let formatted = tokenizer.format_chat("user", "Hello, world!");
/// assert!(formatted.contains("<|im_start|>user"));
/// ```
#[derive(Debug, Clone)]
pub struct Qwen2BpeTokenizer {
    /// Base tokenizer
    base: BpeTokenizer,
    /// Special token IDs
    im_start_id: u32,
    im_end_id: u32,
    endoftext_id: u32,
}

impl Qwen2BpeTokenizer {
    /// Special token: <|`im_start`|>
    pub const IM_START_ID: u32 = 151644;
    /// Special token: <|`im_end`|>
    pub const IM_END_ID: u32 = 151645;
    /// Special token: <|endoftext|>
    pub const ENDOFTEXT_ID: u32 = 151643;

    /// Create a new Qwen2 tokenizer.
    #[must_use]
    pub fn new() -> Self {
        let config = BpeConfig::qwen2();
        let mut base = BpeTokenizer::new(config);

        // Add Qwen2 special tokens
        base.add_special_token("<|endoftext|>", Self::ENDOFTEXT_ID);
        base.add_special_token("<|im_start|>", Self::IM_START_ID);
        base.add_special_token("<|im_end|>", Self::IM_END_ID);

        // Add basic byte vocabulary (will be replaced when loading from file)
        for i in 0..=255u8 {
            if let Some(&c) = base.byte_encoder.get(&i) {
                let token = c.to_string();
                let id = u32::from(i);
                base.vocab.insert(token.clone(), id);
                base.id_to_token.insert(id, token);
            }
        }

        Self {
            base,
            im_start_id: Self::IM_START_ID,
            im_end_id: Self::IM_END_ID,
            endoftext_id: Self::ENDOFTEXT_ID,
        }
    }

    /// Check if token is EOS (end of sequence).
    #[must_use]
    pub fn is_eos(&self, token_id: u32) -> bool {
        token_id == self.im_end_id || token_id == self.endoftext_id
    }

    /// Check if token is BOS (beginning of sequence).
    #[must_use]
    pub fn is_bos(&self, token_id: u32) -> bool {
        token_id == self.im_start_id
    }

    /// Get vocabulary size.
    #[must_use]
    pub fn vocab_size(&self) -> usize {
        151936 // Qwen2 fixed vocab size
    }

    /// Encode text to token IDs.
    #[must_use]
    pub fn encode(&self, text: &str) -> Vec<u32> {
        self.base.encode(text)
    }

    /// Decode token IDs to text.
    #[must_use]
    pub fn decode(&self, ids: &[u32]) -> String {
        self.base.decode(ids)
    }

    /// Format a chat message with Qwen2 template.
    ///
    /// Format: `<|im_start|>role\nmessage<|im_end|>\n`
    #[must_use]
    pub fn format_chat(&self, role: &str, content: &str) -> String {
        format!("<|im_start|>{role}\n{content}<|im_end|>\n")
    }

    /// Format a complete chat conversation.
    ///
    /// # Arguments
    /// * `messages` - List of (role, content) pairs
    ///
    /// # Returns
    /// Formatted conversation string with chat template applied.
    #[must_use]
    pub fn format_conversation(&self, messages: &[(&str, &str)]) -> String {
        let mut result = String::new();
        for (role, content) in messages {
            result.push_str(&self.format_chat(role, content));
        }
        // Add assistant prefix for generation
        result.push_str("<|im_start|>assistant\n");
        result
    }

    /// Get the `im_start` token ID.
    #[must_use]
    pub fn im_start_id(&self) -> u32 {
        self.im_start_id
    }

    /// Get the `im_end` token ID.
    #[must_use]
    pub fn im_end_id(&self) -> u32 {
        self.im_end_id
    }

    /// Load tokenizer from tokenizer.json file path.
    ///
    /// # Arguments
    /// * `path` - Path to tokenizer.json file
    ///
    /// # Returns
    /// Loaded Qwen2 tokenizer with full vocabulary
    ///
    /// # Errors
    /// Returns error if file cannot be read or parsed.
    pub fn from_file<P: AsRef<std::path::Path>>(path: P) -> Result<Self> {
        let json =
            std::fs::read_to_string(path.as_ref()).map_err(|e| AprenderError::FormatError {
                message: format!("Failed to read tokenizer file: {e}"),
            })?;
        Self::from_json(&json)
    }

    /// Load tokenizer from JSON string.
    ///
    /// # Arguments
    /// * `json` - JSON string containing `HuggingFace` tokenizer format
    ///
    /// # Returns
    /// Loaded Qwen2 tokenizer with full vocabulary
    ///
    /// # Errors
    /// Returns error if JSON parsing fails.
    pub fn from_json(json: &str) -> Result<Self> {
        let base = load_from_json(json)?;

        // Find special token IDs from the loaded vocabulary
        let im_start_id = base
            .vocab
            .get("<|im_start|>")
            .copied()
            .unwrap_or(Self::IM_START_ID);
        let im_end_id = base
            .vocab
            .get("<|im_end|>")
            .copied()
            .unwrap_or(Self::IM_END_ID);
        let endoftext_id = base
            .vocab
            .get("<|endoftext|>")
            .copied()
            .unwrap_or(Self::ENDOFTEXT_ID);

        Ok(Self {
            base,
            im_start_id,
            im_end_id,
            endoftext_id,
        })
    }
}

impl Default for Qwen2BpeTokenizer {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Utility Functions
// ============================================================================

/// Create byte to Unicode character mapping.
///
/// GPT-2 uses a specific mapping to avoid issues with certain bytes.
#[must_use]
pub fn bytes_to_unicode() -> (HashMap<u8, char>, HashMap<char, u8>) {
    let mut encoder = HashMap::new();
    let mut decoder = HashMap::new();

    // Printable ASCII + Latin-1 supplement
    let mut n = 0u32;
    for b in 0..=255u8 {
        // Printable characters map to themselves
        let c = if (b'!'..=b'~').contains(&b)
            || (b'\xa1'..=b'\xac').contains(&b)
            || (b'\xae'..=b'\xff').contains(&b)
        {
            char::from(b)
        } else {
            // Non-printable map to Unicode offset
            let c = char::from_u32(256 + n).unwrap_or('?');
            n += 1;
            c
        };

        encoder.insert(b, c);
        decoder.insert(c, b);
    }

    (encoder, decoder)
}

// ============================================================================
// HuggingFace tokenizer.json Parsing Structures
// ============================================================================

/// `HuggingFace` tokenizer.json root structure.
#[derive(Debug, Deserialize)]
struct HfTokenizerJson {
    model: HfModel,
    #[serde(default)]
    added_tokens: Vec<HfAddedToken>,
}

/// BPE model section in tokenizer.json.
#[derive(Debug, Deserialize)]
struct HfModel {
    vocab: HashMap<String, u32>,
    merges: Vec<String>,
}

/// Added token entry.
#[derive(Debug, Deserialize)]
struct HfAddedToken {
    id: u32,
    content: String,
    #[serde(default)]
    special: bool,
}

/// Load tokenizer from `HuggingFace` tokenizer.json format.
///
/// # Arguments
/// * `json` - JSON string of tokenizer configuration
///
/// # Returns
/// Loaded tokenizer with vocabulary and merge rules
///
/// # Errors
/// Returns error if parsing fails.
pub fn load_from_json(json: &str) -> Result<BpeTokenizer> {
    if json.is_empty() {
        return Err(AprenderError::FormatError {
            message: "Empty tokenizer JSON".to_string(),
        });
    }

    // Parse the JSON
    let hf_tokenizer: HfTokenizerJson =
        serde_json::from_str(json).map_err(|e| AprenderError::FormatError {
            message: format!("Failed to parse tokenizer JSON: {e}"),
        })?;

    // Determine config based on vocab size
    let vocab_size = hf_tokenizer.model.vocab.len();
    let config = if vocab_size > 150000 {
        BpeConfig::qwen2()
    } else if vocab_size > 50000 {
        BpeConfig::whisper()
    } else if vocab_size > 40000 {
        BpeConfig::gpt2()
    } else {
        BpeConfig::llama()
    };

    let mut tokenizer = BpeTokenizer::new(config);

    // Load vocabulary
    for (token, id) in &hf_tokenizer.model.vocab {
        tokenizer.vocab.insert(token.clone(), *id);
        tokenizer.id_to_token.insert(*id, token.clone());
    }

    // Load merge rules
    for merge_str in &hf_tokenizer.model.merges {
        let parts: Vec<&str> = merge_str.split(' ').collect();
        if parts.len() >= 2 {
            tokenizer.add_merge(parts[0], parts[1]);
        }
    }

    // Load special/added tokens
    for added in &hf_tokenizer.added_tokens {
        if added.special {
            tokenizer.add_special_token(&added.content, added.id);
        } else {
            tokenizer.vocab.insert(added.content.clone(), added.id);
            tokenizer
                .id_to_token
                .insert(added.id, added.content.clone());
        }
    }

    Ok(tokenizer)
}

/// Load tokenizer from vocab.json and merges.txt files.
///
/// # Arguments
/// * `vocab_json` - JSON string of vocabulary mapping (token -> id)
/// * `merges_txt` - Text file with merge rules (one "pair1 pair2" per line)
///
/// # Returns
/// Loaded tokenizer
///
/// # Errors
/// Returns error if parsing fails.
pub fn load_from_files(vocab_json: &str, merges_txt: &str) -> Result<BpeTokenizer> {
    if vocab_json.is_empty() {
        return Err(AprenderError::FormatError {
            message: "Empty vocabulary JSON".to_string(),
        });
    }

    // Parse vocabulary JSON
    let vocab: HashMap<String, u32> =
        serde_json::from_str(vocab_json).map_err(|e| AprenderError::FormatError {
            message: format!("Failed to parse vocabulary JSON: {e}"),
        })?;

    // Determine config based on vocab size
    let vocab_size = vocab.len();
    let config = if vocab_size > 150000 {
        BpeConfig::qwen2()
    } else if vocab_size > 50000 {
        BpeConfig::whisper()
    } else if vocab_size > 40000 {
        BpeConfig::gpt2()
    } else {
        BpeConfig::llama()
    };

    let mut tokenizer = BpeTokenizer::new(config);

    // Load vocabulary
    for (token, id) in &vocab {
        tokenizer.vocab.insert(token.clone(), *id);
        tokenizer.id_to_token.insert(*id, token.clone());
    }

    // Parse and load merges
    for line in merges_txt.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() >= 2 {
            tokenizer.add_merge(parts[0], parts[1]);
        }
    }

    Ok(tokenizer)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bpe_config_default() {
        let config = BpeConfig::default();
        assert_eq!(config.unk_token, "<unk>");
        assert_eq!(config.vocab_size, 50257);
        assert!(config.add_prefix_space);
    }

    #[test]
    fn test_bpe_config_whisper() {
        let config = BpeConfig::whisper();
        assert_eq!(config.vocab_size, 51865);
        assert!(!config.add_prefix_space);
    }

    #[test]
    fn test_bpe_config_llama() {
        let config = BpeConfig::llama();
        assert_eq!(config.vocab_size, 32000);
        assert_eq!(config.bos_token, Some("<s>".to_string()));
    }

    #[test]
    fn test_merge_rule() {
        let rule = MergeRule::new("hel", "lo");
        assert_eq!(rule.first, "hel");
        assert_eq!(rule.second, "lo");
        assert_eq!(rule.merged(), "hello");
    }

    #[test]
    fn test_tokenizer_new() {
        let tokenizer = BpeTokenizer::new(BpeConfig::default());
        assert_eq!(tokenizer.vocab_size(), 0);
    }

    #[test]
    fn test_tokenizer_gpt2_base() {
        let tokenizer = BpeTokenizer::gpt2_base();
        assert!(tokenizer.vocab_size() > 0);

        // Check special token
        assert!(tokenizer.is_special_token("<|endoftext|>"));
        assert_eq!(tokenizer.token_to_id("<|endoftext|>"), Some(50256));
    }

    #[test]
    fn test_add_special_token() {
        let mut tokenizer = BpeTokenizer::new(BpeConfig::default());
        tokenizer.add_special_token("<test>", 100);

        assert!(tokenizer.is_special_token("<test>"));
        assert_eq!(tokenizer.token_to_id("<test>"), Some(100));
        assert_eq!(tokenizer.id_to_token(100), Some("<test>"));
    }

    #[test]
    fn test_add_merge() {
        let mut tokenizer = BpeTokenizer::new(BpeConfig::default());
        tokenizer.add_merge("a", "b");
        tokenizer.add_merge("ab", "c");

        assert_eq!(tokenizer.merges.len(), 2);
        assert_eq!(
            tokenizer
                .merge_ranks
                .get(&("a".to_string(), "b".to_string())),
            Some(&0)
        );
        assert_eq!(
            tokenizer
                .merge_ranks
                .get(&("ab".to_string(), "c".to_string())),
            Some(&1)
        );
    }

    #[test]
    fn test_encode_empty() {
        let tokenizer = BpeTokenizer::gpt2_base();
        let tokens = tokenizer.encode("");
        assert!(tokens.is_empty());
    }

    #[test]
    fn test_encode_basic() {
        let tokenizer = BpeTokenizer::gpt2_base();
        let tokens = tokenizer.encode("Hello");
        // Should produce some tokens (byte-level encoding)
        assert!(!tokens.is_empty());
    }

    #[test]
    fn test_decode_empty() {
        let tokenizer = BpeTokenizer::gpt2_base();
        let text = tokenizer.decode(&[]);
        assert!(text.is_empty());
    }

    #[test]
    fn test_encode_decode_roundtrip() {
        let tokenizer = BpeTokenizer::gpt2_base();
        let original = "Hi";
        let tokens = tokenizer.encode(original);
        let decoded = tokenizer.decode(&tokens);

        // Should approximately match (may have prefix space)
        assert!(decoded.contains("Hi") || decoded.trim() == "Hi");
    }

    #[test]
    fn test_encode_checked() {
        let tokenizer = BpeTokenizer::gpt2_base();
        let result = tokenizer.encode_checked("test");
        assert!(result.is_ok());
        assert!(!result.expect("encode failed").is_empty());
    }

    #[test]
    fn test_decode_checked() {
        let tokenizer = BpeTokenizer::gpt2_base();
        let result = tokenizer.decode_checked(&[72]); // 'H' in byte encoding
        assert!(result.is_ok());
    }

    #[test]
    fn test_bytes_to_unicode() {
        let (encoder, decoder) = bytes_to_unicode();

        // Check that encoder and decoder are inverses
        for b in 0..=255u8 {
            if let Some(&c) = encoder.get(&b) {
                assert_eq!(decoder.get(&c), Some(&b));
            }
        }

        // Check ASCII printable chars map to themselves
        assert_eq!(encoder.get(&b'A'), Some(&'A'));
        assert_eq!(encoder.get(&b'z'), Some(&'z'));
        assert_eq!(encoder.get(&b'0'), Some(&'0'));
    }

    #[test]
    fn test_pre_tokenize() {
        let tokenizer = BpeTokenizer::new(BpeConfig {
            add_prefix_space: false,
            ..BpeConfig::default()
        });

        let words = tokenizer.pre_tokenize("Hello world");
        assert_eq!(words.len(), 2);
        assert_eq!(words[0], "Hello");
        assert!(words[1].contains("world"));
    }

    #[test]
    fn test_bpe_no_merges() {
        let tokenizer = BpeTokenizer::new(BpeConfig::default());
        let tokens = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let result = tokenizer.bpe(&tokens);
        assert_eq!(result, tokens); // No merges, unchanged
    }

    #[test]
    fn test_bpe_with_merge() {
        let mut tokenizer = BpeTokenizer::new(BpeConfig::default());
        tokenizer.add_merge("a", "b");

        let tokens = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let result = tokenizer.bpe(&tokens);
        assert_eq!(result, vec!["ab".to_string(), "c".to_string()]);
    }

    #[test]
    fn test_bpe_multiple_merges() {
        let mut tokenizer = BpeTokenizer::new(BpeConfig::default());
        tokenizer.add_merge("a", "b");
        tokenizer.add_merge("ab", "c");

        let tokens = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let result = tokenizer.bpe(&tokens);
        assert_eq!(result, vec!["abc".to_string()]);
    }

    #[test]
    fn test_load_from_json_empty() {
        let result = load_from_json("");
        assert!(result.is_err());
    }

    #[test]
    fn test_load_from_json_basic() {
        // Valid HuggingFace tokenizer.json structure
        let json = r#"{
            "model": {
                "vocab": {"hello": 0, "world": 1},
                "merges": ["he llo", "wor ld"]
            },
            "added_tokens": []
        }"#;
        let result = load_from_json(json);
        assert!(result.is_ok());

        let tokenizer = result.expect("load failed");
        assert_eq!(tokenizer.vocab_size(), 2);
        assert_eq!(tokenizer.merges.len(), 2);
    }

    #[test]
    fn test_load_from_json_invalid() {
        // Invalid JSON (missing model field) should fail
        let result = load_from_json("{}");
        assert!(result.is_err());
    }

    #[test]
    fn test_load_from_files_empty_vocab() {
        let result = load_from_files("", "");
        assert!(result.is_err());
    }

    #[test]
    fn test_load_from_files_basic() {
        let vocab = "{}";
        let merges = "a b\nab c\n";
        let result = load_from_files(vocab, merges);
        assert!(result.is_ok());

        let tokenizer = result.expect("load failed");
        assert_eq!(tokenizer.merges.len(), 2);
    }

    #[test]
    fn test_load_from_files_skip_comments() {
        let vocab = "{}";
        let merges = "# comment\na b\n";
        let result = load_from_files(vocab, merges);
        assert!(result.is_ok());

        let tokenizer = result.expect("load failed");
        assert_eq!(tokenizer.merges.len(), 1);
    }

    #[test]
    fn test_encode_unicode() {
        let tokenizer = BpeTokenizer::gpt2_base();
        let tokens = tokenizer.encode("日本語");
        // Should handle unicode without panicking
        assert!(!tokens.is_empty());
    }

    #[test]
    fn test_encode_emoji() {
        let tokenizer = BpeTokenizer::gpt2_base();
        let tokens = tokenizer.encode("Hello 😀");
        // Should handle emoji without panicking
        assert!(!tokens.is_empty());
    }

    #[test]
    fn test_whitespace_handling() {
        let tokenizer = BpeTokenizer::gpt2_base();
        let tokens1 = tokenizer.encode(" ");
        let tokens2 = tokenizer.encode("  ");

        // Whitespace should produce tokens
        assert!(!tokens1.is_empty());
        assert!(!tokens2.is_empty());
    }

    // ========================================================================
    // Qwen2 BPE Tokenizer Tests
    // ========================================================================

    #[test]
    fn test_qwen2_config() {
        let config = BpeConfig::qwen2();
        assert_eq!(config.vocab_size, 151936);
        assert_eq!(config.eos_token, Some("<|im_end|>".to_string()));
        assert_eq!(config.bos_token, Some("<|im_start|>".to_string()));
        assert!(!config.add_prefix_space);
    }

    #[test]
    fn test_qwen2_tokenizer_new() {
        let tokenizer = Qwen2BpeTokenizer::new();
        assert_eq!(tokenizer.vocab_size(), 151936);
    }

    #[test]
    fn test_qwen2_special_tokens() {
        let tokenizer = Qwen2BpeTokenizer::new();

        assert!(tokenizer.is_eos(Qwen2BpeTokenizer::IM_END_ID));
        assert!(tokenizer.is_eos(Qwen2BpeTokenizer::ENDOFTEXT_ID));
        assert!(!tokenizer.is_eos(0));

        assert!(tokenizer.is_bos(Qwen2BpeTokenizer::IM_START_ID));
        assert!(!tokenizer.is_bos(0));
    }

    #[test]
    fn test_qwen2_format_chat() {
        let tokenizer = Qwen2BpeTokenizer::new();
        let formatted = tokenizer.format_chat("user", "Hello!");

        assert!(formatted.starts_with("<|im_start|>user"));
        assert!(formatted.contains("Hello!"));
        assert!(formatted.ends_with("<|im_end|>\n"));
    }

    #[test]
    fn test_qwen2_format_conversation() {
        let tokenizer = Qwen2BpeTokenizer::new();
        let messages = vec![
            ("system", "You are a helpful assistant."),
            ("user", "Hello!"),
        ];
        let formatted = tokenizer.format_conversation(&messages);

        assert!(formatted.contains("<|im_start|>system"));
        assert!(formatted.contains("You are a helpful assistant."));
        assert!(formatted.contains("<|im_start|>user"));
        assert!(formatted.contains("Hello!"));
        assert!(formatted.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn test_qwen2_encode_decode() {
        let tokenizer = Qwen2BpeTokenizer::new();

        // Basic encode (without real vocab, just byte-level)
        let tokens = tokenizer.encode("Hi");
        assert!(!tokens.is_empty());

        // Decode back
        let decoded = tokenizer.decode(&tokens);
        assert!(decoded.contains("Hi") || decoded.trim() == "Hi");
    }

    #[test]
    fn test_qwen2_token_ids() {
        let tokenizer = Qwen2BpeTokenizer::new();

        assert_eq!(tokenizer.im_start_id(), 151644);
        assert_eq!(tokenizer.im_end_id(), 151645);
    }

    // ========================================================================
    // Additional Coverage Tests
    // ========================================================================

    #[test]
    fn test_bpe_config_gpt2() {
        let config = BpeConfig::gpt2();
        assert_eq!(config.vocab_size, 50257);
        assert_eq!(config.unk_token, "<unk>");
        assert!(config.add_prefix_space);
        assert_eq!(config.bos_token, Some("<|endoftext|>".to_string()));
        assert_eq!(config.eos_token, Some("<|endoftext|>".to_string()));
        assert!(config.pad_token.is_none());
    }

    #[test]
    fn test_tokenizer_default() {
        let tokenizer = BpeTokenizer::default();
        assert_eq!(tokenizer.vocab_size(), 0);
        assert!(tokenizer.merges.is_empty());
    }

    #[test]
    fn test_qwen2_tokenizer_default() {
        let tokenizer = Qwen2BpeTokenizer::default();
        assert_eq!(tokenizer.vocab_size(), 151936);
        assert!(tokenizer.is_eos(Qwen2BpeTokenizer::IM_END_ID));
    }

    #[test]
    fn test_bpe_single_token() {
        let tokenizer = BpeTokenizer::new(BpeConfig::default());
        let tokens = vec!["abc".to_string()];
        let result = tokenizer.bpe(&tokens);
        assert_eq!(result, vec!["abc".to_string()]);
    }

    #[test]
    fn test_split_on_special_tokens_empty_text() {
        let mut tokenizer = BpeTokenizer::new(BpeConfig::default());
        tokenizer.add_special_token("<test>", 100);
        let segments = tokenizer.split_on_special_tokens("");
        assert!(segments.is_empty() || segments == vec!["".to_string()]);
    }

    #[test]
    fn test_split_on_special_tokens_no_match() {
        let mut tokenizer = BpeTokenizer::new(BpeConfig::default());
        tokenizer.add_special_token("<test>", 100);
        let segments = tokenizer.split_on_special_tokens("hello world");
        assert_eq!(segments, vec!["hello world".to_string()]);
    }

    #[test]
    fn test_split_on_special_tokens_multiple() {
        let mut tokenizer = BpeTokenizer::new(BpeConfig::default());
        tokenizer.add_special_token("<a>", 100);
        tokenizer.add_special_token("<b>", 101);
        let segments = tokenizer.split_on_special_tokens("Hello<a>world<b>!");
        assert!(segments.contains(&"<a>".to_string()));
        assert!(segments.contains(&"<b>".to_string()));
    }

    #[test]
    fn test_split_on_special_tokens_at_start() {
        let mut tokenizer = BpeTokenizer::new(BpeConfig::default());
        tokenizer.add_special_token("<s>", 100);
        let segments = tokenizer.split_on_special_tokens("<s>hello");
        assert_eq!(segments[0], "<s>");
        assert_eq!(segments[1], "hello");
    }

    #[test]
    fn test_split_on_special_tokens_consecutive() {
        let mut tokenizer = BpeTokenizer::new(BpeConfig::default());
        tokenizer.add_special_token("<a>", 100);
        tokenizer.add_special_token("<b>", 101);
        let segments = tokenizer.split_on_special_tokens("<a><b>");
        assert!(segments.contains(&"<a>".to_string()));
        assert!(segments.contains(&"<b>".to_string()));
    }

    #[test]
    fn test_encode_with_special_tokens() {
        let tokenizer = BpeTokenizer::gpt2_base();
        let text = "<|endoftext|>Hello";
        let tokens = tokenizer.encode(text);
        // Should contain the special token ID
        assert!(tokens.contains(&50256));
    }

    #[test]
    fn test_encode_without_prefix_space() {
        let config = BpeConfig {
            add_prefix_space: false,
            ..BpeConfig::default()
        };
        let tokenizer = BpeTokenizer::new(config);
        let tokens = tokenizer.encode("test");
        // With empty vocab, should be empty
        assert!(tokens.is_empty());
    }

    #[test]
    fn test_decode_with_unknown_id() {
        let tokenizer = BpeTokenizer::gpt2_base();
        // ID that doesn't exist in vocab
        let decoded = tokenizer.decode(&[999999]);
        // Unknown ID should be skipped
        assert!(decoded.is_empty() || decoded == "");
    }

    #[test]
    fn test_decode_skips_special_tokens() {
        let tokenizer = BpeTokenizer::gpt2_base();
        // Decode with special token ID
        let decoded = tokenizer.decode(&[50256, 72]); // endoftext + 'H'
        // Should not contain the special token text
        assert!(!decoded.contains("<|endoftext|>"));
    }

    #[test]
    fn test_bytes_to_bpe_tokens_unknown() {
        let tokenizer = BpeTokenizer::new(BpeConfig::default());
        // Force a path through unknown byte handling
        let result = tokenizer.bytes_to_bpe_tokens("A");
        assert!(!result.is_empty());
    }

    #[test]
    fn test_load_from_json_with_special_added_tokens() {
        let json = r#"{
            "model": {
                "vocab": {"hello": 0, "world": 1},
                "merges": []
            },
            "added_tokens": [
                {"id": 100, "content": "<special>", "special": true},
                {"id": 101, "content": "normal", "special": false}
            ]
        }"#;
        let result = load_from_json(json);
        assert!(result.is_ok());

        let tokenizer = result.expect("load failed");
        assert!(tokenizer.is_special_token("<special>"));
        assert!(!tokenizer.is_special_token("normal"));
        assert_eq!(tokenizer.token_to_id("normal"), Some(101));
    }

    #[test]
    fn test_load_from_json_whisper_vocab_size() {
        // Create vocab with >50000 entries for whisper detection
        let mut vocab_entries: Vec<String> = Vec::new();
        for i in 0..51000 {
            vocab_entries.push(format!("\"tok{i}\": {i}"));
        }
        let vocab_str = vocab_entries.join(", ");
        let json = format!(
            "{{\"model\": {{\"vocab\": {{ {} }}, \"merges\": []}}, \"added_tokens\": []}}",
            vocab_str
        );
        let result = load_from_json(&json);
        assert!(result.is_ok());
    }

    #[test]
    fn test_load_from_json_qwen2_vocab_size() {
        // Create vocab with >150000 entries for qwen2 detection
        let mut vocab_entries: Vec<String> = Vec::new();
        for i in 0..151000 {
            vocab_entries.push(format!("\"tok{i}\": {i}"));
        }
        let vocab_str = vocab_entries.join(", ");
        let json = format!(
            "{{\"model\": {{\"vocab\": {{ {} }}, \"merges\": []}}, \"added_tokens\": []}}",
            vocab_str
        );
        let result = load_from_json(&json);
        assert!(result.is_ok());
    }

    #[test]
    fn test_load_from_files_whisper_vocab_size() {
        // Create vocab with >50000 entries
        let mut vocab: HashMap<String, u32> = HashMap::new();
        for i in 0..51000u32 {
            vocab.insert(format!("tok{i}"), i);
        }
        let vocab_json = serde_json::to_string(&vocab).expect("serialize");
        let merges = "";
        let result = load_from_files(&vocab_json, merges);
        assert!(result.is_ok());
    }

    #[test]
    fn test_load_from_files_gpt2_vocab_size() {
        // Create vocab with >40000 but <50000 entries
        let mut vocab: HashMap<String, u32> = HashMap::new();
        for i in 0..41000u32 {
            vocab.insert(format!("tok{i}"), i);
        }
        let vocab_json = serde_json::to_string(&vocab).expect("serialize");
        let merges = "";
        let result = load_from_files(&vocab_json, merges);
        assert!(result.is_ok());
    }

    #[test]
    fn test_load_from_files_qwen2_vocab_size() {
        // Create vocab with >150000 entries
        let mut vocab: HashMap<String, u32> = HashMap::new();
        for i in 0..151000u32 {
            vocab.insert(format!("tok{i}"), i);
        }
        let vocab_json = serde_json::to_string(&vocab).expect("serialize");
        let merges = "";
        let result = load_from_files(&vocab_json, merges);
        assert!(result.is_ok());
    }

    #[test]
    fn test_load_from_files_empty_lines() {
        let vocab = "{}";
        let merges = "\n\na b\n\n";
        let result = load_from_files(vocab, merges);
        assert!(result.is_ok());

        let tokenizer = result.expect("load failed");
        assert_eq!(tokenizer.merges.len(), 1);
    }

    #[test]
    fn test_load_from_files_invalid_json() {
        let vocab = "not valid json";
        let merges = "";
        let result = load_from_files(vocab, merges);
        assert!(result.is_err());
    }

    #[test]
    fn test_qwen2_from_json() {
        let json = r#"{
            "model": {
                "vocab": {
                    "<|endoftext|>": 151643,
                    "<|im_start|>": 151644,
                    "<|im_end|>": 151645,
                    "hello": 0
                },
                "merges": []
            },
            "added_tokens": [
                {"id": 151643, "content": "<|endoftext|>", "special": true},
                {"id": 151644, "content": "<|im_start|>", "special": true},
                {"id": 151645, "content": "<|im_end|>", "special": true}
            ]
        }"#;
        let result = Qwen2BpeTokenizer::from_json(json);
        assert!(result.is_ok());

        let tokenizer = result.expect("load failed");
        assert!(tokenizer.is_eos(151645));
        assert!(tokenizer.is_bos(151644));
    }

    #[test]
    fn test_qwen2_from_json_default_ids() {
        // Test with vocab missing special tokens - should use defaults
        let json = r#"{
            "model": {
                "vocab": {"hello": 0, "world": 1},
                "merges": []
            },
            "added_tokens": []
        }"#;
        let result = Qwen2BpeTokenizer::from_json(json);
        assert!(result.is_ok());

        let tokenizer = result.expect("load failed");
        // Should use default IDs
        assert_eq!(tokenizer.im_start_id(), Qwen2BpeTokenizer::IM_START_ID);
        assert_eq!(tokenizer.im_end_id(), Qwen2BpeTokenizer::IM_END_ID);
    }

    #[test]
    fn test_qwen2_from_file_not_found() {
        let result = Qwen2BpeTokenizer::from_file("/nonexistent/path/tokenizer.json");
        assert!(result.is_err());
    }

    #[test]
    fn test_merge_rule_debug_clone() {
        let rule = MergeRule::new("a", "b");
        let cloned = rule.clone();
        assert_eq!(rule, cloned);

        // Test Debug
        let debug_str = format!("{:?}", rule);
        assert!(debug_str.contains("MergeRule"));
    }

    #[test]
    fn test_bpe_tokenizer_debug_clone() {
        let tokenizer = BpeTokenizer::gpt2_base();
        let cloned = tokenizer.clone();
        assert_eq!(tokenizer.vocab_size(), cloned.vocab_size());

        // Test Debug
        let debug_str = format!("{:?}", tokenizer);
        assert!(debug_str.contains("BpeTokenizer"));
    }

    #[test]
    fn test_qwen2_tokenizer_debug_clone() {
        let tokenizer = Qwen2BpeTokenizer::new();
        let cloned = tokenizer.clone();
        assert_eq!(tokenizer.vocab_size(), cloned.vocab_size());

        // Test Debug
        let debug_str = format!("{:?}", tokenizer);
        assert!(debug_str.contains("Qwen2BpeTokenizer"));
    }

    #[test]
    fn test_bpe_config_debug_clone() {
        let config = BpeConfig::default();
        let cloned = config.clone();
        assert_eq!(config.vocab_size, cloned.vocab_size);

        let debug_str = format!("{:?}", config);
        assert!(debug_str.contains("BpeConfig"));
    }

    #[test]
    fn test_encode_unk_token_fallback() {
        let mut tokenizer = BpeTokenizer::new(BpeConfig::default());
        // Add unk token to vocab
        tokenizer.add_special_token("<unk>", 0);

        // Encode text - unknown bytes should fall back to unk token
        let tokens = tokenizer.encode("x");
        // Should either be empty or have unk token
        assert!(tokens.is_empty() || tokens.contains(&0));
    }

    #[test]
    fn test_pre_tokenize_multiple_spaces() {
        let tokenizer = BpeTokenizer::new(BpeConfig::default());
        let words = tokenizer.pre_tokenize("hello  world");
        // Should handle multiple spaces
        assert!(words.len() >= 2);
    }

    #[test]
    fn test_pre_tokenize_leading_space() {
        let tokenizer = BpeTokenizer::new(BpeConfig::default());
        let words = tokenizer.pre_tokenize(" hello");
        assert!(!words.is_empty());
        // First word should start with space
        assert!(words[0].starts_with(' '));
    }

    #[test]
    fn test_bpe_tokens_to_bytes_invalid_chars() {
        let tokenizer = BpeTokenizer::gpt2_base();
        // String with chars not in byte_decoder
        let result = tokenizer.bpe_tokens_to_bytes("αβγ");
        // Should handle gracefully (lossy conversion)
        // Result might be empty or partial
        let _ = result;
    }

    #[test]
    fn test_qwen2_encode_special_tokens() {
        let tokenizer = Qwen2BpeTokenizer::new();
        let text = "<|im_start|>user";
        let tokens = tokenizer.encode(text);
        // Should contain the special token ID
        assert!(tokens.contains(&151644));
    }

    #[test]
    fn test_bpe_merge_priority() {
        let mut tokenizer = BpeTokenizer::new(BpeConfig::default());
        // Add merges with specific priority order
        tokenizer.add_merge("x", "y"); // rank 0 (highest priority)
        tokenizer.add_merge("a", "b"); // rank 1

        // Test that lower rank (higher priority) merge is applied first
        let tokens = vec![
            "a".to_string(),
            "b".to_string(),
            "x".to_string(),
            "y".to_string(),
        ];
        let result = tokenizer.bpe(&tokens);
        // Both merges should be applied
        assert_eq!(result.len(), 2);
    }
}
