//! Tokenizer for text encoding and decoding
//!
//! Implements tokenization for transformer models:
//! - BPE (Byte Pair Encoding) - Used by GPT models
//! - Vocabulary management
//! - Special token handling
//!
//! ## Example
//!
//! ```rust,ignore
//! use realizar::Tokenizer;
//!
//! let tokenizer = Tokenizer::from_vocab(vocab);
//! let token_ids = tokenizer.encode("Hello, world!")?;
//! let text = tokenizer.decode(&token_ids)?;
//! ```

use std::collections::HashMap;

use crate::error::{RealizarError, Result};

/// Vocabulary mapping between tokens and IDs
#[derive(Debug, Clone)]
pub struct Vocabulary {
    /// Token to ID mapping
    token_to_id: HashMap<String, u32>,
    /// ID to token mapping
    id_to_token: HashMap<u32, String>,
}

impl Vocabulary {
    /// Create a new vocabulary from token list
    ///
    /// # Arguments
    ///
    /// * `tokens` - List of tokens in order (index = token ID)
    ///
    /// # Errors
    ///
    /// Returns error if tokens list is empty or contains duplicates
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let vocab = Vocabulary::from_tokens(vec![
    ///     "<unk>".to_string(),
    ///     "hello".to_string(),
    ///     "world".to_string(),
    /// ])?;
    /// ```
    pub fn from_tokens(tokens: Vec<String>) -> Result<Self> {
        if tokens.is_empty() {
            return Err(RealizarError::UnsupportedOperation {
                operation: "create_vocabulary".to_string(),
                reason: "Vocabulary cannot be empty".to_string(),
            });
        }

        let mut token_to_id = HashMap::new();
        let mut id_to_token = HashMap::new();

        for (id, token) in tokens.into_iter().enumerate() {
            let id = u32::try_from(id).map_err(|_| RealizarError::UnsupportedOperation {
                operation: "convert_token_id".to_string(),
                reason: format!("Token ID {id} exceeds u32 limit"),
            })?;

            if token_to_id.contains_key(&token) {
                return Err(RealizarError::UnsupportedOperation {
                    operation: "create_vocabulary".to_string(),
                    reason: format!("Duplicate token: {token}"),
                });
            }

            token_to_id.insert(token.clone(), id);
            id_to_token.insert(id, token);
        }

        Ok(Self {
            token_to_id,
            id_to_token,
        })
    }

    /// Get token ID for a token
    ///
    /// Returns `None` if token not in vocabulary
    #[must_use]
    pub fn get_id(&self, token: &str) -> Option<u32> {
        self.token_to_id.get(token).copied()
    }

    /// Get token for a token ID
    ///
    /// Returns `None` if ID not in vocabulary
    #[must_use]
    pub fn get_token(&self, id: u32) -> Option<&str> {
        self.id_to_token.get(&id).map(String::as_str)
    }

    /// Get vocabulary size
    #[must_use]
    pub fn size(&self) -> usize {
        self.token_to_id.len()
    }
}

/// Tokenizer for encoding and decoding text
#[derive(Debug, Clone)]
pub struct Tokenizer {
    /// Vocabulary
    vocab: Vocabulary,
    /// Unknown token ID
    unk_token_id: u32,
}

/// BPE (Byte Pair Encoding) tokenizer
///
/// Implements subword tokenization using byte pair encoding algorithm.
/// Used by GPT-2, GPT-3, and many other models.
///
/// ## vLLM-Style Property Caching (PMAT-805)
///
/// Properties like vocab_size and max_token_id are cached at construction
/// to avoid repeated HashMap operations during hot inference paths.
/// This is especially important for large vocabularies (e.g., Qwen2's 151K tokens).
#[derive(Debug, Clone)]
pub struct BPETokenizer {
    /// Token to ID mapping
    token_to_id: HashMap<String, u32>,
    /// ID to token mapping
    id_to_token: HashMap<u32, String>,
    /// Unknown token ID
    unk_token_id: u32,
    /// Cached vocabulary size (PMAT-805: vLLM pattern)
    vocab_size: usize,
    /// Cached maximum token ID (PMAT-805: vLLM pattern)
    max_token_id: u32,
    /// GH-88: Optional BPE merge rules for proper encoding.
    /// When present, `encode()` uses merge-based BPE instead of greedy longest-match.
    /// Required for HuggingFace vocabularies (SafeTensors/APR imports).
    merge_rules: Vec<(String, String)>,
    /// GH-88: Special tokens for atomic tokenization (not split by BPE).
    special_tokens: HashMap<String, u32>,
}

impl BPETokenizer {
    /// Create a new BPE tokenizer
    ///
    /// # Arguments
    ///
    /// * `vocab` - List of tokens (index = token ID)
    /// * `merges` - List of merge pairs in priority order
    /// * `unk_token` - Unknown token string
    ///
    /// # Errors
    ///
    /// Returns error if vocabulary is empty or unknown token not found
    pub fn new(
        vocab: Vec<String>,
        _merges: Vec<(String, String)>,
        unk_token: &str,
    ) -> Result<Self> {
        if vocab.is_empty() {
            return Err(RealizarError::UnsupportedOperation {
                operation: "create_bpe_tokenizer".to_string(),
                reason: "Vocabulary cannot be empty".to_string(),
            });
        }

        let mut token_to_id = HashMap::new();
        let mut id_to_token = HashMap::new();

        for (id, token) in vocab.into_iter().enumerate() {
            let id = u32::try_from(id).map_err(|_| RealizarError::UnsupportedOperation {
                operation: "convert_token_id".to_string(),
                reason: format!("Token ID {id} exceeds u32 limit"),
            })?;
            token_to_id.insert(token.clone(), id);
            id_to_token.insert(id, token);
        }

        let unk_token_id =
            *token_to_id
                .get(unk_token)
                .ok_or_else(|| RealizarError::UnsupportedOperation {
                    operation: "create_bpe_tokenizer".to_string(),
                    reason: format!("Unknown token '{unk_token}' not in vocabulary"),
                })?;

        // PMAT-805: Cache vocabulary properties at construction (vLLM pattern)
        // This avoids repeated HashMap operations during inference
        let vocab_size = token_to_id.len();
        let max_token_id = id_to_token.keys().copied().max().unwrap_or(0);

        Ok(Self {
            token_to_id,
            id_to_token,
            unk_token_id,
            vocab_size,
            max_token_id,
            merge_rules: Vec::new(),
            special_tokens: HashMap::new(),
        })
    }

    /// GH-88: Create a BPE tokenizer with merge rules for proper encoding.
    ///
    /// HuggingFace vocabularies require merge-based BPE (not greedy longest-match).
    /// When merge rules are provided, `encode()` delegates to `bpe_encode()` from
    /// `apr::tokenizer` which handles special tokens atomically and applies merges.
    pub fn with_merges(
        vocab: Vec<String>,
        merges: Vec<(String, String)>,
        unk_token: &str,
    ) -> Result<Self> {
        let mut tokenizer = Self::new(vocab, vec![], unk_token)?;
        // Extract special tokens from vocabulary for atomic tokenization
        let special_tokens: HashMap<String, u32> = tokenizer
            .token_to_id
            .iter()
            .filter(|(k, _)| {
                (k.starts_with("<|") && k.ends_with("|>"))
                    || (k.starts_with("<")
                        && k.ends_with(">")
                        && k.len() > 2
                        && !k.starts_with("<0x"))
            })
            .map(|(k, v)| (k.clone(), *v))
            .collect();
        tokenizer.merge_rules = merges;
        tokenizer.special_tokens = special_tokens;
        Ok(tokenizer)
    }

    /// Get maximum token ID (cached, O(1))
    ///
    /// PMAT-805: Returns pre-computed max token ID for fast access
    #[inline]
    #[must_use]
    pub fn max_token_id(&self) -> u32 {
        self.max_token_id
    }

    /// Encode text to token IDs.
    ///
    /// GH-88: When merge rules are present (HuggingFace tokenizers), uses proper
    /// BPE encoding with special token support. Otherwise falls back to greedy
    /// longest-match (sufficient for GGUF vocabularies).
    ///
    /// # Arguments
    ///
    /// * `text` - Input text to encode
    ///
    /// # Returns
    ///
    /// Vector of token IDs
    #[must_use]
    pub fn encode(&self, text: &str) -> Vec<u32> {
        contract_pre_encode!();
        if text.is_empty() {
            return Vec::new();
        }

        // GH-88: Use proper BPE when merge rules are available
        if !self.merge_rules.is_empty() {
            return crate::apr::tokenizer::bpe_encode(
                text,
                &self.token_to_id,
                &self.merge_rules,
                &self.special_tokens,
            );
        }

        // Convert to GPT-2 encoding: space -> Ġ, newline -> Ċ
        let processed: String = text
            .chars()
            .map(|c| match c {
                ' ' => 'Ġ',  // U+0120
                '\n' => 'Ċ', // U+010A
                '\r' => 'Ḃ', // U+1E02
                _ => c,
            })
            .collect();

        let mut tokens = Vec::new();
        let mut remaining = processed.as_str();

        while !remaining.is_empty() {
            // Greedy longest match
            let mut best_len = 0;
            let mut best_id = None;

            // Collect character byte offsets for proper slicing
            let char_indices: Vec<usize> = remaining
                .char_indices()
                .map(|(i, _)| i)
                .chain(std::iter::once(remaining.len()))
                .collect();

            // Try all prefixes up to 32 chars
            for char_count in 1..=char_indices.len().saturating_sub(1).min(32) {
                let byte_end = char_indices[char_count];
                let prefix = &remaining[..byte_end];
                if let Some(&id) = self.token_to_id.get(prefix) {
                    best_len = byte_end;
                    best_id = Some(id);
                }
            }

            if let Some(id) = best_id {
                tokens.push(id);
                remaining = &remaining[best_len..];
            } else {
                // No match - try byte tokens like <0x48>
                let ch = remaining.chars().next().expect("non-empty");
                let ch_len = ch.len_utf8();

                for byte in remaining[..ch_len].bytes() {
                    let byte_token = format!("<0x{byte:02X}>");
                    if let Some(&id) = self.token_to_id.get(&byte_token) {
                        tokens.push(id);
                    } else {
                        tokens.push(self.unk_token_id);
                    }
                }
                remaining = &remaining[ch_len..];
            }
        }

        contract_post_encode!(&tokens);
        tokens
    }

    /// Decode token IDs to text
    ///
    /// # Arguments
    ///
    /// * `token_ids` - Token IDs to decode
    ///
    /// # Errors
    ///
    /// Returns error if any token ID is invalid
    pub fn decode(&self, token_ids: &[u32]) -> Result<String> {
        contract_pre_decode!();
        let mut bytes: Vec<u8> = Vec::new();

        for &id in token_ids {
            let token =
                self.id_to_token
                    .get(&id)
                    .ok_or_else(|| RealizarError::UnsupportedOperation {
                        operation: "decode_bpe_token".to_string(),
                        reason: format!("Invalid token ID: {id}"),
                    })?;

            // Skip special tokens
            if token.starts_with("<|") && token.ends_with("|>") {
                continue;
            }
            if token == "<s>" || token == "</s>" || token == "<unk>" || token == "<pad>" {
                continue;
            }

            // Handle byte tokens like <0xE6>
            if token.starts_with("<0x") && token.ends_with('>') && token.len() == 6 {
                if let Ok(byte_val) = u8::from_str_radix(
                    token
                        .get(3..5)
                        .expect("byte token <0xNN> has len 6, indices 3..5 always valid"),
                    16,
                ) {
                    bytes.push(byte_val);
                    continue;
                }
            }

            // Decode GPT-2 style byte-level BPE
            for c in token.chars() {
                match c {
                    'Ġ' => bytes.push(b' '),  // U+0120 -> space
                    'Ċ' => bytes.push(b'\n'), // U+010A -> newline
                    'ċ' => bytes.push(b'\n'), // lowercase variant
                    'Ḃ' => bytes.push(b'\r'), // U+1E02 -> carriage return
                    '▁' => bytes.push(b' '),  // U+2581 SentencePiece -> space
                    _ => {
                        // Try GPT-2 unicode-to-byte mapping
                        if let Some(byte) = Self::gpt2_char_to_byte(c) {
                            bytes.push(byte);
                        } else {
                            // Regular UTF-8 character
                            let mut buf = [0u8; 4];
                            let encoded = c.encode_utf8(&mut buf);
                            bytes.extend_from_slice(encoded.as_bytes());
                        }
                    },
                }
            }
        }

        // Decode as UTF-8, replacing invalid sequences
        let result = String::from_utf8_lossy(&bytes).into_owned();
        contract_post_decode!(&result);
        Ok(result)
    }

    /// Convert GPT-2 unicode character to original byte value
    fn gpt2_char_to_byte(c: char) -> Option<u8> {
        // GPT-2 maps bytes 0-255 to unicode characters
        // Printable ASCII (33-126) maps to itself
        // Other bytes map to unicode range starting at U+0100
        let code = c as u32;
        if (33..=126).contains(&code) || code == 32 {
            Some(code as u8)
        } else if (0x100..=0x100 + 255).contains(&code) {
            // GPT-2 remapped bytes
            let byte = (code - 0x100) as u8;
            // Map back based on GPT-2's byte_encoder
            match byte {
                0..=32 => Some(byte),    // Control chars + space
                127..=160 => Some(byte), // DEL + extended ASCII
                173 => Some(173),        // Soft hyphen
                _ => None,
            }
        } else {
            None
        }
    }

    /// Get vocabulary size (cached, O(1))
    ///
    /// PMAT-805: Returns pre-computed vocab size for fast access.
    /// Avoids repeated HashMap::len() calls during inference.
    #[inline]
    #[must_use]
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Get token ID for a token
    #[must_use]
    pub fn get_token_id(&self, token: &str) -> Option<u32> {
        self.token_to_id.get(token).copied()
    }

    /// Get token for a token ID
    #[must_use]
    pub fn get_token(&self, id: u32) -> Option<&str> {
        self.id_to_token.get(&id).map(String::as_str)
    }
}

/// Viterbi algorithm state: `(best_score, best_token)` at each position
type ViterbiState = (Vec<f32>, Vec<Option<String>>);

/// `SentencePiece` tokenizer (Unigram model)
///
/// Implements subword tokenization using unigram language model.
/// Used by `LLaMA`, T5, ALBERT, and many other models.
///
/// Unlike BPE which uses greedy merges, `SentencePiece` finds the
/// most likely segmentation using token scores (log probabilities).
#[derive(Debug, Clone)]
pub struct SentencePieceTokenizer {
    /// Token to ID mapping
    token_to_id: HashMap<String, u32>,
    /// ID to token mapping
    id_to_token: HashMap<u32, String>,
    /// Token scores (log probabilities)
    scores: HashMap<String, f32>,
    /// Unknown token ID
    unk_token_id: u32,
}

include!("tokenizer_sentence_piece.rs");
include!("tokenizer_sentencepiece_encode.rs");
include!("tokenizer_04.rs");
include!("tokenizer_contract_tests.rs");
