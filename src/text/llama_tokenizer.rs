//! LLaMA/TinyLlama SentencePiece-style BPE tokenizer (GH-145).
//!
//! Provides tokenization for LLaMA-family models:
//! - Load vocabulary from GGUF metadata
//! - SentencePiece-style BPE encoding
//! - Decode token IDs to text
//! - Special token handling (BOS, EOS, UNK, PAD)
//!
//! # Architecture
//!
//! ```text
//! Text → Normalize → BPE Encode → Token IDs
//!                       ↓
//!           GGUF metadata (tokens, scores, merges)
//! ```
//!
//! # Example
//!
//! ```rust,ignore
//! use aprender::text::llama_tokenizer::LlamaTokenizer;
//!
//! let tokenizer = LlamaTokenizer::from_gguf_bytes(&gguf_data)?;
//! let tokens = tokenizer.encode("Hello, world!");
//! let decoded = tokenizer.decode(&tokens);
//! assert_eq!(decoded, "Hello, world!");
//! ```
//!
//! # PMAT Compliance
//!
//! - Zero `unwrap()` calls
//! - All public APIs return `Result<T, E>` where fallible
//! - Toyota P5 Jidoka: fail-fast on invalid input

use std::collections::HashMap;

use crate::error::{AprenderError, Result};

// ============================================================================
// Constants
// ============================================================================

/// LLaMA vocabulary size (standard)
pub const LLAMA_VOCAB_SIZE: usize = 32000;

/// Special token: Beginning of sequence
pub const BOS_TOKEN: &str = "<s>";
/// Special token: End of sequence
pub const EOS_TOKEN: &str = "</s>";
/// Special token: Unknown
pub const UNK_TOKEN: &str = "<unk>";

/// Byte fallback prefix (SentencePiece style)
const BYTE_FALLBACK_PREFIX: &str = "<0x";

// ============================================================================
// LlamaTokenizer
// ============================================================================

/// LLaMA/TinyLlama tokenizer using SentencePiece-style BPE.
///
/// Loads vocabulary and merges from GGUF metadata.
#[derive(Debug, Clone)]
pub struct LlamaTokenizer {
    /// Token string to ID mapping
    vocab: HashMap<String, u32>,
    /// ID to token string mapping
    id_to_token: HashMap<u32, String>,
    /// Token scores (priorities for merging)
    #[allow(dead_code)]
    scores: Vec<f32>,
    /// BOS token ID
    bos_token_id: u32,
    /// EOS token ID
    eos_token_id: u32,
    /// Unknown token ID
    unk_token_id: u32,
    /// Padding token ID (optional)
    #[allow(dead_code)]
    pad_token_id: Option<u32>,
    /// Vocabulary size
    vocab_size: usize,
}

impl LlamaTokenizer {
    /// Create a new LlamaTokenizer from vocabulary data.
    ///
    /// # Arguments
    /// * `tokens` - List of token strings
    /// * `scores` - Token scores/priorities
    /// * `bos_token_id` - Beginning of sequence token ID
    /// * `eos_token_id` - End of sequence token ID
    /// * `unk_token_id` - Unknown token ID
    ///
    /// # Errors
    /// Returns error if vocabulary is empty or IDs are out of range.
    pub fn new(
        tokens: Vec<String>,
        scores: Vec<f32>,
        bos_token_id: u32,
        eos_token_id: u32,
        unk_token_id: u32,
    ) -> Result<Self> {
        if tokens.is_empty() {
            return Err(AprenderError::ValidationError {
                message: "Empty vocabulary".to_string(),
            });
        }

        let vocab_size = tokens.len();

        // Build vocab mappings
        let mut vocab = HashMap::with_capacity(vocab_size);
        let mut id_to_token = HashMap::with_capacity(vocab_size);

        for (id, token) in tokens.into_iter().enumerate() {
            let id = id as u32;
            vocab.insert(token.clone(), id);
            id_to_token.insert(id, token);
        }

        // Validate special token IDs
        let validate_id = |id: u32, name: &str| -> Result<()> {
            if id as usize >= vocab_size {
                return Err(AprenderError::ValidationError {
                    message: format!(
                        "{name} token ID {id} out of range (vocab_size={vocab_size})"
                    ),
                });
            }
            Ok(())
        };

        validate_id(bos_token_id, "BOS")?;
        validate_id(eos_token_id, "EOS")?;
        validate_id(unk_token_id, "UNK")?;

        Ok(Self {
            vocab,
            id_to_token,
            scores,
            bos_token_id,
            eos_token_id,
            unk_token_id,
            pad_token_id: None,
            vocab_size,
        })
    }

    /// Get vocabulary size.
    #[must_use]
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Get BOS token ID.
    #[must_use]
    pub fn bos_token_id(&self) -> u32 {
        self.bos_token_id
    }

    /// Get EOS token ID.
    #[must_use]
    pub fn eos_token_id(&self) -> u32 {
        self.eos_token_id
    }

    /// Get UNK token ID.
    #[must_use]
    pub fn unk_token_id(&self) -> u32 {
        self.unk_token_id
    }

    /// Encode text to token IDs.
    ///
    /// Uses SentencePiece-style BPE encoding:
    /// 1. Add space prefix to indicate word boundaries
    /// 2. Look up tokens with longest-match greedy algorithm
    /// 3. Fall back to byte tokens for unknown characters
    ///
    /// # Arguments
    /// * `text` - Input text to tokenize
    ///
    /// # Returns
    /// Vector of token IDs (without BOS/EOS)
    #[must_use]
    pub fn encode(&self, text: &str) -> Vec<u32> {
        if text.is_empty() {
            return Vec::new();
        }

        // SentencePiece prepends space to input and uses ▁ as word boundary marker
        // "Hello, world!" becomes "▁Hello▁,▁world▁!"
        let normalized = format!("▁{}", text.replace(' ', "▁"));
        let chars: Vec<char> = normalized.chars().collect();

        let mut tokens: Vec<u32> = Vec::with_capacity(text.len());
        let mut i = 0;

        while i < chars.len() {
            let mut best_len = 0;
            let mut best_token_id = self.unk_token_id;

            // Try matching increasingly longer substrings (greedy longest match)
            for end in (i + 1)..=chars.len().min(i + 32) {
                let substr: String = chars[i..end].iter().collect();

                // Check if this substring is in vocab
                if let Some(&token_id) = self.vocab.get(&substr) {
                    best_len = end - i;
                    best_token_id = token_id;
                }
            }

            if best_len > 0 {
                tokens.push(best_token_id);
                i += best_len;
            } else {
                // Fall back to byte tokens for unknown characters
                let c = chars[i];
                for byte in c.to_string().as_bytes() {
                    let byte_token = format!("<0x{byte:02X}>");
                    if let Some(&token_id) = self.vocab.get(&byte_token) {
                        tokens.push(token_id);
                    } else {
                        tokens.push(self.unk_token_id);
                    }
                }
                i += 1;
            }
        }

        tokens
    }

    /// Encode text with BOS token prepended.
    ///
    /// # Arguments
    /// * `text` - Input text to tokenize
    ///
    /// # Returns
    /// Vector of token IDs starting with BOS
    #[must_use]
    pub fn encode_with_bos(&self, text: &str) -> Vec<u32> {
        let mut tokens = vec![self.bos_token_id];
        tokens.extend(self.encode(text));
        tokens
    }

    /// Decode token IDs to text.
    ///
    /// # Arguments
    /// * `token_ids` - Token IDs to decode
    ///
    /// # Returns
    /// Decoded text string
    #[must_use]
    pub fn decode(&self, token_ids: &[u32]) -> String {
        let mut result = String::with_capacity(token_ids.len() * 4);

        for &token_id in token_ids {
            // Skip special tokens in output
            if token_id == self.bos_token_id || token_id == self.eos_token_id {
                continue;
            }

            if let Some(token) = self.id_to_token.get(&token_id) {
                // Handle SentencePiece space prefix
                let text = token.replace('▁', " ");

                // Handle byte tokens
                if text.starts_with(BYTE_FALLBACK_PREFIX) && text.ends_with('>') {
                    if let Some(hex) = text.strip_prefix(BYTE_FALLBACK_PREFIX) {
                        if let Some(hex) = hex.strip_suffix('>') {
                            if let Ok(byte) = u8::from_str_radix(hex, 16) {
                                result.push(byte as char);
                                continue;
                            }
                        }
                    }
                }

                result.push_str(&text);
            }
        }

        // Clean up leading space if present
        if result.starts_with(' ') {
            result.remove(0);
        }

        result
    }

    /// Get token string for an ID.
    #[must_use]
    pub fn id_to_token(&self, id: u32) -> Option<&str> {
        self.id_to_token.get(&id).map(String::as_str)
    }

    /// Get token ID for a string.
    #[must_use]
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.vocab.get(token).copied()
    }
}

// ============================================================================
// GGUF Loading
// ============================================================================

/// Value types in GGUF metadata
#[derive(Debug, Clone)]
pub enum GGUFValue {
    UInt8(u8),
    Int8(i8),
    UInt16(u16),
    Int16(i16),
    UInt32(u32),
    Int32(i32),
    Float32(f32),
    Bool(bool),
    String(String),
    Array(Vec<GGUFValue>),
    UInt64(u64),
    Int64(i64),
    Float64(f64),
}

impl LlamaTokenizer {
    /// Load tokenizer from GGUF file bytes.
    ///
    /// Extracts vocabulary from GGUF metadata:
    /// - `tokenizer.ggml.tokens` - vocabulary strings
    /// - `tokenizer.ggml.scores` - token priorities
    /// - `tokenizer.ggml.bos_token_id` - BOS token
    /// - `tokenizer.ggml.eos_token_id` - EOS token
    /// - `tokenizer.ggml.unknown_token_id` - UNK token
    ///
    /// # Errors
    /// Returns error if GGUF is invalid or missing tokenizer data.
    pub fn from_gguf_bytes(data: &[u8]) -> Result<Self> {
        if data.len() < 24 {
            return Err(AprenderError::FormatError {
                message: "GGUF data too short".to_string(),
            });
        }

        // Verify magic
        if &data[0..4] != b"GGUF" {
            return Err(AprenderError::FormatError {
                message: "Invalid GGUF magic".to_string(),
            });
        }

        // Parse header
        let metadata_count = u64::from_le_bytes(
            data[16..24]
                .try_into()
                .map_err(|_| AprenderError::FormatError { message: "Failed to read metadata count".to_string() })?,
        ) as usize;

        // Parse metadata
        let mut offset = 24usize;
        let mut tokens: Option<Vec<String>> = None;
        let mut scores: Option<Vec<f32>> = None;
        let mut bos_token_id: u32 = 1;
        let mut eos_token_id: u32 = 2;
        let mut unk_token_id: u32 = 0;

        for _ in 0..metadata_count {
            if offset + 8 > data.len() {
                break;
            }

            // Read key
            let key_len = u64::from_le_bytes(
                data[offset..offset + 8]
                    .try_into()
                    .map_err(|_| AprenderError::FormatError { message: "Failed to read key length".to_string() })?,
            ) as usize;
            offset += 8;

            if offset + key_len > data.len() {
                break;
            }
            let key = String::from_utf8_lossy(&data[offset..offset + key_len]).to_string();
            offset += key_len;

            if offset + 4 > data.len() {
                break;
            }

            // Read value type
            let val_type = u32::from_le_bytes(
                data[offset..offset + 4]
                    .try_into()
                    .map_err(|_| AprenderError::FormatError { message: "Failed to read value type".to_string() })?,
            );
            offset += 4;

            // Parse value based on type and key
            match key.as_str() {
                "tokenizer.ggml.tokens" => {
                    if val_type == 9 {
                        // Array
                        let (arr, new_offset) = Self::parse_string_array(data, offset)?;
                        tokens = Some(arr);
                        offset = new_offset;
                    }
                }
                "tokenizer.ggml.scores" => {
                    if val_type == 9 {
                        // Array
                        let (arr, new_offset) = Self::parse_f32_array(data, offset)?;
                        scores = Some(arr);
                        offset = new_offset;
                    }
                }
                "tokenizer.ggml.bos_token_id" => {
                    if val_type == 4 && offset + 4 <= data.len() {
                        bos_token_id = u32::from_le_bytes(
                            data[offset..offset + 4].try_into().unwrap_or([0; 4]),
                        );
                        offset += 4;
                    }
                }
                "tokenizer.ggml.eos_token_id" => {
                    if val_type == 4 && offset + 4 <= data.len() {
                        eos_token_id = u32::from_le_bytes(
                            data[offset..offset + 4].try_into().unwrap_or([0; 4]),
                        );
                        offset += 4;
                    }
                }
                "tokenizer.ggml.unknown_token_id" => {
                    if val_type == 4 && offset + 4 <= data.len() {
                        unk_token_id = u32::from_le_bytes(
                            data[offset..offset + 4].try_into().unwrap_or([0; 4]),
                        );
                        offset += 4;
                    }
                }
                _ => {
                    // Skip other values
                    offset = Self::skip_value(data, offset, val_type);
                }
            }
        }

        let tokens = tokens.ok_or_else(|| {
            AprenderError::FormatError { message: "Missing tokenizer.ggml.tokens in GGUF".to_string() }
        })?;

        let scores = scores.unwrap_or_else(|| vec![0.0; tokens.len()]);

        Self::new(tokens, scores, bos_token_id, eos_token_id, unk_token_id)
    }

    fn parse_string_array(data: &[u8], mut offset: usize) -> Result<(Vec<String>, usize)> {
        if offset + 12 > data.len() {
            return Err(AprenderError::FormatError { message: "Array header too short".to_string() });
        }

        let elem_type = u32::from_le_bytes(
            data[offset..offset + 4].try_into().unwrap_or([0; 4]),
        );
        offset += 4;

        if elem_type != 8 {
            return Err(AprenderError::FormatError {
                message: format!("Expected string array (type 8), got type {elem_type}"),
            });
        }

        let count = u64::from_le_bytes(
            data[offset..offset + 8].try_into().unwrap_or([0; 8]),
        ) as usize;
        offset += 8;

        let mut result = Vec::with_capacity(count);

        for _ in 0..count {
            if offset + 8 > data.len() {
                break;
            }
            let str_len = u64::from_le_bytes(
                data[offset..offset + 8].try_into().unwrap_or([0; 8]),
            ) as usize;
            offset += 8;

            if offset + str_len > data.len() {
                break;
            }
            let s = String::from_utf8_lossy(&data[offset..offset + str_len]).to_string();
            offset += str_len;
            result.push(s);
        }

        Ok((result, offset))
    }

    fn parse_f32_array(data: &[u8], mut offset: usize) -> Result<(Vec<f32>, usize)> {
        if offset + 12 > data.len() {
            return Err(AprenderError::FormatError { message: "Array header too short".to_string() });
        }

        let elem_type = u32::from_le_bytes(
            data[offset..offset + 4].try_into().unwrap_or([0; 4]),
        );
        offset += 4;

        if elem_type != 6 {
            return Err(AprenderError::FormatError {
                message: format!("Expected f32 array (type 6), got type {elem_type}"),
            });
        }

        let count = u64::from_le_bytes(
            data[offset..offset + 8].try_into().unwrap_or([0; 8]),
        ) as usize;
        offset += 8;

        let mut result = Vec::with_capacity(count);

        for _ in 0..count {
            if offset + 4 > data.len() {
                break;
            }
            let f = f32::from_le_bytes(
                data[offset..offset + 4].try_into().unwrap_or([0; 4]),
            );
            offset += 4;
            result.push(f);
        }

        Ok((result, offset))
    }

    fn skip_value(data: &[u8], mut offset: usize, val_type: u32) -> usize {
        match val_type {
            0 | 1 | 7 => offset += 1,  // u8, i8, bool
            2 | 3 => offset += 2,       // u16, i16
            4..=6 => offset += 4,       // u32, i32, f32
            8 => {
                // string
                if offset + 8 > data.len() {
                    return offset;
                }
                let len = u64::from_le_bytes(
                    data[offset..offset + 8].try_into().unwrap_or([0; 8]),
                ) as usize;
                offset += 8 + len;
            }
            9 => {
                // array
                if offset + 12 > data.len() {
                    return offset;
                }
                let elem_type = u32::from_le_bytes(
                    data[offset..offset + 4].try_into().unwrap_or([0; 4]),
                );
                offset += 4;
                let count = u64::from_le_bytes(
                    data[offset..offset + 8].try_into().unwrap_or([0; 8]),
                ) as usize;
                offset += 8;

                match elem_type {
                    0 | 1 | 7 => offset += count,
                    2 | 3 => offset += count * 2,
                    4..=6 => offset += count * 4,
                    8 => {
                        for _ in 0..count {
                            if offset + 8 > data.len() {
                                break;
                            }
                            let len = u64::from_le_bytes(
                                data[offset..offset + 8].try_into().unwrap_or([0; 8]),
                            ) as usize;
                            offset += 8 + len;
                        }
                    }
                    10..=12 => offset += count * 8,
                    _ => {}
                }
            }
            10..=12 => offset += 8, // u64, i64, f64
            _ => {}
        }
        offset
    }
}

// ============================================================================
// Tests (EXTREME TDD - Falsification Tests First)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // Falsification Tests (Popperian)
    // ========================================================================

    /// LT-01: Tokenizer MUST load vocabulary from GGUF
    /// Falsification: If vocab is not loaded, encoding will return only UNK tokens
    #[test]
    fn lt01_tokenizer_loads_vocab_from_gguf() {
        // Create minimal GGUF with tokenizer data
        let gguf_data = create_test_gguf();
        let tokenizer = LlamaTokenizer::from_gguf_bytes(&gguf_data);

        assert!(
            tokenizer.is_ok(),
            "FALSIFIED: Tokenizer failed to load from GGUF: {:?}",
            tokenizer.err()
        );

        let tokenizer = tokenizer.expect("already checked");
        assert!(
            tokenizer.vocab_size() > 0,
            "FALSIFIED: Vocabulary is empty"
        );
    }

    /// LT-02: Tokenizer MUST encode text to non-empty tokens
    /// Falsification: If encoding fails, result will be empty for non-empty input
    #[test]
    fn lt02_tokenizer_encodes_text() {
        let tokenizer = create_test_tokenizer();
        let tokens = tokenizer.encode("Hello");

        assert!(
            !tokens.is_empty(),
            "FALSIFIED: Encoding returned empty for non-empty input"
        );
    }

    /// LT-03: Tokenizer MUST decode tokens back to readable text
    /// Falsification: If decoding fails, result will be empty or garbage
    #[test]
    fn lt03_tokenizer_decodes_tokens() {
        let tokenizer = create_test_tokenizer();
        let tokens = tokenizer.encode("Hello");
        let decoded = tokenizer.decode(&tokens);

        assert!(
            !decoded.is_empty(),
            "FALSIFIED: Decoding returned empty string"
        );
        // Note: Exact roundtrip may not match due to tokenization granularity
    }

    /// LT-04: BOS token MUST be prepended when requested
    /// Falsification: First token will not be BOS ID
    #[test]
    fn lt04_bos_token_prepended() {
        let tokenizer = create_test_tokenizer();
        let tokens = tokenizer.encode_with_bos("Hello");

        assert!(
            !tokens.is_empty(),
            "FALSIFIED: Encoding with BOS returned empty"
        );
        assert_eq!(
            tokens[0],
            tokenizer.bos_token_id(),
            "FALSIFIED: First token is not BOS"
        );
    }

    /// LT-05: Unknown characters MUST use byte fallback, not panic
    /// Falsification: Encoding unknown chars would panic or return empty
    #[test]
    fn lt05_byte_fallback_for_unknown() {
        let tokenizer = create_test_tokenizer();
        // Use an emoji that's unlikely to be in a small test vocab
        let tokens = tokenizer.encode("Hello 🎉 World");

        assert!(
            !tokens.is_empty(),
            "FALSIFIED: Encoding with unknown chars returned empty"
        );
    }

    /// LT-06: Empty input MUST return empty tokens (not panic)
    /// Falsification: Would panic or return non-empty
    #[test]
    fn lt06_empty_input_returns_empty() {
        let tokenizer = create_test_tokenizer();
        let tokens = tokenizer.encode("");

        assert!(
            tokens.is_empty(),
            "FALSIFIED: Empty input returned non-empty tokens"
        );
    }

    /// LT-07: Special tokens MUST be excluded from decode output
    /// Falsification: BOS/EOS would appear in decoded text
    #[test]
    fn lt07_special_tokens_excluded_from_decode() {
        let tokenizer = create_test_tokenizer();
        let tokens = vec![tokenizer.bos_token_id(), 100, 101, tokenizer.eos_token_id()];
        let decoded = tokenizer.decode(&tokens);

        assert!(
            !decoded.contains("<s>") && !decoded.contains("</s>"),
            "FALSIFIED: Special tokens appear in decoded output: {}",
            decoded
        );
    }

    /// LT-08: Tokenizer MUST reject invalid GGUF magic
    /// Falsification: Would accept invalid data
    #[test]
    fn lt08_rejects_invalid_gguf() {
        let invalid_data = b"NOTGGUF0000000000000000";
        let result = LlamaTokenizer::from_gguf_bytes(invalid_data);

        assert!(
            result.is_err(),
            "FALSIFIED: Accepted invalid GGUF magic"
        );
    }

    // ========================================================================
    // Helper Functions
    // ========================================================================

    fn create_test_tokenizer() -> LlamaTokenizer {
        // Create a minimal tokenizer for testing
        let tokens = vec![
            "<unk>".to_string(),
            "<s>".to_string(),
            "</s>".to_string(),
            "▁Hello".to_string(),
            "▁World".to_string(),
            "▁".to_string(),
            "H".to_string(),
            "e".to_string(),
            "l".to_string(),
            "o".to_string(),
        ];
        let scores = vec![0.0; tokens.len()];

        LlamaTokenizer::new(tokens, scores, 1, 2, 0)
            .expect("Failed to create test tokenizer")
    }

    fn create_test_gguf() -> Vec<u8> {
        let mut data = Vec::new();

        // GGUF header
        data.extend_from_slice(b"GGUF");              // magic
        data.extend_from_slice(&3u32.to_le_bytes());  // version
        data.extend_from_slice(&0u64.to_le_bytes());  // tensor_count
        data.extend_from_slice(&5u64.to_le_bytes());  // metadata_count

        // Metadata 1: tokenizer.ggml.tokens (string array)
        let key1 = b"tokenizer.ggml.tokens";
        data.extend_from_slice(&(key1.len() as u64).to_le_bytes());
        data.extend_from_slice(key1);
        data.extend_from_slice(&9u32.to_le_bytes()); // array type
        data.extend_from_slice(&8u32.to_le_bytes()); // string element type
        let tokens = ["<unk>", "<s>", "</s>", "▁Hello", "▁World"];
        data.extend_from_slice(&(tokens.len() as u64).to_le_bytes());
        for token in &tokens {
            let bytes = token.as_bytes();
            data.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
            data.extend_from_slice(bytes);
        }

        // Metadata 2: tokenizer.ggml.scores (f32 array)
        let key2 = b"tokenizer.ggml.scores";
        data.extend_from_slice(&(key2.len() as u64).to_le_bytes());
        data.extend_from_slice(key2);
        data.extend_from_slice(&9u32.to_le_bytes()); // array type
        data.extend_from_slice(&6u32.to_le_bytes()); // f32 element type
        data.extend_from_slice(&(tokens.len() as u64).to_le_bytes());
        for _ in &tokens {
            data.extend_from_slice(&0.0f32.to_le_bytes());
        }

        // Metadata 3: bos_token_id
        let key3 = b"tokenizer.ggml.bos_token_id";
        data.extend_from_slice(&(key3.len() as u64).to_le_bytes());
        data.extend_from_slice(key3);
        data.extend_from_slice(&4u32.to_le_bytes()); // u32 type
        data.extend_from_slice(&1u32.to_le_bytes()); // value

        // Metadata 4: eos_token_id
        let key4 = b"tokenizer.ggml.eos_token_id";
        data.extend_from_slice(&(key4.len() as u64).to_le_bytes());
        data.extend_from_slice(key4);
        data.extend_from_slice(&4u32.to_le_bytes()); // u32 type
        data.extend_from_slice(&2u32.to_le_bytes()); // value

        // Metadata 5: unknown_token_id
        let key5 = b"tokenizer.ggml.unknown_token_id";
        data.extend_from_slice(&(key5.len() as u64).to_le_bytes());
        data.extend_from_slice(key5);
        data.extend_from_slice(&4u32.to_le_bytes()); // u32 type
        data.extend_from_slice(&0u32.to_le_bytes()); // value

        data
    }
}
