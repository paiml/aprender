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

/// `LLaMA` vocabulary size (standard)
pub const LLAMA_VOCAB_SIZE: usize = 32000;

/// Special token: Beginning of sequence
pub const BOS_TOKEN: &str = "<s>";
/// Special token: End of sequence
pub const EOS_TOKEN: &str = "</s>";
/// Special token: Unknown
pub const UNK_TOKEN: &str = "<unk>";

/// Byte fallback prefix (`SentencePiece` style)
const BYTE_FALLBACK_PREFIX: &str = "<0x";

// ============================================================================
// GPT-2 BPE Byte Decoding
// ============================================================================

/// Build the GPT-2 byte-to-unicode mapping.
///
/// GPT-2 BPE encodes bytes as unicode characters to ensure all byte sequences
/// are valid unicode. This maps the special unicode chars back to bytes.
///
/// The mapping is:
/// - Printable ASCII without space (33-126): Maps to same codepoint
/// - All other bytes (0-32, 127-255): Maps to U+0100..U+01FF range
fn build_gpt2_byte_decoder() -> HashMap<char, u8> {
    let mut decoder = HashMap::with_capacity(256);

    // Printable ASCII (33-126) and some extras (161-172, 174-255) map directly
    // Per GPT-2 encoder.py bytes_to_unicode()
    let direct_bytes: Vec<u8> = (b'!'..=b'~')
        .chain(0xA1u8..=0xACu8)
        .chain(0xAEu8..=0xFFu8)
        .collect();

    let mut n = 0u32;
    for b in 0u8..=255u8 {
        if direct_bytes.contains(&b) {
            decoder.insert(char::from(b), b);
        } else {
            // Non-printable bytes map to 256 + n
            if let Some(c) = char::from_u32(256 + n) {
                decoder.insert(c, b);
            }
            n += 1;
        }
    }

    decoder
}

/// Decode a GPT-2 BPE token string to actual bytes.
///
/// GPT-2 BPE uses unicode characters to represent bytes. This function
/// converts the unicode representation back to the original bytes.
fn decode_gpt2_token(token: &str) -> Vec<u8> {
    static GPT2_DECODER: std::sync::LazyLock<HashMap<char, u8>> =
        std::sync::LazyLock::new(build_gpt2_byte_decoder);

    let mut bytes = Vec::with_capacity(token.len());
    for c in token.chars() {
        if let Some(&b) = GPT2_DECODER.get(&c) {
            bytes.push(b);
        } else {
            // Character not in GPT-2 mapping, encode as UTF-8
            let mut buf = [0u8; 4];
            let s = c.encode_utf8(&mut buf);
            bytes.extend_from_slice(s.as_bytes());
        }
    }
    bytes
}

// ============================================================================
// LlamaTokenizer
// ============================================================================

/// Tokenizer model type (from GGUF metadata)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TokenizerModel {
    /// SentencePiece-style BPE (LLaMA, Mistral)
    #[default]
    SentencePiece,
    /// GPT-2 BPE (Qwen, GPT-2, etc.)
    Gpt2,
}

/// LLaMA/TinyLlama tokenizer using SentencePiece-style BPE.
///
/// Also supports GPT-2 BPE for models like Qwen2.5.
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
    /// Tokenizer model type (SentencePiece or GPT-2)
    model: TokenizerModel,
}

impl LlamaTokenizer {
    /// Create a new `LlamaTokenizer` from vocabulary data.
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
                    message: format!("{name} token ID {id} out of range (vocab_size={vocab_size})"),
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
            model: TokenizerModel::SentencePiece, // Default, updated by GGUF loader
        })
    }

    /// Set the tokenizer model type.
    pub fn set_model(&mut self, model: TokenizerModel) {
        self.model = model;
    }

    /// Get the tokenizer model type.
    #[must_use]
    pub fn model(&self) -> TokenizerModel {
        self.model
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
    /// Handles both SentencePiece-style (LLaMA) and GPT-2 BPE (Qwen) tokenizers.
    ///
    /// # Arguments
    /// * `token_ids` - Token IDs to decode
    ///
    /// # Returns
    /// Decoded text string
    #[must_use]
    pub fn decode(&self, token_ids: &[u32]) -> String {
        match self.model {
            TokenizerModel::Gpt2 => self.decode_gpt2(token_ids),
            TokenizerModel::SentencePiece => self.decode_sentencepiece(token_ids),
        }
    }

    /// Decode using GPT-2 byte-level BPE (Qwen, GPT-2, etc.)
    fn decode_gpt2(&self, token_ids: &[u32]) -> String {
        let mut bytes = Vec::with_capacity(token_ids.len() * 4);

        for &token_id in token_ids {
            // Skip special tokens in output
            if token_id == self.bos_token_id || token_id == self.eos_token_id {
                continue;
            }

            if let Some(token) = self.id_to_token.get(&token_id) {
                // GPT-2 BPE: decode unicode chars to original bytes
                bytes.extend(decode_gpt2_token(token));
            }
        }

        // Convert bytes to UTF-8 string, replacing invalid sequences
        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// Decode using SentencePiece-style BPE (LLaMA, Mistral, etc.)
    fn decode_sentencepiece(&self, token_ids: &[u32]) -> String {
        let mut result = String::with_capacity(token_ids.len() * 4);

        for &token_id in token_ids {
            // Skip special tokens in output
            if token_id == self.bos_token_id || token_id == self.eos_token_id {
                continue;
            }

            if let Some(token) = self.id_to_token.get(&token_id) {
                // Handle SentencePiece space prefix (U+2581)
                let text = token.replace('▁', " ");
                // Handle GPT-2 BPE space prefix (U+0120 'Ġ') for hybrid tokenizers
                let text = text.replace('Ġ', " ");

                // Handle byte tokens like <0x0A> for newlines
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

        // Clean up leading space if present (SentencePiece adds leading space)
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
        let metadata_count =
            u64::from_le_bytes(
                data[16..24]
                    .try_into()
                    .map_err(|_| AprenderError::FormatError {
                        message: "Failed to read metadata count".to_string(),
                    })?,
            ) as usize;

        // Parse metadata
        let mut offset = 24usize;
        let mut tokens: Option<Vec<String>> = None;
        let mut scores: Option<Vec<f32>> = None;
        let mut bos_token_id: u32 = 1;
        let mut eos_token_id: u32 = 2;
        let mut unk_token_id: u32 = 0;
        let mut tokenizer_model = TokenizerModel::SentencePiece; // Default

        for _ in 0..metadata_count {
            if offset + 8 > data.len() {
                break;
            }

            // Read key
            let key_len = u64::from_le_bytes(data[offset..offset + 8].try_into().map_err(|_| {
                AprenderError::FormatError {
                    message: "Failed to read key length".to_string(),
                }
            })?) as usize;
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
            let val_type =
                u32::from_le_bytes(data[offset..offset + 4].try_into().map_err(|_| {
                    AprenderError::FormatError {
                        message: "Failed to read value type".to_string(),
                    }
                })?);
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
                "tokenizer.ggml.model" => {
                    // Detect tokenizer type: "gpt2" for GPT-2 BPE, "llama" for SentencePiece
                    if val_type == 8 && offset + 8 <= data.len() {
                        let str_len = u64::from_le_bytes(
                            data[offset..offset + 8].try_into().unwrap_or([0; 8]),
                        ) as usize;
                        offset += 8;
                        if offset + str_len <= data.len() {
                            let model_str =
                                String::from_utf8_lossy(&data[offset..offset + str_len]);
                            if model_str == "gpt2" {
                                tokenizer_model = TokenizerModel::Gpt2;
                            }
                            offset += str_len;
                        }
                    }
                }
                _ => {
                    // Skip other values
                    offset = Self::skip_value(data, offset, val_type);
                }
            }
        }

        let tokens = tokens.ok_or_else(|| AprenderError::FormatError {
            message: "Missing tokenizer.ggml.tokens in GGUF".to_string(),
        })?;

        let scores = scores.unwrap_or_else(|| vec![0.0; tokens.len()]);

        let mut tokenizer = Self::new(tokens, scores, bos_token_id, eos_token_id, unk_token_id)?;
        tokenizer.set_model(tokenizer_model);
        Ok(tokenizer)
    }

    fn parse_string_array(data: &[u8], mut offset: usize) -> Result<(Vec<String>, usize)> {
        if offset + 12 > data.len() {
            return Err(AprenderError::FormatError {
                message: "Array header too short".to_string(),
            });
        }

        let elem_type = u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap_or([0; 4]));
        offset += 4;

        if elem_type != 8 {
            return Err(AprenderError::FormatError {
                message: format!("Expected string array (type 8), got type {elem_type}"),
            });
        }

        let count =
            u64::from_le_bytes(data[offset..offset + 8].try_into().unwrap_or([0; 8])) as usize;
        offset += 8;

        let mut result = Vec::with_capacity(count);

        for _ in 0..count {
            if offset + 8 > data.len() {
                break;
            }
            let str_len =
                u64::from_le_bytes(data[offset..offset + 8].try_into().unwrap_or([0; 8])) as usize;
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
            return Err(AprenderError::FormatError {
                message: "Array header too short".to_string(),
            });
        }

        let elem_type = u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap_or([0; 4]));
        offset += 4;

        if elem_type != 6 {
            return Err(AprenderError::FormatError {
                message: format!("Expected f32 array (type 6), got type {elem_type}"),
            });
        }

        let count =
            u64::from_le_bytes(data[offset..offset + 8].try_into().unwrap_or([0; 8])) as usize;
        offset += 8;

        let mut result = Vec::with_capacity(count);

        for _ in 0..count {
            if offset + 4 > data.len() {
                break;
            }
            let f = f32::from_le_bytes(data[offset..offset + 4].try_into().unwrap_or([0; 4]));
            offset += 4;
            result.push(f);
        }

        Ok((result, offset))
    }

    fn skip_value(data: &[u8], mut offset: usize, val_type: u32) -> usize {
        match val_type {
            0 | 1 | 7 => offset += 1, // u8, i8, bool
            2 | 3 => offset += 2,     // u16, i16
            4..=6 => offset += 4,     // u32, i32, f32
            8 => {
                // string
                if offset + 8 > data.len() {
                    return offset;
                }
                let len = u64::from_le_bytes(data[offset..offset + 8].try_into().unwrap_or([0; 8]))
                    as usize;
                offset += 8 + len;
            }
            9 => {
                // array
                if offset + 12 > data.len() {
                    return offset;
                }
                let elem_type =
                    u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap_or([0; 4]));
                offset += 4;
                let count =
                    u64::from_le_bytes(data[offset..offset + 8].try_into().unwrap_or([0; 8]))
                        as usize;
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
        assert!(tokenizer.vocab_size() > 0, "FALSIFIED: Vocabulary is empty");
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

        assert!(result.is_err(), "FALSIFIED: Accepted invalid GGUF magic");
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

        LlamaTokenizer::new(tokens, scores, 1, 2, 0).expect("Failed to create test tokenizer")
    }

    fn create_test_gguf() -> Vec<u8> {
        let mut data = Vec::new();

        // GGUF header
        data.extend_from_slice(b"GGUF"); // magic
        data.extend_from_slice(&3u32.to_le_bytes()); // version
        data.extend_from_slice(&0u64.to_le_bytes()); // tensor_count
        data.extend_from_slice(&5u64.to_le_bytes()); // metadata_count

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

    // ========================================================================
    // GPT-2 BPE Decoding Tests
    // ========================================================================

    /// GPT-01: GPT-2 byte decoder MUST correctly map unicode to bytes
    /// Falsification: If mapping is wrong, decoded text will be garbage
    #[test]
    fn gpt01_byte_decoder_maps_correctly() {
        let decoder = build_gpt2_byte_decoder();

        // Printable ASCII should map to itself
        assert_eq!(decoder.get(&'A'), Some(&b'A'));
        assert_eq!(decoder.get(&'z'), Some(&b'z'));
        assert_eq!(decoder.get(&'0'), Some(&b'0'));

        // GPT-2 space marker (Ġ = U+0120) should map to space (0x20)
        assert_eq!(decoder.get(&'Ġ'), Some(&b' '));

        // Newline marker (Ċ = U+010A) should map to newline (0x0A)
        assert_eq!(decoder.get(&'Ċ'), Some(&b'\n'));

        // Tab marker (ĉ = U+0109) should map to tab (0x09)
        assert_eq!(decoder.get(&'ĉ'), Some(&b'\t'));
    }

    /// GPT-02: GPT-2 token decoding MUST produce valid UTF-8
    /// Falsification: If decoding fails, String::from_utf8_lossy will show replacement chars
    #[test]
    fn gpt02_token_decoding_produces_utf8() {
        // "Hello" in GPT-2 BPE
        let token = "Hello";
        let bytes = decode_gpt2_token(token);
        assert_eq!(bytes, b"Hello");

        // " world" in GPT-2 BPE (Ġ prefix for space)
        let token_with_space = "Ġworld";
        let bytes = decode_gpt2_token(token_with_space);
        assert_eq!(bytes, b" world");
    }

    /// GPT-03: GPT-2 tokenizer MUST decode complete sentences correctly
    /// Falsification: Sentence will be garbled
    #[test]
    fn gpt03_gpt2_tokenizer_decodes_sentences() {
        // Create a GPT-2 tokenizer with some tokens
        let tokens = vec![
            "<unk>".to_string(),
            "<|endoftext|>".to_string(), // BOS/EOS for GPT-2
            "</s>".to_string(),
            "Hello".to_string(),
            "Ġworld".to_string(), // " world" in GPT-2
            "!".to_string(),
        ];
        let scores = vec![0.0; tokens.len()];

        let mut tokenizer =
            LlamaTokenizer::new(tokens, scores, 1, 2, 0).expect("Failed to create tokenizer");
        tokenizer.set_model(TokenizerModel::Gpt2);

        // Decode token IDs
        let token_ids = vec![3, 4, 5]; // "Hello", " world", "!"
        let decoded = tokenizer.decode(&token_ids);

        assert_eq!(decoded, "Hello world!");
    }

    /// GPT-04: Model type detection from GGUF MUST work
    /// Falsification: Would default to SentencePiece even for GPT-2 models
    #[test]
    fn gpt04_model_type_detection() {
        let gguf_data = create_gpt2_test_gguf();
        let tokenizer = LlamaTokenizer::from_gguf_bytes(&gguf_data);

        assert!(tokenizer.is_ok());
        let tokenizer = tokenizer.expect("already checked");
        assert_eq!(
            tokenizer.model(),
            TokenizerModel::Gpt2,
            "FALSIFIED: GPT-2 model type not detected"
        );
    }

    fn create_gpt2_test_gguf() -> Vec<u8> {
        let mut data = Vec::new();

        // GGUF header
        data.extend_from_slice(b"GGUF");
        data.extend_from_slice(&3u32.to_le_bytes()); // version
        data.extend_from_slice(&0u64.to_le_bytes()); // tensor_count
        data.extend_from_slice(&6u64.to_le_bytes()); // metadata_count (added one more)

        // Metadata 1: tokenizer.ggml.tokens
        let key1 = b"tokenizer.ggml.tokens";
        data.extend_from_slice(&(key1.len() as u64).to_le_bytes());
        data.extend_from_slice(key1);
        data.extend_from_slice(&9u32.to_le_bytes()); // array type
        data.extend_from_slice(&8u32.to_le_bytes()); // string element type
        let tokens = ["<unk>", "<|endoftext|>", "</s>", "Hello", "Ġworld"];
        data.extend_from_slice(&(tokens.len() as u64).to_le_bytes());
        for token in &tokens {
            let bytes = token.as_bytes();
            data.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
            data.extend_from_slice(bytes);
        }

        // Metadata 2: tokenizer.ggml.scores
        let key2 = b"tokenizer.ggml.scores";
        data.extend_from_slice(&(key2.len() as u64).to_le_bytes());
        data.extend_from_slice(key2);
        data.extend_from_slice(&9u32.to_le_bytes());
        data.extend_from_slice(&6u32.to_le_bytes());
        data.extend_from_slice(&(tokens.len() as u64).to_le_bytes());
        for _ in &tokens {
            data.extend_from_slice(&0.0f32.to_le_bytes());
        }

        // Metadata 3: bos_token_id
        let key3 = b"tokenizer.ggml.bos_token_id";
        data.extend_from_slice(&(key3.len() as u64).to_le_bytes());
        data.extend_from_slice(key3);
        data.extend_from_slice(&4u32.to_le_bytes());
        data.extend_from_slice(&1u32.to_le_bytes());

        // Metadata 4: eos_token_id
        let key4 = b"tokenizer.ggml.eos_token_id";
        data.extend_from_slice(&(key4.len() as u64).to_le_bytes());
        data.extend_from_slice(key4);
        data.extend_from_slice(&4u32.to_le_bytes());
        data.extend_from_slice(&2u32.to_le_bytes());

        // Metadata 5: unknown_token_id
        let key5 = b"tokenizer.ggml.unknown_token_id";
        data.extend_from_slice(&(key5.len() as u64).to_le_bytes());
        data.extend_from_slice(key5);
        data.extend_from_slice(&4u32.to_le_bytes());
        data.extend_from_slice(&0u32.to_le_bytes());

        // Metadata 6: tokenizer.ggml.model = "gpt2"
        let key6 = b"tokenizer.ggml.model";
        data.extend_from_slice(&(key6.len() as u64).to_le_bytes());
        data.extend_from_slice(key6);
        data.extend_from_slice(&8u32.to_le_bytes()); // string type
        let model_str = b"gpt2";
        data.extend_from_slice(&(model_str.len() as u64).to_le_bytes());
        data.extend_from_slice(model_str);

        data
    }
}
