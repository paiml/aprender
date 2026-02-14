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
    static GPT2_DECODER: std::sync::OnceLock<HashMap<char, u8>> = std::sync::OnceLock::new();

    let decoder = GPT2_DECODER.get_or_init(build_gpt2_byte_decoder);
    let mut bytes = Vec::with_capacity(token.len());
    for c in token.chars() {
        if let Some(&b) = decoder.get(&c) {
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

        // Normalize based on tokenizer model type:
        // - SentencePiece: Uses ▁ (U+2581) as word boundary marker
        //   "Hello, world!" becomes "▁Hello▁,▁world▁!"
        // - GPT-2: Uses Ġ (U+0120) as space prefix
        //   "Hello, world!" becomes "Hello,Ġworld!"
        let normalized = match self.model {
            TokenizerModel::SentencePiece => format!("▁{}", text.replace(' ', "▁")),
            TokenizerModel::Gpt2 => text.replace(' ', "\u{0120}").replace('\n', "\u{010A}"),
        };
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
    ///
    /// # BUG-TOK-001 FIX: Correct byte token handling for multibyte UTF-8
    ///
    /// Previous implementation used `byte as char` which is WRONG for bytes >= 128.
    /// For example, decoding `<0xE4><0xB8><0x96>` (UTF-8 for "世") would produce:
    ///   - 0xE4 as char = 'ä' (Latin Extended)
    ///   - 0xB8 as char = '¸' (cedilla)
    ///   - 0x96 as char = control char
    ///
    /// Instead of the correct output "世".
    ///
    /// The fix collects bytes and converts to UTF-8 at boundaries.
    fn decode_sentencepiece(&self, token_ids: &[u32]) -> String {
        let mut result = String::with_capacity(token_ids.len() * 4);
        let mut pending_bytes: Vec<u8> = Vec::new();

        for &token_id in token_ids {
            // Skip special tokens in output
            if token_id == self.bos_token_id || token_id == self.eos_token_id {
                continue;
            }

            if let Some(token) = self.id_to_token.get(&token_id) {
                // Handle byte tokens like <0x0A> for newlines
                if token.starts_with(BYTE_FALLBACK_PREFIX) && token.ends_with('>') {
                    if let Some(hex) = token.strip_prefix(BYTE_FALLBACK_PREFIX) {
                        if let Some(hex) = hex.strip_suffix('>') {
                            if let Ok(byte) = u8::from_str_radix(hex, 16) {
                                // BUG-TOK-001 FIX: Collect bytes instead of casting to char
                                pending_bytes.push(byte);
                                continue;
                            }
                        }
                    }
                }

                // Flush pending bytes as UTF-8 before adding regular token
                if !pending_bytes.is_empty() {
                    result.push_str(&String::from_utf8_lossy(&pending_bytes));
                    pending_bytes.clear();
                }

                // Handle SentencePiece space prefix (U+2581)
                let text = token.replace('▁', " ");
                // Handle GPT-2 BPE space prefix (U+0120 'Ġ') for hybrid tokenizers
                let text = text.replace('Ġ', " ");

                result.push_str(&text);
            }
        }

        // Flush any remaining pending bytes
        if !pending_bytes.is_empty() {
            result.push_str(&String::from_utf8_lossy(&pending_bytes));
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
        if data.get(0..4) != Some(b"GGUF".as_slice()) {
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
mod tests;
