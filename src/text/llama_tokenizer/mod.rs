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

include!("gguf_value.rs");
include!("mod_part_03.rs");
