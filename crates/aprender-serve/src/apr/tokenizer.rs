//! BPE Tokenizer for APR models (PMAT-802)
//!
//! Byte Pair Encoding tokenizer supporting APR v2 format models.
//!
//! ## Tokenizer Types
//!
//! - `BpeTokenizer`: Full tokenizer with encode/decode, requires tokenizer.json
//! - `SimpleTokenizer`: Decode-only tokenizer using embedded vocabulary (GH-156)
//!
//! For APR models, prefer `SimpleTokenizer` via `AprV2Model::load_embedded_tokenizer()`
//! as it uses the vocabulary embedded in the .apr file - no sibling files needed.

use super::AprV2Model;
use std::collections::HashMap;

// ============================================================================
// SimpleTokenizer (GH-156): Decode-only tokenizer from embedded APR vocabulary
// ============================================================================

/// Simple decode-only tokenizer for APR models with embedded vocabulary.
///
/// Unlike `BpeTokenizer`, this doesn't require tokenizer.json - it uses
/// the vocabulary embedded directly in the APR file's metadata section.
///
/// # Example
///
/// ```rust,ignore
/// let model = AprV2Model::load("model.apr")?;
/// if let Some(tokenizer) = model.load_embedded_tokenizer() {
///     let text = tokenizer.decode(&[1, 2, 3]);
/// }
/// ```
#[derive(Debug, Clone)]
pub struct SimpleTokenizer {
    /// Vocabulary: index = token ID, value = token string
    pub id_to_token: Vec<String>,
    /// Beginning-of-sequence token ID (optional)
    pub bos_token_id: Option<u32>,
    /// End-of-sequence token ID (optional)
    pub eos_token_id: Option<u32>,
}

impl SimpleTokenizer {
    /// Create a new simple tokenizer from vocabulary
    #[must_use]
    pub fn new(vocab: Vec<String>, bos_id: Option<u32>, eos_id: Option<u32>) -> Self {
        Self {
            id_to_token: vocab,
            bos_token_id: bos_id,
            eos_token_id: eos_id,
        }
    }

    /// Decode token IDs to text
    ///
    /// Handles byte-level BPE encoding (Ġ = space prefix, Ċ = newline, etc.)
    #[must_use]
    pub fn decode(&self, token_ids: &[u32]) -> String {
        AprV2Model::decode_tokens(&self.id_to_token, token_ids)
    }

    /// Get vocabulary size
    #[must_use]
    pub fn vocab_size(&self) -> usize {
        self.id_to_token.len()
    }

    /// Check if token ID is end-of-sequence
    #[must_use]
    pub fn is_eos(&self, token_id: u32) -> bool {
        self.eos_token_id.is_some_and(|eos| token_id == eos)
    }

    /// Check if token ID is beginning-of-sequence
    #[must_use]
    pub fn is_bos(&self, token_id: u32) -> bool {
        self.bos_token_id.is_some_and(|bos| token_id == bos)
    }
}

// ============================================================================
// BpeTokenizer: Full encode/decode from tokenizer.json
// ============================================================================

/// BPE Tokenizer for encoding and decoding text
#[derive(Debug, Clone)]
pub struct BpeTokenizer {
    /// Token string to ID mapping
    pub token_to_id: HashMap<String, u32>,
    /// ID to token string mapping (index = ID)
    pub id_to_token: Vec<String>,
    /// BPE merge rules (first, second) pairs
    pub merge_rules: Vec<(String, String)>,
    /// Beginning-of-sequence token ID
    pub bos_id: Option<u32>,
    /// End-of-sequence token ID
    pub eos_id: Option<u32>,
    /// Special tokens (e.g., <|im_start|>, <|im_end|>) - GH-189 fix
    /// These are tokenized atomically, not split by BPE
    pub special_tokens: HashMap<String, u32>,
}

impl BpeTokenizer {
    /// Encode text to token IDs
    pub fn encode(&self, text: &str) -> Vec<u32> {
        // GH-189: Use special_tokens for atomic tokenization of chat markers
        bpe_encode(
            text,
            &self.token_to_id,
            &self.merge_rules,
            &self.special_tokens,
        )
    }

    /// Decode token IDs to text
    pub fn decode(&self, token_ids: &[u32]) -> String {
        AprV2Model::decode_tokens(&self.id_to_token, token_ids)
    }
}

/// Byte-level BPE encoding with special token support (F-REGR-231 fix)
///
/// Special tokens (like `<|im_start|>`, `<|im_end|>`) are handled atomically
/// and not split by BPE. This is critical for chat template markers.
pub(crate) fn bpe_encode(
    text: &str,
    vocab: &HashMap<String, u32>,
    merges: &[(String, String)],
    special_tokens: &HashMap<String, u32>,
) -> Vec<u32> {
    // F-REGR-231: Split text by special tokens first so that multi-char
    // sequences like <|im_start|> are preserved as single tokens
    let segments = split_by_special_tokens(text, special_tokens);

    let mut result = Vec::new();
    for segment in segments {
        match segment {
            TextSegment::Special(id) => {
                result.push(id);
            },
            TextSegment::Regular(s) => {
                result.extend(bpe_encode_segment(&s, vocab, merges));
            },
        }
    }
    result
}

/// Segment type for special token handling
enum TextSegment {
    Special(u32),
    Regular(String),
}

/// Split text by special tokens, preserving order
/// Try to match a special token at the start of remaining text
///
/// Returns (token_id, bytes_consumed) if matched.
fn try_match_special_at_start<'a>(
    remaining: &str,
    sorted_tokens: &[(&'a String, &'a u32)],
) -> Option<(u32, usize)> {
    for (token_str, &token_id) in sorted_tokens {
        if remaining.starts_with(token_str.as_str()) {
            return Some((token_id, token_str.len()));
        }
    }
    None
}

/// Find the byte position of the earliest special token in remaining text
fn find_earliest_special_pos(remaining: &str, sorted_tokens: &[(&String, &u32)]) -> usize {
    let mut earliest = remaining.len();
    for (token_str, _) in sorted_tokens {
        if let Some(pos) = remaining.find(token_str.as_str()) {
            earliest = earliest.min(pos);
        }
    }
    earliest
}

fn split_by_special_tokens(text: &str, special_tokens: &HashMap<String, u32>) -> Vec<TextSegment> {
    if special_tokens.is_empty() {
        return vec![TextSegment::Regular(text.to_string())];
    }

    let mut sorted_tokens: Vec<(&String, &u32)> = special_tokens.iter().collect();
    sorted_tokens.sort_by(|a, b| b.0.len().cmp(&a.0.len()));

    let mut segments = Vec::new();
    let mut remaining = text;

    while !remaining.is_empty() {
        if let Some((token_id, consumed)) = try_match_special_at_start(remaining, &sorted_tokens) {
            segments.push(TextSegment::Special(token_id));
            remaining = &remaining[consumed..];
        } else {
            let next_pos = find_earliest_special_pos(remaining, &sorted_tokens);
            if next_pos > 0 {
                segments.push(TextSegment::Regular(remaining[..next_pos].to_string()));
                remaining = &remaining[next_pos..];
            }
        }
    }

    segments
}

/// Convert a character to its byte-level BPE token representation (GPT-2/Qwen style)
fn char_to_bpe_token(c: char) -> String {
    match c {
        ' ' => "Ġ".to_string(),
        '\n' => "Ċ".to_string(),
        '\t' => "ĉ".to_string(),
        c if c.is_ascii() => c.to_string(),
        c => {
            let mut buf = [0u8; 4];
            let s = c.encode_utf8(&mut buf);
            s.chars()
                .map(|byte_char| byte_to_bpe_char(byte_char as u8))
                .collect()
        },
    }
}

/// Apply a single BPE merge rule to the token list, returning true if any merge was applied
fn apply_bpe_merge(tokens: &mut Vec<String>, first: &str, second: &str, merged: &str) -> bool {
    let mut found = false;
    let mut i = 0;
    while i + 1 < tokens.len() {
        if tokens[i] == first && tokens[i + 1] == second {
            tokens[i] = merged.to_string();
            tokens.remove(i + 1);
            found = true;
        }
        i += 1;
    }
    found
}

/// BPE encode a regular text segment (no special tokens)
fn bpe_encode_segment(
    text: &str,
    vocab: &HashMap<String, u32>,
    merges: &[(String, String)],
) -> Vec<u32> {
    let mut tokens: Vec<String> = text.chars().map(char_to_bpe_token).collect();

    // Apply BPE merges iteratively
    for (first, second) in merges {
        let merged = format!("{}{}", first, second);
        while apply_bpe_merge(&mut tokens, first, second, &merged) {}
    }

    // Convert tokens to IDs
    tokens
        .iter()
        .filter_map(|t| vocab.get(t).copied())
        .collect()
}

/// Convert byte to BPE character representation
pub fn byte_to_bpe_char(b: u8) -> String {
    // GPT-2/Qwen byte-level BPE uses specific unicode mappings
    // This is a simplified version - real tokenizers use a full byte-to-unicode table
    match b {
        b' ' => "Ġ".to_string(),
        b'\n' => "Ċ".to_string(),
        b'\t' => "ĉ".to_string(),
        _ if b.is_ascii_graphic() || b.is_ascii_alphanumeric() => (b as char).to_string(),
        _ => format!("<0x{:02X}>", b),
    }
}

include!("tokenizer_tests.rs");
