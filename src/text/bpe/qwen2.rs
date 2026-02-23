#[allow(clippy::wildcard_imports)]
use super::*;
use crate::error::{AprenderError, Result};
use serde::Deserialize;
use std::collections::HashMap;

impl Qwen2BpeTokenizer {
    // Source of truth: SpecialTokens::qwen2() from special-tokens-registry-v1.yaml
    /// Special token: <|`im_start`|>
    pub const IM_START_ID: u32 = crate::demo::SpecialTokens::qwen2().im_start_id;
    /// Special token: <|`im_end`|>
    pub const IM_END_ID: u32 = crate::demo::SpecialTokens::qwen2().im_end_id;
    /// Special token: <|endoftext|>
    pub const ENDOFTEXT_ID: u32 = crate::demo::SpecialTokens::qwen2().bos_id;

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
        crate::demo::Qwen2Config::VOCAB_SIZE
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

    let hf_tokenizer: HfTokenizerJson =
        serde_json::from_str(json).map_err(|e| AprenderError::FormatError {
            message: format!("Failed to parse tokenizer JSON: {e}"),
        })?;

    let config = config_from_vocab_size(hf_tokenizer.model.vocab.len());
    let mut tokenizer = BpeTokenizer::new(config);

    load_vocab_into(&mut tokenizer, &hf_tokenizer.model.vocab);
    load_merges_from_strings(&mut tokenizer, &hf_tokenizer.model.merges);
    load_added_tokens(&mut tokenizer, &hf_tokenizer.added_tokens);

    Ok(tokenizer)
}

/// Determine BPE config based on vocabulary size.
fn config_from_vocab_size(vocab_size: usize) -> BpeConfig {
    if vocab_size > 150_000 {
        BpeConfig::qwen2()
    } else if vocab_size > 50_000 {
        BpeConfig::whisper()
    } else if vocab_size > 40_000 {
        BpeConfig::gpt2()
    } else {
        BpeConfig::llama()
    }
}

/// Load vocabulary entries into a tokenizer.
fn load_vocab_into(tokenizer: &mut BpeTokenizer, vocab: &HashMap<String, u32>) {
    for (token, id) in vocab {
        tokenizer.vocab.insert(token.clone(), *id);
        tokenizer.id_to_token.insert(*id, token.clone());
    }
}

/// Load merge rules from space-separated merge strings.
fn load_merges_from_strings(tokenizer: &mut BpeTokenizer, merges: &[String]) {
    for merge_str in merges {
        let parts: Vec<&str> = merge_str.split(' ').collect();
        if parts.len() >= 2 {
            tokenizer.add_merge(parts[0], parts[1]);
        }
    }
}

/// Load added tokens (special and regular) into a tokenizer.
fn load_added_tokens(tokenizer: &mut BpeTokenizer, added_tokens: &[HfAddedToken]) {
    for added in added_tokens {
        if added.special {
            tokenizer.add_special_token(&added.content, added.id);
        } else {
            tokenizer.vocab.insert(added.content.clone(), added.id);
            tokenizer
                .id_to_token
                .insert(added.id, added.content.clone());
        }
    }
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

    let vocab: HashMap<String, u32> =
        serde_json::from_str(vocab_json).map_err(|e| AprenderError::FormatError {
            message: format!("Failed to parse vocabulary JSON: {e}"),
        })?;

    let config = config_from_vocab_size(vocab.len());
    let mut tokenizer = BpeTokenizer::new(config);

    load_vocab_into(&mut tokenizer, &vocab);
    load_merges_from_text(&mut tokenizer, merges_txt);

    Ok(tokenizer)
}

/// Parse merge rules from merges.txt format (one "pair1 pair2" per line, skipping comments).
fn load_merges_from_text(tokenizer: &mut BpeTokenizer, merges_txt: &str) {
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
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
