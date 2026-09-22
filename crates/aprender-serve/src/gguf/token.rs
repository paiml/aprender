
impl GGUFModel {

    /// Get normalization epsilon from metadata
    /// Different models use different values (LLaMA: 1e-5, Qwen2: 1e-6)
    /// GH-278: Also checks `layer_norm_epsilon` for GPT-2/phi-2 style models
    pub fn rms_epsilon(&self) -> Option<f32> {
        let arch = self.architecture()?;
        let rms_key = crate::gguf::keys::arch_key(arch, crate::gguf::keys::ATTENTION_LAYER_NORM_RMS_EPSILON);
        if let Some(GGUFValue::Float32(eps)) = self.metadata.get(&rms_key) {
            return Some(*eps);
        }
        let ln_key = crate::gguf::keys::arch_key(arch, crate::gguf::keys::ATTENTION_LAYER_NORM_EPSILON);
        if let Some(GGUFValue::Float32(eps)) = self.metadata.get(&ln_key) {
            return Some(*eps);
        }
        None
    }

    /// Get RoPE type from metadata or infer from architecture
    /// Returns: 0 = NORM (adjacent pairs), 2 = NEOX (split halves)
    /// Per llama.cpp: LLAMA_ROPE_TYPE_NORM = 0, LLAMA_ROPE_TYPE_NEOX = 2
    ///
    /// GH-329: Delegates to shared `infer_rope_type()` for architecture inference.
    pub fn rope_type(&self) -> Option<u32> {
        let arch = self.architecture()?;
        let key = crate::gguf::keys::arch_key(arch, crate::gguf::keys::ROPE_SCALING_TYPE);
        // Try rope type from scaling type first
        if let Some(GGUFValue::String(s)) = self.metadata.get(&key) {
            match s.as_str() {
                "none" | "linear" => return Some(0), // NORM style
                "yarn" | "neox" => return Some(2),   // NEOX style
                _ => {},
            }
        }
        // GH-329: Use shared inference function (single source of truth)
        Some(crate::gguf::infer_rope_type(arch))
    }

    /// Get BOS (beginning of sentence) token ID
    #[must_use]
    pub fn bos_token_id(&self) -> Option<u32> {
        if let Some(GGUFValue::UInt32(id)) = self.metadata.get(crate::gguf::keys::TOKENIZER_BOS_ID) {
            Some(*id)
        } else {
            None
        }
    }

    /// Get EOS (end of sentence) token ID
    #[must_use]
    pub fn eos_token_id(&self) -> Option<u32> {
        if let Some(GGUFValue::UInt32(id)) = self.metadata.get(crate::gguf::keys::TOKENIZER_EOS_ID) {
            Some(*id)
        } else {
            None
        }
    }

    /// Get vocabulary tokens from metadata
    ///
    /// Returns the token strings indexed by token ID.
    /// Uses "tokenizer.ggml.tokens" key from GGUF metadata.
    #[must_use]
    pub fn vocabulary(&self) -> Option<Vec<String>> {
        if let Some(GGUFValue::Array(arr)) = self.metadata.get(crate::gguf::keys::TOKENIZER_TOKENS) {
            let tokens: Vec<String> = arr
                .iter()
                .filter_map(|v| {
                    if let GGUFValue::String(s) = v {
                        Some(s.clone())
                    } else {
                        None
                    }
                })
                .collect();
            if tokens.is_empty() {
                None
            } else {
                Some(tokens)
            }
        } else {
            None
        }
    }

    /// PMAT-341: Get BPE merge rules from metadata.
    ///
    /// Returns merge pairs as (first, second) tuples.
    /// Uses "tokenizer.ggml.merges" key from GGUF metadata.
    #[must_use]
    pub fn merge_rules(&self) -> Option<Vec<(String, String)>> {
        contract_pre_merge_rule!();
        if let Some(GGUFValue::Array(arr)) = self.metadata.get("tokenizer.ggml.merges") {
            let merges: Vec<(String, String)> = arr
                .iter()
                .filter_map(|v| {
                    if let GGUFValue::String(s) = v {
                        let parts: Vec<&str> = s.splitn(2, ' ').collect();
                        if parts.len() == 2 {
                            Some((parts[0].to_string(), parts[1].to_string()))
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                })
                .collect();
            if merges.is_empty() { None } else { Some(merges) }
        } else {
            None
        }
    }

    /// Decode token IDs to text using vocabulary
    ///
    /// Returns decoded string. Unknown tokens are replaced with "�".
    /// Handles BPE markers:
    /// - GPT-2 style: Ġ (U+0120) → space, Ċ (U+010A) → newline
    /// - SentencePiece: ▁ (U+2581) → space
    /// - Byte tokens: <0xHH> → actual byte value
    #[must_use]
    pub fn decode(&self, token_ids: &[u32]) -> String {
        if let Some(vocab) = self.vocabulary() {
            // Detect tokenizer type from metadata
            let is_gpt2_style = self
                .metadata
                .get(crate::gguf::keys::TOKENIZER_MODEL)
                .is_some_and(|v| matches!(v, GGUFValue::String(s) if s == "gpt2" || s == "bpe"));

            // Collect raw tokens and convert byte tokens to actual bytes
            let mut bytes: Vec<u8> = Vec::new();

            for &id in token_ids {
                let token = vocab
                    .get(id as usize)
                    .map_or("�", std::string::String::as_str);
                push_token_bytes(token, is_gpt2_style, &mut bytes);
            }

            // Decode bytes as UTF-8 (lossy for invalid sequences)
            let raw = String::from_utf8_lossy(&bytes).into_owned();

            // Post-process BPE markers (only for SentencePiece, GPT-2 already handled)
            if !is_gpt2_style {
                raw.replace('▁', " ") // SentencePiece word boundary
            } else {
                raw
            }
        } else {
            // Fallback to ASCII if no vocabulary
            token_ids
                .iter()
                .map(|&t| char::from_u32(t.min(127)).unwrap_or('?'))
                .collect()
        }
    }

    /// #3726: the canonical encoding of a byte-level (`gpt2`) vocabulary, or `None` when the
    /// file is not byte-level or its pre-tokenizer is not implemented (said once on stderr).
    /// Greedy longest-match gave these vocabularies ids the model never saw in training
    /// (" quorum" -> `Ġquo|rum`, where the merges give `Ġqu|orum`; every byte outside the
    /// printable glyphs -> id 0), and the model quoted identifiers back corrupted (#3693).
    fn encode_byte_level(&self, text: &str, vocab: &[String]) -> Option<Vec<u32>> {
        let byte_level = self
            .metadata
            .get("tokenizer.ggml.model")
            .is_some_and(|v| matches!(v, GGUFValue::String(s) if s == "gpt2" || s == "bpe"));
        if !byte_level {
            return None;
        }
        crate::gguf::byte_level_bpe::ByteLevelBpe::from_gguf(&self.metadata, vocab)
            .map_err(|refusal| warn_greedy_tokenizer_fallback_once(&refusal))
            .ok()
            .map(|bpe| bpe.encode(text))
    }

    /// Encode text to token IDs using vocabulary
    ///
    /// A byte-level (`tokenizer.ggml.model = "gpt2"`) vocabulary whose pre-tokenizer is
    /// implemented is encoded canonically by [`crate::gguf::byte_level_bpe`]: special tokens,
    /// then the model's pre-tokenizer, then its ranked merges, identical to llama.cpp (#3726).
    /// Everything else still takes the greedy longest-match below, which is NOT the model's
    /// tokenization; that fallback says so once on stderr.
    /// Returns None if no vocabulary is available.
    ///
    /// Supports both tokenizer types:
    /// - SentencePiece (llama): Uses `▁` (U+2581) for word boundaries
    /// - GPT-2 (qwen2, gpt2): Uses `Ġ` (U+0120) for space prefixes
    #[must_use]
    pub fn encode(&self, text: &str) -> Option<Vec<u32>> {
        let vocab = self.vocabulary()?;

        if let Some(ids) = self.encode_byte_level(text, &vocab) {
            return Some(ids);
        }

        // Build reverse lookup: token string -> token ID
        let token_to_id: std::collections::HashMap<&str, u32> = vocab
            .iter()
            .enumerate()
            .map(|(id, token)| (token.as_str(), id as u32))
            .collect();

        // GH-320: Identify special tokens by pattern, not by hardcoded ID threshold.
        // Matches <|...|> tokens at any ID position in the vocabulary.
        let special_tokens: Vec<(&str, u32)> = vocab
            .iter()
            .enumerate()
            .filter(|(_id, tok)| tok.starts_with("<|") && tok.ends_with("|>"))
            .map(|(id, tok)| (tok.as_str(), id as u32))
            .collect();

        // Detect tokenizer type from metadata
        // GPT-2 style uses Ġ (U+0120), SentencePiece uses ▁ (U+2581)
        let is_gpt2_style = self
            .metadata
            .get("tokenizer.ggml.model")
            .is_some_and(|v| matches!(v, GGUFValue::String(s) if s == "gpt2" || s == "bpe"));

        let space_char = if is_gpt2_style { '\u{0120}' } else { '▁' };

        // Split text on special tokens first, preserving them
        let segments = split_on_special_tokens(text, &special_tokens);

        let mut tokens = Vec::new();

        for (is_special, segment) in segments {
            if is_special {
                // Direct lookup for special token
                if let Some(&id) = token_to_id.get(segment) {
                    tokens.push(id);
                }
                continue;
            }

            // Process non-special segment with character replacement
            let text_with_prefix = if is_gpt2_style {
                segment.to_string()
            } else if segment.starts_with(' ') {
                segment.to_string()
            } else {
                format!(" {}", segment)
            };

            let processed = if is_gpt2_style {
                text_with_prefix
                    .replace(' ', &space_char.to_string())
                    .replace('\n', "\u{010A}") // Ċ = GPT-2 newline
            } else {
                text_with_prefix.replace(' ', &space_char.to_string())
            };

            let mut remaining = processed.as_str();

            while !remaining.is_empty() {
                // Greedy longest match using character boundaries (not byte indices)
                let mut best_byte_len = 0;
                let mut best_id = None;

                // Collect character byte offsets for proper slicing
                let char_indices: Vec<usize> = remaining
                    .char_indices()
                    .map(|(i, _)| i)
                    .chain(std::iter::once(remaining.len()))
                    .collect();

                // Try all prefixes up to 32 chars (reasonable max token length)
                for char_count in 1..=char_indices.len().saturating_sub(1).min(32) {
                    let byte_end = char_indices[char_count];
                    let prefix = &remaining[..byte_end];
                    if let Some(&id) = token_to_id.get(prefix) {
                        best_byte_len = byte_end;
                        best_id = Some(id);
                    }
                }

                if let Some(id) = best_id {
                    tokens.push(id);
                    remaining = &remaining[best_byte_len..];
                } else {
                    // No match found - try single UTF-8 char as byte tokens
                    // SAFETY: remaining is non-empty (loop condition guarantees this)
                    let ch = remaining
                        .chars()
                        .next()
                        .expect("loop invariant: remaining non-empty");
                    let ch_len = ch.len_utf8();

                    // Look for byte tokens like <0x48> for 'H'
                    for byte in remaining[..ch_len].bytes() {
                        let byte_token = format!("<0x{:02X}>", byte);
                        if let Some(&id) = token_to_id.get(byte_token.as_str()) {
                            tokens.push(id);
                        } else {
                            // Unknown byte - use a common unknown token ID (usually 0 or 1)
                            tokens.push(0);
                        }
                    }
                    remaining = &remaining[ch_len..];
                }
            }
        }

        Some(tokens)
    }

    /// The vocabulary as schema-constrained decoding sees it (#3568): each
    /// token's raw bytes from [`push_token_bytes`], the same per-token decoding
    /// [`Self::decode`] uses, so the mask and the decoded text agree about what a
    /// token is. Control and unused tokens (`tokenizer.ggml.token_type` 3 and 5)
    /// are special and can never be emitted as constrained text. Without that
    /// metadata, a `<|…|>` token is special: the same rule `encode` uses (GH-320).
    ///
    /// `None` when the file has no vocabulary or no end-of-sequence id.
    #[must_use]
    pub fn constraint_vocab(&self) -> Option<crate::constrain::ConstraintVocab> {
        let vocab = self.vocabulary()?;
        let eos = self.eos_token_id()?;
        let is_gpt2_style = self
            .metadata
            .get(crate::gguf::keys::TOKENIZER_MODEL)
            .is_some_and(|v| matches!(v, GGUFValue::String(s) if s == "gpt2" || s == "bpe"));
        let token_types: Option<Vec<i32>> = match self.metadata.get(TOKENIZER_TOKEN_TYPE) {
            Some(GGUFValue::Array(arr)) => arr
                .iter()
                .map(|v| match v {
                    GGUFValue::Int32(t) => Some(*t),
                    _ => None,
                })
                .collect(),
            _ => None,
        };
        let mut token_bytes = Vec::with_capacity(vocab.len());
        let mut special = Vec::with_capacity(vocab.len());
        for (id, token) in vocab.iter().enumerate() {
            let is_special = match token_types.as_ref().and_then(|t| t.get(id)) {
                Some(&t) => t == 3 || t == 5, // CONTROL, UNUSED
                None => token.starts_with("<|") && token.ends_with("|>"),
            };
            let mut bytes = Vec::new();
            if is_special {
                bytes.extend_from_slice(token.as_bytes());
            } else {
                push_token_bytes(token, is_gpt2_style, &mut bytes);
                if !is_gpt2_style {
                    // decode() maps SentencePiece's word boundary `▁` (E2 96 81) to a space after
                    // joining; done per token here, on BYTES, so a lone <0xNN> byte token survives
                    bytes = replace_sp_word_boundary(&bytes);
                }
            }
            token_bytes.push(bytes);
            special.push(is_special);
        }
        Some(crate::constrain::ConstraintVocab {
            token_bytes,
            special,
            eos,
        })
    }
}

/// `tokenizer.ggml.token_type`: one llama.cpp token type per token (1 normal,
/// 3 control, 5 unused, 6 byte, …).
const TOKENIZER_TOKEN_TYPE: &str = "tokenizer.ggml.token_type";

/// Append one token's raw bytes, exactly as [`GGUFModel::decode`] renders it: a
/// `<0xNN>` byte token is that byte; a GPT-2 byte-level BPE token maps each char
/// back to its byte; a SentencePiece token is its UTF-8 text. `decode` and the
/// constraint vocabulary (#3568) both call this, so they cannot drift apart.
fn push_token_bytes(token: &str, is_gpt2_style: bool, bytes: &mut Vec<u8>) {
    if token.starts_with("<0x") && token.ends_with('>') && token.len() == 6 {
        if let Ok(byte_val) = u8::from_str_radix(
            token
                .get(3..5)
                .expect("byte token <0xNN> has len 6, indices 3..5 always valid"),
            16,
        ) {
            bytes.push(byte_val);
            return;
        }
    }
    if is_gpt2_style {
        // Each unicode character in a byte-level BPE token represents a raw byte
        bytes.extend(token.chars().filter_map(gpt2_unicode_to_byte));
    } else {
        bytes.extend_from_slice(token.as_bytes());
    }
}

/// SentencePiece's word boundary `▁` (U+2581, bytes E2 96 81) as a space, on raw bytes.
fn replace_sp_word_boundary(bytes: &[u8]) -> Vec<u8> {
    const BOUNDARY: [u8; 3] = [0xE2, 0x96, 0x81];
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i..].starts_with(&BOUNDARY) {
            out.push(b' ');
            i += BOUNDARY.len();
        } else {
            out.push(bytes[i]);
            i += 1;
        }
    }
    out
}

use crate::gguf::{
    OwnedQKVWeights, OwnedQuantizedLayer, OwnedQuantizedModel, OwnedQuantizedTensor,
    QuantizedGGUFTransformer,
};

include!("loader_parse.rs");
include!("metadata.rs");

/// The greedy longest-match fallback is not the model's tokenization; say so, once (#3726).
fn warn_greedy_tokenizer_fallback_once(refusal: &crate::gguf::byte_level_bpe::ByteLevelBpeRefusal) {
    static WARNED: std::sync::Once = std::sync::Once::new();
    WARNED.call_once(|| {
        eprintln!(
            "warning: byte-level BPE tokenizer: {refusal}; falling back to greedy longest-match, \
             which does NOT reproduce the model's tokenization (#3726)"
        );
    });
}

/// Split `text` at the earliest occurrence of any special token, repeatedly, keeping the
/// specials as their own `(true, token)` segments (the greedy path's special handling).
fn split_on_special_tokens<'t>(text: &'t str, special_tokens: &[(&'t str, u32)]) -> Vec<(bool, &'t str)> {
    let mut segments = Vec::new();
    let mut rest = text;
    while !rest.is_empty() {
        let earliest = special_tokens
            .iter()
            .filter_map(|&(tok, _)| rest.find(tok).map(|pos| (pos, tok)))
            .min_by_key(|&(pos, _)| pos);
        let Some((pos, tok)) = earliest else {
            segments.push((false, rest));
            break;
        };
        if pos > 0 {
            segments.push((false, &rest[..pos]));
        }
        segments.push((true, tok));
        rest = &rest[pos + tok.len()..];
    }
    segments
}

