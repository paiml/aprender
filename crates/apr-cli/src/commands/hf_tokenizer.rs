//! #3742: the one tokenizer.json encoder apr-cli uses.
//!
//! HuggingFace byte-level vocabularies (Qwen, GPT-2, Llama 3) split the text with a regex
//! before their merges run. aprender-core's `BpeTokenizer` only splits on whitespace, so it
//! refuses such a vocabulary (`load_from_json`, #3742) instead of producing ids the model
//! never saw. Every apr-cli path that loads a tokenizer.json goes through this adapter, which
//! uses realizar's loader: it builds the canonical byte-level BPE (#3726, identical to
//! llama.cpp) from the tokenizer.json's own pre-tokenizer.

use std::path::Path;

/// A tokenizer.json loaded through realizar's canonical byte-level BPE.
#[derive(Debug, Clone)]
pub(crate) struct HfTokenizer {
    inner: realizar::apr::BpeTokenizer,
}

impl HfTokenizer {
    /// Load `path` (a HuggingFace tokenizer.json).
    ///
    /// # Errors
    /// The file is missing or is not a BPE tokenizer.json.
    pub(crate) fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, String> {
        let path = path.as_ref();
        realizar::apr::AprV2Model::load_tokenizer_from_path(path)
            .map(|inner| Self { inner })
            .ok_or_else(|| format!("{} is not a readable BPE tokenizer.json", path.display()))
    }

    /// Token ids for `text` (special tokens matched first).
    pub(crate) fn encode(&self, text: &str) -> Vec<u32> {
        self.inner.encode(text)
    }

    /// Text for `ids`.
    pub(crate) fn decode(&self, ids: &[u32]) -> String {
        self.inner.decode(ids)
    }

    /// Vocabulary size, added tokens included.
    pub(crate) fn vocab_size(&self) -> usize {
        let top_special = self
            .inner
            .special_tokens
            .values()
            .map(|&id| id as usize + 1)
            .max()
            .unwrap_or(0);
        self.inner.id_to_token.len().max(top_special)
    }

    /// The id of `token`, special or ordinary.
    pub(crate) fn token_to_id(&self, token: &str) -> Option<u32> {
        self.inner
            .special_tokens
            .get(token)
            .or_else(|| self.inner.token_to_id.get(token))
            .copied()
    }

    /// Whether the canonical encoder was built (the tokenizer.json's pre-tokenizer is one
    /// realizar implements). `false` means the legacy merge loop, which realizar has already
    /// named on stderr.
    pub(crate) fn is_canonical(&self) -> bool {
        self.inner.canonical.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::HfTokenizer;

    /// A minimal byte-level tokenizer.json with the real Qwen2 Split regex: the 256 GPT-2
    /// byte glyphs, one merge, and one added special token.
    fn qwen2_style_tokenizer_json() -> String {
        let printable = |b: u8| matches!(b, b'!'..=b'~' | 0xA1..=0xAC | 0xAE..=0xFF);
        let mut next = 0u32;
        let mut vocab = serde_json::Map::new();
        for b in 0..=255u8 {
            let glyph = if printable(b) {
                char::from(b)
            } else {
                next += 1;
                char::from_u32(255 + next).expect("valid")
            };
            vocab.insert(glyph.to_string(), serde_json::json!(b));
        }
        vocab.insert("ab".to_string(), serde_json::json!(256));
        serde_json::json!({
            "model": {"type": "BPE", "vocab": vocab, "merges": ["a b"]},
            "added_tokens": [{"id": 257, "content": "<|im_start|>", "special": true}],
            "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
                {"type": "Split", "pattern": {"Regex": r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"},
                 "behavior": "Isolated", "invert": false},
                {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false}
            ]}
        })
        .to_string()
    }

    /// #3742 guard, both halves: a byte-level tokenizer.json reaches the canonical encoder
    /// through apr-cli's adapter, and aprender-core's BPE refuses the same file. A consumer
    /// added later that calls core's BPE on a byte-level vocabulary gets the refusal, not
    /// silently wrong ids.
    #[test]
    fn byte_level_tokenizer_json_reaches_the_canonical_encoder_and_core_refuses_it() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("tokenizer.json");
        let json = qwen2_style_tokenizer_json();
        std::fs::write(&path, &json).expect("write");

        let tok = HfTokenizer::from_file(&path).expect("loads");
        assert!(
            tok.is_canonical(),
            "the Qwen2 Split regex is implemented: canonical encoder"
        );
        assert_eq!(tok.encode("<|im_start|>ab"), vec![257, 256]);
        assert_eq!(
            tok.decode(&tok.encode("<|im_start|>ab é")),
            "<|im_start|>ab é"
        );
        assert_eq!(tok.vocab_size(), 258);
        assert_eq!(tok.token_to_id("<|im_start|>"), Some(257));

        let core = aprender::text::bpe::load_from_json(&json);
        assert!(
            core.as_ref()
                .is_err_and(|e| e.to_string().contains("#3742")),
            "aprender-core must refuse a regex pre-tokenizer vocabulary: {:?}",
            core.map(|_| ())
        );
    }
}
