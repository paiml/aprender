//! BPE (Byte Pair Encoding) tokenizer implementation.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use unicode_normalization::UnicodeNormalization;

use super::config::{Normalization, TokenizerConfig};
use super::error::{Result, TokenizerError};
use super::traits::{TokenId, Tokenizer};

/// BPE (Byte Pair Encoding) tokenizer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BPETokenizer {
    config: TokenizerConfig,
    /// Token to ID mapping
    vocab: HashMap<String, TokenId>,
    /// ID to token mapping
    id_to_token_map: HashMap<TokenId, String>,
    /// Merge rules (pair -> merged token)
    merges: Vec<(String, String)>,
    /// Whether the tokenizer is trained
    trained: bool,
}

impl BPETokenizer {
    /// Create a new BPE tokenizer
    pub fn new(config: TokenizerConfig) -> Self {
        Self {
            config,
            vocab: HashMap::new(),
            id_to_token_map: HashMap::new(),
            merges: Vec::new(),
            trained: false,
        }
    }

    /// Initialize vocabulary with special tokens and bytes
    fn init_vocab(&mut self) {
        let mut id: TokenId = 0;

        // Add special tokens
        let special = [
            &self.config.special_tokens.unk,
            &self.config.special_tokens.bos,
            &self.config.special_tokens.eos,
            &self.config.special_tokens.pad,
            &self.config.special_tokens.mask,
        ];

        for token in special {
            self.vocab.insert(token.clone(), id);
            self.id_to_token_map.insert(id, token.clone());
            id += 1;
        }

        // Add all single bytes as base vocabulary
        for byte in 0..=255u8 {
            let token = format!("{byte:02x}");
            if !self.vocab.contains_key(&token) {
                self.vocab.insert(token.clone(), id);
                self.id_to_token_map.insert(id, token);
                id += 1;
            }
        }
    }

    /// Get pair frequencies from tokenized corpus
    fn get_pair_freqs(&self, tokenized: &[Vec<String>]) -> HashMap<(String, String), usize> {
        let mut freqs = HashMap::new();

        for tokens in tokenized {
            for pair in tokens.windows(2) {
                let key = (pair[0].clone(), pair[1].clone());
                *freqs.entry(key).or_insert(0) += 1;
            }
        }

        freqs
    }

    /// Merge the most frequent pair
    fn merge_pair(&self, tokenized: &mut [Vec<String>], pair: &(String, String), merged: &str) {
        for tokens in tokenized.iter_mut() {
            let mut i = 0;
            while i < tokens.len().saturating_sub(1) {
                if tokens[i] == pair.0 && tokens[i + 1] == pair.1 {
                    tokens[i] = merged.to_string();
                    tokens.remove(i + 1);
                }
                i += 1;
            }
        }
    }

    /// Apply the configured Unicode normalization, then optional lowercasing.
    ///
    /// NFC is applied BEFORE lowercasing because `char::to_lowercase()` is not
    /// closed over non-NFC input for every grapheme — normalizing first makes
    /// the pipeline deterministic for composed/decomposed variants of the
    /// same visible text.
    fn preprocess(&self, text: &str) -> String {
        let normalized = match self.config.normalization {
            Normalization::None => text.to_string(),
            Normalization::NFC => text.nfc().collect(),
        };
        if self.config.lowercase {
            normalized.to_lowercase()
        } else {
            normalized
        }
    }

    /// Tokenize text to bytes (initial tokenization)
    fn to_bytes(&self, text: &str) -> Vec<String> {
        text.as_bytes().iter().map(|b| format!("{b:02x}")).collect()
    }

    /// Apply all learned merges
    fn apply_merges(&self, mut tokens: Vec<String>) -> Vec<String> {
        for (a, b) in &self.merges {
            let merged = format!("{a}{b}");
            let mut i = 0;
            while i < tokens.len().saturating_sub(1) {
                if &tokens[i] == a && &tokens[i + 1] == b {
                    tokens[i] = merged.clone();
                    tokens.remove(i + 1);
                } else {
                    i += 1;
                }
            }
        }
        tokens
    }

    /// Borrow the learned `token → id` vocabulary map.
    ///
    /// Exposed so callers (e.g. `apr tokenize train`) can emit the HuggingFace
    /// `vocab.json` artifact mandated by `contracts/tokenizer-bpe-v1.yaml` without
    /// serializing the whole `BPETokenizer` struct. Read-only by design — training
    /// and encoding continue to own the `HashMap`.
    pub fn vocab(&self) -> &HashMap<String, TokenId> {
        &self.vocab
    }

    /// Borrow the ordered list of learned merge rules (`(left, right)` pairs in
    /// merge order).
    ///
    /// Exposed so callers can write the HuggingFace `merges.txt` artifact. The
    /// order is load-bearing: `merges.txt` consumers apply pairs top-to-bottom.
    pub fn merges(&self) -> &[(String, String)] {
        &self.merges
    }

    /// Save tokenizer to file
    pub fn save(&self, path: &str) -> Result<()> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| TokenizerError::Serialization(e.to_string()))?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Load tokenizer from file
    pub fn load(path: &str) -> Result<Self> {
        let json = std::fs::read_to_string(path)?;
        serde_json::from_str(&json).map_err(|e| TokenizerError::Serialization(e.to_string()))
    }

    /// Reconstruct a trained `BPETokenizer` from the HuggingFace-style pair of
    /// `vocab.json` + `merges.txt` emitted by `apr tokenize train`.
    ///
    /// # Format
    /// - `vocab.json`: JSON object mapping token string → token id (u32). Order
    ///   is informational; the loader inverts the map to build `id_to_token`.
    /// - `merges.txt`: leading `#version: 0.2\n` header line, then one merge per
    ///   line in apply order. Each line is `"<left> <right>"` with a single
    ///   ASCII space separator (tokens in the aprender-train hex
    ///   representation never contain spaces, so space-split is unambiguous).
    ///
    /// # Parameters
    /// - `vocab_path`: path to `vocab.json`
    /// - `merges_path`: path to `merges.txt`
    /// - `config`: caller-supplied config (normalization, special tokens, etc.)
    ///   since those fields are not recorded in the HF-style files. MUST match
    ///   the config used at training time — wrong normalization here produces
    ///   silently-wrong encodings.
    ///
    /// # Invariants
    /// - C-PRETOK-BIN INV-PRETOK-001: every loaded vocab id < returned
    ///   tokenizer's `vocab_size()`.
    /// - Every merge's `(left, right)` concatenation is present in the loaded
    ///   vocab (otherwise applying the merge during encode would produce a
    ///   token the vocab cannot resolve). Enforced; mismatch returns an error.
    pub fn from_vocab_merges(
        vocab_path: &str,
        merges_path: &str,
        config: TokenizerConfig,
    ) -> Result<Self> {
        let vocab_json = std::fs::read_to_string(vocab_path)?;
        let vocab: HashMap<String, TokenId> = serde_json::from_str(&vocab_json)
            .map_err(|e| TokenizerError::Serialization(e.to_string()))?;

        let id_to_token_map: HashMap<TokenId, String> =
            vocab.iter().map(|(tok, &id)| (id, tok.clone())).collect();

        if id_to_token_map.len() != vocab.len() {
            return Err(TokenizerError::Serialization(
                "vocab.json contains duplicate token ids (collision detected after inverting map)"
                    .to_string(),
            ));
        }

        let merges_text = std::fs::read_to_string(merges_path)?;
        let mut merges: Vec<(String, String)> = Vec::new();
        for (line_no, line) in merges_text.lines().enumerate() {
            if line.is_empty() || line.starts_with("#") {
                continue;
            }
            let mut parts = line.splitn(2, ' ');
            let left = parts
                .next()
                .ok_or_else(|| {
                    TokenizerError::Serialization(format!(
                        "merges.txt line {}: missing left token",
                        line_no + 1
                    ))
                })?
                .to_string();
            let right = parts
                .next()
                .ok_or_else(|| {
                    TokenizerError::Serialization(format!(
                        "merges.txt line {}: missing right token (expected '<left> <right>')",
                        line_no + 1
                    ))
                })?
                .to_string();

            let merged = format!("{left}{right}");
            if !vocab.contains_key(&merged) {
                return Err(TokenizerError::Serialization(format!(
                    "merges.txt line {}: merged token {:?} not present in vocab.json",
                    line_no + 1,
                    merged
                )));
            }
            merges.push((left, right));
        }

        Ok(Self { config, vocab, id_to_token_map, merges, trained: true })
    }
}

impl Tokenizer for BPETokenizer {
    fn train(&mut self, corpus: &[&str]) -> Result<()> {
        self.init_vocab();

        // Tokenize corpus to bytes
        let mut tokenized: Vec<Vec<String>> =
            corpus.iter().map(|text| self.to_bytes(&self.preprocess(text))).collect();

        // Learn merges until we reach target vocab size
        let target = self.config.vocab_size;
        while self.vocab.len() < target {
            let freqs = self.get_pair_freqs(&tokenized);

            // Find most frequent pair
            let best = freqs
                .iter()
                .filter(|(_, &count)| count >= self.config.min_frequency)
                .max_by_key(|(_, count)| *count);

            match best {
                Some((pair, _)) => {
                    let merged = format!("{}{}", pair.0, pair.1);

                    // Add to vocabulary
                    let id = self.vocab.len() as TokenId;
                    self.vocab.insert(merged.clone(), id);
                    self.id_to_token_map.insert(id, merged.clone());

                    // Record merge
                    self.merges.push(pair.clone());

                    // Apply merge
                    self.merge_pair(&mut tokenized, pair, &merged);
                }
                None => break, // No more pairs meet frequency threshold
            }
        }

        self.trained = true;
        Ok(())
    }

    fn encode(&self, text: &str) -> Result<Vec<TokenId>> {
        if !self.trained {
            return Err(TokenizerError::NotTrained);
        }

        let tokens = self.to_bytes(&self.preprocess(text));
        let tokens = self.apply_merges(tokens);

        let unk_id = *self
            .vocab
            .get(&self.config.special_tokens.unk)
            .expect("UNK token must exist in trained vocabulary");

        let ids: Vec<TokenId> =
            tokens.iter().map(|t| *self.vocab.get(t).unwrap_or(&unk_id)).collect();

        Ok(ids)
    }

    fn decode(&self, ids: &[TokenId]) -> Result<String> {
        if !self.trained {
            return Err(TokenizerError::NotTrained);
        }

        let mut hex_string = String::new();

        for &id in ids {
            if let Some(token) = self.id_to_token_map.get(&id) {
                // Skip special tokens
                if token.starts_with('<') && token.ends_with('>') {
                    continue;
                }
                hex_string.push_str(token);
            }
        }

        // Convert hex string back to bytes
        let bytes: Vec<u8> = (0..hex_string.len())
            .step_by(2)
            .filter_map(|i| {
                if i + 2 <= hex_string.len() {
                    u8::from_str_radix(&hex_string[i..i + 2], 16).ok()
                } else {
                    None
                }
            })
            .collect();

        String::from_utf8(bytes).map_err(|e| TokenizerError::Training(e.to_string()))
    }

    fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    fn is_trained(&self) -> bool {
        self.trained
    }

    fn id_to_token(&self, id: TokenId) -> Option<&str> {
        self.id_to_token_map.get(&id).map(String::as_str)
    }

    fn token_to_id(&self, token: &str) -> Option<TokenId> {
        self.vocab.get(token).copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bpe_new() {
        let config = TokenizerConfig::bpe();
        let tokenizer = BPETokenizer::new(config);
        assert!(!tokenizer.is_trained());
    }

    #[test]
    fn test_bpe_train() {
        let config = TokenizerConfig::bpe().with_vocab_size(300).with_min_frequency(1);
        let mut tokenizer = BPETokenizer::new(config);

        let corpus = vec!["hello hello", "hello world", "world hello"];
        tokenizer.train(&corpus).expect("operation should succeed");

        assert!(tokenizer.is_trained());
        assert!(tokenizer.vocab_size() > 256); // Base bytes + some merges
    }

    #[test]
    fn test_bpe_encode_not_trained() {
        let config = TokenizerConfig::bpe();
        let tokenizer = BPETokenizer::new(config);

        let result = tokenizer.encode("hello");
        assert!(result.is_err());
    }

    #[test]
    fn test_bpe_encode_decode() {
        let config = TokenizerConfig::bpe().with_vocab_size(300).with_min_frequency(1);
        let mut tokenizer = BPETokenizer::new(config);

        let corpus = vec!["hello world", "hello there"];
        tokenizer.train(&corpus).expect("operation should succeed");

        let text = "hello";
        let encoded = tokenizer.encode(text).expect("encoding should succeed");
        let decoded = tokenizer.decode(&encoded).expect("encoding should succeed");

        assert_eq!(decoded, text);
    }

    #[test]
    fn test_bpe_lowercase() {
        let config =
            TokenizerConfig::bpe().with_vocab_size(300).with_min_frequency(1).with_lowercase(true);
        let mut tokenizer = BPETokenizer::new(config);

        let corpus = vec!["Hello World"];
        tokenizer.train(&corpus).expect("operation should succeed");

        let encoded = tokenizer.encode("HELLO").expect("encoding should succeed");
        let decoded = tokenizer.decode(&encoded).expect("encoding should succeed");

        assert_eq!(decoded, "hello");
    }

    #[test]
    fn test_bpe_id_to_token() {
        let config = TokenizerConfig::bpe().with_vocab_size(300).with_min_frequency(1);
        let mut tokenizer = BPETokenizer::new(config);

        let corpus = vec!["test"];
        tokenizer.train(&corpus).expect("operation should succeed");

        // ID 0 should be <unk>
        assert_eq!(tokenizer.id_to_token(0), Some("<unk>"));
    }

    #[test]
    fn test_bpe_token_to_id() {
        let config = TokenizerConfig::bpe().with_vocab_size(300).with_min_frequency(1);
        let mut tokenizer = BPETokenizer::new(config);

        let corpus = vec!["test"];
        tokenizer.train(&corpus).expect("operation should succeed");

        assert_eq!(tokenizer.token_to_id("<unk>"), Some(0));
    }

    // C-TOK-BPE-001 INV-TOK-003: NFC normalization makes composed and decomposed
    // variants of the same grapheme hash to identical byte sequences, so a
    // tokenizer trained on one form encodes the other form identically.
    #[test]
    fn test_bpe_nfc_composed_decomposed_parity() {
        let composed = "café"; // U+00E9
        let decomposed = "cafe\u{0301}"; // e + combining acute

        let config = TokenizerConfig::bpe()
            .with_vocab_size(300)
            .with_min_frequency(1)
            .with_normalization(Normalization::NFC);
        let mut tokenizer = BPETokenizer::new(config);
        tokenizer.train(&[composed]).expect("operation should succeed");

        let ids_composed = tokenizer.encode(composed).expect("encoding should succeed");
        let ids_decomposed = tokenizer.encode(decomposed).expect("encoding should succeed");

        assert_eq!(
            ids_composed, ids_decomposed,
            "NFC must map composed and decomposed café to identical token IDs"
        );

        let decoded = tokenizer.decode(&ids_composed).expect("decoding should succeed");
        assert_eq!(decoded, composed, "NFC round-trip must recover composed form");
    }

    // Without NFC, composed and decomposed café MUST diverge — this is the
    // exact drift INV-TOK-003 is defending against at training/inference boundary.
    #[test]
    fn test_bpe_without_nfc_composed_decomposed_diverge() {
        let composed = "café";
        let decomposed = "cafe\u{0301}";

        let config = TokenizerConfig::bpe()
            .with_vocab_size(300)
            .with_min_frequency(1)
            .with_normalization(Normalization::None);
        let mut tokenizer = BPETokenizer::new(config);
        tokenizer.train(&[composed]).expect("operation should succeed");

        let ids_composed = tokenizer.encode(composed).expect("encoding should succeed");
        let ids_decomposed = tokenizer.encode(decomposed).expect("encoding should succeed");

        assert_ne!(
            ids_composed, ids_decomposed,
            "Without NFC, composed and decomposed café MUST diverge (falsification witness for INV-TOK-003)"
        );
    }

    // C-PRETOK-BIN GATE-PRETOK-003 prerequisite: reloading a trained
    // tokenizer from its emitted vocab.json + merges.txt MUST yield
    // byte-identical encodings vs the original in-memory tokenizer.
    // Any drift here means `apr tokenize encode-corpus` (which loads
    // via from_vocab_merges) would produce shards that differ from
    // what the tokenizer intended — ShardBatchIter round-trip fails.
    #[test]
    fn test_bpe_from_vocab_merges_roundtrip() {
        use std::fmt::Write;
        let config = TokenizerConfig::bpe()
            .with_vocab_size(400)
            .with_min_frequency(1)
            .with_normalization(Normalization::NFC);
        let mut original = BPETokenizer::new(config.clone());
        let corpus = vec!["def hello():\n    return 1\n", "def world():\n    return 2\n"];
        original.train(&corpus).expect("training should succeed");

        let tmp = std::env::temp_dir().join(format!(
            "bpe_roundtrip_{}_{}",
            std::process::id(),
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()
        ));
        std::fs::create_dir_all(&tmp).unwrap();
        let vocab_path = tmp.join("vocab.json");
        let merges_path = tmp.join("merges.txt");

        let mut entries: Vec<(&String, &TokenId)> = original.vocab().iter().collect();
        entries.sort_by_key(|(_, id)| *id);
        let ordered: serde_json::Map<String, serde_json::Value> = entries
            .into_iter()
            .map(|(k, v)| (k.clone(), serde_json::Value::Number((*v).into())))
            .collect();
        let vocab_json = serde_json::to_string_pretty(&ordered).unwrap();
        std::fs::write(&vocab_path, vocab_json).unwrap();

        let mut merges_content = String::from("#version: 0.2\n");
        for (left, right) in original.merges() {
            writeln!(merges_content, "{left} {right}").unwrap();
        }
        std::fs::write(&merges_path, merges_content).unwrap();

        let reloaded = BPETokenizer::from_vocab_merges(
            vocab_path.to_str().unwrap(),
            merges_path.to_str().unwrap(),
            config,
        )
        .expect("from_vocab_merges should succeed");

        assert_eq!(reloaded.vocab_size(), original.vocab_size(), "reloaded vocab size must match");

        for text in &corpus {
            let original_ids = original.encode(text).expect("original encode");
            let reloaded_ids = reloaded.encode(text).expect("reloaded encode");
            assert_eq!(
                original_ids, reloaded_ids,
                "reloaded encoding must byte-equal original encoding for {text:?}"
            );
        }

        let _ = std::fs::remove_dir_all(&tmp);
    }

    // Negative: from_vocab_merges must reject a merges.txt with a merged
    // token not present in vocab.json — that's a corrupt pair, and encoding
    // would silently emit <unk> instead of the intended token.
    #[test]
    fn test_bpe_from_vocab_merges_rejects_orphan_merge() {
        let tmp = std::env::temp_dir().join(format!(
            "bpe_orphan_{}_{}",
            std::process::id(),
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()
        ));
        std::fs::create_dir_all(&tmp).unwrap();
        let vocab_path = tmp.join("vocab.json");
        let merges_path = tmp.join("merges.txt");

        std::fs::write(&vocab_path, r#"{"<unk>": 0, "aa": 1, "bb": 2}"#).unwrap();
        std::fs::write(&merges_path, "#version: 0.2\naa bb\n").unwrap();

        let result = BPETokenizer::from_vocab_merges(
            vocab_path.to_str().unwrap(),
            merges_path.to_str().unwrap(),
            TokenizerConfig::bpe(),
        );

        assert!(
            result.is_err(),
            "from_vocab_merges must reject merges.txt with merged token not in vocab.json"
        );
        let err_msg = format!("{:?}", result.unwrap_err());
        assert!(
            err_msg.contains("aabb"),
            "error should name the offending merged token, got: {err_msg}"
        );

        let _ = std::fs::remove_dir_all(&tmp);
    }
}

#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(50))]

        #[test]
        fn prop_bpe_encode_produces_valid_ids(text in "[a-zA-Z ]{1,20}") {
            let config = TokenizerConfig::bpe()
                .with_vocab_size(300)
                .with_min_frequency(1);
            let mut tokenizer = BPETokenizer::new(config);
            tokenizer.train(&[&text]).expect("operation should succeed");

            let encoded = tokenizer.encode(&text).expect("encoding should succeed");

            for id in encoded {
                prop_assert!(tokenizer.id_to_token(id).is_some());
            }
        }

        #[test]
        fn prop_vocab_size_bounded(target_size in 261usize..500) {
            let config = TokenizerConfig::bpe()
                .with_vocab_size(target_size)
                .with_min_frequency(1);
            let mut tokenizer = BPETokenizer::new(config);

            let corpus = vec!["hello world hello world test test"];
            tokenizer.train(&corpus).expect("operation should succeed");

            prop_assert!(tokenizer.vocab_size() <= target_size);
        }
    }
}
