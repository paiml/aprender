
impl AprV2Model {

    /// Encode text to token IDs using embedded BPE tokenizer (PMAT-172: Fail-Fast)
    ///
    /// APR files MUST have embedded tokenizer. NO FALLBACK to external files.
    /// This prevents Silent Failure Recovery where wrong tokenizer produces garbage.
    ///
    /// # Design Principle
    ///
    /// APR format is designed to be ONE self-contained file. If the embedded
    /// tokenizer is missing, the APR file is BROKEN and should be re-converted.
    ///
    /// # Returns
    ///
    /// - `Some(tokens)` if APR has embedded tokenizer
    /// - `None` if file doesn't exist or isn't APR format
    ///
    /// # Panics
    ///
    /// Prints error and returns None if APR is missing embedded tokenizer.
    /// This is intentional - we want users to see the error, not garbage output.
    pub fn encode_text(model_path: &Path, text: &str) -> Option<Vec<u32>> {
        // Validate model path exists
        if !model_path.exists() {
            eprintln!(
                "[PMAT-172] Error: Model file not found: {}",
                model_path.display()
            );
            return None;
        }

        // PMAT-172: APR files MUST use embedded tokenizer - NO FALLBACK
        if model_path.extension().is_some_and(|e| e == "apr") {
            match Self::load(model_path) {
                Ok(model) => {
                    // Try BPE tokenizer first
                    if let Some(tokenizer) = model.load_embedded_bpe_tokenizer() {
                        return Some(tokenizer.encode(text));
                    }
                    // GH-366: Try SentencePiece tokenizer (Unigram models)
                    if let Some(tokenizer) = model.load_embedded_sentencepiece_tokenizer() {
                        return Some(tokenizer.encode(text));
                    }
                    // PMAT-172: FAIL FAST - No embedded tokenizer found
                    eprintln!("\n[PMAT-172] ERROR: APR file missing embedded tokenizer.");
                    eprintln!("           APR format requires self-contained tokenizer.");
                    eprintln!(
                        "           Re-convert with: apr convert <source>.gguf -o {}",
                        model_path.display()
                    );
                    eprintln!("           Or use the original GGUF file directly.\n");
                    return None;
                },
                Err(e) => {
                    eprintln!("[PMAT-172] Error loading APR file: {}", e);
                    return None;
                },
            }
        }

        // For non-APR files (SafeTensors), use sibling tokenizer.json ONLY
        // NO fallback to HuggingFace cache (PMAT-172: removed Silent Failure Recovery)
        // GAP-UX-002: Try hash-prefixed first, then plain filename
        let tokenizer_path = match find_sibling_file(model_path, "tokenizer.json") {
            Some(path) => path,
            None => {
                eprintln!(
                    "\n[PMAT-172] ERROR: No tokenizer found for {}.",
                    model_path.display()
                );
                let stem = model_path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("model");
                eprintln!(
                    "           Expected sibling file: {}.tokenizer.json or tokenizer.json",
                    stem
                );
                eprintln!(
                    "           For SafeTensors models, tokenizer.json must be in same directory.\n"
                );
                return None;
            },
        };

        let content = match fs::read_to_string(&tokenizer_path) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("[PMAT-172] Error reading tokenizer.json: {}", e);
                return None;
            },
        };

        // #3742: the same loader as every other tokenizer.json consumer, so the model's own
        // pre-tokenizer and ranked merges encode the prompt (the canonical byte-level BPE)
        // whenever they are implemented. This branch ran the legacy merge loop over the whole
        // text, which is not the model's tokenization.
        let json: serde_json::Value = match serde_json::from_str(&content) {
            Ok(j) => j,
            Err(e) => {
                eprintln!("[PMAT-172] Error parsing tokenizer.json: {}", e);
                return None;
            },
        };
        let Some(tokenizer) = Self::load_tokenizer_from_value(&json) else {
            eprintln!(
                "[PMAT-172] Error: {} is not a BPE tokenizer.json (no model.vocab / model.merges)",
                tokenizer_path.display()
            );
            return None;
        };
        Some(tokenizer.encode(text))
    }

    // PMAT-172: Removed find_tokenizer_json_in_cache() — loading a stale
    // HuggingFace cache tokenizer produced garbage output. Now requires
    // explicit tokenizer path or embedded APR vocabulary.

    /// Load tokenizer from embedded APR metadata (GH-156)
    ///
    /// APR files can contain embedded tokenizer data - this is the preferred
    /// way to decode tokens since it doesn't require sibling files.
    ///
    /// Returns a simple decode-only tokenizer (no BPE encoding support).
    pub fn load_embedded_tokenizer(&self) -> Option<SimpleTokenizer> {
        let vocab = self.metadata.get_embedded_vocabulary()?;
        let bos_id = self.metadata.get_embedded_bos_token_id();
        let eos_id = self.metadata.get_embedded_eos_token_id();

        Some(SimpleTokenizer {
            id_to_token: vocab,
            bos_token_id: bos_id,
            eos_token_id: eos_id,
        })
    }

    /// Load a full BPE tokenizer from embedded APR metadata (PMAT-171)
    ///
    /// APR files converted from GGUF can contain both vocabulary AND BPE merge
    /// rules embedded in metadata. This enables standalone encoding without
    /// needing sibling tokenizer.json files.
    ///
    /// Returns `Some(BpeTokenizer)` if both vocab and merges are embedded.
    /// Returns `None` if either is missing (fall back to sibling file).
    pub fn load_embedded_bpe_tokenizer(&self) -> Option<BpeTokenizer> {
        let vocab_list = self.metadata.get_embedded_vocabulary()?;
        let merges = self.metadata.get_embedded_merges()?;

        // Build token_to_id and id_to_token maps
        let mut token_to_id: HashMap<String, u32> = HashMap::new();
        let mut id_to_token: Vec<String> = Vec::with_capacity(vocab_list.len());

        for (id, token) in vocab_list.iter().enumerate() {
            token_to_id.insert(token.clone(), id as u32);
            id_to_token.push(token.clone());
        }

        let bos_id = self.metadata.get_embedded_bos_token_id();
        let eos_id = self.metadata.get_embedded_eos_token_id();

        // GH-189: Extract special tokens from vocabulary for atomic tokenization
        // Special tokens like <|im_start|>, <|im_end|> must not be split by BPE
        let special_tokens = extract_special_tokens_from_vocab(&token_to_id);

        eprintln!(
            "[PMAT-171] Loaded embedded BPE tokenizer: {} vocab, {} merges, {} special tokens",
            id_to_token.len(),
            merges.len(),
            special_tokens.len()
        );
        // #3742: the model's pre-tokenizer and ranked merges, as the GGUF path encodes.
        let canonical =
            crate::apr::canonical_tokenizer::canonical_for_apr(&self.metadata, &id_to_token, &merges);

        Some(BpeTokenizer {
            token_to_id,
            id_to_token,
            merge_rules: merges,
            bos_id,
            eos_id,
            special_tokens,
            canonical,
        })
    }

    /// GH-366: Load a SentencePiece tokenizer from embedded APR metadata
    ///
    /// APR files converted from SafeTensors models with tokenizer.model
    /// contain vocabulary + scores for Unigram/Viterbi encoding.
    ///
    /// Returns `Some(SentencePieceTokenizer)` if vocab and scores are embedded.
    pub fn load_embedded_sentencepiece_tokenizer(&self) -> Option<SentencePieceTokenizer> {
        let vocab_list = self.metadata.get_embedded_vocabulary()?;
        let scores = self.metadata.get_embedded_scores()?;

        if vocab_list.len() != scores.len() {
            eprintln!(
                "[GH-366] Vocab/scores length mismatch: {} vs {}",
                vocab_list.len(),
                scores.len()
            );
            return None;
        }

        let vocab_with_scores: Vec<(String, f32)> = vocab_list
            .into_iter()
            .zip(scores)
            .collect();

        match SentencePieceTokenizer::new(vocab_with_scores, "<unk>") {
            Ok(tokenizer) => {
                eprintln!(
                    "[GH-366] Loaded embedded SentencePiece tokenizer: {} vocab tokens",
                    tokenizer.vocab_size()
                );
                Some(tokenizer)
            }
            Err(e) => {
                eprintln!("[GH-366] Failed to create SentencePiece tokenizer: {e}");
                None
            }
        }
    }

    /// Load a full tokenizer struct from sibling tokenizer.json
    ///
    /// GAP-UX-002: Tries hash-prefixed companion first (`{stem}.tokenizer.json`),
    /// then falls back to non-prefixed (`tokenizer.json`) for backwards compatibility.
    ///
    /// Returns a BpeTokenizer that can be reused for multiple encode/decode calls.
    /// For decode-only operations, prefer `load_embedded_tokenizer()` first.
    pub fn load_tokenizer(model_path: &Path) -> Option<BpeTokenizer> {
        let tokenizer_path = find_sibling_file(model_path, "tokenizer.json")?;
        Self::load_tokenizer_from_path(&tokenizer_path)
    }

    /// Load a BPE tokenizer from an explicit tokenizer.json path
    ///
    /// This is used for loading tokenizers from HuggingFace cache or other locations.
    /// (PMAT-SHOWCASE-TOKENIZER-001)
    pub fn load_tokenizer_from_path(tokenizer_path: &Path) -> Option<BpeTokenizer> {
        if !tokenizer_path.exists() {
            return None;
        }

        let content = fs::read_to_string(tokenizer_path).ok()?;
        let tokenizer = Self::load_tokenizer_from_json(&content)?;
        eprintln!(
            "[GH-189] Loaded tokenizer from {}: {} special tokens",
            tokenizer_path.display(),
            tokenizer.special_tokens.len()
        );
        Some(tokenizer)
    }

    /// Load a tokenizer from the text of a HuggingFace tokenizer.json (#3742: entrenar's
    /// tokenizer reaches the canonical byte-level BPE through here).
    pub fn load_tokenizer_from_json(content: &str) -> Option<BpeTokenizer> {
        Self::load_tokenizer_from_value(&serde_json::from_str(content).ok()?)
    }

    fn load_tokenizer_from_value(json: &serde_json::Value) -> Option<BpeTokenizer> {
        let (token_to_id, id_to_token) = read_vocab(json)?;
        let merge_rules: Vec<(String, String)> = json
            .get("model")?
            .get("merges")?
            .as_array()?
            .iter()
            .filter_map(merge_pair)
            .collect();
        let (added, bos_id, eos_id) = read_added_tokens(json);
        // GH-189: ALL added_tokens are special tokens, for atomic tokenization.
        let special_tokens: HashMap<String, u32> =
            added.iter().map(|(content, id, _)| (content.clone(), *id)).collect();

        // #3742: the tokenizer.json's own pre-tokenizer and ranked merges, when implemented.
        let canonical = crate::apr::canonical_tokenizer::canonical_for_tokenizer_json(
            json,
            &id_to_token,
            &merge_rules,
            &added,
        );

        Some(BpeTokenizer {
            token_to_id,
            id_to_token,
            merge_rules,
            bos_id,
            eos_id,
            special_tokens,
            canonical,
        })
    }
}

/// A tokenizer.json `model.vocab` as (token -> id, id -> token); an id the map skips is an
/// empty string.
fn read_vocab(json: &serde_json::Value) -> Option<(HashMap<String, u32>, Vec<String>)> {
    let mut vocab_vec: Vec<(String, u32)> = json
        .get("model")?
        .get("vocab")?
        .as_object()?
        .iter()
        .filter_map(|(token, id)| Some((token.clone(), id.as_u64()? as u32)))
        .collect();
    vocab_vec.sort_by_key(|(_, id)| *id);

    let mut token_to_id: HashMap<String, u32> = HashMap::new();
    let mut id_to_token: Vec<String> = Vec::new();
    for (token, id) in vocab_vec {
        token_to_id.insert(token.clone(), id);
        if id_to_token.len() <= id as usize {
            id_to_token.resize(id as usize + 1, String::new());
        }
        id_to_token[id as usize] = token;
    }
    Some((token_to_id, id_to_token))
}

/// A tokenizer.json's `added_tokens` as (content, id, special), and the bos and eos ids among
/// them (the last match wins, as before).
fn read_added_tokens(json: &serde_json::Value) -> (Vec<(String, u32, bool)>, Option<u32>, Option<u32>) {
    let mut added = Vec::new();
    let (mut bos_id, mut eos_id) = (None, None);
    for token in json
        .get("added_tokens")
        .and_then(|v| v.as_array())
        .into_iter()
        .flatten()
    {
        let content = token.get("content").and_then(|v| v.as_str());
        let id = token
            .get("id")
            .and_then(serde_json::Value::as_u64)
            .map(|v| v as u32);
        let (Some(content), Some(id)) = (content, id) else {
            continue;
        };
        let special = token
            .get("special")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(true);
        if matches!(content, "<|endoftext|>" | "</s>" | "<eos>") {
            eos_id = Some(id);
        }
        if matches!(content, "<s>" | "<bos>") {
            bos_id = Some(id);
        }
        added.push((content.to_string(), id, special));
    }
    (added, bos_id, eos_id)
}

/// One tokenizer.json merge: the string form `"a b"` or the array form `["a", "b"]` that
/// `tokenizers` 0.20+ writes (#3742). Reading only the string form built a canonical encoder
/// with no merges from every array-form file, without a word.
fn merge_pair(m: &serde_json::Value) -> Option<(String, String)> {
    if let Some(pair) = m.as_array() {
        return match pair.as_slice() {
            [a, b] => Some((a.as_str()?.to_string(), b.as_str()?.to_string())),
            _ => None,
        };
    }
    let (a, b) = m.as_str()?.split_once(' ')?;
    Some((a.to_string(), b.to_string()))
}

include!("loading_mmap.rs");
include!("forward.rs");
