
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
                    match model.load_embedded_bpe_tokenizer() {
                        Some(tokenizer) => {
                            return Some(tokenizer.encode(text));
                        },
                        None => {
                            // PMAT-172: FAIL FAST - Do not fall back to external tokenizers
                            eprintln!("\n[PMAT-172] ERROR: APR file missing embedded tokenizer.");
                            eprintln!("           APR format requires self-contained tokenizer.");
                            eprintln!(
                                "           Re-convert with: apr convert <source>.gguf -o {}",
                                model_path.display()
                            );
                            eprintln!("           Or use the original GGUF file directly.\n");
                            return None;
                        },
                    }
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

        let json: serde_json::Value = match serde_json::from_str(&content) {
            Ok(j) => j,
            Err(e) => {
                eprintln!("[PMAT-172] Error parsing tokenizer.json: {}", e);
                return None;
            },
        };

        // Extract vocabulary (token -> id)
        let vocab_obj = json.get("model")?.get("vocab")?;
        let vocab_map = vocab_obj.as_object()?;
        let token_to_id: HashMap<String, u32> = vocab_map
            .iter()
            .filter_map(|(token, id)| Some((token.clone(), id.as_u64()? as u32)))
            .collect();

        // F-REGR-231: Extract added_tokens (special tokens like <|im_start|>, <|im_end|>)
        let special_tokens: HashMap<String, u32> = json
            .get("added_tokens")
            .and_then(|arr| arr.as_array())
            .map(|tokens| {
                tokens
                    .iter()
                    .filter_map(|t| {
                        let content = t.get("content")?.as_str()?;
                        let id = t.get("id")?.as_u64()? as u32;
                        Some((content.to_string(), id))
                    })
                    .collect()
            })
            .unwrap_or_default();

        // Extract merges
        let merges = json.get("model")?.get("merges")?.as_array()?;
        let merge_rules: Vec<(String, String)> = merges
            .iter()
            .filter_map(|m| {
                let s = m.as_str()?;
                let parts: Vec<&str> = s.splitn(2, ' ').collect();
                if parts.len() == 2 {
                    Some((parts[0].to_string(), parts[1].to_string()))
                } else {
                    None
                }
            })
            .collect();

        let tokens = bpe_encode(text, &token_to_id, &merge_rules, &special_tokens);
        Some(tokens)
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

        Some(BpeTokenizer {
            token_to_id,
            id_to_token,
            merge_rules: merges,
            bos_id,
            eos_id,
            special_tokens,
        })
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
        let json: serde_json::Value = serde_json::from_str(&content).ok()?;

        // Extract vocabulary
        let vocab_obj = json.get("model")?.get("vocab")?;
        let vocab_map = vocab_obj.as_object()?;

        let mut token_to_id: HashMap<String, u32> = HashMap::new();
        let mut id_to_token: Vec<String> = Vec::new();

        let mut vocab_vec: Vec<(String, u32)> = vocab_map
            .iter()
            .filter_map(|(token, id)| Some((token.clone(), id.as_u64()? as u32)))
            .collect();
        vocab_vec.sort_by_key(|(_, id)| *id);

        for (token, id) in vocab_vec {
            token_to_id.insert(token.clone(), id);
            // Pad id_to_token if needed
            while id_to_token.len() <= id as usize {
                id_to_token.push(String::new());
            }
            id_to_token[id as usize] = token;
        }

        // Extract merges
        let merges = json.get("model")?.get("merges")?.as_array()?;
        let merge_rules: Vec<(String, String)> = merges
            .iter()
            .filter_map(|m| {
                let s = m.as_str()?;
                let parts: Vec<&str> = s.splitn(2, ' ').collect();
                if parts.len() == 2 {
                    Some((parts[0].to_string(), parts[1].to_string()))
                } else {
                    None
                }
            })
            .collect();

        // GH-189: Extract ALL added_tokens as special tokens for atomic tokenization
        let mut bos_id = None;
        let mut eos_id = None;
        let mut special_tokens: HashMap<String, u32> = HashMap::new();

        if let Some(added_tokens) = json.get("added_tokens").and_then(|v| v.as_array()) {
            for token in added_tokens {
                let content = token.get("content").and_then(|v| v.as_str());
                let id = token
                    .get("id")
                    .and_then(serde_json::Value::as_u64)
                    .map(|v| v as u32);

                if let (Some(content), Some(id)) = (content, id) {
                    // Add ALL added_tokens to special_tokens map for atomic tokenization
                    special_tokens.insert(content.to_string(), id);

                    // Also track bos/eos specifically
                    if content == "<|endoftext|>" || content == "</s>" || content == "<eos>" {
                        eos_id = Some(id);
                    }
                    if content == "<s>" || content == "<bos>" {
                        bos_id = Some(id);
                    }
                }
            }
        }

        eprintln!(
            "[GH-189] Loaded tokenizer from {}: {} special tokens",
            tokenizer_path.display(),
            special_tokens.len()
        );

        Some(BpeTokenizer {
            token_to_id,
            id_to_token,
            merge_rules,
            bos_id,
            eos_id,
            special_tokens,
        })
    }
}

include!("loading_mmap.rs");
include!("forward.rs");
