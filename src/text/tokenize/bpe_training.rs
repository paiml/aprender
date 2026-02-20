impl BpeTokenizer {
    /// Train a BPE tokenizer on the given corpus.
    ///
    /// # Arguments
    ///
    /// * `corpus` - Slice of text documents to train on
    /// * `vocab_size` - Target vocabulary size (including special tokens)
    ///
    /// # Returns
    ///
    /// * `Ok(BpeTokenizer)` - Trained tokenizer
    /// * `Err(AprenderError)` - If `vocab_size` is too small
    ///
    /// # Examples
    ///
    /// ```
    /// use aprender::text::tokenize::BpeTokenizer;
    ///
    /// let corpus = vec!["hello world", "hello there", "world wide web"];
    /// let tokenizer = BpeTokenizer::train(&corpus, 100).expect("training should succeed");
    ///
    /// assert!(tokenizer.vocab_size() >= 26); // At least all letters
    /// ```
    pub fn train(corpus: &[&str], vocab_size: usize) -> Result<Self, AprenderError> {
        Self::train_with_special_tokens(corpus, vocab_size, SpecialTokens::default())
    }

    /// Train BPE with custom special tokens.
    ///
    /// # Arguments
    ///
    /// * `corpus` - Slice of text documents to train on
    /// * `vocab_size` - Target vocabulary size (including special tokens)
    /// * `special_tokens` - Custom special tokens configuration
    pub fn train_with_special_tokens(
        corpus: &[&str],
        vocab_size: usize,
        special_tokens: SpecialTokens,
    ) -> Result<Self, AprenderError> {
        if vocab_size < 10 {
            return Err(AprenderError::InvalidHyperparameter {
                param: "vocab_size".to_string(),
                value: vocab_size.to_string(),
                constraint: "must be at least 10".to_string(),
            });
        }

        let end_of_word = "</w>".to_string();

        // Initialize vocab with special tokens
        let mut vocab: HashMap<String, u32> = HashMap::new();
        let mut next_id: u32 = 0;

        // Add special tokens
        vocab.insert(special_tokens.unk.clone(), next_id);
        next_id += 1;

        if let Some(ref bos) = special_tokens.bos {
            vocab.insert(bos.clone(), next_id);
            next_id += 1;
        }
        if let Some(ref eos) = special_tokens.eos {
            vocab.insert(eos.clone(), next_id);
            next_id += 1;
        }
        if let Some(ref pad) = special_tokens.pad {
            vocab.insert(pad.clone(), next_id);
            next_id += 1;
        }

        // Count character frequencies and build initial word splits
        let mut word_freqs: HashMap<String, usize> = HashMap::new();
        for doc in corpus {
            for word in doc.split_whitespace() {
                *word_freqs.entry(word.to_string()).or_insert(0) += 1;
            }
        }

        // Convert words to character sequences with end-of-word marker
        // word_splits: word -> (frequency, character sequence)
        let mut word_splits: HashMap<String, (usize, Vec<String>)> = HashMap::new();
        for (word, freq) in &word_freqs {
            let mut chars: Vec<String> = word.chars().map(|c| c.to_string()).collect();
            if !chars.is_empty() {
                // Add end-of-word marker to last character
                if let Some(last) = chars.last_mut() {
                    last.push_str(&end_of_word);
                }
            }
            word_splits.insert(word.clone(), (*freq, chars));
        }

        // Add all characters to vocab
        for (_, splits) in word_splits.values() {
            for token in splits {
                if !vocab.contains_key(token) {
                    vocab.insert(token.clone(), next_id);
                    next_id += 1;
                }
            }
        }

        // Iteratively merge most frequent pairs
        let mut merges: Vec<(String, String)> = Vec::new();

        while vocab.len() < vocab_size {
            // Count pair frequencies
            let mut pair_freqs: HashMap<(String, String), usize> = HashMap::new();
            for (freq, splits) in word_splits.values() {
                if splits.len() < 2 {
                    continue;
                }
                for window in splits.windows(2) {
                    let pair = (window[0].clone(), window[1].clone());
                    *pair_freqs.entry(pair).or_insert(0) += freq;
                }
            }

            // Find most frequent pair
            let best_pair = pair_freqs
                .iter()
                .max_by_key(|(_, freq)| *freq)
                .map(|(pair, _)| pair.clone());

            let Some((left, right)) = best_pair else {
                break; // No more pairs to merge
            };

            // Create merged token
            let merged = format!("{left}{right}");

            // Add to vocab
            if !vocab.contains_key(&merged) {
                vocab.insert(merged.clone(), next_id);
                next_id += 1;
            }

            // Record merge rule
            merges.push((left.clone(), right.clone()));

            // Apply merge to all word splits
            for (_, splits) in word_splits.values_mut() {
                let mut i = 0;
                while i < splits.len().saturating_sub(1) {
                    if splits[i] == left && splits[i + 1] == right {
                        merged.clone_into(&mut splits[i]);
                        splits.remove(i + 1);
                    } else {
                        i += 1;
                    }
                }
            }
        }

        // Build inverse vocab
        let inverse_vocab: HashMap<u32, String> =
            vocab.iter().map(|(k, v)| (*v, k.clone())).collect();

        Ok(Self {
            vocab,
            inverse_vocab,
            merges,
            special_tokens,
            end_of_word,
        })
    }

    /// Create a BPE tokenizer from pre-built vocabulary and merges.
    ///
    /// # Arguments
    ///
    /// * `vocab` - Token to ID mapping
    /// * `merges` - Ordered list of merge rules
    #[must_use]
    pub fn from_vocab(vocab: HashMap<String, u32>, merges: Vec<(String, String)>) -> Self {
        let inverse_vocab: HashMap<u32, String> =
            vocab.iter().map(|(k, v)| (*v, k.clone())).collect();
        Self {
            vocab,
            inverse_vocab,
            merges,
            special_tokens: SpecialTokens::default(),
            end_of_word: "</w>".to_string(),
        }
    }

    /// Load a BPE tokenizer from `HuggingFace` vocab.json and merges.txt files.
    ///
    /// This is the standard format used by GPT-2, Whisper, and many other models.
    ///
    /// # Arguments
    ///
    /// * `vocab_json` - JSON content of vocab.json (token -> id mapping)
    /// * `merges_txt` - Content of merges.txt (one merge per line)
    ///
    /// # Returns
    ///
    /// * `Ok(BpeTokenizer)` - Loaded tokenizer
    /// * `Err(AprenderError)` - If parsing fails
    ///
    /// # Examples
    ///
    /// ```
    /// use aprender::text::tokenize::BpeTokenizer;
    ///
    /// let vocab_json = r#"{"hello": 0, "world": 1, "<|endoftext|>": 2}"#;
    /// let merges_txt = "h e\nhe l\nhel lo";
    ///
    /// let tokenizer = BpeTokenizer::from_huggingface(vocab_json, merges_txt)
    ///     .expect("loading should succeed");
    /// assert_eq!(tokenizer.vocab_size(), 3);
    /// ```
    ///
    /// # Format Details
    ///
    /// The vocab.json file is a JSON object mapping tokens to IDs:
    /// ```json
    /// {"hello": 0, "world": 1, "<|endoftext|>": 50256}
    /// ```
    ///
    /// The merges.txt file contains one merge rule per line:
    /// ```text
    /// #version: 0.2
    /// h e
    /// he l
    /// hel lo
    /// ```
    ///
    /// # References
    ///
    /// - Sennrich et al. (2016): Neural Machine Translation of Rare Words with Subword Units
    /// - `HuggingFace` Tokenizers: <https://huggingface.co/docs/tokenizers>
    pub fn from_huggingface(vocab_json: &str, merges_txt: &str) -> Result<Self, AprenderError> {
        // Parse vocab.json - simple JSON parsing without external dependency
        let vocab = Self::parse_vocab_json(vocab_json)?;

        // Parse merges.txt
        let merges = Self::parse_merges_txt(merges_txt);

        // Detect end-of-word marker from vocab (GPT-2 uses "Ġ", others use "</w>")
        let end_of_word = if vocab.keys().any(|k| k.contains("Ġ")) {
            "Ġ".to_string()
        } else {
            "</w>".to_string()
        };

        // Detect special tokens from vocab
        let unk = vocab
            .keys()
            .find(|k| k.contains("unk") || k.contains("UNK"))
            .cloned()
            .unwrap_or_else(|| "<unk>".to_string());

        let eos = vocab
            .keys()
            .find(|k| k.contains("endoftext") || k.contains("</s>") || k.contains("eos"))
            .cloned();

        let bos = vocab
            .keys()
            .find(|k| k.contains("startoftext") || k.contains("<s>") || k.contains("bos"))
            .cloned();

        let pad = vocab
            .keys()
            .find(|k| k.contains("pad") || k.contains("PAD"))
            .cloned();

        let special_tokens = SpecialTokens { unk, bos, eos, pad };

        let inverse_vocab: HashMap<u32, String> =
            vocab.iter().map(|(k, v)| (*v, k.clone())).collect();

        Ok(Self {
            vocab,
            inverse_vocab,
            merges,
            special_tokens,
            end_of_word,
        })
    }

    /// Parse vocab.json content into a `HashMap`.
    ///
    /// Simple JSON parsing without external dependencies. Handles basic JSON format:
    /// `{"token1": 0, "token2": 1, ...}`
    fn parse_vocab_json(json: &str) -> Result<HashMap<String, u32>, AprenderError> {
        let json = json.trim();
        if !json.starts_with('{') || !json.ends_with('}') {
            return Err(AprenderError::Serialization(
                "Invalid vocab.json: must be a JSON object".to_string(),
            ));
        }

        let mut vocab = HashMap::new();
        let content = &json[1..json.len() - 1]; // Remove { and }

        if content.trim().is_empty() {
            return Ok(vocab);
        }

        // Parse key-value pairs
        let chars = content.chars();
        let mut in_string = false;
        let mut escape_next = false;
        let mut current_key = String::new();
        let mut current_value = String::new();
        let mut parsing_key = true;

        for c in chars {
            if escape_next {
                if parsing_key {
                    current_key.push(c);
                } else {
                    current_value.push(c);
                }
                escape_next = false;
                continue;
            }

            match c {
                '\\' => {
                    escape_next = true;
                }
                '"' => {
                    in_string = !in_string;
                }
                ':' if !in_string => {
                    parsing_key = false;
                }
                ',' if !in_string => {
                    // End of pair
                    let key = current_key.trim().to_string();
                    let value: u32 = current_value.trim().parse().map_err(|_| {
                        AprenderError::Serialization(format!(
                            "Invalid token ID for '{key}': '{current_value}'"
                        ))
                    })?;

                    if !key.is_empty() {
                        vocab.insert(key, value);
                    }

                    current_key.clear();
                    current_value.clear();
                    parsing_key = true;
                }
                _ if in_string => {
                    if parsing_key {
                        current_key.push(c);
                    } else {
                        current_value.push(c);
                    }
                }
                _ if !in_string && !c.is_whitespace() => {
                    if !parsing_key {
                        current_value.push(c);
                    }
                }
                _ => {}
            }
        }

        // Handle last pair
        let key = current_key.trim().to_string();
        if !key.is_empty() && !current_value.is_empty() {
            let value: u32 = current_value.trim().parse().map_err(|_| {
                AprenderError::Serialization(format!(
                    "Invalid token ID for '{key}': '{current_value}'"
                ))
            })?;
            vocab.insert(key, value);
        }

        Ok(vocab)
    }

    /// Parse merges.txt content into a list of merge rules.
    ///
    /// The format is one merge per line: "token1 token2"
    /// Lines starting with # are treated as comments.
    fn parse_merges_txt(content: &str) -> Vec<(String, String)> {
        let mut merges = Vec::new();

        for line in content.lines() {
            let line = line.trim();

            // Skip empty lines and comments
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            // Split on first space
            let parts: Vec<&str> = line.splitn(2, ' ').collect();
            if parts.len() != 2 {
                continue; // Skip malformed lines
            }

            let left = parts[0].to_string();
            let right = parts[1].to_string();

            if !left.is_empty() && !right.is_empty() {
                merges.push((left, right));
            }
        }

        merges
    }

    /// Check if a token is a special token (UNK, BOS, EOS, PAD).
    #[must_use]
    pub fn is_special_token(&self, token: &str) -> bool {
        token == self.special_tokens.unk
            || self.special_tokens.bos.as_deref() == Some(token)
            || self.special_tokens.eos.as_deref() == Some(token)
            || self.special_tokens.pad.as_deref() == Some(token)
    }

    /// Get the EOS token if configured.
    #[must_use]
    pub fn eos_token(&self) -> Option<&str> {
        self.special_tokens.eos.as_deref()
    }

    /// Get the BOS token if configured.
    #[must_use]
    pub fn bos_token(&self) -> Option<&str> {
        self.special_tokens.bos.as_deref()
    }

    /// Get the UNK token.
    #[must_use]
    pub fn unk_token(&self) -> &str {
        &self.special_tokens.unk
    }

    /// Get the end-of-word marker used by this tokenizer.
    ///
    /// Returns `</w>` for standard BPE or "Ġ" for GPT-2 style tokenizers.
    #[must_use]
    pub fn end_of_word_marker(&self) -> &str {
        &self.end_of_word
    }
}
