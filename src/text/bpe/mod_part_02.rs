
impl BpeTokenizer {
    /// Create a new BPE tokenizer with given config
    #[must_use]
    pub fn new(config: BpeConfig) -> Self {
        let (byte_encoder, byte_decoder) = bytes_to_unicode();

        Self {
            config,
            vocab: HashMap::new(),
            id_to_token: HashMap::new(),
            merges: Vec::new(),
            merge_ranks: HashMap::new(),
            special_tokens: HashMap::new(),
            byte_encoder,
            byte_decoder,
        }
    }

    /// Create tokenizer with GPT-2 base vocabulary (stub)
    ///
    /// # Note
    /// Real implementation requires loading vocabulary files.
    #[must_use]
    pub fn gpt2_base() -> Self {
        let config = BpeConfig::gpt2();
        let mut tokenizer = Self::new(config);

        // Add basic ASCII characters as initial vocab
        for i in 0..=255u8 {
            if let Some(&c) = tokenizer.byte_encoder.get(&i) {
                let token = c.to_string();
                let id = u32::from(i);
                tokenizer.vocab.insert(token.clone(), id);
                tokenizer.id_to_token.insert(id, token);
            }
        }

        // Add special tokens
        tokenizer.add_special_token("<|endoftext|>", 50256);

        tokenizer
    }

    /// Add a special token
    pub fn add_special_token(&mut self, token: &str, id: u32) {
        self.special_tokens.insert(token.to_string(), id);
        self.vocab.insert(token.to_string(), id);
        self.id_to_token.insert(id, token.to_string());
    }

    /// Add a merge rule
    pub fn add_merge(&mut self, first: &str, second: &str) {
        let rank = self.merges.len();
        let rule = MergeRule::new(first, second);
        self.merge_ranks
            .insert((first.to_string(), second.to_string()), rank);
        self.merges.push(rule);
    }

    /// Get vocabulary size
    #[must_use]
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// Get token ID for a token
    #[must_use]
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.vocab.get(token).copied()
    }

    /// Get token for an ID
    #[must_use]
    pub fn id_to_token(&self, id: u32) -> Option<&str> {
        self.id_to_token.get(&id).map(String::as_str)
    }

    /// Check if token is a special token
    #[must_use]
    pub fn is_special_token(&self, token: &str) -> bool {
        self.special_tokens.contains_key(token)
    }

    /// Encode text to token IDs.
    ///
    /// # Arguments
    /// * `text` - Text to encode
    ///
    /// # Returns
    /// Vector of token IDs
    #[must_use]
    pub fn encode(&self, text: &str) -> Vec<u32> {
        if text.is_empty() {
            return vec![];
        }

        let mut ids = Vec::new();

        // PMAT-114: Handle special tokens FIRST before BPE tokenization
        // This ensures tokens like <|im_start|> are encoded as single tokens (151644)
        // rather than being split into characters (27, 91, 318, 4906, 91, 29)
        let segments = self.split_on_special_tokens(text);

        for segment in segments {
            if let Some(&special_id) = self.special_tokens.get(&segment) {
                // Special token - output its ID directly
                ids.push(special_id);
            } else {
                // Regular text - apply BPE tokenization
                let segment_text = if self.config.add_prefix_space
                    && !segment.starts_with(' ')
                    && ids.is_empty()
                {
                    format!(" {segment}")
                } else {
                    segment
                };

                for word in self.pre_tokenize(&segment_text) {
                    let byte_word = self.bytes_to_bpe_tokens(&word);
                    let tokens = self.bpe(&byte_word);

                    for token in tokens {
                        if let Some(&id) = self.vocab.get(&token) {
                            ids.push(id);
                        } else if let Some(&id) = self.vocab.get(&self.config.unk_token) {
                            ids.push(id);
                        }
                    }
                }
            }
        }

        ids
    }

    /// Split text on special tokens while preserving them as separate segments.
    /// Returns vec of segments where special tokens are their own elements.
    fn split_on_special_tokens(&self, text: &str) -> Vec<String> {
        if self.special_tokens.is_empty() {
            return vec![text.to_string()];
        }

        // Sort special tokens by length (longest first) to avoid partial matches
        let mut sorted_tokens: Vec<_> = self.special_tokens.keys().collect();
        sorted_tokens.sort_by_key(|t| std::cmp::Reverse(t.len()));

        let mut result = Vec::new();
        let mut remaining = text;

        while !remaining.is_empty() {
            // Find the earliest special token occurrence
            let mut earliest_match: Option<(usize, &str)> = None;

            for token in &sorted_tokens {
                if let Some(pos) = remaining.find(token.as_str()) {
                    match earliest_match {
                        None => earliest_match = Some((pos, token)),
                        Some((prev_pos, _)) if pos < prev_pos => {
                            earliest_match = Some((pos, token));
                        }
                        _ => {}
                    }
                }
            }

            match earliest_match {
                Some((pos, token)) => {
                    // Add text before the special token (if any)
                    if pos > 0 {
                        result.push(remaining[..pos].to_string());
                    }
                    // Add the special token itself
                    result.push(token.to_string());
                    // Continue with remaining text
                    remaining = &remaining[pos + token.len()..];
                }
                None => {
                    // No more special tokens - add remaining text
                    result.push(remaining.to_string());
                    break;
                }
            }
        }

        result
    }

    /// Decode token IDs to text.
    ///
    /// # Arguments
    /// * `ids` - Token IDs to decode
    ///
    /// # Returns
    /// Decoded text
    #[must_use]
    pub fn decode(&self, ids: &[u32]) -> String {
        if ids.is_empty() {
            return String::new();
        }

        let mut text = String::new();

        for &id in ids {
            if let Some(token) = self.id_to_token.get(&id) {
                // Skip special tokens in output
                if !self.special_tokens.contains_key(token) {
                    text.push_str(token);
                }
            }
        }

        // Convert byte tokens back to UTF-8
        self.bpe_tokens_to_bytes(&text)
    }

    /// Encode text to token IDs with error handling.
    ///
    /// # Errors
    /// Returns error if encoding fails.
    pub fn encode_checked(&self, text: &str) -> Result<Vec<u32>> {
        Ok(self.encode(text))
    }

    /// Decode token IDs to text with error handling.
    ///
    /// # Errors
    /// Returns error if decoding fails.
    pub fn decode_checked(&self, ids: &[u32]) -> Result<String> {
        Ok(self.decode(ids))
    }

    /// Pre-tokenize text into words
    fn pre_tokenize(&self, text: &str) -> Vec<String> {
        // Simple regex-like pattern: split on whitespace, keeping punctuation
        // Future: Use self.config for model-specific pre-tokenization rules
        let _ = &self.config;
        let mut words = Vec::new();
        let mut current = String::new();

        for c in text.chars() {
            if c.is_whitespace() {
                if !current.is_empty() {
                    words.push(current.clone());
                    current.clear();
                }
                // Include the space as part of next word
                current.push(c);
            } else {
                current.push(c);
            }
        }

        if !current.is_empty() {
            words.push(current);
        }

        words
    }

    /// Convert string to byte-encoded tokens
    fn bytes_to_bpe_tokens(&self, word: &str) -> Vec<String> {
        word.bytes()
            .map(|b| {
                self.byte_encoder
                    .get(&b)
                    .map_or_else(|| format!("?{b}"), |&c| c.to_string())
            })
            .collect()
    }

    /// Convert byte-encoded tokens back to string
    fn bpe_tokens_to_bytes(&self, text: &str) -> String {
        let bytes: Vec<u8> = text
            .chars()
            .filter_map(|c| self.byte_decoder.get(&c).copied())
            .collect();

        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// Apply BPE merges to token list
    fn bpe(&self, tokens: &[String]) -> Vec<String> {
        if tokens.len() <= 1 {
            return tokens.to_vec();
        }

        let mut result = tokens.to_vec();

        loop {
            // Find best merge (lowest rank)
            let mut best_merge: Option<(usize, usize)> = None;
            let mut best_rank = usize::MAX;

            for i in 0..result.len().saturating_sub(1) {
                let pair = (result[i].clone(), result[i + 1].clone());
                if let Some(&rank) = self.merge_ranks.get(&pair) {
                    if rank < best_rank {
                        best_rank = rank;
                        best_merge = Some((i, rank));
                    }
                }
            }

            // Apply best merge or stop
            match best_merge {
                Some((idx, _)) => {
                    let merged = format!("{}{}", result[idx], result[idx + 1]);
                    result.splice(idx..=idx + 1, std::iter::once(merged));
                }
                None => break,
            }
        }

        result
    }
}

impl Default for BpeTokenizer {
    fn default() -> Self {
        Self::new(BpeConfig::default())
    }
}

// ============================================================================
// Qwen2 BPE Tokenizer
// ============================================================================

/// Qwen2-specific BPE tokenizer with chat template support.
///
/// Extends the base BPE tokenizer with Qwen2's special tokens and
/// chat formatting conventions.
///
/// # Example
///
/// ```rust
/// use aprender::text::bpe::Qwen2BpeTokenizer;
///
/// let tokenizer = Qwen2BpeTokenizer::new();
///
/// // Check special tokens
/// assert!(tokenizer.is_eos(151645)); // <|im_end|>
///
/// // Format a chat message
/// let formatted = tokenizer.format_chat("user", "Hello, world!");
/// assert!(formatted.contains("<|im_start|>user"));
/// ```
#[derive(Debug, Clone)]
pub struct Qwen2BpeTokenizer {
    /// Base tokenizer
    base: BpeTokenizer,
    /// Special token IDs
    im_start_id: u32,
    im_end_id: u32,
    endoftext_id: u32,
}
