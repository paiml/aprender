
impl SentencePieceTokenizer {
    /// Create a new `SentencePiece` tokenizer
    ///
    /// # Arguments
    ///
    /// * `vocab` - List of (token, score) pairs where score is log probability
    /// * `unk_token` - Unknown token string
    ///
    /// # Errors
    ///
    /// Returns error if vocabulary is empty or unknown token not found
    pub fn new(vocab: Vec<(String, f32)>, unk_token: &str) -> Result<Self> {
        if vocab.is_empty() {
            return Err(RealizarError::UnsupportedOperation {
                operation: "create_sentencepiece_tokenizer".to_string(),
                reason: "Vocabulary cannot be empty".to_string(),
            });
        }

        let mut token_to_id = HashMap::new();
        let mut id_to_token = HashMap::new();
        let mut scores = HashMap::new();

        for (id, (token, score)) in vocab.into_iter().enumerate() {
            let id = u32::try_from(id).map_err(|_| RealizarError::UnsupportedOperation {
                operation: "convert_token_id".to_string(),
                reason: format!("Token ID {id} exceeds u32 limit"),
            })?;
            token_to_id.insert(token.clone(), id);
            id_to_token.insert(id, token.clone());
            scores.insert(token, score);
        }

        let unk_token_id =
            *token_to_id
                .get(unk_token)
                .ok_or_else(|| RealizarError::UnsupportedOperation {
                    operation: "create_sentencepiece_tokenizer".to_string(),
                    reason: format!("Unknown token '{unk_token}' not in vocabulary"),
                })?;

        Ok(Self {
            token_to_id,
            id_to_token,
            scores,
            unk_token_id,
        })
    }

    /// Encode text to token IDs using Viterbi algorithm
    ///
    /// Finds the most likely segmentation based on token scores.
    ///
    /// # Arguments
    ///
    /// * `text` - Input text to encode
    ///
    /// # Returns
    ///
    /// Vector of token IDs
    #[must_use]
    pub fn encode(&self, text: &str) -> Vec<u32> {
        if text.is_empty() {
            return Vec::new();
        }

        let chars: Vec<char> = text.chars().collect();
        let (_best_score, best_token) = self.viterbi_forward(&chars);
        let tokens = Self::viterbi_backtrack(&chars, &best_token);

        // Convert tokens to IDs
        tokens
            .into_iter()
            .map(|t| {
                self.token_to_id
                    .get(&t)
                    .copied()
                    .unwrap_or(self.unk_token_id)
            })
            .collect()
    }

    /// Viterbi forward pass: find best score and token at each position
    ///
    /// # Arguments
    ///
    /// * `chars` - Character array of input text
    ///
    /// # Returns
    ///
    /// Tuple of `(best_score, best_token)` vectors
    fn viterbi_forward(&self, chars: &[char]) -> ViterbiState {
        let n = chars.len();
        let mut best_score = vec![f32::NEG_INFINITY; n + 1];
        let mut best_token: Vec<Option<String>> = vec![None; n + 1];
        best_score[0] = 0.0;

        for end in 1..=n {
            // Try all possible start positions for token ending at `end`
            for start in 0..end {
                let substr: String = chars[start..end].iter().collect();
                if let Some(&score) = self.scores.get(&substr) {
                    let new_score = best_score[start] + score;
                    if new_score > best_score[end] {
                        best_score[end] = new_score;
                        best_token[end] = Some(substr);
                    }
                }
            }

            // If no token found ending at this position, use single character as unknown
            if best_token[end].is_none() && best_score[end - 1] > f32::NEG_INFINITY {
                let char_str: String = chars[end - 1..end].iter().collect();
                best_score[end] = best_score[end - 1] - 100.0; // Penalty for unknown
                best_token[end] = Some(char_str);
            }
        }

        (best_score, best_token)
    }

    /// Viterbi backtracking: reconstruct token sequence from `best_token`
    ///
    /// # Arguments
    ///
    /// * `chars` - Character array of input text
    /// * `best_token` - Best token at each position (from forward pass)
    ///
    /// # Returns
    ///
    /// Vector of tokens in forward order
    fn viterbi_backtrack(chars: &[char], best_token: &[Option<String>]) -> Vec<String> {
        let n = chars.len();
        let mut tokens = Vec::new();
        let mut pos = n;

        while pos > 0 {
            if let Some(token) = &best_token[pos] {
                tokens.push(token.clone());
                pos -= token.chars().count();
            } else {
                // Fallback: single character
                let char_str: String = chars[pos - 1..pos].iter().collect();
                tokens.push(char_str);
                pos -= 1;
            }
        }

        tokens.reverse();
        tokens
    }

    /// Decode token IDs to text
    ///
    /// # Arguments
    ///
    /// * `token_ids` - Token IDs to decode
    ///
    /// # Errors
    ///
    /// Returns error if any token ID is invalid
    pub fn decode(&self, token_ids: &[u32]) -> Result<String> {
        let mut result = String::new();

        for &id in token_ids {
            let token =
                self.id_to_token
                    .get(&id)
                    .ok_or_else(|| RealizarError::UnsupportedOperation {
                        operation: "decode_sentencepiece_token".to_string(),
                        reason: format!("Invalid token ID: {id}"),
                    })?;
            result.push_str(token);
        }

        Ok(result)
    }

    /// Get vocabulary size
    #[must_use]
    pub fn vocab_size(&self) -> usize {
        self.token_to_id.len()
    }

    /// Get token ID for a token
    #[must_use]
    pub fn get_token_id(&self, token: &str) -> Option<u32> {
        self.token_to_id.get(token).copied()
    }

    /// Get token for a token ID
    #[must_use]
    pub fn get_token(&self, id: u32) -> Option<&str> {
        self.id_to_token.get(&id).map(String::as_str)
    }

    /// Get score for a token
    #[must_use]
    pub fn get_score(&self, token: &str) -> Option<f32> {
        self.scores.get(token).copied()
    }
}

impl Tokenizer {
    /// Create a new tokenizer from vocabulary
    ///
    /// # Arguments
    ///
    /// * `vocab` - Vocabulary mapping
    /// * `unk_token` - Unknown token (default: `"<unk>"`)
    ///
    /// # Errors
    ///
    /// Returns error if unknown token not in vocabulary
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let vocab = Vocabulary::from_tokens(vec!["<unk>".to_string(), "hello".to_string()])?;
    /// let tokenizer = Tokenizer::new(vocab, "<unk>")?;
    /// ```
    pub fn new(vocab: Vocabulary, unk_token: &str) -> Result<Self> {
        let unk_token_id =
            vocab
                .get_id(unk_token)
                .ok_or_else(|| RealizarError::UnsupportedOperation {
                    operation: "create_tokenizer".to_string(),
                    reason: format!("Unknown token '{unk_token}' not in vocabulary"),
                })?;

        Ok(Self {
            vocab,
            unk_token_id,
        })
    }

    /// Encode text to token IDs (simple word-level tokenization)
    ///
    /// # Arguments
    ///
    /// * `text` - Input text to encode
    ///
    /// # Returns
    ///
    /// Vector of token IDs
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let token_ids = tokenizer.encode("hello world")?;
    /// assert_eq!(token_ids, vec![1, 2]);
    /// ```
    #[must_use]
    pub fn encode(&self, text: &str) -> Vec<u32> {
        text.split_whitespace()
            .map(|word| self.vocab.get_id(word).unwrap_or(self.unk_token_id))
            .collect()
    }

    /// Decode token IDs to text
    ///
    /// # Arguments
    ///
    /// * `token_ids` - Token IDs to decode
    ///
    /// # Returns
    ///
    /// Decoded text string
    ///
    /// # Errors
    ///
    /// Returns error if any token ID is invalid
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let text = tokenizer.decode(&[1, 2])?;
    /// assert_eq!(text, "hello world");
    /// ```
    pub fn decode(&self, token_ids: &[u32]) -> Result<String> {
        let tokens: Result<Vec<&str>> = token_ids
            .iter()
            .map(|&id| {
                self.vocab
                    .get_token(id)
                    .ok_or_else(|| RealizarError::UnsupportedOperation {
                        operation: "decode_token".to_string(),
                        reason: format!("Invalid token ID: {id}"),
                    })
            })
            .collect();

        Ok(tokens?.join(" "))
    }

    /// Get vocabulary size
    #[must_use]
    pub fn vocab_size(&self) -> usize {
        self.vocab.size()
    }
}
