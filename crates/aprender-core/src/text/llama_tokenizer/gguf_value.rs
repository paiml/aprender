
impl LlamaTokenizer {
    /// Create a new `LlamaTokenizer` from vocabulary data.
    ///
    /// # Arguments
    /// * `tokens` - List of token strings
    /// * `scores` - Token scores/priorities
    /// * `bos_token_id` - Beginning of sequence token ID
    /// * `eos_token_id` - End of sequence token ID
    /// * `unk_token_id` - Unknown token ID
    ///
    /// # Errors
    /// Returns error if vocabulary is empty or IDs are out of range.
    pub fn new(
        tokens: Vec<String>,
        scores: Vec<f32>,
        bos_token_id: u32,
        eos_token_id: u32,
        unk_token_id: u32,
    ) -> Result<Self> {
        if tokens.is_empty() {
            return Err(AprenderError::ValidationError {
                message: "Empty vocabulary".to_string(),
            });
        }

        let vocab_size = tokens.len();

        // Build vocab mappings
        let mut vocab = HashMap::with_capacity(vocab_size);
        let mut id_to_token = HashMap::with_capacity(vocab_size);

        for (id, token) in tokens.into_iter().enumerate() {
            let id = id as u32;
            vocab.insert(token.clone(), id);
            id_to_token.insert(id, token);
        }

        // Validate special token IDs
        let validate_id = |id: u32, name: &str| -> Result<()> {
            if id as usize >= vocab_size {
                return Err(AprenderError::ValidationError {
                    message: format!("{name} token ID {id} out of range (vocab_size={vocab_size})"),
                });
            }
            Ok(())
        };

        validate_id(bos_token_id, "BOS")?;
        validate_id(eos_token_id, "EOS")?;
        validate_id(unk_token_id, "UNK")?;

        Ok(Self {
            vocab,
            id_to_token,
            scores,
            bos_token_id,
            eos_token_id,
            unk_token_id,
            pad_token_id: None,
            vocab_size,
            model: TokenizerModel::SentencePiece, // Default, updated by GGUF loader
        })
    }

    /// Set the tokenizer model type.
    pub fn set_model(&mut self, model: TokenizerModel) {
        self.model = model;
    }

    /// Get the tokenizer model type.
    #[must_use]
    pub fn model(&self) -> TokenizerModel {
        self.model
    }

    /// Get vocabulary size.
    #[must_use]
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Get BOS token ID.
    #[must_use]
    pub fn bos_token_id(&self) -> u32 {
        self.bos_token_id
    }

    /// Get EOS token ID.
    #[must_use]
    pub fn eos_token_id(&self) -> u32 {
        self.eos_token_id
    }

    /// Get UNK token ID.
    #[must_use]
    pub fn unk_token_id(&self) -> u32 {
        self.unk_token_id
    }

    /// Encode text to token IDs.
    ///
    /// Uses SentencePiece-style BPE encoding:
    /// 1. Add space prefix to indicate word boundaries
    /// 2. Look up tokens with longest-match greedy algorithm
    /// 3. Fall back to byte tokens for unknown characters
    ///
    /// # Arguments
    /// * `text` - Input text to tokenize
    ///
    /// # Returns
    /// Vector of token IDs (without BOS/EOS)
    #[must_use]
    pub fn encode(&self, text: &str) -> Vec<u32> {
        if text.is_empty() {
            return Vec::new();
        }

        // Normalize based on tokenizer model type:
        // - SentencePiece: Uses ▁ (U+2581) as word boundary marker
        //   "Hello, world!" becomes "▁Hello▁,▁world▁!"
        // - GPT-2: Uses Ġ (U+0120) as space prefix
        //   "Hello, world!" becomes "Hello,Ġworld!"
        let normalized = match self.model {
            TokenizerModel::SentencePiece => format!("▁{}", text.replace(' ', "▁")),
            TokenizerModel::Gpt2 => text.replace(' ', "\u{0120}").replace('\n', "\u{010A}"),
        };
        let chars: Vec<char> = normalized.chars().collect();

        let mut tokens: Vec<u32> = Vec::with_capacity(text.len());
        let mut i = 0;

        while i < chars.len() {
            let mut best_len = 0;
            let mut best_token_id = self.unk_token_id;

            // Try matching increasingly longer substrings (greedy longest match)
            for end in (i + 1)..=chars.len().min(i + 32) {
                let substr: String = chars[i..end].iter().collect();

                // Check if this substring is in vocab
                if let Some(&token_id) = self.vocab.get(&substr) {
                    best_len = end - i;
                    best_token_id = token_id;
                }
            }

            if best_len > 0 {
                tokens.push(best_token_id);
                i += best_len;
            } else {
                // Fall back to byte tokens for unknown characters
                let c = chars[i];
                for byte in c.to_string().as_bytes() {
                    let byte_token = format!("<0x{byte:02X}>");
                    if let Some(&token_id) = self.vocab.get(&byte_token) {
                        tokens.push(token_id);
                    } else {
                        tokens.push(self.unk_token_id);
                    }
                }
                i += 1;
            }
        }

        tokens
    }

    /// Encode text with BOS token prepended.
    ///
    /// # Arguments
    /// * `text` - Input text to tokenize
    ///
    /// # Returns
    /// Vector of token IDs starting with BOS
    #[must_use]
    pub fn encode_with_bos(&self, text: &str) -> Vec<u32> {
        let mut tokens = vec![self.bos_token_id];
        tokens.extend(self.encode(text));
        tokens
    }

    /// Decode token IDs to text.
    ///
    /// Handles both SentencePiece-style (LLaMA) and GPT-2 BPE (Qwen) tokenizers.
    ///
    /// # Arguments
    /// * `token_ids` - Token IDs to decode
    ///
    /// # Returns
    /// Decoded text string
    #[must_use]
    pub fn decode(&self, token_ids: &[u32]) -> String {
        match self.model {
            TokenizerModel::Gpt2 => self.decode_gpt2(token_ids),
            TokenizerModel::SentencePiece => self.decode_sentencepiece(token_ids),
        }
    }

    /// Decode using GPT-2 byte-level BPE (Qwen, GPT-2, etc.)
    fn decode_gpt2(&self, token_ids: &[u32]) -> String {
        let mut bytes = Vec::with_capacity(token_ids.len() * 4);

        for &token_id in token_ids {
            // Skip special tokens in output
            if token_id == self.bos_token_id || token_id == self.eos_token_id {
                continue;
            }

            if let Some(token) = self.id_to_token.get(&token_id) {
                // GPT-2 BPE: decode unicode chars to original bytes
                bytes.extend(decode_gpt2_token(token));
            }
        }

        // Convert bytes to UTF-8 string, replacing invalid sequences
        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// Decode using SentencePiece-style BPE (LLaMA, Mistral, etc.)
    ///
    /// # BUG-TOK-001 FIX: Correct byte token handling for multibyte UTF-8
    ///
    /// Previous implementation used `byte as char` which is WRONG for bytes >= 128.
    /// For example, decoding `<0xE4><0xB8><0x96>` (UTF-8 for "世") would produce:
    ///   - 0xE4 as char = 'ä' (Latin Extended)
    ///   - 0xB8 as char = '¸' (cedilla)
    ///   - 0x96 as char = control char
    ///
    /// Instead of the correct output "世".
    ///
    /// The fix collects bytes and converts to UTF-8 at boundaries.
    fn decode_sentencepiece(&self, token_ids: &[u32]) -> String {
        let mut result = String::with_capacity(token_ids.len() * 4);
        let mut pending_bytes: Vec<u8> = Vec::new();

        for &token_id in token_ids {
            // Skip special tokens in output
            if token_id == self.bos_token_id || token_id == self.eos_token_id {
                continue;
            }

            if let Some(token) = self.id_to_token.get(&token_id) {
                // Handle byte tokens like <0x0A> for newlines
                if token.starts_with(BYTE_FALLBACK_PREFIX) && token.ends_with('>') {
                    if let Some(hex) = token.strip_prefix(BYTE_FALLBACK_PREFIX) {
                        if let Some(hex) = hex.strip_suffix('>') {
                            if let Ok(byte) = u8::from_str_radix(hex, 16) {
                                // BUG-TOK-001 FIX: Collect bytes instead of casting to char
                                pending_bytes.push(byte);
                                continue;
                            }
                        }
                    }
                }

                // Flush pending bytes as UTF-8 before adding regular token
                if !pending_bytes.is_empty() {
                    result.push_str(&String::from_utf8_lossy(&pending_bytes));
                    pending_bytes.clear();
                }

                // Handle SentencePiece space prefix (U+2581)
                let text = token.replace('▁', " ");
                // Handle GPT-2 BPE space prefix (U+0120 'Ġ') for hybrid tokenizers
                let text = text.replace('Ġ', " ");

                result.push_str(&text);
            }
        }

        // Flush any remaining pending bytes
        if !pending_bytes.is_empty() {
            result.push_str(&String::from_utf8_lossy(&pending_bytes));
        }

        // Clean up leading space if present (SentencePiece adds leading space)
        if result.starts_with(' ') {
            result.remove(0);
        }

        result
    }

    /// Get token string for an ID.
    #[must_use]
    pub fn id_to_token(&self, id: u32) -> Option<&str> {
        self.id_to_token.get(&id).map(String::as_str)
    }

    /// Get token ID for a string.
    #[must_use]
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.vocab.get(token).copied()
    }
}

// ============================================================================
// GGUF Loading
// ============================================================================

/// Value types in GGUF metadata
#[derive(Debug, Clone)]
pub enum GGUFValue {
    UInt8(u8),
    Int8(i8),
    UInt16(u16),
    Int16(i16),
    UInt32(u32),
    Int32(i32),
    Float32(f32),
    Bool(bool),
    String(String),
    Array(Vec<GGUFValue>),
    UInt64(u64),
    Int64(i64),
    Float64(f64),
}
