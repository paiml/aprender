impl BpeTokenizer {
    /// Encode text into token IDs.
    ///
    /// # Arguments
    ///
    /// * `text` - Text to encode
    ///
    /// # Returns
    ///
    /// * `Ok(Vec<u32>)` - Token IDs
    /// * `Err(AprenderError)` - If encoding fails
    ///
    /// # Examples
    ///
    /// ```
    /// use aprender::text::tokenize::BpeTokenizer;
    ///
    /// let corpus = vec!["hello", "world"];
    /// let tokenizer = BpeTokenizer::train(&corpus, 50).expect("train");
    ///
    /// let ids = tokenizer.encode("hello").expect("encode");
    /// assert!(!ids.is_empty());
    /// ```
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, AprenderError> {
        if self.byte_level {
            return self.encode_byte_level(text);
        }

        let mut token_ids = Vec::new();

        for word in text.split_whitespace() {
            // Convert word to character sequence with end-of-word marker
            let mut tokens: Vec<String> = word.chars().map(|c| c.to_string()).collect();
            if !tokens.is_empty() {
                if let Some(last) = tokens.last_mut() {
                    last.push_str(&self.end_of_word);
                }
            }

            // Apply merges in order
            for (left, right) in &self.merges {
                let merged = format!("{left}{right}");
                let mut i = 0;
                while i < tokens.len().saturating_sub(1) {
                    if &tokens[i] == left && &tokens[i + 1] == right {
                        merged.clone_into(&mut tokens[i]);
                        tokens.remove(i + 1);
                    } else {
                        i += 1;
                    }
                }
            }

            // Convert tokens to IDs
            let unk_id = self
                .vocab
                .get(&self.special_tokens.unk)
                .copied()
                .unwrap_or(0);

            for token in tokens {
                let id = self.vocab.get(&token).copied().unwrap_or(unk_id);
                token_ids.push(id);
            }
        }

        Ok(token_ids)
    }

    /// Byte-level encode path (tokenizer-bpe-v1 INV-BPE-003 / INV-BPE-007).
    fn encode_byte_level(&self, text: &str) -> Result<Vec<u32>, AprenderError> {
        let (byte_enc, _) = crate::text::bpe::bytes_to_unicode();
        let mut tokens: Vec<String> = text
            .bytes()
            .map(|b| byte_enc.get(&b).copied().unwrap_or('?').to_string())
            .collect();

        for (left, right) in &self.merges {
            let merged = format!("{left}{right}");
            let mut i = 0;
            while i < tokens.len().saturating_sub(1) {
                if &tokens[i] == left && &tokens[i + 1] == right {
                    merged.clone_into(&mut tokens[i]);
                    tokens.remove(i + 1);
                } else {
                    i += 1;
                }
            }
        }

        let unk_id = self
            .vocab
            .get(&self.special_tokens.unk)
            .copied()
            .unwrap_or(0);

        let ids = tokens
            .into_iter()
            .map(|tok| self.vocab.get(&tok).copied().unwrap_or(unk_id))
            .collect();
        Ok(ids)
    }

    /// Encode text and add special tokens (BOS/EOS).
    ///
    /// # Arguments
    ///
    /// * `text` - Text to encode
    /// * `add_bos` - Add beginning-of-sequence token
    /// * `add_eos` - Add end-of-sequence token
    pub fn encode_with_special(
        &self,
        text: &str,
        add_bos: bool,
        add_eos: bool,
    ) -> Result<Vec<u32>, AprenderError> {
        let mut ids = Vec::new();

        if add_bos {
            if let Some(ref bos) = self.special_tokens.bos {
                if let Some(&id) = self.vocab.get(bos) {
                    ids.push(id);
                }
            }
        }

        ids.extend(self.encode(text)?);

        if add_eos {
            if let Some(ref eos) = self.special_tokens.eos {
                if let Some(&id) = self.vocab.get(eos) {
                    ids.push(id);
                }
            }
        }

        Ok(ids)
    }

    /// Decode token IDs back to text.
    ///
    /// # Arguments
    ///
    /// * `ids` - Token IDs to decode
    ///
    /// # Returns
    ///
    /// * `Ok(String)` - Decoded text
    /// * `Err(AprenderError)` - If decoding fails
    ///
    /// # Examples
    ///
    /// ```
    /// use aprender::text::tokenize::BpeTokenizer;
    ///
    /// let corpus = vec!["hello"];
    /// let tokenizer = BpeTokenizer::train(&corpus, 50).expect("train");
    ///
    /// let ids = tokenizer.encode("hello").expect("encode");
    /// let text = tokenizer.decode(&ids).expect("decode");
    /// assert_eq!(text, "hello");
    /// ```
    pub fn decode(&self, ids: &[u32]) -> Result<String, AprenderError> {
        if self.byte_level {
            return self.decode_byte_level(ids);
        }

        let mut result = String::new();
        let mut need_space = false;

        for &id in ids {
            // Skip special tokens in output
            if let Some(ref bos) = self.special_tokens.bos {
                if self.vocab.get(bos) == Some(&id) {
                    continue;
                }
            }
            if let Some(ref eos) = self.special_tokens.eos {
                if self.vocab.get(eos) == Some(&id) {
                    continue;
                }
            }
            if let Some(ref pad) = self.special_tokens.pad {
                if self.vocab.get(pad) == Some(&id) {
                    continue;
                }
            }

            let token = self
                .inverse_vocab
                .get(&id)
                .map_or_else(|| self.special_tokens.unk.clone(), Clone::clone);

            // Handle end-of-word marker
            if token.ends_with(&self.end_of_word) {
                if need_space {
                    result.push(' ');
                }
                let cleaned = token.trim_end_matches(self.end_of_word.as_str());
                result.push_str(cleaned);
                need_space = true;
            } else {
                if need_space && result.ends_with(' ') {
                    // Already has space
                } else if need_space {
                    // Continue building word
                }
                result.push_str(&token);
            }
        }

        Ok(result)
    }

    /// Byte-level decode path: invert `bytes_to_unicode`, rebuild bytes, then
    /// UTF-8-decode. Unknown tokens are skipped (their byte contribution is
    /// already captured by UNK in the id stream).
    fn decode_byte_level(&self, ids: &[u32]) -> Result<String, AprenderError> {
        let (_, byte_dec) = crate::text::bpe::bytes_to_unicode();
        let mut bytes: Vec<u8> = Vec::with_capacity(ids.len());

        let unk_id = self.vocab.get(&self.special_tokens.unk).copied();

        for &id in ids {
            if let Some(ref bos) = self.special_tokens.bos {
                if self.vocab.get(bos) == Some(&id) {
                    continue;
                }
            }
            if let Some(ref eos) = self.special_tokens.eos {
                if self.vocab.get(eos) == Some(&id) {
                    continue;
                }
            }
            if let Some(ref pad) = self.special_tokens.pad {
                if self.vocab.get(pad) == Some(&id) {
                    continue;
                }
            }
            if Some(id) == unk_id {
                continue;
            }

            let Some(token) = self.inverse_vocab.get(&id) else {
                continue;
            };
            for ch in token.chars() {
                if let Some(&b) = byte_dec.get(&ch) {
                    bytes.push(b);
                }
            }
        }

        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }

    /// Get the vocabulary size.
    #[must_use]
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// Get a reference to the vocabulary.
    #[must_use]
    pub fn vocab(&self) -> &HashMap<String, u32> {
        &self.vocab
    }

    /// Get the merge rules.
    #[must_use]
    pub fn merges(&self) -> &[(String, String)] {
        &self.merges
    }

    /// Check if a token exists in the vocabulary.
    #[must_use]
    pub fn contains(&self, token: &str) -> bool {
        self.vocab.contains_key(token)
    }

    /// Get the ID for a token.
    #[must_use]
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.vocab.get(token).copied()
    }

    /// Get the token for an ID.
    #[must_use]
    pub fn id_to_token(&self, id: u32) -> Option<&str> {
        self.inverse_vocab.get(&id).map(String::as_str)
    }
}
