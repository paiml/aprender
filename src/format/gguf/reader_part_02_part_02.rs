impl GgufReader {
    /// Load and parse a GGUF file
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let mut file = File::open(path.as_ref()).map_err(AprenderError::Io)?;
        let mut data = Vec::new();
        file.read_to_end(&mut data).map_err(AprenderError::Io)?;
        Self::from_bytes(data)
    }

    /// Parse GGUF from bytes
    pub fn from_bytes(data: Vec<u8>) -> Result<Self> {
        if data.len() < 24 {
            return Err(AprenderError::FormatError {
                message: "GGUF file too small (< 24 bytes)".to_string(),
            });
        }

        // Check magic (GH-183: Enhanced error message for debugging)
        let magic = read_u32(&data, 0)?;
        if magic != GGUF_MAGIC {
            // Show both hex and ASCII for easier debugging
            let magic_bytes = &data[0..4.min(data.len())];
            let magic_ascii: String = magic_bytes
                .iter()
                .map(|&b| if b.is_ascii_graphic() { b as char } else { '.' })
                .collect();
            return Err(AprenderError::FormatError {
                message: format!(
                    "Invalid GGUF magic: 0x{magic:08X} (bytes: {magic_bytes:02X?}, ascii: \"{magic_ascii}\"), \
                     expected 0x{GGUF_MAGIC:08X} (\"GGUF\")"
                ),
            });
        }

        let version = read_u32(&data, 4)?;
        let tensor_count = read_u64(&data, 8)?;
        let metadata_kv_count = read_u64(&data, 16)?;

        // BUG-GGUF-001 FIX: Validate counts before allocation to prevent OOM attacks
        if tensor_count > MAX_TENSOR_COUNT {
            return Err(AprenderError::FormatError {
                message: format!(
                    "GGUF tensor_count {} exceeds maximum allowed {} (possible corrupted/malicious file)",
                    tensor_count, MAX_TENSOR_COUNT
                ),
            });
        }
        if metadata_kv_count > MAX_METADATA_COUNT {
            return Err(AprenderError::FormatError {
                message: format!(
                    "GGUF metadata_kv_count {} exceeds maximum allowed {} (possible corrupted/malicious file)",
                    metadata_kv_count, MAX_METADATA_COUNT
                ),
            });
        }

        // Parse metadata section (extract vocabulary and other tokenizer data)
        let mut offset = 24;
        let mut metadata = BTreeMap::new();
        for _ in 0..metadata_kv_count {
            // Read key
            let (key, key_len) = read_string(&data, offset)?;
            offset += key_len;

            // Read value type
            let value_type = read_u32(&data, offset)?;
            offset += 4;

            // Parse value for tokenizer, general, and model architecture keys
            // We parse: tokenizer.*, general.*, and per-arch keys for full model config
            if key.starts_with("tokenizer.")
                || key.starts_with("general.")
                || key.starts_with("llama.")
                || key.starts_with("qwen2.")
                || key.starts_with("qwen3.")
                || key.starts_with("phi.")
                || key.starts_with("mistral.")
                || key.starts_with("gpt2.")
            {
                let (value, value_len) = read_metadata_value(&data, offset, value_type)?;
                metadata.insert(key, value);
                offset += value_len;
            } else {
                // Skip other metadata for efficiency
                let value_len = skip_metadata_value(&data, offset, value_type)?;
                offset += value_len;
            }
        }

        // Parse tensor infos
        let mut tensors = Vec::with_capacity(tensor_count as usize);
        for _ in 0..tensor_count {
            // Read name
            let (name, name_len) = read_string(&data, offset)?;
            offset += name_len;

            // Read n_dims
            let n_dims = read_u32(&data, offset)?;
            offset += 4;

            // BUG-GGUF-001 FIX: Validate n_dims to prevent allocation attacks
            if n_dims > MAX_DIMS {
                return Err(AprenderError::FormatError {
                    message: format!(
                        "Tensor '{}' has {} dimensions, exceeds maximum {} (possible corrupted file)",
                        name, n_dims, MAX_DIMS
                    ),
                });
            }

            // Read dimensions
            let mut dims = Vec::with_capacity(n_dims as usize);
            for _ in 0..n_dims {
                dims.push(read_u64(&data, offset)?);
                offset += 8;
            }

            // Read dtype
            let dtype = read_u32(&data, offset)?;
            offset += 4;

            // Read offset
            let tensor_offset = read_u64(&data, offset)?;
            offset += 8;

            tensors.push(GgufTensorMeta {
                name,
                dims,
                dtype,
                offset: tensor_offset,
            });
        }

        // Align to GGUF_DEFAULT_ALIGNMENT for tensor data
        let padding = padding_for_alignment(offset, GGUF_DEFAULT_ALIGNMENT);
        let data_offset = offset + padding;

        Ok(Self {
            data,
            version,
            tensor_count,
            tensors,
            data_offset,
            metadata,
        })
    }

    /// Get vocabulary tokens from metadata
    ///
    /// Returns the token strings indexed by token ID.
    /// Uses "tokenizer.ggml.tokens" key from GGUF metadata.
    #[must_use]
    pub fn vocabulary(&self) -> Option<Vec<String>> {
        if let Some(GgufValue::ArrayString(tokens)) = self.metadata.get("tokenizer.ggml.tokens") {
            if tokens.is_empty() {
                None
            } else {
                Some(tokens.clone())
            }
        } else {
            None
        }
    }

    /// Get tokenizer model type (e.g., "llama", "gpt2")
    #[must_use]
    pub fn tokenizer_model(&self) -> Option<String> {
        if let Some(GgufValue::String(model)) = self.metadata.get("tokenizer.ggml.model") {
            Some(model.clone())
        } else {
            None
        }
    }

    /// Get BOS (beginning of sequence) token ID
    #[must_use]
    pub fn bos_token_id(&self) -> Option<u32> {
        if let Some(GgufValue::Uint32(id)) = self.metadata.get("tokenizer.ggml.bos_token_id") {
            Some(*id)
        } else {
            None
        }
    }

    /// Get EOS (end of sequence) token ID
    #[must_use]
    pub fn eos_token_id(&self) -> Option<u32> {
        if let Some(GgufValue::Uint32(id)) = self.metadata.get("tokenizer.ggml.eos_token_id") {
            Some(*id)
        } else {
            None
        }
    }

    /// Get BPE merge rules from metadata (PMAT-171)
    ///
    /// Returns the merge rules as "token1 token2" strings for BPE encoding.
    /// Uses "tokenizer.ggml.merges" key from GGUF metadata.
    #[must_use]
    pub fn merges(&self) -> Option<Vec<String>> {
        if let Some(GgufValue::ArrayString(merges)) = self.metadata.get("tokenizer.ggml.merges") {
            if merges.is_empty() {
                None
            } else {
                Some(merges.clone())
            }
        } else {
            None
        }
    }

    /// Get general architecture name (e.g., "llama", "qwen2")
    #[must_use]
    pub fn architecture(&self) -> Option<String> {
        if let Some(GgufValue::String(arch)) = self.metadata.get("general.architecture") {
            Some(arch.clone())
        } else {
            None
        }
    }

    /// Get model name from metadata
    #[must_use]
    pub fn model_name(&self) -> Option<String> {
        if let Some(GgufValue::String(name)) = self.metadata.get("general.name") {
            Some(name.clone())
        } else {
            None
        }
    }

    // ========================================================================
    // Transformer Model Config (CRITICAL for APR inference)
    // ========================================================================

    /// Get hidden dimension (embedding_length)
    #[must_use]
    pub fn hidden_size(&self) -> Option<usize> {
        let arch = self.architecture().unwrap_or_else(|| "llama".to_string());
        let key = format!("{arch}.embedding_length");
        match self.metadata.get(&key) {
            Some(GgufValue::Uint32(v)) => Some(*v as usize),
            Some(GgufValue::Uint64(v)) => Some(*v as usize),
            _ => None,
        }
    }

    /// Get number of transformer layers (block_count)
    #[must_use]
    pub fn num_layers(&self) -> Option<usize> {
        let arch = self.architecture().unwrap_or_else(|| "llama".to_string());
        let key = format!("{arch}.block_count");
        match self.metadata.get(&key) {
            Some(GgufValue::Uint32(v)) => Some(*v as usize),
            Some(GgufValue::Uint64(v)) => Some(*v as usize),
            _ => None,
        }
    }

    /// Get number of attention heads
    #[must_use]
    pub fn num_heads(&self) -> Option<usize> {
        let arch = self.architecture().unwrap_or_else(|| "llama".to_string());
        let key = format!("{arch}.attention.head_count");
        match self.metadata.get(&key) {
            Some(GgufValue::Uint32(v)) => Some(*v as usize),
            Some(GgufValue::Uint64(v)) => Some(*v as usize),
            _ => None,
        }
    }

    /// Get number of key-value heads (for GQA)
    #[must_use]
    pub fn num_kv_heads(&self) -> Option<usize> {
        let arch = self.architecture().unwrap_or_else(|| "llama".to_string());
        let key = format!("{arch}.attention.head_count_kv");
        match self.metadata.get(&key) {
            Some(GgufValue::Uint32(v)) => Some(*v as usize),
            Some(GgufValue::Uint64(v)) => Some(*v as usize),
            _ => self.num_heads(), // Default to num_heads if not GQA
        }
    }

    /// Get vocabulary size
    #[must_use]
    pub fn vocab_size(&self) -> Option<usize> {
        // Try architecture-specific key first
        let arch = self.architecture().unwrap_or_else(|| "llama".to_string());
        let key = format!("{arch}.vocab_size");
        if let Some(GgufValue::Uint32(v)) = self.metadata.get(&key) {
            return Some(*v as usize);
        }
        if let Some(GgufValue::Uint64(v)) = self.metadata.get(&key) {
            return Some(*v as usize);
        }
        // Fall back to vocabulary length
        self.vocabulary().map(|v| v.len())
    }

    /// Get FFN intermediate dimension
    #[must_use]
    pub fn intermediate_size(&self) -> Option<usize> {
        let arch = self.architecture().unwrap_or_else(|| "llama".to_string());
        let key = format!("{arch}.feed_forward_length");
        match self.metadata.get(&key) {
            Some(GgufValue::Uint32(v)) => Some(*v as usize),
            Some(GgufValue::Uint64(v)) => Some(*v as usize),
            _ => None,
        }
    }

    /// Get maximum context length
    #[must_use]
    pub fn context_length(&self) -> Option<usize> {
        let arch = self.architecture().unwrap_or_else(|| "llama".to_string());
        let key = format!("{arch}.context_length");
        match self.metadata.get(&key) {
            Some(GgufValue::Uint32(v)) => Some(*v as usize),
            Some(GgufValue::Uint64(v)) => Some(*v as usize),
            _ => None,
        }
    }

    /// Get RoPE theta (frequency base)
    #[must_use]
    pub fn rope_theta(&self) -> Option<f32> {
        let arch = self.architecture().unwrap_or_else(|| "llama".to_string());
        let key = format!("{arch}.rope.freq_base");
        match self.metadata.get(&key) {
            Some(GgufValue::Float32(v)) => Some(*v),
            Some(GgufValue::Uint32(v)) => Some(*v as f32),
            _ => None,
        }
    }

    /// Get RMS norm epsilon (or standard LayerNorm epsilon for GPT-2)
    #[must_use]
    pub fn rms_norm_eps(&self) -> Option<f32> {
        let arch = self.architecture().unwrap_or_else(|| "llama".to_string());
        // GH-277: Try RMSNorm key first, then standard LayerNorm key (for GPT-2)
        let rms_key = format!("{arch}.attention.layer_norm_rms_epsilon");
        let ln_key = format!("{arch}.attention.layer_norm_epsilon");
        match self.metadata.get(&rms_key) {
            Some(GgufValue::Float32(v)) => Some(*v),
            _ => match self.metadata.get(&ln_key) {
                Some(GgufValue::Float32(v)) => Some(*v),
                _ => None,
            },
        }
    }

    // ========================================================================
    // GH-253: Tokenizer metadata accessors for GGUF export round-trip
    // ========================================================================

    /// Get per-token type array (tokenizer.ggml.token_type)
    /// Values: 1=normal, 2=unknown, 3=control/special, 4=user_defined, etc.
    #[must_use]
    pub fn token_type(&self) -> Option<Vec<i32>> {
        if let Some(GgufValue::ArrayInt32(types)) = self.metadata.get("tokenizer.ggml.token_type") {
            if types.is_empty() {
                None
            } else {
                Some(types.clone())
            }
        } else {
            None
        }
    }

    /// Get padding token ID (tokenizer.ggml.padding_token_id)
    #[must_use]
    pub fn padding_token_id(&self) -> Option<u32> {
        if let Some(GgufValue::Uint32(id)) = self.metadata.get("tokenizer.ggml.padding_token_id") {
            Some(*id)
        } else {
            None
        }
    }

    /// Get add_bos_token flag (tokenizer.ggml.add_bos_token)
    #[must_use]
    pub fn add_bos_token(&self) -> Option<bool> {
        if let Some(GgufValue::Bool(v)) = self.metadata.get("tokenizer.ggml.add_bos_token") {
            Some(*v)
        } else {
            None
        }
    }

    /// Get chat template (tokenizer.chat_template)
    #[must_use]
    pub fn chat_template(&self) -> Option<String> {
        if let Some(GgufValue::String(tmpl)) = self.metadata.get("tokenizer.chat_template") {
            Some(tmpl.clone())
        } else {
            None
        }
    }

    /// GH-277: Get pre-tokenizer type (tokenizer.ggml.pre)
    #[must_use]
    pub fn pre_tokenizer_type(&self) -> Option<String> {
        if let Some(GgufValue::String(pre)) = self.metadata.get("tokenizer.ggml.pre") {
            Some(pre.clone())
        } else {
            None
        }
    }
}
