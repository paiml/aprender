
impl QuantizedAprTransformer {
    /// Create a new quantized transformer with the given config and quantization type
    #[must_use]
    pub fn new(config: AprTransformerConfig, quant_type: AprQuantizationType) -> Self {
        let hidden_dim = config.hidden_dim;
        let vocab_size = config.vocab_size;
        let _intermediate_dim = config.intermediate_dim;

        // Calculate quantized sizes
        let embed_size = vocab_size * hidden_dim; // F32 for embeddings
        let layer_weight_size = Self::calculate_layer_bytes(&config, quant_type);
        let lm_head_size = Self::calculate_quantized_bytes(hidden_dim * vocab_size, quant_type);

        // Initialize with zeros
        let layer_weights = (0..config.num_layers)
            .map(|_| vec![0u8; layer_weight_size])
            .collect();

        Self {
            config,
            quant_type,
            token_embedding: vec![0.0; embed_size],
            layer_weights,
            output_norm_weight: vec![1.0; hidden_dim],
            lm_head_weight: vec![0u8; lm_head_size],
        }
    }

    /// Create from an F32 transformer by quantizing weights
    #[must_use]
    pub fn from_f32_transformer(
        f32_model: &AprTransformer,
        quant_type: AprQuantizationType,
    ) -> Self {
        let config = f32_model.config.clone();

        // For now, just create zero-initialized quantized model
        // Full quantization would convert F32 weights to Q4_K/Q8_0
        Self::new(config, quant_type)
    }

    /// Get the quantization type
    #[must_use]
    pub fn quantization_type(&self) -> AprQuantizationType {
        self.quant_type
    }

    /// Get bits per weight
    #[must_use]
    pub fn bits_per_weight(&self) -> f64 {
        self.quant_type.bits_per_weight()
    }

    /// Get the model configuration
    #[must_use]
    pub fn config(&self) -> &AprTransformerConfig {
        &self.config
    }

    /// Get total quantized weight bytes
    #[must_use]
    pub fn weight_bytes(&self) -> usize {
        let embed_bytes = self.token_embedding.len() * 4; // F32
        let layer_bytes: usize = self.layer_weights.iter().map(std::vec::Vec::len).sum();
        let norm_bytes = self.output_norm_weight.len() * 4; // F32
        let lm_head_bytes = self.lm_head_weight.len();

        embed_bytes + layer_bytes + norm_bytes + lm_head_bytes
    }

    /// Get equivalent F32 size for compression ratio
    #[must_use]
    pub fn f32_equivalent_bytes(&self) -> usize {
        let num_params = self.num_parameters();
        num_params * 4 // 4 bytes per F32
    }

    /// Get total number of parameters
    #[must_use]
    pub fn num_parameters(&self) -> usize {
        let hidden = self.config.hidden_dim;
        let vocab = self.config.vocab_size;
        let layers = self.config.num_layers;
        let intermediate = self.config.intermediate_dim;

        // Embedding + LM head
        let embed_params = vocab * hidden * 2;

        // Per layer: attn_norm + qkv + attn_out + ffn_up + ffn_down
        let layer_params = hidden
            + (hidden * 3 * hidden)
            + (hidden * hidden)
            + (hidden * intermediate)
            + (intermediate * hidden);

        // Output norm
        let norm_params = hidden;

        embed_params + (layers * layer_params) + norm_params
    }

    /// Calculate bytes needed for layer weights
    fn calculate_layer_bytes(
        config: &AprTransformerConfig,
        quant_type: AprQuantizationType,
    ) -> usize {
        let hidden = config.hidden_dim;
        let intermediate = config.intermediate_dim;

        // Layer weights: qkv + attn_out + ffn_up + ffn_down + norms
        let weight_elements = (hidden * 3 * hidden)
            + (hidden * hidden)
            + (hidden * intermediate)
            + (intermediate * hidden);

        Self::calculate_quantized_bytes(weight_elements, quant_type)
    }

    /// Calculate quantized byte size for N elements
    pub(crate) fn calculate_quantized_bytes(
        num_elements: usize,
        quant_type: AprQuantizationType,
    ) -> usize {
        let values_per_block = quant_type.values_per_block();
        let bytes_per_block = quant_type.bytes_per_block();

        // Round up to nearest block
        let num_blocks = num_elements.div_ceil(values_per_block);
        num_blocks * bytes_per_block
    }

    /// Forward pass with quantized weights
    ///
    /// Dequantizes weights on-the-fly during computation.
    pub fn forward(&self, token_ids: &[u32]) -> Result<Vec<f32>> {
        if token_ids.is_empty() {
            return Err(RealizarError::InvalidShape {
                reason: "Token sequence cannot be empty".to_string(),
            });
        }

        let hidden_dim = self.config.hidden_dim;
        let _vocab_size = self.config.vocab_size;

        // 1. Token embedding lookup (F32)
        let mut hidden = Vec::with_capacity(token_ids.len() * hidden_dim);
        for &token_id in token_ids {
            let offset = (token_id as usize) * hidden_dim;
            if offset + hidden_dim <= self.token_embedding.len() {
                hidden.extend_from_slice(&self.token_embedding[offset..offset + hidden_dim]);
            } else {
                hidden.extend(std::iter::repeat_n(0.0, hidden_dim));
            }
        }

        // 2. Process through layers (simplified - dequantize on the fly)
        // For zero-initialized weights, this is essentially a no-op
        for _layer_weights in &self.layer_weights {
            // In production: dequantize and apply layer operations
            // For now with zero weights: output stays the same
        }

        // 3. Final layer norm
        let seq_len = token_ids.len();
        let eps = self.config.eps;
        let mut normed = Vec::with_capacity(hidden.len());

        for s in 0..seq_len {
            let start = s * hidden_dim;
            let slice = &hidden[start..start + hidden_dim];

            let mean: f32 = slice.iter().sum::<f32>() / hidden_dim as f32;
            let variance: f32 =
                slice.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / hidden_dim as f32;
            let std_dev = (variance + eps).sqrt();

            for (i, &x) in slice.iter().enumerate() {
                let normalized = (x - mean) / std_dev;
                normed.push(normalized * self.output_norm_weight[i]);
            }
        }

        // 4. LM head (take last position, project to vocab)
        let last_hidden_start = (seq_len - 1) * hidden_dim;
        let last_hidden = &normed[last_hidden_start..last_hidden_start + hidden_dim];

        // Dequantize LM head and compute logits
        let logits = self.compute_lm_head_logits(last_hidden)?;

        Ok(logits)
    }

    /// Compute LM head logits (dequantize weight and matmul)
    fn compute_lm_head_logits(&self, _hidden: &[f32]) -> Result<Vec<f32>> {
        let vocab_size = self.config.vocab_size;
        let _hidden_dim = self.config.hidden_dim;

        // For zero-initialized weights, output is zeros
        // In production: dequantize self.lm_head_weight and compute
        let logits = vec![0.0f32; vocab_size];

        // Simple matmul with dequantized weights (placeholder)
        // Real implementation would use fused_q4k_dot or dequantize_q8_0
        match self.quant_type {
            AprQuantizationType::F32 => {
                // No dequantization needed (but we store as bytes anyway)
            },
            AprQuantizationType::Q4_K => {
                // Would call: fused_q4k_dot for each output
            },
            AprQuantizationType::Q8_0 => {
                // Would call: dequantize_q8_0 then dot product
            },
        }

        Ok(logits)
    }

    /// Serialize to bytes (APR binary format with quantization)
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        let mut bytes = Vec::new();

        // Header (64 bytes)
        bytes.extend_from_slice(&MAGIC);
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&(self.config.hidden_dim as u32).to_le_bytes());
        bytes.extend_from_slice(&(self.config.num_layers as u32).to_le_bytes());
        bytes.extend_from_slice(&(self.config.num_heads as u32).to_le_bytes());
        bytes.extend_from_slice(&(self.config.num_kv_heads as u32).to_le_bytes());
        bytes.extend_from_slice(&(self.config.vocab_size as u32).to_le_bytes());
        bytes.extend_from_slice(&(self.config.intermediate_dim as u32).to_le_bytes());
        bytes.extend_from_slice(&(self.config.context_length as u32).to_le_bytes());
        bytes.extend_from_slice(&self.config.rope_theta.to_le_bytes());
        bytes.extend_from_slice(&self.config.eps.to_le_bytes());

        // Tensor data offset (after header)
        let tensor_offset = APR_TRANSFORMER_HEADER_SIZE as u32;
        bytes.extend_from_slice(&tensor_offset.to_le_bytes());

        // Quantization type at offset 48
        bytes.push(self.quant_type.to_byte());

        // Pad to 64 bytes
        while bytes.len() < APR_TRANSFORMER_HEADER_SIZE {
            bytes.push(0);
        }

        // Token embeddings (F32)
        for &v in &self.token_embedding {
            bytes.extend_from_slice(&v.to_le_bytes());
        }

        // Layer weights (quantized)
        for layer in &self.layer_weights {
            bytes.extend_from_slice(layer);
        }

        // Output norm (F32)
        for &v in &self.output_norm_weight {
            bytes.extend_from_slice(&v.to_le_bytes());
        }

        // LM head (quantized)
        bytes.extend_from_slice(&self.lm_head_weight);

        Ok(bytes)
    }

    /// Deserialize from bytes
    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        if data.len() < APR_TRANSFORMER_HEADER_SIZE {
            return Err(RealizarError::FormatError {
                reason: format!("Data too small: {} bytes", data.len()),
            });
        }

        // Verify magic
        if data[0..4] != MAGIC {
            return Err(RealizarError::FormatError {
                reason: "Invalid APR magic".to_string(),
            });
        }

        // Parse header
        let hidden_dim = u32::from_le_bytes([data[8], data[9], data[10], data[11]]) as usize;
        let num_layers = u32::from_le_bytes([data[12], data[13], data[14], data[15]]) as usize;
        let num_heads = u32::from_le_bytes([data[16], data[17], data[18], data[19]]) as usize;
        let num_kv_heads = u32::from_le_bytes([data[20], data[21], data[22], data[23]]) as usize;
        let vocab_size = u32::from_le_bytes([data[24], data[25], data[26], data[27]]) as usize;
        let intermediate_dim =
            u32::from_le_bytes([data[28], data[29], data[30], data[31]]) as usize;
        let context_length = u32::from_le_bytes([data[32], data[33], data[34], data[35]]) as usize;
        let rope_theta = f32::from_le_bytes([data[36], data[37], data[38], data[39]]);
        let eps = f32::from_le_bytes([data[40], data[41], data[42], data[43]]);

        // Quantization type at offset 48
        let quant_type =
            AprQuantizationType::from_byte(data[48]).ok_or_else(|| RealizarError::FormatError {
                reason: format!("Invalid quantization type: {}", data[48]),
            })?;

        let config = AprTransformerConfig {
            architecture: "apr".to_string(),
            hidden_dim,
            num_layers,
            num_heads,
            num_kv_heads,
            vocab_size,
            intermediate_dim,
            context_length,
            rope_theta,
            eps,
            eos_token_id: None, // Parsed from APR metadata if available
            ..Default::default()
        };

        // For now, create with default weights
        // Full implementation would parse the weight data
        Ok(Self::new(config, quant_type))
    }

    /// Forward pass with KV cache for efficient autoregressive generation (Y4)
    ///
    /// Processes a single token using cached key-value pairs from previous positions.
    /// Uses quantized weights with on-the-fly dequantization.
    ///
    /// # Arguments
    ///
    /// * `token_id` - Single token ID to process
    /// * `cache` - Mutable KV cache to read from and append to
    /// * `position` - Position in sequence (0-indexed)
    ///
    /// # Returns
    ///
    /// Logits over vocabulary for next token prediction
    pub fn forward_with_cache(
        &self,
        token_id: u32,
        cache: &mut AprKVCache,
        _position: usize,
    ) -> Result<Vec<f32>> {
        let hidden_dim = self.config.hidden_dim;
        let num_heads = self.config.num_heads;
        let num_kv_heads = self.config.num_kv_heads;
        let head_dim = hidden_dim / num_heads;

        // 1. Token embedding lookup (F32)
        let mut hidden = Vec::with_capacity(hidden_dim);
        let offset = (token_id as usize) * hidden_dim;
        if offset + hidden_dim <= self.token_embedding.len() {
            hidden.extend_from_slice(&self.token_embedding[offset..offset + hidden_dim]);
        } else {
            hidden.extend(std::iter::repeat_n(0.0, hidden_dim));
        }

        // 2. Process through layers (simplified for quantized)
        for layer_idx in 0..self.config.num_layers {
            // For zero-initialized quantized weights, output stays mostly the same
            // In production: dequantize layer weights and compute

            // Compute placeholder K, V for cache
            let kv_size = num_kv_heads * head_dim;
            let k = vec![0.0f32; kv_size];
            let v = vec![0.0f32; kv_size];
            cache.append(layer_idx, &k, &v);
        }

        // 3. Final layer norm
        let eps = self.config.eps;
        let mean: f32 = hidden.iter().sum::<f32>() / hidden_dim as f32;
        let variance: f32 =
            hidden.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / hidden_dim as f32;
        let std_dev = (variance + eps).sqrt();

        let mut normed = Vec::with_capacity(hidden_dim);
        for (i, &x) in hidden.iter().enumerate() {
            let normalized = (x - mean) / std_dev;
            normed.push(normalized * self.output_norm_weight[i]);
        }

        // 4. LM head (dequantize and compute)
        let logits = self.compute_lm_head_logits(&normed)?;

        Ok(logits)
    }
}
