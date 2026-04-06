
impl Model {
    /// Create a new transformer model
    ///
    /// # Arguments
    ///
    /// * `config` - Model configuration
    ///
    /// # Errors
    ///
    /// Returns error if configuration is invalid
    pub fn new(config: ModelConfig) -> Result<Self> {
        let embedding = Embedding::new(config.vocab_size, config.hidden_dim)?;

        let mut blocks = Vec::with_capacity(config.num_layers);
        for _ in 0..config.num_layers {
            blocks.push(TransformerBlock::new(
                config.hidden_dim,
                config.num_heads,
                config.intermediate_dim,
                config.eps,
            )?);
        }

        let final_norm = LayerNorm::new(config.hidden_dim, config.eps)?;
        let lm_head = Linear::new(config.hidden_dim, config.vocab_size)?;

        Ok(Self {
            embedding,
            blocks,
            final_norm,
            lm_head,
            config,
        })
    }

    /// Forward pass through the model
    ///
    /// # Arguments
    ///
    /// * `token_ids` - Input token IDs
    ///
    /// # Returns
    ///
    /// Logits tensor with shape `[seq_len, vocab_size]`
    ///
    /// # Errors
    ///
    /// Returns error if input is invalid
    pub fn forward(&self, token_ids: &[usize]) -> Result<Tensor<f32>> {
        // Embed tokens
        let mut hidden = self.embedding.forward(token_ids)?;

        // Pass through transformer blocks
        for block in &self.blocks {
            hidden = block.forward(&hidden)?;
        }

        // Final layer norm
        hidden = self.final_norm.forward(&hidden)?;

        // Project to vocabulary
        self.lm_head.forward(&hidden)
    }

    /// Get model configuration
    #[must_use]
    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    /// Get mutable reference to embedding layer
    pub fn embedding_mut(&mut self) -> &mut Embedding {
        &mut self.embedding
    }

    /// Get mutable reference to transformer blocks
    pub fn blocks_mut(&mut self) -> &mut [TransformerBlock] {
        &mut self.blocks
    }

    /// Get mutable reference to final layer norm
    pub fn final_norm_mut(&mut self) -> &mut LayerNorm {
        &mut self.final_norm
    }

    /// Get mutable reference to LM head
    pub fn lm_head_mut(&mut self) -> &mut Linear {
        &mut self.lm_head
    }

    /// Get number of parameters in the model (approximate)
    #[must_use]
    pub fn num_parameters(&self) -> usize {
        let embed_params = self.config.vocab_size * self.config.hidden_dim;
        let block_params = self.config.num_layers
            * (
                // Attention (Q, K, V, O projections would be here in full impl)
                // For now just count layer norms and FFN
                2 * self.config.hidden_dim  // Layer norm weights
                + self.config.hidden_dim * self.config.intermediate_dim  // fc1
                + self.config.intermediate_dim * self.config.hidden_dim
                // fc2
            );
        let head_params = self.config.hidden_dim * self.config.vocab_size;

        embed_params + block_params + head_params
    }

    /// Generate tokens autoregressively
    ///
    /// # Arguments
    ///
    /// * `prompt` - Initial token IDs
    /// * `config` - Generation configuration
    ///
    /// # Returns
    ///
    /// Vector of generated token IDs (including prompt)
    ///
    /// # Errors
    ///
    /// Returns error if generation fails
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let generated = model.generate(&[1, 2, 3], &GenerationConfig::greedy())?;
    /// ```
    pub fn generate(&self, prompt: &[usize], config: &GenerationConfig) -> Result<Vec<usize>> {
        if prompt.is_empty() {
            return Err(RealizarError::InvalidShape {
                reason: "Prompt cannot be empty".to_string(),
            });
        }

        let mut tokens = prompt.to_vec();
        let mut rng_state = config.seed.unwrap_or(42);

        for _ in 0..config.max_tokens {
            // Forward pass
            let logits = self.forward(&tokens)?;

            // Get logits for last position
            let seq_len = tokens.len();
            let vocab_size = self.config.vocab_size;
            let last_logits_start = (seq_len - 1) * vocab_size;
            let last_logits = &logits.data()[last_logits_start..last_logits_start + vocab_size];

            let last_logits_tensor = Tensor::from_vec(vec![vocab_size], last_logits.to_vec())?;

            // Simple LCG for random number generation
            rng_state = rng_state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            #[allow(clippy::cast_precision_loss)]
            let rng_value = (rng_state >> 33) as f32 / (1u64 << 31) as f32;

            // Sample next token
            let next_token = sample_token(&last_logits_tensor, config, rng_value)?;

            // Check for EOS
            if let Some(eos_id) = config.eos_token_id {
                if next_token == eos_id {
                    break;
                }
            }

            tokens.push(next_token);
        }

        Ok(tokens)
    }
}
