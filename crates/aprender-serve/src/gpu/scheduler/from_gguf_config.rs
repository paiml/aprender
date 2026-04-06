impl GpuModel {

    /// Create GPU model from GGUF config (M13: Real Model Loading)
    ///
    /// This is a convenience constructor that creates a model with zero-initialized
    /// weights from a config. Use `from_mapped_gguf()` to load actual weights.
    ///
    /// # Arguments
    ///
    /// * `config` - Model configuration
    ///
    /// # Errors
    ///
    /// Returns error if GPU initialization fails
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let config = GpuModelConfig {
    ///     vocab_size: 32000,
    ///     hidden_dim: 4096,
    ///     num_heads: 32,
    ///     num_kv_heads: 32,
    ///     num_layers: 32,
    ///     intermediate_dim: 11008,
    ///     eps: 1e-5,
    ///     rope_theta: 10000.0,
    ///     explicit_head_dim: None,
    ///     layer_types: None,
    ///     linear_key_head_dim: None,
    ///     linear_value_head_dim: None,
    ///     linear_num_key_heads: None,
    ///     linear_num_value_heads: None,
    ///     linear_conv_kernel_dim: None,
    ///     constraints: None,
    ///     num_experts: None,
    ///     num_experts_per_tok: None,
    ///     expert_intermediate_size: None,
    /// };
    /// let model = GpuModel::from_gguf_config(config)?;
    /// ```
    pub fn from_gguf_config(config: GpuModelConfig) -> Result<Self> {
        // Delegate to new() which handles initialization
        Self::new(config)
    }

    /// Load GPU model from memory-mapped GGUF file (M13: Real Model Loading)
    ///
    /// This is the primary method for loading real GGUF models to GPU.
    /// It dequantizes weights on-the-fly and uploads them to GPU buffers.
    ///
    /// # Arguments
    ///
    /// * `mapped` - Memory-mapped GGUF model
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Required tensors are missing
    /// - Tensor shapes don't match expected dimensions
    /// - GPU initialization or upload fails
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let mapped = MappedGGUFModel::from_path("model.gguf")?;
    /// let model = GpuModel::from_mapped_gguf(&mapped)?;
    /// let logits = model.forward_gpu_owned(&[1, 2, 3])?;
    /// ```
    pub fn from_mapped_gguf(mapped: &crate::gguf::MappedGGUFModel) -> Result<Self> {
        let w = super::loading::load_weights_from_gguf(mapped)?;

        // GH-279: Contract gate — validate architecture and dimensions
        let _proof = crate::contract_gate::validate_model_load_basic(
            w.config.constraints.as_ref().map_or("llama", |c| {
                if c.has_qk_norm { "qwen3" } else if c.has_bias { "qwen2" } else { "llama" }
            }),
            w.config.num_layers,
            w.config.hidden_dim,
            w.config.num_heads,
            w.config.num_kv_heads,
            w.config.intermediate_dim,
            w.config.vocab_size,
        )
        .map_err(crate::contract_gate::gate_error)?;

        let scheduler = HybridScheduler::new()?;

        // Pre-compute transposed LM head for fast CPU inference
        let lm_head_weight_t =
            Self::transpose_weights(&w.lm_head_weight, w.config.hidden_dim, w.config.vocab_size);

        Ok(Self {
            embedding_weights: w.embedding_weights,
            block_weights: w.block_weights,
            final_norm_weight: w.final_norm_weight,
            final_norm_bias: w.final_norm_bias,
            lm_head_weight: w.lm_head_weight,
            lm_head_weight_t,
            lm_head_bias: w.lm_head_bias,
            scheduler,
            #[cfg(feature = "cuda")]
            cuda_scheduler: None,
            config: w.config,
            attention_buffers: None,
            test_executor: None,
            linear_attn_state: None,
        })
    }

    /// Create GpuModel from pre-extracted APR weights (PMAT-106)
    ///
    /// This constructor is used by `AprToGpuAdapter` to create a `GpuModel`
    /// from dequantized APR weights.
    ///
    /// # Arguments
    ///
    /// * `config` - GPU model configuration
    /// * `embedding_weights` - Token embedding weights
    /// * `block_weights` - Transformer block weights
    /// * `final_norm_weight` - Final layer norm weight
    /// * `final_norm_bias` - Final layer norm bias
    /// * `lm_head_weight` - LM head weight (row-major)
    /// * `lm_head_weight_t` - LM head weight transposed (for fast CPU inference)
    /// * `lm_head_bias` - LM head bias
    ///
    /// # Errors
    ///
    /// Returns error if GPU scheduler initialization fails
    #[allow(clippy::too_many_arguments)]
    /// Create GpuModel from validated APR weights (PMAT-284).
    ///
    /// Accepts `ValidatedGpuWeights` which proves dimensions match config.
    /// LM head newtypes (`LmHeadWeight` / `LmHeadWeightTransposed`) prevent
    /// the PMAT-216 argument swap at compile time.
    pub fn from_validated_weights(weights: ValidatedGpuWeights) -> Result<Self> {
        // GH-279: Contract gate — validate architecture and dimensions
        let _proof = crate::contract_gate::validate_model_load_basic(
            weights.config.constraints.as_ref().map_or("llama", |c| {
                if c.has_qk_norm { "qwen3" } else if c.has_bias { "qwen2" } else { "llama" }
            }),
            weights.config.num_layers,
            weights.config.hidden_dim,
            weights.config.num_heads,
            weights.config.num_kv_heads,
            weights.config.intermediate_dim,
            weights.config.vocab_size,
        )
        .map_err(crate::contract_gate::gate_error)?;

        let scheduler = HybridScheduler::new()?;

        #[cfg(feature = "cuda")]
        let cuda_scheduler = match CudaScheduler::new() {
            Ok(cs) => {
                eprintln!("[PHASE21] CudaScheduler initialized for APR model");
                Some(cs)
            },
            Err(e) => {
                eprintln!(
                    "[PHASE21] CudaScheduler init failed (using HybridScheduler fallback): {}",
                    e
                );
                None
            },
        };

        let lm_head_weight = weights.lm_head_weight.into_inner();
        let lm_head_weight_t = weights.lm_head_weight_t.into_inner();

        Ok(Self {
            embedding_weights: weights.embedding_weights,
            block_weights: weights.block_weights,
            final_norm_weight: weights.final_norm_weight,
            final_norm_bias: weights.final_norm_bias,
            lm_head_weight,
            lm_head_weight_t,
            lm_head_bias: weights.lm_head_bias,
            scheduler,
            #[cfg(feature = "cuda")]
            cuda_scheduler,
            config: weights.config,
            attention_buffers: None,
            test_executor: None,
            linear_attn_state: None,
        })
    }

    /// Legacy constructor — accepts raw Vec<f32> for backwards compatibility.
    ///
    /// Prefer `from_validated_weights()` for new code. This function retains
    /// the PMAT-216 runtime transpose check for callers that haven't migrated.
    pub fn from_apr_weights(
        config: GpuModelConfig,
        embedding_weights: Vec<f32>,
        block_weights: Vec<BlockWeights>,
        final_norm_weight: Vec<f32>,
        final_norm_bias: Vec<f32>,
        lm_head_weight: Vec<f32>,
        lm_head_weight_t: Vec<f32>,
        lm_head_bias: Vec<f32>,
    ) -> Result<Self> {
        // GH-279: Contract gate — validate architecture and dimensions
        let _proof = crate::contract_gate::validate_model_load_basic(
            config.constraints.as_ref().map_or("llama", |c| {
                if c.has_qk_norm { "qwen3" } else if c.has_bias { "qwen2" } else { "llama" }
            }),
            config.num_layers,
            config.hidden_dim,
            config.num_heads,
            config.num_kv_heads,
            config.intermediate_dim,
            config.vocab_size,
        )
        .map_err(crate::contract_gate::gate_error)?;

        let scheduler = HybridScheduler::new()?;

        #[cfg(feature = "cuda")]
        let cuda_scheduler = match CudaScheduler::new() {
            Ok(cs) => {
                eprintln!("[PHASE21] CudaScheduler initialized for APR model");
                Some(cs)
            },
            Err(e) => {
                eprintln!(
                    "[PHASE21] CudaScheduler init failed (using HybridScheduler fallback): {}",
                    e
                );
                None
            },
        };

        // PMAT-216: Runtime transpose check (legacy path only — new code should use
        // from_validated_weights with LmHeadWeight/LmHeadWeightTransposed newtypes)
        if lm_head_weight.len() >= config.hidden_dim
            && lm_head_weight_t.len() >= config.vocab_size * config.hidden_dim
        {
            let orig_00 = lm_head_weight[0];
            let trans_00 = lm_head_weight_t[0];
            if (orig_00 - trans_00).abs() > 1e-6 {
                return Err(RealizarError::InvalidShape {
                    reason: format!(
                        "PMAT-216: lm_head_weight[0,0]={} != lm_head_weight_t[0,0]={}. \
                         Arguments may be swapped.",
                        orig_00, trans_00
                    ),
                });
            }

            if config.hidden_dim > 1 && config.vocab_size > 1 {
                let orig_01 = lm_head_weight[1];
                let trans_01 = lm_head_weight_t[1];
                let trans_10 = lm_head_weight_t[config.vocab_size];

                let is_transposed = (orig_01 - trans_10).abs() < 1e-5;
                let is_same = (orig_01 - trans_01).abs() < 1e-5;

                if is_same && !is_transposed && (orig_01 - trans_01).abs() < 1e-6 {
                    let msg = "PMAT-216: lm_head_weight_t appears to NOT be transposed. Check argument order in from_apr_weights call.";
                    return Err(RealizarError::InvalidShape { reason: msg.to_string() });
                }
            }
        }

        Ok(Self {
            embedding_weights,
            block_weights,
            final_norm_weight,
            final_norm_bias,
            lm_head_weight,
            lm_head_weight_t,
            lm_head_bias,
            scheduler,
            #[cfg(feature = "cuda")]
            cuda_scheduler,
            config,
            attention_buffers: None,
            test_executor: None,
            linear_attn_state: None,
        })
    }

    /// Get model configuration (M13: Real Model Loading)
    #[must_use]
    pub fn config(&self) -> &GpuModelConfig {
        &self.config
    }

    // ============================================================================
    // Phase 8: Optimized Incremental Decoding (M17)
    // ============================================================================

    /// Create GPU model with pre-allocated attention buffers (M17)
    ///
    /// Allocates reusable buffers for incremental decoding, eliminating
    /// per-token memory allocation overhead.
    ///
    /// # Arguments
    ///
    /// * `config` - Model configuration
    /// * `max_seq_len` - Maximum sequence length to support
    ///
    /// # Errors
    ///
    /// Returns error if GPU initialization fails
    pub fn with_attention_buffers(config: GpuModelConfig, max_seq_len: usize) -> Result<Self> {
        let buffers = AttentionBuffers::new(&config, max_seq_len);
        let mut model = Self::new(config)?;
        model.attention_buffers = Some(buffers);
        Ok(model)
    }

    /// Check if model has pre-allocated attention buffers (M17)
    #[must_use]
    pub fn has_attention_buffers(&self) -> bool {
        self.attention_buffers.is_some()
    }

    /// Optimized text generation using pre-allocated buffers (M17)
    ///
    /// Uses the optimized incremental forward pass with pre-allocated buffers
    /// and batched multi-head attention for better performance.
    ///
    /// # Arguments
    ///
    /// * `prompt` - Initial token IDs
    /// * `config` - Generation configuration
    ///
    /// # Errors
    ///
    /// Returns error if generation fails
    pub fn generate_optimized(
        &mut self,
        prompt: &[usize],
        config: &GpuGenerateConfig,
    ) -> Result<Vec<usize>> {
        if prompt.is_empty() {
            return Err(RealizarError::InvalidShape {
                reason: "Prompt cannot be empty".to_string(),
            });
        }

        // Initialize KV cache
        // IMP-093: For GQA, use num_kv_heads since K/V have fewer heads than Q
        let head_dim = self.config.head_dim();
        let max_seq_len = self
            .attention_buffers
            .as_ref()
            .map_or(512, |b| b.max_seq_len);
        let mut kv_cache = StreamingKVCache::new(
            self.config.num_layers,
            max_seq_len,
            self.config.num_kv_heads, // GQA: K/V have fewer heads
            head_dim,
        );

        let mut tokens = prompt.to_vec();

        // Process prompt with cache - returns logits for final position only [vocab_size]
        let logits = self.forward_gpu_with_cache(prompt, &mut kv_cache)?;

        // Sample first token (logits is already for last position only)
        let mut next_token = if config.temperature == 0.0 || config.top_k == 1 {
            Self::argmax(&logits)
        } else {
            Self::sample_topk_generate(&logits, config.temperature, config.top_k)
        };

        if config.stop_tokens.contains(&next_token) {
            return Ok(tokens);
        }

        tokens.push(next_token);

        // Generate remaining tokens using optimized incremental forward
        for _ in 1..config.max_tokens {
            let logits = self.forward_gpu_incremental_optimized(next_token, &mut kv_cache)?;

            next_token = if config.temperature == 0.0 || config.top_k == 1 {
                Self::argmax(&logits)
            } else {
                Self::sample_topk_generate(&logits, config.temperature, config.top_k)
            };

            if config.stop_tokens.contains(&next_token) {
                break;
            }

            tokens.push(next_token);
        }

        Ok(tokens)
    }

    /// Optimized incremental forward pass using pre-allocated buffers (M17)
    ///
    /// Single-token forward pass optimized by:
    /// - Reusing pre-allocated attention buffers
    /// - Direct KV cache access without copying
    /// - Batched multi-head attention computation
    ///
    /// # Arguments
    ///
    /// * `token_id` - Single token to process
    /// * `kv_cache` - Mutable reference to KV cache
    ///
    /// # Errors
    ///
    /// Returns error if forward pass fails
    pub fn forward_gpu_incremental_optimized(
        &mut self,
        token_id: usize,
        kv_cache: &mut StreamingKVCache,
    ) -> Result<Vec<f32>> {
        if token_id >= self.config.vocab_size {
            return Err(RealizarError::InvalidShape {
                reason: format!(
                    "Token ID {} out of bounds (vocab_size={})",
                    token_id, self.config.vocab_size
                ),
            });
        }

        let hidden_dim = self.config.hidden_dim;

        // Get embedding for single token
        let offset = token_id * hidden_dim;
        let mut hidden: Vec<f32> = self.embedding_weights[offset..offset + hidden_dim].to_vec();

        // Process through all blocks with optimized attention
        for block_idx in 0..self.block_weights.len() {
            hidden = self.forward_block_incremental_optimized(&hidden, block_idx, kv_cache)?;
        }

        // Final layer norm
        hidden = self.layer_norm(&hidden, &self.final_norm_weight, &self.final_norm_bias);

        // LM head projection (single token)
        // IMP-090, IMP-096: Use CPU fallback with SIMD for large vocab
        let lm_head_elements = hidden_dim * self.config.vocab_size;
        let output = if exceeds_gpu_buffer_limit(lm_head_elements) {
            // IMP-096: CPU path with transposed weights + SIMD + fused bias
            cpu_matmul_transposed_simd(
                &hidden,
                &self.lm_head_weight_t,
                &self.lm_head_bias,
                hidden_dim,
                self.config.vocab_size,
            )
        } else {
            // IMP-1006: Use do_matmul to route to CudaScheduler when available
            let lm_weight = self.lm_head_weight.clone();
            let vocab_size = self.config.vocab_size;
            let logits = self.do_matmul(&hidden, &lm_weight, 1, hidden_dim, vocab_size)?;
            // Add bias
            logits
                .into_iter()
                .zip(self.lm_head_bias.iter())
                .map(|(l, &b)| l + b)
                .collect()
        };

        Ok(output)
    }
}
