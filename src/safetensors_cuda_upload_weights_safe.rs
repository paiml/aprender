impl SafeTensorsCudaModel {

    /// GH-201: Upload only essential weights for streaming mode.
    ///
    /// Uploads:
    /// - LM head (always needed for logits)
    /// - Output norm (always needed)
    /// - Layer norms (small, needed for RMS norm)
    /// - Biases (small, kept on CPU)
    ///
    /// Does NOT upload:
    /// - Per-layer QKV, O, FFN weights (streamed on-demand)
    #[allow(clippy::type_complexity)]
    fn upload_weights_streaming(
        executor: &mut CudaExecutor,
        st_model: &MappedSafeTensorsModel,
        config: &SafeTensorsCudaConfig,
    ) -> Result<(
        Vec<f32>,
        std::collections::HashMap<String, Vec<f32>>,
        std::collections::HashMap<String, Vec<f32>>,
        std::collections::HashMap<String, Vec<f32>>,
    )> {
        let hidden_dim = config.hidden_dim;
        let num_layers = config.num_layers;
        let num_kv_heads = config.num_kv_heads;
        let num_heads = config.num_heads;
        let vocab_size = config.vocab_size;
        let head_dim = hidden_dim / num_heads;
        let kv_dim = num_kv_heads * head_dim;

        let mut gamma_cache = std::collections::HashMap::new();
        let mut qkv_bias_cache = std::collections::HashMap::new();
        let mut o_bias_cache = std::collections::HashMap::new();

        // Embedding table (keep on CPU for token lookup)
        let embedding = st_model.get_tensor_auto("model.embed_tokens.weight")?;

        // Output norm - upload to GPU AND keep CPU copy
        let output_norm = st_model.get_tensor_auto("model.norm.weight")?;
        gamma_cache.insert("output".to_string(), output_norm.clone());
        executor.preload_output_norm(&output_norm).map_err(|e| {
            RealizarError::UnsupportedOperation {
                operation: "preload_output_norm".to_string(),
                reason: format!("Failed to upload output_norm: {e}"),
            }
        })?;

        // LM head (always needed) - upload to GPU
        let lm_head = if config.tie_word_embeddings {
            Self::transpose_for_gemm(&embedding, vocab_size, hidden_dim)
        } else if st_model.has_tensor("lm_head.weight") {
            let raw = st_model.get_tensor_auto("lm_head.weight")?;
            Self::transpose_for_gemm(&raw, vocab_size, hidden_dim)
        } else {
            Self::transpose_for_gemm(&embedding, vocab_size, hidden_dim)
        };
        executor.load_weights("lm_head", &lm_head).map_err(|e| {
            RealizarError::UnsupportedOperation {
                operation: "load_weights".to_string(),
                reason: format!("Failed to upload lm_head: {e}"),
            }
        })?;

        // Per-layer: only upload norms and cache biases (small tensors)
        // Layer weights (QKV, O, FFN) will be streamed on-demand
        for layer_idx in 0..num_layers {
            let prefix = format!("model.layers.{layer_idx}");

            // Attention norm (small - upload to GPU)
            let attn_norm =
                st_model.get_tensor_auto(&format!("{prefix}.input_layernorm.weight"))?;
            gamma_cache.insert(format!("attn.{layer_idx}"), attn_norm.clone());
            let attn_norm_key = format!("blk.{layer_idx}.attn_norm.gamma");
            executor
                .cache_rmsnorm_gamma(&attn_norm_key, &attn_norm)
                .map_err(|e| RealizarError::UnsupportedOperation {
                    operation: "cache_rmsnorm_gamma".to_string(),
                    reason: format!("Failed to upload layer {layer_idx} attn_norm: {e}"),
                })?;

            // FFN norm (small - upload to GPU)
            let ffn_norm =
                st_model.get_tensor_auto(&format!("{prefix}.post_attention_layernorm.weight"))?;
            gamma_cache.insert(format!("ffn.{layer_idx}"), ffn_norm.clone());
            let ffn_norm_key = format!("blk.{layer_idx}.ffn_norm.gamma");
            executor
                .cache_rmsnorm_gamma(&ffn_norm_key, &ffn_norm)
                .map_err(|e| RealizarError::UnsupportedOperation {
                    operation: "cache_rmsnorm_gamma".to_string(),
                    reason: format!("Failed to upload layer {layer_idx} ffn_norm: {e}"),
                })?;

            // Cache biases on CPU (small)
            let q_bias = st_model
                .get_tensor_auto(&format!("{prefix}.self_attn.q_proj.bias"))
                .unwrap_or_else(|_| vec![0.0f32; hidden_dim]);
            let k_bias = st_model
                .get_tensor_auto(&format!("{prefix}.self_attn.k_proj.bias"))
                .unwrap_or_else(|_| vec![0.0f32; kv_dim]);
            let v_bias = st_model
                .get_tensor_auto(&format!("{prefix}.self_attn.v_proj.bias"))
                .unwrap_or_else(|_| vec![0.0f32; kv_dim]);
            let mut qkv_bias = Vec::with_capacity(hidden_dim + 2 * kv_dim);
            qkv_bias.extend_from_slice(&q_bias);
            qkv_bias.extend_from_slice(&k_bias);
            qkv_bias.extend_from_slice(&v_bias);
            qkv_bias_cache.insert(format!("qkv_bias.{layer_idx}"), qkv_bias);

            if let Ok(o_bias) = st_model.get_tensor_auto(&format!("{prefix}.self_attn.o_proj.bias"))
            {
                o_bias_cache.insert(format!("o_bias.{layer_idx}"), o_bias);
            }
        }

        Ok((embedding, gamma_cache, qkv_bias_cache, o_bias_cache))
    }

    /// Get GPU device name.
    #[must_use]
    pub fn device_name(&self) -> &str {
        &self.device_name
    }

    /// Get VRAM in MB.
    #[must_use]
    pub fn vram_mb(&self) -> u64 {
        (self.memory_info.1 / (1024 * 1024)) as u64
    }

    /// Get model configuration.
    #[must_use]
    pub fn config(&self) -> &SafeTensorsCudaConfig {
        &self.config
    }

    /// Reset KV cache for new conversation.
    pub fn reset_kv_cache(&mut self) {
        self.kv_position = 0;
        self.executor.reset_kv_cache_gpu();
    }

    /// Generate tokens with GPU acceleration (F-QUAL-066 to F-QUAL-080).
    ///
    /// # Arguments
    ///
    /// * `input_ids` - Input token IDs
    /// * `max_tokens` - Maximum tokens to generate
    /// * `eos_id` - End-of-sequence token ID
    ///
    /// # Returns
    ///
    /// All tokens (input + generated).
    pub fn generate(
        &mut self,
        input_ids: &[u32],
        max_tokens: usize,
        eos_id: u32,
    ) -> Result<Vec<u32>> {
        let mut tokens = input_ids.to_vec();

        // PMAT-120 FIX: Prefill processes all input tokens, keeping logits from last one.
        // Previously, logits were discarded during prefill and the last input token was
        // processed AGAIN in the decode loop, causing duplicate KV entries and wrong RoPE.
        let mut last_logits = vec![];
        for &token in input_ids {
            last_logits = self.forward_single(token)?;
        }

        // Sample first new token from prefill logits (not from re-processing last input)
        let first_next = last_logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map_or(0, |(i, _)| i as u32);

        if first_next == eos_id {
            return Ok(tokens);
        }
        tokens.push(first_next);

        // Decode: generate remaining tokens
        for _ in 1..max_tokens {
            let last_token = *tokens.last().unwrap_or(&1);
            let logits = self.forward_single(last_token)?;

            // Greedy sampling (argmax)
            let next_token = logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map_or(0, |(i, _)| i as u32);

            if next_token == eos_id {
                break;
            }

            tokens.push(next_token);
        }

        Ok(tokens)
    }

    /// Forward pass for a single token.
    fn forward_single(&mut self, token: u32) -> Result<Vec<f32>> {
        let hidden_dim = self.config.hidden_dim;
        let vocab_size = self.config.vocab_size;

        // 1. Embedding lookup (CPU)
        let start = (token as usize) * hidden_dim;
        let end = start + hidden_dim;
        if end > self.embedding_cache.len() {
            return Err(RealizarError::UnsupportedOperation {
                operation: "embedding_lookup".to_string(),
                reason: format!("Token {} out of range", token),
            });
        }
        let mut hidden = self.embedding_cache[start..end].to_vec();

        // 2. Transformer layers (GPU)
        // Position tracking is handled internally by incremental_attention_gpu
        for layer_idx in 0..self.config.num_layers {
            hidden = self.forward_layer(layer_idx, &hidden)?;
        }

        // 3. Output norm (CPU path)
        hidden = self.apply_rms_norm_cpu(&hidden)?;

        // 4. LM head projection (GPU) - C = A × B where B is cached lm_head
        let mut logits = vec![0.0f32; vocab_size];
        self.executor
            .gemm_b_cached(
                "lm_head",
                &hidden,           // A: [1, hidden_dim] row vector
                &mut logits,       // C: [1, vocab_size]
                1,                 // m = 1 (single token)
                vocab_size as u32, // n = vocab_size
                hidden_dim as u32, // k = hidden_dim
            )
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "lm_head".to_string(),
                reason: format!("LM head GEMM failed: {e}"),
            })?;

        self.kv_position += 1;
        Ok(logits)
    }

    /// GH-201: Load layer weights on-demand for streaming mode.
    ///
    /// In streaming mode, we don't pre-cache all layer weights. Instead, we load
    /// them from the SafeTensors file on-demand for each layer.
    fn ensure_layer_weights_loaded(&mut self, layer_idx: usize) -> Result<()> {
        if !self.streaming_mode {
            return Ok(()); // Weights already pre-cached
        }

        let model_path =
            self.model_path
                .as_ref()
                .ok_or_else(|| RealizarError::UnsupportedOperation {
                    operation: "ensure_layer_weights_loaded".to_string(),
                    reason: "Streaming mode enabled but model_path not set".to_string(),
                })?;

        // Reload SafeTensors (mmap is cheap, it just maps the file)
        let st_model = MappedSafeTensorsModel::load(model_path)?;

        let hidden_dim = self.config.hidden_dim;
        let num_heads = self.config.num_heads;
        let num_kv_heads = self.config.num_kv_heads;
        let intermediate_dim = self.config.intermediate_dim;
        let head_dim = hidden_dim / num_heads;
        let kv_dim = num_kv_heads * head_dim;

        let prefix = format!("model.layers.{layer_idx}");

        // Upload QKV weights (reuses buffer slot from previous layer)
        let q = st_model.get_tensor_auto(&format!("{prefix}.self_attn.q_proj.weight"))?;
        let k = st_model.get_tensor_auto(&format!("{prefix}.self_attn.k_proj.weight"))?;
        let v = st_model.get_tensor_auto(&format!("{prefix}.self_attn.v_proj.weight"))?;
        let qkv = Self::concat_qkv_transposed(&q, &k, &v, hidden_dim, kv_dim);
        self.executor
            .load_weights(&format!("blk.{layer_idx}.attn_qkv"), &qkv)
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "load_weights".to_string(),
                reason: format!("Failed to upload layer {layer_idx} qkv: {e}"),
            })?;

        // Upload O projection
        let o_raw = st_model.get_tensor_auto(&format!("{prefix}.self_attn.o_proj.weight"))?;
        let o = Self::transpose_for_gemm(&o_raw, hidden_dim, hidden_dim);
        self.executor
            .load_weights(&format!("blk.{layer_idx}.attn_output"), &o)
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "load_weights".to_string(),
                reason: format!("Failed to upload layer {layer_idx} attn_output: {e}"),
            })?;

        // Upload FFN gate
        let gate_raw = st_model.get_tensor_auto(&format!("{prefix}.mlp.gate_proj.weight"))?;
        let gate = Self::transpose_for_gemm(&gate_raw, intermediate_dim, hidden_dim);
        self.executor
            .load_weights(&format!("blk.{layer_idx}.ffn_gate"), &gate)
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "load_weights".to_string(),
                reason: format!("Failed to upload layer {layer_idx} ffn_gate: {e}"),
            })?;

        // Upload FFN up
        let up_raw = st_model.get_tensor_auto(&format!("{prefix}.mlp.up_proj.weight"))?;
        let up = Self::transpose_for_gemm(&up_raw, intermediate_dim, hidden_dim);
        self.executor
            .load_weights(&format!("blk.{layer_idx}.ffn_up"), &up)
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "load_weights".to_string(),
                reason: format!("Failed to upload layer {layer_idx} ffn_up: {e}"),
            })?;

        // Upload FFN down
        let down_raw = st_model.get_tensor_auto(&format!("{prefix}.mlp.down_proj.weight"))?;
        let down = Self::transpose_for_gemm(&down_raw, hidden_dim, intermediate_dim);
        self.executor
            .load_weights(&format!("blk.{layer_idx}.ffn_down"), &down)
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "load_weights".to_string(),
                reason: format!("Failed to upload layer {layer_idx} ffn_down: {e}"),
            })?;

        Ok(())
    }
}
