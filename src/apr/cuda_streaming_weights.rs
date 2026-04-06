impl AprV2ModelCuda {

    /// GH-201: Pre-cache only essential weights in streaming mode.
    ///
    /// In streaming mode, we only cache:
    /// - LM head (required for every token)
    /// - Output norm gamma (required for every token)
    ///
    /// Per-layer weights are loaded on-demand via `ensure_layer_weights_loaded()`.
    /// This reduces VRAM usage from ~6GB to ~1.2GB for 1.5B models.
    fn pre_cache_weights_streaming(&mut self) -> Result<()> {
        let hidden_dim = self.model.metadata.hidden_size.unwrap_or(0);
        let _num_layers = self.model.metadata.num_layers.unwrap_or(0);
        let vocab_size = self.model.metadata.vocab_size.unwrap_or(0);

        if hidden_dim == 0 {
            return Ok(()); // Non-transformer model
        }

        let mut total_bytes = 0usize;

        // Cache output norm (always needed)
        let output_norm_patterns = [
            "model.norm.weight",
            "norm.weight",
            "transformer.ln_f.weight",
            "output_norm.weight",
        ];
        if let Ok(src_name) = self.model.find_tensor_name(&output_norm_patterns) {
            if let Ok(gamma) = self.model.get_tensor_f32(&src_name) {
                if let Ok(bytes) = self
                    .executor
                    .cache_rmsnorm_gamma("output_norm.gamma", &gamma)
                {
                    total_bytes += bytes;
                }
            }
        }

        // Cache LM head (always needed - may be quantized or F32)
        let lm_head_patterns = [
            "lm_head.weight",
            "output.weight",
            "token_embd.weight", // GGUF (tied embeddings)
        ];
        if let Ok(src_name) = self.model.find_tensor_name(&lm_head_patterns) {
            if let Some(entry) = self.model.get_tensor(&src_name) {
                if let Some(qtype) = dtype_to_ggml_qtype(&entry.dtype) {
                    // Quantized LM head
                    if let Ok(bytes) = self.model.get_tensor_bytes(&src_name) {
                        if let Ok(size) = self.executor.load_quantized_weights_with_type(
                            "output.weight",
                            bytes,
                            qtype,
                        ) {
                            total_bytes += size;
                        }
                    }
                } else if let Ok(w) = self.model.get_tensor_f32(&src_name) {
                    // F32 LM head
                    // SAFETY: f32 slice to u8 view - valid because f32 has no padding
                    let w_bytes: &[u8] = unsafe {
                        std::slice::from_raw_parts(
                            w.as_ptr().cast::<u8>(),
                            w.len() * std::mem::size_of::<f32>(),
                        )
                    };
                    if let Ok(size) =
                        self.executor
                            .load_quantized_weights_with_type("output.weight", w_bytes, 0)
                    {
                        total_bytes += size;
                    }
                }
            }
        }

        let lm_head_mb = vocab_size * hidden_dim * 4 / (1024 * 1024);
        eprintln!(
            "[AprV2ModelCuda] GH-201: Streaming mode - cached {} MB (LM head ~{} MB, norms)",
            total_bytes / (1024 * 1024),
            lm_head_mb
        );
        eprintln!("[AprV2ModelCuda] GH-201: Layer weights will be streamed on-demand");

        Ok(())
    }

    /// GH-201: Ensure a specific layer's weights are loaded on GPU.
    ///
    /// In streaming mode, this uploads the layer's weights if not already cached.
    /// The previously cached layer's weights are replaced.
    ///
    /// In full cache mode, this is a no-op (all weights pre-cached).
    fn ensure_layer_weights_loaded(&mut self, layer_idx: usize) -> Result<()> {
        if !self.streaming_mode {
            return Ok(()); // Full cache mode - weights already on GPU
        }

        // Check if this layer is already cached
        if self.cached_streaming_layer == Some(layer_idx) {
            return Ok(());
        }

        let hidden_dim = self.model.metadata.hidden_size.unwrap_or(0);
        let num_heads = self.model.metadata.num_heads.unwrap_or(1);
        let num_kv_heads = self.model.metadata.num_kv_heads.unwrap_or(num_heads);
        let _intermediate_dim = self
            .model
            .metadata
            .intermediate_size
            .unwrap_or(hidden_dim * 4);
        let head_dim = if num_heads > 0 {
            hidden_dim / num_heads
        } else {
            0
        };
        let kv_dim = num_kv_heads * head_dim;

        let prefix = format!("blk.{layer_idx}");
        let mut total_bytes = 0usize;

        // Clear previous layer's weights from GPU cache
        // (The executor will reuse the memory)

        // Helper to upload a weight tensor
        let upload_weight = |executor: &mut crate::cuda::CudaExecutor,
                             model: &AprV2Model,
                             src_name: &str,
                             cache_name: &str|
         -> usize {
            if let Some(entry) = model.get_tensor(src_name) {
                if let Some(qtype) = dtype_to_ggml_qtype(&entry.dtype) {
                    // Quantized weight
                    if let Ok(bytes) = model.get_tensor_bytes(src_name) {
                        executor
                            .load_quantized_weights_with_type(cache_name, bytes, qtype)
                            .unwrap_or(0)
                    } else {
                        0
                    }
                } else if let Ok(weights) = model.get_tensor_f32(src_name) {
                    // F32 weight - transpose for GPU GEMM
                    let final_weights = if entry.shape.len() == 2 {
                        let rows = entry.shape[0];
                        let cols = entry.shape[1];
                        let mut transposed = vec![0.0f32; weights.len()];
                        for i in 0..rows {
                            for j in 0..cols {
                                transposed[j * rows + i] = weights[i * cols + j];
                            }
                        }
                        transposed
                    } else {
                        weights
                    };
                    executor
                        .load_weights(cache_name, &final_weights)
                        .unwrap_or(0)
                } else {
                    0
                }
            } else {
                0
            }
        };

        // Upload attention weights
        let weight_mappings = [
            (
                vec![
                    format!("model.layers.{layer_idx}.self_attn.q_proj.weight"),
                    format!("blk.{layer_idx}.attn_q.weight"),
                ],
                "attn_q.weight",
            ),
            (
                vec![
                    format!("model.layers.{layer_idx}.self_attn.k_proj.weight"),
                    format!("blk.{layer_idx}.attn_k.weight"),
                ],
                "attn_k.weight",
            ),
            (
                vec![
                    format!("model.layers.{layer_idx}.self_attn.v_proj.weight"),
                    format!("blk.{layer_idx}.attn_v.weight"),
                ],
                "attn_v.weight",
            ),
            (
                vec![
                    format!("model.layers.{layer_idx}.self_attn.o_proj.weight"),
                    format!("blk.{layer_idx}.attn_output.weight"),
                ],
                "attn_output.weight",
            ),
            (
                vec![
                    format!("model.layers.{layer_idx}.mlp.gate_proj.weight"),
                    format!("blk.{layer_idx}.ffn_gate.weight"),
                ],
                "ffn_gate.weight",
            ),
            (
                vec![
                    format!("model.layers.{layer_idx}.mlp.up_proj.weight"),
                    format!("blk.{layer_idx}.ffn_up.weight"),
                ],
                "ffn_up.weight",
            ),
            (
                vec![
                    format!("model.layers.{layer_idx}.mlp.down_proj.weight"),
                    format!("blk.{layer_idx}.ffn_down.weight"),
                ],
                "ffn_down.weight",
            ),
        ];

        for (patterns, suffix) in weight_mappings {
            let patterns_ref: Vec<&str> = patterns.iter().map(String::as_str).collect();
            if let Ok(src_name) = self.model.find_tensor_name(&patterns_ref) {
                let cache_name = format!("{prefix}.{suffix}");
                total_bytes +=
                    upload_weight(&mut self.executor, &self.model, &src_name, &cache_name);
            }
        }

        // Handle fused QKV if present
        let fused_qkv_patterns = vec![format!(
            "model.layers.{layer_idx}.self_attn.qkv_proj.weight"
        )];
        let fused_patterns_ref: Vec<&str> = fused_qkv_patterns.iter().map(String::as_str).collect();
        if let Ok(src_name) = self.model.find_tensor_name(&fused_patterns_ref) {
            if let Ok(qkv_weight) = self.model.get_tensor_f32(&src_name) {
                let q_size = hidden_dim * hidden_dim;
                let k_size = kv_dim * hidden_dim;
                let v_size = kv_dim * hidden_dim;

                if qkv_weight.len() >= q_size + k_size + v_size {
                    let q_weight: Vec<f32> = qkv_weight[0..q_size].to_vec();
                    let k_weight: Vec<f32> = qkv_weight[q_size..q_size + k_size].to_vec();
                    let v_weight: Vec<f32> =
                        qkv_weight[q_size + k_size..q_size + k_size + v_size].to_vec();

                    // Transpose for GPU GEMM
                    let q_weight_t = transpose_matrix(&q_weight, hidden_dim, hidden_dim);
                    let k_weight_t = transpose_matrix(&k_weight, kv_dim, hidden_dim);
                    let v_weight_t = transpose_matrix(&v_weight, kv_dim, hidden_dim);

                    let q_cache_name = format!("blk.{layer_idx}.attn_q.weight");
                    let k_cache_name = format!("blk.{layer_idx}.attn_k.weight");
                    let v_cache_name = format!("blk.{layer_idx}.attn_v.weight");

                    total_bytes += self
                        .executor
                        .load_weights(&q_cache_name, &q_weight_t)
                        .unwrap_or(0);
                    total_bytes += self
                        .executor
                        .load_weights(&k_cache_name, &k_weight_t)
                        .unwrap_or(0);
                    total_bytes += self
                        .executor
                        .load_weights(&v_cache_name, &v_weight_t)
                        .unwrap_or(0);
                }
            }
        }

        // Upload RMSNorm gamma weights
        let norm_mappings = [
            (
                vec![
                    format!("model.layers.{layer_idx}.input_layernorm.weight"),
                    format!("blk.{layer_idx}.attn_norm.weight"),
                ],
                "attn_norm.gamma",
            ),
            (
                vec![
                    format!("model.layers.{layer_idx}.post_attention_layernorm.weight"),
                    format!("blk.{layer_idx}.ffn_norm.weight"),
                ],
                "ffn_norm.gamma",
            ),
            // GH-279: QK norm weights (Qwen3 per-head RMSNorm on Q and K)
            (
                vec![
                    format!("model.layers.{layer_idx}.self_attn.q_norm.weight"),
                    format!("blk.{layer_idx}.attn_q_norm.weight"),
                ],
                "attn_q_norm.gamma",
            ),
            (
                vec![
                    format!("model.layers.{layer_idx}.self_attn.k_norm.weight"),
                    format!("blk.{layer_idx}.attn_k_norm.weight"),
                ],
                "attn_k_norm.gamma",
            ),
        ];

        for (patterns, suffix) in norm_mappings {
            let patterns_ref: Vec<&str> = patterns.iter().map(String::as_str).collect();
            if let Ok(src_name) = self.model.find_tensor_name(&patterns_ref) {
                if let Ok(gamma) = self.model.get_tensor_f32(&src_name) {
                    let cache_name = format!("{prefix}.{suffix}");
                    total_bytes += self
                        .executor
                        .cache_rmsnorm_gamma(&cache_name, &gamma)
                        .unwrap_or(0);
                }
            }
        }

        self.cached_streaming_layer = Some(layer_idx);

        // Only log for first few layers to avoid spam
        if layer_idx < 3 {
            eprintln!(
                "[AprV2ModelCuda] GH-201: Streamed layer {} weights ({} KB)",
                layer_idx,
                total_bytes / 1024
            );
        }

        Ok(())
    }

    /// GH-201: Check if model is in streaming mode.
    #[must_use]
    pub fn is_streaming_mode(&self) -> bool {
        self.streaming_mode
    }

    /// Pre-cache embedding table for fast token lookup.
    ///
    /// This reads the embedding table once and stores it in memory, eliminating
    /// repeated disk/mmap reads during generation (~450ms → ~0.05ms per token).
    fn cache_embeddings(&mut self) -> Result<()> {
        let embed_name = self.model.find_tensor_name(&[
            "model.embed_tokens.weight",
            "embed_tokens.weight",
            "token_embd.weight", // GGUF naming
        ])?;

        let embeddings = self.model.get_tensor_f32(&embed_name)?;
        let embed_mb = embeddings.len() * 4 / (1024 * 1024);
        eprintln!("[AprV2ModelCuda] Cached embedding table: {} MB", embed_mb);

        self.embedding_cache = Some(embeddings);
        Ok(())
    }

    /// Get embedding for a token ID from cache.
    #[inline]
    fn get_embedding(&self, token_id: u32) -> Option<&[f32]> {
        self.embedding_cache.as_ref().and_then(|cache| {
            let offset = (token_id as usize) * self.hidden_dim;
            if offset + self.hidden_dim <= cache.len() {
                Some(&cache[offset..offset + self.hidden_dim])
            } else {
                None
            }
        })
    }

    /// Check if weights are cached on GPU.
    #[must_use]
    pub fn weights_cached(&self) -> bool {
        self.executor.cached_weight_count() > 0
    }

    /// Get total cached weight size in MB.
    #[must_use]
    pub fn cached_weight_mb(&self) -> usize {
        self.executor.cached_weight_bytes() / (1024 * 1024)
    }
}
