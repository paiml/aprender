impl OwnedQuantizedModel {
    /// Forward pass for a single token using KV cache with BrickProfiler instrumentation (GH-104)
    ///
    /// Identical to `forward_single_with_cache()` but wraps each operation with
    /// `profiler.start()`/`profiler.stop()` using BrickId-aligned names.
    ///
    /// This method mirrors `forward_profiled()` (batch path) but for the autoregressive
    /// decode path. Operations profiled:
    /// - `Embedding` — token embedding + position embedding
    /// - `RmsNorm` / `LayerNorm` — attention norm (fused into QKV when possible)
    /// - `QkvProjection` — Q/K/V projection (fused norm+matmul for RMSNorm models)
    /// - `RopeEmbedding` — rotary position embedding
    /// - `AttentionScore` — cached GQA attention
    /// - `OutputProjection` — attention output projection
    /// - `FFN` — full FFN block (norm + gate/up + activation + down)
    /// - `LmHead` — final layer norm + LM head projection
    ///
    /// # Arguments
    /// * `token_id` - Single input token ID
    /// * `cache` - Mutable reference to KV cache
    /// * `position` - Position in sequence for RoPE
    /// * `profiler` - BrickProfiler instance for timing
    ///
    /// # Returns
    /// Logits for next token prediction `[vocab_size]`
    ///
    /// # Errors
    /// Returns error if tensor operations fail
    pub fn forward_single_with_cache_profiled(
        &self,
        token_id: u32,
        cache: &mut OwnedQuantizedKVCache,
        position: usize,
        profiler: &mut BrickProfiler,
    ) -> Result<Vec<f32>> {
        let hidden_dim = self.config.hidden_dim;

        // 1. Token embedding lookup
        profiler.start("Embedding");
        let mut hidden = self.embed(&[token_id]);

        // GH-278: Add learned position embedding for absolute encoding (GPT-2, BERT, whisper)
        if self.config.constraints.uses_absolute_positions() {
            if let Some(ref pos_emb) = self.position_embedding {
                let start = position * hidden_dim;
                let end = start + hidden_dim;
                if end <= pos_emb.len() {
                    for i in 0..hidden_dim {
                        hidden[i] += pos_emb[start + i];
                    }
                }
            }
        }
        profiler.stop("Embedding");

        // GH-278: Use contract-derived norm type.
        let use_rmsnorm = self.config.constraints.uses_rmsnorm();

        // PMAT-305/307: Pre-allocate workspace buffers — reused across all layers.
        let mut attn_out_buffer = vec![0.0f32; self.config.q_dim()];
        let mut o_proj_buffer = vec![0.0f32; hidden_dim];
        let mut ffn_down_buffer = vec![0.0f32; hidden_dim];
        // PMAT-307: QKV workspace — eliminates 28 Vec allocs per token
        let qkv_dim = self.config.q_dim() + 2 * self.config.kv_dim();
        let mut qkv_buffer = vec![0.0f32; qkv_dim];

        // 2. Process through transformer layers
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            profiler.set_current_layer(layer_idx);

            // 2a+2b. Fused attention layer norm + QKV projection
            profiler.start("QkvProjection");
            #[allow(unused_variables)]
            let qkv_actual_dim;
            let mut qkv = if use_rmsnorm {
                match &layer.qkv_weight {
                    crate::gguf::quantized::OwnedQKVWeights::Fused(ref w) => {
                        ops::rms_norm_into(
                            &hidden,
                            &layer.attn_norm_weight,
                            self.config.eps,
                            &mut o_proj_buffer[..hidden_dim],
                        );
                        qkv_actual_dim = w.out_dim;
                        self.fused_matmul_into(
                            &o_proj_buffer[..hidden_dim],
                            w,
                            &mut qkv_buffer[..w.out_dim],
                        )?;
                        &mut qkv_buffer[..w.out_dim]
                    }
                    _ => {
                        qkv_actual_dim = 0;
                        let v = self.fused_rmsnorm_qkv_matmul(
                            &hidden,
                            &layer.attn_norm_weight,
                            self.config.eps,
                            &layer.qkv_weight,
                        )?;
                        qkv_buffer[..v.len()].copy_from_slice(&v);
                        &mut qkv_buffer[..v.len()]
                    }
                }
            } else {
                let normed = ops::layer_norm(
                    &hidden,
                    &layer.attn_norm_weight,
                    layer.attn_norm_bias.as_deref(),
                    self.config.eps,
                );
                let v = self.qkv_matmul(&normed, &layer.qkv_weight)?;
                qkv_actual_dim = 0;
                qkv_buffer[..v.len()].copy_from_slice(&v);
                &mut qkv_buffer[..v.len()]
            };

            if let Some(ref bias) = layer.qkv_bias {
                ops::add_bias(&mut qkv, bias);
            }
            profiler.stop("QkvProjection");

            // 2c. Extract Q, K, V with GQA-aware sizes and apply RoPE
            let num_kv_heads = self.config.num_kv_heads;
            let head_dim = self.config.head_dim();
            let q_dim = self.config.q_dim();
            let kv_dim = self.config.kv_dim();

            // GH-479: Per-head QK RMSNorm (Qwen3) — after bias, before RoPE
            if let Some(ref q_norm) = layer.attn_q_norm_weight {
                ops::apply_per_head_rms_norm(
                    &mut qkv[0..q_dim],
                    q_norm,
                    self.config.num_heads,
                    self.config.eps,
                );
            }
            if let Some(ref k_norm) = layer.attn_k_norm_weight {
                ops::apply_per_head_rms_norm(
                    &mut qkv[q_dim..q_dim + kv_dim],
                    k_norm,
                    num_kv_heads,
                    self.config.eps,
                );
            }

            // GH-278: Skip RoPE for models with learned position embeddings (GPT-2)
            profiler.start("RopeEmbedding");
            if self.config.constraints.uses_rope() {
                self.apply_rope(&mut qkv[0..q_dim], position, self.config.num_heads);
                self.apply_rope(
                    &mut qkv[q_dim..q_dim + kv_dim],
                    position,
                    num_kv_heads,
                );
            }
            profiler.stop("RopeEmbedding");

            // Use slices to avoid copies
            let q = &qkv[0..q_dim];
            let k = &qkv[q_dim..q_dim + kv_dim];
            let v = &qkv[q_dim + kv_dim..q_dim + 2 * kv_dim];

            // 2d. Get cached K/V and compute attention with GQA support
            profiler.start("AttentionScore");
            let k_cache = cache.get_k(layer_idx);
            let v_cache = cache.get_v(layer_idx);

            if k_cache.is_empty() {
                let q_per_kv = self.config.num_heads / num_kv_heads;
                for q_head in 0..self.config.num_heads {
                    let kv_head = q_head / q_per_kv;
                    let v_start = kv_head * head_dim;
                    let out_start = q_head * head_dim;
                    attn_out_buffer[out_start..out_start + head_dim]
                        .copy_from_slice(&v[v_start..v_start + head_dim]);
                }
            } else {
                self.attention_with_cache_gqa_into(
                    q, k_cache, v_cache, k, v, &mut attn_out_buffer,
                );
            }
            profiler.stop("AttentionScore");

            // 2e. Store K and V in cache for future tokens
            cache.append(layer_idx, k, v);

            // 2f. Attention output projection
            profiler.start("OutputProjection");
            self.fused_matmul_into(
                &attn_out_buffer,
                &layer.attn_output_weight,
                &mut o_proj_buffer,
            )?;
            if let Some(ref bias) = layer.attn_output_bias {
                ops::add_bias(&mut o_proj_buffer, bias);
            }
            profiler.stop("OutputProjection");

            // 2g. Residual connection
            for i in 0..hidden_dim {
                hidden[i] += o_proj_buffer[i];
            }

            // 2h+2i. FFN with optional layer norm and SwiGLU/GELU activation
            profiler.start("FFN");
            let ffn_activated =
                self.single_cache_ffn_block(&hidden, layer_idx, use_rmsnorm)?;
            profiler.stop("FFN");

            // 2j. FFN down projection
            profiler.start("DownProjection");
            self.fused_matmul_into(
                &ffn_activated,
                &layer.ffn_down_weight,
                &mut ffn_down_buffer,
            )?;
            if let Some(ref bias) = layer.ffn_down_bias {
                ops::add_bias(&mut ffn_down_buffer, bias);
            }
            profiler.stop("DownProjection");

            // Residual
            for i in 0..hidden_dim {
                hidden[i] += ffn_down_buffer[i];
            }
        }

        // Advance cache position after processing all layers
        cache.advance();

        // 3+4. Final layer norm + LM head
        profiler.start("LmHead");
        let logits =
            self.single_cache_final_output(&hidden, position, use_rmsnorm)?;
        profiler.stop("LmHead");

        Ok(logits)
    }
}
