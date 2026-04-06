impl AprTransformer {

    /// Forward pass with KV cache for efficient autoregressive generation (Y4)
    ///
    /// Processes a single token using cached key-value pairs from previous positions.
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
        position: usize,
    ) -> Result<Vec<f32>> {
        let force_f32 = std::env::var("APR_FORCE_F32").is_ok();
        let trace_enabled = std::env::var("REALIZE_TRACE").is_ok();

        let hidden_dim = self.config.hidden_dim;
        let num_heads = self.config.num_heads;
        let num_kv_heads = self.config.num_kv_heads;
        let head_dim = hidden_dim / num_heads;

        // 1. Token embedding lookup
        let mut hidden = self.embed(&[token_id]);

        // 2. Process through transformer layers
        let layers_start = std::time::Instant::now();
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let q4k_layer = self.q4k_layers.as_ref().and_then(|l| l.get(layer_idx));

            // 2a. Attention layer norm
            let normed = self.layer_norm(
                &hidden,
                &layer.attn_norm_weight,
                layer.attn_norm_bias.as_deref(),
                self.config.eps,
            );

            // 2b. QKV projection with bias and RoPE
            let (q, k, v) = self.project_qkv_with_cache(
                &normed, layer, q4k_layer,
                hidden_dim, num_heads, num_kv_heads, head_dim,
                position, force_f32,
            )?;

            // 2c. Append K, V to cache (K now has RoPE applied)
            cache.append(layer_idx, &k, &v);

            // 2d. Compute attention with full cache
            let (k_cache, v_cache) = cache.get(layer_idx);
            let cache_len = cache.len();

            let attn_out = self.compute_attention_with_cache(
                &q, &k, &v, k_cache, v_cache,
                cache_len, position,
                num_heads, num_kv_heads, head_dim, hidden_dim,
            );

            // 2e. Attention output projection
            let mut attn_output = self.project_with_q4k_or_f32(
                if force_f32 { None } else { q4k_layer.and_then(|q| q.attn_output_weight.as_deref()) },
                None,
                &layer.attn_output_weight,
                &attn_out,
                hidden_dim,
                hidden_dim,
                force_f32,
            )?;
            if let Some(ref bias) = layer.attn_output_bias {
                self.add_bias(&mut attn_output, bias);
            }

            // 2f. Residual connection
            for i in 0..hidden.len() {
                hidden[i] += attn_output[i];
            }

            // 2g. Apply FFN norm if present (post_attention_layernorm)
            let ffn_input = if let Some(ref ffn_norm) = layer.ffn_norm_weight {
                self.layer_norm(
                    &hidden,
                    ffn_norm,
                    layer.ffn_norm_bias.as_deref(),
                    self.config.eps,
                )
            } else {
                hidden.clone()
            };

            // 2h. FFN projection (SwiGLU or standard GELU)
            let ffn_output = self.forward_ffn_block(
                &ffn_input, layer, q4k_layer,
                hidden_dim, self.config.intermediate_dim, force_f32,
            )?;

            // 2i. Residual connection
            for i in 0..hidden.len() {
                hidden[i] += ffn_output[i];
            }

        }
        if trace_enabled {
            eprintln!(
                "[TRACE-CACHE] pos={}: {} layers took {:?}",
                position,
                self.layers.len(),
                layers_start.elapsed()
            );
        }

        // 3. Final layer norm
        let normed = self.layer_norm(
            &hidden,
            &self.output_norm_weight,
            self.output_norm_bias.as_deref(),
            self.config.eps,
        );

        // 4. LM head projection
        self.project_lm_head(&normed, hidden_dim, force_f32)
    }
}

include!("mod_project_apr_transformer.rs");
include!("cache_attention.rs");
