impl AprV2ModelCuda {

    /// GPU-accelerated forward pass.
    ///
    /// Computes logits for the given token sequence using GPU acceleration
    /// for matrix multiplications. Achieves 2x+ Ollama performance by using
    /// GPU GEMM for QKV, attention output, and FFN projections.
    ///
    /// # Arguments
    ///
    /// * `token_ids` - Input token IDs
    ///
    /// # Returns
    ///
    /// Logits vector of size `vocab_size` for next token prediction.
    pub fn forward_cuda(&mut self, token_ids: &[u32]) -> Result<Vec<f32>> {
        // GH-282: Ensure CUDA context is current for this thread.
        // Without this, tokio worker threads in `apr serve` get error 201
        // (CUDA_ERROR_INVALID_CONTEXT) when compiling PTX modules.
        self.executor
            .make_current()
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "cuda_make_current".to_string(),
                reason: format!("Failed to set CUDA context current: {e}"),
            })?;

        if token_ids.is_empty() {
            return Err(RealizarError::InvalidShape {
                reason: "Token sequence cannot be empty".to_string(),
            });
        }

        if !self.model.metadata.is_transformer() {
            return Err(RealizarError::FormatError {
                reason: "Model is not a transformer (missing config)".to_string(),
            });
        }

        let hidden_dim = self.model.metadata.hidden_size.unwrap_or(0);
        let num_layers = self.model.metadata.num_layers.unwrap_or(0);
        let num_heads = self.model.metadata.num_heads.unwrap_or(1);
        let num_kv_heads = self.model.metadata.num_kv_heads.unwrap_or(num_heads);
        let vocab_size = self.model.metadata.vocab_size.unwrap_or(0);
        let intermediate_dim = self
            .model
            .metadata
            .intermediate_size
            .unwrap_or(hidden_dim * 4);
        let eps = self.model.metadata.rms_norm_eps.unwrap_or(1e-6);
        let seq_len = token_ids.len();
        let head_dim = hidden_dim / num_heads;
        let kv_dim = num_kv_heads * head_dim;

        // FAST PATH: Use indexed Q4K GEMV kernels with CUDA graph capture
        // Phase 45: Skip fast path when test_executor is present
        // PMAT-110: Skip fast path if KV cache was populated via fallback path
        // GH-201: Skip fast path in streaming mode (layer weights not pre-cached)
        if self.test_executor.is_none()
            && self.executor.has_indexed_weights()
            && seq_len == 1
            && !self.fallback_kv_used
            && !self.streaming_mode
        {
            return self.forward_cuda_indexed_decode(
                token_ids[0],
                vocab_size,
                num_layers,
                hidden_dim as u32,
                intermediate_dim as u32,
                eps,
            );
        }

        // FALLBACK PATH: Original F32 GEMM path (for prefill or non-indexed models)
        let profiling = self.executor.is_profiling_enabled();
        let trace_layers = std::env::var("APR_TRACE_LAYERS").is_ok();

        // 1. Token embedding lookup
        let mut hidden = self.forward_cuda_embed(token_ids, hidden_dim, profiling, trace_layers)?;

        // 2. Process through transformer layers
        for layer_idx in 0..num_layers {
            let attn_norm_name = self.model.find_tensor_name(&[
                &format!("model.layers.{layer_idx}.input_layernorm.weight"),
                &format!("layers.{layer_idx}.input_layernorm.weight"),
                &format!("transformer.h.{layer_idx}.ln_1.weight"),
                &format!("layers.{layer_idx}.attention_norm.weight"),
                &format!("blk.{layer_idx}.attn_norm.weight"),
            ])?;

            let o_name = self.model.find_tensor_name(&[
                &format!("model.layers.{layer_idx}.self_attn.o_proj.weight"),
                &format!("layers.{layer_idx}.self_attn.o_proj.weight"),
                &format!("transformer.h.{layer_idx}.attn.out_proj.weight"),
                &format!("layers.{layer_idx}.attention.wo.weight"),
                &format!("blk.{layer_idx}.attn_output.weight"),
            ])?;
            let o_cache_name = format!("blk.{}.attn_output.weight", layer_idx);

            // RMSNorm
            let norm_weight = self.model.get_tensor_f32(&attn_norm_name)?;
            let normed = rms_norm(&hidden, &norm_weight, eps);

            // QKV projections (delegated to helper for cached/fused/separate paths)
            let (mut q, mut k, mut v) =
                self.forward_cuda_qkv_projection(layer_idx, &normed, seq_len, hidden_dim, kv_dim)?;

            self.apply_qkv_bias_for_layer(
                layer_idx, &mut q, &mut k, &mut v,
                hidden_dim, kv_dim, seq_len, trace_layers,
            )?;

            // Attention + output projection + residual
            self.forward_cuda_attention_layer(
                layer_idx, &q, &k, &v, &mut hidden,
                seq_len, hidden_dim, kv_dim,
                num_heads, num_kv_heads, head_dim,
                &o_name, &o_cache_name, profiling,
            )?;

            // FFN + residual
            self.forward_cuda_ffn_layer(
                layer_idx, &mut hidden,
                seq_len, hidden_dim, intermediate_dim,
                eps, num_layers, profiling, trace_layers,
            )?;
        }

        // Update KV cache position
        self.kv_position += seq_len as u32;
        self.fallback_kv_used = true;

        // 3. Final layer norm
        let final_norm_name = self.model.find_tensor_name(&[
            "model.norm.weight",
            "norm.weight",
            "transformer.ln_f.weight",
            "output_norm.weight",
        ])?;
        let final_norm = self.model.get_tensor_f32(&final_norm_name)?;
        let hidden = rms_norm(&hidden, &final_norm, eps);

        // 4. LM head projection
        let last_hidden = &hidden[hidden.len() - hidden_dim..];
        self.forward_cuda_lm_head(last_hidden, hidden_dim, vocab_size)
    }

    /// Embedding lookup for token IDs.
    fn forward_cuda_embed(
        &mut self,
        token_ids: &[u32],
        hidden_dim: usize,
        profiling: bool,
        trace_layers: bool,
    ) -> Result<Vec<f32>> {
        let timer_embed = if profiling {
            let _ = self.executor.synchronize();
            Some(self.executor.profiler_mut().start("apr.Embed"))
        } else {
            None
        };

        let embed_name = self.model.find_tensor_name(&[
            "model.embed_tokens.weight",
            "embed_tokens.weight",
            "transformer.wte.weight",
            "embeddings.word_embeddings.weight",
            "tok_embeddings.weight",
            "token_embd.weight",
        ])?;
        let embeddings = self.model.get_tensor_f32(&embed_name)?;
        let seq_len = token_ids.len();

        let mut hidden = Vec::with_capacity(seq_len * hidden_dim);
        for &token_id in token_ids {
            let offset = (token_id as usize) * hidden_dim;
            if offset + hidden_dim <= embeddings.len() {
                hidden.extend_from_slice(&embeddings[offset..offset + hidden_dim]);
            } else {
                // N-09: OOB token → zeros. Contract: embedding-lookup-v1.yaml
                eprintln!(
                    "Warning: forward_cuda_embed token_id {} OOB (offset={offset}, embed_len={}). N-09 escape.",
                    token_id, embeddings.len()
                );
                hidden.extend(std::iter::repeat_n(0.0, hidden_dim));
            }
        }

        if let Some(t) = timer_embed {
            self.executor.profiler_mut().stop(t, seq_len as u64);
        }

        if trace_layers {
            let last_hidden = &hidden[hidden.len() - hidden_dim..];
            let sum: f32 = last_hidden.iter().sum();
            let mean = sum / hidden_dim as f32;
            let min = last_hidden.iter().cloned().fold(f32::INFINITY, f32::min);
            let max = last_hidden
                .iter()
                .cloned()
                .fold(f32::NEG_INFINITY, f32::max);
            eprintln!(
                "[PMAT-114] After embed: mean={:.6}, min={:.6}, max={:.6}, first5={:?}",
                mean, min, max,
                &last_hidden[..5.min(hidden_dim)]
            );
        }

        Ok(hidden)
    }
}

include!("forward_from_cuda_helpers.rs");
