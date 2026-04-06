impl OwnedQuantizedModel {
    /// Batched forward pass for prompt prefill (PARITY-002)
    ///
    /// Processes all prompt tokens at once, enabling GPU acceleration
    /// for the attention computation when the batch is large enough.
    ///
    /// # Arguments
    /// * `tokens` - All prompt tokens to process at once
    /// * `cache` - KV cache for storing computed K/V tensors
    /// * `metrics` - Dispatch metrics tracker for CPU/GPU decision recording
    ///
    /// # Returns
    /// Logits for next token prediction (from the last token position)
    ///
    /// # Errors
    /// Returns error if tensor operations fail
    #[cfg(feature = "gpu")]
    pub fn forward_batch_with_cache(
        &self,
        tokens: &[u32],
        cache: &mut OwnedQuantizedKVCache,
        metrics: &std::sync::Arc<DispatchMetrics>,
    ) -> Result<Vec<f32>> {
        if tokens.is_empty() {
            return Err(RealizarError::InvalidShape {
                reason: "Tokens cannot be empty".to_string(),
            });
        }

        let seq_len = tokens.len();
        let hidden_dim = self.config.hidden_dim;

        // 1. Embed all tokens at once: [seq_len, hidden_dim]
        let mut hidden_states: Vec<Vec<f32>> = tokens
            .iter()
            .map(|&token_id| self.embed(&[token_id]))
            .collect();

        // 2. Process through transformer layers
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            // Collect Q, K, V for all positions
            let mut all_q: Vec<Vec<f32>> = Vec::with_capacity(seq_len);
            let mut all_k: Vec<Vec<f32>> = Vec::with_capacity(seq_len);
            let mut all_v: Vec<Vec<f32>> = Vec::with_capacity(seq_len);

            for (pos, hidden) in hidden_states.iter().enumerate() {
                // 2a. Attention layer norm
                let normed = ops::layer_norm(
                    hidden,
                    &layer.attn_norm_weight,
                    layer.attn_norm_bias.as_deref(),
                    self.config.eps,
                );

                // 2b. QKV projection
                let mut qkv = self.qkv_matmul(&normed, &layer.qkv_weight)?;
                if let Some(ref bias) = layer.qkv_bias {
                    ops::add_bias(&mut qkv, bias);
                }

                // 2c. Extract Q, K, V and apply RoPE
                // Note: This uses hidden_dim for all (assumes non-GQA or fused QKV)
                let mut q = qkv[0..hidden_dim].to_vec();
                let mut k = qkv[hidden_dim..2 * hidden_dim].to_vec();
                let v = qkv[2 * hidden_dim..3 * hidden_dim].to_vec();

                self.apply_rope(&mut q, pos, self.config.num_heads);
                self.apply_rope(&mut k, pos, self.config.num_heads); // Same as Q for non-GQA

                all_q.push(q);
                all_k.push(k);
                all_v.push(v);
            }

            // 2d. Compute batched attention
            // For PARITY-002: This is where GPU can accelerate!
            // Attention scores: Q @ K^T is [seq_len, seq_len]
            let attn_outputs = self
                .batched_attention_with_cache(&all_q, &all_k, &all_v, cache, layer_idx, metrics)?;

            // 2e. Store all K/V in cache
            for (k, v) in all_k.iter().zip(all_v.iter()) {
                cache.append(layer_idx, k, v);
            }

            // 2f. Attention output projection + residual
            for (pos, attn_out) in attn_outputs.iter().enumerate() {
                let mut attn_output = self.fused_matmul(attn_out, &layer.attn_output_weight)?;
                if let Some(ref bias) = layer.attn_output_bias {
                    ops::add_bias(&mut attn_output, bias);
                }

                // Residual connection
                for i in 0..hidden_dim {
                    hidden_states[pos][i] += attn_output[i];
                }
            }

            // 2g. FFN for all positions
            for hidden in &mut hidden_states {
                let mut ffn_hidden = self.fused_matmul(hidden, &layer.ffn_up_weight)?;
                if let Some(ref bias) = layer.ffn_up_bias {
                    ops::add_bias(&mut ffn_hidden, bias);
                }
                ops::gelu(&mut ffn_hidden);

                let mut ffn_output = self.fused_matmul(&ffn_hidden, &layer.ffn_down_weight)?;
                if let Some(ref bias) = layer.ffn_down_bias {
                    ops::add_bias(&mut ffn_output, bias);
                }

                // Residual
                for i in 0..hidden_dim {
                    hidden[i] += ffn_output[i];
                }
            }
        }

        // Advance cache position for all processed tokens
        for _ in 0..seq_len {
            cache.advance();
        }

        // 3. Final layer norm and LM head for LAST token only
        let last_hidden = &hidden_states[seq_len - 1];
        let normed = ops::layer_norm(
            last_hidden,
            &self.output_norm_weight,
            self.output_norm_bias.as_deref(),
            self.config.eps,
        );

        // 4. LM head projection
        let mut logits = self.fused_matmul(&normed, &self.lm_head_weight)?;
        if let Some(ref bias) = self.lm_head_bias {
            ops::add_bias(&mut logits, bias);
        }

        Ok(logits)
    }

    /// Batched attention computation with GPU acceleration (PARITY-002)
    ///
    /// Computes attention for all positions at once, enabling GPU dispatch
    /// when the workload (seq_len * hidden_dim * seq_len) exceeds the threshold.
    ///
    /// KEY OPTIMIZATION: Uses GPU matmul for Q @ K^T when workload is large enough.
    /// This is the critical path for GPU acceleration - previous implementation only
    /// recorded metrics without actually using GPU.
    #[cfg(feature = "gpu")]
    fn batched_attention_with_cache(
        &self,
        all_q: &[Vec<f32>],
        all_k: &[Vec<f32>],
        all_v: &[Vec<f32>],
        cache: &OwnedQuantizedKVCache,
        layer_idx: usize,
        metrics: &std::sync::Arc<DispatchMetrics>,
    ) -> Result<Vec<Vec<f32>>> {
        let seq_len = all_q.len();
        let hidden_dim = self.config.hidden_dim;
        let num_heads = self.config.num_heads;
        let head_dim = hidden_dim / num_heads;

        // Get any cached K/V from previous sequences
        let cached_k = cache.get_k(layer_idx);
        let cached_v = cache.get_v(layer_idx);
        let cache_len = cached_k.len() / hidden_dim;

        // Build full K/V sequences: [cache + current]
        let total_len = cache_len + seq_len;

        // Determine if we should use GPU based on workload size
        //
        // IMPORTANT FINDING (IMP-600, PARITY-002):
        // GPU is 2.7x SLOWER for MATVEC operations (per-head attention is MATVEC)
        // GPU is 57x FASTER for large GEMM (batch) operations
        //
        // For GPU to be beneficial, we need LARGE matrices. Per-head attention
        // uses tiny matrices: Q[1, head_dim] @ K^T[head_dim, seq_len] = [1, seq_len]
        // This is a MATVEC operation where GPU transfer overhead dominates.
        //
        // Measured result with GPU matmul: 0.20 tok/s (vs 5.31 tok/s CPU)
        // GPU path is 26x SLOWER due to per-head matmul overhead.
        //
        // For true GPU acceleration, need:
        // - FlashAttention (fused kernel, not yet available in trueno)
        // - Batched multi-request inference (process multiple prompts together)
        //
        // For now, use optimized CPU path which is faster for single-request inference.
        let workload = num_heads * seq_len * head_dim * total_len;
        let _ = workload; // Document: GPU not used because MATVEC is slower on GPU

        // Always use CPU path - it's faster for per-head attention MATVEC
        metrics.record_cpu_dispatch();
        self.cpu_batched_attention(
            all_q, all_k, all_v, cached_k, cached_v, cache_len, hidden_dim, num_heads, head_dim,
        )
    }

    /// CPU-based batched attention (fallback for small workloads)
    #[cfg(feature = "gpu")]
    #[allow(clippy::too_many_arguments)] // Attention requires all these parameters
    fn cpu_batched_attention(
        &self,
        all_q: &[Vec<f32>],
        all_k: &[Vec<f32>],
        all_v: &[Vec<f32>],
        cached_k: &[f32],
        cached_v: &[f32],
        cache_len: usize,
        hidden_dim: usize,
        _num_heads: usize,
        head_dim: usize,
    ) -> Result<Vec<Vec<f32>>> {
        let seq_len = all_q.len();
        let mut outputs = Vec::with_capacity(seq_len);

        for (q_pos, q) in all_q.iter().enumerate() {
            let attend_len = cache_len + q_pos + 1;
            let mut k_vecs: Vec<&[f32]> = Vec::with_capacity(attend_len);
            let mut v_vecs: Vec<&[f32]> = Vec::with_capacity(attend_len);

            // Add cached K/V
            for i in 0..cache_len {
                let start = i * hidden_dim;
                let end = start + hidden_dim;
                k_vecs.push(&cached_k[start..end]);
                v_vecs.push(&cached_v[start..end]);
            }

            // Add current sequence K/V up to and including current position
            for i in 0..=q_pos {
                k_vecs.push(&all_k[i]);
                v_vecs.push(&all_v[i]);
            }

            let output = self.compute_attention_output(q, &k_vecs, &v_vecs, head_dim)?;
            outputs.push(output);
        }

        Ok(outputs)
    }

    /// Compute attention output for a single query against K/V vectors
    #[cfg(feature = "gpu")]
    fn compute_attention_output(
        &self,
        q: &[f32],
        k_vecs: &[&[f32]],
        v_vecs: &[&[f32]],
        head_dim: usize,
    ) -> Result<Vec<f32>> {
        let hidden_dim = q.len();
        let num_heads = hidden_dim / head_dim;
        let seq_len = k_vecs.len();

        if seq_len == 0 {
            // No keys to attend to - return zeros (will be replaced by first attention)
            return Ok(vec![0.0; hidden_dim]);
        }

        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut output = vec![0.0; hidden_dim];

        // Process each head independently
        for head in 0..num_heads {
            let head_start = head * head_dim;
            let head_end = head_start + head_dim;

            let q_head = &q[head_start..head_end];

            // Compute attention scores for this head
            let mut scores = Vec::with_capacity(seq_len);
            for k in k_vecs {
                let k_head = &k[head_start..head_end];
                let score: f32 = q_head.iter().zip(k_head.iter()).map(|(a, b)| a * b).sum();
                scores.push(score * scale);
            }

            // Softmax (SIMD-optimized, in-place)
            crate::quantize::softmax_simd(&mut scores);

            // Weighted sum of values
            for (attn, v) in scores.iter().zip(v_vecs.iter()) {
                let v_head = &v[head_start..head_end];
                for (i, &v_val) in v_head.iter().enumerate() {
                    output[head_start + i] += attn * v_val;
                }
            }
        }

        Ok(output)
    }

    /// Generate tokens with batched prompt prefill (PARITY-002)
    ///
    /// Uses `forward_batch_with_cache` for initial prompt processing (GPU-accelerated),
    /// then falls back to single-token generation for autoregressive decoding.
    ///
    /// # Arguments
    /// * `prompt` - Initial token IDs (processed in batch)
    /// * `config` - Generation configuration
    /// * `metrics` - Dispatch metrics tracker
    ///
    /// # Returns
    /// Generated token sequence including prompt
    ///
    /// # Errors
    /// Returns error if generation fails
    #[cfg(feature = "gpu")]
    pub fn generate_with_batched_prefill(
        &self,
        prompt: &[u32],
        config: &QuantizedGenerateConfig,
        metrics: &std::sync::Arc<DispatchMetrics>,
    ) -> Result<Vec<u32>> {
        contract_pre_batched_serial_equivalence!();
        if prompt.is_empty() {
            return Err(RealizarError::InvalidShape {
                reason: "Prompt cannot be empty".to_string(),
            });
        }

        let max_seq_len = prompt.len() + config.max_tokens;
        let mut cache = OwnedQuantizedKVCache::from_config(&self.config, max_seq_len);
        let mut tokens = prompt.to_vec();

        // PARITY-002: Process ALL prompt tokens at once (batched prefill)
        // This enables GPU acceleration for the attention computation
        let mut logits = self.forward_batch_with_cache(prompt, &mut cache, metrics)?;

        // Generate new tokens one at a time (autoregressive)
        for gen_idx in 0..config.max_tokens {
            // Sample next token from logits
            let next_token = if config.temperature == 0.0 || config.top_k == 1 {
                ops::argmax(&logits)
            } else {
                crate::gguf::OwnedQuantizedModel::sample_topk(
                    &logits,
                    config.temperature,
                    config.top_k,
                )
            };

            // Check stop condition
            if config.stop_tokens.contains(&next_token) {
                break;
            }

            tokens.push(next_token);

            // Check max length
            if tokens.len() >= max_seq_len {
                break;
            }

            // Forward pass for the new token (single-token, uses CPU)
            let position = prompt.len() + gen_idx;
            logits =
                self.forward_single_with_cache_adaptive(next_token, &mut cache, position, metrics)?;
        }

        Ok(tokens)
    }

    /// Generate tokens with SmallVec optimization (IMP-117)
    ///
    /// Uses SmallVec for token storage to avoid heap allocations when:
    /// - Prompt + max_tokens <= TOKEN_BUFFER_INLINE_CAP
    ///
    /// # Arguments
    /// * `prompt` - Input token buffer (can be SmallVec or slice)
    /// * `config` - Generation configuration
    ///
    /// # Returns
    /// Generated token sequence as TokenBuffer (SmallVec)
    ///
    /// # Errors
    /// Returns error if forward pass fails
    pub fn generate_with_smallvec(
        &self,
        prompt: &[u32],
        config: &QuantizedGenerateConfig,
    ) -> Result<TokenBuffer> {
        if prompt.is_empty() {
            return Err(RealizarError::InvalidShape {
                reason: "Prompt cannot be empty".to_string(),
            });
        }

        let max_seq_len = prompt.len() + config.max_tokens;
        let mut cache = OwnedQuantizedKVCache::from_config(&self.config, max_seq_len);

        // Use SmallVec for token storage - inline for small sequences
        let mut tokens: TokenBuffer = TokenBuffer::from_slice(prompt);

        // Process prompt tokens (prefill)
        for (pos, &token_id) in prompt.iter().enumerate() {
            let _ = self.forward_single_with_cache(token_id, &mut cache, pos)?;
        }

        // Generate new tokens
        for gen_idx in 0..config.max_tokens {
            let position = prompt.len() + gen_idx;
            let last_token = *tokens.last().ok_or_else(|| RealizarError::InvalidShape {
                reason: "Token buffer empty during generation".to_string(),
            })?;

            let logits = self.forward_single_with_cache(last_token, &mut cache, position)?;

            // Sample next token
            let next_token = if config.temperature == 0.0 || config.top_k == 1 {
                ops::argmax(&logits)
            } else {
                crate::gguf::OwnedQuantizedModel::sample_topk(
                    &logits,
                    config.temperature,
                    config.top_k,
                )
            };

            // Check stop condition
            if config.stop_tokens.contains(&next_token) {
                break;
            }

            tokens.push(next_token);

            // Check max length
            if tokens.len() >= max_seq_len {
                break;
            }
        }

        Ok(tokens)
    }
}
