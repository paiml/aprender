
#[cfg(feature = "gpu")]
impl OwnedQuantizedModelCachedSync {

    /// Batched forward pass with GPU FFN optimization (PARITY-021)
    ///
    /// Processes multiple tokens in parallel with GPU-accelerated FFN.
    /// Attention is still per-token with CPU KV cache, but FFN uses GPU GEMM.
    ///
    /// # Arguments
    /// * `token_ids` - Token IDs for each prompt [batch_size]
    /// * `caches` - Per-prompt KV caches
    /// * `positions` - Position for each prompt [batch_size]
    ///
    /// # Returns
    /// Logits for each prompt [batch_size][vocab_size]
    ///
    /// # GPU Dispatch
    /// - batch_size >= 32: GPU GEMM for FFN (10x speedup)
    /// - batch_size < 32: CPU fallback
    pub fn forward_batch_with_gpu_ffn(
        &self,
        token_ids: &[u32],
        caches: &mut [OwnedQuantizedKVCache],
        positions: &[usize],
    ) -> Result<Vec<Vec<f32>>> {
        let batch_size = token_ids.len();
        if batch_size == 0 {
            return Ok(Vec::new());
        }
        if batch_size != caches.len() || batch_size != positions.len() {
            return Err(RealizarError::InvalidShape {
                reason: format!(
                    "Batch size mismatch: tokens={}, caches={}, positions={}",
                    batch_size,
                    caches.len(),
                    positions.len()
                ),
            });
        }

        let hidden_dim = self.model.config.hidden_dim;
        let num_layers = self.model.layers.len();

        // Threshold for GPU dispatch (based on IMP-600 analysis)
        const GPU_BATCH_THRESHOLD: usize = 32;
        let use_gpu = batch_size >= GPU_BATCH_THRESHOLD && self.is_gpu_cache_warm();

        // PARITY-098: Parallel embedding using rayon
        use rayon::prelude::*;
        let mut hidden_states: Vec<Vec<f32>> = token_ids
            .par_iter()
            .map(|&tid| self.model.embed(&[tid]))
            .collect();

        // 2. Process through transformer layers
        for layer_idx in 0..num_layers {
            let layer = &self.model.layers[layer_idx];

            // PARITY-024: GPU batch attention path vs CPU sequential path
            if use_gpu {
                // GPU path: batch QKV projection, per-prompt attention, batch output projection

                // 2a. PARITY-098: Parallel batch layer norm
                let normed_batch: Vec<Vec<f32>> = hidden_states
                    .par_iter()
                    .map(|hidden| {
                        self.model.layer_norm(
                            hidden,
                            &layer.attn_norm_weight,
                            layer.attn_norm_bias.as_deref(),
                            self.model.config.eps,
                        )
                    })
                    .collect();

                // 2b. Batch QKV projection using GPU GEMM (PARITY-024)
                let batch_normed: Vec<f32> = normed_batch.iter().flatten().copied().collect();
                let batch_qkv = self.batch_qkv_projection_gpu(&batch_normed, layer_idx)?;

                // 2c-2e. PARITY-099: Parallel attention computation per prompt
                // Each prompt has its own KV cache, so we can parallelize
                let qkv_dim = 3 * hidden_dim;

                let attention_outputs: Vec<Vec<f32>> = caches
                    .par_iter_mut()
                    .enumerate()
                    .map(|(prompt_idx, cache)| {
                        let qkv_start = prompt_idx * qkv_dim;
                        let qkv = &batch_qkv[qkv_start..qkv_start + qkv_dim];

                        // Extract Q, K, V
                        let mut q = qkv[0..hidden_dim].to_vec();
                        let mut k = qkv[hidden_dim..2 * hidden_dim].to_vec();
                        let v = qkv[2 * hidden_dim..3 * hidden_dim].to_vec();

                        // Apply RoPE (position-dependent, must be per-prompt)
                        // Note: Uses num_heads for both (non-GQA code path)
                        self.model.apply_rope(
                            &mut q,
                            positions[prompt_idx],
                            self.model.config.num_heads,
                        );
                        self.model.apply_rope(
                            &mut k,
                            positions[prompt_idx],
                            self.model.config.num_heads,
                        );

                        // Attention with KV cache (must be per-prompt, different caches)
                        // PARITY-027: Use FlashAttention for long sequences (O(N) memory)
                        let k_cache = cache.get_k(layer_idx);
                        let v_cache = cache.get_v(layer_idx);

                        // FlashAttention threshold: use for sequences >= 512 tokens
                        const FLASH_ATTENTION_THRESHOLD: usize = 512;
                        let cache_len = k_cache.len() / hidden_dim;
                        let use_flash_attention = cache_len >= FLASH_ATTENTION_THRESHOLD;

                        let attn_out = if k_cache.is_empty() {
                            v.clone()
                        } else if use_flash_attention {
                            // FlashAttention: O(N) memory, tiled computation
                            const FLASH_BLOCK_SIZE: usize = 64;
                            self.model.flash_attention_tiled(
                                &q,
                                k_cache,
                                v_cache,
                                &k,
                                &v,
                                FLASH_BLOCK_SIZE,
                            )
                        } else {
                            // Standard attention: O(N²) memory but faster for short sequences
                            self.model
                                .attention_with_cache(&q, k_cache, v_cache, &k, &v)
                        };

                        // Store K and V in cache
                        cache.append(layer_idx, &k, &v);
                        attn_out
                    })
                    .collect();

                // 2f. Batch attention output projection using GPU GEMM (PARITY-024)
                let batch_attn: Vec<f32> = attention_outputs.iter().flatten().copied().collect();
                let batch_output = self.batch_attention_output_gpu(&batch_attn, layer_idx)?;

                // 2g. PARITY-100: Parallel residual connection
                hidden_states
                    .par_iter_mut()
                    .enumerate()
                    .for_each(|(prompt_idx, hidden)| {
                        let start = prompt_idx * hidden_dim;
                        for i in 0..hidden_dim {
                            hidden[i] += batch_output[start + i];
                        }
                    });
            } else {
                // CPU sequential path (original implementation)
                for (prompt_idx, hidden) in hidden_states.iter_mut().enumerate() {
                    // Attention layer norm
                    let normed = self.model.layer_norm(
                        hidden,
                        &layer.attn_norm_weight,
                        layer.attn_norm_bias.as_deref(),
                        self.model.config.eps,
                    );

                    // QKV projection
                    let mut qkv = self.model.qkv_matmul(&normed, &layer.qkv_weight)?;
                    if let Some(ref bias) = layer.qkv_bias {
                        self.model.add_bias(&mut qkv, bias);
                    }

                    // Extract Q, K, V and apply RoPE
                    // Note: Uses num_heads for both (non-GQA code path)
                    let mut q = qkv[0..hidden_dim].to_vec();
                    let mut k = qkv[hidden_dim..2 * hidden_dim].to_vec();
                    let v = qkv[2 * hidden_dim..3 * hidden_dim].to_vec();

                    self.model.apply_rope(
                        &mut q,
                        positions[prompt_idx],
                        self.model.config.num_heads,
                    );
                    self.model.apply_rope(
                        &mut k,
                        positions[prompt_idx],
                        self.model.config.num_heads,
                    );

                    // Get cached K/V and compute attention
                    let k_cache = caches[prompt_idx].get_k(layer_idx);
                    let v_cache = caches[prompt_idx].get_v(layer_idx);

                    let attn_out = if k_cache.is_empty() {
                        v.clone()
                    } else {
                        self.model
                            .attention_with_cache(&q, k_cache, v_cache, &k, &v)
                    };

                    // Store K and V in cache
                    caches[prompt_idx].append(layer_idx, &k, &v);

                    // Attention output projection
                    let mut attn_output = self
                        .model
                        .fused_matmul(&attn_out, &layer.attn_output_weight)?;
                    if let Some(ref bias) = layer.attn_output_bias {
                        self.model.add_bias(&mut attn_output, bias);
                    }

                    // Residual connection
                    for i in 0..hidden_dim {
                        hidden[i] += attn_output[i];
                    }
                }
            }

            // 2h. FFN - GPU batch or CPU sequential
            if use_gpu {
                // GPU batch FFN: collect hidden states, process together, scatter back
                let batch_hidden: Vec<f32> = hidden_states.iter().flatten().copied().collect();
                let ffn_output = self.batch_ffn_gpu(&batch_hidden, layer_idx)?;

                // PARITY-100: Parallel scatter and residual
                hidden_states
                    .par_iter_mut()
                    .enumerate()
                    .for_each(|(prompt_idx, hidden)| {
                        let start = prompt_idx * hidden_dim;
                        for i in 0..hidden_dim {
                            hidden[i] += ffn_output[start + i];
                        }
                    });
            } else {
                // CPU sequential FFN
                for hidden in &mut hidden_states {
                    let mut ffn_hidden = self.model.fused_matmul(hidden, &layer.ffn_up_weight)?;
                    if let Some(ref bias) = layer.ffn_up_bias {
                        self.model.add_bias(&mut ffn_hidden, bias);
                    }
                    self.model.gelu(&mut ffn_hidden);

                    let mut ffn_output = self
                        .model
                        .fused_matmul(&ffn_hidden, &layer.ffn_down_weight)?;
                    if let Some(ref bias) = layer.ffn_down_bias {
                        self.model.add_bias(&mut ffn_output, bias);
                    }

                    // Residual
                    for i in 0..hidden_dim {
                        hidden[i] += ffn_output[i];
                    }
                }
            }
        }

        // PARITY-100: Parallel cache advance
        caches.par_iter_mut().for_each(|cache| {
            cache.advance();
        });

        // 3. Final layer norm and LM head for each prompt
        // PARITY-025: Use GPU batch LM head when batch >= threshold
        let vocab_size = self.model.config.vocab_size;

        let all_logits: Vec<Vec<f32>> = if use_gpu {
            // GPU path: batch layer norm and LM head projection

            // 3a. PARITY-098: Parallel final layer norm
            let normed_batch: Vec<Vec<f32>> = hidden_states
                .par_iter()
                .map(|hidden| {
                    self.model.layer_norm(
                        hidden,
                        &self.model.output_norm_weight,
                        self.model.output_norm_bias.as_deref(),
                        self.model.config.eps,
                    )
                })
                .collect();

            // 3b. Batch LM head projection using GPU GEMM (PARITY-025)
            let batch_normed: Vec<f32> = normed_batch.iter().flatten().copied().collect();
            let batch_logits = self.batch_lm_head_gpu(&batch_normed)?;

            // 3c. PARITY-098: Parallel scatter logits back to per-prompt vectors
            (0..batch_size)
                .into_par_iter()
                .map(|i| {
                    let start = i * vocab_size;
                    batch_logits[start..start + vocab_size].to_vec()
                })
                .collect()
        } else {
            // CPU path: sequential per-prompt processing
            let mut result = Vec::with_capacity(batch_size);
            for hidden in &hidden_states {
                let normed = self.model.layer_norm(
                    hidden,
                    &self.model.output_norm_weight,
                    self.model.output_norm_bias.as_deref(),
                    self.model.config.eps,
                );

                let mut logits = self
                    .model
                    .fused_matmul(&normed, &self.model.lm_head_weight)?;
                if let Some(ref bias) = self.model.lm_head_bias {
                    self.model.add_bias(&mut logits, bias);
                }
                result.push(logits);
            }
            result
        };

        Ok(all_logits)
    }

    /// Get batch generation statistics
    ///
    /// Returns information about the batch processing capabilities.
    pub fn batch_stats(&self) -> BatchGenerationStats {
        let is_cached = self.is_gpu_cache_warm();
        let memory_gb = self.gpu_cache_memory() as f64 / 1_000_000_000.0;
        let num_layers = self.model.layers.len();
        let hidden_dim = self.model.config.hidden_dim;
        let intermediate_dim = self.model.config.intermediate_dim;

        BatchGenerationStats {
            gpu_cache_ready: is_cached,
            cache_memory_gb: memory_gb,
            num_layers,
            hidden_dim,
            intermediate_dim,
            recommended_batch_size: 32, // GPU GEMM threshold
            max_batch_size: 64,         // Memory-limited
        }
    }
}

include!("thread-safe.rs");
include!("hidden_dim.rs");
include!("sync_owned_quantized_02.rs");
