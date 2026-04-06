impl OwnedQuantizedModelCuda {
    /// Forward pass using CUDA GEMM acceleration (IMP-800a)
    ///
    /// Uses CudaExecutor for matrix multiplications in the FFN layers.
    /// Attention and embedding remain on CPU for now.
    ///
    /// # Arguments
    ///
    /// * `token_ids` - Input token IDs
    ///
    /// # Returns
    ///
    /// Logits for next token prediction [vocab_size]
    ///
    /// # Errors
    ///
    /// Returns error if CUDA operations fail
    pub fn forward_cuda(&mut self, token_ids: &[u32]) -> Result<Vec<f32>> {
        let hidden_dim = self.model.config.hidden_dim;

        // 1. Token embedding lookup (CPU - fast enough)
        let mut hidden = self.model.embed(token_ids);

        // 2. Process through transformer layers
        for layer in &self.model.layers {
            // 2a. Attention layer norm (CPU)
            let normed = ops::layer_norm(
                &hidden,
                &layer.attn_norm_weight,
                layer.attn_norm_bias.as_deref(),
                self.model.config.eps,
            );

            // 2b. QKV projection (CPU - fused Q4_K for now)
            // GQA-aware dimensions: Q has num_heads, K/V have num_kv_heads
            let num_kv_heads = self.model.config.num_kv_heads;
            // GH-479: Use config methods for head_dim/q_dim/kv_dim (Qwen3 head_dim != hidden/heads)
            let head_dim = self.model.config.head_dim();
            let q_dim = self.model.config.q_dim();
            let kv_dim = self.model.config.kv_dim();
            let qkv_dim = q_dim + 2 * kv_dim; // Q + K + V with GQA
            let mut qkv = self.model.qkv_matmul(&normed, &layer.qkv_weight)?;
            if let Some(ref bias) = layer.qkv_bias {
                ops::add_bias(&mut qkv, bias);
            }

            // 2c. Attention (CPU - complex control flow)
            let seq_len = token_ids.len();
            let mut q_all = Vec::with_capacity(seq_len * q_dim);
            let mut k_all = Vec::with_capacity(seq_len * kv_dim);
            let mut v_all = Vec::with_capacity(seq_len * kv_dim);

            for s in 0..seq_len {
                let qkv_start = s * qkv_dim;
                let mut q = qkv[qkv_start..qkv_start + q_dim].to_vec();
                let mut k = qkv[qkv_start + q_dim..qkv_start + q_dim + kv_dim].to_vec();
                let v = &qkv[qkv_start + q_dim + kv_dim..qkv_start + q_dim + 2 * kv_dim];

                // GH-479: Per-head QK RMSNorm (Qwen3) — before RoPE
                if let Some(ref w) = layer.attn_q_norm_weight {
                    ops::apply_per_head_rms_norm(&mut q, w, self.model.config.num_heads, self.model.config.eps);
                }
                if let Some(ref w) = layer.attn_k_norm_weight {
                    ops::apply_per_head_rms_norm(&mut k, w, num_kv_heads, self.model.config.eps);
                }

                // GQA-aware RoPE: Q uses num_heads, K uses num_kv_heads
                self.model
                    .apply_rope(&mut q, s, self.model.config.num_heads);
                self.model.apply_rope(&mut k, s, num_kv_heads);

                q_all.extend_from_slice(&q);
                k_all.extend_from_slice(&k);
                v_all.extend_from_slice(v);
            }

            let attn_out = self.model.causal_attention(&q_all, &k_all, &v_all, seq_len);

            // 2d. Attention output projection (CPU - fused Q4_K)
            let mut attn_output = self
                .model
                .fused_matmul(&attn_out, &layer.attn_output_weight)?;
            if let Some(ref bias) = layer.attn_output_bias {
                ops::add_bias(&mut attn_output, bias);
            }

            // 2e. Residual connection
            for i in 0..hidden.len() {
                hidden[i] += attn_output[i];
            }

            // 2f. FFN up projection - try GPU GEMM if weights are dequantized
            // For now, use CPU fused ops (GPU overhead too high for m=1)
            let mut ffn_hidden = self.model.fused_matmul(&hidden, &layer.ffn_up_weight)?;
            if let Some(ref bias) = layer.ffn_up_bias {
                ops::add_bias(&mut ffn_hidden, bias);
            }

            // GELU activation (CPU)
            ops::gelu(&mut ffn_hidden);

            // 2g. FFN down projection (CPU fused)
            let mut ffn_output = self
                .model
                .fused_matmul(&ffn_hidden, &layer.ffn_down_weight)?;
            if let Some(ref bias) = layer.ffn_down_bias {
                ops::add_bias(&mut ffn_output, bias);
            }

            // Residual connection
            for i in 0..hidden.len() {
                hidden[i] += ffn_output[i];
            }
        }

        // 3. Final layer norm (CPU)
        let normed = ops::layer_norm(
            &hidden,
            &self.model.output_norm_weight,
            self.model.output_norm_bias.as_deref(),
            self.model.config.eps,
        );

        // 4. LM head projection (CPU fused)
        let seq_len = token_ids.len();
        let last_hidden_start = (seq_len - 1) * hidden_dim;
        let last_hidden = &normed[last_hidden_start..last_hidden_start + hidden_dim];

        let mut logits = self
            .model
            .fused_matmul(last_hidden, &self.model.lm_head_weight)?;
        if let Some(ref bias) = self.model.lm_head_bias {
            ops::add_bias(&mut logits, bias);
        }

        Ok(logits)
    }

    /// Generate tokens using CUDA acceleration (IMP-800a)
    ///
    /// # Arguments
    ///
    /// * `prompt` - Initial token IDs
    /// * `config` - Generation configuration
    ///
    /// # Returns
    ///
    /// Generated token sequence including prompt
    pub fn forward_single_cuda_with_cache(
        &mut self,
        token_id: u32,
        cache: &mut OwnedQuantizedKVCache,
        position: usize,
    ) -> Result<Vec<f32>> {
        let hidden_dim = self.model.config.hidden_dim;
        let num_heads = self.model.config.num_heads;
        let num_kv_heads = self.model.config.num_kv_heads;
        // GH-479: Use config methods for head_dim/q_dim/kv_dim (Qwen3 head_dim != hidden/heads)
        let head_dim = self.model.config.head_dim();
        let q_dim = self.model.config.q_dim();
        let kv_dim = self.model.config.kv_dim();
        let num_layers = self.model.layers.len();
        let eps = self.model.config.eps;

        // 1. Token embedding lookup (CPU - fast enough)
        let mut hidden = self.model.embed(&[token_id]);

        // 2. Process through transformer layers (index-based to avoid borrow issues)
        for layer_idx in 0..num_layers {
            // 2a. Attention layer norm (CPU)
            let normed = ops::layer_norm(
                &hidden,
                &self.model.layers[layer_idx].attn_norm_weight,
                self.model.layers[layer_idx].attn_norm_bias.as_deref(),
                eps,
            );

            // 2b. QKV projection (CPU - fused Q4_K)
            let mut qkv = self
                .model
                .qkv_matmul(&normed, &self.model.layers[layer_idx].qkv_weight)?;
            if let Some(ref bias) = self.model.layers[layer_idx].qkv_bias {
                ops::add_bias(&mut qkv, bias);
            }

            // 2c. Extract Q, K, V and apply RoPE (GQA-aware dimensions)
            // GH-479: Use q_dim (may differ from hidden_dim for Qwen3)
            let mut q = qkv[0..q_dim].to_vec();
            let mut k = qkv[q_dim..q_dim + kv_dim].to_vec();
            let v = qkv[q_dim + kv_dim..q_dim + 2 * kv_dim].to_vec();

            // GH-479: Per-head QK RMSNorm (Qwen3) — before RoPE
            if let Some(ref w) = self.model.layers[layer_idx].attn_q_norm_weight {
                ops::apply_per_head_rms_norm(&mut q, w, num_heads, eps);
            }
            if let Some(ref w) = self.model.layers[layer_idx].attn_k_norm_weight {
                ops::apply_per_head_rms_norm(&mut k, w, num_kv_heads, eps);
            }

            self.model
                .apply_rope(&mut q, position, self.model.config.num_heads);
            self.model
                .apply_rope(&mut k, position, self.model.config.num_kv_heads);

            // 2d. Get cached K/V and compute attention
            let k_cache = cache.get_k(layer_idx);
            let v_cache = cache.get_v(layer_idx);

            let attn_out = if k_cache.is_empty() {
                // First token - no cache yet, GQA expansion needed for output
                if num_kv_heads < num_heads {
                    // Expand V to match num_heads by repeating KV groups
                    let q_per_kv = num_heads / num_kv_heads;
                    // GH-479: Use q_dim for expanded output size
                    let mut expanded = vec![0.0f32; q_dim];
                    for q_head in 0..num_heads {
                        let kv_head = q_head / q_per_kv;
                        let src_offset = kv_head * head_dim;
                        let dst_offset = q_head * head_dim;
                        expanded[dst_offset..dst_offset + head_dim]
                            .copy_from_slice(&v[src_offset..src_offset + head_dim]);
                    }
                    expanded
                } else {
                    v.clone()
                }
            } else {
                // Use GPU multi-head attention if cache is large enough (PARITY-044)
                let cache_len = if kv_dim > 0 {
                    k_cache.len() / kv_dim
                } else {
                    0
                };
                let total_len = cache_len + 1;

                // PAR-017: Lower GPU attention threshold for more consistent GPU usage
                // Previous: 32 tokens caused high variance with short sequences
                const GPU_ATTN_THRESHOLD: usize = 8;

                if total_len >= GPU_ATTN_THRESHOLD && num_kv_heads == num_heads {
                    // GPU path only works for non-GQA models currently
                    self.cuda_attention_with_cache(
                        &q, k_cache, v_cache, &k, &v, total_len, num_heads, head_dim,
                    )?
                } else {
                    // CPU path for short sequences or GQA models
                    // Use GQA-aware version that handles grouped KV heads correctly
                    self.model
                        .attention_with_cache_gqa(&q, k_cache, v_cache, &k, &v)
                }
            };

            // 2e. Store K and V in cache for future tokens
            cache.append(layer_idx, &k, &v);

            // 2f. Attention output projection (CPU fused)
            let mut attn_output = self
                .model
                .fused_matmul(&attn_out, &self.model.layers[layer_idx].attn_output_weight)?;
            if let Some(ref bias) = self.model.layers[layer_idx].attn_output_bias {
                ops::add_bias(&mut attn_output, bias);
            }

            // 2g. Residual connection
            for i in 0..hidden_dim {
                hidden[i] += attn_output[i];
            }

            // PAR-047: FFN with proper SwiGLU/GELU detection
            // LLaMA-family models use SwiGLU (ffn_gate_weight present)
            // Phi-2 style models use GELU (no gate weight)
            let ffn_activated =
                if let Some(ref gate_weight) = self.model.layers[layer_idx].ffn_gate_weight {
                    // SwiGLU path (LLaMA, TinyLlama, Mistral, Qwen, etc.)
                    // Apply FFN norm if present (separate from attention norm in LLaMA-style)
                    let ffn_input =
                        if let Some(ref ffn_norm) = self.model.layers[layer_idx].ffn_norm_weight {
                            ops::layer_norm(
                                &hidden,
                                ffn_norm,
                                self.model.layers[layer_idx].ffn_norm_bias.as_deref(),
                                eps,
                            )
                        } else {
                            hidden.clone()
                        };

                    let mut ffn_up = self
                        .model
                        .fused_matmul(&ffn_input, &self.model.layers[layer_idx].ffn_up_weight)?;
                    if let Some(ref bias) = self.model.layers[layer_idx].ffn_up_bias {
                        ops::add_bias(&mut ffn_up, bias);
                    }

                    let mut ffn_gate = self.model.fused_matmul(&ffn_input, gate_weight)?;
                    if let Some(ref bias) = self.model.layers[layer_idx].ffn_gate_bias {
                        ops::add_bias(&mut ffn_gate, bias);
                    }

                    // SwiGLU: silu(gate) * up
                    ops::silu(&mut ffn_gate);
                    for i in 0..ffn_gate.len() {
                        ffn_gate[i] *= ffn_up[i];
                    }
                    ffn_gate
                } else {
                    // GELU path (phi-2 style, no gate weight)
                    let mut ffn_hidden = self
                        .model
                        .fused_matmul(&hidden, &self.model.layers[layer_idx].ffn_up_weight)?;
                    if let Some(ref bias) = self.model.layers[layer_idx].ffn_up_bias {
                        ops::add_bias(&mut ffn_hidden, bias);
                    }
                    ops::gelu(&mut ffn_hidden);
                    ffn_hidden
                };

            // 2i. FFN down projection (CPU fused)
            let mut ffn_output = self.model.fused_matmul(
                &ffn_activated,
                &self.model.layers[layer_idx].ffn_down_weight,
            )?;
            if let Some(ref bias) = self.model.layers[layer_idx].ffn_down_bias {
                ops::add_bias(&mut ffn_output, bias);
            }

            // Residual
            for i in 0..hidden_dim {
                hidden[i] += ffn_output[i];
            }
        }

        // Advance cache position after processing all layers
        cache.advance();

        // 3. Final layer norm (CPU)
        let normed = ops::layer_norm(
            &hidden,
            &self.model.output_norm_weight,
            self.model.output_norm_bias.as_deref(),
            self.model.config.eps,
        );

        // 4. LM head projection (CPU fused)
        let mut logits = self
            .model
            .fused_matmul(&normed, &self.model.lm_head_weight)?;
        if let Some(ref bias) = self.model.lm_head_bias {
            ops::add_bias(&mut logits, bias);
        }

        Ok(logits)
    }

    /// IMP-1010: GPU-accelerated fused Q4_K matmul
    ///
    /// Uses `CudaExecutor::q4k_matvec` to execute quantized matrix-vector
    /// multiplication directly on GPU, avoiding CPU SIMD overhead.
    ///
    /// # Performance Impact
    ///
    /// - CPU SIMD path: ~5 tok/s (limited by memory bandwidth)
    /// - GPU CUDA path: ~200 tok/s (theoretical, matching Ollama)
    /// - Key: Dequantize on GPU, not on CPU
    ///
    /// # Arguments
    ///
    /// * `input` - Input vector (f32)
    /// * `weight` - Quantized weight tensor (Q4_K format)
    ///
    /// # Returns
    ///
    /// Output vector [out_dim]
    fn fused_matmul_cuda(
        &mut self,
        input: &[f32],
        weight: &OwnedQuantizedTensor,
    ) -> Result<Vec<f32>> {
        // Only Q4_K is supported for GPU acceleration (PARITY-041)
        const GGUF_TYPE_Q4_K: u32 = 12;

        if weight.qtype != GGUF_TYPE_Q4_K {
            // Fallback to CPU for non-Q4_K weights
            return self.model.fused_matmul(input, weight);
        }

        let in_dim = weight.in_dim;
        let out_dim = weight.out_dim;

        // GPU kernel expects single input (seq_len=1 during token generation)
        if input.len() != in_dim {
            return Err(RealizarError::InvalidShape {
                reason: format!(
                    "IMP-1010: Input length {} doesn't match weight in_dim {}",
                    input.len(),
                    in_dim
                ),
            });
        }

        // Allocate output buffer
        let mut output = vec![0.0f32; out_dim];

        // PAR-014: Use cached GEMV for weight reuse (avoids re-transfer each call)
        // Cache key is based on weight data pointer (stable since model owns data)
        let cache_key = format!("q4k_{:016x}", weight.data.as_ptr() as usize);

        // Lazy cache - upload weight on first use
        if !self.executor.has_quantized_weights(&cache_key) {
            self.executor
                .load_quantized_weights(&cache_key, &weight.data)
                .map_err(|e| RealizarError::UnsupportedOperation {
                    operation: "cuda_q4k_cache".to_string(),
                    reason: format!("Failed to cache Q4_K weights: {e}"),
                })?;
        }

        // Execute Q4_K matmul on GPU using cached weights
        self.executor
            .q4k_gemv_cached(
                &cache_key,
                input,
                &mut output,
                out_dim as u32,
                in_dim as u32,
            )
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "q4k_gemv_cached".to_string(),
                reason: format!("CUDA Q4_K GEMV failed: {e}"),
            })?;

        Ok(output)
    }
}
